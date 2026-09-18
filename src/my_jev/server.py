from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from pydantic import BaseModel, Field

from .agent_policy import (
    AgentPolicyState,
    PolicyConstraints,
    build_agent_policy_record,
    resolve_agent_policy,
    scores_from_predictions,
)
from .assistx_adapter import (
    assistx_policy_payload,
    hermes_turn_directive,
)
from .calibration import load_temperature
from .checkpoint import load_checkpoint


class AgentPolicyRequest(BaseModel):
    state: AgentPolicyState
    constraints: PolicyConstraints = Field(
        default_factory=PolicyConstraints
    )


@dataclass
class PolicyRuntime:
    model: object
    temperature: float
    checkpoint: str
    device: str

    @classmethod
    def load(
        cls,
        checkpoint: str,
        *,
        calibration: str | None = None,
        device: str | None = None,
    ) -> PolicyRuntime:
        selected = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )
        model = load_checkpoint(
            checkpoint,
            device=selected,
        )
        return cls(
            model=model,
            temperature=load_temperature(
                calibration
            ),
            checkpoint=checkpoint,
            device=str(selected),
        )

    def agent_policy(
        self,
        request: AgentPolicyRequest,
    ) -> dict[str, object]:
        record = build_agent_policy_record(
            request.state
        )
        predictions = self.model.predict(
            [record],
            temperature=self.temperature,
        )
        scores = scores_from_predictions(
            predictions
        )
        resolved = resolve_agent_policy(
            scores,
            request.constraints,
        )
        return {
            "contract": "assistx-agent-policy-v1",
            "checkpoint": self.checkpoint,
            "temperature": self.temperature,
            "device": self.device,
            "predictions": predictions,
            "scores": scores.model_dump(mode="json"),
            "resolved": resolved.model_dump(mode="json"),
            "assistx": assistx_policy_payload(
                resolved
            ),
            "hermes": hermes_turn_directive(
                resolved
            ),
        }


def create_app(runtime: PolicyRuntime):
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise RuntimeError(
            "Install my-jev with the serve extra: "
            "pip install -e '.[serve]'"
        ) from exc

    app = FastAPI(
        title="my-jev policy service",
        version="0.1.0",
    )

    @app.get("/healthz")
    def healthz():
        return {
            "ok": True,
            "contract": "assistx-agent-policy-v1",
            "checkpoint": runtime.checkpoint,
            "temperature": runtime.temperature,
            "device": runtime.device,
        }

    @app.post("/v1/agent-policy")
    def agent_policy(
        request: AgentPolicyRequest,
    ):
        return runtime.agent_policy(request)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Serve my-jev as a Hermes/AssistX "
            "agent-policy sidecar"
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument("--calibration")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8088,
    )
    parser.add_argument("--device")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "Install my-jev with the serve extra: "
            "pip install -e '.[serve]'"
        ) from exc

    runtime = PolicyRuntime.load(
        args.checkpoint,
        calibration=args.calibration,
        device=args.device,
    )
    uvicorn.run(
        create_app(runtime),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
