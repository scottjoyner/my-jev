from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .bonsai_contract import (
    BonsaiDecisionContract,
    contract_payload,
    inspect_prism_llama_cpp,
)


def _git_revision(
    root: Path,
) -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None

    value = (
        result.stdout.strip()
    )
    return (
        value
        if value
        else None
    )


def probe_bonsai_runtime(
    llama_root: str | Path,
    *,
    model_path: (
        str | Path | None
    ) = None,
) -> dict[str, object]:
    root = Path(
        llama_root
    ).expanduser().resolve()

    contract = (
        BonsaiDecisionContract()
    )
    contract.validate()

    runtime = (
        inspect_prism_llama_cpp(
            root
        )
    )

    model: dict[str, object] = {
        "required": (
            model_path
            is not None
        ),
        "path": (
            str(
                Path(
                    model_path
                ).expanduser().resolve()
            )
            if model_path
            is not None
            else None
        ),
        "exists": (
            Path(
                model_path
            ).expanduser().is_file()
            if model_path
            is not None
            else None
        ),
    }

    ready = bool(
        runtime[
            "ready"
        ]
    )
    if (
        model_path
        is not None
        and not model[
            "exists"
        ]
    ):
        ready = False

    blockers: list[str] = []
    if not runtime[
        "ready"
    ]:
        blockers.append(
            "PrismML llama.cpp staging "
            "activation APIs are missing"
        )
    if (
        model_path
        is not None
        and not model[
            "exists"
        ]
    ):
        blockers.append(
            "Bonsai 2 GGUF model file "
            "does not exist"
        )

    return {
        "schema_version": 1,
        "ready_for_activation_bridge": (
            ready
        ),
        "contract": (
            contract_payload(
                contract
            )
        ),
        "runtime": {
            **runtime,
            "git_revision": (
                _git_revision(
                    root
                )
            ),
        },
        "model": model,
        "blockers": blockers,
        "next_boundary": (
            "build the C/C++ activation extractor "
            "against llama-ext.h and stream "
            "frozen activations into the Python "
            "decision head"
            if ready
            else (
                "resolve runtime/model blockers "
                "before implementing the extractor"
            )
        ),
        "authority_boundary": {
            "evidence_only": True,
            "dispatch_allowed": False,
            "runtime_authority_changed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether a PrismML llama.cpp "
            "checkout exposes the hidden-state "
            "hooks required for a Bonsai 2 "
            "decision-attention bridge"
        )
    )
    parser.add_argument(
        "--llama-root",
        required=True,
    )
    parser.add_argument(
        "--model-path",
    )
    parser.add_argument(
        "--output",
    )
    args = parser.parse_args()

    result = probe_bonsai_runtime(
        args.llama_root,
        model_path=(
            args.model_path
        ),
    )
    rendered = (
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    if args.output:
        output = Path(
            args.output
        )
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output.write_text(
            rendered,
            encoding="utf-8",
        )

    print(
        rendered,
        end="",
    )

    if not result[
        "ready_for_activation_bridge"
    ]:
        raise SystemExit(
            2
        )


if __name__ == "__main__":
    main()
