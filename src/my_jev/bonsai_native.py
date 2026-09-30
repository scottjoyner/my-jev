from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from .activation_protocol import (
    ActivationFrame,
    read_activation_frame,
)
from .bonsai_contract import (
    BonsaiDecisionContract,
)
from .prompt_contract import (
    PROMPT_CONTRACT_VERSION,
    candidate_text,
)
from .schema import DecisionRecord


def _sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()
    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )
    return digest.hexdigest()


def _git_revision(
    root: Path,
) -> str:
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
    value = (
        result.stdout.strip()
    )
    if (
        len(
            value
        )
        != 40
    ):
        raise ValueError(
            "runtime git revision must "
            "be a full 40-character SHA"
        )
    return value


def _load_record(
    path: Path,
) -> DecisionRecord:
    text = path.read_text(
        encoding="utf-8"
    ).strip()
    if not text:
        raise ValueError(
            "record file is empty"
        )

    if text.startswith(
        "{"
    ):
        try:
            return (
                DecisionRecord
                .model_validate_json(
                    text
                )
            )
        except Exception:
            pass

    rows = [
        row
        for row in (
            text.splitlines()
        )
        if row.strip()
    ]
    if len(
        rows
    ) != 1:
        raise ValueError(
            "record input must contain "
            "exactly one DecisionRecord"
        )
    return (
        DecisionRecord
        .model_validate_json(
            rows[
                0
            ]
        )
    )


class BonsaiNativeProvider:
    """Capture typed decision activations from a frozen Bonsai GGUF runtime."""

    provider_name = (
        "bonsai2-ternary-llama-cpp"
    )

    def __init__(
        self,
        *,
        bridge: str | Path,
        model: str | Path,
        llama_root: str | Path,
        contract: (
            BonsaiDecisionContract
            | None
        ) = None,
        context_size: int = 2048,
        gpu_layers: int = 999,
    ) -> None:
        self.bridge = Path(
            bridge
        ).expanduser().resolve()
        self.model = Path(
            model
        ).expanduser().resolve()
        self.llama_root = Path(
            llama_root
        ).expanduser().resolve()
        self.contract = (
            contract
            or BonsaiDecisionContract()
        )
        self.contract.validate()

        if not self.bridge.is_file():
            raise ValueError(
                "Bonsai bridge executable "
                f"does not exist: {self.bridge}"
            )
        if not self.model.is_file():
            raise ValueError(
                "Bonsai GGUF does not exist: "
                f"{self.model}"
            )
        if not (
            self.llama_root
            / "src"
            / "llama-ext.h"
        ).is_file():
            raise ValueError(
                "PrismML llama.cpp staging "
                "header is missing"
            )
        if context_size < 32:
            raise ValueError(
                "context_size must be >= 32"
            )

        self.context_size = (
            context_size
        )
        self.gpu_layers = (
            gpu_layers
        )
        self.model_sha256 = (
            _sha256(
                self.model
            )
        )
        self.runtime_revision = (
            _git_revision(
                self.llama_root
            )
        )

    def _bridge_command(
        self,
        *,
        record: DecisionRecord,
        state_file: Path,
        option_files: list[
            tuple[
                str,
                str,
                str,
                Path,
            ]
        ],
        output: Path,
    ) -> list[str]:
        command = [
            str(
                self.bridge
            ),
            "--model",
            str(
                self.model
            ),
            "--model-sha256",
            self.model_sha256,
            "--runtime-revision",
            self.runtime_revision,
            "--state-file",
            str(
                state_file
            ),
            "--output",
            str(
                output
            ),
            "--hidden-size",
            str(
                self.contract.hidden_size
            ),
            "--num-layers",
            str(
                self.contract.num_layers
            ),
            "--ctx",
            str(
                self.context_size
            ),
            "--gpu-layers",
            str(
                self.gpu_layers
            ),
        ]

        for layer in (
            self.contract.layer_taps
        ):
            command.extend(
                [
                    "--tap",
                    str(
                        layer
                    ),
                ]
            )
        if (
            self.contract
            .include_final_prenorm
        ):
            command.append(
                "--final-prenorm"
            )

        current: (
            tuple[
                str,
                str,
            ]
            | None
        ) = None
        for (
            question_name,
            question_type,
            option_label,
            path,
        ) in option_files:
            identity = (
                question_name,
                question_type,
            )
            if (
                current
                != identity
            ):
                command.extend(
                    [
                        "--question",
                        question_name,
                        question_type,
                    ]
                )
                current = identity

            command.extend(
                [
                    "--option",
                    option_label,
                    str(
                        path
                    ),
                ]
            )

        return command

    def capture(
        self,
        record: DecisionRecord,
        *,
        output: (
            str | Path | None
        ) = None,
    ) -> ActivationFrame:
        with tempfile.TemporaryDirectory(
            prefix=(
                "my-jev-bonsai-"
            )
        ) as temp:
            root = Path(
                temp
            )
            state_file = (
                root
                / "state.txt"
            )
            state_file.write_text(
                record.state,
                encoding="utf-8",
            )

            option_files: list[
                tuple[
                    str,
                    str,
                    str,
                    Path,
                ]
            ] = []

            option_index = 0
            for (
                question_name,
                question,
            ) in (
                record.questions.items()
            ):
                for (
                    local_index,
                    option,
                ) in enumerate(
                    question.options
                    or []
                ):
                    path = (
                        root
                        / (
                            "option-"
                            f"{option_index:04d}"
                            ".txt"
                        )
                    )
                    path.write_text(
                        candidate_text(
                            question,
                            option,
                            local_index,
                        ),
                        encoding="utf-8",
                    )
                    option_files.append(
                        (
                            question_name,
                            question.type.value,
                            option,
                            path,
                        )
                    )
                    option_index += 1

            if not option_files:
                raise ValueError(
                    "DecisionRecord has no "
                    "runtime options"
                )

            frame_path = (
                Path(
                    output
                ).expanduser().resolve()
                if output
                is not None
                else (
                    root
                    / "frame.bin"
                )
            )
            frame_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            command = (
                self._bridge_command(
                    record=record,
                    state_file=(
                        state_file
                    ),
                    option_files=(
                        option_files
                    ),
                    output=(
                        frame_path
                    ),
                )
            )
            subprocess.run(
                command,
                check=True,
            )

            with frame_path.open(
                "rb"
            ) as handle:
                frame = (
                    read_activation_frame(
                        handle
                    )
                )

            self.validate_frame(
                frame,
                record=record,
            )
            return frame

    def validate_frame(
        self,
        frame: ActivationFrame,
        *,
        record: DecisionRecord,
    ) -> None:
        frame.validate()

        if (
            frame.provider
            != self.provider_name
        ):
            raise ValueError(
                "activation provider mismatch"
            )
        if (
            frame.model_sha256
            != self.model_sha256
        ):
            raise ValueError(
                "activation model SHA mismatch"
            )
        if (
            frame.runtime_revision
            != self.runtime_revision
        ):
            raise ValueError(
                "activation runtime SHA mismatch"
            )
        if (
            frame.prompt_contract
            != PROMPT_CONTRACT_VERSION
        ):
            raise ValueError(
                "activation prompt contract "
                "mismatch"
            )
        if (
            frame.hidden_size
            != self.contract.hidden_size
        ):
            raise ValueError(
                "activation hidden size "
                "mismatch"
            )
        if (
            frame.tap_count
            != self.contract.tap_count
        ):
            raise ValueError(
                "activation tap count mismatch"
            )

        expected_groups: list[
            tuple[
                str,
                str,
                tuple[
                    str,
                    ...,
                ],
            ]
        ] = []
        for (
            name,
            question,
        ) in (
            record.questions.items()
        ):
            expected_groups.append(
                (
                    name,
                    question.type.value,
                    tuple(
                        question.options
                        or []
                    ),
                )
            )

        actual_groups = [
            (
                group.name,
                group.type,
                group.options,
            )
            for group in (
                frame.groups
            )
        ]
        if (
            actual_groups
            != expected_groups
        ):
            raise ValueError(
                "activation typed question "
                "groups do not match record"
            )


def capture_receipt(
    provider: BonsaiNativeProvider,
    record: DecisionRecord,
    frame: ActivationFrame,
    *,
    frame_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": (
            "bonsai2-native-activation-capture"
        ),
        "provider": (
            provider.provider_name
        ),
        "model": {
            "path": str(
                provider.model
            ),
            "sha256": (
                provider.model_sha256
            ),
        },
        "runtime": {
            "root": str(
                provider.llama_root
            ),
            "git_sha": (
                provider.runtime_revision
            ),
        },
        "bridge": str(
            provider.bridge
        ),
        "prompt_contract": (
            PROMPT_CONTRACT_VERSION
        ),
        "activation": {
            "path": str(
                frame_path
            ),
            "hidden_size": (
                frame.hidden_size
            ),
            "tap_count": (
                frame.tap_count
            ),
            "state_tokens": int(
                frame.state_taps.shape[
                    2
                ]
            ),
            "options": int(
                frame.option_taps.shape[
                    1
                ]
            ),
            "questions": len(
                frame.groups
            ),
        },
        "record": {
            "source": (
                record.source
            ),
            "metadata": (
                record.metadata
            ),
        },
        "authority_boundary": {
            "evidence_only": True,
            "dispatch_allowed": False,
            "runtime_authority_changed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Capture one typed DecisionRecord "
            "through the native Bonsai 2 "
            "activation bridge"
        )
    )
    parser.add_argument(
        "--bridge",
        required=True,
    )
    parser.add_argument(
        "--model",
        required=True,
    )
    parser.add_argument(
        "--llama-root",
        required=True,
    )
    parser.add_argument(
        "--record",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--receipt",
        required=True,
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=999,
    )
    args = parser.parse_args()

    record = _load_record(
        Path(
            args.record
        )
    )
    provider = (
        BonsaiNativeProvider(
            bridge=args.bridge,
            model=args.model,
            llama_root=(
                args.llama_root
            ),
            context_size=args.ctx,
            gpu_layers=(
                args.gpu_layers
            ),
        )
    )
    output = Path(
        args.output
    ).expanduser().resolve()

    frame = provider.capture(
        record,
        output=output,
    )
    receipt = capture_receipt(
        provider,
        record,
        frame,
        frame_path=output,
    )

    receipt_path = Path(
        args.receipt
    ).expanduser().resolve()
    receipt_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    receipt_path.write_text(
        json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
