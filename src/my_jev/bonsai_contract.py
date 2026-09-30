from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BONSAI2_MODEL_REPO = (
    "prism-ml/Ternary-Bonsai-2-27B-gguf"
)
BONSAI2_BASE_MODEL = (
    "Qwen/Qwen3.8-27B"
)
BONSAI2_ARCHITECTURE = "qwen35"
BONSAI2_HIDDEN_SIZE = 5120
BONSAI2_NUM_LAYERS = 64
BONSAI2_DEFAULT_LAYER_TAPS = (
    31,
    47,
    63,
)
BONSAI2_INCLUDE_FINAL_PRENORM = True

_REQUIRED_STAGING_SYMBOLS = (
    "llama_set_embeddings_layer_inp",
    "llama_get_embeddings_layer_inp",
    "llama_set_embeddings_nextn",
    "llama_get_embeddings_nextn",
)


@dataclass(frozen=True)
class BonsaiDecisionContract:
    model_repo: str = (
        BONSAI2_MODEL_REPO
    )
    base_model: str = (
        BONSAI2_BASE_MODEL
    )
    architecture: str = (
        BONSAI2_ARCHITECTURE
    )
    hidden_size: int = (
        BONSAI2_HIDDEN_SIZE
    )
    num_layers: int = (
        BONSAI2_NUM_LAYERS
    )
    layer_taps: tuple[int, ...] = (
        BONSAI2_DEFAULT_LAYER_TAPS
    )
    include_final_prenorm: bool = (
        BONSAI2_INCLUDE_FINAL_PRENORM
    )
    option_pooling: str = (
        "masked_mean"
    )
    state_pooling: str = (
        "token_memory"
    )
    backbone_trainable: bool = False

    @property
    def tap_count(
        self,
    ) -> int:
        return (
            len(
                self.layer_taps
            )
            + int(
                self.include_final_prenorm
            )
        )

    def validate(
        self,
    ) -> None:
        if (
            self.hidden_size
            != BONSAI2_HIDDEN_SIZE
        ):
            raise ValueError(
                "Bonsai 2 hidden size must "
                f"remain {BONSAI2_HIDDEN_SIZE}"
            )
        if (
            self.num_layers
            != BONSAI2_NUM_LAYERS
        ):
            raise ValueError(
                "Bonsai 2 layer count must "
                f"remain {BONSAI2_NUM_LAYERS}"
            )
        if (
            not self.layer_taps
            and not self.include_final_prenorm
        ):
            raise ValueError(
                "at least one activation tap "
                "is required"
            )
        if len(
            set(
                self.layer_taps
            )
        ) != len(
            self.layer_taps
        ):
            raise ValueError(
                "activation layer taps must "
                "be unique"
            )
        for layer in (
            self.layer_taps
        ):
            if (
                layer < 0
                or layer
                >= self.num_layers
            ):
                raise ValueError(
                    "activation layer tap "
                    f"{layer} is out of range"
                )
        if (
            self.option_pooling
            != "masked_mean"
        ):
            raise ValueError(
                "v0 Bonsai decision contract "
                "requires masked_mean option "
                "pooling"
            )
        if (
            self.state_pooling
            != "token_memory"
        ):
            raise ValueError(
                "v0 Bonsai decision contract "
                "requires token_memory state "
                "activations"
            )
        if self.backbone_trainable:
            raise ValueError(
                "ternary Bonsai backbone must "
                "remain frozen in v0"
            )


def inspect_prism_llama_cpp(
    root: str | Path,
) -> dict[str, object]:
    source_root = Path(
        root
    ).expanduser().resolve()

    ext_header = (
        source_root
        / "src"
        / "llama-ext.h"
    )
    context_header = (
        source_root
        / "src"
        / "llama-context.h"
    )

    missing_files = [
        str(
            path
        )
        for path in (
            ext_header,
            context_header,
        )
        if not path.is_file()
    ]
    if missing_files:
        return {
            "ready": False,
            "root": str(
                source_root
            ),
            "missing_files": (
                missing_files
            ),
            "required_symbols": list(
                _REQUIRED_STAGING_SYMBOLS
            ),
            "missing_symbols": list(
                _REQUIRED_STAGING_SYMBOLS
            ),
        }

    text = (
        ext_header.read_text(
            encoding="utf-8"
        )
        + "\n"
        + context_header.read_text(
            encoding="utf-8"
        )
    )
    missing_symbols = [
        symbol
        for symbol in (
            _REQUIRED_STAGING_SYMBOLS
        )
        if symbol not in text
    ]

    return {
        "ready": (
            not missing_symbols
        ),
        "root": str(
            source_root
        ),
        "missing_files": [],
        "required_symbols": list(
            _REQUIRED_STAGING_SYMBOLS
        ),
        "missing_symbols": (
            missing_symbols
        ),
    }


def contract_payload(
    contract: (
        BonsaiDecisionContract
        | None
    ) = None,
) -> dict[str, object]:
    selected = (
        contract
        or BonsaiDecisionContract()
    )
    selected.validate()

    taps = [
        {
            "kind": (
                "layer_input"
            ),
            "layer": layer,
        }
        for layer in (
            selected.layer_taps
        )
    ]
    if (
        selected
        .include_final_prenorm
    ):
        taps.append(
            {
                "kind": (
                    "final_prenorm"
                ),
                "layer": (
                    selected.num_layers
                ),
            }
        )

    return {
        "schema_version": 1,
        "provider": (
            "bonsai2-ternary-llama-cpp"
        ),
        "model_repo": (
            selected.model_repo
        ),
        "base_model": (
            selected.base_model
        ),
        "architecture": (
            selected.architecture
        ),
        "hidden_size": (
            selected.hidden_size
        ),
        "num_layers": (
            selected.num_layers
        ),
        "tap_count": (
            selected.tap_count
        ),
        "taps": taps,
        "state_representation": (
            selected.state_pooling
        ),
        "option_pooling": (
            selected.option_pooling
        ),
        "backbone_trainable": (
            selected.backbone_trainable
        ),
        "authority_boundary": {
            "evidence_only": True,
            "dispatch_allowed": False,
            "runtime_authority_changed": False,
        },
    }
