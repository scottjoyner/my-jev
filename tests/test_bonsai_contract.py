from pathlib import Path

import pytest

from my_jev.bonsai_contract import (
    BONSAI2_HIDDEN_SIZE,
    BONSAI2_NUM_LAYERS,
    BonsaiDecisionContract,
    contract_payload,
    inspect_prism_llama_cpp,
)
from my_jev.bonsai_probe import (
    probe_bonsai_runtime,
)


def _fake_llama_root(
    tmp_path: Path,
    *,
    complete: bool = True,
) -> Path:
    root = tmp_path / "llama.cpp"
    src = root / "src"
    src.mkdir(
        parents=True,
    )

    symbols = [
        "llama_set_embeddings_layer_inp",
        "llama_get_embeddings_layer_inp",
        "llama_set_embeddings_nextn",
        "llama_get_embeddings_nextn",
    ]
    if not complete:
        symbols = symbols[:-1]

    (
        src
        / "llama-ext.h"
    ).write_text(
        "\n".join(
            symbols
        )
        + "\n",
        encoding="utf-8",
    )
    (
        src
        / "llama-context.h"
    ).write_text(
        "staging context\n",
        encoding="utf-8",
    )
    return root


def test_bonsai_contract_matches_qwen38_shape_and_is_frozen():
    contract = (
        BonsaiDecisionContract()
    )
    contract.validate()

    assert (
        contract.hidden_size
        == BONSAI2_HIDDEN_SIZE
        == 5120
    )
    assert (
        contract.num_layers
        == BONSAI2_NUM_LAYERS
        == 64
    )
    assert (
        contract.tap_count
        == 4
    )
    assert (
        contract.backbone_trainable
        is False
    )

    payload = contract_payload(
        contract
    )
    assert (
        payload[
            "architecture"
        ]
        == "qwen35"
    )
    assert (
        payload[
            "authority_boundary"
        ][
            "dispatch_allowed"
        ]
        is False
    )


def test_bonsai_contract_rejects_trainable_ternary_backbone():
    with pytest.raises(
        ValueError,
        match="remain frozen",
    ):
        BonsaiDecisionContract(
            backbone_trainable=True
        ).validate()


def test_runtime_inspection_requires_all_staging_symbols(
    tmp_path: Path,
):
    complete = _fake_llama_root(
        tmp_path / "complete",
        complete=True,
    )
    incomplete = (
        _fake_llama_root(
            tmp_path / "incomplete",
            complete=False,
        )
    )

    assert (
        inspect_prism_llama_cpp(
            complete
        )["ready"]
        is True
    )

    report = (
        inspect_prism_llama_cpp(
            incomplete
        )
    )
    assert (
        report["ready"]
        is False
    )
    assert (
        "llama_get_embeddings_nextn"
        in report[
            "missing_symbols"
        ]
    )


def test_probe_can_require_local_gguf(
    tmp_path: Path,
):
    root = _fake_llama_root(
        tmp_path
    )
    model = (
        tmp_path
        / "bonsai2.gguf"
    )

    missing = (
        probe_bonsai_runtime(
            root,
            model_path=model,
        )
    )
    assert (
        missing[
            "ready_for_activation_bridge"
        ]
        is False
    )

    model.write_bytes(
        b"gguf-placeholder"
    )
    ready = (
        probe_bonsai_runtime(
            root,
            model_path=model,
        )
    )
    assert (
        ready[
            "ready_for_activation_bridge"
        ]
        is True
    )
