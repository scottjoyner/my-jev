"""`train_causal.main` end to end, with no 4B backbone and no `peft`.

Same problem `train.main` had and now does not: the causal lane built its scorer
through `from_pretrained` for a decoder backbone, so the entry point could not be
exercised without a multi-gigabyte download. `CausalScalarSystemOneModel` now accepts
a pre-built tokenizer and encoder, so the backbone here is a tiny
`BertForSequenceClassification` built from a config.

**What this does not cover:** LoRA itself. `train_causal.main` hardcodes
`enable_lora=True`, and the offline factory here bypasses it, so the adapter
configuration, the peft wrapping and the adapter checkpoint are all outside what these
tests reach. The `peft`-missing error paths *are* covered, because those are the ones
an operator hits and they are pure refusal logic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

import my_jev.train_causal as causal_train
from my_jev.causal_scalar import CausalScalarSystemOneModel
from my_jev.data import dump_jsonl
from my_jev.synth import generate_synthetic_records

VOCAB = 64
SEQUENCE = 8


class TinyTokenizer:
    """Answers the two shapes the causal scorer asks for.

    It needs `pad_token_id`/`eos_token_id` because the constructor repairs a
    tokenizer without a pad token, and that repair is real logic worth exercising --
    so this one reports an id rather than pretending the branch does not exist.
    """

    pad_token_id = 0
    eos_token_id = 2
    pad_token = "<pad>"
    eos_token = "</s>"

    def __init__(self) -> None:
        self.saved_to: list[str] = []

    def save_pretrained(self, path):
        self.saved_to.append(str(path))
        Path(path).mkdir(parents=True, exist_ok=True)

    def __call__(self, texts, *, padding=True, truncation=True,
                 max_length=None, return_tensors="pt"):
        from transformers.tokenization_utils_base import BatchEncoding

        width = min(SEQUENCE, max_length or SEQUENCE)
        ids = torch.zeros((len(texts), width), dtype=torch.long)
        mask = torch.ones((len(texts), width), dtype=torch.long)
        for row, text in enumerate(texts):
            for column, token in enumerate(text.split()[:width]):
                ids[row, column] = 1 + (sum(token.lower().encode()) % (VOCAB - 1))
        return BatchEncoding(
            {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": torch.zeros((len(texts), width), dtype=torch.long),
            }
        )


def _tiny_encoder():
    from transformers import BertConfig, BertForSequenceClassification

    return BertForSequenceClassification(
        BertConfig(
            vocab_size=VOCAB,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=SEQUENCE * 4,
            num_labels=1,
            problem_type="regression",
        )
    )


@pytest.fixture
def offline_model(monkeypatch):
    """Build the scorer locally, without LoRA.

    `enable_lora` is forced off because peft is an optional extra and LoRA wrapping is
    not what these tests are about; the LoRA arguments still travel through
    `run_config`, which is asserted.
    """
    built: list[CausalScalarSystemOneModel] = []

    def factory(**kwargs):
        kwargs.pop("enable_lora", None)
        # `max_length` comes through from main(), so it is overridden rather than
        # passed twice -- a smaller one keeps the tiny encoder's position table big
        # enough without changing what main() thinks it asked for.
        kwargs["max_length"] = min(kwargs.get("max_length", SEQUENCE * 2), SEQUENCE * 2)
        model = CausalScalarSystemOneModel(
            tokenizer=TinyTokenizer(),
            encoder=_tiny_encoder(),
            enable_lora=False,
            **kwargs,
        )
        built.append(model)
        return model

    monkeypatch.setattr(causal_train, "CausalScalarSystemOneModel", factory)
    return built


@pytest.fixture
def corpus(tmp_path):
    train_file = tmp_path / "train.jsonl"
    valid_file = tmp_path / "valid.jsonl"
    dump_jsonl(generate_synthetic_records(8, seed=5), train_file)
    dump_jsonl(generate_synthetic_records(4, seed=6), valid_file)
    return train_file, valid_file


def _argv(tmp_path, corpus, *extra):
    return [
        "my_jev.train-causal",
        "--train", str(corpus[0]),
        "--valid", str(corpus[1]),
        "--output", str(tmp_path / "run"),
        "--epochs", "1",
        "--batch-size", "2",
        "--grad-accum", "2",
        "--device", "cpu",
        "--max-length", "16",
        *extra,
    ]


def _run(monkeypatch, tmp_path, corpus, *extra):
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, corpus, *extra))
    causal_train.main()


# --- the happy path --------------------------------------------------------


def test_a_causal_run_writes_a_checkpoint_and_a_run_record(
    monkeypatch, tmp_path, corpus, offline_model
):
    _run(monkeypatch, tmp_path, corpus)

    run = tmp_path / "run"
    assert (run / "run_config.json").exists()
    assert (run / "last").is_dir()
    assert (run / "best").is_dir()


def test_the_causal_run_record_carries_the_lora_arguments(
    monkeypatch, tmp_path, corpus, offline_model
):
    """LoRA is bypassed offline, but its settings are still recorded.

    So the run record describes the run that was *asked for*, which is what someone
    reading it months later is relying on.
    """
    _run(monkeypatch, tmp_path, corpus, "--lora-r", "8", "--lora-alpha", "16",
         "--target-modules", "query,value")

    recorded = json.loads((tmp_path / "run" / "run_config.json").read_text())
    assert recorded["lora_r"] == 8
    assert recorded["lora_alpha"] == 16
    assert recorded["target_modules"] == "query,value"
    assert recorded["backbone"] == "Qwen/Qwen3.5-4B-Base"
    assert recorded["device"] == "cpu"
    assert recorded["train_records"] == 8
    assert recorded["valid_records"] == 4


def test_the_checkpoint_is_self_contained(monkeypatch, tmp_path, corpus, offline_model):
    """Weights, config and tokenizer all travel together.

    Otherwise reloading reaches for the hub, which is exactly what the offline
    construction was introduced to avoid.
    """
    _run(monkeypatch, tmp_path, corpus)
    saved = offline_model[0].tokenizer.saved_to
    assert saved, "the tokenizer was not persisted with the checkpoint"


def test_two_runs_at_the_same_seed_agree(monkeypatch, tmp_path, corpus, offline_model):
    def weights(seed: str) -> bytes:
        destination = tmp_path / seed
        monkeypatch.setattr(
            sys, "argv",
            ["my_jev.train-causal",
             "--train", str(corpus[0]), "--valid", str(corpus[1]),
             "--output", str(destination), "--epochs", "1",
             "--batch-size", "2", "--device", "cpu", "--max-length", "16",
             "--seed", "11"],
        )
        causal_train.main()
        files = sorted(f for f in (destination / "last").rglob("*") if f.is_file())
        return b"".join(
            f.name.encode() + b":" + f.read_bytes()
            for f in files
        )

    assert weights("a") == weights("b")


# --- what it refuses -------------------------------------------------------


def test_an_empty_training_corpus_is_refused(monkeypatch, tmp_path, offline_model):
    """Parity with `train.main`, which was guarded first for the same reason."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    valid = tmp_path / "valid.jsonl"
    dump_jsonl(generate_synthetic_records(2, seed=1), valid)

    monkeypatch.setattr(
        sys, "argv",
        ["my_jev.train-causal", "--train", str(empty), "--valid", str(valid),
         "--output", str(tmp_path / "run"), "--device", "cpu"],
    )
    with pytest.raises(SystemExit, match="no training records"):
        causal_train.main()


def test_an_empty_validation_corpus_is_refused(monkeypatch, tmp_path, corpus, offline_model):
    # An existing but empty file, not a missing one: a wrong path raises on load and
    # never reaches the emptiness check, so it would prove nothing here.
    empty = tmp_path / "empty-valid.jsonl"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv",
        ["my-jev.train-causal", "--train", str(corpus[0]),
         "--valid", str(empty),
         "--output", str(tmp_path / "run"), "--device", "cpu"],
    )
    with pytest.raises(SystemExit, match="no validation records"):
        causal_train.main()


def test_a_missing_training_file_is_not_swallowed(monkeypatch, tmp_path, corpus, offline_model):
    monkeypatch.setattr(
        sys, "argv",
        ["my-jev.train-causal", "--train", str(tmp_path / "nope.jsonl"),
         "--valid", str(corpus[1]), "--output", str(tmp_path / "run"),
         "--device", "cpu"],
    )
    with pytest.raises((FileNotFoundError, OSError)):
        causal_train.main()


# --- the peft refusals, which are the part an operator actually hits ------


def _without_peft(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "peft" or name.startswith("peft."):
            raise ImportError("no peft")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)


def test_asking_for_lora_without_the_extra_names_the_extra(monkeypatch):
    """An unactionable error here means a person downloads nothing and reruns."""
    _without_peft(monkeypatch)
    with pytest.raises(RuntimeError, match=r"\[?causal'?\]? extra|casual extra"):
        CausalScalarSystemOneModel(
            backbone="unused/offline",
            tokenizer=TinyTokenizer(),
            encoder=_tiny_encoder(),
            enable_lora=True,
        )


def test_loading_a_lora_adapter_without_the_extra_names_the_extra(monkeypatch, tmp_path):
    _without_peft(monkeypatch)
    with pytest.raises(RuntimeError, match="causal"):
        CausalScalarSystemOneModel(
            backbone="unused/offline",
            tokenizer=TinyTokenizer(),
            encoder=_tiny_encoder(),
            adapter_path=tmp_path / "adapter",
        )


def test_lora_can_be_switched_off_without_the_extra():
    """The documented way to use this lane without peft, and it works."""
    model = CausalScalarSystemOneModel(
        backbone="unused/offline",
        tokenizer=TinyTokenizer(),
        encoder=_tiny_encoder(),
        enable_lora=False,
    )
    assert model.head_kind == "causal_scalar"


def test_a_zero_lora_rank_also_bypasses_the_extra(monkeypatch):
    """`lora_r=0` means no adapter, so peft is not needed either."""
    _without_peft(monkeypatch)
    model = CausalScalarSystemOneModel(
        backbone="unused/offline",
        tokenizer=TinyTokenizer(),
        encoder=_tiny_encoder(),
        lora_r=0,
    )
    assert model.lora_r == 0


# --- construction ----------------------------------------------------------


def test_tokenizer_and_encoder_must_be_supplied_together():
    with pytest.raises(ValueError, match="together"):
        CausalScalarSystemOneModel(tokenizer=TinyTokenizer())
    with pytest.raises(ValueError, match="together"):
        CausalScalarSystemOneModel(encoder=_tiny_encoder())


def test_a_tokenizer_with_neither_pad_nor_eos_is_refused():
    """The scorer needs a pad token to batch at all, so it says so up front.

    A tokenizer with neither cannot be repaired, and padding it with something else
    would silently change every sequence length.
    """

    class Unpaddable(TinyTokenizer):
        pad_token_id = None
        eos_token_id = None

    with pytest.raises(ValueError, match="pad or EOS"):
        CausalScalarSystemOneModel(
            backbone="unused/offline",
            tokenizer=Unpaddable(),
            encoder=_tiny_encoder(),
            enable_lora=False,
        )
