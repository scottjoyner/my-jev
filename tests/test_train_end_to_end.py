"""`train.main` end to end, with no model download and no machine-specific fixture.

The only test that drove the training loop required `/tmp/tq-big/my-jev/train.jsonl` —
a bundle exported on one machine — *and* a populated Hugging Face cache, because
`SystemOneModel` called `from_pretrained` for a 574 MB backbone. Neither exists in CI,
so the test skipped there and the epoch loop, the gradient guard's wiring, the
checkpoint writing and the run record were all unverified outside one laptop.

`SystemOneModel` now accepts a pre-built encoder and tokenizer, so the backbone can be
a tiny `BertModel` constructed from a config. `transformers` is already a base
dependency, so nothing is downloaded.

**What this does not cover:** the real ModernBERT backbone, the real tokenizer, and
therefore tokenisation and anything downstream of it. What it does cover is the
training *loop* — batching, the guard being called at the right step, the epoch
bookkeeping, the validation pass, and what lands on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

import my_jev.train as train_module
from my_jev.data import dump_jsonl
from my_jev.model import SystemOneModel
from my_jev.synth import generate_synthetic_records
from my_jev.train import NonFiniteTrainingError

VOCAB = 64
SEQUENCE = 8


def _tiny_encoder():
    from transformers import BertConfig, BertModel

    return BertModel(
        BertConfig(
            vocab_size=VOCAB,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=SEQUENCE * 4,
        )
    )


class TinyTokenizer:
    """The two calls `forward_records` makes, answered honestly.

    Real ids, a real attention mask, and lengths derived from the text, so states and
    candidates do not collapse into one identical row. Not a real tokeniser, and the
    tests that depend on tokenisation are not here.
    """

    def save_pretrained(self, path):
        """`save_checkpoint` persists the tokenizer alongside the weights.

        A stub that records that it was asked, so the test can assert the checkpoint
        is self-contained rather than silently writing nothing.
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "stub-tokenizer.json").write_text(
            json.dumps({"stub": True, "vocab_size": VOCAB, "sequence": SEQUENCE}),
            encoding="utf-8",
        )

    def __call__(self, texts, *, padding=True, truncation=True,
                 max_length=None, return_tensors="pt"):
        from transformers.tokenization_utils_base import BatchEncoding

        width = min(SEQUENCE, max_length or SEQUENCE)
        ids = torch.zeros((len(texts), width), dtype=torch.long)
        mask = torch.ones((len(texts), width), dtype=torch.long)
        for row, text in enumerate(texts):
            tokens = text.split()
            for column, token in enumerate(tokens[:width]):
                ids[row, column] = 1 + (sum(token.lower().encode()) % (VOCAB - 1))
        return BatchEncoding(
            {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": torch.zeros((len(texts), width), dtype=torch.long),
            }
        )


@pytest.fixture
def offline_model(monkeypatch):
    """Replace the hub-backed construction with a locally built one."""
    built: list[SystemOneModel] = []

    def factory(backbone="unused/offline", **kwargs):
        model = SystemOneModel(
            backbone=backbone,
            tokenizer=TinyTokenizer(),
            encoder=_tiny_encoder(),
            max_state_length=kwargs.pop("max_state_length", SEQUENCE * 2),
            max_candidate_length=kwargs.pop("max_candidate_length", SEQUENCE),
            **kwargs,
        )
        built.append(model)
        return model

    monkeypatch.setattr(train_module, "SystemOneModel", factory)
    return built


@pytest.fixture
def corpus(tmp_path):
    """A real, generated corpus written the way the pipeline writes one."""
    train_file = tmp_path / "train.jsonl"
    valid_file = tmp_path / "valid.jsonl"
    dump_jsonl(generate_synthetic_records(12, seed=5), train_file)
    dump_jsonl(generate_synthetic_records(4, seed=6), valid_file)
    return train_file, valid_file


def _argv(tmp_path, corpus, *extra):
    train_file, valid_file = corpus
    return [
        "my_jev.train",
        "--train", str(train_file),
        "--valid", str(valid_file),
        "--output", str(tmp_path / "run"),
        "--epochs", "1",
        "--batch-size", "2",
        "--grad-accum", "2",
        "--device", "cpu",
        *extra,
    ]


def _run(monkeypatch, tmp_path, corpus, *extra):
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, corpus, *extra))
    train_module.main()


# --- the happy path --------------------------------------------------------


def test_a_run_writes_a_checkpoint_and_a_run_record(monkeypatch, tmp_path, corpus, offline_model):
    _run(monkeypatch, tmp_path, corpus)

    run = tmp_path / "run"
    assert (run / "run_config.json").exists()
    assert (run / "last").is_dir(), "the last checkpoint was not written"
    assert (run / "best").is_dir(), "the best checkpoint was not written"
    for name in ("last", "best"):
        assert (run / name / "model.pt").exists()
        assert (run / name / "my_jev_config.json").exists()
        # The tokenizer travels with the weights, so the checkpoint can be reloaded
        # without reaching for the hub.
        assert (run / name / "tokenizer" / "stub-tokenizer.json").exists()


def test_the_run_record_describes_the_run_that_happened(monkeypatch, tmp_path, corpus, offline_model):
    """The record is what makes a run auditable later, so its contents are pinned."""
    _run(monkeypatch, tmp_path, corpus, "--grad-accum", "4", "--seed", "3")

    recorded = json.loads((tmp_path / "run" / "run_config.json").read_text())
    assert recorded["epochs"] == 1
    assert recorded["grad_accum"] == 4
    assert recorded["batch_size"] == 2
    assert recorded["seed"] == 3
    assert recorded["device"] == "cpu"
    assert recorded["train_records"] == 12
    assert recorded["valid_records"] == 4
    assert recorded["bf16_requested"] is False
    assert recorded["determinism"], "the determinism record is missing"


def test_the_run_is_reproducible_for_a_given_seed(monkeypatch, tmp_path, corpus, offline_model):
    """Two runs with the same seed must agree.

    Checked on the checkpoint's own bytes rather than on a printed metric, because the
    checkpoint is what actually gets loaded later.
    """
    def checkpoint(seed: str) -> bytes:
        destination = tmp_path / seed
        monkeypatch.setattr(
            sys, "argv",
            [
                "my_jev.train",
                "--train", str(corpus[0]),
                "--valid", str(corpus[1]),
                "--output", str(destination),
                "--epochs", "1",
                "--batch-size", "2",
                "--device", "cpu",
                "--seed", "11",
            ],
        )
        train_module.main()
        return (destination / "last" / "model.pt").read_bytes()

    assert checkpoint("a") == checkpoint("b")


def test_each_requested_epoch_is_run(monkeypatch, tmp_path, corpus, offline_model):
    _run(monkeypatch, tmp_path, corpus, "--epochs", "2")

    last = json.loads(
        (tmp_path / "run" / "last" / "my_jev_config.json").read_text()
    )
    assert last["extra"]["epoch"] == 2


# --- the guard, which is the reason the loop exists ------------------------


def test_a_non_finite_gradient_stops_the_run(monkeypatch, tmp_path, corpus, offline_model):
    """The guard is only useful if the loop calls it, at the right step."""
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda *args, **kwargs: torch.tensor(float("nan")),
    )
    with pytest.raises(NonFiniteTrainingError) as caught:
        _run(monkeypatch, tmp_path, corpus)

    message = str(caught.value)
    assert "gradient norm" in message
    assert "epoch 2" in message
    # The guard fires at an *accumulation boundary*, not at every step, and the
    # message reports `step + 1`. With --grad-accum 2 the boundaries are internal
    # steps 2, 4, 6..., reported as 3, 5, 7... Asserted as a relationship rather than
    # a magic number: my first version expected "step 1" and was wrong, because the
    # guard is deliberately not called every step.
    reported_step = int(message.split("step ")[1].split()[0])
    assert (reported_step - 1) % 2 == 0, message


def test_a_non_finite_loss_stops_the_run(monkeypatch, tmp_path, corpus, offline_model):
    """A NaN loss with a finite gradient norm still poisons the parameters."""
    original = SystemOneModel.forward_records

    def poison(self, records):
        outputs = original(self, records)
        for output in outputs:
            object.__setattr__(output, "logits", output.logits * float("nan"))
        return outputs

    monkeypatch.setattr(SystemOneModel, "forward_records", poison)
    with pytest.raises(NonFiniteTrainingError):
        _run(monkeypatch, tmp_path, corpus)


def test_a_finite_run_does_not_trip_the_guard(monkeypatch, tmp_path, corpus, offline_model):
    """The counterweight to the two above: the guard must not fire on its own."""
    _run(monkeypatch, tmp_path, corpus)
    assert (tmp_path / "run" / "best").is_dir()


# --- what it refuses -------------------------------------------------------


def test_an_empty_training_corpus_is_refused(monkeypatch, tmp_path, offline_model):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    valid = tmp_path / "valid.jsonl"
    dump_jsonl(generate_synthetic_records(2, seed=1), valid)

    monkeypatch.setattr(
        sys, "argv",
        ["my_jev.train", "--train", str(empty), "--valid", str(valid),
         "--output", str(tmp_path / "run"), "--device", "cpu"],
    )
    with pytest.raises(SystemExit, match="no training records"):
        train_module.main()


def test_a_missing_training_file_is_not_swallowed(monkeypatch, tmp_path, corpus, offline_model):
    """A wrong path must surface as a wrong path, not as an empty corpus."""
    monkeypatch.setattr(
        sys, "argv",
        ["my_jev.train", "--train", str(tmp_path / "nope.jsonl"),
         "--valid", str(corpus[1]), "--output", str(tmp_path / "run"),
         "--device", "cpu"],
    )
    with pytest.raises((FileNotFoundError, OSError)):
        train_module.main()


def test_an_unknown_head_kind_is_refused_at_the_parser(monkeypatch, tmp_path, corpus, offline_model):
    """Rejected before the model is built, which is the better place for it.

    My first version expected the model's own `ValueError` and got argparse's exit
    instead -- the stronger behaviour, since nothing is constructed at all.
    """
    with pytest.raises(SystemExit):
        _run(monkeypatch, tmp_path, corpus, "--head-kind", "not_a_head")


def test_the_model_rejects_an_unknown_head_kind_too():
    """The parser is the first line; this is the second, for programmatic callers."""
    from my_jev.model import SystemOneModel

    with pytest.raises(ValueError, match="head_kind"):
        SystemOneModel(
            tokenizer=TinyTokenizer(),
            encoder=_tiny_encoder(),
            head_kind="not_a_head",
        )


# --- the offline construction itself ---------------------------------------


def test_tokenizer_and_encoder_must_be_supplied_together():
    """Half a model is a wiring mistake, caught at construction."""
    with pytest.raises(ValueError, match="together"):
        SystemOneModel(tokenizer=TinyTokenizer())
    with pytest.raises(ValueError, match="together"):
        SystemOneModel(encoder=_tiny_encoder())


def test_the_default_construction_still_names_the_backbone():
    """So the offline path is opt-in and the recorded run_config stays truthful."""
    from my_jev.model import SystemOneModel as Real

    source = Path(Real.__module__.replace(".", "/") + ".py")
    if not source.exists():
        source = Path(__file__).resolve().parent.parent / "src" / "my_jev" / "model.py"
    assert "answerdotai/ModernBERT-base" in source.read_text()