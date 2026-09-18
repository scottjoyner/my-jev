from pathlib import Path

from my_jev.agentic_synth import (
    generate_agent_policy_records,
)
from my_jev.data import dump_jsonl
from my_jev.experiment import (
    _train_command,
    load_experiment_spec,
    run_experiment,
)


def _write_dataset(path: Path):
    dump_jsonl(
        iter(
            generate_agent_policy_records(
                8,
                seed=11,
            )
        ),
        path,
    )


def test_experiment_dry_run_records_manifest_and_commands(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    for split in (
        "train",
        "validation",
        "calibration",
        "test",
    ):
        _write_dataset(
            data / f"{split}.jsonl"
        )

    spec = tmp_path / "experiment.toml"
    spec.write_text(
        """
[experiment]
name = "assistx-policy-smoke"
output_root = "runs"
registry_path = "runs/registry.json"
lock_dir = "runs/.locks"

[model]
backend = "encoder_option_query"
backbone = "answerdotai/ModernBERT-base"
head_kind = "option_query"
head_rank = 128

[data]
train = "data/train.jsonl"
validation = "data/validation.jsonl"
calibration = "data/calibration.jsonl"
test = "data/test.jsonl"

[train]
epochs = 1
batch_size = 2
grad_accum = 1
learning_rate = 0.00002
weight_decay = 0.01
max_state_length = 512
max_candidate_length = 96
seed = 17
bf16 = false
freeze_backbone = true
gradient_checkpointing = false

[benchmark]
batch_size = 4

[gates]
min_accuracy = 0.50
max_ece = 0.20
min_accuracy_delta_vs_shuffled = 0.05
min_choice_order_top1_agreement = 0.95
max_policy_consistency_violation_rate = 0.10
""".strip()
        + "\n",
        encoding="utf-8",
    )

    parsed, sha = load_experiment_spec(
        spec
    )
    assert parsed.experiment.name == (
        "assistx-policy-smoke"
    )
    assert len(sha) == 64

    result = run_experiment(
        spec,
        dry_run=True,
    )
    run_dir = Path(
        result["run_dir"]
    )
    assert (
        run_dir
        / "manifest.json"
    ).exists()
    assert (
        run_dir
        / "stages"
        / "train.command.json"
    ).exists()
    assert (
        "my_jev.train"
        in result[
            "commands"
        ]["train"]
    )
    assert (
        tmp_path
        / "runs"
        / "registry.json"
    ).exists()


def test_causal_experiment_builds_qwen_scalar_train_command(
    tmp_path,
):
    spec_path = tmp_path / "causal.toml"
    spec_path.write_text(
        """
[experiment]
name = "qwen-smoke"

[model]
backend = "causal_scalar"
backbone = "Qwen/Qwen3.5-4B-Base"
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = "all-linear"

[data]
train = "data/train.jsonl"
validation = "data/validation.jsonl"
calibration = "data/calibration.jsonl"
test = "data/test.jsonl"

[train]
epochs = 2
batch_size = 1
grad_accum = 16
learning_rate = 0.0001
weight_decay = 0.01
max_sequence_length = 1024
seed = 17
bf16 = true
gradient_checkpointing = true

[gates]
min_accuracy = 0.80
""".strip()
        + "\n",
        encoding="utf-8",
    )
    spec, _ = load_experiment_spec(
        spec_path
    )
    command = _train_command(
        spec,
        tmp_path,
        tmp_path / "run",
    )

    assert "my_jev.train_causal" in command
    assert "--lora-r" in command
    assert command[
        command.index(
            "--lora-r"
        )
        + 1
    ] == "16"
    assert "--max-length" in command
    assert command[
        command.index(
            "--max-length"
        )
        + 1
    ] == "1024"
