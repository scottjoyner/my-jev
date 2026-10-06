"""Training defaults, and the recipes that document them.

`train_causal.py` sat at 20% coverage and `train.py` at 68%, with `parse_args`
untested in both. That is the wrong way round: the defaults *are* the contract for
what a bare invocation does, and a drifted default changes what a training run does
with no error anywhere.

The two trainers share five flags and diverge on six, and every divergence is
legitimate -- a LoRA run on a decoder backbone is not the same problem as a full
fine-tune of an encoder. Those are recorded explicitly below so a future reader does
not "fix" one toward the other, and so a *new* divergence has to be declared rather
than appearing by accident.

The recipes in `TRAINING_PLAN.md` currently restate the defaults exactly, so a bare
invocation and the documented one agree. That is asserted rather than assumed: if a
default moves and the recipe does not, the recipe silently becomes a pin of a value
nothing produces.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

from my_jev import train, train_causal

ROOT = Path(__file__).resolve().parent.parent
BASE = ("--train", "a.jsonl", "--valid", "b.jsonl", "--output", "c.pt")


def _namespace(module, *extra: str):
    """Call the module's real `parse_args` with a patched argv.

    Reading the defaults out of a constructed Namespace rather than re-declaring them
    here means the test cannot drift into asserting values the parser does not have.
    """
    with mock.patch.object(sys, "argv", ["prog", *BASE, *extra]):
        return module.parse_args()


def _defaults(module) -> dict[str, object]:
    return dict(vars(_namespace(module)))


TRAINERS = {"train": train, "train_causal": train_causal}


# --- required arguments ----------------------------------------------------


@pytest.mark.parametrize("name", sorted(TRAINERS))
def test_required_arguments_are_required(name):
    """A run with no data paths is not a run, and must not start one."""
    module = TRAINERS[name]
    for flag in ("--train", "--valid", "--output"):
        # The same argv with just that one flag dropped.
        argv = ["prog"]
        for index in range(0, len(BASE), 2):
            if BASE[index] != flag:
                argv += [BASE[index], BASE[index + 1]]
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit):
                module.parse_args()


# --- the shared contract ---------------------------------------------------


@pytest.mark.parametrize("name", sorted(TRAINERS))
def test_seed_and_weight_decay_agree_across_the_trainers(name):
    """The two flags that describe the run rather than the model.

    Reproducibility is the point of `--seed`, so it is one documented value rather
    than one per trainer.
    """
    defaults = _defaults(TRAINERS[name])
    assert defaults["seed"] == 17
    assert defaults["weight_decay"] == 0.01


@pytest.mark.parametrize("name", sorted(TRAINERS))
def test_device_defaults_to_unset_preserving_the_historic_autoselect(name):
    """`None` means "cuda when visible", which is how both behaved before --device.

    Pinned in both directions: an explicit `"cpu"` would silently move every default
    run off the accelerator, and an explicit `"cuda"` would break CPU-only hosts.
    """
    assert _defaults(TRAINERS[name])["device"] is None


@pytest.mark.parametrize("name", sorted(TRAINERS))
def test_an_explicit_device_is_carried_through(name):
    """The steering that was missing when a shared card could only cause an OOM."""
    assert _namespace(TRAINERS[name], "--device", "cpu").device == "cpu"


# --- the documented defaults ----------------------------------------------


#: What a bare invocation must keep. Each of these changes what a training run does
#: with no error if it moves, which is exactly the default worth writing down.
EXPECTED_DEFAULTS = {
    "train": {
        "backbone": "answerdotai/ModernBERT-base",
        "epochs": 3,
        "batch_size": 2,
        "grad_accum": 8,
        "lr": 2e-05,
        "weight_decay": 0.01,
        "max_state_length": 2048,
        "max_candidate_length": 192,
        "seed": 17,
        "head_kind": "option_query",
        "device": None,
    },
    "train_causal": {
        "backbone": "Qwen/Qwen3.5-4B-Base",
        "epochs": 2,
        "batch_size": 1,
        "grad_accum": 16,
        "lr": 1e-04,
        "weight_decay": 0.01,
        "max_length": 1024,
        "seed": 17,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": "all-linear",
        "device": None,
    },
}


@pytest.mark.parametrize("name", sorted(EXPECTED_DEFAULTS))
def test_documented_defaults_are_what_the_code_uses(name):
    defaults = _defaults(TRAINERS[name])
    for key, expected in EXPECTED_DEFAULTS[name].items():
        assert key in defaults, f"{name} has no {key} any more"
        actual = defaults[key]
        if isinstance(expected, float):
            assert actual == pytest.approx(expected), f"{name}.{key}"
        else:
            assert actual == expected, f"{name}.{key}"


def test_the_two_trainers_diverge_only_where_they_must():
    """Every shared-flag divergence is deliberate, and each one is named.

    A LoRA run on a decoder backbone and a full fine-tune of an encoder are different
    problems: the learning rates differ by an order of magnitude and the accumulation
    by two, because the gradient shapes are not comparable. Recording them means a
    later reader who notices the difference knows it is intended.
    """
    expected_divergences = {
        "backbone",     # encoder family vs decoder family
        "epochs",       # 3 vs 2
        "batch_size",   # 2 vs 1
        "grad_accum",   # 8 vs 16, compensating for the batch size
        "lr",           # 2e-5 for a full fine-tune vs 1e-4 for LoRA
    }
    plain, causal = _defaults(train), _defaults(train_causal)
    shared = set(plain) & set(causal)
    diverging = {flag for flag in shared if plain[flag] != causal[flag]}
    assert diverging == expected_divergences, (
        f"the trainers now diverge on {sorted(diverging - expected_divergences)}; if "
        "that is intended, say why in this test rather than leaving it to be found"
    )


def test_numeric_flags_are_parsed_as_numbers_not_strings():
    """A stringified hyperparameter trains something other than what was asked."""
    causal = _namespace(train_causal, "--lr", "5e-5", "--epochs", "7")
    assert causal.lr == pytest.approx(5e-5)
    assert causal.epochs == 7
    # And a non-numeric value is refused rather than silently coerced.
    with pytest.raises(SystemExit):
        _namespace(train_causal, "--epochs", "many")


def test_an_unrecognised_flag_is_refused_rather_than_ignored():
    """A typo in a hyperparameter is a silent change to the run."""
    with pytest.raises(SystemExit):
        _namespace(train_causal, "--lora-rn", "8")


# --- the documented recipes -----------------------------------------------


#: Flags the recipes set to something other than the default. Empty today: every
#: value shown is the default. Anything added here needs a reason, because an
#: undeclared divergence means the recipe no longer describes a bare invocation.
RECIPE_OVERRIDES: dict[str, set[str]] = {
    "my-jev-train": set(),
    "my-jev-train-causal": set(),
}

_PATH_FLAGS = {"--train", "--valid", "--output"}


def _recipes() -> list[tuple[str, dict[str, str]]]:
    """Every documented invocation, parsed line by line.

    Written line-based rather than as one regex because the obvious non-greedy regex
    matches a bare `--` from the first flag in every recipe, which silently reduced
    both recipes to a single empty flag name.
    """
    lines = (ROOT / "docs" / "TRAINING_PLAN.md").read_text().splitlines()
    found: list[tuple[str, dict[str, str]]] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not (stripped.startswith("my-jev-train")):
            index += 1
            continue
        command = stripped.rstrip(chr(92)).strip()
        flags: dict[str, str] = {}
        index += 1
        while index < len(lines):
            candidate = lines[index].strip().rstrip(chr(92)).strip()
            if not candidate.startswith("--"):
                break
            parts = candidate.split(None, 1)
            flags[parts[0]] = parts[1].strip() if len(parts) > 1 else ""
            index += 1
        found.append((command, flags))
    return found


def test_the_documented_recipes_exist():
    """Otherwise every check below passes by finding nothing."""
    commands = {command for command, _ in _recipes()}
    assert {"my-jev-train", "my-jev-train-causal"} <= commands


@pytest.mark.parametrize(
    ("command", "flags"), _recipes(), ids=[c for c, _ in _recipes()]
)
def test_every_documented_flag_still_exists(command, flags):
    """A renamed or removed flag would leave the recipe prescribing nothing."""
    defaults = _defaults(train if command == "my-jev-train" else train_causal)
    for flag in flags:
        dest = flag.lstrip("-").replace("-", "_")
        assert dest in defaults, f"{command} documents {flag}, which no longer exists"


@pytest.mark.parametrize(
    ("command", "flags"), _recipes(), ids=[c for c, _ in _recipes()]
)
def test_documented_recipe_values_are_the_defaults(command, flags):
    """So the recipe and a bare invocation agree.

    If this fails after a default moves, that is the question being asked: was the
    documented number the real intent -- in which case it belongs in the default too
    -- or a copy of a value that has since changed, in which case the recipe should be
    trimmed to the flags that actually matter.
    """
    defaults = _defaults(train if command == "my-jev-train" else train_causal)
    overrides = RECIPE_OVERRIDES[command]
    for flag, value in flags.items():
        if flag in _PATH_FLAGS or flag in overrides or not value:
            continue
        dest = flag.lstrip("-").replace("-", "_")
        if dest not in defaults or isinstance(defaults[dest], bool):
            continue
        assert str(defaults[dest]) == value, (
            f"{command} {flag} is documented as {value!r} but the code defaults to "
            f"{defaults[dest]!r}; declare it in RECIPE_OVERRIDES if that is intended"
        )