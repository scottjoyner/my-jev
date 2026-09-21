#!/usr/bin/env bash
set -euo pipefail

BACKBONE="${MY_JEV_R9700_BACKBONE:-answerdotai/ModernBERT-base}"
MIN_GIB="${MY_JEV_R9700_MIN_GIB:-28}"
DEVICE_PATTERN="${MY_JEV_R9700_DEVICE_PATTERN:-R9700}"
MIN_FREE_GIB="${MY_JEV_R9700_MIN_FREE_GIB:-20}"
WORK_ROOT="${MY_JEV_R9700_WORKTREE_ROOT:-$HOME/my-jev-r9700-worktrees}"
PREFETCH_MODEL=true

usage() {
  cat >&2 <<EOF
usage: $0 [--no-prefetch-model]

Checks the local R9700/ROCm Python, required runtime packages, BF16,
writable cache/work space, free disk, and the pinned ModernBERT backbone.

Environment overrides:
  MY_JEV_R9700_BACKBONE      default: ${BACKBONE}
  MY_JEV_R9700_MIN_GIB       default: ${MIN_GIB}
  MY_JEV_R9700_DEVICE_PATTERN default: ${DEVICE_PATTERN}
  MY_JEV_R9700_MIN_FREE_GIB  default: ${MIN_FREE_GIB}
  MY_JEV_R9700_WORKTREE_ROOT default: ${WORK_ROOT}
EOF
  exit 64
}

case "${1:-}" in
  "")
    ;;
  --no-prefetch-model)
    PREFETCH_MODEL=false
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage
    ;;
esac

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "R9700 host preflight requires Linux" >&2
  exit 65
fi

for command in python git df; do
  command -v "${command}" >/dev/null || {
    echo "${command} is required" >&2
    exit 66
  }
done

mkdir -p "${WORK_ROOT}"

FREE_KIB="$(
  df -Pk "${WORK_ROOT}"     | awk 'NR == 2 {print $4}'
)"
MIN_FREE_KIB="$(
  python - "${MIN_FREE_GIB}" <<'PY'
import sys

print(
    int(
        float(sys.argv[1])
        * 1024
        * 1024
    )
)
PY
)"

if [[ "${FREE_KIB}" -lt "${MIN_FREE_KIB}" ]]; then
  FREE_GIB="$(
    python - "${FREE_KIB}" <<'PY'
import sys

print(
    f"{int(sys.argv[1]) / (1024 * 1024):.2f}"
)
PY
  )"
  echo "insufficient free space for R9700 run: ${FREE_GIB} GiB < ${MIN_FREE_GIB} GiB" >&2
  exit 67
fi

ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd
)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

PREFETCH_FLAG=0
if ${PREFETCH_MODEL}; then
  PREFETCH_FLAG=1
fi

python -   "${BACKBONE}"   "${MIN_GIB}"   "${DEVICE_PATTERN}"   "${PREFETCH_FLAG}"   "${WORK_ROOT}"   <<'PY'
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

backbone = sys.argv[1]
minimum_gib = float(sys.argv[2])
device_pattern = sys.argv[3].lower()
prefetch = bool(int(sys.argv[4]))
work_root = Path(sys.argv[5]).expanduser().resolve()

required_modules = (
    "torch",
    "transformers",
    "pydantic",
    "numpy",
    "safetensors",
    "tqdm",
)

missing = []
for name in required_modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(
            f"{name}: {exc}"
        )

if missing:
    raise SystemExit(
        "missing/broken R9700 runtime dependencies:\n- "
        + "\n- ".join(missing)
    )

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)

print(
    json.dumps(
        {
            "python": sys.executable,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "transformers": transformers.__version__,
            "work_root": str(work_root),
            "backbone": backbone,
            "device_pattern": device_pattern,
            "prefetch_model": prefetch,
        },
        indent=2,
        sort_keys=True,
    )
)

if not torch.version.hip:
    raise SystemExit(
        "PyTorch is not reporting a ROCm/HIP runtime"
    )
if not torch.cuda.is_available():
    raise SystemExit(
        "ROCm accelerator is not visible through torch.cuda"
    )
if not torch.cuda.is_bf16_supported():
    raise SystemExit(
        "BF16 is not reported as supported"
    )

matches = []
for index in range(
    torch.cuda.device_count()
):
    props = torch.cuda.get_device_properties(
        index
    )
    name = str(
        props.name
    )
    gib = (
        float(
            props.total_memory
        )
        / (1024**3)
    )
    print(
        f"device[{index}]={name} "
        f"memory_gib={gib:.2f}"
    )
    if (
        device_pattern in name.lower()
        and gib >= minimum_gib
    ):
        matches.append(
            index
        )

if not matches:
    raise SystemExit(
        "no visible accelerator matched "
        f"{device_pattern!r} with >= "
        f"{minimum_gib:.1f} GiB"
    )

cache_home = Path(
    os.environ.get(
        "HF_HOME",
        Path.home()
        / ".cache"
        / "huggingface",
    )
).expanduser()
cache_home.mkdir(
    parents=True,
    exist_ok=True,
)

probe = (
    cache_home
    / ".my-jev-write-probe"
)
probe.write_text(
    "ok\n",
    encoding="utf-8",
)
probe.unlink()

if not prefetch:
    print(
        "model prefetch skipped by request"
    )
    raise SystemExit(0)

print(
    f"prefetching/loading {backbone}"
)

config = AutoConfig.from_pretrained(
    backbone
)
tokenizer = (
    AutoTokenizer.from_pretrained(
        backbone
    )
)
model = AutoModel.from_pretrained(
    backbone,
    config=config,
)

device = torch.device(
    f"cuda:{matches[0]}"
)
model = model.to(
    device=device,
    dtype=torch.bfloat16,
)
model.eval()

inputs = tokenizer(
    "my-jev R9700 readiness probe",
    return_tensors="pt",
)
inputs = {
    key: value.to(
        device
    )
    for key, value in inputs.items()
}

with torch.inference_mode():
    output = model(
        **inputs
    )

hidden = getattr(
    output,
    "last_hidden_state",
    None,
)
if hidden is None:
    raise SystemExit(
        "backbone readiness probe returned "
        "no last_hidden_state"
    )
if not torch.isfinite(
    hidden
).all():
    raise SystemExit(
        "backbone readiness probe produced "
        "non-finite values"
    )

print(
    "backbone BF16 GPU readiness probe passed: "
    f"shape={tuple(hidden.shape)} "
    f"dtype={hidden.dtype} "
    f"device={hidden.device}"
)

del output
del model
torch.cuda.empty_cache()
PY

echo "R9700 local host preflight passed"
echo "rocm_python: $(command -v python)"
echo "work_root: ${WORK_ROOT}"
echo "free_kib: ${FREE_KIB}"
