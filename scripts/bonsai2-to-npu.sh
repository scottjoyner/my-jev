#!/usr/bin/env bash
# Convert Ternary Bonsai 2 to an AMD NPU (Ryzen AI) model directory.
#
# Why this is a script and not a note: the conversion is the only step between a
# model that runs today and one that runs on the XDNA2 NPU. Everything upstream is
# measured and working -- see the notes at the bottom. This makes the remaining step
# a single command the moment hybrid-orchestrator is available, instead of a research
# task.
#
# Status on this host: hybrid-orchestrator is NOT installed and is NOT on PyPI
#   ERROR: No matching distribution found for hybrid-orchestrator
# It ships with the Ryzen AI NPU SDK / AMD GitHub releases, so this script fails
# fast with the exact remedy rather than something cryptic.
#
# Usage:
#   scripts/bonsai2-to-npu.sh [GGUF] [OUT_DIR]
# Defaults:
#   GGUF     /media/scott/data/models/prism-ml/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf
#   OUT_DIR  /media/scott/data/amd-npu-models/Ternary-Bonsai-2-27B-npu

set -euo pipefail

GGUF="${1:-/media/scott/data/models/prism-ml/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf}"
OUT="${2:-/media/scott/data/amd-npu-models/Ternary-Bonsai-2-27B-npu}"

die() { echo "error: $*" >&2; exit 1; }

[ -f "$GGUF" ] || die "no such GGUF: $GGUF"

if ! command -v hybrid-orchestrator >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: hybrid-orchestrator is not on PATH.

It is the tool that turns a GGUF/HF model into an NPU model directory
(model.onnx stub + model.bin + dd_metastate_*.meta/.ctrlpkt/.super/.fconst/.state),
and it is not distributed on PyPI. Install it from the Ryzen AI NPU SDK or the
AMD GitHub release that matches this EP build, then re-run.

EP build here: amdxdna 2.25.260102.56, NPU firmware 1.1.2.65, aie2p 6x8
genai: onnxruntime-genai-ryzenai 0.14.0, onnxruntime 1.27.0

Reference for the output shape -- AMD's shipped model at
/media/scott/data/amd-npu-models/Llama-3.2-3B-Instruct has:
  model.onnx                 4 nodes (Shape, Constant, Equal, If) -- a stub
  model.bin                  1.7 GB
  dd_metastate_*             the actual model
  genai_config.json          provider "RyzenAI", hybrid_opt_token_backend=npu
EOF
  exit 1
fi

mkdir -p "$OUT"
echo "converting $GGUF -> $OUT"

# --npu-onnx emits the NPU model bundle rather than a CPU ONNX graph.
hybrid-orchestrator \
  --gguf "$GGUF" \
  --npu-onnx \
  --output-dir "$OUT" \
  --tokenizer /media/scott/data/finetune-staging/models/K2-Horizon-0.9B/tokenizer.json

echo
echo "verify with:"
echo "  cd $OUT && python3 - <<'EOF'"
echo "  import onnxruntime_genai as og, time"
echo "  og.register_execution_provider_library('RyzenAI', EP_SO)"
echo "  m = og.Model('.'); tk = og.Tokenizer(m)"
echo "  ids = tk.encode('The capital of France is')"
echo "  p = og.GeneratorParams(m)"
echo "  p.set_search_options(do_sample=False, max_length=len(ids)+32)"
echo "  g = og.Generator(m, p); g.append_tokens(ids)"
echo "  while not g.is_done():"
echo "      g.get_logits(); g.generate_next_token(); print(tk.decode([g.get_next_tokens()[0]]), end='')"
echo "  EOF"
echo
echo "then confirm the NPU is actually doing the work by watching the rail:"
echo "  watch -n0.2 cat /sys/bus/pci/devices/0000:c4:00.1/hwmon/hwmon7/power1_input"
echo "Residency looks like ~2.6 W. A graph that only got a few nodes onto the NPU"
echo "reads ~0.5 W -- that ratio is the check that matters."

# ---------------------------------------------------------------------------
# Measured context, so this is not re-derived from scratch later.
#
#   Ternary Bonsai 2 27B, 1.75 bpw ternary, group 128, 5.53 GiB, 26.90B params
#     CPU only  (-ngl 0)   4.87 tok/s decode, 3.65 W package
#     iGPU      (-ngl -1) 15.69 tok/s decode
#
#   AMD Llama-3.2-3B on the NPU via genai+metastate
#     17.59 tok/s, NPU rail 2.65 W avg / 3.09 W peak, CPU package 3.54 W
#     (our hand-built K2 graph drew 0.50 W -- that is the "not resident" tell)
#
# Why Bonsai 2 and not a dense transformer: it is qwen35, a hybrid with SSM
# layers, so its operators are the ones the EP's fused kernels are built for
# (SSMLPMul, SSMlpFuse, LinearAttention, Conv). Its dimensions -- d_model 5120,
# ssm inner 6144, group_count 16, state_size 128 -- hit the JIT's compiled shape
# table exactly.
# ---------------------------------------------------------------------------
