#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: build-bonsai-bridge.sh <PrismML-Eng/llama.cpp checkout> [build-dir]

Environment:
  MY_JEV_BONSAI_CMAKE_ARGS   Extra CMake arguments, shell-split.
EOF
  exit 64
}

LLAMA_ROOT="${1:-}"
BUILD_DIR="${2:-runs/build/bonsai-bridge}"

[[ -n "${LLAMA_ROOT}" ]] || usage
LLAMA_ROOT="$(cd "${LLAMA_ROOT}" && pwd)"

if [[ ! -f "${LLAMA_ROOT}/src/llama-ext.h" ]]; then
  echo "PrismML staging header not found: ${LLAMA_ROOT}/src/llama-ext.h" >&2
  exit 65
fi

for symbol in   llama_set_embeddings_layer_inp   llama_get_embeddings_layer_inp   llama_set_embeddings_nextn   llama_get_embeddings_nextn
do
  if ! grep -q "${symbol}" "${LLAMA_ROOT}/src/llama-ext.h"; then
    echo "required PrismML staging symbol missing: ${symbol}" >&2
    exit 66
  fi
done

mkdir -p "${BUILD_DIR}"

EXTRA_ARGS=()
if [[ -n "${MY_JEV_BONSAI_CMAKE_ARGS:-}" ]]; then
  # Intentional operator-controlled split for CMake flags.
  read -r -a EXTRA_ARGS <<<"${MY_JEV_BONSAI_CMAKE_ARGS}"
fi

cmake   -S native/bonsai_bridge   -B "${BUILD_DIR}"   -DPRISM_LLAMA_ROOT="${LLAMA_ROOT}"   -DCMAKE_BUILD_TYPE=Release   "${EXTRA_ARGS[@]}"

cmake   --build "${BUILD_DIR}"   --target my-jev-bonsai-bridge   -j "${MY_JEV_BUILD_JOBS:-$(nproc)}"

BRIDGE="${BUILD_DIR}/bin/my-jev-bonsai-bridge"
if [[ ! -x "${BRIDGE}" ]]; then
  candidate="$(
    find "${BUILD_DIR}" -type f -name my-jev-bonsai-bridge -perm -111 | head -n 1
  )"
  if [[ -z "${candidate}" ]]; then
    echo "bridge binary was not produced" >&2
    exit 67
  fi
  BRIDGE="${candidate}"
fi

RUNTIME_SHA="$(
  git -C "${LLAMA_ROOT}" rev-parse HEAD
)"
if [[ ! "${RUNTIME_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "could not resolve PrismML llama.cpp exact SHA" >&2
  exit 68
fi

cat > "${BUILD_DIR}/bridge-build-receipt.json" <<EOF
{
  "schema_version": 1,
  "kind": "my-jev-bonsai-native-bridge-build",
  "runtime_root": "${LLAMA_ROOT}",
  "runtime_git_sha": "${RUNTIME_SHA}",
  "bridge": "${BRIDGE}",
  "dispatch_allowed": false,
  "runtime_authority_changed": false
}
EOF

echo "Bonsai native activation bridge built"
echo "bridge: ${BRIDGE}"
echo "runtime_sha: ${RUNTIME_SHA}"
echo "receipt: ${BUILD_DIR}/bridge-build-receipt.json"
