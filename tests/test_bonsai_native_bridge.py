from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import my_jev.bonsai_native as bonsai_native
from my_jev.agentic_synth import (
    generate_agent_policy_records,
)
from my_jev.bonsai_contract import (
    BonsaiDecisionContract,
)
from my_jev.prompt_contract import (
    PROMPT_CONTRACT_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "native"
    / "bonsai_bridge"
    / "bonsai_activation_bridge.cpp"
)
BUILD_SCRIPT = (
    ROOT
    / "scripts"
    / "build-bonsai-bridge.sh"
)


def _write_stub_headers(
    root: Path,
) -> None:
    (
        root
        / "ggml-backend.h"
    ).write_text(
        """
#pragma once
inline void ggml_backend_load_all() {}
""",
        encoding="utf-8",
    )

    (
        root
        / "llama.h"
    ).write_text(
        """
#pragma once
#include <cstddef>
#include <cstdint>

using llama_token = int32_t;
using llama_memory_t = void *;

struct llama_model {};
struct llama_context {};
struct llama_vocab {};

enum llama_pooling_type {
    LLAMA_POOLING_TYPE_NONE = 0,
};

struct llama_model_params {
    int32_t n_gpu_layers;
};

struct llama_context_params {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    llama_pooling_type pooling_type;
    bool no_perf;
};

struct llama_batch {
    int32_t n_tokens;
};

llama_model_params llama_model_default_params();
llama_model * llama_model_load_from_file(
    const char *,
    llama_model_params
);
void llama_model_free(llama_model *);
int32_t llama_model_n_embd(
    const llama_model *
);
int32_t llama_model_n_layer(
    const llama_model *
);
const llama_vocab * llama_model_get_vocab(
    const llama_model *
);
llama_context_params llama_context_default_params();
llama_context * llama_init_from_model(
    llama_model *,
    llama_context_params
);
void llama_free(llama_context *);
void llama_set_causal_attn(
    llama_context *,
    bool
);
llama_memory_t llama_get_memory(
    const llama_context *
);
void llama_memory_clear(
    llama_memory_t,
    bool
);
int32_t llama_tokenize(
    const llama_vocab *,
    const char *,
    int32_t,
    llama_token *,
    int32_t,
    bool,
    bool
);
llama_batch llama_batch_get_one(
    llama_token *,
    int32_t
);
int32_t llama_decode(
    llama_context *,
    llama_batch
);
void llama_synchronize(
    llama_context *
);
""",
        encoding="utf-8",
    )

    (
        root
        / "llama-ext.h"
    ).write_text(
        """
#pragma once
#include "llama.h"

void llama_set_embeddings_layer_inp(
    llama_context *,
    uint32_t,
    bool
);
float * llama_get_embeddings_layer_inp(
    llama_context *,
    uint32_t
);
void llama_set_embeddings_nextn(
    llama_context *,
    bool,
    bool
);
float * llama_get_embeddings_nextn(
    llama_context *
);
""",
        encoding="utf-8",
    )


def test_native_bridge_is_valid_cpp_against_prism_api_shape(
    tmp_path: Path,
):
    compiler = shutil.which(
        "g++"
    )
    if compiler is None:
        pytest.skip(
            "g++ is not available"
        )

    include = (
        tmp_path
        / "include"
    )
    include.mkdir()
    _write_stub_headers(
        include
    )

    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-fsyntax-only",
            "-I",
            str(
                include
            ),
            str(
                SOURCE
            ),
        ],
        cwd=ROOT,
        check=True,
    )


def test_bridge_build_script_is_valid_bash():
    subprocess.run(
        [
            "bash",
            "-n",
            str(
                BUILD_SCRIPT
            ),
        ],
        cwd=ROOT,
        check=True,
    )


def test_native_bridge_source_uses_exact_hidden_state_hooks():
    text = SOURCE.read_text(
        encoding="utf-8"
    )

    for symbol in (
        "llama_set_embeddings_layer_inp",
        "llama_get_embeddings_layer_inp",
        "llama_set_embeddings_nextn",
        "llama_get_embeddings_nextn",
        "llama_set_causal_attn",
        "llama_memory_clear",
    ):
        assert symbol in text

    assert (
        PROMPT_CONTRACT_VERSION
        in text
    )
    assert (
        "llama_model_n_embd"
        in text
    )
    assert (
        "llama_model_n_layer"
        in text
    )


def test_native_provider_builds_grouped_bridge_command(
    tmp_path: Path,
    monkeypatch,
):
    bridge = (
        tmp_path
        / "bridge"
    )
    bridge.write_text(
        "#!/bin/sh\n",
        encoding="utf-8",
    )
    bridge.chmod(
        0o755
    )
    model = (
        tmp_path
        / "bonsai.gguf"
    )
    model.write_bytes(
        b"model"
    )
    llama_root = (
        tmp_path
        / "llama"
    )
    (
        llama_root
        / "src"
    ).mkdir(
        parents=True,
    )
    (
        llama_root
        / "src"
        / "llama-ext.h"
    ).write_text(
        "staging\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        bonsai_native,
        "_sha256",
        lambda _: "a" * 64,
    )
    monkeypatch.setattr(
        bonsai_native,
        "_git_revision",
        lambda _: "b" * 40,
    )

    provider = (
        bonsai_native
        .BonsaiNativeProvider(
            bridge=bridge,
            model=model,
            llama_root=(
                llama_root
            ),
        )
    )
    record = (
        generate_agent_policy_records(
            1,
            seed=127,
        )[0]
    )

    state_file = (
        tmp_path
        / "state.txt"
    )
    state_file.write_text(
        record.state,
        encoding="utf-8",
    )

    option_files = []
    index = 0
    for name, question in (
        record.questions.items()
    ):
        for option in (
            question.options
            or []
        ):
            path = (
                tmp_path
                / f"option-{index}.txt"
            )
            path.write_text(
                option,
                encoding="utf-8",
            )
            option_files.append(
                (
                    name,
                    question.type.value,
                    option,
                    path,
                )
            )
            index += 1

    command = (
        provider._bridge_command(
            record=record,
            state_file=(
                state_file
            ),
            option_files=(
                option_files
            ),
            output=(
                tmp_path
                / "frame.bin"
            ),
        )
    )

    assert (
        command.count(
            "--tap"
        )
        == 3
    )
    assert (
        "--final-prenorm"
        in command
    )
    assert (
        command.count(
            "--question"
        )
        == len(
            record.questions
        )
    )
    assert (
        command.count(
            "--option"
        )
        == index
    )

    contract = (
        BonsaiDecisionContract()
    )
    assert (
        str(
            contract.hidden_size
        )
        in command
    )


def test_bridge_builder_auto_enables_hip_when_available():
    text = BUILD_SCRIPT.read_text(
        encoding="utf-8"
    )

    assert (
        "command -v hipcc"
        in text
    )
    assert (
        'EXTRA_ARGS+=("-DGGML_HIP=ON")'
        in text
    )
    assert (
        "-DGGML_HIP=*"
        in text
    )
