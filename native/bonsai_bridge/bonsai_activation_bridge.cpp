#include "ggml-backend.h"
#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char * kFrameVersion = "my-jev-activation-frame-v1";
constexpr const char * kProvider = "bonsai2-ternary-llama-cpp";
constexpr const char * kPromptContract = "my-jev-typed-decision-prompt-v1";

struct Question {
    std::string name;
    std::string type;
    std::vector<std::string> labels;
    std::vector<std::string> files;
};

struct Args {
    std::string model;
    std::string model_sha256;
    std::string runtime_revision;
    std::string state_file;
    std::string output;
    int hidden_size = 5120;
    int num_layers = 64;
    int n_ctx = 2048;
    int n_gpu_layers = 999;
    bool final_prenorm = false;
    std::vector<uint32_t> taps;
    std::vector<Question> questions;
};

struct PromptActivations {
    int32_t n_tokens = 0;
    std::vector<std::vector<float>> taps;
};

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

std::string read_text(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail("cannot open text input: " + path);
    }
    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

std::string json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 16);
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (ch < 0x20) {
                    static constexpr char hex[] = "0123456789abcdef";
                    out += "\\u00";
                    out += hex[(ch >> 4) & 0x0f];
                    out += hex[ch & 0x0f];
                } else {
                    out += static_cast<char>(ch);
                }
        }
    }
    return out;
}

int parse_int(const std::string & value, const char * name) {
    try {
        size_t used = 0;
        const long parsed = std::stol(value, &used, 10);
        if (used != value.size() ||
            parsed < std::numeric_limits<int>::min() ||
            parsed > std::numeric_limits<int>::max()) {
            fail(std::string("invalid ") + name + ": " + value);
        }
        return static_cast<int>(parsed);
    } catch (const std::exception &) {
        fail(std::string("invalid ") + name + ": " + value);
    }
}

Args parse_args(int argc, char ** argv) {
    Args args;
    Question * current = nullptr;

    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];

        auto require = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                fail(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (flag == "--model") {
            args.model = require("--model");
        } else if (flag == "--model-sha256") {
            args.model_sha256 = require("--model-sha256");
        } else if (flag == "--runtime-revision") {
            args.runtime_revision = require("--runtime-revision");
        } else if (flag == "--state-file") {
            args.state_file = require("--state-file");
        } else if (flag == "--output") {
            args.output = require("--output");
        } else if (flag == "--hidden-size") {
            args.hidden_size = parse_int(require("--hidden-size"), "hidden size");
        } else if (flag == "--num-layers") {
            args.num_layers = parse_int(require("--num-layers"), "layer count");
        } else if (flag == "--ctx") {
            args.n_ctx = parse_int(require("--ctx"), "context size");
        } else if (flag == "--gpu-layers") {
            args.n_gpu_layers = parse_int(require("--gpu-layers"), "GPU layer count");
        } else if (flag == "--tap") {
            const int tap = parse_int(require("--tap"), "tap");
            if (tap < 0) {
                fail("tap must be non-negative");
            }
            args.taps.push_back(static_cast<uint32_t>(tap));
        } else if (flag == "--final-prenorm") {
            args.final_prenorm = true;
        } else if (flag == "--question") {
            Question question;
            question.name = require("--question name");
            question.type = require("--question type");
            args.questions.push_back(std::move(question));
            current = &args.questions.back();
        } else if (flag == "--option") {
            if (current == nullptr) {
                fail("--option requires a preceding --question");
            }
            current->labels.push_back(require("--option label"));
            current->files.push_back(require("--option file"));
        } else if (flag == "--help" || flag == "-h") {
            std::cout
                << "my-jev Bonsai activation bridge\n"
                << "  --model MODEL.gguf\n"
                << "  --model-sha256 HEX\n"
                << "  --runtime-revision GIT_SHA\n"
                << "  --state-file STATE.txt\n"
                << "  --tap LAYER (repeatable)\n"
                << "  --final-prenorm\n"
                << "  --question NAME TYPE --option LABEL FILE ...\n"
                << "  --output FRAME.bin\n";
            std::exit(0);
        } else {
            fail("unknown argument: " + flag);
        }
    }

    if (args.model.empty() ||
        args.model_sha256.empty() ||
        args.runtime_revision.empty() ||
        args.state_file.empty() ||
        args.output.empty()) {
        fail("model, provenance, state, and output arguments are required");
    }
    if (args.hidden_size <= 0 || args.num_layers <= 0 || args.n_ctx <= 0) {
        fail("hidden size, layer count, and context size must be positive");
    }
    if (args.taps.empty() && !args.final_prenorm) {
        fail("at least one activation tap is required");
    }
    if (args.questions.empty()) {
        fail("at least one typed question is required");
    }
    for (const Question & question : args.questions) {
        if (question.name.empty() || question.type.empty()) {
            fail("question name/type must be non-empty");
        }
        if (question.labels.empty() ||
            question.labels.size() != question.files.size()) {
            fail("every question must have one or more labeled option files");
        }
    }

    std::sort(args.taps.begin(), args.taps.end());
    if (std::adjacent_find(args.taps.begin(), args.taps.end()) != args.taps.end()) {
        fail("activation taps must be unique");
    }
    for (const uint32_t tap : args.taps) {
        if (tap >= static_cast<uint32_t>(args.num_layers)) {
            fail("activation tap exceeds configured layer count");
        }
    }

    return args;
}

std::vector<llama_token> tokenize(
    const llama_vocab * vocab,
    const std::string & text) {
    const int32_t count = -llama_tokenize(
        vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        nullptr,
        0,
        true,
        true);
    if (count <= 0) {
        fail("failed to size tokenized prompt");
    }

    std::vector<llama_token> tokens(static_cast<size_t>(count));
    const int32_t actual = llama_tokenize(
        vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        true,
        true);
    if (actual < 0) {
        fail("failed to tokenize prompt");
    }
    tokens.resize(static_cast<size_t>(actual));
    return tokens;
}

PromptActivations capture_prompt(
    llama_context * ctx,
    const llama_vocab * vocab,
    const std::string & text,
    const std::vector<uint32_t> & taps,
    bool final_prenorm,
    int hidden_size,
    int n_ctx) {
    std::vector<llama_token> tokens = tokenize(vocab, text);
    if (tokens.empty()) {
        fail("prompt tokenized to zero tokens");
    }
    if (tokens.size() > static_cast<size_t>(n_ctx)) {
        fail(
            "prompt token count " + std::to_string(tokens.size()) +
            " exceeds configured context " + std::to_string(n_ctx));
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_get_one(
        tokens.data(),
        static_cast<int32_t>(tokens.size()));

    const int32_t rc = llama_decode(ctx, batch);
    if (rc != 0) {
        fail("llama_decode failed with code " + std::to_string(rc));
    }
    llama_synchronize(ctx);

    PromptActivations result;
    result.n_tokens = static_cast<int32_t>(tokens.size());
    result.taps.reserve(taps.size() + (final_prenorm ? 1 : 0));

    const size_t row_count =
        static_cast<size_t>(result.n_tokens) * static_cast<size_t>(hidden_size);

    for (const uint32_t layer : taps) {
        const float * data = llama_get_embeddings_layer_inp(ctx, layer);
        if (data == nullptr) {
            fail("layer input activation pointer is null for layer " + std::to_string(layer));
        }
        result.taps.emplace_back(data, data + row_count);
    }

    if (final_prenorm) {
        const float * data = llama_get_embeddings_nextn(ctx);
        if (data == nullptr) {
            fail("final pre-norm activation pointer is null");
        }
        result.taps.emplace_back(data, data + row_count);
    }

    return result;
}

std::vector<float> mean_pool(
    const PromptActivations & prompt,
    size_t tap_index,
    int hidden_size) {
    if (tap_index >= prompt.taps.size()) {
        fail("tap index out of range during option pooling");
    }
    if (prompt.n_tokens <= 0) {
        fail("cannot pool empty option prompt");
    }

    std::vector<float> pooled(static_cast<size_t>(hidden_size), 0.0f);
    const std::vector<float> & values = prompt.taps[tap_index];

    for (int32_t token = 0; token < prompt.n_tokens; ++token) {
        const float * row =
            values.data() + static_cast<size_t>(token) * static_cast<size_t>(hidden_size);
        for (int hidden = 0; hidden < hidden_size; ++hidden) {
            pooled[static_cast<size_t>(hidden)] += row[hidden];
        }
    }

    const float scale = 1.0f / static_cast<float>(prompt.n_tokens);
    for (float & value : pooled) {
        value *= scale;
    }
    return pooled;
}

void write_u32_be(std::ostream & output, uint32_t value) {
    const unsigned char bytes[4] = {
        static_cast<unsigned char>((value >> 24) & 0xff),
        static_cast<unsigned char>((value >> 16) & 0xff),
        static_cast<unsigned char>((value >> 8) & 0xff),
        static_cast<unsigned char>(value & 0xff),
    };
    output.write(
        reinterpret_cast<const char *>(bytes),
        sizeof(bytes));
}

std::string build_header(
    const Args & args,
    int32_t state_tokens,
    size_t tap_count,
    size_t option_count,
    size_t state_bytes,
    size_t state_mask_bytes,
    size_t option_bytes,
    size_t option_mask_bytes) {
    std::string json;
    json += "{";
    json += "\"dtype\":\"float32\",";
    json += "\"groups\":[";

    size_t option_offset = 0;
    for (size_t q = 0; q < args.questions.size(); ++q) {
        if (q > 0) {
            json += ",";
        }
        const Question & question = args.questions[q];
        const size_t start = option_offset;
        const size_t end = start + question.labels.size();

        json += "{";
        json += "\"name\":\"" + json_escape(question.name) + "\",";
        json += "\"option_end\":" + std::to_string(end) + ",";
        json += "\"option_start\":" + std::to_string(start) + ",";
        json += "\"options\":[";
        for (size_t i = 0; i < question.labels.size(); ++i) {
            if (i > 0) {
                json += ",";
            }
            json += "\"" + json_escape(question.labels[i]) + "\"";
        }
        json += "],";
        json += "\"record_index\":0,";
        json += "\"type\":\"" + json_escape(question.type) + "\"";
        json += "}";

        option_offset = end;
    }

    json += "],";
    json += "\"model_sha256\":\"" + json_escape(args.model_sha256) + "\",";
    json += "\"option_bytes\":" + std::to_string(option_bytes) + ",";
    json += "\"option_mask_bytes\":" + std::to_string(option_mask_bytes) + ",";
    json += "\"option_mask_shape\":[1," + std::to_string(option_count) + "],";
    json += "\"option_shape\":[1," + std::to_string(option_count) + "," +
        std::to_string(tap_count) + "," + std::to_string(args.hidden_size) + "],";
    json += "\"prompt_contract\":\"" + std::string(kPromptContract) + "\",";
    json += "\"provider\":\"" + std::string(kProvider) + "\",";
    json += "\"runtime_revision\":\"" + json_escape(args.runtime_revision) + "\",";
    json += "\"state_bytes\":" + std::to_string(state_bytes) + ",";
    json += "\"state_mask_bytes\":" + std::to_string(state_mask_bytes) + ",";
    json += "\"state_mask_shape\":[1," + std::to_string(state_tokens) + "],";
    json += "\"state_shape\":[1," + std::to_string(tap_count) + "," +
        std::to_string(state_tokens) + "," + std::to_string(args.hidden_size) + "],";
    json += "\"version\":\"" + std::string(kFrameVersion) + "\"";
    json += "}";

    return json;
}

void write_frame(
    const Args & args,
    const PromptActivations & state,
    const std::vector<std::vector<std::vector<float>>> & options) {
    const size_t tap_count = state.taps.size();
    const size_t option_count = options.size();

    if (tap_count == 0 || option_count == 0) {
        fail("cannot write an empty activation frame");
    }

    const size_t state_float_count =
        tap_count *
        static_cast<size_t>(state.n_tokens) *
        static_cast<size_t>(args.hidden_size);
    const size_t option_float_count =
        option_count *
        tap_count *
        static_cast<size_t>(args.hidden_size);

    const size_t state_bytes = state_float_count * sizeof(float);
    const size_t state_mask_bytes = static_cast<size_t>(state.n_tokens);
    const size_t option_bytes = option_float_count * sizeof(float);
    const size_t option_mask_bytes = option_count;

    const std::string header = build_header(
        args,
        state.n_tokens,
        tap_count,
        option_count,
        state_bytes,
        state_mask_bytes,
        option_bytes,
        option_mask_bytes);

    if (header.size() > std::numeric_limits<uint32_t>::max()) {
        fail("activation frame header is too large");
    }

    std::ofstream output(args.output, std::ios::binary);
    if (!output) {
        fail("cannot create activation frame: " + args.output);
    }

    write_u32_be(output, static_cast<uint32_t>(header.size()));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));

    for (const std::vector<float> & tap : state.taps) {
        output.write(
            reinterpret_cast<const char *>(tap.data()),
            static_cast<std::streamsize>(tap.size() * sizeof(float)));
    }

    const std::vector<uint8_t> state_mask(
        static_cast<size_t>(state.n_tokens),
        static_cast<uint8_t>(1));
    output.write(
        reinterpret_cast<const char *>(state_mask.data()),
        static_cast<std::streamsize>(state_mask.size()));

    for (const auto & option : options) {
        if (option.size() != tap_count) {
            fail("option tap count does not match state tap count");
        }
        for (const std::vector<float> & tap : option) {
            if (tap.size() != static_cast<size_t>(args.hidden_size)) {
                fail("pooled option hidden size mismatch");
            }
            output.write(
                reinterpret_cast<const char *>(tap.data()),
                static_cast<std::streamsize>(tap.size() * sizeof(float)));
        }
    }

    const std::vector<uint8_t> option_mask(
        option_count,
        static_cast<uint8_t>(1));
    output.write(
        reinterpret_cast<const char *>(option_mask.data()),
        static_cast<std::streamsize>(option_mask.size()));

    if (!output) {
        fail("failed while writing activation frame");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const Args args = parse_args(argc, argv);

        ggml_backend_load_all();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = args.n_gpu_layers;

        llama_model * model = llama_model_load_from_file(
            args.model.c_str(),
            model_params);
        if (model == nullptr) {
            fail("unable to load Bonsai model");
        }

        const int32_t actual_hidden = llama_model_n_embd(model);
        const int32_t actual_layers = llama_model_n_layer(model);
        if (actual_hidden != args.hidden_size) {
            llama_model_free(model);
            fail(
                "model hidden size " + std::to_string(actual_hidden) +
                " does not match contract " + std::to_string(args.hidden_size));
        }
        if (actual_layers != args.num_layers) {
            llama_model_free(model);
            fail(
                "model layer count " + std::to_string(actual_layers) +
                " does not match contract " + std::to_string(args.num_layers));
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = static_cast<uint32_t>(args.n_ctx);
        ctx_params.n_batch = static_cast<uint32_t>(args.n_ctx);
        ctx_params.n_ubatch = static_cast<uint32_t>(args.n_ctx);
        ctx_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        ctx_params.no_perf = false;

        llama_context * ctx = llama_init_from_model(model, ctx_params);
        if (ctx == nullptr) {
            llama_model_free(model);
            fail("unable to initialize Bonsai context");
        }

        llama_set_causal_attn(ctx, true);
        for (const uint32_t tap : args.taps) {
            llama_set_embeddings_layer_inp(ctx, tap, true);
        }
        if (args.final_prenorm) {
            llama_set_embeddings_nextn(ctx, true, false);
        }

        const PromptActivations state = capture_prompt(
            ctx,
            vocab,
            read_text(args.state_file),
            args.taps,
            args.final_prenorm,
            args.hidden_size,
            args.n_ctx);

        std::vector<std::vector<std::vector<float>>> option_vectors;
        for (const Question & question : args.questions) {
            for (const std::string & path : question.files) {
                const PromptActivations option = capture_prompt(
                    ctx,
                    vocab,
                    read_text(path),
                    args.taps,
                    args.final_prenorm,
                    args.hidden_size,
                    args.n_ctx);

                std::vector<std::vector<float>> pooled;
                pooled.reserve(option.taps.size());
                for (size_t tap = 0; tap < option.taps.size(); ++tap) {
                    pooled.push_back(
                        mean_pool(
                            option,
                            tap,
                            args.hidden_size));
                }
                option_vectors.push_back(std::move(pooled));
            }
        }

        write_frame(args, state, option_vectors);

        std::cerr
            << "my-jev Bonsai activation bridge: "
            << "state_tokens=" << state.n_tokens
            << " options=" << option_vectors.size()
            << " taps=" << state.taps.size()
            << " hidden=" << args.hidden_size
            << " output=" << args.output
            << "\n";

        llama_free(ctx);
        llama_model_free(model);
        return 0;
    } catch (const std::exception & exc) {
        std::cerr << "my-jev Bonsai activation bridge error: "
                  << exc.what() << "\n";
        return 1;
    }
}
