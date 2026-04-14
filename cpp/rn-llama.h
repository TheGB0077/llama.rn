#ifndef RNLLAMA_H
#define RNLLAMA_H

#include <sstream>
#include <iostream>
#include <thread>
#include <codecvt>
#include "common.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"
#include "llama-model.h"
#include "llama-impl.h"
#include "sampling.h"
#include "nlohmann/json.hpp"
#include "rn-tts.h"
#include "rn-completion.h"
#if defined(__ANDROID__)
#include <android/log.h>
#endif

using json = nlohmann::ordered_json;

namespace rnllama {

std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token);

std::string tokens_to_str(llama_context *ctx, const std::vector<llama_token>::const_iterator begin, const std::vector<llama_token>::const_iterator end);

lm_ggml_type kv_cache_type_from_str(const std::string & s);

enum llama_flash_attn_type flash_attn_type_from_str(const std::string & s);

struct llama_rn_tokenize_result {
  std::vector<llama_token> tokens;
};

struct llama_rn_context {
    llama_model *model = nullptr;
    float loading_progress = 0;
    bool is_load_interrupted = false;
    common_params params;
    common_init_result_ptr llama_init;
    llama_context *ctx = nullptr;
    int n_ctx;

    lm_ggml_threadpool *threadpool = nullptr;
    lm_ggml_threadpool *threadpool_batch = nullptr;

    ~llama_rn_context();

    bool loadModel(common_params &params_);
    void cleanupThreadpools();
    bool attachThreadpoolsIfAvailable();

    llama_rn_tokenize_result tokenize(const std::string &text);

    // Completion
    llama_rn_context_completion* completion = nullptr;

    // TTS
    llama_rn_context_tts *tts_wrapper = nullptr;
    bool has_vocoder = false;
    bool initVocoder(const std::string &vocoder_model_path, int batch_size = -1);
    bool isVocoderEnabled() const;
    void releaseVocoder();

    void clearCache(bool clear_data = false);
};

inline void llama_batch_add(llama_batch *batch, llama_token id, llama_pos pos, std::vector<llama_seq_id> seq_ids, bool logits) {
    batch->token   [batch->n_tokens] = id;
    batch->pos     [batch->n_tokens] = pos;
    batch->n_seq_id[batch->n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); i++) {
        batch->seq_id[batch->n_tokens][i] = seq_ids[i];
    }
    batch->logits  [batch->n_tokens] = logits ? 1 : 0;
    batch->n_tokens += 1;
}

std::string get_backend_devices_info();

void log(const char *level, const char *function, int line, const char *format, ...);

extern bool rnllama_verbose;

#if RNLLAMA_VERBOSE != 1
#define LOG_VERBOSE(MSG, ...)
#else
#define LOG_VERBOSE(MSG, ...)                                       \
    do                                                              \
    {                                                               \
        if (rnllama_verbose)                                        \
        {                                                           \
            log("VERBOSE", __func__, __LINE__, MSG, ##__VA_ARGS__); \
        }                                                           \
    } while (0)
#endif

#define LOG_ERROR(MSG, ...) log("ERROR", __func__, __LINE__, MSG, ##__VA_ARGS__)
#define LOG_WARNING(MSG, ...) log("WARNING", __func__, __LINE__, MSG, ##__VA_ARGS__)
#define LOG_INFO(MSG, ...) log("INFO", __func__, __LINE__, MSG, ##__VA_ARGS__)

} // namespace rnllama

#endif /* RNLLAMA_H */
