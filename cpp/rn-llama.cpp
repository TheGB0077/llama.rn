#include "rn-llama.h"
#include "ggml-cpu.h"
#include "rn-tts.h"
#include "rn-common.hpp"

#include <cstdarg>

namespace rnllama {

std::string get_backend_devices_info() {
    return backend_devices_info();
}

static const std::vector<lm_ggml_type> kv_cache_types = {
    LM_GGML_TYPE_F32,
    LM_GGML_TYPE_F16,
    LM_GGML_TYPE_BF16,
    LM_GGML_TYPE_Q8_0,
    LM_GGML_TYPE_Q4_0,
    LM_GGML_TYPE_Q4_1,
    LM_GGML_TYPE_IQ4_NL,
    LM_GGML_TYPE_Q5_0,
    LM_GGML_TYPE_Q5_1,
};

lm_ggml_type kv_cache_type_from_str(const std::string & s) {
    if (s.empty()) {
        return LM_GGML_TYPE_F16;
    }

    for (const auto & type : kv_cache_types) {
        if (lm_ggml_type_name(type) == s) {
            return type;
        }
    }

    return LM_GGML_TYPE_F16;
}

enum llama_flash_attn_type flash_attn_type_from_str(const std::string & s) {
    if (s == "on" || s == "true" || s == "1") return LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (s == "off" || s == "false" || s == "0") return LLAMA_FLASH_ATTN_TYPE_DISABLED;
    return LLAMA_FLASH_ATTN_TYPE_AUTO;
}

void log(const char *level, const char *function, int line,
                       const char *format, ...)
{
    va_list args;
    #if defined(__ANDROID__)
        char prefix[256];
        snprintf(prefix, sizeof(prefix), "%s:%d %s", function, line, format);

        va_start(args, format);
        android_LogPriority priority;
        if (strcmp(level, "ERROR") == 0) {
            priority = ANDROID_LOG_ERROR;
        } else if (strcmp(level, "WARNING") == 0) {
            priority = ANDROID_LOG_WARN;
        } else if (strcmp(level, "INFO") == 0) {
            priority = ANDROID_LOG_INFO;
        } else {
            priority = ANDROID_LOG_DEBUG;
        }
        __android_log_vprint(priority, "RNLlama", prefix, args);
        va_end(args);
    #else
        printf("[%s] %s:%d ", level, function, line);
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
        printf("\n");
    #endif
}

std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token)
{
    std::string out = token == -1 ? "" : common_token_to_piece(ctx, token);
    if (out.size() == 1 && (out[0] & 0x80) == 0x80)
    {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "byte: \\x" + res;
    }
    return out;
}

std::string tokens_to_str(llama_context *ctx, const std::vector<llama_token>::const_iterator begin, const std::vector<llama_token>::const_iterator end)
{
    std::string ret;
    for (auto it = begin; it != end; ++it)
    {
        ret += common_token_to_piece(ctx, *it);
    }
    return ret;
}

void llama_rn_context::cleanupThreadpools() {
    if (ctx != nullptr && (threadpool != nullptr || threadpool_batch != nullptr)) {
        llama_detach_threadpool(ctx);
    }

    if (threadpool_batch != nullptr) {
        lm_ggml_threadpool_free(threadpool_batch);
        threadpool_batch = nullptr;
    }

    if (threadpool != nullptr) {
        lm_ggml_threadpool_free(threadpool);
        threadpool = nullptr;
    }
}

bool llama_rn_context::attachThreadpoolsIfAvailable() {
    if (ctx == nullptr) {
        return false;
    }

    lm_ggml_backend_dev_t cpu_dev = lm_ggml_backend_dev_by_type(LM_GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        LOG_WARNING("No CPU backend available; skipping threadpool attachment");
        return false;
    }

    cleanupThreadpools();

    lm_ggml_threadpool_params tpp =
        lm_ggml_threadpool_params_from_cpu_params(params.cpuparams);
    lm_ggml_threadpool_params tpp_batch =
        lm_ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);

    if (tpp.n_threads <= 0) {
        LOG_WARNING("Skipping threadpool attachment (n_threads = %d)", tpp.n_threads);
        return false;
    }

    bool need_batch_pool =
        !lm_ggml_threadpool_params_match(&tpp, &tpp_batch) && tpp_batch.n_threads > 0;

    lm_ggml_threadpool *new_batch = nullptr;
    if (need_batch_pool) {
        new_batch = lm_ggml_threadpool_new(&tpp_batch);
        if (new_batch == nullptr) {
            LOG_WARNING("Failed to create batch threadpool (n_threads=%d)", tpp_batch.n_threads);
            return false;
        }
        tpp.paused = true;
    }

    lm_ggml_threadpool *new_threadpool = lm_ggml_threadpool_new(&tpp);
    if (new_threadpool == nullptr) {
        LOG_WARNING("Failed to create threadpool (n_threads=%d)", tpp.n_threads);
        if (new_batch != nullptr) {
            lm_ggml_threadpool_free(new_batch);
        }
        return false;
    }

    llama_attach_threadpool(ctx, new_threadpool, new_batch);
    threadpool = new_threadpool;
    threadpool_batch = new_batch;
    LOG_INFO("Attached ggml threadpool (n_threads=%d, n_threads_batch=%d)",
             tpp.n_threads,
             threadpool_batch ? tpp_batch.n_threads : tpp.n_threads);
    return true;
}

llama_rn_context::~llama_rn_context() {
    cleanupThreadpools();
    releaseVocoder();
    delete completion;
}

bool llama_rn_context::loadModel(common_params &params_)
{
    params = params_;

    if (params.n_parallel < 1) {
        params.n_parallel = 1;
    }

    llama_init = common_init_from_params(params);
    model = llama_init->model();
    ctx = llama_init->context();
    if (model == nullptr)
    {
        LOG_ERROR("unable to load model: %s", params_.model.path.c_str());
        return false;
    }
    n_ctx = llama_n_ctx(ctx);
    completion = new llama_rn_context_completion(this);

    return true;
}

llama_rn_tokenize_result llama_rn_context::tokenize(const std::string &text) {
    std::vector<llama_token> text_tokens;
    text_tokens = common_tokenize(ctx, text, false, true);
    llama_rn_tokenize_result tokenize_result;
    tokenize_result.tokens = text_tokens;
    return tokenize_result;
}

bool llama_rn_context::initVocoder(const std::string &vocoder_model_path, int batch_size) {
    try {
        tts_wrapper = new llama_rn_context_tts(vocoder_model_path, batch_size);
        has_vocoder = true;
        return true;
    } catch (const std::exception& e) {
        has_vocoder = false;
        return false;
    }
}

bool llama_rn_context::isVocoderEnabled() const {
    return has_vocoder && tts_wrapper != nullptr;
}

void llama_rn_context::releaseVocoder() {
    if (tts_wrapper != nullptr) {
        delete tts_wrapper;
        tts_wrapper = nullptr;
    }
    has_vocoder = false;
}

void llama_rn_context::clearCache(bool clear_data) {
    if (ctx == nullptr) {
        LOG_WARNING("Cannot clear cache: context not initialized");
        return;
    }

    auto * kv = llama_get_memory(ctx);
    if (kv == nullptr) {
        LOG_WARNING("Cannot clear cache: memory not available");
        return;
    }

    llama_memory_clear(kv, clear_data);
    LOG_INFO("Cache cleared (clear_data=%s)", clear_data ? "true" : "false");
}

}
