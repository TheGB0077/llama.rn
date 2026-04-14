#ifndef RN_COMPLETION_H
#define RN_COMPLETION_H

#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "nlohmann/json.hpp"

using json = nlohmann::ordered_json;

namespace rnllama {

static inline void llama_batch_clear(llama_batch *batch) {
    batch->n_tokens = 0;
}

struct llama_rn_context;

enum stop_type
{
    STOP_FULL,
    STOP_PARTIAL,
};

struct completion_token_output
{
    struct token_prob
    {
        llama_token tok;
        float prob;
    };

    std::vector<token_prob> probs;
    llama_token tok;
    std::string text;
    int32_t request_id = -1;
};

struct llama_rn_context_completion {
    llama_rn_context* parent_ctx;

    bool is_predicting = false;
    bool is_interrupted = false;
    bool has_next_token = false;
    std::string prefill_text;
    std::string generated_text;
    std::vector<completion_token_output> generated_token_probs;
    size_t num_prompt_tokens = 0;
    size_t num_tokens_predicted = 0;
    llama_pos n_past = 0;
    size_t n_remain = 0;
    std::vector<llama_token> embd;
    bool incomplete = false;
    bool context_full = false;
    bool truncated = false;
    bool stopped_eos = false;
    bool stopped_word = false;
    bool stopped_limit = false;
    std::string stopping_word;
    common_sampler *ctx_sampling = nullptr;

    llama_rn_context_completion(llama_rn_context* parent);
    ~llama_rn_context_completion();

    void rewind();
    bool initSampling();
    void truncatePrompt(std::vector<llama_token> &prompt_tokens);
    void loadPrompt();
    void beginCompletion();
    void endCompletion();
    completion_token_output nextToken();
    size_t findStoppingStrings(const std::string &text, const size_t last_token_size, const stop_type type);
    completion_token_output doCompletion();
};

} // namespace rnllama

#endif /* RN_COMPLETION_H */
