#ifndef RNTTS_H
#define RNTTS_H

#include <vector>
#include <string>
#include "llama.h"
#include "nlohmann/json.hpp"

using json = nlohmann::ordered_json;

struct codec_model;
struct codec_context;

namespace rnllama {

struct llama_rn_context;

enum tts_type {
    UNKNOWN = -1,
    OUTETTS_V1_0 = 1,
};

struct llama_rn_audio_completion_result {
    std::string prompt;
    std::string grammar;
};

struct llama_rn_context_tts {
    std::vector<llama_token> audio_tokens;
    std::vector<llama_token> guide_tokens;

    ::codec_model *codec_model = nullptr;
    ::codec_context *codec_ctx = nullptr;
    tts_type type = UNKNOWN;

    int c1_0_token_id = -1;
    int c2_0_token_id = -1;
    int audio_end_token_id = -1;
    bool token_ids_resolved = false;

    llama_rn_context_tts(const std::string &vocoder_model_path, int batch_size = -1);
    ~llama_rn_context_tts();

    void resolveTokenIds(llama_rn_context* main_ctx);
    tts_type getTTSType(llama_rn_context* main_ctx);
    llama_rn_audio_completion_result getFormattedAudioCompletion(llama_rn_context* main_ctx, const std::string &speaker_json_str, const std::string &text_to_speak);
    std::vector<llama_token> getAudioCompletionGuideTokens(llama_rn_context* main_ctx, const std::string &text_to_speak);
    std::vector<float> decodeAudioTokens(llama_rn_context* main_ctx, const std::vector<llama_token> &tokens);
    void collectAudioToken(llama_rn_context* main_ctx, llama_token token);
    void setGuideTokens(const std::vector<llama_token> &tokens);
};

}

#endif /* RNTTS_H */
