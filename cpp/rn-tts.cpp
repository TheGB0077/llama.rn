#include "rn-tts.h"
#include "rn-llama.h"
#include "common.h"
#include "codec.h"
#include <regex>
#include <sstream>
#include <algorithm>

namespace rnllama {

static std::string text_normalization(const std::string &input) {
    std::string result = std::regex_replace(input, std::regex("\\s+"), " ");
    result = std::regex_replace(result, std::regex("\\…"), "...");
    result = std::regex_replace(result, std::regex("\\s+$|^\\s+"), "");

    static const std::pair<std::string, std::string> patterns[] = {
        {"\u201C", "\""}, {"\u201D", "\""},
        {"\u2018", "'"},  {"\u2019", "'"},
        {"\u2013", "-"},  {"\u2014", "-"},
    };
    for (const auto &p : patterns) {
        size_t pos = 0;
        while ((pos = result.find(p.first, pos)) != std::string::npos) {
            result.replace(pos, p.first.length(), p.second);
            pos += p.second.length();
        }
    }
    return result;
}

static std::string format_special(const std::string &tmpl, int value) {
    size_t pos = tmpl.find("{}");
    if (pos == std::string::npos) return tmpl;
    return tmpl.substr(0, pos) + std::to_string(value) + tmpl.substr(pos + 2);
}

static std::string format_special(const std::string &tmpl, double value) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f", value);
    size_t pos = tmpl.find("{}");
    if (pos == std::string::npos) return tmpl;
    return tmpl.substr(0, pos) + buf + tmpl.substr(pos + 2);
}

struct SpecialTokens {
    std::string bos = "<|im_start|>";
    std::string text_start = "<|text_start|>";
    std::string text_end = "<|text_end|>";
    std::string audio_start = "<|audio_start|>";
    std::string audio_end = "<|audio_end|>";
    std::string word_start = "<|word_start|>";
    std::string word_end = "<|word_end|>";
    std::string c1 = "<|c1_{}|>";
    std::string c2 = "<|c2_{}|>";
    std::string time = "<|t_{}|>";
    std::string code = "<|code|>";
    std::string features = "<|features|>";
    std::string energy = "<|energy_{}|>";
    std::string spectral_centroid = "<|spectral_centroid_{}|>";
    std::string pitch = "<|pitch_{}|>";
    std::string global_features_start = "<|global_features_start|>";
    std::string global_features_end = "<|global_features_end|>";
};

static std::string create_codes(const json &words, const SpecialTokens &st) {
    std::string result;
    for (size_t i = 0; i < words.size(); i++) {
        const auto &w = words[i];
        std::string entry = st.word_start;
        entry += w["word"].get<std::string>() + st.features;
        entry += format_special(st.time, w["duration"].get<float>());

        if (w.contains("features") && w["features"].is_object()) {
            const auto &f = w["features"];
            if (f.contains("energy")) entry += format_special(st.energy, f["energy"].get<int>());
            if (f.contains("spectral_centroid")) entry += format_special(st.spectral_centroid, f["spectral_centroid"].get<int>());
            if (f.contains("pitch")) entry += format_special(st.pitch, f["pitch"].get<int>());
        }

        entry += st.code;
        if (w.contains("c1") && w.contains("c2")) {
            size_t n = std::min(w["c1"].size(), w["c2"].size());
            for (size_t idx = 0; idx < n; idx++) {
                entry += format_special(st.c1, w["c1"][idx].get<int>());
                entry += format_special(st.c2, w["c2"][idx].get<int>());
            }
        }

        entry += st.word_end;
        result += entry;
        if (i < words.size() - 1) result += "\n";
    }
    return result;
}

static std::string get_separator(const std::string &text) {
    for (unsigned char c : text) {
        if (c >= 0xE3) return "\u3002";
        if (c >= 0xEA) return ". ";
        if (c >= 0xE4 && c <= 0xE9) return "\u3002";
    }
    return ". ";
}

static std::string merge_speaker_text(const std::string &input_text, const std::string &speaker_text) {
    std::string st = speaker_text;
    st = std::regex_replace(st, std::regex("\\s+$|^\\s+"), "");

    if (st.empty()) return input_text;

    std::string sep = get_separator(st);
    bool ends_ok = false;
    if (sep == "\u3002") {
        for (auto &e : {"\u3002", "\uFF1F", "\uFF01", "?", "!"}) {
            if (st.size() >= 1 && st.compare(st.size() - 1, 1, e) == 0) { ends_ok = true; break; }
            if (st.size() >= 3 && st.compare(st.size() - 3, 3, e) == 0) { ends_ok = true; break; }
        }
    } else {
        for (auto &e : {".", "?", "!"}) {
            if (st.size() >= 1 && st.compare(st.size() - 1, 1, e) == 0) { ends_ok = true; break; }
        }
    }

    std::string rs = ends_ok ? "" : (sep == "\u3002" ? "" : " ");
    std::string output = st + rs + input_text;
    std::string trimmed_rs = std::regex_replace(rs, std::regex("^\\s+|\\s+$"), "");
    return output;
}

static std::string init_prompt(const std::string &text, const SpecialTokens &st) {
    return st.bos + "\n" + st.text_start + text + st.text_end + "\n" + st.audio_start + "\n";
}

static std::string get_completion_prompt(const std::string &text, json &speaker) {
    SpecialTokens st;
    std::string normalized = text_normalization(text);

    std::string prompt;
    if (speaker.is_object() && speaker.contains("words") && speaker["words"].is_array()) {
        std::string spk_text = speaker.contains("text") ? speaker["text"].get<std::string>() : "";
        normalized = merge_speaker_text(normalized, spk_text);
        prompt = init_prompt(normalized, st);
        prompt += create_codes(speaker["words"], st) + "\n" + st.word_start;
    } else {
        prompt = init_prompt(normalized, st);
    }
    return prompt;
}

extern bool rnllama_verbose;
void log(const char *level, const char *function, int line, const char *format, ...);

#define LOG_ERROR(MSG, ...) log("ERROR", __func__, __LINE__, MSG, ##__VA_ARGS__)
#define LOG_WARNING(MSG, ...) log("WARNING", __func__, __LINE__, MSG, ##__VA_ARGS__)
#define LOG_INFO(MSG, ...) log("INFO", __func__, __LINE__, MSG, ##__VA_ARGS__)

llama_rn_context_tts::llama_rn_context_tts(const std::string &vocoder_model_path, int /* batch_size */) {
  struct codec_model_params model_params = codec_model_default_params();
  codec_model = codec_model_load_from_file(vocoder_model_path.c_str(), model_params);
  if (codec_model == nullptr) {
      throw std::runtime_error("Failed to load codec model");
  }

  struct codec_context_params context_params = codec_context_default_params();
  codec_ctx = codec_init_from_model(codec_model, context_params);
  if (codec_ctx == nullptr) {
      codec_model_free(codec_model);
      codec_model = nullptr;
      throw std::runtime_error("Failed to initialize codec context");
  }

  type = UNKNOWN;
}

llama_rn_context_tts::~llama_rn_context_tts() {
  if (codec_ctx != nullptr) {
      codec_free(codec_ctx);
      codec_ctx = nullptr;
  }
  if (codec_model != nullptr) {
      codec_model_free(codec_model);
      codec_model = nullptr;
  }
  type = UNKNOWN;
}

void llama_rn_context_tts::setGuideTokens(const std::vector<llama_token> &tokens) {
    guide_tokens = tokens;
}

void llama_rn_context_tts::resolveTokenIds(llama_rn_context* main_ctx) {
    if (token_ids_resolved) return;
    const llama_vocab * vocab = llama_model_get_vocab(main_ctx->model);
    for (int i = 0; i < (int)llama_vocab_n_tokens(vocab); i++) {
        std::string piece = common_token_to_piece(vocab, (llama_token)i);
        if (piece == "<|c1_0|>") { c1_0_token_id = i; }
        else if (piece == "<|c2_0|>") { c2_0_token_id = i; }
        else if (piece == "<|audio_end|>") { audio_end_token_id = i; }
    }
    token_ids_resolved = true;
    LOG_INFO("Resolved token IDs: c1_0=%d, c2_0=%d, audio_end=%d", c1_0_token_id, c2_0_token_id, audio_end_token_id);
}

tts_type llama_rn_context_tts::getTTSType(llama_rn_context* main_ctx) {
    if (type != UNKNOWN) return type;
    resolveTokenIds(main_ctx);
    if (c1_0_token_id >= 0 && c2_0_token_id >= 0) {
        type = OUTETTS_V1_0;
        return type;
    }
    LOG_ERROR("Unknown TTS type — could not find c1/c2 token IDs");
    return UNKNOWN;
}

llama_rn_audio_completion_result llama_rn_context_tts::getFormattedAudioCompletion(llama_rn_context* main_ctx, const std::string &speaker_json_str, const std::string &text_to_speak) {
    json speaker = speaker_json_str.empty() ? json::object() : json::parse(speaker_json_str);

    tts_type tts_t = getTTSType(main_ctx);
    if (tts_t == UNKNOWN) {
        LOG_ERROR("Unknown TTS version");
        return {"", ""};
    }

    json spk_copy = speaker;
    std::string prompt = get_completion_prompt(text_to_speak, spk_copy);

    return {prompt, ""};
}

std::vector<llama_token> llama_rn_context_tts::getAudioCompletionGuideTokens(llama_rn_context* main_ctx, const std::string &text_to_speak) {
    const llama_vocab * vocab = llama_model_get_vocab(main_ctx->model);
    std::string normalized = text_normalization(text_to_speak);

    std::vector<llama_token> result;

    std::vector<llama_token> text_tokens = common_tokenize(vocab, normalized, false, true);
    for (auto tok : text_tokens) {
        result.push_back(tok);
    }

    if (audio_end_token_id >= 0) {
        result.push_back(audio_end_token_id);
    }

    return result;
}

std::vector<float> llama_rn_context_tts::decodeAudioTokens(llama_rn_context* main_ctx, const std::vector<llama_token> &tokens) {
    if (codec_ctx == nullptr || codec_model == nullptr) {
        LOG_ERROR("Codec context is not initialized");
        return std::vector<float>();
    }

    resolveTokenIds(main_ctx);

    std::vector<int> codebook1;
    std::vector<int> codebook2;

    int codebook_size = codec_model_codebook_size(codec_model);
    int c1_last = c1_0_token_id + codebook_size - 1;
    int c2_last = c2_0_token_id + codebook_size - 1;

    LOG_INFO("[decodeAudio] tokens=%zu c1_0=%d c2_0=%d codebook_size=%d c1_range=[%d,%d] c2_range=[%d,%d]",
        tokens.size(), c1_0_token_id, c2_0_token_id, codebook_size,
        c1_0_token_id, c1_last, c2_0_token_id, c2_last);

    for (auto tok : tokens) {
        if (tok >= c1_0_token_id && tok <= c1_last) {
            codebook1.push_back(tok - c1_0_token_id);
        } else if (tok >= c2_0_token_id && tok <= c2_last) {
            codebook2.push_back(tok - c2_0_token_id);
        }
    }

    LOG_INFO("[decodeAudio] codebook1=%zu codebook2=%zu first_5_tokens:", codebook1.size(), codebook2.size());
    for (size_t i = 0; i < std::min((size_t)10, tokens.size()); i++) {
        LOG_INFO("[decodeAudio]   token[%zu]=%d", i, tokens[i]);
    }

    size_t n_frames = std::min(codebook1.size(), codebook2.size());
    codebook1.resize(n_frames);
    codebook2.resize(n_frames);

    if (n_frames == 0) {
        LOG_ERROR("[decodeAudio] n_frames=0, no matching tokens");
        return std::vector<float>();
    }

    const int n_q = 2;
    std::vector<int32_t> interleaved;
    interleaved.reserve(n_frames * n_q);
    for (size_t i = 0; i < n_frames; i++) {
        interleaved.push_back(codebook1[i]);
        interleaved.push_back(codebook2[i]);
    }

    struct codec_token_buffer token_buffer = {};
    token_buffer.data = interleaved.data();
    token_buffer.n_tokens = (int32_t)interleaved.size();
    token_buffer.n_frames = (int32_t)n_frames;
    token_buffer.n_q = n_q;
    token_buffer.codebook_size = codebook_size;
    token_buffer.sample_rate = codec_model_sample_rate(codec_model);
    token_buffer.hop_size = codec_model_hop_size(codec_model);

    struct codec_decode_params decode_params = codec_decode_default_params();
    decode_params.n_q = n_q;
    if (main_ctx->params.cpuparams.n_threads > 0) {
        decode_params.n_threads = main_ctx->params.cpuparams.n_threads;
    }

    struct codec_pcm_buffer pcm = {};
    const enum codec_status status = codec_decode(codec_ctx, &token_buffer, &pcm, decode_params);
    if (status != CODEC_STATUS_SUCCESS) {
        const char *err = codec_get_last_error(codec_ctx);
        LOG_ERROR("codec_decode() failed: %s", err != nullptr ? err : "unknown error");
        return std::vector<float>();
    }

    std::vector<float> audio(pcm.data, pcm.data + pcm.n_samples);
    codec_pcm_buffer_free(&pcm);
    return audio;
}

void llama_rn_context_tts::collectAudioToken(llama_rn_context* main_ctx, llama_token token) {
    resolveTokenIds(main_ctx);
    if (c1_0_token_id >= 0 && c2_0_token_id >= 0) {
        int codebook_size = codec_model_codebook_size(codec_model);
        if ((token >= c1_0_token_id && token <= c1_0_token_id + codebook_size - 1) ||
            (token >= c2_0_token_id && token <= c2_0_token_id + codebook_size - 1)) {
            audio_tokens.push_back(token);
        }
    }
}

}
