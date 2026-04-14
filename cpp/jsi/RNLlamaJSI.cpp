#include "RNLlamaJSI.h"
#include "JSIContext.h"
#include "ThreadPool.h"
#include "JSIUtils.h"
#include "JSIParams.h"
#include "JSIHelpers.h"
#include "JSICompletion.h"
#include "JSIRequestManager.h"
#include "JSITaskManager.h"
#include "JSINativeHeaders.h"

#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__ANDROID__)
#include <android/log.h>
#include <cstring>
#endif

using namespace facebook;
using json = nlohmann::ordered_json;

enum class LogLevel { LOG_DEBUG, LOG_INFO, LOG_ERROR };

static void log(LogLevel level, const char* format, ...) {
    va_list args;
    va_start(args, format);

#if defined(__ANDROID__)
    int androidLevel = (level == LogLevel::LOG_DEBUG) ? ANDROID_LOG_DEBUG :
                      (level == LogLevel::LOG_INFO) ? ANDROID_LOG_INFO : ANDROID_LOG_ERROR;
    __android_log_vprint(androidLevel, "RNWhisperJSI", format, args);
#else
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    const char* levelStr = (level == LogLevel::LOG_DEBUG) ? "DEBUG" :
                          (level == LogLevel::LOG_INFO) ? "INFO" : "ERROR";
    printf("RNWhisperJSI %s: %s\n", levelStr, buffer);
#endif

    va_end(args);
}

#define logInfo(format, ...) log(LogLevel::LOG_INFO, format, ##__VA_ARGS__)
#define logError(format, ...) log(LogLevel::LOG_ERROR, format, ##__VA_ARGS__)
#define logDebug(format, ...) log(LogLevel::LOG_DEBUG, format, ##__VA_ARGS__)

static std::once_flag backend_init_once;

static std::string stripFileScheme(const std::string& path) {
    const std::string prefix = "file://";
    if (path.rfind(prefix, 0) == 0) {
        return path.substr(prefix.size());
    }
    return path;
}

#if defined(__ANDROID__)
static std::vector<lm_ggml_backend_dev_t> getFilteredDefaultDevices() {
    std::vector<lm_ggml_backend_dev_t> rpc_servers;
    std::vector<lm_ggml_backend_dev_t> gpus;
    std::vector<lm_ggml_backend_dev_t> igpus;

    for (size_t i = 0; i < lm_ggml_backend_dev_count(); ++i) {
        lm_ggml_backend_dev_t dev = lm_ggml_backend_dev_get(i);

        switch (lm_ggml_backend_dev_type(dev)) {
            case LM_GGML_BACKEND_DEVICE_TYPE_CPU:
            case LM_GGML_BACKEND_DEVICE_TYPE_ACCEL:
            case LM_GGML_BACKEND_DEVICE_TYPE_META:
                break;
            case LM_GGML_BACKEND_DEVICE_TYPE_GPU: {
                lm_ggml_backend_reg_t reg = lm_ggml_backend_dev_backend_reg(dev);
                const char *reg_name = reg ? lm_ggml_backend_reg_name(reg) : nullptr;
                if (reg_name != nullptr && strcmp(reg_name, "RPC") == 0) {
                    rpc_servers.push_back(dev);
                } else {
                    lm_ggml_backend_dev_props props;
                    lm_ggml_backend_dev_get_props(dev, &props);
                    auto it = std::find_if(gpus.begin(), gpus.end(), [&props](lm_ggml_backend_dev_t other) {
                        lm_ggml_backend_dev_props other_props;
                        lm_ggml_backend_dev_get_props(other, &other_props);
                        return props.device_id != nullptr &&
                               other_props.device_id != nullptr &&
                               strcmp(props.device_id, other_props.device_id) == 0;
                    });

                    if (it == gpus.end()) {
                        gpus.push_back(dev);
                    }
                }
                break;
            }
            case LM_GGML_BACKEND_DEVICE_TYPE_IGPU:
                igpus.push_back(dev);
                break;
        }
    }

    std::vector<lm_ggml_backend_dev_t> devices;
    devices.insert(devices.end(), rpc_servers.begin(), rpc_servers.end());
    devices.insert(devices.end(), gpus.begin(), gpus.end());

    if (devices.empty()) {
        devices.insert(devices.end(), igpus.begin(), igpus.end());
    }

    if (!devices.empty()) {
        devices.push_back(nullptr);
    }

    return devices;
}
#endif

namespace rnllama_jsi {
    static std::atomic<int64_t> g_context_limit(-1);
#if defined(__ANDROID__)
    static std::string g_android_loaded_library;
#endif
    static std::mutex g_log_mutex;
    static std::weak_ptr<react::CallInvoker> g_log_invoker;
    static std::shared_ptr<jsi::Function> g_log_handler;
    static std::shared_ptr<jsi::Runtime> g_log_runtime;

    struct ProgressCallbackData {
        std::shared_ptr<jsi::Function> callback;
        std::weak_ptr<react::CallInvoker> callInvoker;
        std::shared_ptr<jsi::Runtime> runtime;
        int contextId;
        std::atomic<int> lastProgress{0};
        int progressEvery = 1;
    };

    void setContextLimit(int64_t limit) {
        g_context_limit.store(limit);
    }

#if defined(__ANDROID__)
    void setAndroidLoadedLibrary(const std::string& name) {
        g_android_loaded_library = name;
    }
#endif

    static bool isContextLimitReached() {
        int64_t limit = g_context_limit.load();
        if (limit < 0) {
            return false;
        }
        return g_llamaContexts.size() >= static_cast<size_t>(limit);
    }

    static bool isContextBusy(rnllama::llama_rn_context* ctx) {
        if (ctx == nullptr) {
            return false;
        }
        return ctx->completion && ctx->completion->is_predicting;
    }

    static void throwIfContextBusy(rnllama::llama_rn_context* ctx) {
        if (isContextBusy(ctx)) {
            throw std::runtime_error("Context is busy");
        }
    }

    static void ensureBackendInitialized() {
        std::call_once(backend_init_once, []() {
            llama_backend_init();
        });
    }

    static void logToJsCallback(enum lm_ggml_log_level level, const char* text, void* /*data*/) {
        llama_log_callback_default(level, text, nullptr);

        std::shared_ptr<react::CallInvoker> invoker;
        std::shared_ptr<jsi::Function> handler;
        std::shared_ptr<jsi::Runtime> runtime;
        {
            std::lock_guard<std::mutex> lock(g_log_mutex);
            invoker = g_log_invoker.lock();
            handler = g_log_handler;
            runtime = g_log_runtime;
        }

        if (!invoker || !handler || !runtime) {
            return;
        }

        std::string levelStr = "info";
        switch (level) {
            case LM_GGML_LOG_LEVEL_ERROR: levelStr = "error"; break;
            case LM_GGML_LOG_LEVEL_WARN: levelStr = "warn"; break;
            case LM_GGML_LOG_LEVEL_INFO: levelStr = "info"; break;
            default: break;
        }

        std::string message = text ? text : "";

        invoker->invokeAsync([handler, levelStr, message, runtime]() {
            auto& rt = *runtime;
            handler->call(
                rt,
                jsi::String::createFromUtf8(rt, levelStr),
                jsi::String::createFromUtf8(rt, message)
            );
        });
    }

    static jsi::Array toJsStringArray(jsi::Runtime& runtime, const std::vector<std::string>& values) {
        jsi::Array arr(runtime, values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            arr.setValueAtIndex(runtime, i, jsi::String::createFromUtf8(runtime, values[i]));
        }
        return arr;
    }

    static jsi::Object createModelDetails(jsi::Runtime& runtime, rnllama::llama_rn_context* ctx) {
        jsi::Object model(runtime);

        char desc[1024];
        llama_model_desc(ctx->model, desc, sizeof(desc));
        model.setProperty(runtime, "desc", jsi::String::createFromUtf8(runtime, desc));
        model.setProperty(runtime, "size", (double)llama_model_size(ctx->model));
        model.setProperty(runtime, "nEmbd", (double)llama_model_n_embd(ctx->model));
        model.setProperty(runtime, "nParams", (double)llama_model_n_params(ctx->model));
        model.setProperty(runtime, "is_recurrent", llama_model_is_recurrent(ctx->model));
        model.setProperty(runtime, "is_hybrid", llama_model_is_hybrid(ctx->model));

        jsi::Object metadata(runtime);
        int metaCount = llama_model_meta_count(ctx->model);
        for (int i = 0; i < metaCount; ++i) {
            char key[256];
            llama_model_meta_key_by_index(ctx->model, i, key, sizeof(key));
            char val[16384];
            llama_model_meta_val_str_by_index(ctx->model, i, val, sizeof(val));
            metadata.setProperty(runtime, key, jsi::String::createFromUtf8(runtime, val));
        }
        model.setProperty(runtime, "metadata", metadata);

        return model;
    }

    static std::vector<lm_ggml_backend_dev_t> buildDeviceOverrides(
        const std::vector<std::string>& requestedDevices,
        bool skipGpuDevices,
        bool& anyGpuAvailable
    ) {
        std::vector<lm_ggml_backend_dev_t> selected;
        anyGpuAvailable = false;

        const size_t devCount = lm_ggml_backend_dev_count();
        for (size_t i = 0; i < devCount; ++i) {
            lm_ggml_backend_dev_t dev = lm_ggml_backend_dev_get(i);
            const auto type = lm_ggml_backend_dev_type(dev);
#if TARGET_OS_SIMULATOR
            if (type == LM_GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                continue;
            }
#endif
            const bool isGpuType = type == LM_GGML_BACKEND_DEVICE_TYPE_GPU || type == LM_GGML_BACKEND_DEVICE_TYPE_IGPU;
            if (isGpuType) {
                anyGpuAvailable = true;
            }
            if (skipGpuDevices && isGpuType) {
                continue;
            }

            if (!requestedDevices.empty()) {
                const char* name = lm_ggml_backend_dev_name(dev);
                std::string nameStr = name ? name : "";
                auto it = std::find(requestedDevices.begin(), requestedDevices.end(), nameStr);
                if (it == requestedDevices.end()) {
                    continue;
                }
            }

            selected.push_back(dev);
        }

        if (!selected.empty()) {
            selected.push_back(nullptr);
        }

        return selected;
    }

    void addContext(int contextId, long contextPtr) {
        g_llamaContexts.add(contextId, contextPtr);
    }

    void removeContext(int contextId) {
        g_llamaContexts.remove(contextId);
    }

    rnllama::llama_rn_context* getContextOrThrow(int contextId) {
        long ctxPtr = g_llamaContexts.get(contextId);
        if (!ctxPtr) {
            throw std::runtime_error("Context not found");
        }
        return reinterpret_cast<rnllama::llama_rn_context*>(ctxPtr);
    }

    void installJSIBindings(
        jsi::Runtime& runtime,
        std::shared_ptr<react::CallInvoker> callInvoker
    ) {
        TaskManager::getInstance().reset();
        auto initContext = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaInitContext"),
            3,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                jsi::Object params = arguments[1].asObject(runtime);
                bool isModelAsset = getPropertyAsBool(runtime, params, "is_model_asset", false);

                bool useProgressCallback = getPropertyAsBool(runtime, params, "use_progress_callback", false);
                int progressCallbackEvery = getPropertyAsInt(runtime, params, "progress_callback_every", 1);
                std::shared_ptr<ProgressCallbackData> progressData;
                if (count > 2 && arguments[2].isObject() && arguments[2].asObject(runtime).isFunction(runtime)) {
                    useProgressCallback = true;
                    progressData = std::make_shared<ProgressCallbackData>();
                    progressData->callback = makeJsiFunction(runtime, arguments[2], callInvoker);
                    progressData->callInvoker = callInvoker;
                    progressData->runtime = std::shared_ptr<jsi::Runtime>(&runtime, [](jsi::Runtime*){});
                    progressData->contextId = contextId;
                    progressData->progressEvery = std::max(1, progressCallbackEvery);
                    progressData->lastProgress.store(0);
                } else if (useProgressCallback) {
                    useProgressCallback = false;
                }

                ensureBackendInitialized();

                common_params cparams;
                parseCommonParams(runtime, params, cparams);

#if defined(__APPLE__)
                if (isModelAsset) {
                    cparams.model.path = resolveIosModelPath(cparams.model.path, true);
                }
#endif

                bool skipGpuDevices = getPropertyAsBool(runtime, params, "no_gpu_devices", false);
                if (skipGpuDevices) {
                    cparams.n_gpu_layers = 0;
                }

#if defined(__APPLE__)
                auto metalAvailability = getMetalAvailability(skipGpuDevices);
                std::string appleGpuReason = metalAvailability.available ? "" : metalAvailability.reason;
                if (!metalAvailability.available && !skipGpuDevices) {
                    skipGpuDevices = true;
                    cparams.n_gpu_layers = 0;
                }
#endif

                std::vector<std::string> requestedDevices;
                bool devicesProvided = false;
                if (params.hasProperty(runtime, "devices") && params.getProperty(runtime, "devices").isObject()) {
                    jsi::Array devicesArr = params.getProperty(runtime, "devices").asObject(runtime).asArray(runtime);
                    if (devicesArr.size(runtime) > 0) {
                        devicesProvided = true;
                        for (size_t i = 0; i < devicesArr.size(runtime); ++i) {
                            auto val = devicesArr.getValueAtIndex(runtime, i);
                            if (val.isString()) {
                                requestedDevices.push_back(val.asString(runtime).utf8(runtime));
                            }
                        }
                    }
                }
                bool anyGpuAvailable = false;
                std::vector<lm_ggml_backend_dev_t> overrideDevices;
                if (devicesProvided) {
                    overrideDevices = buildDeviceOverrides(requestedDevices, skipGpuDevices, anyGpuAvailable);
                    if (!overrideDevices.empty()) {
                        cparams.devices = overrideDevices;
                    }
                }
                if (overrideDevices.empty() && !skipGpuDevices) {
#if defined(__ANDROID__)
                    auto defaultDevices = getFilteredDefaultDevices();
                    if (!defaultDevices.empty()) {
                        cparams.devices = defaultDevices;
                        for (auto dev : defaultDevices) {
                            if (dev == nullptr) continue;
                            auto type = lm_ggml_backend_dev_type(dev);
                            if (type == LM_GGML_BACKEND_DEVICE_TYPE_GPU || type == LM_GGML_BACKEND_DEVICE_TYPE_IGPU) {
                                anyGpuAvailable = true;
                                break;
                            }
                        }
                    }
#endif
                }

                if (overrideDevices.empty() && anyGpuAvailable == false) {
                    const size_t devCount = lm_ggml_backend_dev_count();
                    for (size_t i = 0; i < devCount; ++i) {
                        auto dev = lm_ggml_backend_dev_get(i);
                        auto type = lm_ggml_backend_dev_type(dev);
                        if (type == LM_GGML_BACKEND_DEVICE_TYPE_GPU || type == LM_GGML_BACKEND_DEVICE_TYPE_IGPU) {
                            anyGpuAvailable = true;
                            break;
                        }
                    }
                }

                return createPromiseTask(runtime, callInvoker, [contextId, cparams, skipGpuDevices, anyGpuAvailable, useProgressCallback, progressData
#if defined(__APPLE__)
                    , appleGpuReason
#endif
                ]() mutable -> PromiseResultGenerator {
                    if (isContextLimitReached()) {
                        throw std::runtime_error("Context limit reached");
                    }

                    if (useProgressCallback && progressData && progressData->callback) {
                        cparams.progress_callback = [](float progress, void * user_data) {
                            auto *data = static_cast<ProgressCallbackData *>(user_data);
                            if (!data) {
                                return true;
                            }

                            int percentage = (int) (progress * 100.0f);
                            int last = data->lastProgress.load();
                            if (percentage < 100 && percentage - last < data->progressEvery) {
                                return true;
                            }
                            if (percentage <= last) {
                                return true;
                            }

                            data->lastProgress.store(percentage);

                            auto invoker = data->callInvoker.lock();
                            auto cb = data->callback;
                            auto runtime = data->runtime;
                            if (invoker && cb && runtime) {
                                invoker->invokeAsync([cb, percentage, runtime]() {
                                    auto& rt = *runtime;
                                    cb->call(rt, jsi::Value((double) percentage));
                                });
                            }

                            return true;
                        };
                        cparams.progress_callback_user_data = progressData.get();
                    }

                    auto ctx = new rnllama::llama_rn_context();
                    if (ctx->loadModel(cparams)) {
                         ctx->attachThreadpoolsIfAvailable();

                         std::vector<std::string> usedDevices;
                         bool gpuEnabled = false;
                         if (ctx->llama_init->model() != nullptr) {
                             for (const auto & dev_info : ctx->llama_init->model()->devices) {
                                 auto dev = dev_info.dev;
                                 if (dev == nullptr) continue;
                                 const char* used_name = lm_ggml_backend_dev_name(dev);
                                 if (used_name != nullptr) {
                                     usedDevices.push_back(used_name);
                                 }
                                 auto devType = lm_ggml_backend_dev_type(dev);
                                 if (devType == LM_GGML_BACKEND_DEVICE_TYPE_GPU || devType == LM_GGML_BACKEND_DEVICE_TYPE_IGPU) {
                                     gpuEnabled = true;
                                 }
                             }
                         }

                         std::string reasonNoGPU;
#if defined(__APPLE__)
                         const std::string platformReason = appleGpuReason;
#endif
                         if (!gpuEnabled) {
#if defined(__APPLE__)
                             if (!platformReason.empty()) {
                                 reasonNoGPU = platformReason;
                             } else
#endif
                             if (skipGpuDevices) {
                                 reasonNoGPU = "GPU devices disabled by user";
                             } else if (anyGpuAvailable) {
                                 reasonNoGPU = "GPU backend is available but was not selected";
                             } else {
                                 reasonNoGPU = "GPU backend is not available";
                             }
                         }

                         addContext(contextId, (long)ctx);

                         std::string system_info = common_params_get_system_info(ctx->params);

                         return [gpuEnabled, reasonNoGPU, system_info, usedDevices, contextId](jsi::Runtime& rt) {
                             jsi::Object result(rt);
                             result.setProperty(rt, "gpu", gpuEnabled);
                             result.setProperty(rt, "reasonNoGPU", jsi::String::createFromUtf8(rt, reasonNoGPU));
                             result.setProperty(rt, "systemInfo", jsi::String::createFromUtf8(rt, system_info));

                             long ctxPtr = g_llamaContexts.get(contextId);
                             if (ctxPtr) {
                                 auto ctx = reinterpret_cast<rnllama::llama_rn_context*>(ctxPtr);
                                 result.setProperty(rt, "model", createModelDetails(rt, ctx));
                             }

                             result.setProperty(rt, "devices", toJsStringArray(rt, usedDevices));
                             std::string androidLibName = "";
                             #if defined(__ANDROID__)
                             androidLibName = g_android_loaded_library;
                             #endif
                             result.setProperty(rt, "androidLib", jsi::String::createFromUtf8(rt, androidLibName));
                             return result;
                         };
                    } else {
                        delete ctx;
                        throw std::runtime_error("Failed to load model");
                    }
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaInitContext", initContext);

        auto modelInfo = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaModelInfo"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                std::string path = arguments[0].asString(runtime).utf8(runtime);
                std::vector<std::string> skip;
                if (count > 1 && arguments[1].isObject()) {
                    jsi::Array skipArr = arguments[1].asObject(runtime).asArray(runtime);
                    for (size_t i = 0; i < skipArr.size(runtime); i++) {
                        skip.push_back(skipArr.getValueAtIndex(runtime, i).asString(runtime).utf8(runtime));
                    }
                }

                return createPromiseTask(runtime, callInvoker, [path, skip]() -> PromiseResultGenerator {
                    return [path, skip](jsi::Runtime& rt) {
                        return createModelInfo(rt, path, skip);
                    };
                }, -1, false);
            }
        );
        runtime.global().setProperty(runtime, "llamaModelInfo", modelInfo);

        auto getBackendDevicesInfo = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaGetBackendDevicesInfo"),
            0,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                 return createPromiseTask(runtime, callInvoker, [callInvoker]() -> PromiseResultGenerator {
                     ensureBackendInitialized();

                     std::string info = rnllama::get_backend_devices_info();

                     return [info](jsi::Runtime& rt) {
                         return jsi::String::createFromUtf8(rt, info);
                     };
                 }, -1, false);
            }
        );
        runtime.global().setProperty(runtime, "llamaGetBackendDevicesInfo", getBackendDevicesInfo);

        auto tokenize = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaTokenize"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                std::string text = arguments[1].asString(runtime).utf8(runtime);

                return createPromiseTask(runtime, callInvoker, [contextId, text]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    auto result = ctx->tokenize(text);
                    return [result](jsi::Runtime& rt) {
                        return createTokenizeResult(rt, result);
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaTokenize", tokenize);

        auto detokenize = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaDetokenize"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                std::vector<llama_token> tokens;
                jsi::Array tokensArr = arguments[1].asObject(runtime).asArray(runtime);
                for (size_t i = 0; i < tokensArr.size(runtime); ++i) {
                    tokens.push_back((llama_token)tokensArr.getValueAtIndex(runtime, i).asNumber());
                }

                return createPromiseTask(runtime, callInvoker, [contextId, tokens]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    std::string text = rnllama::tokens_to_str(ctx->ctx, tokens.cbegin(), tokens.cend());
                    return [text](jsi::Runtime& rt) {
                        return jsi::String::createFromUtf8(rt, text);
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaDetokenize", detokenize);

        auto completion = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaCompletion"),
            3,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                jsi::Object params = arguments[1].asObject(runtime);
                std::shared_ptr<jsi::Function> onToken;

                if (count > 2 && arguments[2].isObject() && arguments[2].asObject(runtime).isFunction(runtime)) {
                    onToken = makeJsiFunction(runtime, arguments[2], callInvoker);
                }

                bool emitPartial = getPropertyAsBool(runtime, params, "emit_partial_completion", false);

                auto ctx = getContextOrThrow(contextId);
                throwIfContextBusy(ctx);

                if (ctx->completion == nullptr) {
                    throw std::runtime_error("Completion not initialized");
                }
                ctx->completion->rewind();

                parseCompletionParams(runtime, params, ctx);

                std::string prefill_text = getPropertyAsString(runtime, params, "prefill_text");
                std::vector<llama_token> guide_tokens;
                if (params.hasProperty(runtime, "guide_tokens")) {
                    auto guideVal = params.getProperty(runtime, "guide_tokens");
                    if (guideVal.isObject() && guideVal.asObject(runtime).isArray(runtime)) {
                        jsi::Array guideArr = guideVal.asObject(runtime).asArray(runtime);
                        guide_tokens.reserve(guideArr.size(runtime));
                        for (size_t i = 0; i < guideArr.size(runtime); i++) {
                            auto tokVal = guideArr.getValueAtIndex(runtime, i);
                            if (tokVal.isNumber()) {
                                guide_tokens.push_back((llama_token)tokVal.asNumber());
                            }
                        }
                    }
                }

                return createPromiseTask(runtime, callInvoker, [runtimePtr = std::shared_ptr<jsi::Runtime>(&runtime, [](jsi::Runtime*){}), contextId, onToken, emitPartial, prefill_text, guide_tokens, callInvoker]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);

                    if (ctx->completion == nullptr) {
                        throw std::runtime_error("Completion not initialized");
                    }
                    throwIfContextBusy(ctx);

                    if (!guide_tokens.empty() && ctx->tts_wrapper != nullptr) {
                        ctx->params.vocoder.use_guide_tokens = true;
                        ctx->tts_wrapper->setGuideTokens(guide_tokens);
                    }

                    if (!ctx->completion->initSampling()) {
                        throw std::runtime_error("Failed to initialize sampling");
                    }

                    ctx->completion->prefill_text = prefill_text;
                    ctx->completion->beginCompletion();

                    try {
                        ctx->completion->loadPrompt();
                    } catch (const std::exception &e) {
                        ctx->completion->endCompletion();
                        throw std::runtime_error(e.what());
                    }

                    if (ctx->completion->context_full) {
                        ctx->completion->endCompletion();
                        throw std::runtime_error("Context is full");
                    }

                    size_t sent_count = 0;

                    while (ctx->completion->has_next_token && !ctx->completion->is_interrupted) {
                        const rnllama::completion_token_output token_with_probs = ctx->completion->doCompletion();
                        if (token_with_probs.tok == -1 || ctx->completion->incomplete) {
                            continue;
                        }

                        const std::string token_text = common_token_to_piece(ctx->ctx, token_with_probs.tok);
                        size_t pos = std::min(sent_count, ctx->completion->generated_text.size());
                        const std::string str_test = ctx->completion->generated_text.substr(pos);

                        bool is_stop_full = false;
                        size_t stop_pos = ctx->completion->findStoppingStrings(str_test, token_text.size(), rnllama::STOP_FULL);
                        if (stop_pos != std::string::npos) {
                            is_stop_full = true;
                            ctx->completion->generated_text.erase(
                                ctx->completion->generated_text.begin() + pos + stop_pos,
                                ctx->completion->generated_text.end());
                            pos = std::min(sent_count, ctx->completion->generated_text.size());
                        } else {
                             stop_pos = ctx->completion->findStoppingStrings(str_test, token_text.size(), rnllama::STOP_PARTIAL);
                        }

                        if (stop_pos == std::string::npos || (!ctx->completion->has_next_token && !is_stop_full && stop_pos > 0)) {
                            const std::string to_send = ctx->completion->generated_text.substr(pos, std::string::npos);
                            sent_count += to_send.size();

                            if (emitPartial && onToken) {
                                rnllama::completion_token_output output_copy = token_with_probs;
                                output_copy.text = to_send;

                                auto runtime = runtimePtr;
                                if (runtime) {
                                    callInvoker->invokeAsync([onToken, output_copy, contextId, runtime]() {
                                        long ctxPtr = g_llamaContexts.get(contextId);
                                        if (!ctxPtr) {
                                            return;
                                        }
                                        auto ctx = reinterpret_cast<rnllama::llama_rn_context*>(ctxPtr);
                                        auto& rt = *runtime;
                                        jsi::Object res = createTokenResult(rt, ctx, output_copy);
                                        onToken->call(rt, res);
                                    });
                                }
                            }
                        }
                    }

                    common_perf_print(ctx->ctx, ctx->completion->ctx_sampling);
                    ctx->completion->endCompletion();

                    return [contextId](jsi::Runtime& rt) -> jsi::Value {
                        long ctxPtr = g_llamaContexts.get(contextId);
                        if (!ctxPtr) {
                            jsi::Object res(rt);
                            res.setProperty(rt, "text", jsi::String::createFromUtf8(rt, ""));
                            res.setProperty(rt, "interrupted", true);
                            res.setProperty(rt, "context_released", true);
                            return jsi::Value(std::move(res));
                        }
                        auto ctx = reinterpret_cast<rnllama::llama_rn_context*>(ctxPtr);
                        return jsi::Value(std::move(createCompletionResult(rt, ctx)));
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaCompletion", completion);

        auto stopCompletion = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaStopCompletion"),
            1,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                auto ctx = getContextOrThrow(contextId);
                if (ctx->completion) {
                    ctx->completion->is_interrupted = true;
                }
                return jsi::Value::undefined();
            }
        );
        runtime.global().setProperty(runtime, "llamaStopCompletion", stopCompletion);

        auto toggleNativeLog = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaToggleNativeLog"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                bool enabled = count > 0 && arguments[0].isBool() ? arguments[0].getBool() : false;
                std::shared_ptr<jsi::Function> onLog;
                if (enabled && count > 1 && arguments[1].isObject() && arguments[1].asObject(runtime).isFunction(runtime)) {
                    onLog = makeJsiFunction(runtime, arguments[1], callInvoker);
                }

                return createPromiseTask(runtime, callInvoker, [enabled, onLog, callInvoker, runtimePtr = std::shared_ptr<jsi::Runtime>(&runtime, [](jsi::Runtime*){})]() -> PromiseResultGenerator {
                    if (enabled && onLog) {
                        {
                            std::lock_guard<std::mutex> lock(g_log_mutex);
                            g_log_handler = onLog;
                            g_log_invoker = callInvoker;
                            g_log_runtime = runtimePtr;
                        }
                        llama_log_set(logToJsCallback, nullptr);
                    } else {
                        {
                            std::lock_guard<std::mutex> lock(g_log_mutex);
                            g_log_handler.reset();
                            g_log_invoker.reset();
                            g_log_runtime.reset();
                        }
                        llama_log_set(llama_log_callback_default, nullptr);
                    }
                    return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                }, -1, false);
            }
        );
        runtime.global().setProperty(runtime, "llamaToggleNativeLog", toggleNativeLog);

        auto releaseContext = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaReleaseContext"),
            1,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                 int contextId = (int)arguments[0].asNumber();
                 return createPromiseTask(runtime, callInvoker, [contextId]() -> PromiseResultGenerator {
                     RequestManager::getInstance().clearContext(contextId);
                     long ctxPtr = g_llamaContexts.get(contextId);
                     if (ctxPtr) {
                         auto ctx = reinterpret_cast<rnllama::llama_rn_context*>(ctxPtr);
                         if (ctx->completion) {
                             ctx->completion->is_interrupted = true;
                         }
                     }

                     TaskManager::getInstance().waitForContext(contextId, 0);
                     if (TaskManager::getInstance().isShuttingDown()) {
                         return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                     }

                     if (ctxPtr) {
                         auto ctx = reinterpret_cast<rnllama::llama_rn_context*>(ctxPtr);
                         removeContext(contextId);
                         delete ctx;
                     }
                     return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                 }, contextId, false);
            }
        );
        runtime.global().setProperty(runtime, "llamaReleaseContext", releaseContext);

        auto releaseAllContexts = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaReleaseAllContexts"),
            0,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                 return createPromiseTask(runtime, callInvoker, []() -> PromiseResultGenerator {
                     RequestManager::getInstance().clearAll();

                     auto contexts = g_llamaContexts.snapshot();
                     for (const auto& entry : contexts) {
                         long ctxPtr = entry.second;
                         if (!ctxPtr) {
                             continue;
                         }
                         auto ctx = reinterpret_cast<rnllama::llama_rn_context*>(ctxPtr);
                         if (ctx->completion) {
                             ctx->completion->is_interrupted = true;
                         }
                     }

                     TaskManager::getInstance().waitForAll(0);
                     if (TaskManager::getInstance().isShuttingDown()) {
                         return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                     }

                     g_llamaContexts.clear([](long ptr) {
                        if (ptr) {
                            auto ctx = reinterpret_cast<rnllama::llama_rn_context*>(ptr);
                            delete ctx;
                        }
                     });
                     return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                 }, -1, false);
            }
        );
        runtime.global().setProperty(runtime, "llamaReleaseAllContexts", releaseAllContexts);

        auto setContextLimitFn = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaSetContextLimit"),
            1,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int64_t limit = (int64_t)arguments[0].asNumber();
                return createPromiseTask(runtime, callInvoker, [limit]() -> PromiseResultGenerator {
                    setContextLimit(limit);
                    return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                });
            }
        );
        runtime.global().setProperty(runtime, "llamaSetContextLimit", setContextLimitFn);

        // Vocoder
        auto initVocoder = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaInitVocoder"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                jsi::Object params = arguments[1].asObject(runtime);
                std::string path = getPropertyAsString(runtime, params, "path");
                int n_batch = getPropertyAsInt(runtime, params, "n_batch", 512);

                return createPromiseTask(runtime, callInvoker, [contextId, path, n_batch]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    throwIfContextBusy(ctx);
                    bool result = ctx->initVocoder(path, n_batch);
                    return [result](jsi::Runtime& rt) { return jsi::Value(result); };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaInitVocoder", initVocoder);

        auto isVocoderEnabled = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaIsVocoderEnabled"),
            1,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                return createPromiseTask(runtime, callInvoker, [contextId]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    bool result = ctx->isVocoderEnabled();
                    return [result](jsi::Runtime& rt) { return jsi::Value(result); };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaIsVocoderEnabled", isVocoderEnabled);

        auto getFormattedAudioCompletion = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaGetFormattedAudioCompletion"),
            3,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                std::string speakerJsonStr = arguments[1].asString(runtime).utf8(runtime);
                std::string textToSpeak = arguments[2].asString(runtime).utf8(runtime);

                return createPromiseTask(runtime, callInvoker, [contextId, speakerJsonStr, textToSpeak]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->isVocoderEnabled()) throw std::runtime_error("Vocoder is not enabled");

                    try {
                        auto audio_result = ctx->tts_wrapper->getFormattedAudioCompletion(ctx, speakerJsonStr, textToSpeak);
                        return [audio_result](jsi::Runtime& rt) {
                            jsi::Object res(rt);
                            res.setProperty(rt, "prompt", jsi::String::createFromUtf8(rt, audio_result.prompt));
                            res.setProperty(rt, "grammar", jsi::String::createFromUtf8(rt, audio_result.grammar));
                            return res;
                        };
                    } catch (const std::exception &e) {
                        throw std::runtime_error(e.what());
                    }
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaGetFormattedAudioCompletion", getFormattedAudioCompletion);

        auto getAudioCompletionGuideTokens = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaGetAudioCompletionGuideTokens"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                std::string textToSpeak = arguments[1].asString(runtime).utf8(runtime);

                return createPromiseTask(runtime, callInvoker, [contextId, textToSpeak]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->isVocoderEnabled()) throw std::runtime_error("Vocoder is not enabled");

                    try {
                        auto guide_tokens = ctx->tts_wrapper->getAudioCompletionGuideTokens(ctx, textToSpeak);
                        return [guide_tokens](jsi::Runtime& rt) {
                            jsi::Array res(rt, guide_tokens.size());
                            for (size_t i = 0; i < guide_tokens.size(); i++) {
                                res.setValueAtIndex(rt, i, (double)guide_tokens[i]);
                            }
                            return res;
                        };
                    } catch (const std::exception &e) {
                        throw std::runtime_error(e.what());
                    }
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaGetAudioCompletionGuideTokens", getAudioCompletionGuideTokens);

        auto decodeAudioTokens = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaDecodeAudioTokens"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                jsi::Array tokensArr = arguments[1].asObject(runtime).asArray(runtime);
                std::vector<llama_token> tokens;
                for (size_t i = 0; i < tokensArr.size(runtime); i++) {
                    tokens.push_back((llama_token)tokensArr.getValueAtIndex(runtime, i).asNumber());
                }

                return createPromiseTask(runtime, callInvoker, [contextId, tokens]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->isVocoderEnabled()) throw std::runtime_error("Vocoder is not enabled");

                    try {
                        auto audio_data = ctx->tts_wrapper->decodeAudioTokens(ctx, tokens);
                        return [audio_data](jsi::Runtime& rt) {
                            jsi::Array res(rt, audio_data.size());
                            for (size_t i = 0; i < audio_data.size(); i++) {
                                res.setValueAtIndex(rt, i, (double)audio_data[i]);
                            }
                            return res;
                        };
                    } catch (const std::exception &e) {
                        throw std::runtime_error(e.what());
                    }
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaDecodeAudioTokens", decodeAudioTokens);

        auto clearCache = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaClearCache"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                bool clearData = count > 1 && arguments[1].isBool() ? arguments[1].asBool() : false;
                return createPromiseTask(runtime, callInvoker, [contextId, clearData]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    throwIfContextBusy(ctx);
                    ctx->clearCache(clearData);
                    return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaClearCache", clearCache);

        auto releaseVocoder = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaReleaseVocoder"),
            1,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                return createPromiseTask(runtime, callInvoker, [contextId]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    throwIfContextBusy(ctx);
                    ctx->releaseVocoder();
                    return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaReleaseVocoder", releaseVocoder);
    }

    void cleanupJSIBindings() {
        TaskManager::getInstance().beginShutdown();
        {
            std::lock_guard<std::mutex> lock(g_log_mutex);
            g_log_handler.reset();
            g_log_invoker.reset();
            g_log_runtime.reset();
        }
        llama_log_set(llama_log_callback_default, nullptr);

        RequestManager::getInstance().clearAll();
        auto contexts = g_llamaContexts.snapshot();
        for (const auto& entry : contexts) {
            long ctxPtr = entry.second;
            if (!ctxPtr) {
                continue;
            }
            auto ctx = reinterpret_cast<rnllama::llama_rn_context*>(ctxPtr);
            if (ctx->completion) {
                ctx->completion->is_interrupted = true;
            }
        }

        if (contexts.empty()) {
            g_context_limit.store(-1);
            return;
        }
        ThreadPool::getInstance().shutdown();

        g_llamaContexts.clear([](long ptr) {
            if (ptr) {
                auto ctx = reinterpret_cast<rnllama::llama_rn_context*>(ptr);
                delete ctx;
            }
        });
        g_context_limit.store(-1);
    }
}
