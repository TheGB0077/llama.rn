export type NativeContextParams = {
  model: string
  is_model_asset?: boolean
  use_progress_callback?: boolean
  n_ctx?: number
  n_batch?: number
  n_ubatch?: number
  n_threads?: number
  cpu_mask?: string
  cpu_strict?: boolean
  n_gpu_layers?: number
  devices?: Array<string>
  no_gpu_devices?: boolean
  flash_attn_type?: string
  flash_attn?: boolean
  cache_type_k?: string
  cache_type_v?: string
  use_mlock?: boolean
  use_mmap?: boolean
  vocab_only?: boolean
  no_extra_bufts?: boolean
  rope_freq_base?: number
  rope_freq_scale?: number
  ctx_shift?: boolean
  n_cpu_moe?: number
}

export type NativeCompletionParams = {
  prompt: string
  n_threads?: number
  json_schema?: string
  grammar?: string
  grammar_lazy?: boolean
  stop?: Array<string>
  n_predict?: number
  n_probs?: number
  top_k?: number
  top_p?: number
  min_p?: number
  xtc_probability?: number
  xtc_threshold?: number
  typical_p?: number
  temperature?: number
  penalty_last_n?: number
  penalty_repeat?: number
  penalty_freq?: number
  penalty_present?: number
  mirostat?: number
  mirostat_tau?: number
  mirostat_eta?: number
  dry_multiplier?: number
  dry_base?: number
  dry_allowed_length?: number
  dry_penalty_last_n?: number
  dry_sequence_breakers?: Array<string>
  top_n_sigma?: number
  ignore_eos?: boolean
  logit_bias?: Array<Array<number>>
  seed?: number
  guide_tokens?: Array<number>
  emit_partial_completion: boolean
}

export type NativeCompletionTokenProbItem = {
  tok_str: string
  prob: number
}

export type NativeCompletionTokenProb = {
  content: string
  probs: Array<NativeCompletionTokenProbItem>
}

export type NativeCompletionResultTimings = {
  cache_n: number
  prompt_n: number
  prompt_ms: number
  prompt_per_token_ms: number
  prompt_per_second: number
  predicted_n: number
  predicted_ms: number
  predicted_per_token_ms: number
  predicted_per_second: number
}

export type NativeCompletionResult = {
  text: string
  tokens_predicted: number
  tokens_evaluated: number
  truncated: boolean
  stopped_eos: boolean
  stopped_word: string
  stopped_limit: number
  stopping_word: string
  context_full: boolean
  interrupted: boolean
  tokens_cached: number
  timings: NativeCompletionResultTimings
  completion_probabilities?: Array<NativeCompletionTokenProb>
  audio_tokens?: Array<number>
}

export type NativeTokenizeResult = {
  tokens: Array<number>
}

export type NativeLlamaContext = {
  contextId: number
  model: {
    desc: string
    size: number
    nEmbd: number
    nParams: number
    is_recurrent: boolean
    is_hybrid: boolean
    metadata: Object
  }
  androidLib?: string
  devices?: Array<string>
  gpu: boolean
  reasonNoGPU: string
  systemInfo: string
}

export type NativeBackendDeviceInfo = {
  backend: string
  type: string
  deviceName: string
  maxMemorySize: number
  metadata?: Record<string, any>
}
