import RNLlama from './NativeRNLlama'
import './jsi'
import type {
  NativeContextParams,
  NativeLlamaContext,
  NativeCompletionParams,
  NativeCompletionResult,
  NativeTokenizeResult,
  NativeCompletionTokenProb,
  NativeCompletionResultTimings,
  NativeBackendDeviceInfo,
} from './types'
import { BUILD_NUMBER, BUILD_COMMIT } from './version'

export type {
  NativeContextParams,
  NativeLlamaContext,
  NativeCompletionParams,
  NativeCompletionTokenProb,
  NativeCompletionResult,
  NativeTokenizeResult,
  NativeCompletionResultTimings,
  NativeBackendDeviceInfo,
}

const logListeners: Array<(level: string, text: string) => void> = []
const emitNativeLog = (level: string, text: string) => {
  logListeners.forEach((listener) => listener(level, text))
}

const jsiBindingKeys = [
  'llamaInitContext',
  'llamaReleaseContext',
  'llamaReleaseAllContexts',
  'llamaModelInfo',
  'llamaGetBackendDevicesInfo',
  'llamaTokenize',
  'llamaDetokenize',
  'llamaToggleNativeLog',
  'llamaSetContextLimit',
  'llamaCompletion',
  'llamaStopCompletion',
  'llamaInitVocoder',
  'llamaIsVocoderEnabled',
  'llamaGetFormattedAudioCompletion',
  'llamaGetAudioCompletionGuideTokens',
  'llamaDecodeAudioTokens',
  'llamaReleaseVocoder',
  'llamaClearCache',
] as const

type JsiBindingKey = (typeof jsiBindingKeys)[number]
type JsiBindings = { [K in JsiBindingKey]: NonNullable<(typeof globalThis)[K]> }

let jsiBindings: JsiBindings | null = null

const bindJsiFromGlobal = () => {
  const bindings: Partial<JsiBindings> = {}
  const missing: string[] = []

  jsiBindingKeys.forEach((key) => {
    const value = global[key]
    if (typeof value === 'function') {
      ;(bindings as Record<string, unknown>)[key] =
        value as JsiBindings[typeof key]
      delete global[key]
    } else {
      missing.push(key)
    }
  })

  if (missing.length > 0) {
    throw new Error(`[RNLlama] Missing JSI bindings: ${missing.join(', ')}`)
  }

  jsiBindings = bindings as JsiBindings
}

const getJsi = (): JsiBindings => {
  if (!jsiBindings) {
    throw new Error('JSI bindings not installed')
  }
  return jsiBindings
}

let isJsiInstalled = false
export const installJsi = async () => {
  if (isJsiInstalled) return
  if (typeof global.llamaInitContext !== 'function') {
    const installed = await RNLlama.install()
    if (!installed && typeof global.llamaInitContext !== 'function') {
      throw new Error('JSI bindings not installed')
    }
  }
  bindJsiFromGlobal()
  isJsiInstalled = true
}

export type TokenData = {
  token: string
  completion_probabilities?: Array<NativeCompletionTokenProb>
}

export type ContextParams = Omit<
  NativeContextParams,
  'flash_attn_type' | 'cache_type_k' | 'cache_type_v'
> & {
  flash_attn_type?: 'auto' | 'on' | 'off'
  cache_type_k?:
    | 'f16'
    | 'f32'
    | 'q8_0'
    | 'q4_0'
    | 'q4_1'
    | 'iq4_nl'
    | 'q5_0'
    | 'q5_1'
  cache_type_v?:
    | 'f16'
    | 'f32'
    | 'q8_0'
    | 'q4_0'
    | 'q4_1'
    | 'iq4_nl'
    | 'q5_0'
    | 'q5_1'
}

const validCacheTypes = [
  'f16',
  'f32',
  'bf16',
  'q8_0',
  'q4_0',
  'q4_1',
  'iq4_nl',
  'q5_0',
  'q5_1',
]

export type CompletionParams = Omit<
  NativeCompletionParams,
  'emit_partial_completion' | 'prompt'
> & {
  prompt?: string
}

type NativeCompletionRequestParams = NativeCompletionParams

export class LlamaContext {
  id: number
  gpu: boolean = false
  reasonNoGPU: string = ''
  devices: NativeLlamaContext['devices']
  model: NativeLlamaContext['model']
  androidLib: NativeLlamaContext['androidLib']
  systemInfo: NativeLlamaContext['systemInfo']

  constructor({
    contextId,
    gpu,
    devices,
    reasonNoGPU,
    model,
    androidLib,
    systemInfo,
  }: NativeLlamaContext) {
    this.id = contextId
    this.gpu = gpu
    this.devices = devices
    this.reasonNoGPU = reasonNoGPU
    this.model = model
    this.androidLib = androidLib
    this.systemInfo = systemInfo
  }

  async completion(
    params: CompletionParams,
    callback?: (data: TokenData) => void,
  ): Promise<NativeCompletionResult> {
    const nativeParams: NativeCompletionRequestParams = {
      ...params,
      prompt: params.prompt || '',
      emit_partial_completion: !!callback,
    }

    if (!nativeParams.prompt) throw new Error('Prompt is required')

    const { llamaCompletion } = getJsi()
    return llamaCompletion(this.id, nativeParams, callback)
  }

  stopCompletion(): Promise<void> {
    const { llamaStopCompletion } = getJsi()
    return llamaStopCompletion(this.id)
  }

  tokenize(text: string): Promise<NativeTokenizeResult> {
    const { llamaTokenize } = getJsi()
    return llamaTokenize(this.id, text)
  }

  detokenize(tokens: number[]): Promise<string> {
    const { llamaDetokenize } = getJsi()
    return llamaDetokenize(this.id, tokens)
  }

  async initVocoder({
    path,
    n_batch: nBatch,
  }: {
    path: string
    n_batch?: number
  }): Promise<boolean> {
    const { llamaInitVocoder } = getJsi()
    if (path.startsWith('file://')) path = path.slice(7)
    return await llamaInitVocoder(this.id, { path, n_batch: nBatch })
  }

  async isVocoderEnabled(): Promise<boolean> {
    const { llamaIsVocoderEnabled } = getJsi()
    return await llamaIsVocoderEnabled(this.id)
  }

  async getFormattedAudioCompletion(
    speaker: object | null,
    textToSpeak: string,
  ): Promise<{
    prompt: string
    grammar?: string
  }> {
    const { llamaGetFormattedAudioCompletion } = getJsi()
    return await llamaGetFormattedAudioCompletion(
      this.id,
      speaker ? JSON.stringify(speaker) : '',
      textToSpeak,
    )
  }

  async getAudioCompletionGuideTokens(
    textToSpeak: string,
  ): Promise<Array<number>> {
    const { llamaGetAudioCompletionGuideTokens } = getJsi()
    return await llamaGetAudioCompletionGuideTokens(this.id, textToSpeak)
  }

  async decodeAudioTokens(tokens: number[]): Promise<Array<number>> {
    const { llamaDecodeAudioTokens } = getJsi()
    return await llamaDecodeAudioTokens(this.id, tokens)
  }

  async releaseVocoder(): Promise<void> {
    const { llamaReleaseVocoder } = getJsi()
    return await llamaReleaseVocoder(this.id)
  }

  async clearCache(clearData: boolean = false): Promise<void> {
    const { llamaClearCache } = getJsi()
    return llamaClearCache(this.id, clearData)
  }

  async release(): Promise<void> {
    const { llamaReleaseContext } = getJsi()
    return llamaReleaseContext(this.id)
  }
}

export async function toggleNativeLog(enabled: boolean): Promise<void> {
  await installJsi()
  const { llamaToggleNativeLog } = getJsi()
  return llamaToggleNativeLog(enabled, emitNativeLog)
}

export function addNativeLogListener(
  listener: (level: string, text: string) => void,
): { remove: () => void } {
  logListeners.push(listener)
  return {
    remove: () => {
      logListeners.splice(logListeners.indexOf(listener), 1)
    },
  }
}

export async function setContextLimit(limit: number): Promise<void> {
  await installJsi()
  const { llamaSetContextLimit } = getJsi()
  return llamaSetContextLimit(limit)
}

let contextIdCounter = 0
const contextIdRandom = () =>
  /* @ts-ignore */
  process.env.NODE_ENV === 'test' ? 0 : Math.floor(Math.random() * 100000)

const modelInfoSkip = [
  'tokenizer.ggml.tokens',
  'tokenizer.ggml.token_type',
  'tokenizer.ggml.merges',
  'tokenizer.ggml.scores',
]
export async function loadLlamaModelInfo(model: string): Promise<Object> {
  await installJsi()
  const { llamaModelInfo } = getJsi()
  let path = model
  if (path.startsWith('file://')) path = path.slice(7)
  return llamaModelInfo(path, modelInfoSkip)
}

export async function getBackendDevicesInfo(): Promise<
  Array<NativeBackendDeviceInfo>
> {
  await installJsi()
  const { llamaGetBackendDevicesInfo } = getJsi()
  try {
    const jsonString = await llamaGetBackendDevicesInfo()
    return JSON.parse(jsonString as string)
  } catch (e) {
    console.warn(
      '[RNLlama] Failed to parse backend devices info, falling back to empty list',
      e,
    )
    return []
  }
}

export async function initLlama(
  {
    model,
    is_model_asset: isModelAsset,
    devices,
    ...rest
  }: ContextParams,
  onProgress?: (progress: number) => void,
): Promise<LlamaContext> {
  await installJsi()
  const { llamaInitContext } = getJsi()
  let path = model
  if (path.startsWith('file://')) path = path.slice(7)

  const contextId = contextIdCounter + contextIdRandom()
  contextIdCounter += 1

  let lastProgress = 0
  const progressCallback = onProgress
    ? (progress: number) => {
        lastProgress = progress
        try {
          onProgress(progress)
        } catch (err) {
          console.warn('[RNLlama] onProgress callback failed', err)
        }
      }
    : undefined

  if (progressCallback) progressCallback(0)

  if (rest.cache_type_k && !validCacheTypes.includes(rest.cache_type_k)) {
    console.warn(
      `[RNLlama] initLlama: Invalid cache K type: ${rest.cache_type_k}, falling back to f16`,
    )
    delete rest.cache_type_k
  }
  if (rest.cache_type_v && !validCacheTypes.includes(rest.cache_type_v)) {
    console.warn(
      `[RNLlama] initLlama: Invalid cache V type: ${rest.cache_type_v}, falling back to f16`,
    )
    delete rest.cache_type_v
  }

  const {
    gpu,
    devices: usedDevices,
    reasonNoGPU,
    model: modelDetails,
    androidLib,
    systemInfo,
  } = await llamaInitContext(
    contextId,
    {
      model: path,
      is_model_asset: !!isModelAsset,
      use_progress_callback: !!progressCallback,
      devices: devices,
      ...rest,
    },
    progressCallback,
  )

  if (progressCallback && lastProgress < 100) progressCallback(100)

  return new LlamaContext({
    contextId,
    gpu,
    devices: usedDevices,
    reasonNoGPU,
    model: modelDetails,
    androidLib,
    systemInfo,
  })
}

export async function releaseAllLlama(): Promise<void> {
  if (!isJsiInstalled) return
  const { llamaReleaseAllContexts } = getJsi()
  return llamaReleaseAllContexts()
}

export const BuildInfo = {
  number: BUILD_NUMBER,
  commit: BUILD_COMMIT,
}
