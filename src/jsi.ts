/* eslint-disable no-var */
import type {
  NativeContextParams,
  NativeCompletionParams,
  NativeCompletionResult,
  NativeTokenizeResult,
} from './types'

declare global {
  var llamaInitContext: (
    contextId: number,
    params: NativeContextParams,
    onProgress?: (progress: number) => void,
  ) => Promise<any>
  var llamaReleaseContext: (contextId: number) => Promise<void>
  var llamaReleaseAllContexts: () => Promise<void>
  var llamaModelInfo: (path: string, skip: string[]) => Promise<object>
  var llamaGetBackendDevicesInfo: () => Promise<string>
  var llamaTokenize: (
    contextId: number,
    text: string,
  ) => Promise<NativeTokenizeResult>
  var llamaDetokenize: (contextId: number, tokens: number[]) => Promise<string>
  var llamaToggleNativeLog: (
    enabled: boolean,
    onLog?: (level: string, text: string) => void,
  ) => Promise<void>
  var llamaSetContextLimit: (limit: number) => Promise<void>
  var llamaCompletion: (
    contextId: number,
    params: NativeCompletionParams,
    onToken?: (token: any) => void,
  ) => Promise<NativeCompletionResult>
  var llamaStopCompletion: (contextId: number) => Promise<void>
  var llamaInitVocoder: (
    contextId: number,
    params: { path: string; n_batch?: number },
  ) => Promise<boolean>
  var llamaIsVocoderEnabled: (contextId: number) => Promise<boolean>
  var llamaGetFormattedAudioCompletion: (
    contextId: number,
    speaker: string,
    text: string,
  ) => Promise<{ prompt: string; grammar?: string }>
  var llamaGetAudioCompletionGuideTokens: (
    contextId: number,
    text: string,
  ) => Promise<number[]>
  var llamaDecodeAudioTokens: (
    contextId: number,
    tokens: number[],
  ) => Promise<number[]>
  var llamaReleaseVocoder: (contextId: number) => Promise<void>
  var llamaClearCache: (contextId: number, clearData: boolean) => Promise<void>
}
