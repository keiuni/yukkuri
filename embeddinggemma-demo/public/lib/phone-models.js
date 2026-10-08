// Models tried by the phone check page (public/phone.html) and scripts/phone-check.mjs.
//
// sizeMB is what a text-only run downloads (ONNX graph + weights) at the pinned revision.
// maxTensorMB is the largest single weight in those files: on WebGPU every weight becomes a GPU
// buffer, so it has to fit in adapter.limits.maxBufferSize and maxStorageBufferBindingSize.
// Both were read from the ONNX files on the Hub (2026-10-08).
//
// The 4-bit and 8-bit Gemma LLM exports (and EmbeddingGemma 2's) keep their embedding table as
// GatherBlockQuantized, which the WASM (CPU) build of onnxruntime-web does not implement, so they only
// run on WebGPU. On WASM that leaves fp32 (Gemma 3 1B: 4.1 GB, Gemma 4 E2B: 20.5 GB), which is only
// practical for the small models.
import { MODELS } from './model-config.js';

// The text-generation tests answer this (or what the user types instead), greedily, with at most
// GENERATION_MAX_TOKENS tokens unless a longer answer is chosen.
export const GENERATION_PROMPT = 'スマートフォンの中でAIを動かす利点を、短く3つ挙げてください。';
export const GENERATION_MAX_TOKENS = 64;
export const GENERATION_TOKEN_CHOICES = [64, 160, 320];

export const PHONE_MODELS = [
  {
    key: 'embeddinggemma',
    label: 'EmbeddingGemma 300M',
    task: 'embedding',
    // `embedding` is the key in model-config.js MODELS.
    embedding: 'v1',
    id: MODELS.v1.id,
    revision: MODELS.v1.revision,
    sessions: ['model'],
    variants: [
      // q4 uses the "no_gather" export because WASM cannot run GatherBlockQuantized.
      { dtype: 'q4', device: 'wasm', sizeMB: 195, maxTensorMB: 101, fileName: 'model_no_gather' },
      { dtype: 'q8', device: 'wasm', sizeMB: 309, maxTensorMB: 201 },
    ],
  },
  {
    key: 'embeddinggemma-2',
    label: 'EmbeddingGemma 2（テキスト）',
    task: 'embedding',
    embedding: 'v2',
    id: MODELS.v2.id,
    revision: MODELS.v2.revision,
    // Only the text model is loaded, not the image and audio encoders.
    sessions: ['model'],
    // The largest weight is the embedding table (262144 x 512): 4-bit 67 MB, 8-bit exactly 128 MiB.
    variants: [
      { dtype: 'q4', device: 'webgpu', sizeMB: 175, maxTensorMB: 67 },
      { dtype: 'q8', device: 'webgpu', sizeMB: 314, maxTensorMB: 134 },
      { dtype: 'fp32', device: 'wasm', sizeMB: 1085, maxTensorMB: 537 },
    ],
  },
  {
    key: 'gemma-3-270m',
    label: 'Gemma 3 270M',
    task: 'text-generation',
    id: 'onnx-community/gemma-3-270m-it-ONNX',
    revision: '2dbbfdb1b59bd034eb959428c6a7da9dd7ea27f0',
    sessions: ['model'],
    variants: [
      { dtype: 'q4f16', device: 'webgpu', sizeMB: 273, maxTensorMB: 84, f16: true },
      { dtype: 'q4', device: 'webgpu', sizeMB: 323, maxTensorMB: 84 },
      // The only export that runs on WASM: q4 / q8 store the embedding table as GatherBlockQuantized.
      { dtype: 'fp32', device: 'wasm', sizeMB: 1140, maxTensorMB: 671 },
    ],
  },
  {
    key: 'gemma-3-1b',
    label: 'Gemma 3 1B',
    task: 'text-generation',
    id: 'onnx-community/gemma-3-1b-it-ONNX',
    revision: 'a58439f40017d3b99c7d378ff525e54e0ba08ebf',
    sessions: ['model'],
    variants: [
      { dtype: 'q4f16', device: 'webgpu', sizeMB: 763, maxTensorMB: 151, f16: true },
      { dtype: 'q4', device: 'webgpu', sizeMB: 859, maxTensorMB: 151 },
    ],
  },
  {
    key: 'lfm2.5-1.2b-jp',
    label: 'LFM2.5-1.2B-JP（Liquid AI・日本語）',
    task: 'text-generation',
    id: 'LiquidAI/LFM2.5-1.2B-JP-202606-ONNX',
    revision: '01b5a0a33d22253d5ee52b367d2375eeb113b7cd',
    sessions: ['model'],
    // A Japanese chat model (2026-06). Its 4-bit embedding table (65536 x 2048) is the largest weight, and
    // it is stored as GatherBlockQuantized, so it needs WebGPU.
    variants: [
      { dtype: 'q4f16', device: 'webgpu', sizeMB: 744, maxTensorMB: 67, f16: true },
      { dtype: 'q4', device: 'webgpu', sizeMB: 834, maxTensorMB: 67 },
    ],
  },
  {
    key: 'lfm2.5-2.6b',
    label: 'LFM2.5-2.6B（Liquid AI）',
    task: 'text-generation',
    id: 'LiquidAI/LFM2.5-2.6B-ONNX',
    revision: '66826372fd4fa166f53be0371c9315745c07cace',
    sessions: ['model'],
    // A reasoning model: it always thinks first, so short generations are all thinking. Its 4-bit embedding
    // table (128000 x 1024 bytes) is the largest weight and fits WebGPU's default 128 MiB limit.
    variants: [
      { dtype: 'q4f16', device: 'webgpu', sizeMB: 1534, maxTensorMB: 131, f16: true },
      { dtype: 'q4', device: 'webgpu', sizeMB: 1855, maxTensorMB: 131 },
    ],
  },
  {
    key: 'gemma-4-e2b',
    label: 'Gemma 4 E2B',
    task: 'text-generation',
    id: 'onnx-community/gemma-4-E2B-it-ONNX',
    revision: '7c6d3d1d4092253ea241428e88312ab34bfa9c26',
    // Text-only runs load these two of the repo's four sessions (no vision or audio encoder).
    sessions: ['embed_tokens', 'decoder_model_merged'],
    // The per-layer embedding table (262144 x 4480, 4-bit) is one 1174 MB tensor.
    variants: [
      { dtype: 'q4f16', device: 'webgpu', sizeMB: 3110, maxTensorMB: 1174, f16: true },
      { dtype: 'q4', device: 'webgpu', sizeMB: 3627, maxTensorMB: 1174 },
    ],
  },
];

// Google's own browser build of Gemma 4 E2B (LiteRT-LM, text only). It runs in its own page, so the
// check page only links to it. Its loader requires shader-f16 and warns below 524550144-byte buffers.
export const LITERT_GEMMA4 = {
  label: 'Gemma 4 E2B（LiteRT-LM Web 版）',
  url: 'https://huggingface.co/spaces/tylermullen/Gemma4',
  file: 'litert-community/gemma-4-E2B-it-litert-lm / gemma-4-E2B-it-gpu.litertlm',
  sizeMB: 2008,
  minBufferBytes: 524550144,
};

// wasm32 memory tops out at 4 GiB, and the weights are copied into it.
const WASM_LIMIT_MB = 3000;
// Above this the run usually still works on a PC but risks the tab being killed on a phone.
const PHONE_WASM_COMFORT_MB = 600;
const BIG_DOWNLOAD_MB = 1500;

export const variantKey = (model, variant) => `${model.key}:${variant.dtype}:${variant.device}`;

export function findVariant(key) {
  const [modelKey, dtype, device] = key.split(':');
  const model = PHONE_MODELS.find((m) => m.key === modelKey);
  const variant = model?.variants.find((v) => v.dtype === dtype && v.device === device);
  return model && variant ? { model, variant } : null;
}

const mb = (bytes) => bytes / 1e6;
/** Whole megabytes, rounded down (a limit is never rounded up), with thousands separators. */
export const formatMB = (megabytes) => `${Math.floor(megabytes).toLocaleString('en-US')} MB`;

/**
 * Rough verdict for one variant on this device, from the published limits only.
 * `device`: { mobile, deviceMemoryGB|null, webgpu: null | { f16, maxBufferSize, maxStorageBufferBindingSize } }
 * Returns { level: 'yes' | 'maybe' | 'no', reasons: string[] }.
 */
export function assessVariant(variant, device) {
  const reasons = [];
  let level = 'yes';
  const lower = (to) => {
    if (to === 'no' || (to === 'maybe' && level === 'yes')) level = to;
  };

  if (variant.device === 'webgpu') {
    const gpu = device.webgpu;
    if (!gpu) return { level: 'no', reasons: ['WebGPU が使えない'] };
    if (variant.f16 && !gpu.f16) {
      lower('no');
      reasons.push('GPU が f16（shader-f16）に対応していない');
    }
    const limit = Math.min(gpu.maxBufferSize, gpu.maxStorageBufferBindingSize);
    if (variant.maxTensorMB > mb(limit)) {
      lower('no');
      reasons.push(`最大の重み ${formatMB(variant.maxTensorMB)} が GPU の 1 バッファの上限 ${formatMB(mb(limit))} を超える`);
    }
  } else if (variant.sizeMB > WASM_LIMIT_MB) {
    lower('no');
    reasons.push(`重み ${(variant.sizeMB / 1000).toFixed(1)} GB が WASM のメモリ上限（4 GB）に収まらない`);
  } else if (device.mobile && variant.sizeMB > PHONE_WASM_COMFORT_MB) {
    lower('maybe');
    reasons.push('WASM で重みを全部メモリに載せるので、スマホではタブが落ちることがある');
  }

  if (device.deviceMemoryGB && variant.sizeMB / 1000 > device.deviceMemoryGB * 0.5) {
    lower('maybe');
    reasons.push(`端末メモリ（約 ${device.deviceMemoryGB} GB）に対して大きい`);
  }
  if (variant.sizeMB > BIG_DOWNLOAD_MB) {
    lower('maybe');
    reasons.push(`ダウンロードが ${(variant.sizeMB / 1000).toFixed(1)} GB ある`);
  }
  return { level, reasons };
}

/** Same idea for the LiteRT-LM build, which only runs on WebGPU with f16. */
export function assessLiteRt(device) {
  const gpu = device.webgpu;
  if (!gpu) return { level: 'no', reasons: ['WebGPU が使えない'] };
  const reasons = [];
  let level = 'maybe';
  if (!gpu.f16) {
    level = 'no';
    reasons.push('GPU が f16（shader-f16）に対応していない');
  }
  if (Math.min(gpu.maxBufferSize, gpu.maxStorageBufferBindingSize) < LITERT_GEMMA4.minBufferBytes) {
    level = 'no';
    reasons.push(`GPU の 1 バッファの上限が ${formatMB(mb(LITERT_GEMMA4.minBufferBytes))} 未満`);
  }
  if (level !== 'no') reasons.push('ダウンロード約 2.0 GB、GPU メモリ約 1.8 GB（公式の M4 Max での値）');
  return { level, reasons };
}
