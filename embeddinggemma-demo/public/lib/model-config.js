// Shared by the browser app, the embedding worker and the Node scripts.

export const TRANSFORMERS_VERSION = '4.3.1';
export const TRANSFORMERS_URL = `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${TRANSFORMERS_VERSION}/dist/transformers.min.js`;

// The two EmbeddingGemma generations. Both take the same prompts and return 768-dimensional
// vectors that can be cut to 512/256/128 dimensions (Matryoshka). Revisions are pinned so that
// results stay reproducible.
//
// fp16 / q4f16 exports exist on the Hub for both but are left out on purpose: both model cards
// state that the activations overflow float16.
export const MODELS = {
  v1: {
    name: 'EmbeddingGemma 300M',
    label: 'EmbeddingGemma 初代（300M）',
    id: 'onnx-community/embeddinggemma-300m-ONNX',
    revision: '5090578d9565bb06545b4552f76e6bc2c93e4a66', // 2025-09-04
    card: 'https://huggingface.co/google/embeddinggemma-300m',
    files: ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'],
    charVectors: 'data/char-vectors.json',
    // q4 uses the "no_gather" export: model_q4.onnx stores the embedding table as
    // GatherBlockQuantized, which the WASM backend cannot run.
    dtypes: {
      fp32: { label: 'fp32', modelFileName: 'model', suffix: '', sizeMB: 1235 },
      q8: { label: 'q8 (int8)', modelFileName: 'model', suffix: '_quantized', sizeMB: 309 },
      q4: { label: 'q4 (4-bit)', modelFileName: 'model_no_gather', suffix: '_q4', sizeMB: 195 },
    },
  },
  v2: {
    name: 'EmbeddingGemma 2',
    label: 'EmbeddingGemma 2（テキスト部分 270M）',
    id: 'onnx-community/embeddinggemma-2-ONNX',
    revision: 'daa72c51243991dfcaf9f9137d2c573d8f7790c0', // 2026-10-06
    card: 'https://huggingface.co/google/embeddinggemma-2',
    files: ['config.json', 'tokenizer.json', 'tokenizer_config.json'],
    charVectors: 'data/char-vectors-v2.json',
    // The repo also holds the image and audio encoders; only the text model is loaded (see textOnlyConfig).
    textOnly: true,
    // Every quantized export keeps the embedding table as GatherBlockQuantized and there is no
    // "no_gather" variant, so in the browser q8 and q4 need WebGPU; WASM runs fp32 only.
    dtypes: {
      fp32: { label: 'fp32', modelFileName: 'model', suffix: '', sizeMB: 1085 },
      q8: { label: 'q8 (int8)', modelFileName: 'model', suffix: '_quantized', sizeMB: 314, webgpuOnly: true },
      q4: { label: 'q4 (4-bit)', modelFileName: 'model', suffix: '_q4', sizeMB: 175, webgpuOnly: true },
    },
  },
};

export const DEFAULT_MODEL = 'v1';

/** Path of the ONNX graph for a dtype inside the model repo (weights live in `${path}_data`). */
export function onnxFile(model, dtype) {
  const { modelFileName, suffix } = MODELS[model].dtypes[dtype];
  return `onnx/${modelFileName}${suffix}.onnx`;
}

/**
 * The config to pass to AutoModel.from_pretrained: for EmbeddingGemma 2 the image and audio
 * encoders are dropped so that only the text model is downloaded and loaded (as its model card shows).
 * `AutoConfig` comes from transformers.js; returns undefined when the default config is fine.
 */
export async function textOnlyConfig(AutoConfig, model, options = {}) {
  const { id, revision, textOnly } = MODELS[model];
  if (!textOnly) return undefined;
  const config = await AutoConfig.from_pretrained(id, { ...options, revision });
  config.vision_config = config.audio_config = null;
  return config;
}

// Matryoshka dimensions supported by the model (truncate, then re-normalize).
export const MRL_DIMS = [768, 512, 256, 128];

// Prompts from the EmbeddingGemma model card ("Prompt Instructions"); EmbeddingGemma 2 uses the same ones.
export const QUERY_TASKS = {
  'search result': 'task: search result | query: ',
  'question answering': 'task: question answering | query: ',
  'fact checking': 'task: fact checking | query: ',
  classification: 'task: classification | query: ',
  clustering: 'task: clustering | query: ',
  'sentence similarity': 'task: sentence similarity | query: ',
  'code retrieval': 'task: code retrieval | query: ',
};

export const NO_PROMPT = 'none';

export function queryPrompt(task) {
  return task === NO_PROMPT ? '' : QUERY_TASKS[task] ?? QUERY_TASKS['search result'];
}

export function documentPrompt(title, enabled = true) {
  return enabled ? `title: ${title?.trim() || 'none'} | text: ` : '';
}
