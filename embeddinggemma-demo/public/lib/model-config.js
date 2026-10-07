// Shared by the browser app, the embedding worker and the Node scripts.

export const MODEL_ID = 'onnx-community/embeddinggemma-300m-ONNX';
// Pinned Hub revision (2025-09-04) so that results stay reproducible.
export const MODEL_REVISION = '5090578d9565bb06545b4552f76e6bc2c93e4a66';

export const TRANSFORMERS_VERSION = '4.3.1';
export const TRANSFORMERS_URL = `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${TRANSFORMERS_VERSION}/dist/transformers.min.js`;

// fp16 / q4f16 exports exist on the Hub but are left out on purpose:
// the model card states that EmbeddingGemma activations do not support fp16.
// q4 uses the "no_gather" export: model_q4.onnx stores the embedding table as
// GatherBlockQuantized, which the WASM backend cannot run.
export const DTYPES = {
  fp32: { label: 'fp32', modelFileName: 'model', suffix: '', sizeMB: 1235 },
  q8: { label: 'q8 (int8)', modelFileName: 'model', suffix: '_quantized', sizeMB: 309 },
  q4: { label: 'q4 (4-bit)', modelFileName: 'model_no_gather', suffix: '_q4', sizeMB: 195 },
};

/** Path of the ONNX graph for a dtype inside the model repo (weights live in `${path}_data`). */
export function onnxFile(dtype) {
  return `onnx/${DTYPES[dtype].modelFileName}${DTYPES[dtype].suffix}.onnx`;
}

// Matryoshka dimensions supported by the model (truncate, then re-normalize).
export const MRL_DIMS = [768, 512, 256, 128];

// Prompts from the EmbeddingGemma model card ("Prompt Instructions").
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
