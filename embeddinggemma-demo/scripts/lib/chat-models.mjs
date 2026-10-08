// Small chat models that run on a phone (in an app or in the browser), as GGUF files for llama.cpp's
// llama-server, for the Japanese reply and conversation test (scripts/ja-chat.mjs).
// Each model gets the sampling its model card recommends, with thinking turned off: a chat app wants
// the reply right away. Revisions are pinned (2026-10-08).
export const CHAT_MODELS = {
  'gemma4-e2b': {
    name: 'Gemma 4 E2B',
    repo: 'google/gemma-4-E2B-it-qat-q4_0-gguf',
    revision: '675cff42a74c774d6cb76f76d8eacb49b48c9b93',
    file: 'gemma-4-E2B_q4_0-it.gguf',
    // Google's quantization-aware-trained 4-bit weights (the phone builds use a similar mixed 2/4/8-bit scheme).
    quant: 'Q4_0（QAT）',
    phone: 'アプリ（Google AI Edge Gallery、Android / iOS）。ブラウザはハイエンド機の LiteRT-LM Web 版だけ',
    sampling: { temperature: 1.0, top_p: 0.95, top_k: 64 },
    templateKwargs: { enable_thinking: false },
  },
  'gemma4-e4b': {
    name: 'Gemma 4 E4B',
    repo: 'google/gemma-4-E4B-it-qat-q4_0-gguf',
    revision: '4b4a2c1d584be7264f87aac328a1bc739ce81b6c',
    file: 'gemma-4-E4B_q4_0-it.gguf',
    quant: 'Q4_0（QAT）',
    phone: 'アプリ（Google AI Edge Gallery）。メモリに余裕のあるハイエンド機向け',
    sampling: { temperature: 1.0, top_p: 0.95, top_k: 64 },
    templateKwargs: { enable_thinking: false },
  },
  'gemma3-1b': {
    name: 'Gemma 3 1B',
    repo: 'ggml-org/gemma-3-1b-it-GGUF',
    revision: 'f9c28bcd85737ffc5aef028638d3341d49869c27',
    file: 'gemma-3-1b-it-Q4_K_M.gguf',
    quant: 'Q4_K_M',
    phone: 'ブラウザ（WebGPU、transformers.js の q4 は 0.8 GB）',
    sampling: { temperature: 1.0, top_p: 0.95, top_k: 64 },
  },
  'gemma3-270m': {
    name: 'Gemma 3 270M',
    repo: 'unsloth/gemma-3-270m-it-GGUF',
    revision: 'c90975dbd40c0c7b275fefaae758c3415c906238',
    file: 'gemma-3-270m-it-Q8_0.gguf',
    quant: 'Q8_0',
    phone: 'ブラウザ（WebGPU で 0.3 GB）',
    sampling: { temperature: 1.0, top_p: 0.95, top_k: 64 },
  },
  'lfm25-jp': {
    name: 'LFM2.5-1.2B-JP',
    repo: 'LiquidAI/LFM2.5-1.2B-JP-202606-GGUF',
    revision: '448ba3f7d408c2f5c32cec8038612f7c1ed9f054',
    file: 'LFM2.5-1.2B-JP-202606-Q4_K_M.gguf',
    quant: 'Q4_K_M',
    phone: 'ブラウザ（WebGPU、ONNX q4f16 は 0.74 GB）、アプリ（LiteRT-LM int4 0.74 GB）',
    sampling: { temperature: 0.1, top_k: 50, repeat_penalty: 1.05 },
  },
  tinyswallow: {
    name: 'TinySwallow-1.5B',
    repo: 'SakanaAI/TinySwallow-1.5B-Instruct-GGUF',
    revision: '38c003aaf8be9d17af11dece1fbabeb873c567fa',
    file: 'tinyswallow-1.5b-instruct-q5_k_m.gguf',
    quant: 'Q5_K_M',
    phone: 'ブラウザ（Sakana AI の WebGPU デモあり）',
    // generation_config.json of the model.
    sampling: { temperature: 0.7, top_p: 0.8, top_k: 20, repeat_penalty: 1.1 },
  },
  'qwen35-2b': {
    name: 'Qwen3.5-2B',
    repo: 'unsloth/Qwen3.5-2B-GGUF',
    revision: 'f6d5376be1edb4d416d56da11e5397a961aca8ae',
    file: 'Qwen3.5-2B-Q4_K_M.gguf',
    quant: 'Q4_K_M',
    phone: 'アプリ（llama.cpp 系）、ブラウザ（WebGPU）',
    // The model card's non-thinking settings for text.
    sampling: { temperature: 1.0, top_p: 1.0, top_k: 20, min_p: 0, presence_penalty: 2.0 },
    templateKwargs: { enable_thinking: false },
  },
};

/** The llama-server spec of a chat model: llama.mjs starts it like a Liquid AI model. */
export function serverSpec(key) {
  const model = CHAT_MODELS[key];
  return { ...model, kind: 'chat', args: ['-c', '8192', '--jinja'] };
}
