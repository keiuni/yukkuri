// The server a chat model of scripts/lib/chat-models.mjs runs on: llama-server for the GGUF files,
// LiteRT-LM's server for Gemma 4's phone builds. Both answer the same OpenAI-style chat requests.
import { LlamaServer } from '../../agent/llama.mjs';
import { CHAT_MODELS, serverSpec } from './chat-models.mjs';
import { LiteRtServer } from './litert-server.mjs';

export function startChatServer(key, port) {
  const model = CHAT_MODELS[key];
  return model.runtime === 'litert-lm' ? LiteRtServer.start({ key, ...model }, port) : LlamaServer.start(serverSpec(key), port);
}

/** What every chat request for `model` carries besides the messages: its sampling, thinking off, a length cap. */
export function requestOptions(model, { maxTokens, seed }) {
  return {
    max_tokens: maxTokens,
    max_completion_tokens: maxTokens, // the name LiteRT-LM's server reads
    seed,
    ...model.sampling,
    ...(model.templateKwargs ? { chat_template_kwargs: model.templateKwargs } : {}),
    ...model.extraBody,
  };
}
