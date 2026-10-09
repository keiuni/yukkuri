#!/usr/bin/env node
// Downloads the GGUF files of the phone-sized chat models (scripts/lib/chat-models.mjs) into ./models for
// the Japanese chat test (scripts/ja-chat.mjs). About 13 GB in total; Gemma 4 E4B alone is 5.2 GB.
//
//   npm run download-chat-models                          # all the GGUF files
//   npm run download-chat-models -- gemma4-e2b lfm25-jp   # some of them
//   npm run download-chat-models -- gemma4-e4b-litert gemma4-e2b-litert   # Gemma 4's phone builds (3.7 + 2.6 GB)
//
// llama-server comes from a llama.cpp build (see scripts/download-liquid-models.mjs).
// Behind an HTTP proxy, run it with NODE_USE_ENV_PROXY=1 (Node >= 22.21).
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { CHAT_MODELS, GGUF_MODEL_KEYS } from './lib/chat-models.mjs';
import { downloadFile } from './lib/hub.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const keys = process.argv.slice(2).filter((arg) => !arg.startsWith('-'));
for (const key of keys) {
  if (!CHAT_MODELS[key]) {
    console.error(`Unknown model "${key}". Choose from: ${Object.keys(CHAT_MODELS).join(', ')}`);
    process.exit(1);
  }
}

for (const key of keys.length ? keys : GGUF_MODEL_KEYS) {
  const { repo, revision, file } = CHAT_MODELS[key];
  console.log(`Downloading ${repo}@${revision.slice(0, 7)} ${file}`);
  await downloadFile({ modelId: repo, revision, file, targetDir: path.join(ROOT, 'models', repo), force: process.argv.includes('--force') });
}
console.log('All files are ready.');
