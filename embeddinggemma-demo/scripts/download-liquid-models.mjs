#!/usr/bin/env node
// Downloads the GGUF files of the Liquid AI models that agent/run.mjs serves with llama-server
// (d1-3B, d1-omni-600M, LFM2.5-2.6B) into ./models. About 5 GB in total.
//
//   npm run download-liquid-models               # all three
//   npm run download-liquid-models -- d1-3b lfm  # some of them
//
// llama-server itself comes from a llama.cpp build with the /v1/systemone endpoint (2026-10 or later):
//   git clone --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp
//   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target llama-server -j
// Then point LLAMA_SERVER at build/bin/llama-server (or put it on PATH).
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { LIQUID_MODELS } from '../agent/llama.mjs';
import { downloadFile } from './lib/hub.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const keys = process.argv.slice(2).filter((arg) => !arg.startsWith('-'));
for (const key of keys) {
  if (!LIQUID_MODELS[key]) {
    console.error(`Unknown model "${key}". Choose from: ${Object.keys(LIQUID_MODELS).join(', ')}`);
    process.exit(1);
  }
}

for (const key of keys.length ? keys : Object.keys(LIQUID_MODELS)) {
  const { repo, revision, file } = LIQUID_MODELS[key];
  console.log(`Downloading ${repo}@${revision.slice(0, 7)} ${file}`);
  await downloadFile({ modelId: repo, revision, file, targetDir: path.join(ROOT, 'models', repo), force: process.argv.includes('--force') });
}
console.log('All files are ready.');
