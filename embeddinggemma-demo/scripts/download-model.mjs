#!/usr/bin/env node
// Downloads the EmbeddingGemma ONNX files into ./models so that server.mjs can
// serve them from localhost. This is optional: without it the demo downloads the
// same files from the Hugging Face Hub and keeps them in the browser cache.
//
//   npm run download-model                 # q8 + q4 (about 530 MB)
//   npm run download-model -- fp32 q8 q4   # every variant used by the demo (about 1.8 GB)
//
// Behind an HTTP proxy, run it with NODE_USE_ENV_PROXY=1 (Node >= 22.21).
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { DTYPES, MODEL_ID, MODEL_REVISION, onnxFile } from '../public/lib/model-config.js';
import { downloadFile } from './lib/hub.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TARGET_DIR = path.join(ROOT, 'models', MODEL_ID);
const BASE_FILES = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'];

const requested = process.argv.slice(2).filter((arg) => !arg.startsWith('-'));
const force = process.argv.includes('--force');
const dtypes = requested.length > 0 ? requested : ['q8', 'q4'];

for (const dtype of dtypes) {
  if (!DTYPES[dtype]) {
    console.error(`Unknown dtype "${dtype}". Choose from: ${Object.keys(DTYPES).join(', ')}`);
    process.exit(1);
  }
}

const files = [
  ...BASE_FILES,
  ...dtypes.flatMap((dtype) => [onnxFile(dtype), `${onnxFile(dtype)}_data`]),
];

console.log(`Downloading ${MODEL_ID}@${MODEL_REVISION.slice(0, 7)} [${dtypes.join(', ')}] -> ${path.relative(ROOT, TARGET_DIR)}`);
for (const file of files) {
  await downloadFile({ modelId: MODEL_ID, revision: MODEL_REVISION, file, targetDir: TARGET_DIR, force });
}
console.log('All files are ready. Start the demo with: npm start');
