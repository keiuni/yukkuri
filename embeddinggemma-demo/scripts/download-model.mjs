#!/usr/bin/env node
// Downloads the EmbeddingGemma ONNX files into ./models so that server.mjs can
// serve them from localhost. This is optional: without it the demo downloads the
// same files from the Hugging Face Hub and keeps them in the browser cache.
//
//   npm run download-model                              # q8 + q4 (about 530 MB)
//   npm run download-model -- fp32 q8 q4                # every variant used by the demo (about 1.8 GB)
//   npm run download-model -- --model v2 fp32 q8 q4     # EmbeddingGemma 2, text model only (about 1.6 GB)
//
// Behind an HTTP proxy, run it with NODE_USE_ENV_PROXY=1 (Node >= 22.21).
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { DEFAULT_MODEL, MODELS, onnxFile } from '../public/lib/model-config.js';
import { downloadFile } from './lib/hub.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const args = process.argv.slice(2);
const modelIndex = args.indexOf('--model');
const model = modelIndex >= 0 ? args[modelIndex + 1] : DEFAULT_MODEL;
if (!MODELS[model]) {
  console.error(`Unknown model "${model}". Choose from: ${Object.keys(MODELS).join(', ')}`);
  process.exit(1);
}
const { id, revision, files: baseFiles, dtypes: known } = MODELS[model];
const targetDir = path.join(ROOT, 'models', id);

const requested = args.filter((arg, i) => !arg.startsWith('-') && !(modelIndex >= 0 && i === modelIndex + 1));
const force = args.includes('--force');
const dtypes = requested.length > 0 ? requested : ['q8', 'q4'];

for (const dtype of dtypes) {
  if (!known[dtype]) {
    console.error(`Unknown dtype "${dtype}". Choose from: ${Object.keys(known).join(', ')}`);
    process.exit(1);
  }
}

const files = [
  ...baseFiles,
  ...dtypes.flatMap((dtype) => [onnxFile(model, dtype), `${onnxFile(model, dtype)}_data`]),
];

console.log(`Downloading ${id}@${revision.slice(0, 7)} [${dtypes.join(', ')}] -> ${path.relative(ROOT, targetDir)}`);
for (const file of files) {
  await downloadFile({ modelId: id, revision, file, targetDir, force });
}
console.log('All files are ready. Start the demo with: npm start');
