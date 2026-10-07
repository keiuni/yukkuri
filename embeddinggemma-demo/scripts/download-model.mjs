#!/usr/bin/env node
// Downloads the EmbeddingGemma ONNX files into ./models so that server.mjs can
// serve them from localhost. This is optional: without it the demo downloads the
// same files from the Hugging Face Hub and keeps them in the browser cache.
//
//   npm run download-model                 # q8 + q4 (about 530 MB)
//   npm run download-model -- fp32 q8 q4   # every variant used by the demo (about 1.8 GB)
//
// Behind an HTTP proxy, run it with NODE_USE_ENV_PROXY=1 (Node >= 22.21).
import { createWriteStream } from 'node:fs';
import { mkdir, rename, stat } from 'node:fs/promises';
import path from 'node:path';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';

import { DTYPES, MODEL_ID, MODEL_REVISION, onnxFile } from '../public/lib/model-config.js';

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

const formatMB = (bytes) => `${(bytes / 1e6).toFixed(1)} MB`;

async function exists(file) {
  try {
    return (await stat(file)).isFile();
  } catch {
    return false;
  }
}

async function download(file) {
  const destination = path.join(TARGET_DIR, file);
  if (!force && (await exists(destination))) {
    console.log(`skip  ${file} (already downloaded)`);
    return;
  }
  await mkdir(path.dirname(destination), { recursive: true });

  const url = `https://huggingface.co/${MODEL_ID}/resolve/${MODEL_REVISION}/${file}`;
  const response = await fetch(url);
  if (!response.ok || !response.body) {
    throw new Error(`GET ${url} failed: ${response.status} ${response.statusText}`);
  }

  const total = Number(response.headers.get('content-length')) || 0;
  let received = 0;
  let lastReport = 0;
  const progress = new Transform({
    transform(chunk, _encoding, callback) {
      received += chunk.length;
      if (total > 50e6 && received - lastReport > 100e6) {
        lastReport = received;
        console.log(`      ${file}: ${formatMB(received)} / ${formatMB(total)}`);
      }
      callback(null, chunk);
    },
  });

  // Write to a temporary name first so an interrupted download is never mistaken for a complete one.
  const partial = `${destination}.part`;
  await pipeline(Readable.fromWeb(response.body), progress, createWriteStream(partial));
  await rename(partial, destination);
  console.log(`done  ${file} (${formatMB(received)})`);
}

console.log(`Downloading ${MODEL_ID}@${MODEL_REVISION.slice(0, 7)} [${dtypes.join(', ')}] -> ${path.relative(ROOT, TARGET_DIR)}`);
for (const file of files) {
  await download(file);
}
console.log('All files are ready. Start the demo with: npm start');
