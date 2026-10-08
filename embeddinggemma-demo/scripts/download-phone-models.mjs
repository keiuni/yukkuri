#!/usr/bin/env node
// Downloads the models of the phone check page into ./models, so that `npm start` serves them from
// localhost and scripts/phone-check.mjs does not fetch them from the Hub on every run.
//
//   npm run download-phone-models                         # the variants the phone check runs by default
//   npm run download-phone-models -- gemma-3-1b:q4:wasm   # specific variants (model:dtype:device)
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { PHONE_MODELS, findVariant, variantKey } from '../public/lib/phone-models.js';
import { downloadFile } from './lib/hub.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const BASE_FILES = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json'];
const OPTIONAL_FILES = ['chat_template.jinja', 'special_tokens_map.json'];
const SUFFIX = { fp32: '', fp16: '_fp16', q8: '_quantized', q4: '_q4', q4f16: '_q4f16' };
// f16 variants need a GPU with shader-f16, so they are skipped unless asked for by name.
const DEFAULT_KEYS = PHONE_MODELS.flatMap((model) =>
  model.variants.filter((variant) => !variant.f16).map((variant) => variantKey(model, variant)),
);

const keys = process.argv.slice(2).filter((arg) => !arg.startsWith('-'));
const force = process.argv.includes('--force');
const wanted = (keys.length ? keys : DEFAULT_KEYS).map((key) => {
  const found = findVariant(key);
  if (!found) {
    console.error(`Unknown variant "${key}". Choose from: ${PHONE_MODELS.flatMap((m) => m.variants.map((v) => variantKey(m, v))).join(', ')}`);
    process.exit(1);
  }
  return found;
});

const seen = new Set();
for (const { model, variant } of wanted) {
  // WebGPU and WASM runs of one dtype read the same files.
  if (seen.has(`${model.key}:${variant.dtype}`)) continue;
  seen.add(`${model.key}:${variant.dtype}`);
  const targetDir = path.join(ROOT, 'models', model.id);
  const graphs = model.sessions.map((session) => `onnx/${variant.fileName ?? session}${SUFFIX[variant.dtype]}.onnx`);
  console.log(`Downloading ${model.id}@${model.revision.slice(0, 7)} [${variant.dtype}] -> ${path.relative(ROOT, targetDir)}`);
  for (const file of [...BASE_FILES, ...graphs.flatMap((graph) => [graph, `${graph}_data`])]) {
    await downloadFile({ modelId: model.id, revision: model.revision, file, targetDir, force });
  }
  for (const file of OPTIONAL_FILES) {
    await downloadFile({ modelId: model.id, revision: model.revision, file, targetDir, force }).catch((error) => {
      if (!/\b404\b/.test(error.message)) throw error;
    });
  }
}
console.log('All files are ready.');
