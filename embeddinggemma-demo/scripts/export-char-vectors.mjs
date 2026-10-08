#!/usr/bin/env node
// Writes public/data/char-vectors.json (char-vectors-v2.json for EmbeddingGemma 2) for the 文字生成
// tab: the embedding of every character in public/data/char-vocab.json, stored as int8 so the browser
// does not have to embed 2,000+ characters before it can start. The tab only uses them to shortlist
// kanji for a target.
//
//   node scripts/export-char-vectors.mjs [--model v2]
// All 768 dimensions are kept: cut to 128 (Matryoshka), only 7-9 of the full top 16 survived;
// with 768 in int8, 15-16 did.
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { EmbeddingGemma } from '../agent/gemma.mjs';
import { vocabChars } from '../public/lib/char-gen.js';
import { DEFAULT_MODEL, MODELS } from '../public/lib/model-config.js';
import { charVectorsFor } from './lib/char-vectors.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIM = 768;

const modelIndex = process.argv.indexOf('--model');
const model = modelIndex >= 0 ? process.argv[modelIndex + 1] : DEFAULT_MODEL;
if (!MODELS[model]) {
  console.error(`Unknown model "${model}". Choose from: ${Object.keys(MODELS).join(', ')}`);
  process.exit(1);
}
const { id, revision, charVectors } = MODELS[model];

const vocab = JSON.parse(await readFile(path.join(ROOT, 'public', 'data', 'char-vocab.json'), 'utf8'));
const chars = vocabChars(vocab);
// The model is only loaded when the vectors are not cached yet.
const lazyEmbedder = { embed: async (texts) => (await EmbeddingGemma.load(model)).embed(texts) };
const vectors = await charVectorsFor(lazyEmbedder, chars, revision);

const packed = new Int8Array(chars.length * DIM);
vectors.forEach((vector, i) => {
  const head = vector.subarray(0, DIM);
  const norm = Math.sqrt(head.reduce((sum, value) => sum + value * value, 0)) || 1;
  for (let d = 0; d < DIM; d++) packed[i * DIM + d] = Math.max(-127, Math.min(127, Math.round((head[d] / norm) * 127)));
});

const out = path.join(ROOT, 'public', charVectors);
await writeFile(
  out,
  `${JSON.stringify({ model: id, revision, prompt: 'title: none | text: ', dim: DIM, chars: chars.join(''), data: Buffer.from(packed.buffer).toString('base64') })}\n`,
);
console.log(`${chars.length} characters x ${DIM} dims -> ${path.relative(ROOT, out)}`);
