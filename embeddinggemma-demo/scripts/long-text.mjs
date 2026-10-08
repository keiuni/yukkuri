#!/usr/bin/env node
// Does a long document lose to short ones? The relevant document of every mini-benchmark query
// (public/data/benchmark.json) is buried among k unrelated documents from other topics, at the start,
// in the middle or at the end, and must still rank first against the other, unchanged documents
// (which include the topic's hard negatives). EmbeddingGemma averages over all tokens, so the
// relevant part is diluted as the text grows.
//
//   npm run long-text -- --model v2 --dtype fp32
//   npm run long-text -- --model v1 --dtype fp32
//
// Results go to test-output/long-text/<model>-<dtype>.json.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { EmbeddingGemma } from '../agent/gemma.mjs';
import { MODELS, documentPrompt, queryPrompt } from '../public/lib/model-config.js';
import { mean } from '../public/lib/metrics.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'test-output', 'long-text');
// Unrelated documents added around the relevant one (each is one or two sentences, 40-50 tokens).
const PADDING = [0, 3, 10, 30];
const POSITIONS = ['start', 'middle', 'end'];

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : fallback;
};
const modelKey = option('model', 'v2');
const dtype = option('dtype', 'fp32');
if (!MODELS[modelKey]?.dtypes[dtype]) {
  console.error(`Unknown model or dtype: ${modelKey} ${dtype}`);
  process.exit(1);
}

const { docs, queries } = JSON.parse(await readFile(path.join(ROOT, 'public', 'data', 'benchmark.json'), 'utf8'));
const byId = new Map(docs.map((doc) => [doc.id, doc]));

/** The relevant document with `k` documents of other topics around it (fixed order, so runs compare). */
function bury(relevant, k, position) {
  const others = docs.filter((doc) => doc.topic !== relevant.topic);
  // Rotate by the document's index so that each query gets a different, but fixed, selection.
  const start = docs.indexOf(relevant) * 7;
  const fillers = Array.from({ length: k }, (_, i) => others[(start + i) % others.length].text);
  const at = position === 'start' ? 0 : position === 'end' ? k : Math.floor(k / 2);
  return [...fillers.slice(0, at), relevant.text, ...fillers.slice(at)].join('\n');
}

const embedder = await EmbeddingGemma.load(modelKey, dtype);
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const started = performance.now();
const queryVectors = await embedder.embed(queries.map((query) => queryPrompt('search result') + query.text));
const docVectors = new Map();
(await embedder.embed(docs.map((doc) => documentPrompt(null) + doc.text))).forEach((vector, i) => docVectors.set(docs[i].id, vector));

const cells = [];
for (const k of PADDING) {
  for (const position of k === 0 ? ['start'] : POSITIONS) {
    const longTexts = queries.map((query) => bury(byId.get(query.relevant[0]), k, position));
    const longVectors = await embedder.embed(longTexts.map((text) => documentPrompt(null) + text));
    const tokens = await Promise.all(longTexts.map(async (text) => (await embedder.tokenizer(documentPrompt(null) + text)).input_ids.dims.at(-1)));
    const rows = queries.map((query, qi) => {
      const relevantId = query.relevant[0];
      const relevantScore = dot(queryVectors[qi], longVectors[qi]);
      const others = docs.filter((doc) => doc.id !== relevantId).map((doc) => ({ id: doc.id, score: dot(queryVectors[qi], docVectors.get(doc.id)) }));
      const best = others.reduce((a, b) => (b.score > a.score ? b : a));
      return { id: query.id, relevantScore, rank: 1 + others.filter((doc) => doc.score > relevantScore).length, beatenBy: best.score > relevantScore ? best.id : null, bestOther: best.score };
    });
    const cell = {
      k,
      position,
      tokens: mean(tokens),
      maxTokens: Math.max(...tokens),
      acc1: mean(rows.map((row) => (row.rank === 1 ? 1 : 0))),
      meanRank: mean(rows.map((row) => row.rank)),
      relevantScore: mean(rows.map((row) => row.relevantScore)),
      rows,
    };
    cells.push(cell);
    console.log(`k=${String(k).padStart(2)} ${position.padEnd(6)} tokens≈${cell.tokens.toFixed(0).padStart(5)}  Top-1 ${(cell.acc1 * 100).toFixed(1)}%  mean rank ${cell.meanRank.toFixed(2)}  relevant sim ${cell.relevantScore.toFixed(3)}`);
  }
}

await mkdir(OUT, { recursive: true });
const result = {
  createdAt: new Date().toISOString(),
  model: { key: modelKey, name: MODELS[modelKey].name, id: MODELS[modelKey].id, revision: MODELS[modelKey].revision, dtype, device: 'cpu (Node)' },
  queries: queries.length,
  seconds: (performance.now() - started) / 1000,
  cells,
};
await writeFile(path.join(OUT, `${modelKey}-${dtype}.json`), `${JSON.stringify(result, null, 2)}\n`);
console.log(`done in ${result.seconds.toFixed(0)} s`);
await embedder.model.dispose();
