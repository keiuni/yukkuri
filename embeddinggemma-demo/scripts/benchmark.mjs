#!/usr/bin/env node
// The 「ミニベンチマーク」 tab's accuracy test in Node (onnxruntime, CPU): same data, prompts, metrics and
// Matryoshka dimensions. It is there for precisions the browser can only run on WebGPU (EmbeddingGemma 2
// q8 / q4), which a machine without a GPU cannot time or run in reasonable time. Speed numbers are left
// to the browser runs; here only accuracy is compared.
//
//   npm run benchmark -- --model v2 --dtype q8,q4
//   npm run benchmark -- --model v1 --dtype fp32,q8,q4
//
// Results go to test-output/benchmark-node-<model>-<dtype>.json.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { EmbeddingGemma } from '../agent/gemma.mjs';
import { evaluateRetrieval, truncateAndNormalize } from '../public/lib/metrics.js';
import { DEFAULT_MODEL, MODELS, MRL_DIMS, documentPrompt, queryPrompt } from '../public/lib/model-config.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'test-output');

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : fallback;
};
const model = option('model', DEFAULT_MODEL);
const dtypes = option('dtype', 'fp32').split(',');
for (const dtype of dtypes) {
  if (!MODELS[model]?.dtypes[dtype]) {
    console.error(`Unknown model or dtype: ${model} ${dtype}`);
    process.exit(1);
  }
}

const { docs, queries, name, version } = JSON.parse(await readFile(path.join(ROOT, 'public', 'data', 'benchmark.json'), 'utf8'));
const pct = (x) => `${(x * 100).toFixed(1)}%`;
await mkdir(OUT, { recursive: true });

for (const dtype of dtypes) {
  const embedder = await EmbeddingGemma.load(model, dtype);
  /** One flat Float32Array per text list, as the browser worker returns them. */
  const embed = async (texts) => {
    const vectors = await embedder.embed(texts);
    const flat = new Float32Array(vectors.length * vectors[0].length);
    vectors.forEach((vector, i) => flat.set(vector, i * vector.length));
    return { vectors: flat, dim: vectors[0].length };
  };
  const inputs = {
    prompt: [docs.map((doc) => documentPrompt(null) + doc.text), queries.map((query) => queryPrompt('search result') + query.text)],
    raw: [docs.map((doc) => doc.text), queries.map((query) => query.text)],
  };

  const evaluations = [];
  for (const [mode, [docTexts, queryTexts]] of Object.entries(inputs)) {
    const [docOut, queryOut] = [await embed(docTexts), await embed(queryTexts)];
    for (const dims of MRL_DIMS) {
      evaluations.push({
        mode,
        dims,
        ...evaluateRetrieval({
          queries,
          docs,
          queryVectors: truncateAndNormalize(queryOut.vectors, queryOut.dim, dims),
          docVectors: truncateAndNormalize(docOut.vectors, docOut.dim, dims),
          dims,
        }),
      });
    }
  }

  const result = {
    createdAt: new Date().toISOString(),
    model: { model, id: MODELS[model].id, revision: MODELS[model].revision, dtype, device: 'cpu (Node)' },
    dataset: { name, version, docs: docs.length, queries: queries.length },
    evaluations: evaluations.map(({ perQuery, ...summary }) => summary),
    perQuery: evaluations.find((evaluation) => evaluation.mode === 'prompt' && evaluation.dims === 768).perQuery,
  };
  await writeFile(path.join(OUT, `benchmark-node-${model}-${dtype}.json`), `${JSON.stringify(result, null, 2)}\n`);

  console.log(`${MODELS[model].name} ${dtype}`);
  for (const mode of ['prompt', 'raw']) {
    const row = MRL_DIMS.map((dims) => {
      const { acc1, mrr10 } = evaluations.find((e) => e.mode === mode && e.dims === dims).overall;
      return `${dims}: ${pct(acc1)} / ${mrr10.toFixed(3)}`;
    });
    console.log(`  ${mode === 'prompt' ? 'prompt' : 'raw   '}  ${row.join('  ')}`);
  }
  const misses = result.perQuery.filter((row) => row.rank !== 1);
  for (const row of misses) console.log(`  miss: ${row.text} → rank ${row.rank} (${row.relevantScore.toFixed(3)} vs ${row.top[0].id} ${row.top[0].score.toFixed(3)})`);
  await embedder.model.dispose();
}
