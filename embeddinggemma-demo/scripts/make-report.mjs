#!/usr/bin/env node
// Summarizes the JSON files written by the E2E runs (test-output/e2e-<model>-<dtype>-<device>.json)
// into test-output/REPORT.md, one row per model, precision and device.
import { readdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { MODELS, MRL_DIMS } from '../public/lib/model-config.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUTPUT_DIR = path.join(ROOT, 'test-output');
const ORDER = Object.entries(MODELS).flatMap(([model, { dtypes }]) =>
  Object.keys(dtypes).flatMap((dtype) => ['wasm', 'webgpu'].map((device) => `${model}:${dtype}:${device}`)),
);
const DEVICE_LABELS = { wasm: 'WASM', webgpu: 'WebGPU' };
const TYPES = [
  ['ja→ja', '日→日'],
  ['en→ja', '英→日'],
  ['ja→en', '日→英'],
  ['en→en', '英→英'],
];

const files = (await readdir(OUTPUT_DIR).catch(() => [])).filter((file) => /^e2e-.+\.json$/.test(file));
if (files.length === 0) {
  console.error('No test-output/e2e-*.json found. Run the E2E tests first: npm run test:e2e');
  process.exit(1);
}
const keyOf = (record) => `${record.model}:${record.dtype}:${record.device}`;
const records = (await Promise.all(files.map(async (file) => JSON.parse(await readFile(path.join(OUTPUT_DIR, file), 'utf8')))))
  .filter((record) => record.loaded && record.benchmark)
  .sort((a, b) => ORDER.indexOf(keyOf(a)) - ORDER.indexOf(keyOf(b)));
// "初代 fp32" / "2 q4（WebGPU）": the device is only named when it is not WASM.
const label = (record) =>
  `${record.model === 'v1' ? '初代' : MODELS[record.model].name.replace('EmbeddingGemma ', '')} ${record.dtype}${record.device === 'wasm' ? '' : `（${DEVICE_LABELS[record.device]}）`}`;

const pct = (x) => `${(x * 100).toFixed(1)}%`;
const num = (x, digits = 3) => (Number.isFinite(x) ? x.toFixed(digits) : '–');
const ms = (x) => (x >= 1000 ? `${(x / 1000).toFixed(1)} s` : `${Math.round(x)} ms`);
const table = (headers, rows) =>
  [`| ${headers.join(' | ')} |`, `| ${headers.map((_, i) => (i === 0 ? '---' : '---:')).join(' | ')} |`, ...rows.map((row) => `| ${row.join(' | ')} |`)].join('\n');
const evaluation = (record, mode, dims) => record.benchmark.evaluations.find((e) => e.mode === mode && e.dims === dims);

const first = records[0];
const chrome = first.userAgent.match(/(?:HeadlessChrome|Chrome)\/([\d.]+)/)?.[1] ?? 'unknown';
const wasmRecord = records.find((record) => record.device === 'wasm') ?? first;
const models = [...new Set(records.map((record) => record.model))];
const times = records.map((record) => record.benchmark.createdAt).sort();
const lines = [
  '# EmbeddingGemma ブラウザ検証レポート',
  '',
  `- 実行日時: ${times[0]} 〜 ${times.at(-1)}`,
  `- ブラウザ: Chromium ${chrome}（Playwright / headless）、CPU論理コア ${first.benchmark.environment.hardwareConcurrency}、WASMスレッド ${wasmRecord.loaded.numThreads ?? '既定'}、crossOriginIsolated=${first.loaded.crossOriginIsolated}`,
  `- モデル: ${models.map((model) => `${MODELS[model].id}@${MODELS[model].revision.slice(0, 7)}`).join('、')}、Transformers.js ${first.loaded.transformersVersion}`,
  `- データセット: ${first.benchmark.dataset.name} v${first.benchmark.dataset.version}（文書${first.benchmark.dataset.docs}件・クエリ${first.benchmark.dataset.queries}件）`,
  '',
  '## まとめ（768次元・プロンプトあり）',
  '',
  table(
    ['モデル・精度', 'ファイル', '読み込み', '初回推論', 'Top-1', 'MRR@10', 'nDCG@10', '文書 ms/件（バッチ8）', 'クエリ p50 / p95'],
    records.map((record) => {
      const main = evaluation(record, 'prompt', 768).overall;
      const { documents, queries } = record.benchmark.speed;
      return [
        label(record),
        `${(record.loaded.downloadedBytes / 1e6).toFixed(0)} MB`,
        ms(record.loaded.loadMs),
        ms(record.loaded.warmupMs),
        `${pct(main.acc1)}（${Math.round(main.acc1 * main.count)}/${main.count}）`,
        num(main.mrr10),
        num(main.ndcg10),
        `${Math.round(documents.msPerText)}`,
        `${Math.round(queries.p50Ms)} / ${Math.round(queries.p95Ms)} ms`,
      ];
    }),
  ),
  '',
  '## プロンプトと次元（MRL）: Top-1 / MRR@10',
  '',
  table(
    ['モデル・精度', 'プロンプト', ...MRL_DIMS.map((dims) => `${dims}次元`)],
    records.flatMap((record) =>
      ['prompt', 'raw'].map((mode) => [
        label(record),
        mode === 'prompt' ? 'あり' : 'なし',
        ...MRL_DIMS.map((dims) => {
          const { acc1, mrr10 } = evaluation(record, mode, dims).overall;
          return `${pct(acc1)} / ${num(mrr10)}`;
        }),
      ]),
    ),
  ),
  '',
  '## 言語ペア別 Top-1（768次元・プロンプトあり）',
  '',
  table(
    ['モデル・精度', ...TYPES.map(([type, name]) => `${name}（${evaluation(first, 'prompt', 768).byType[type]?.count ?? 0}問）`)],
    records.map((record) => [label(record), ...TYPES.map(([type]) => pct(evaluation(record, 'prompt', 768).byType[type]?.acc1))]),
  ),
  '',
  '## モデルカードの例（Which planet is known as the Red Planet?）',
  '',
  '最大誤差は、そのモデルのモデルカードに載っている値（初代は fp32、2 は q4）との差です。',
  '',
  table(
    ['', 'Venus', 'Mars', 'Jupiter', 'Saturn', '最大誤差'],
    models.flatMap((model) => {
      const own = records.filter((record) => record.model === model);
      const { reference } = own[0].modelCard;
      return [
        [`${model === 'v1' ? '初代' : '2'} モデルカード（${reference.dtype}）`, ...reference.scores.map((score) => num(score)), '–'],
        ...own.map((record) => [label(record), ...record.modelCard.scores.map((score) => num(score)), num(record.modelCard.maxDiff, 4)]),
      ];
    }),
  ),
  '',
  '## 類似度マトリクス（日英の言い換え8文）',
  '',
  table(
    ['モデル・精度', '同じ意味の組の最小値', '違う意味の組の最大値', '差'],
    records.map((record) => [
      label(record),
      num(record.similarity.minSameMeaning),
      num(record.similarity.maxDifferentMeaning),
      num(record.similarity.minSameMeaning - record.similarity.maxDifferentMeaning),
    ]),
  ),
  '',
  '## サンプル検索で想定解が何位だったか',
  '',
  table(
    ['モデル・精度', 'ECサイトFAQ（日本語）', '猫と玉ねぎ（日→英）', 'アニメ制作の工程（日本語）'],
    records.map((record) => [
      label(record),
      ...['faq-ja', 'cross-lingual', 'anime-ja'].map((preset) => `${record.presets[preset].expectedRank}位`),
    ]),
  ),
  '',
  '## 1位を外したクエリ（768次元・プロンプトあり）',
  '',
];

for (const record of records) {
  const misses = record.benchmark.perQuery.filter((row) => row.rank !== 1);
  lines.push(`### ${label(record)}（${misses.length}件）`, '');
  if (misses.length === 0) lines.push('- なし');
  for (const row of misses) {
    lines.push(`- 「${row.text}」[${row.type}] 正解 \`${row.relevant[0]}\` は${Number.isFinite(row.rank) ? `${row.rank}位` : '圏外'}（${num(row.relevantScore)}）、1位は \`${row.top[0].id}\`（${num(row.top[0].score)}）`);
  }
  lines.push('');
}

const report = `${lines.join('\n').trimEnd()}\n`;
await writeFile(path.join(OUTPUT_DIR, 'REPORT.md'), report);
console.log(report);
