#!/usr/bin/env node
// Summarizes the JSON files written by the E2E run (test-output/e2e-<dtype>.json)
// into test-output/REPORT.md.
import { readdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { DTYPES, MODEL_REVISION, MRL_DIMS } from '../public/lib/model-config.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUTPUT_DIR = path.join(ROOT, 'test-output');
const ORDER = Object.keys(DTYPES);
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
const records = (await Promise.all(files.map(async (file) => JSON.parse(await readFile(path.join(OUTPUT_DIR, file), 'utf8')))))
  .filter((record) => record.model && record.benchmark)
  .sort((a, b) => ORDER.indexOf(a.dtype) - ORDER.indexOf(b.dtype));

const pct = (x) => `${(x * 100).toFixed(1)}%`;
const num = (x, digits = 3) => (Number.isFinite(x) ? x.toFixed(digits) : '–');
const ms = (x) => (x >= 1000 ? `${(x / 1000).toFixed(1)} s` : `${Math.round(x)} ms`);
const table = (headers, rows) =>
  [`| ${headers.join(' | ')} |`, `| ${headers.map((_, i) => (i === 0 ? '---' : '---:')).join(' | ')} |`, ...rows.map((row) => `| ${row.join(' | ')} |`)].join('\n');
const evaluation = (record, mode, dims) => record.benchmark.evaluations.find((e) => e.mode === mode && e.dims === dims);

const first = records[0];
const chrome = first.userAgent.match(/(?:HeadlessChrome|Chrome)\/([\d.]+)/)?.[1] ?? 'unknown';
const lines = [
  '# EmbeddingGemma ブラウザ検証レポート',
  '',
  `- 実行日時: ${first.benchmark.createdAt}`,
  `- ブラウザ: Chromium ${chrome}（Playwright / headless）、CPU論理コア ${first.benchmark.environment.hardwareConcurrency}、WASMスレッド ${first.model.numThreads ?? '既定'}、crossOriginIsolated=${first.model.crossOriginIsolated}`,
  `- モデル: ${first.model.modelId}@${MODEL_REVISION.slice(0, 7)}、Transformers.js ${first.model.transformersVersion}、実行環境 ${first.model.device}`,
  `- データセット: ${first.benchmark.dataset.name} v${first.benchmark.dataset.version}（文書${first.benchmark.dataset.docs}件・クエリ${first.benchmark.dataset.queries}件）`,
  '',
  '## まとめ（768次元・プロンプトあり）',
  '',
  table(
    ['精度', 'モデル', '読み込み', '初回推論', 'Top-1', 'MRR@10', 'nDCG@10', '文書 ms/件（バッチ8）', 'クエリ p50 / p95'],
    records.map((record) => {
      const main = evaluation(record, 'prompt', 768).overall;
      const { documents, queries } = record.benchmark.speed;
      return [
        record.dtype,
        `${(record.model.downloadedBytes / 1e6).toFixed(0)} MB`,
        ms(record.model.loadMs),
        ms(record.model.warmupMs),
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
    ['精度', 'プロンプト', ...MRL_DIMS.map((dims) => `${dims}次元`)],
    records.flatMap((record) =>
      ['prompt', 'raw'].map((mode) => [
        record.dtype,
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
    ['精度', ...TYPES.map(([type, label]) => `${label}（${evaluation(first, 'prompt', 768).byType[type]?.count ?? 0}問）`)],
    records.map((record) => [record.dtype, ...TYPES.map(([type]) => pct(evaluation(record, 'prompt', 768).byType[type]?.acc1))]),
  ),
  '',
  '## モデルカードの例（Which planet is known as the Red Planet?）',
  '',
  table(
    ['', 'Venus', 'Mars', 'Jupiter', 'Saturn', '最大誤差'],
    [
      ['モデルカード（fp32）', '0.301', '0.636', '0.493', '0.489', '–'],
      ...records.map((record) => [record.dtype, ...record.modelCard.scores.map((score) => num(score)), num(record.modelCard.maxDiff, 4)]),
    ],
  ),
  '',
  '## 類似度マトリクス（日英の言い換え8文）',
  '',
  table(
    ['精度', '同じ意味の組の最小値', '違う意味の組の最大値', '差'],
    records.map((record) => [
      record.dtype,
      num(record.similarity.minSameMeaning),
      num(record.similarity.maxDifferentMeaning),
      num(record.similarity.minSameMeaning - record.similarity.maxDifferentMeaning),
    ]),
  ),
  '',
  '## サンプル検索で想定解が何位だったか',
  '',
  table(
    ['精度', 'ECサイトFAQ（日本語）', '猫と玉ねぎ（日→英）', 'アニメ制作の工程（日本語）'],
    records.map((record) => [
      record.dtype,
      ...['faq-ja', 'cross-lingual', 'anime-ja'].map((preset) => `${record.presets[preset].expectedRank}位`),
    ]),
  ),
  '',
  '## 1位を外したクエリ（768次元・プロンプトあり）',
  '',
];

for (const record of records) {
  const misses = record.benchmark.perQuery.filter((row) => row.rank !== 1);
  lines.push(`### ${record.dtype}（${misses.length}件）`, '');
  if (misses.length === 0) lines.push('- なし');
  for (const row of misses) {
    lines.push(`- 「${row.text}」[${row.type}] 正解 \`${row.relevant[0]}\` は${Number.isFinite(row.rank) ? `${row.rank}位` : '圏外'}（${num(row.relevantScore)}）、1位は \`${row.top[0].id}\`（${num(row.top[0].score)}）`);
  }
  lines.push('');
}

const report = `${lines.join('\n').trimEnd()}\n`;
await writeFile(path.join(OUTPUT_DIR, 'REPORT.md'), report);
console.log(report);
