#!/usr/bin/env node
// Markdown tables from the failure-probe runs (test-output/failure-probes/*.json, scripts/failure-probes.mjs)
// and the long-text runs (test-output/long-text/*.json, scripts/long-text.mjs), for the report in docs/.
//
//   node scripts/failure-report.mjs > test-output/failure-probes/report.md
import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { charOverlap, rocAuc } from '../public/lib/probe-metrics.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIR = path.join(ROOT, 'test-output', 'failure-probes');
const probes = JSON.parse(await readFile(path.join(ROOT, 'public', 'data', 'failure-probes.json'), 'utf8'));
const load = async (dir, name) => JSON.parse(await readFile(path.join(dir, name), 'utf8'));
const files = (await readdir(DIR)).filter((name) => name.endsWith('.json'));
const runs = Object.fromEntries(await Promise.all(files.map(async (name) => [name.replace(/\.json$/, ''), await load(DIR, name)])));

const pct = (x) => (Number.isFinite(x) ? `${(x * 100).toFixed(0)}%` : '–');
const label = (key) => {
  const run = runs[key];
  if (!run) return key;
  return run.kind === 'decision' ? run.model.name : `${run.model.key === 'v1' ? '初代' : '2'} ${run.model.dtype}`;
};
const table = (header, rows) => [`| ${header.join(' | ')} |`, `| ${header.map((_, i) => (i === 0 ? '---' : '---:')).join(' | ')} |`, ...rows.map((row) => `| ${row.join(' | ')} |`)].join('\n');
const out = [];

// A baseline without any model: pick the candidate that shares the most character bigrams with the query
// (a tie for first counts as a guess among the tied candidates).
const overlapAccuracy = (setFilter) => {
  const scores = probes.sets.filter(setFilter).flatMap((set) =>
    set.queries.map((query) => {
      const overlaps = set.candidates.map((candidate) => charOverlap(query.text, candidate));
      const best = Math.max(...overlaps);
      const tied = overlaps.map((value, i) => (value === best ? i : -1)).filter((i) => i >= 0);
      return tied.includes(query.answer) ? 1 / tied.length : 0;
    }),
  );
  return scores.reduce((a, b) => a + b, 0) / scores.length;
};

// 1. Accuracy per category.
const main = ['v2-fp32', 'v2-q4', 'v1-fp32', 'v1-q8', 'd1-3b'].filter((key) => runs[key]);
out.push('## 種類ごとの正答率（エージェントと同じプロンプト）\n');
out.push(
  table(
    ['種類（問題数）', '偶然', '文字の重なりだけ', ...main.map(label), '2 fp32 の「違いを無視」した組'],
    [
      ...probes.categories.map((category) => {
        const ref = runs['v2-fp32'].byCategory[category.id];
        return [
          `${category.label}（${ref.queries}）`,
          pct(ref.chance),
          pct(overlapAccuracy((set) => set.category === category.id)),
          ...main.map((key) => pct(runs[key].byCategory[category.id].accuracy)),
          ref.pairedSets ? `${ref.blindSets}/${ref.pairedSets}` : '–',
        ];
      }),
      ['**全体**（219）', pct(runs['v2-fp32'].overall.chance), pct(overlapAccuracy(() => true)), ...main.map((key) => `**${pct(runs[key].overall.accuracy)}**`), `${runs['v2-fp32'].overall.blindSets}/${runs['v2-fp32'].overall.pairedSets}`],
    ],
  ),
);

// 2. Prompts.
const promptKeys = ['search', 'qa', 'similarity', 'none'];
out.push('\n## プロンプト別の全体正答率\n');
out.push(
  table(
    ['モデル', ...promptKeys.map((key) => runs[`v2-fp32${key === 'search' ? '' : `-prompt-${key}`}`]?.prompt.label ?? key)],
    ['v2', 'v1'].map((model) => [model === 'v1' ? '初代 fp32' : '2 fp32', ...promptKeys.map((key) => pct(runs[`${model}-fp32${key === 'search' ? '' : `-prompt-${key}`}`]?.overall.accuracy))]),
  ),
);
const promptCats = ['negation', 'antonym', 'role', 'relative-date', 'intent', 'lexical-trap'];
out.push('\n' + table(['2 fp32 のプロンプト', ...promptCats.map((id) => probes.categories.find((c) => c.id === id).label)], promptKeys.map((key) => {
  const run = runs[`v2-fp32${key === 'search' ? '' : `-prompt-${key}`}`];
  return [run?.prompt.label ?? key, ...promptCats.map((id) => pct(run?.byCategory[id].accuracy))];
})));

// 3. Matryoshka dims and crowded sets.
const emb = ['v2-fp32', 'v2-q8', 'v2-q4', 'v1-fp32', 'v1-q8', 'v1-q4'].filter((key) => runs[key]);
out.push('\n## 次元・候補の数\n');
out.push(
  table(
    ['モデル', '768 次元', '512', '256', '128', '無関係な候補 +10', '無関係な候補 +全部（UI 51 / 文 36）'],
    emb.map((key) => [label(key), ...[768, 512, 256, 128].map((d) => pct(runs[key].dims[d].accuracy)), pct(runs[key].crowded[10].accuracy), pct(runs[key].crowded.all.accuracy)]),
  ),
);

// 4. Unanswerable queries.
/** The cut-off that best separates answerable from unanswerable queries (Youden's J). */
function bestCutoff(answerable, unanswerable) {
  let best = { j: -1 };
  for (const t of [...answerable, ...unanswerable].sort((a, b) => a - b)) {
    const keep = answerable.filter((s) => s >= t).length / answerable.length;
    const block = unanswerable.filter((s) => s < t).length / unanswerable.length;
    if (keep + block - 1 > best.j) best = { j: keep + block - 1, t, keep, block };
  }
  return best;
}
out.push('\n## 「該当なし」を見分けられるか（18 問）\n');
out.push(
  table(
    ['モデル', '1 位の類似度（正解あり, 中央値）', '1 位の類似度（該当なし, 中央値）', 'AUC', 'しきい値 0.3 で止まる（該当なし / 正解あり）', '最良のしきい値', 'そのとき止まる（該当なし / 正解あり）'],
    emb.concat(runs['d1-3b'] ? ['d1-3b'] : []).map((key) => {
      const run = runs[key];
      const answerable = run.rows.map((row) => Math.max(...row.scores));
      const unanswerable = run.noAnswer.items.map((item) => item.max);
      const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
      const cut = bestCutoff(answerable, unanswerable);
      return [
        label(key),
        median(answerable).toFixed(3),
        median(unanswerable).toFixed(3),
        rocAuc(answerable, unanswerable).toFixed(2),
        run.kind === 'decision' ? '–' : `${run.noAnswer.agentCutoff.unanswerableBlocked}/18 / ${run.noAnswer.agentCutoff.answerableBlocked}/219`,
        cut.t.toFixed(3),
        `${pct(cut.block)} / ${pct(1 - cut.keep)}`,
      ];
    }),
  ),
);

// 5. Escalating the least confident picks.
out.push('\n## 自信のない判断だけ強いモデルに回したら（1 位と 2 位の差が小さい順）\n');
out.push(
  table(
    ['モデル', '誤りの数', '自信と正誤の AUC', ...[0.1, 0.2, 0.3, 0.5].map((share) => `${share * 100}% を回す`)],
    main.map((key) => {
      const run = runs[key];
      return [label(key), String(run.overall.queries - run.overall.correct), run.confidenceAuc.toFixed(2), ...run.escalation.map((e) => `誤りの ${pct(e.caught)}（正答率 ${pct(e.accuracyAfter)}）`)];
    }),
  ),
);

// 6. Errors that follow the character overlap.
out.push('\n## 文字の重なりに引きずられた誤り\n');
out.push(
  table(
    ['モデル', '誤り', '文字の重なりで 1 位が決まる誤り', 'うち、その 1 位を選んだ'],
    main.map((key) => {
      const run = runs[key];
      return [label(key), String(run.overall.queries - run.overall.correct), String(run.errorsFollowingOverlap.errors), `${run.errorsFollowingOverlap.followed}（${pct(run.errorsFollowingOverlap.followed / run.errorsFollowingOverlap.errors)}）`];
    }),
  ),
);

// 7. Where the two generations differ.
if (runs['v1-fp32']) {
  const v1 = new Map(runs['v1-fp32'].rows.map((row) => [`${row.set}|${row.query}`, row]));
  const v2Only = runs['v2-fp32'].rows.filter((row) => !row.correct && v1.get(`${row.set}|${row.query}`).correct);
  const v1Only = runs['v2-fp32'].rows.filter((row) => row.correct && !v1.get(`${row.set}|${row.query}`).correct);
  const both = runs['v2-fp32'].rows.filter((row) => !row.correct && !v1.get(`${row.set}|${row.query}`).correct);
  out.push(`\n## 初代と 2 の違い\n\n両方が外した ${both.length} 問、2 だけが外した ${v2Only.length} 問、初代だけが外した ${v1Only.length} 問。\n`);
  const byCat = (rows) => Object.entries(rows.reduce((acc, row) => ({ ...acc, [row.category]: (acc[row.category] ?? 0) + 1 }), {})).map(([id, n]) => `${probes.categories.find((c) => c.id === id).label} ${n}`).join('、');
  out.push(`- 2 だけ: ${byCat(v2Only)}\n- 初代だけ: ${byCat(v1Only)}\n- 両方: ${byCat(both)}`);
}

// 8. Long documents.
const longDir = path.join(ROOT, 'test-output', 'long-text');
const longFiles = (await readdir(longDir).catch(() => [])).filter((name) => name.endsWith('.json'));
if (longFiles.length) {
  const longRuns = await Promise.all(longFiles.map((name) => load(longDir, name)));
  out.push('\n## 長い文書（正解の文書を無関係な文書の中に埋めたとき、ミニベンチマーク 56 問の Top-1）\n');
  const cells = longRuns[0].cells.map((cell) => `${cell.k}|${cell.position}`);
  out.push(
    table(
      ['足した文書（位置）', 'トークン数（平均）', ...longRuns.map((run) => `${run.model.key === 'v1' ? '初代' : '2'} ${run.model.dtype}`)],
      cells.map((id) => {
        const [k, position] = id.split('|');
        const first = longRuns[0].cells.find((cell) => `${cell.k}|${cell.position}` === id);
        const where = { start: '先頭', middle: '中央', end: '末尾' }[position];
        return [k === '0' ? 'なし' : `${k} 件（正解は${where}）`, first.tokens.toFixed(0), ...longRuns.map((run) => pct(run.cells.find((cell) => `${cell.k}|${cell.position}` === id).acc1))];
      }),
    ),
  );
}

console.log(out.join('\n'));
