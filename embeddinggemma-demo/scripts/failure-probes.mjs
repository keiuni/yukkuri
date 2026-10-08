#!/usr/bin/env node
// Where does EmbeddingGemma pick the wrong candidate? Runs the failure probes
// (public/data/failure-probes.json) with an embedding model, or with Liquid AI's decision model d1-3B
// for comparison, and records every pick.
//
//   npm run failure-probes -- --model v2 --dtype fp32,q8,q4
//   npm run failure-probes -- --model v1 --dtype fp32 --prompts all    # also try the other prompts
//   npm run failure-probes -- --decider d1-3b                          # needs llama-server (agent/llama.mjs)
//
// The embedding models get the prompts the browser agent uses (`task: search result | query: ` for the
// step, `title: none | text: ` for the candidates) unless --prompts says otherwise. Besides the plain
// run, an embedding run also scores the probes with the vectors cut to 512/256/128 dimensions, with
// unrelated page elements added to every set, and against queries that have no right answer.
// Results go to test-output/failure-probes/<run>.json; scripts/failure-report.mjs compares them.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { EmbeddingGemma } from '../agent/gemma.mjs';
import { DecisionModel, LIQUID_MODELS } from '../agent/llama.mjs';
import { percentile } from '../public/lib/metrics.js';
import { MODELS, MRL_DIMS, QUERY_TASKS, documentPrompt, queryPrompt } from '../public/lib/model-config.js';
import { escalation, judgeQuery, overlapPick, rocAuc, summarizeBy, summarizeRows } from '../public/lib/probe-metrics.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'test-output', 'failure-probes');
const probes = JSON.parse(await readFile(path.join(ROOT, 'public', 'data', 'failure-probes.json'), 'utf8'));
const categoryOrder = probes.categories.map((category) => category.id);

const PROMPTS = {
  search: { label: '検索（エージェントと同じ）', query: queryPrompt('search result'), doc: documentPrompt(null) },
  none: { label: 'プロンプトなし', query: '', doc: '' },
  similarity: { label: '文の類似度（両側）', query: QUERY_TASKS['sentence similarity'], doc: QUERY_TASKS['sentence similarity'] },
  qa: { label: '質問応答', query: QUERY_TASKS['question answering'], doc: documentPrompt(null) },
};
// The browser agent's "no candidate is good enough" cut-off (agent/run.mjs).
const AGENT_MIN_SIMILARITY = 0.3;

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : fallback;
};
const pct = (x) => `${(x * 100).toFixed(1)}%`;

/** A fixed pseudo-random order (mulberry32), so that every run adds the same unrelated elements. */
function seededShuffle(items, seed) {
  const out = [...items];
  let state = seed >>> 0;
  const next = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32;
  };
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/** One row per probe query, from `scoreSet(set, queryText) -> number[]` (one score per candidate). */
async function scoreProbes(scoreSet) {
  const rows = [];
  for (const set of probes.sets) {
    for (const query of set.queries) {
      const scores = await scoreSet(set, query.text, set.candidates);
      rows.push({
        set: set.id,
        category: set.category,
        style: set.style,
        query: query.text,
        answer: query.answer,
        candidates: set.candidates.length,
        scores,
        overlapPick: overlapPick(query.text, set.candidates),
        ...judgeQuery(scores, query.answer),
      });
    }
  }
  return rows;
}

function describeRun(rows) {
  const overall = summarizeRows(rows);
  const errors = rows.filter((row) => !row.correct);
  const lexical = errors.filter((row) => row.overlapPick >= 0);
  return {
    overall,
    byCategory: summarizeBy(rows, 'category', categoryOrder),
    byStyle: summarizeBy(rows, 'style', ['ui', 'doc']),
    // Of the errors where a pure character match has a clear favourite, how many follow it.
    errorsFollowingOverlap: { errors: lexical.length, followed: lexical.filter((row) => row.predicted === row.overlapPick).length },
    confidenceAuc: rocAuc(
      rows.filter((row) => row.correct).map((row) => row.confidence),
      errors.map((row) => row.confidence),
    ),
    escalation: escalation(rows),
  };
}

function printRun(title, result) {
  console.log(`\n${title}`);
  console.log(`  overall ${pct(result.overall.accuracy)} (${result.overall.correct}/${result.overall.queries}), sets passed ${result.overall.setsPassed}/${result.overall.sets}, blind sets ${result.overall.blindSets}/${result.overall.pairedSets}`);
  for (const [category, s] of Object.entries(result.byCategory)) {
    console.log(`  ${category.padEnd(17)} ${pct(s.accuracy).padStart(6)} (${s.correct}/${s.queries}, chance ${pct(s.chance)})  sets ${s.setsPassed}/${s.sets}  blind ${s.blindSets}/${s.pairedSets}  margin p50 ${s.medianMargin.toFixed(3)}`);
  }
}

async function runEmbedding(modelKey, dtype, promptKeys) {
  const embedder = await EmbeddingGemma.load(modelKey, dtype);
  const runs = [];
  for (const promptKey of promptKeys) {
    const prompt = PROMPTS[promptKey];
    const queryTexts = [...probes.sets.flatMap((set) => set.queries.map((query) => query.text)), ...probes.unanswerable.map((item) => item.text)];
    const docTexts = [...new Set([...probes.sets.flatMap((set) => set.candidates), ...probes.fillers.ui, ...probes.fillers.doc])];
    const started = performance.now();
    const vectors = await embedder.embed([...queryTexts.map((text) => prompt.query + text), ...docTexts.map((text) => prompt.doc + text)]);
    const embedMs = performance.now() - started;
    const byText = new Map();
    queryTexts.forEach((text, i) => byText.set(`q:${text}`, vectors[i]));
    docTexts.forEach((text, i) => byText.set(`d:${text}`, vectors[queryTexts.length + i]));

    const similarity = (a, b, dims) => {
      let dot = 0;
      let na = 0;
      let nb = 0;
      for (let i = 0; i < dims; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
      }
      return dot / Math.sqrt(na * nb);
    };
    const scorer = (dims) => async (set, text, candidates) => candidates.map((candidate) => similarity(byText.get(`q:${text}`), byText.get(`d:${candidate}`), dims));

    const rows = await scoreProbes(scorer(768));
    const result = describeRun(rows);

    // Matryoshka: the same picks with shorter vectors.
    const dims = {};
    for (const d of MRL_DIMS) dims[d] = summarizeRows(await scoreProbes(scorer(d)));

    // Unrelated page elements (or sentences) added to every set.
    const crowded = {};
    for (const extra of [10, 'all']) {
      const scoreWithFillers = async (set, text, candidates) => {
        const pool = probes.fillers[set.style];
        const fillers = extra === 'all' ? pool : seededShuffle(pool, set.id.length * 7919 + text.length).slice(0, extra);
        return scorer(768)(set, text, [...candidates, ...fillers]);
      };
      const crowdedRows = [];
      for (const set of probes.sets) {
        for (const query of set.queries) {
          const scores = await scoreWithFillers(set, query.text, set.candidates);
          crowdedRows.push({ set: set.id, category: set.category, answer: query.answer, candidates: scores.length, ...judgeQuery(scores, query.answer) });
        }
      }
      crowded[extra] = summarizeRows(crowdedRows);
    }

    // Queries with no right answer: does the best score give that away?
    const unanswerable = probes.unanswerable.map((item) => {
      const set = probes.sets.find((s) => s.id === item.set);
      const scores = set.candidates.map((candidate) => similarity(byText.get(`q:${item.text}`), byText.get(`d:${candidate}`), 768));
      return { set: item.set, query: item.text, scores, max: Math.max(...scores) };
    });
    const answerableBest = rows.map((row) => Math.max(...row.scores));
    const noAnswer = {
      auc: rocAuc(answerableBest, unanswerable.map((item) => item.max)),
      // Within the same set: below every right answer's score?
      belowSetMinimum: unanswerable.filter((item) => item.max < Math.min(...rows.filter((row) => row.set === item.set).map((row) => row.scores[row.answer]))).length,
      agentCutoff: {
        value: AGENT_MIN_SIMILARITY,
        unanswerableBlocked: unanswerable.filter((item) => item.max < AGENT_MIN_SIMILARITY).length,
        answerableBlocked: rows.filter((row) => row.scores[row.predicted] < AGENT_MIN_SIMILARITY).length,
      },
      answerableBestMedian: percentile(answerableBest, 50),
      unanswerableMaxMedian: percentile(unanswerable.map((item) => item.max), 50),
      items: unanswerable,
    };

    const run = {
      createdAt: new Date().toISOString(),
      kind: 'embedding',
      model: { key: modelKey, name: MODELS[modelKey].name, id: MODELS[modelKey].id, revision: MODELS[modelKey].revision, dtype, device: 'cpu (Node)' },
      prompt: { key: promptKey, ...prompt },
      embedMs,
      texts: vectors.length,
      ...result,
      dims,
      crowded,
      noAnswer,
      rows,
    };
    const name = `${modelKey}-${dtype}${promptKey === 'search' ? '' : `-prompt-${promptKey}`}`;
    await writeFile(path.join(OUT, `${name}.json`), `${JSON.stringify(run, null, 2)}\n`);
    printRun(`${MODELS[modelKey].name} ${dtype} / ${prompt.label}（${(embedMs / 1000).toFixed(0)} s for ${vectors.length} texts）`, run);
    console.log(`  dims: ${MRL_DIMS.map((d) => `${d} ${pct(dims[d].accuracy)}`).join('  ')}`);
    console.log(`  + unrelated elements: 10 → ${pct(crowded[10].accuracy)}, all → ${pct(crowded.all.accuracy)}`);
    console.log(`  no answer: AUC ${noAnswer.auc.toFixed(3)}, below the set's lowest right answer ${noAnswer.belowSetMinimum}/${unanswerable.length}, agent cut-off blocks ${noAnswer.agentCutoff.unanswerableBlocked}/${unanswerable.length} (and ${noAnswer.agentCutoff.answerableBlocked}/${rows.length} answerable)`);
    console.log(`  errors following the character match: ${run.errorsFollowingOverlap.followed}/${run.errorsFollowingOverlap.errors}; confidence AUC ${run.confidenceAuc.toFixed(3)}`);
    runs.push(run);
  }
  await embedder.model.dispose();
  return runs;
}

// d1 gets the browser agent's question for UI steps and a search question for sentences.
const D1_QUESTIONS = {
  ui: { state: '手順: ', instructions: 'この手順を実行するには、どの操作をすればよいですか？' },
  doc: { state: '質問: ', instructions: 'この質問に最もよく合う文はどれですか？' },
};

async function runDecision(key) {
  const model = await DecisionModel.load(key, 8091);
  try {
    let done = 0;
    const total = probes.sets.reduce((sum, set) => sum + set.queries.length, 0) + probes.unanswerable.length;
    const started = performance.now();
    const ask = async (set, text, candidates) => {
      const question = D1_QUESTIONS[set.style];
      const { probs } = await model.choose(`${question.state}${text}`, question.instructions, candidates);
      if (++done % 20 === 0) console.log(`  ${done}/${total} (${((performance.now() - started) / done / 1000).toFixed(1)} s each)`);
      return probs;
    };
    const rows = await scoreProbes(ask);
    const unanswerable = [];
    for (const item of probes.unanswerable) {
      const set = probes.sets.find((s) => s.id === item.set);
      const scores = await ask(set, item.text, set.candidates);
      unanswerable.push({ set: item.set, query: item.text, scores, max: Math.max(...scores) });
    }
    const run = {
      createdAt: new Date().toISOString(),
      kind: 'decision',
      model: { key, name: LIQUID_MODELS[key].name, id: `${LIQUID_MODELS[key].repo}/${LIQUID_MODELS[key].file}`, revision: LIQUID_MODELS[key].revision, device: 'cpu (llama.cpp)' },
      prompt: { key: 'd1', ...D1_QUESTIONS },
      msPerQuery: (performance.now() - started) / total,
      ...describeRun(rows),
      noAnswer: { auc: rocAuc(rows.map((row) => Math.max(...row.scores)), unanswerable.map((item) => item.max)), items: unanswerable },
      rows,
    };
    await writeFile(path.join(OUT, `${key}.json`), `${JSON.stringify(run, null, 2)}\n`);
    printRun(`${LIQUID_MODELS[key].name}（${(run.msPerQuery / 1000).toFixed(1)} s per query）`, run);
    console.log(`  no answer: AUC of the top probability ${run.noAnswer.auc.toFixed(3)}`);
  } finally {
    model.stop();
  }
}

await mkdir(OUT, { recursive: true });
const decider = option('decider', null);
if (decider) {
  if (LIQUID_MODELS[decider]?.kind !== 'decision') {
    console.error(`Unknown decision model "${decider}". Choose from: ${Object.keys(LIQUID_MODELS).filter((key) => LIQUID_MODELS[key].kind === 'decision').join(', ')}`);
    process.exit(1);
  }
  await runDecision(decider);
} else {
  const modelKey = option('model', 'v2');
  const dtypes = option('dtype', 'fp32').split(',');
  const promptOption = option('prompts', 'search');
  const promptKeys = promptOption === 'all' ? Object.keys(PROMPTS) : promptOption.split(',');
  for (const dtype of dtypes) {
    if (!MODELS[modelKey]?.dtypes[dtype]) {
      console.error(`Unknown model or dtype: ${modelKey} ${dtype}`);
      process.exit(1);
    }
  }
  if (promptKeys.some((key) => !PROMPTS[key])) {
    console.error(`Unknown prompt. Choose from: ${Object.keys(PROMPTS).join(', ')}, all`);
    process.exit(1);
  }
  for (const dtype of dtypes) await runEmbedding(modelKey, dtype, promptKeys);
}
