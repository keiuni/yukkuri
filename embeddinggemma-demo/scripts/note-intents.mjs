#!/usr/bin/env node
// Can EmbeddingGemma sort a sentence typed into a notes app into 予定 / ToDo / メモ? Embeds the held-out sentences of
// public/data/note-intents.json and compares: the label descriptions alone (zero-shot), the average of a few example
// sentences per label (prototypes), and the nearest examples (k-nearest); with the model card's classification prompt, the
// search prompts, or none. Also: how accurate it is when the app asks the user whenever the top two labels are close, and
// how well the rules of public/lib/note-intent.js read the date and time out of the schedule sentences.
//
//   npm run note-intents -- --model v1 --dtype q8,fp32
//
// Results go to test-output/note-intents-<model>-<dtype>.json.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { EmbeddingGemma } from '../agent/gemma.mjs';
import { DEFAULT_MODEL, MODELS, documentPrompt, queryPrompt } from '../public/lib/model-config.js';
import { abstainCurve, centroid, classifyByNeighbors, classifyByPrototype, parseWhen, summarize } from '../public/lib/note-intent.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'test-output');
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : fallback;
};
const model = option('model', DEFAULT_MODEL);
const dtypes = option('dtype', 'q8').split(',');
const data = JSON.parse(await readFile(path.join(ROOT, 'public', 'data', 'note-intents.json'), 'utf8'));
const pct = (x) => `${(x * 100).toFixed(1)}%`;

// How the sentence and the label side are written for the model: [sentence prompt, label / example prompt].
const PROMPTS = {
  classification: [queryPrompt('classification'), queryPrompt('classification')],
  'search result': [queryPrompt('search result'), documentPrompt(null)],
  none: ['', ''],
};

await mkdir(OUT, { recursive: true });
for (const dtype of dtypes) {
  const embedder = await EmbeddingGemma.load(model, dtype);
  const report = { model, dtype, methods: {}, now: data.now };
  const timing = performance.now();
  await embedder.embed(data.test.map((t) => t.text)); // warm up and time the plain sentences
  report.msPerSentence = (performance.now() - timing) / data.test.length;

  for (const [promptName, [sentencePrompt, labelPrompt]] of Object.entries(PROMPTS)) {
    const sentences = await embedder.embed(data.test.map((t) => sentencePrompt + t.text));
    const seeds = await embedder.embed(data.seeds.map((s) => labelPrompt + s.text));
    const descriptions = await embedder.embed(data.labels.map((l) => labelPrompt + `${l.name}：${l.description}`));
    const seedExamples = data.seeds.map((s, i) => ({ label: s.label, vector: seeds[i] }));
    const byLabel = (label) => seedExamples.filter((e) => e.label === label).map((e) => e.vector);
    const methods = {
      'zero-shot (説明文)': (v) => classifyByPrototype(v, Object.fromEntries(data.labels.map((l, i) => [l.id, descriptions[i]]))),
      'prototype (例文 5 つの平均)': (v) => classifyByPrototype(v, Object.fromEntries(data.labels.map((l) => [l.id, centroid(byLabel(l.id))]))),
      'prototype (例文 + 説明文)': (v) => classifyByPrototype(v, Object.fromEntries(data.labels.map((l, i) => [l.id, centroid([...byLabel(l.id), descriptions[i]])]))),
      'k-nearest (k=3)': (v) => classifyByNeighbors(v, seedExamples, 3),
    };
    for (const [methodName, classify] of Object.entries(methods)) {
      const rows = data.test.map((t, i) => ({ text: t.text, hard: t.hard ?? null, truth: t.label, ...classify(sentences[i]) }));
      const rowsForSummary = rows.map((r) => ({ truth: r.truth, predicted: r.label, margin: r.margin }));
      const hard = rows.filter((r) => r.hard);
      report.methods[`${promptName} / ${methodName}`] = {
        ...summarize(rowsForSummary),
        hardAccuracy: hard.filter((r) => r.truth === r.label).length / hard.length,
        abstain: abstainCurve(rowsForSummary, [0, 0.01, 0.02, 0.03, 0.05]),
        errors: rows.filter((r) => r.truth !== r.label).map((r) => ({ text: r.text, truth: r.truth, predicted: r.label, margin: Number(r.margin.toFixed(3)), hard: r.hard })),
      };
    }
  }

  // The rules that read the date and time (they are not part of the model).
  const schedule = data.test.filter((t) => t.label === 'schedule');
  const fields = { date: 0, time: 0, endDate: 0, allDay: 0 };
  const counts = { date: 0, time: 0, endDate: 0, allDay: 0 };
  for (const item of schedule) {
    const got = parseWhen(item.text, data.now);
    for (const key of Object.keys(fields)) {
      if (!(key in item)) continue;
      counts[key]++;
      if (got[key] === item[key]) fields[key]++;
    }
  }
  report.parser = { fields, counts };

  await writeFile(path.join(OUT, `note-intents-${model}-${dtype}.json`), `${JSON.stringify(report, null, 2)}\n`);
  console.log(`\n${MODELS[model].name} ${dtype}: ${report.msPerSentence.toFixed(0)} ms/sentence (CPU, batched)`);
  for (const [name, m] of Object.entries(report.methods)) {
    console.log(`  ${name.padEnd(46)} ${pct(m.accuracy).padStart(6)}  難問 ${pct(m.hardAccuracy).padStart(6)}  recall ${Object.entries(m.recall).map(([k, v]) => `${k} ${pct(v)}`).join(' / ')}`);
  }
}
