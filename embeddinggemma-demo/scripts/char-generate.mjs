#!/usr/bin/env node
// EmbeddingGemma as a text generator: append one character at a time and keep the text whose
// embedding is closest to a target (public/lib/char-gen.js). Two kinds of target:
//   invert  the embedding of a sentence: can the sentence be written back from its embedding?
//   query   the embedding of a question: what text does the model consider the best answer?
// Results go to test-output/char-gen/results.json.
//
//   npm run char-generate                     # every experiment, greedy and beam 4
//   npm run char-generate -- invert-tower     # one experiment
//   npm run char-generate -- --beams 1        # greedy only
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { EmbeddingGemma } from '../agent/gemma.mjs';
import { candidateChars, charRecall, dot, generateByEmbedding, orderScore, vocabChars } from '../public/lib/char-gen.js';
import { documentPrompt, queryPrompt } from '../public/lib/model-config.js';
import { charVectorsFor } from './lib/char-vectors.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'test-output', 'char-gen');

export const EXPERIMENTS = [
  { id: 'invert-tower', kind: 'invert', text: '東京タワーは東京都港区にある電波塔です。' },
  { id: 'invert-cat', kind: 'invert', text: '猫に玉ねぎを食べさせてはいけない。' },
  { id: 'invert-rain', kind: 'invert', text: '明日は雨が降るので傘を持って行こう。' },
  { id: 'invert-english', kind: 'invert', text: 'the cat sat on the mat.', latin: true },
  { id: 'query-mountain', kind: 'query', text: '日本でいちばん高い山は？', reference: '日本でいちばん高い山は富士山で、標高は3776メートルです。' },
  { id: 'query-tower', kind: 'query', text: '東京タワーの高さは？', reference: '東京タワーの高さは333メートルです。' },
  { id: 'query-cat', kind: 'query', text: '猫に食べさせてはいけないものは？', reference: '玉ねぎやチョコレートは猫にとって有害なので、食べさせてはいけない。' },
];

// Brackets, ・ and ？！ are left out: their scores sit within a few thousandths of the best
// character at almost every step, so greedy search fills the text with them.
const ALWAYS_SYMBOLS = '、。々0123456789';

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : fallback;
};
const beams = String(option('beams', '1,4')).split(',').map(Number);
const shortlist = Number(option('shortlist', 48));
const optionValues = new Set([option('beams'), option('shortlist')].filter(Boolean));
const wanted = args.filter((arg) => !arg.startsWith('--') && !optionValues.has(arg));
const experiments = wanted.length ? EXPERIMENTS.filter((e) => wanted.includes(e.id)) : EXPERIMENTS;

await mkdir(OUT, { recursive: true });
const vocabJson = JSON.parse(await readFile(path.join(ROOT, 'public', 'data', 'char-vocab.json'), 'utf8'));
const embedder = await EmbeddingGemma.load();
embedder.batchSize = 32;
const asDocument = (texts) => embedder.embed(texts.map((text) => documentPrompt(null) + text));

const file = path.join(OUT, wanted.length ? `results-${wanted.join('+')}.json` : 'results.json');
const results = [];
// Written after every run, so a long session leaves its finished runs behind even if it is stopped.
const save = () => writeFile(file, `${JSON.stringify({ createdAt: new Date().toISOString(), shortlist, results }, null, 1)}\n`);

for (const experiment of experiments) {
  const vocab = experiment.latin ? [...new Set(vocabJson.latin)] : vocabChars(vocabJson);
  const always = experiment.latin ? vocab : [...new Set([...vocabJson.hiragana, ...vocabJson.katakana, ...ALWAYS_SYMBOLS])];
  const charVectors = await charVectorsFor(embedder, vocab);
  const [target] =
    experiment.kind === 'invert'
      ? await asDocument([experiment.text])
      : await embedder.embed([queryPrompt('search result') + experiment.text]);
  const candidates = candidateChars({ target, vocab, charVectors, always, shortlist: experiment.latin ? 0 : shortlist });
  const reference = experiment.reference ? dot((await asDocument([experiment.reference]))[0], target) : null;
  const maxLength = experiment.kind === 'invert' ? [...experiment.text].length + 4 : 20;

  for (const beam of beams) {
    const started = Date.now();
    process.stdout.write(`${experiment.id} beam=${beam} `);
    const run = await generateByEmbedding({
      target,
      embed: asDocument,
      candidates,
      beam,
      maxLength,
      patience: 4,
      onStep: () => process.stdout.write('.'),
    });
    const seconds = (Date.now() - started) / 1000;
    const entry = {
      id: experiment.id,
      kind: experiment.kind,
      input: experiment.text,
      beam,
      output: run.text,
      score: run.score,
      referenceText: experiment.reference ?? null,
      referenceScore: reference,
      charRecall: experiment.kind === 'invert' ? charRecall(experiment.text, run.text) : null,
      orderScore: experiment.kind === 'invert' ? orderScore(experiment.text, run.text) : null,
      candidates: candidates.length,
      shortlist: candidates.filter((char) => !always.includes(char)).join(''),
      seconds,
      steps: run.steps,
    };
    results.push(entry);
    await save();
    console.log(
      ` ${seconds.toFixed(0)}s  cos=${run.score.toFixed(3)}${reference !== null ? ` (reference ${reference.toFixed(3)})` : ''}` +
        `${entry.charRecall !== null ? ` recall=${entry.charRecall.toFixed(2)} order=${entry.orderScore.toFixed(2)}` : ''}  「${run.text}」`,
    );
  }
}

console.log(`Saved ${path.relative(ROOT, file)}`);
