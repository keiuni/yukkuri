import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { charOverlap, escalation, judgeQuery, overlapPick, rocAuc, summarizeBy, summarizeRows } from '../../public/lib/probe-metrics.js';
import { HARD_TASKS } from '../../agent/tasks.mjs';

const probes = JSON.parse(readFileSync(new URL('../../public/data/failure-probes.json', import.meta.url), 'utf8'));

test('judgeQuery reports the pick, the margin to the best wrong candidate and the top-two gap', () => {
  const right = judgeQuery([0.2, 0.9, 0.5], 1);
  assert.equal(right.predicted, 1);
  assert.equal(right.correct, true);
  assert.ok(Math.abs(right.margin - 0.4) < 1e-9);
  assert.ok(Math.abs(right.confidence - 0.4) < 1e-9);

  const wrong = judgeQuery([0.8, 0.7, 0.1], 1);
  assert.equal(wrong.predicted, 0);
  assert.equal(wrong.correct, false);
  assert.ok(Math.abs(wrong.margin + 0.1) < 1e-9);
});

test('a set whose queries all got the same pick counts as blind', () => {
  const row = (set, answer, predicted) => ({ set, category: 'c', answer, predicted, correct: answer === predicted, margin: 0, candidates: 2 });
  const rows = [row('a', 0, 0), row('a', 1, 0), row('b', 0, 0), row('b', 1, 1), row('c', 0, 1)];
  const summary = summarizeRows(rows);
  assert.equal(summary.queries, 5);
  assert.equal(summary.correct, 3);
  assert.equal(summary.chance, 0.5);
  assert.equal(summary.sets, 3);
  assert.equal(summary.setsPassed, 1);
  // Only sets whose queries point at different candidates can be blind: 'a' is, 'b' is not, 'c' has one query.
  assert.equal(summary.pairedSets, 2);
  assert.equal(summary.blindSets, 1);
  assert.deepEqual(Object.keys(summarizeBy(rows, 'category', ['x', 'c'])), ['c']);
});

test('rocAuc counts wins and half ties', () => {
  assert.equal(rocAuc([0.9, 0.8], [0.1, 0.2]), 1);
  assert.equal(rocAuc([0.1], [0.9]), 0);
  assert.equal(rocAuc([0.5], [0.5]), 0.5);
  assert.ok(Number.isNaN(rocAuc([], [0.5])));
});

test('escalation hands over the least confident picks first', () => {
  const rows = [
    { correct: false, confidence: 0.01 },
    { correct: true, confidence: 0.02 },
    { correct: false, confidence: 0.3 },
    { correct: true, confidence: 0.4 },
  ];
  const [quarter, half] = escalation(rows, [0.25, 0.5]);
  assert.equal(quarter.caught, 0.5);
  assert.equal(quarter.accuracyAfter, 0.75);
  assert.equal(half.threshold, 0.02);
  assert.equal(half.caught, 0.5);
});

test('character overlap favours shared words and reports ties', () => {
  assert.equal(charOverlap('猫が好き', '猫が好き'), 1);
  assert.equal(charOverlap('猫', '犬'), 0);
  assert.equal(overlapPick('パソコンが立ち上がらない', ['電源を入れても画面が暗い', 'パソコンの立ち上がりを速くする']), 1);
  assert.equal(overlapPick('ab', ['xy', 'zw']), -1);
});

test('every probe has a valid answer and the fillers stay apart from the candidates', () => {
  const categories = new Set(probes.categories.map((category) => category.id));
  const ids = new Set();
  for (const set of probes.sets) {
    assert.ok(!ids.has(set.id), `duplicate set ${set.id}`);
    ids.add(set.id);
    assert.ok(categories.has(set.category), set.id);
    assert.ok(['ui', 'doc'].includes(set.style), set.id);
    assert.ok(set.candidates.length >= 2 && new Set(set.candidates).size === set.candidates.length, set.id);
    for (const query of set.queries) assert.ok(Number.isInteger(query.answer) && query.answer >= 0 && query.answer < set.candidates.length, `${set.id}: ${query.text}`);
  }
  for (const item of probes.unanswerable) assert.ok(ids.has(item.set), item.set);
  const candidates = new Set(probes.sets.flatMap((set) => set.candidates));
  for (const filler of [...probes.fillers.ui, ...probes.fillers.doc]) assert.ok(!candidates.has(filler), filler);
});

test('hard tasks say what the right action is and how a page-aware planner would word the steps', () => {
  for (const task of HARD_TASKS) {
    assert.ok(task.expect, task.id);
    assert.ok(task.steps.length > 0, task.id);
    assert.ok(Array.isArray(task.plainSteps), task.id);
    for (const step of [...task.steps, ...task.plainSteps]) assert.equal(typeof step.text, 'string', task.id);
  }
});
