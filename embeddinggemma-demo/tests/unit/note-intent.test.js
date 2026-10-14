import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { abstainCurve, centroid, classifyByNeighbors, classifyByPrototype, parseWhen, summarize } from '../../public/lib/note-intent.js';

const data = JSON.parse(readFileSync(new URL('../../public/data/note-intents.json', import.meta.url), 'utf8'));

test('every schedule sentence in the test set resolves to the date and time it was labelled with', () => {
  const failures = [];
  for (const item of data.test.filter((t) => t.label === 'schedule')) {
    const got = parseWhen(item.text, data.now);
    for (const key of ['date', 'time', 'endDate', 'allDay']) {
      if (key in item && got[key] !== item[key]) failures.push(`${item.text}: ${key} ${got[key]} != ${item[key]}`);
    }
  }
  assert.deepEqual(failures, []);
});

test('deadlines of to-dos resolve, and sentences without a date give nothing', () => {
  for (const item of data.test.filter((t) => t.due)) assert.equal(parseWhen(item.text, data.now).date, item.due, item.text);
  assert.deepEqual(parseWhen('牛乳と卵を買って帰る', data.now), { date: null, time: null, endDate: null, allDay: false });
});

test('weeks run Monday to Sunday and a bare weekday is the next one', () => {
  // 2026-10-14 is a Wednesday.
  assert.equal(parseWhen('来週の月曜', '2026-10-14').date, '2026-10-19');
  assert.equal(parseWhen('今週の月曜', '2026-10-14').date, '2026-10-12');
  assert.equal(parseWhen('水曜', '2026-10-14').date, '2026-10-21');
  assert.equal(parseWhen('来月の3日', '2026-12-20').date, '2027-01-03');
  assert.equal(parseWhen('3月5日', '2026-10-14').date, '2027-03-05');
  assert.equal(parseWhen('5日に', '2026-10-14').date, '2026-11-05');
});

test('prototypes and neighbours vote, and the margin is the gap to the runner-up', () => {
  const a = Float32Array.from([1, 0]);
  const b = Float32Array.from([0, 1]);
  const prototypes = { x: centroid([a, Float32Array.from([0.9, 0.1])]), y: centroid([b]) };
  const result = classifyByPrototype(Float32Array.from([0.8, 0.6]), prototypes);
  assert.equal(result.label, 'x');
  assert.ok(result.margin > 0);
  assert.equal(classifyByNeighbors(a, [{ label: 'x', vector: a }, { label: 'y', vector: b }], 1).label, 'x');
});

test('summaries count the confusion and the abstain curve trades coverage for accuracy', () => {
  const rows = [
    { truth: 'a', predicted: 'a', margin: 0.2 },
    { truth: 'a', predicted: 'b', margin: 0.01 },
    { truth: 'b', predicted: 'b', margin: 0.1 },
    { truth: 'b', predicted: 'a', margin: 0.02 },
  ];
  const s = summarize(rows);
  assert.equal(s.accuracy, 0.5);
  assert.deepEqual(s.confusion.a, { a: 1, b: 1 });
  const [all, sure] = abstainCurve(rows, [0, 0.05]);
  assert.equal(all.coverage, 1);
  assert.equal(sure.coverage, 0.5);
  assert.equal(sure.accuracy, 1);
});

test('phrasings written after the rules: slash dates, noon, the n-th weekday of a month', () => {
  // `unseen` was run against the rules before these three were added: 15 of 18 were right, and the misses were exactly these.
  const failures = [];
  for (const item of data.unseen) {
    const got = parseWhen(item.text, data.now);
    if (got.date !== item.date || got.time !== item.time) failures.push(`${item.text}: ${got.date} ${got.time}`);
  }
  assert.deepEqual(failures, []);
  assert.equal(parseWhen('今月の第3水曜日に勉強会', '2026-10-14').date, '2026-10-21');
  assert.equal(parseWhen('第1月曜日は定例', '2026-10-14').date, '2026-11-02'); // this month's has passed
});
