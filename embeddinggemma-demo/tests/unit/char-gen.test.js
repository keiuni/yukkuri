import assert from 'node:assert/strict';
import { test } from 'node:test';

import { candidateChars, charRecall, dot, generateByEmbedding, orderScore, vocabChars } from '../../public/lib/char-gen.js';

test('vocabChars keeps every character once, kana first, Latin only on request', () => {
  const vocab = { hiragana: 'あい', katakana: 'アイ', symbols: '、。あ', kanji: '日本', latin: 'ab' };
  assert.deepEqual(vocabChars(vocab), ['あ', 'い', 'ア', 'イ', '、', '。', '日', '本']);
  assert.deepEqual(vocabChars(vocab, { latin: true }).slice(-2), ['a', 'b']);
});

test('candidateChars adds the characters closest to the target to the fixed ones', () => {
  const vocab = ['日', '本', '猫', 'の'];
  const charVectors = [
    [1, 0],
    [0.6, 0.8],
    [0, 1],
    [1, 0],
  ];
  const candidates = candidateChars({ target: [0, 1], vocab, charVectors, always: ['の'], shortlist: 2 });
  assert.deepEqual(candidates, ['猫', '本', 'の']);
});

// Toy embedding: one dimension per letter, counting occurrences, then normalized.
// The target is the bag of letters of "cab"; order does not matter to it, as with a real embedding.
const LETTERS = 'abcx';
function toyEmbed(texts) {
  return Promise.resolve(
    texts.map((text) => {
      const vector = [...LETTERS].map((letter) => [...text].filter((char) => char === letter).length);
      const norm = Math.sqrt(dot(vector, vector)) || 1;
      return vector.map((value) => value / norm);
    }),
  );
}

test('greedy search approaches the target and stops when it stops improving', async () => {
  const [target] = await toyEmbed(['cab']);
  const seen = [];
  const run = await generateByEmbedding({
    target,
    embed: toyEmbed,
    candidates: [...LETTERS],
    maxLength: 10,
    patience: 2,
    onStep: (step) => seen.push(step.text),
  });
  assert.equal([...run.text].sort().join(''), 'abc');
  assert.ok(Math.abs(run.score - 1) < 1e-9);
  assert.equal(seen.length, 5, 'three useful steps, then two without improvement');
  assert.equal(run.steps[0].alternatives.length, 4);
});

test('the stop callback ends the search before the next step', async () => {
  const [target] = await toyEmbed(['cab']);
  let steps = 0;
  const run = await generateByEmbedding({
    target,
    embed: toyEmbed,
    candidates: [...LETTERS],
    onStep: () => steps++,
    shouldStop: () => steps >= 1,
  });
  assert.equal(run.steps.length, 1);
});

test('charRecall and orderScore measure content and order separately', () => {
  assert.equal(charRecall('東京タワー', '塔京東'), 2 / 5);
  assert.equal(charRecall('abc', 'cba'), 1);
  assert.equal(orderScore('abc', 'cba'), 1 / 3);
  assert.equal(orderScore('abc', 'xaxbxc'), 1);
});
