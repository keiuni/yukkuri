import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  evaluateRetrieval,
  percentile,
  rankIndices,
  rankingMetrics,
  similarityMatrix,
  truncateAndNormalize,
} from '../../public/lib/metrics.js';

const close = (actual, expected, epsilon = 1e-6) =>
  assert.ok(Math.abs(actual - expected) < epsilon, `expected ${actual} to be close to ${expected}`);

test('truncateAndNormalize keeps the leading dimensions and re-normalizes each row', () => {
  const vectors = new Float32Array([3, 4, 12, 0, 0, 5, 0, 0]);
  const truncated = truncateAndNormalize(vectors, 4, 2);
  assert.deepEqual(Array.from(truncated).map((x) => +x.toFixed(6)), [0.6, 0.8, 0, 1]);
  assert.throws(() => truncateAndNormalize(vectors, 4, 8));
});

test('similarityMatrix returns row-major dot products', () => {
  const a = new Float32Array([1, 0, 0, 1]);
  const b = new Float32Array([1, 0, 0.6, 0.8, 0, 1]);
  assert.deepEqual(Array.from(similarityMatrix(a, b, 2)).map((x) => +x.toFixed(4)), [1, 0.6, 0, 0, 0.8, 1]);
});

test('rankIndices orders by descending score', () => {
  assert.deepEqual(rankIndices(new Float32Array([0.1, 0.9, 0.5])), [1, 2, 0]);
});

test('rankingMetrics handles hits at rank 1, rank 3 and misses', () => {
  const first = rankingMetrics(['a', 'b', 'c'], new Set(['a']));
  assert.deepEqual(first, { rank: 1, acc1: 1, recall5: 1, mrr10: 1, ndcg10: 1 });

  const third = rankingMetrics(['x', 'y', 'a'], new Set(['a']));
  assert.equal(third.rank, 3);
  assert.equal(third.acc1, 0);
  close(third.mrr10, 1 / 3);
  close(third.ndcg10, 1 / Math.log2(4));

  const miss = rankingMetrics(['x', 'y'], new Set(['a']));
  assert.deepEqual(miss, { rank: Infinity, acc1: 0, recall5: 0, mrr10: 0, ndcg10: 0 });
});

test('rankingMetrics normalizes nDCG with several relevant documents', () => {
  const metrics = rankingMetrics(['a', 'x', 'b'], new Set(['a', 'b']));
  const dcg = 1 + 1 / Math.log2(4);
  const idcg = 1 + 1 / Math.log2(3);
  close(metrics.ndcg10, dcg / idcg);
  assert.equal(metrics.recall5, 1);
});

test('percentile uses the nearest rank', () => {
  assert.equal(percentile([5, 1, 3, 2, 4], 50), 3);
  assert.equal(percentile([5, 1, 3, 2, 4], 95), 5);
  assert.ok(Number.isNaN(percentile([], 50)));
});

test('evaluateRetrieval aggregates overall and per language pair', () => {
  const docs = [
    { id: 'd1', lang: 'ja' },
    { id: 'd2', lang: 'en' },
  ];
  const queries = [
    { id: 'q1', lang: 'ja', text: 'q1', relevant: ['d1'] },
    { id: 'q2', lang: 'ja', text: 'q2', relevant: ['d2'] },
  ];
  // q1 points at d1 (hit); q2 also points at d1, so d2 is ranked second (miss at 1).
  const queryVectors = new Float32Array([1, 0, 0.8, 0.6]);
  const docVectors = new Float32Array([1, 0, 0, 1]);
  const result = evaluateRetrieval({ queries, docs, queryVectors, docVectors, dims: 2 });

  assert.equal(result.overall.count, 2);
  close(result.overall.acc1, 0.5);
  close(result.overall.mrr10, 0.75);
  assert.deepEqual(Object.keys(result.byType), ['ja→en', 'ja→ja']);
  assert.equal(result.byType['ja→ja'].acc1, 1);
  assert.equal(result.perQuery[1].rank, 2);
  close(result.perQuery[1].relevantScore, 0.6);
});
