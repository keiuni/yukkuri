// Vector math and retrieval metrics. Pure functions shared by the UI and the unit tests.
// Matrices are flat Float32Arrays in row-major order: row i of an [n, dim] matrix is
// vectors[i * dim ... (i + 1) * dim).

/** Keeps the first `dims` components of every row (Matryoshka truncation) and L2-normalizes them. */
export function truncateAndNormalize(vectors, dim, dims = dim) {
  if (dims > dim) throw new Error(`Cannot truncate ${dim}-d vectors to ${dims} dimensions`);
  const rows = vectors.length / dim;
  const out = new Float32Array(rows * dims);
  for (let i = 0; i < rows; i++) {
    let norm = 0;
    for (let j = 0; j < dims; j++) norm += vectors[i * dim + j] ** 2;
    norm = Math.sqrt(norm) || 1;
    for (let j = 0; j < dims; j++) out[i * dims + j] = vectors[i * dim + j] / norm;
  }
  return out;
}

/** Dot products between every row of `a` and every row of `b` (cosine similarity for normalized rows). */
export function similarityMatrix(a, b, dims) {
  const rowsA = a.length / dims;
  const rowsB = b.length / dims;
  const out = new Float32Array(rowsA * rowsB);
  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < rowsB; j++) {
      let dot = 0;
      for (let k = 0; k < dims; k++) dot += a[i * dims + k] * b[j * dims + k];
      out[i * rowsB + j] = dot;
    }
  }
  return out;
}

/** Indices of `scores` ordered from the highest to the lowest score. */
export function rankIndices(scores) {
  return Array.from(scores.keys()).sort((x, y) => scores[y] - scores[x]);
}

/**
 * Binary-relevance metrics for one ranked list.
 * @param {string[]} rankedIds document ids, best first
 * @param {Set<string>} relevant ids of the relevant documents
 */
export function rankingMetrics(rankedIds, relevant, k = 10) {
  const firstHit = rankedIds.findIndex((id) => relevant.has(id));
  const rank = firstHit === -1 ? Infinity : firstHit + 1;

  let dcg = 0;
  rankedIds.slice(0, k).forEach((id, i) => {
    if (relevant.has(id)) dcg += 1 / Math.log2(i + 2);
  });
  let idcg = 0;
  for (let i = 0; i < Math.min(relevant.size, k); i++) idcg += 1 / Math.log2(i + 2);

  const hitsAt5 = rankedIds.slice(0, 5).filter((id) => relevant.has(id)).length;
  return {
    rank,
    acc1: rank === 1 ? 1 : 0,
    recall5: hitsAt5 / relevant.size,
    mrr10: rank <= k ? 1 / rank : 0,
    ndcg10: idcg > 0 ? dcg / idcg : 0,
  };
}

export const METRIC_KEYS = ['acc1', 'recall5', 'mrr10', 'ndcg10'];

export function mean(values) {
  return values.length === 0 ? NaN : values.reduce((sum, value) => sum + value, 0) / values.length;
}

/** Nearest-rank percentile (p in 0..100). */
export function percentile(values, p) {
  if (values.length === 0) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[index];
}

function averageMetrics(rows) {
  return Object.fromEntries([
    ...METRIC_KEYS.map((key) => [key, mean(rows.map((row) => row[key]))]),
    ['count', rows.length],
  ]);
}

/** Language pair of a query, e.g. "ja→en" for a Japanese query whose answer is an English document. */
export function queryType(query, docsById) {
  const docLang = docsById.get(query.relevant[0])?.lang ?? '?';
  return `${query.lang}→${docLang}`;
}

/**
 * Evaluates retrieval for every query against the whole corpus.
 * `queryVectors` / `docVectors` must already be truncated and normalized to `dims`.
 */
export function evaluateRetrieval({ queries, docs, queryVectors, docVectors, dims }) {
  const docsById = new Map(docs.map((doc) => [doc.id, doc]));
  const scores = similarityMatrix(queryVectors, docVectors, dims);

  const perQuery = queries.map((query, qi) => {
    const row = scores.subarray(qi * docs.length, (qi + 1) * docs.length);
    const order = rankIndices(row);
    const rankedIds = order.map((di) => docs[di].id);
    const metrics = rankingMetrics(rankedIds, new Set(query.relevant));
    return {
      id: query.id,
      text: query.text,
      type: queryType(query, docsById),
      relevant: query.relevant,
      ...metrics,
      top: order.slice(0, 3).map((di) => ({ id: docs[di].id, score: row[di] })),
      relevantScore: Math.max(...query.relevant.map((id) => row[docs.findIndex((doc) => doc.id === id)])),
    };
  });

  const byType = {};
  for (const type of [...new Set(perQuery.map((row) => row.type))].sort()) {
    byType[type] = averageMetrics(perQuery.filter((row) => row.type === type));
  }
  return { overall: averageMetrics(perQuery), byType, perQuery };
}
