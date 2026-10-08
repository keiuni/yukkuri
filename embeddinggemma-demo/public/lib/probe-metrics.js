// Scoring of the failure probes (data/failure-probes.json). Pure functions shared by the Node runner
// (scripts/failure-probes.mjs) and the unit tests. A model gives one score per candidate of a set
// (a cosine similarity, or a probability for a decision model); higher means "more likely the answer".

import { mean, percentile, rankIndices } from './metrics.js';

/** What the model picked for one query, whether it is right, and how sure it was. */
export function judgeQuery(scores, answer) {
  const order = rankIndices(scores);
  const wrong = scores.filter((_, i) => i !== answer);
  return {
    predicted: order[0],
    correct: order[0] === answer,
    // How far the right candidate is ahead of the best wrong one (negative when the model is wrong).
    margin: scores[answer] - Math.max(...wrong),
    // The gap between the first and second pick: all a model can see without knowing the answer.
    confidence: scores[order[0]] - scores[order[1]],
  };
}

/**
 * Per-group totals. `rows` hold { set, predicted, correct, margin, answer, candidates } (candidates =
 * number of candidates). A set passes when every one of its queries is right; it is "blind" when its
 * queries point at different candidates but the model picked the same one for all of them, i.e. the
 * difference the set is about made no difference to the model.
 */
export function summarizeRows(rows) {
  const bySet = new Map();
  for (const row of rows) bySet.set(row.set, [...(bySet.get(row.set) ?? []), row]);
  const sets = [...bySet.values()];
  const multi = sets.filter((setRows) => new Set(setRows.map((row) => row.answer)).size > 1);
  return {
    queries: rows.length,
    correct: rows.filter((row) => row.correct).length,
    accuracy: mean(rows.map((row) => (row.correct ? 1 : 0))),
    chance: mean(rows.map((row) => 1 / row.candidates)),
    sets: sets.length,
    setsPassed: sets.filter((setRows) => setRows.every((row) => row.correct)).length,
    pairedSets: multi.length,
    blindSets: multi.filter((setRows) => new Set(setRows.map((row) => row.predicted)).size === 1).length,
    medianMargin: percentile(rows.map((row) => row.margin), 50),
  };
}

/** summarizeRows for every value of `key` (e.g. the category), in the order of `order` when given. */
export function summarizeBy(rows, key, order) {
  const keys = order ?? [...new Set(rows.map((row) => row[key]))];
  return Object.fromEntries(keys.filter((k) => rows.some((row) => row[key] === k)).map((k) => [k, summarizeRows(rows.filter((row) => row[key] === k))]));
}

/** Area under the ROC curve: the chance that a random positive scores above a random negative (ties count half). */
export function rocAuc(positives, negatives) {
  if (positives.length === 0 || negatives.length === 0) return NaN;
  let wins = 0;
  for (const p of positives) for (const n of negatives) wins += p > n ? 1 : p === n ? 0.5 : 0;
  return wins / (positives.length * negatives.length);
}

/**
 * If the least confident picks were handed to a stronger model (and that model got them right), how many
 * of the errors would be caught? For each share of queries handed over, returns the confidence threshold,
 * the share of errors caught and the accuracy afterwards.
 */
export function escalation(rows, shares = [0.1, 0.2, 0.3, 0.5]) {
  const sorted = [...rows].sort((a, b) => a.confidence - b.confidence);
  const errors = rows.filter((row) => !row.correct).length;
  return shares.map((share) => {
    const handed = sorted.slice(0, Math.ceil(share * rows.length));
    const caught = handed.filter((row) => !row.correct).length;
    return {
      share,
      threshold: handed.at(-1)?.confidence ?? -Infinity,
      caught: errors ? caught / errors : 1,
      accuracyAfter: (rows.filter((row) => row.correct).length + caught) / rows.length,
    };
  });
}

/** Character bigrams of a text, for a crude "how many characters do they share" measure. */
function bigrams(text) {
  const clean = text.normalize('NFKC').replace(/\s+/g, '');
  const out = new Set();
  for (let i = 0; i < clean.length - 1; i++) out.add(clean.slice(i, i + 2));
  return out;
}

/** Jaccard overlap of character bigrams (0..1). */
export function charOverlap(a, b) {
  const x = bigrams(a);
  const y = bigrams(b);
  if (x.size === 0 || y.size === 0) return 0;
  let shared = 0;
  for (const gram of x) if (y.has(gram)) shared++;
  return shared / (x.size + y.size - shared);
}

/** The candidate a pure surface match would pick (most shared bigrams with the query); -1 on a tie for first. */
export function overlapPick(query, candidates) {
  const overlaps = candidates.map((candidate) => charOverlap(query, candidate));
  const best = Math.max(...overlaps);
  return overlaps.filter((value) => value === best).length > 1 ? -1 : overlaps.indexOf(best);
}
