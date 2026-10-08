// Character-by-character generation with an embedding model, no language model involved.
// At every step each candidate character is appended to the current text, the results are
// embedded, and the one closest (cosine) to a target embedding is kept (beam search when beam > 1).
// The only signal is EmbeddingGemma's similarity, so the output shows what the embedding rewards.
//
// Used by the browser demo (the 文字生成 tab) and scripts/char-generate.mjs.

/** All characters of public/data/char-vocab.json, in a stable order without duplicates. */
export function vocabChars(vocab, { latin = false } = {}) {
  const parts = [vocab.hiragana, vocab.katakana, vocab.symbols, vocab.kanji, latin ? vocab.latin : ''];
  return [...new Set(parts.flatMap((part) => [...part]))];
}

export function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

/**
 * The characters tried at every step: those listed in `always` (kana and punctuation, which mean
 * little on their own but are needed to write a sentence), plus the `shortlist` characters whose
 * own embedding is closest to the target.
 */
export function candidateChars({ target, vocab, charVectors, always = [], shortlist = 48 }) {
  const fixed = new Set(always);
  const ranked = vocab
    .map((char, i) => ({ char, score: dot(charVectors[i], target) }))
    .filter(({ char }) => !fixed.has(char))
    .sort((a, b) => b.score - a.score)
    .slice(0, shortlist)
    .map(({ char }) => char);
  return [...ranked, ...fixed];
}

/**
 * @param {object} options
 * @param {Float32Array|number[]} options.target unit vector to approach
 * @param {(texts: string[]) => Promise<ArrayLike<number>[]>} options.embed unit vectors of the texts
 * @param {string[]} options.candidates characters tried at every step (see candidateChars)
 * @param {number} [options.beam=1] texts kept after each step (1 = greedy)
 * @param {number} [options.maxLength=24] characters to add at most
 * @param {number} [options.patience=3] stop after this many steps without a better best text
 * @param {string} [options.prefix=''] text to start from
 * @param {(step: object) => void} [options.onStep] called after every step
 * @param {() => boolean} [options.shouldStop] checked before every step (for a stop button)
 */
export async function generateByEmbedding({
  target,
  embed,
  candidates,
  beam = 1,
  maxLength = 24,
  patience = 3,
  prefix = '',
  onStep,
  shouldStop,
}) {
  const startScore = prefix ? dot((await embed([prefix]))[0], target) : -Infinity;
  let beams = [{ text: prefix, score: startScore }];
  let best = beams[0];
  let stale = 0;
  const steps = [];

  for (let step = 1; step <= maxLength && !shouldStop?.(); step++) {
    const started = Date.now();
    const texts = [...new Set(beams.flatMap((current) => candidates.map((char) => current.text + char)))];
    const vectors = await embed(texts);
    const scored = texts.map((text, i) => ({ text, score: dot(vectors[i], target) })).sort((a, b) => b.score - a.score);
    beams = scored.slice(0, beam);

    // What the leading text could have become, for the step-by-step view.
    const leader = beams[0];
    const parent = leader.text.slice(0, -1);
    const alternatives = scored
      .filter(({ text }) => text.slice(0, -1) === parent)
      .slice(0, 5)
      .map(({ text, score }) => ({ char: text.slice(-1), score }));
    const record = { step, text: leader.text, score: leader.score, alternatives, evaluated: texts.length, ms: Date.now() - started };
    steps.push(record);
    onStep?.(record);

    if (leader.score > best.score + 1e-4) {
      best = leader;
      stale = 0;
    } else if (++stale >= patience) {
      break;
    }
  }
  return { text: best.text, score: best.score, steps };
}

/** Share of the target's distinct characters that appear in `text`. */
export function charRecall(target, text) {
  const wanted = new Set([...target].filter((char) => !/\s/.test(char)));
  if (wanted.size === 0) return 0;
  const got = new Set(text);
  let hit = 0;
  for (const char of wanted) if (got.has(char)) hit++;
  return hit / wanted.size;
}

/** Length of the longest common subsequence, divided by the target's length (order as well as content). */
export function orderScore(target, text) {
  const a = [...target];
  const b = [...text];
  if (a.length === 0) return 0;
  let previous = new Array(b.length + 1).fill(0);
  for (let i = 1; i <= a.length; i++) {
    const current = new Array(b.length + 1).fill(0);
    for (let j = 1; j <= b.length; j++) {
      current[j] = a[i - 1] === b[j - 1] ? previous[j - 1] + 1 : Math.max(previous[j], current[j - 1]);
    }
    previous = current;
  }
  return previous[b.length] / a.length;
}
