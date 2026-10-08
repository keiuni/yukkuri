// 文字生成 tab: EmbeddingGemma writes text one character at a time, always keeping the character
// that brings the text's embedding closest to a target (public/lib/char-gen.js).
import { candidateChars, charRecall, generateByEmbedding, orderScore } from './lib/char-gen.js';
import { documentPrompt, queryPrompt } from './lib/model-config.js';

const EXAMPLES = {
  invert: '東京タワーは東京都港区にある電波塔です。',
  query: '日本でいちばん高い山は？',
};
const INPUT_LABELS = { invert: '元の文（この文の埋め込みを目標にする）', query: '質問（この質問の埋め込みを目標にする）' };
// Kana and a few marks are tried at every step; kanji come from a per-target shortlist.
const ALWAYS_SYMBOLS = '、。々0123456789';

let assets = null;

/** Vocabulary and the int8 single-character embeddings used for the kanji shortlist. */
async function loadAssets() {
  if (assets) return assets;
  const [vocab, packed] = await Promise.all(
    ['data/char-vocab.json', 'data/char-vectors.json'].map((file) => fetch(new URL(file, import.meta.url)).then((response) => response.json())),
  );
  const bytes = Uint8Array.from(atob(packed.data), (char) => char.charCodeAt(0));
  const int8 = new Int8Array(bytes.buffer);
  const chars = [...packed.chars];
  const charVectors = chars.map((_, i) => int8.subarray(i * packed.dim, (i + 1) * packed.dim));
  assets = { vocab, chars, charVectors, dim: packed.dim };
  return assets;
}

function normalize(vector) {
  let norm = 0;
  for (const value of vector) norm += value * value;
  norm = Math.sqrt(norm) || 1;
  return Float32Array.from(vector, (value) => value / norm);
}

export function initCharGen({ $, el, embedCached, runExclusive, fmtMs }) {
  let stopRequested = false;
  const mode = () => $('chargen-mode').value;

  const showExample = () => {
    $('chargen-input').value = EXAMPLES[mode()];
    $('chargen-input-label').textContent = INPUT_LABELS[mode()];
  };
  $('chargen-mode').addEventListener('change', showExample);
  showExample();
  $('chargen-stop').addEventListener('click', () => {
    stopRequested = true;
  });

  /** Unit vectors for texts, through the page's embedding cache. */
  async function embed(texts) {
    const { vectors, dim } = await embedCached(texts, { batchSize: 32 });
    return texts.map((_, i) => normalize(vectors.subarray(i * dim, (i + 1) * dim)));
  }

  async function run() {
    const input = $('chargen-input').value.trim();
    if (!input) return;
    const results = $('chargen-results');
    const beam = Number($('chargen-beam').value);
    const shortlist = Number($('chargen-shortlist').value);
    const maxLength = Number($('chargen-length').value);
    const targetMode = mode();
    stopRequested = false;
    $('chargen-stop').hidden = false;
    $('chargen-note').textContent = '準備中…';

    try {
      const { vocab, chars, charVectors, dim } = await loadAssets();
      const [target] = await embed([(targetMode === 'invert' ? documentPrompt(null) : queryPrompt('search result')) + input]);
      const kana = $('chargen-kana').value === '1' ? [...vocab.hiragana, ...vocab.katakana] : [];
      const always = [...new Set([...kana, ...ALWAYS_SYMBOLS])];
      const candidates = candidateChars({ target: normalize(target.subarray(0, dim)), vocab: chars, charVectors, always, shortlist });
      const kanji = candidates.filter((char) => !always.includes(char)).join('');

      const current = el('p', { className: 'chargen-current' });
      const meta = el('p', { className: 'result-meta', text: `候補 ${candidates.length} 文字（漢字の候補: ${kanji}）` });
      const steps = el('ol', { className: 'chargen-steps' });
      results.replaceChildren(current, meta, steps);
      const started = performance.now();

      const run = await generateByEmbedding({
        target,
        embed: (texts) => embed(texts.map((text) => documentPrompt(null) + text)),
        candidates,
        beam,
        maxLength,
        patience: 4,
        shouldStop: () => stopRequested,
        onStep: (step) => {
          current.replaceChildren(el('span', { className: 'chargen-text', text: step.text }), el('span', { className: 'chargen-score', text: `類似度 ${step.score.toFixed(3)}` }));
          steps.prepend(
            el(
              'li',
              {},
              el('span', { className: 'chargen-step-text', text: `${step.step}. ${step.text}` }),
              el(
                'span',
                { className: 'chargen-alternatives' },
                step.alternatives.map((alt, i) => el('span', { className: i === 0 ? 'chosen' : '', text: `${alt.char} ${alt.score.toFixed(3)}` })),
              ),
            ),
          );
          $('chargen-note').textContent = `${step.step} 文字目（${step.evaluated} 通りを比較、${fmtMs(step.ms)}）`;
        },
      });

      current.replaceChildren(el('span', { className: 'chargen-text', text: run.text }), el('span', { className: 'chargen-score', text: `類似度 ${run.score.toFixed(3)}` }));
      const summary = [`${fmtMs(performance.now() - started)}`, `${run.steps.length} ステップ`];
      if (targetMode === 'invert') {
        summary.push(`元の文の文字を含む割合 ${(charRecall(input, run.text) * 100).toFixed(0)}%`, `順序の一致 ${(orderScore(input, run.text) * 100).toFixed(0)}%`);
      }
      $('chargen-note').textContent = `${stopRequested ? '途中で止めました' : '完了'}（${summary.join('・')}）`;
      results.dataset.done = 'true';
    } catch (error) {
      $('chargen-note').textContent = `エラー: ${error.message}`;
      results.dataset.done = 'error';
    } finally {
      $('chargen-stop').hidden = true;
    }
  }

  $('chargen-button').addEventListener('click', () => {
    $('chargen-results').dataset.done = 'false';
    runExclusive(run);
  });
}
