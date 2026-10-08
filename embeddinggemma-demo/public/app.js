import { initCharGen } from './chargen-tab.js';
import { SEARCH_PRESETS, SIMILARITY_SENTENCES } from './data/presets.js';
import { evaluateRetrieval, mean, percentile, rankIndices, similarityMatrix, truncateAndNormalize } from './lib/metrics.js';
import { DTYPES, MODEL_ID, MRL_DIMS, NO_PROMPT, QUERY_TASKS, documentPrompt, queryPrompt } from './lib/model-config.js';

// URL options (handy for automation): ?dtype=q4&device=wasm&threads=4&autoload=1&cache=0&local=0
const params = new URL(location.href).searchParams;
const $ = (id) => document.getElementById(id);

const MAX_SIMILARITY_SENTENCES = 16;
const DEVICE_LABELS = { wasm: 'WASM（CPU）', webgpu: 'WebGPU' };
const TYPE_LABELS = { 'ja→ja': '日→日', 'en→en': '英→英', 'ja→en': '日→英', 'en→ja': '英→日' };

// ---------------------------------------------------------------- helpers

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === 'className') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key.startsWith('data-') || key.startsWith('aria-') || key === 'title' || key === 'scope') node.setAttribute(key, value);
    else node[key] = value;
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined && child !== false) node.append(child);
  }
  return node;
}

const lines = (text) =>
  text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);

const fmtScore = (x) => (Number.isFinite(x) ? x.toFixed(3) : '–');
const fmtPct = (x) => (Number.isFinite(x) ? `${(x * 100).toFixed(1)}%` : '–');
const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} 秒` : `${Math.round(ms)} ms`);
const fmtMB = (bytes) => `${(bytes / 1e6).toFixed(0)} MB`;

function setProgress(container, fraction, label) {
  container.hidden = false;
  container.querySelector('.progress-fill').style.width = `${Math.round(Math.min(1, Math.max(0, fraction)) * 100)}%`;
  container.querySelector('.progress-label').textContent = label;
}

function download(filename, data) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  const link = el('a', { href: url, download: filename });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------------------------------------------------------------- worker client

class EmbedderClient {
  #worker = null;
  #pending = new Map();
  #nextId = 1;

  start() {
    this.stop();
    this.#worker = new Worker(new URL('./embedder.worker.js', import.meta.url), { type: 'module' });
    this.#worker.addEventListener('message', ({ data }) => {
      const entry = this.#pending.get(data.id);
      if (!entry) return;
      if (data.type === 'progress') {
        entry.onProgress?.(data);
        return;
      }
      this.#pending.delete(data.id);
      if (data.type === 'error') entry.reject(new Error(data.message));
      else entry.resolve(data);
    });
    this.#worker.addEventListener('error', (event) => {
      event.preventDefault();
      this.#rejectAll(new Error(event.message || 'ワーカーの起動に失敗しました'));
    });
  }

  stop() {
    this.#worker?.terminate();
    this.#worker = null;
    this.#rejectAll(new Error('ワーカーを停止しました'));
  }

  request(message, onProgress) {
    if (!this.#worker) return Promise.reject(new Error('モデルが読み込まれていません'));
    const id = this.#nextId++;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject, onProgress });
      this.#worker.postMessage({ ...message, id });
    });
  }

  #rejectAll(error) {
    for (const entry of this.#pending.values()) entry.reject(error);
    this.#pending.clear();
  }
}

const client = new EmbedderClient();

// Everything the E2E tests (and curious users in DevTools) may want to read back.
const state = {
  model: null,
  busy: false,
  dataset: null,
  cache: new Map(), // prompt + text -> Float32Array, valid for the currently loaded model
  last: { search: null, similarity: null, benchmark: null },
  runs: { search: 0, similarity: 0, benchmark: 0 },
};
window.__demo = state;

/** Bumps the run counter of a tab; results containers expose it as data-run so automation can wait for fresh output. */
function markRun(kind, container) {
  state.runs[kind] += 1;
  container.dataset.run = String(state.runs[kind]);
}

/** Embeds texts, reusing vectors computed earlier for the same model and prompt. */
async function embedCached(texts, { batchSize = 8, onProgress } = {}) {
  const missing = [...new Set(texts.filter((text) => !state.cache.has(text)))];
  if (missing.length > 0) {
    const result = await client.request({ type: 'embed', texts: missing, batchSize }, onProgress);
    missing.forEach((text, i) => state.cache.set(text, result.vectors.slice(i * result.dim, (i + 1) * result.dim)));
  }
  const dim = state.cache.get(texts[0]).length;
  const vectors = new Float32Array(texts.length * dim);
  texts.forEach((text, i) => vectors.set(state.cache.get(text), i * dim));
  return { vectors, dim, computed: missing.length };
}

async function runExclusive(task) {
  if (state.busy) return;
  state.busy = true;
  updateButtons();
  try {
    await task();
  } finally {
    state.busy = false;
    updateButtons();
  }
}

function updateButtons() {
  for (const button of document.querySelectorAll('.needs-model')) {
    button.disabled = state.busy || !state.model;
  }
  $('load-button').disabled = state.busy;
}

// ---------------------------------------------------------------- model panel

function setStatus(stateName, text) {
  const status = $('model-status');
  status.dataset.state = stateName;
  status.textContent = text;
}

function fillSelect(select, options, selected) {
  select.replaceChildren(
    ...options.map(({ value, label, disabled }) => el('option', { value, text: label, disabled, selected: value === selected })),
  );
}

async function initModelPanel() {
  fillSelect(
    $('dtype'),
    Object.entries(DTYPES).map(([value, { label, sizeMB }]) => ({
      value,
      label: `${label} · ${sizeMB >= 1000 ? `${(sizeMB / 1000).toFixed(1)} GB` : `${sizeMB} MB`}`,
    })),
    DTYPES[params.get('dtype')] ? params.get('dtype') : 'q8',
  );

  const cores = navigator.hardwareConcurrency || 1;
  const threadChoices = [1, 2, 4, 8, 16].filter((n) => n <= cores);
  if (!threadChoices.includes(cores)) threadChoices.push(cores);
  fillSelect(
    $('threads'),
    [{ value: 'auto', label: '自動（onnxruntime既定）' }, ...threadChoices.map((n) => ({ value: String(n), label: `${n}` }))],
    params.get('threads') ?? 'auto',
  );

  const hints = [];
  if (!self.crossOriginIsolated) {
    $('threads').disabled = true;
    hints.push('このページはクロスオリジン分離されていないため、WASMは1スレッドで動きます（npm start のサーバーを使うとマルチスレッドになります）。');
  }

  const adapter = await navigator.gpu?.requestAdapter().catch(() => null);
  const webgpuOption = $('device').querySelector('option[value="webgpu"]');
  if (!adapter) {
    webgpuOption.disabled = true;
    webgpuOption.textContent = 'WebGPU（このブラウザでは利用不可）';
  }
  const requestedDevice = params.get('device');
  $('device').value = requestedDevice === 'webgpu' && adapter ? 'webgpu' : 'wasm';
  hints.push(`CPU論理コア: ${cores}`);
  $('env-hint').textContent = hints.join(' ');

  $('device').addEventListener('change', () => {
    $('threads').disabled = $('device').value !== 'wasm' || !self.crossOriginIsolated;
  });
  $('load-button').addEventListener('click', () => runExclusive(loadModel));
}

async function loadModel() {
  const dtype = $('dtype').value;
  const device = $('device').value;
  const threads = $('threads').value;
  const progress = $('load-progress');

  state.model = null;
  state.cache.clear();
  $('model-facts').replaceChildren();
  updateButtons();
  setStatus('loading', `${DTYPES[dtype].label} を読み込み中…`);
  setProgress(progress, 0, '準備中…');

  client.start();
  try {
    const { info } = await client.request(
      {
        type: 'load',
        dtype,
        device,
        numThreads: device === 'wasm' && threads !== 'auto' ? Number(threads) : undefined,
        useCache: params.get('cache') !== '0',
        useLocal: params.get('local') !== '0',
      },
      ({ loaded, total }) => {
        if (!total) return;
        const label = loaded >= total ? 'モデルを初期化中…' : `${fmtMB(loaded)} / ${fmtMB(total)}`;
        setProgress(progress, loaded / total, label);
      },
    );
    // The first inference compiles kernels and allocates buffers; do it now so that the
    // first search (and every timing shown afterwards) reflects steady-state speed.
    setProgress(progress, 1, '初回推論を準備中…');
    const warmupStarted = performance.now();
    await client.request({ type: 'embed', texts: ['warm-up'], batchSize: 1 });
    info.warmupMs = performance.now() - warmupStarted;

    state.model = info;
    progress.hidden = true;
    setStatus('ready', `準備完了 — ${DTYPES[dtype].label} · ${DEVICE_LABELS[device]}`);
    renderFacts(info);
    $('load-button').textContent = '別の設定で読み込み直す';
  } catch (error) {
    progress.hidden = true;
    client.stop();
    setStatus('error', `読み込みに失敗しました: ${error.message}`);
  }
}

function renderFacts(info) {
  const facts = [
    ['読み込み時間', fmtMs(info.loadMs)],
    ['初回推論（ウォームアップ）', fmtMs(info.warmupMs)],
    ['精度', DTYPES[info.dtype].label],
    ['実行環境', DEVICE_LABELS[info.device]],
    ['WASMスレッド', info.numThreads ?? '–'],
    ['クロスオリジン分離', info.crossOriginIsolated ? '有効' : '無効'],
    ['モデルファイル', info.downloadedBytes ? fmtMB(info.downloadedBytes) : '–'],
    ['Transformers.js', info.transformersVersion],
  ];
  $('model-facts').replaceChildren(...facts.map(([term, value]) => el('div', {}, el('dt', { text: term }), el('dd', { text: String(value) }))));
}

// ---------------------------------------------------------------- tabs

function initTabs() {
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const select = (tab) => {
    for (const other of tabs) {
      const selected = other === tab;
      other.setAttribute('aria-selected', String(selected));
      other.tabIndex = selected ? 0 : -1;
      $(other.getAttribute('aria-controls')).hidden = !selected;
    }
  };
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => select(tab));
    tab.addEventListener('keydown', (event) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (!step) return;
      const next = tabs[(index + step + tabs.length) % tabs.length];
      select(next);
      next.focus();
    });
  });
}

function taskOptions() {
  return [
    ...Object.keys(QUERY_TASKS).map((task) => ({ value: task, label: task })),
    { value: NO_PROMPT, label: 'なし（プロンプトを付けない）' },
  ];
}

function initDims() {
  for (const select of document.querySelectorAll('.dims-select')) {
    fillSelect(select, MRL_DIMS.map((dims) => ({ value: String(dims), label: `${dims}` })), '768');
  }
}

// ---------------------------------------------------------------- search tab

function currentPreset() {
  return SEARCH_PRESETS.find((preset) => preset.id === $('search-preset').value);
}

function applyPreset(preset) {
  $('search-query').value = preset.query;
  $('search-docs').value = preset.docs.join('\n');
  $('search-task').value = preset.task;
  updatePromptPreview();
}

function updatePromptPreview() {
  const task = $('search-task').value;
  const firstDoc = lines($('search-docs').value)[0] ?? '（文書）';
  $('search-prompt-preview').textContent = [
    `クエリ: ${queryPrompt(task)}${$('search-query').value.trim()}`,
    `文書:   ${documentPrompt(null, task !== NO_PROMPT)}${firstDoc}`,
  ].join('\n');
}

function initSearch() {
  fillSelect($('search-preset'), SEARCH_PRESETS.map(({ id, label }) => ({ value: id, label })), SEARCH_PRESETS[0].id);
  fillSelect($('search-task'), taskOptions(), 'search result');
  applyPreset(SEARCH_PRESETS[0]);

  $('search-preset').addEventListener('change', () => applyPreset(currentPreset()));
  for (const id of ['search-query', 'search-docs', 'search-task']) {
    $(id).addEventListener('input', updatePromptPreview);
  }
  $('search-query').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.isComposing && !$('search-button').disabled) $('search-button').click();
  });
  $('search-button').addEventListener('click', () => runExclusive(runSearch));
}

async function runSearch() {
  const query = $('search-query').value.trim();
  const docs = lines($('search-docs').value);
  const task = $('search-task').value;
  const dims = Number($('search-dims').value);
  if (!query || docs.length === 0) {
    $('search-note').textContent = 'クエリと文書を入力してください。';
    return;
  }
  $('search-note').textContent = '計算中…';

  try {
    const started = performance.now();
    const queryEmbedding = await embedCached([queryPrompt(task) + query], { batchSize: 1 });
    const docEmbedding = await embedCached(
      docs.map((doc) => documentPrompt(null, task !== NO_PROMPT) + doc),
      { batchSize: 8, onProgress: ({ done, total }) => ($('search-note').textContent = `文書を埋め込み中… ${done}/${total}`) },
    );
    const elapsed = performance.now() - started;

    const q = truncateAndNormalize(queryEmbedding.vectors, queryEmbedding.dim, dims);
    const d = truncateAndNormalize(docEmbedding.vectors, docEmbedding.dim, dims);
    const scores = similarityMatrix(q, d, dims);
    const order = rankIndices(scores);

    // For an untouched sample, mark the intended answer and (for the model-card example
    // at full size) show the published scores next to ours.
    const preset = currentPreset();
    const untouched = preset && query === preset.query && docs.join('\n') === preset.docs.join('\n');
    const expectedIndex = untouched ? preset.expectedTop ?? null : null;
    const reference = untouched && preset.referenceScores && dims === 768 && task === preset.task ? preset.referenceScores : null;

    state.last.search = {
      query,
      task,
      dims,
      elapsedMs: elapsed,
      expectedIndex,
      expectedRank: expectedIndex === null ? null : order.indexOf(expectedIndex) + 1,
      ranking: order.map((index) => ({ index, text: docs[index], score: scores[index] })),
    };
    renderSearch({ docs, scores, order, dims, elapsed, queryEmbedding, docEmbedding, reference, expectedIndex });
    markRun('search', $('search-results'));
    $('search-note').textContent = '';
  } catch (error) {
    $('search-note').textContent = `エラー: ${error.message}`;
  }
}

function renderSearch({ docs, scores, order, dims, elapsed, queryEmbedding, docEmbedding, reference, expectedIndex }) {
  const computed = queryEmbedding.computed + docEmbedding.computed;
  const cached = docs.length + 1 - computed;
  const meta = [
    `${docs.length}件を順位付け`,
    `${dims}次元`,
    `処理時間 ${fmtMs(elapsed)}`,
    computed > 0 ? `${computed}件を新たに埋め込み` : null,
    cached > 0 ? `${cached}件はキャッシュを再利用` : null,
  ].filter(Boolean);

  const items = order.map((index, rank) =>
    el(
      'li',
      { className: rank === 0 ? 'top' : '', 'data-index': String(index) },
      el('span', { className: 'rank', text: `${rank + 1}` }),
      el(
        'span',
        { className: 'doc' },
        docs[index],
        index === expectedIndex ? el('span', { className: 'badge', text: '想定解' }) : null,
      ),
      scoreBar(scores[index], reference ? `（モデルカード ${fmtScore(reference[index])}）` : null),
    ),
  );

  $('search-results').replaceChildren(
    el('p', { className: 'result-meta', text: meta.join(' · ') }),
    el('ol', { className: `ranking${reference ? ' with-reference' : ''}`, 'aria-label': '検索結果' }, items),
  );
}

/** A thin bar whose length is the cosine score on a 0..1 scale, with the value at its tip. */
function scoreBar(score, note) {
  const bar = el('span', { className: 'score-bar' });
  bar.style.setProperty('--s', String(Math.min(1, Math.max(0, score))));
  return el(
    'span',
    { className: 'score' },
    bar,
    el('span', { className: 'score-value', text: fmtScore(score) }),
    note ? el('span', { className: 'score-value reference', text: note }) : null,
  );
}

// ---------------------------------------------------------------- similarity tab

function initSimilarity() {
  fillSelect($('similarity-task'), taskOptions(), 'sentence similarity');
  $('similarity-sentences').value = SIMILARITY_SENTENCES.join('\n');
  $('similarity-button').addEventListener('click', () => runExclusive(runSimilarity));
}

async function runSimilarity() {
  const sentences = lines($('similarity-sentences').value).slice(0, MAX_SIMILARITY_SENTENCES);
  const task = $('similarity-task').value;
  const dims = Number($('similarity-dims').value);
  if (sentences.length < 2) {
    $('similarity-note').textContent = '2文以上入力してください。';
    return;
  }
  $('similarity-note').textContent = '計算中…';

  try {
    const started = performance.now();
    const embedding = await embedCached(
      sentences.map((sentence) => queryPrompt(task) + sentence),
      { batchSize: 8, onProgress: ({ done, total }) => ($('similarity-note').textContent = `埋め込み中… ${done}/${total}`) },
    );
    const elapsed = performance.now() - started;
    const vectors = truncateAndNormalize(embedding.vectors, embedding.dim, dims);
    const matrix = similarityMatrix(vectors, vectors, dims);

    state.last.similarity = { sentences, task, dims, matrix: Array.from(matrix) };
    renderHeatmap(sentences, matrix, dims, elapsed);
    markRun('similarity', $('similarity-results'));
    $('similarity-note').textContent = '';
  } catch (error) {
    $('similarity-note').textContent = `エラー: ${error.message}`;
  }
}

function renderHeatmap(sentences, matrix, dims, elapsed) {
  const n = sentences.length;
  const low = Math.min(...matrix);
  const span = Math.max(1e-6, 1 - low);
  const tooltip = $('tooltip');

  const header = el('tr', {}, el('th', { scope: 'col', text: '' }), sentences.map((_, j) => el('th', { scope: 'col', text: `${j + 1}` })));
  const rows = sentences.map((sentence, i) =>
    el(
      'tr',
      {},
      el('th', { scope: 'row', title: sentence }, el('span', { className: 'index', text: `${i + 1}` }), sentence),
      sentences.map((_, j) => {
        const score = matrix[i * n + j];
        const p = Math.min(1, Math.max(0, (score - low) / span));
        const cell = el('td', {
          text: score.toFixed(2),
          className: p > 0.58 ? 'strong' : '',
          'data-i': String(i),
          'data-j': String(j),
        });
        cell.style.setProperty('--p', `${(p * 100).toFixed(1)}%`);
        return cell;
      }),
    ),
  );

  const table = el('table', { className: 'heatmap', 'aria-label': 'コサイン類似度マトリクス' }, el('thead', {}, header), el('tbody', {}, rows));

  table.addEventListener('pointermove', (event) => {
    const cell = event.target.closest('td');
    if (!cell) {
      tooltip.hidden = true;
      return;
    }
    const i = Number(cell.dataset.i);
    const j = Number(cell.dataset.j);
    tooltip.replaceChildren(
      el('strong', { text: matrix[i * n + j].toFixed(4) }),
      el('span', { text: `${i + 1}. ${sentences[i]}` }),
      el('span', { text: `${j + 1}. ${sentences[j]}` }),
    );
    tooltip.hidden = false;
    const x = Math.min(event.clientX + 14, window.innerWidth - tooltip.offsetWidth - 8);
    const y = Math.min(event.clientY + 14, window.innerHeight - tooltip.offsetHeight - 8);
    tooltip.style.left = `${Math.max(8, x)}px`;
    tooltip.style.top = `${Math.max(8, y)}px`;
  });
  table.addEventListener('pointerleave', () => (tooltip.hidden = true));

  $('similarity-results').replaceChildren(
    el('p', { className: 'result-meta', text: `${n}文 · ${dims}次元 · 処理時間 ${fmtMs(elapsed)}` }),
    el('div', { className: 'heatmap-scroll' }, table),
    el(
      'div',
      { className: 'scale', 'aria-hidden': 'true' },
      el('span', { text: low.toFixed(2) }),
      el('span', { className: 'scale-ramp' }),
      el('span', { text: '1.00' }),
    ),
  );
}

// ---------------------------------------------------------------- benchmark tab

async function initBenchmark() {
  const response = await fetch('data/benchmark.json');
  state.dataset = await response.json();
  const { docs, queries } = state.dataset;
  const docLang = new Map(docs.map((doc) => [doc.id, doc.lang]));
  const counts = {};
  for (const query of queries) {
    const type = `${query.lang}→${docLang.get(query.relevant[0])}`;
    counts[type] = (counts[type] ?? 0) + 1;
  }
  const breakdown = Object.entries(counts)
    .map(([type, count]) => `${TYPE_LABELS[type] ?? type} ${count}`)
    .join('、');
  $('benchmark-intro').textContent =
    `日英の小さな検索データセット（文書${docs.length}件・クエリ${queries.length}件：${breakdown}）で、正解文書を何位に出せるかと処理速度を測ります。` +
    '各トピックには「同じ話題だが答えではない」紛らわしい文書を入れてあります。プロンプトの有無と次元（768/512/256/128）を切り替えた結果も比較します。';

  $('benchmark-button').addEventListener('click', () => runExclusive(runBenchmark));
  $('benchmark-download').addEventListener('click', () => {
    const result = state.last.benchmark;
    if (result) download(`embeddinggemma-benchmark-${result.model.dtype}-${result.model.device}.json`, result);
  });
}

async function runBenchmark() {
  const { docs, queries } = state.dataset;
  const batchSize = Number($('benchmark-batch').value);
  const progress = $('benchmark-progress');
  $('benchmark-note').textContent = '';
  $('benchmark-download').hidden = true;

  // Each step: [label, texts, batch size]. Queries are embedded one at a time to measure search latency.
  const steps = [
    ['文書（プロンプトあり）', docs.map((doc) => documentPrompt(null) + doc.text), batchSize],
    ['クエリ（プロンプトあり）', queries.map((query) => queryPrompt('search result') + query.text), 1],
    ['文書（プロンプトなし）', docs.map((doc) => doc.text), batchSize],
    ['クエリ（プロンプトなし）', queries.map((query) => query.text), batchSize],
  ];
  const totalTexts = steps.reduce((sum, [, texts]) => sum + texts.length, 0);

  try {
    const outputs = [];
    let finished = 0;
    for (const [label, texts, size] of steps) {
      const output = await client.request({ type: 'embed', texts, batchSize: size }, ({ done }) =>
        setProgress(progress, (finished + done) / totalTexts, `${label} ${done}/${texts.length}`),
      );
      finished += texts.length;
      outputs.push(output);
    }
    const [docsPrompted, queriesPrompted, docsRaw, queriesRaw] = outputs;

    const evaluations = [];
    for (const [mode, queryOut, docOut] of [
      ['prompt', queriesPrompted, docsPrompted],
      ['raw', queriesRaw, docsRaw],
    ]) {
      for (const dims of MRL_DIMS) {
        evaluations.push({
          mode,
          dims,
          ...evaluateRetrieval({
            queries,
            docs,
            queryVectors: truncateAndNormalize(queryOut.vectors, queryOut.dim, dims),
            docVectors: truncateAndNormalize(docOut.vectors, docOut.dim, dims),
            dims,
          }),
        });
      }
    }

    const docTokens = docsPrompted.tokens.reduce((sum, count) => sum + count, 0);
    const queryLatencies = queriesPrompted.timings.batchMs;
    const speed = {
      documents: {
        count: docs.length,
        batchSize,
        totalMs: docsPrompted.timings.totalMs,
        msPerText: docsPrompted.timings.totalMs / docs.length,
        textsPerSecond: (docs.length / docsPrompted.timings.totalMs) * 1000,
        tokens: docTokens,
        tokensPerSecond: (docTokens / docsPrompted.timings.totalMs) * 1000,
        averageTokens: docTokens / docs.length,
      },
      queries: {
        count: queries.length,
        p50Ms: percentile(queryLatencies, 50),
        p95Ms: percentile(queryLatencies, 95),
        meanMs: mean(queryLatencies),
        averageTokens: mean(queriesPrompted.tokens),
      },
    };

    const result = {
      createdAt: new Date().toISOString(),
      model: { ...state.model, id: MODEL_ID },
      environment: {
        userAgent: navigator.userAgent,
        hardwareConcurrency: navigator.hardwareConcurrency,
        crossOriginIsolated: self.crossOriginIsolated,
      },
      dataset: { name: state.dataset.name, version: state.dataset.version, docs: docs.length, queries: queries.length },
      speed,
      evaluations: evaluations.map(({ perQuery, ...summary }) => summary),
      perQuery: evaluations.find((evaluation) => evaluation.mode === 'prompt' && evaluation.dims === 768).perQuery,
    };
    state.last.benchmark = result;
    progress.hidden = true;
    renderBenchmark(result, evaluations);
    markRun('benchmark', $('benchmark-results'));
    $('benchmark-download').hidden = false;
  } catch (error) {
    progress.hidden = true;
    $('benchmark-note').textContent = `エラー: ${error.message}`;
  }
}

function tile(label, value, note) {
  return el(
    'div',
    { className: 'tile' },
    el('div', { className: 'tile-label', text: label }),
    el('div', { className: 'tile-value', text: value }),
    note ? el('div', { className: 'tile-note', text: note }) : null,
  );
}

function dataTable(caption, headers, rows) {
  return el(
    'div',
    { className: 'table-scroll' },
    el(
      'table',
      { className: 'data', 'aria-label': caption },
      el('thead', {}, el('tr', {}, headers.map((header) => el('th', { scope: 'col', text: header })))),
      el('tbody', {}, rows),
    ),
  );
}

function renderBenchmark(result, evaluations) {
  const docsById = new Map(state.dataset.docs.map((doc) => [doc.id, doc]));
  const find = (mode, dims) => evaluations.find((evaluation) => evaluation.mode === mode && evaluation.dims === dims);
  const main = find('prompt', 768);
  const raw = find('raw', 768);
  const { documents, queries } = result.speed;

  const tiles = el(
    'div',
    { className: 'tiles' },
    tile('Top-1 正解率', fmtPct(main.overall.acc1), `${Math.round(main.overall.acc1 * main.overall.count)}/${main.overall.count}問 · 768次元`),
    tile('MRR@10', fmtScore(main.overall.mrr10), `プロンプトなし ${fmtScore(raw.overall.mrr10)}`),
    tile('nDCG@10', fmtScore(main.overall.ndcg10)),
    tile('文書の埋め込み', `${Math.round(documents.msPerText)} ms/件`, `${documents.textsPerSecond.toFixed(1)}件/秒 · バッチ${documents.batchSize}`),
    tile('検索クエリの遅延', `${Math.round(queries.p50Ms)} ms`, `中央値 · p95 ${Math.round(queries.p95Ms)} ms`),
  );

  const dimsRows = evaluations.map((evaluation) =>
    el(
      'tr',
      { className: evaluation.mode === 'prompt' && evaluation.dims === 768 ? 'highlight' : '' },
      el('td', { text: evaluation.mode === 'prompt' ? 'あり' : 'なし' }),
      el('td', { text: String(evaluation.dims) }),
      el('td', { text: fmtPct(evaluation.overall.acc1) }),
      el('td', { text: fmtPct(evaluation.overall.recall5) }),
      el('td', { text: fmtScore(evaluation.overall.mrr10) }),
      el('td', { text: fmtScore(evaluation.overall.ndcg10) }),
    ),
  );

  const typeRows = Object.entries(main.byType).map(([type, metrics]) =>
    el(
      'tr',
      {},
      el('td', { text: `${TYPE_LABELS[type] ?? type}（${type}）` }),
      el('td', { text: String(metrics.count) }),
      el('td', { text: fmtPct(metrics.acc1) }),
      el('td', { text: fmtScore(metrics.mrr10) }),
      el('td', { text: fmtPct(raw.byType[type]?.acc1) }),
      el('td', { text: fmtScore(raw.byType[type]?.mrr10) }),
    ),
  );

  const speedRows = [
    el(
      'tr',
      {},
      el('td', { text: `文書 ${documents.count}件（バッチ${documents.batchSize}）` }),
      el('td', { text: fmtMs(documents.totalMs) }),
      el('td', { text: `${Math.round(documents.msPerText)} ms` }),
      el('td', { text: documents.textsPerSecond.toFixed(1) }),
      el('td', { text: Math.round(documents.tokensPerSecond).toLocaleString() }),
      el('td', { text: documents.averageTokens.toFixed(1) }),
    ),
    el(
      'tr',
      {},
      el('td', { text: `クエリ ${queries.count}件（1件ずつ）` }),
      el('td', { text: `p50 ${Math.round(queries.p50Ms)} ms` }),
      el('td', { text: `${Math.round(queries.meanMs)} ms` }),
      el('td', { text: (1000 / queries.meanMs).toFixed(1) }),
      el('td', { text: '–' }),
      el('td', { text: queries.averageTokens.toFixed(1) }),
    ),
  ];

  const misses = main.perQuery.filter((row) => row.rank !== 1);
  const missItems = misses.map((row) => {
    const expected = docsById.get(row.relevant[0]);
    const top = docsById.get(row.top[0].id);
    return el(
      'li',
      {},
      el('span', { className: 'q', text: row.text }),
      el('span', { className: 'muted', text: ` [${TYPE_LABELS[row.type] ?? row.type}]` }),
      el('span', {
        className: 'detail',
        text: `正解（${Number.isFinite(row.rank) ? `${row.rank}位` : '圏外'}・${fmtScore(row.relevantScore)}）: ${expected.text}`,
      }),
      el('span', { className: 'detail', text: `1位（${fmtScore(row.top[0].score)}）: ${top.text}` }),
    );
  });

  $('benchmark-results').replaceChildren(
    el(
      'p',
      { className: 'result-meta' },
      `${DTYPES[result.model.dtype].label} · ${DEVICE_LABELS[result.model.device]}` +
        (result.model.numThreads ? ` · ${result.model.numThreads}スレッド` : '') +
        ` · 文書${result.dataset.docs}件 / クエリ${result.dataset.queries}件`,
    ),
    tiles,
    el('h3', { text: 'プロンプトと次元（MRL）の影響' }),
    dataTable('プロンプトと次元の影響', ['プロンプト', '次元', 'Top-1', 'Recall@5', 'MRR@10', 'nDCG@10'], dimsRows),
    el('h3', { text: '言語ペア別（768次元）' }),
    dataTable('言語ペア別', ['クエリ→文書', '問数', 'Top-1', 'MRR@10', 'Top-1（プロンプトなし）', 'MRR@10（プロンプトなし）'], typeRows),
    el('h3', { text: '速度' }),
    dataTable('速度', ['対象', '合計 / 中央値', '1件あたり', '件/秒', 'トークン/秒', '平均トークン'], speedRows),
    el('h3', { text: `1位を外したクエリ（${misses.length}件・768次元・プロンプトあり）` }),
    misses.length > 0 ? el('ol', { className: 'miss-list' }, missItems) : el('p', { className: 'muted', text: 'すべて1位で正解しました。' }),
  );
}

// ---------------------------------------------------------------- boot

initTabs();
initDims();
initSearch();
initSimilarity();
initCharGen({ $, el, embedCached, runExclusive, fmtMs });
await Promise.all([initModelPanel(), initBenchmark()]);
updateButtons();
document.body.dataset.ready = 'true';
if (params.get('autoload') === '1') runExclusive(loadModel);
