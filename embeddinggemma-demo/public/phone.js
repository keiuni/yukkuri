// Phone check page: reports what this browser offers (WebGPU, f16, buffer limits, memory), rates
// each candidate model against it, and runs a model on demand in a fresh worker.
// scripts/phone-check.mjs drives the same page through `window.phoneCheck`.
import {
  GENERATION_MAX_TOKENS,
  GENERATION_PROMPT,
  GENERATION_TOKEN_CHOICES,
  LITERT_GEMMA4,
  PHONE_MODELS,
  assessLiteRt,
  assessVariant,
  formatMB as formatWholeMB,
  variantKey,
} from './lib/phone-models.js';

const $ = (id) => document.getElementById(id);
const LEVEL_LABEL = { yes: '動きそう', maybe: '微妙', no: '動かない見込み' };
const DEVICE_LABEL = { wasm: 'WASM（CPU）', webgpu: 'WebGPU（GPU）' };
const results = [];
// Model files come from this server's /models/ when it has them (npm start), otherwise from the Hub.
const useLocal = ['127.0.0.1', 'localhost'].includes(location.hostname);
const params = new URLSearchParams(location.search);
// ?tokens=16 shortens the generation test (scripts/phone-check.mjs uses it on the slow software GPU).
const initialTokens = Number(params.get('tokens')) || GENERATION_MAX_TOKENS;
// ?cache=0 skips the browser's Cache Storage copy of the model files (to measure its cost).
const useCache = params.get('cache') !== '0';

const formatMB = (bytes) => formatWholeMB(bytes / 1e6);
const formatSize = (mb) => (mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb} MB`);

/** Models run in a worker, so WebGPU has to be there too (true / false, or null when unknown). */
async function workerHasWebGpu() {
  const code = 'navigator.gpu ? navigator.gpu.requestAdapter().then((a) => postMessage(Boolean(a)), () => postMessage(false)) : postMessage(false);';
  const url = URL.createObjectURL(new Blob([code], { type: 'text/javascript' }));
  try {
    const worker = new Worker(url);
    return await new Promise((resolve) => {
      const finish = (value) => {
        clearTimeout(timer);
        worker.terminate();
        resolve(value);
      };
      const timer = setTimeout(() => finish(null), 5000);
      worker.onmessage = ({ data }) => finish(data);
      worker.onerror = () => finish(null);
    });
  } catch {
    return null;
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function detectDevice() {
  const ua = navigator.userAgent;
  const device = {
    userAgent: ua,
    mobile: navigator.userAgentData?.mobile ?? /Android|iPhone|iPad|iPod/i.test(ua),
    threads: navigator.hardwareConcurrency ?? null,
    deviceMemoryGB: navigator.deviceMemory ?? null,
    crossOriginIsolated: self.crossOriginIsolated,
    webgpu: null,
    webgpuNote: null,
    storage: null,
  };
  if (!navigator.gpu) {
    device.webgpuNote = 'このブラウザには WebGPU がない';
  } else {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        device.webgpuNote = 'GPU アダプターが見つからない';
      } else {
        const info = adapter.info ?? {};
        device.webgpu = {
          name: [info.vendor, info.architecture, info.description].filter(Boolean).join(' / ') || '不明',
          f16: adapter.features.has('shader-f16'),
          maxBufferSize: adapter.limits.maxBufferSize,
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        };
      }
    } catch (error) {
      device.webgpuNote = `WebGPU の初期化に失敗: ${error.message}`;
    }
    if (device.webgpu && (await workerHasWebGpu()) === false) {
      device.webgpu = null;
      device.webgpuNote = '画面側では使えるが、モデルを動かすワーカーの中では使えない';
    }
  }
  try {
    const estimate = await navigator.storage?.estimate?.();
    if (estimate) device.storage = { quota: estimate.quota, usage: estimate.usage };
  } catch {
    // Storage estimates are optional.
  }
  return device;
}

function browserName(ua) {
  const match =
    ua.match(/(Edg|EdgA|SamsungBrowser|CriOS|FxiOS|Firefox|OPR)\/([\d.]+)/) ??
    ua.match(/(Chrome)\/([\d.]+)/) ??
    ua.match(/Version\/([\d.]+).*(Safari)/);
  if (!match) return ua.slice(0, 40);
  return match[2] === 'Safari' ? `Safari ${match[1]}` : `${match[1]} ${match[2].split('.')[0]}`;
}

function renderDevice(device) {
  const gpu = device.webgpu;
  const facts = [
    ['ブラウザ', `${browserName(device.userAgent)}${device.mobile ? '（モバイル）' : ''}`],
    ['CPU スレッド', device.threads ?? '不明'],
    ['端末メモリ', device.deviceMemoryGB ? `約 ${device.deviceMemoryGB} GB` : '不明（Safari などは非公開）'],
    ['WebGPU', gpu ? `あり（${gpu.name}）` : `なし（${device.webgpuNote}）`],
    ['f16（shader-f16）', gpu ? (gpu.f16 ? '対応' : '非対応') : '—'],
    ['GPU の 1 バッファの上限', gpu ? formatMB(Math.min(gpu.maxBufferSize, gpu.maxStorageBufferBindingSize)) : '—'],
    ['保存できる容量', device.storage ? formatMB(device.storage.quota - device.storage.usage) : '不明'],
    ['WASM のマルチスレッド', device.crossOriginIsolated ? '使える' : '使えない（1 スレッド）'],
  ];
  $('device-facts').replaceChildren(
    ...facts.map(([term, value]) => {
      const div = document.createElement('div');
      const dt = document.createElement('dt');
      const dd = document.createElement('dd');
      dt.textContent = term;
      dd.textContent = String(value);
      div.append(dt, dd);
      return div;
    }),
  );
  const status = $('device-status');
  status.dataset.state = 'ready';
  status.textContent = gpu ? 'WebGPU が使えます' : 'WebGPU は使えません（CPU で試せます）';
}

function verdictLine(assessment) {
  const p = document.createElement('p');
  p.className = 'verdict';
  const chip = document.createElement('span');
  chip.className = 'chip';
  chip.dataset.level = assessment.level;
  chip.textContent = LEVEL_LABEL[assessment.level];
  const reasons = document.createElement('span');
  reasons.textContent = assessment.reasons.join('。');
  p.append(chip, reasons);
  return p;
}

function renderModels(device) {
  const rows = [];
  for (const model of PHONE_MODELS) {
    for (const variant of model.variants) {
      const key = variantKey(model, variant);
      const assessment = assessVariant(variant, device);
      const li = document.createElement('li');
      li.className = 'model-row';
      li.dataset.key = key;
      li.dataset.level = assessment.level;

      const head = document.createElement('div');
      const name = document.createElement('p');
      name.className = 'model-name';
      name.textContent = `${model.label}（${variant.dtype}）`;
      const meta = document.createElement('p');
      meta.className = 'model-meta';
      meta.textContent = `${DEVICE_LABEL[variant.device]}・${formatSize(variant.sizeMB)}・${model.task === 'embedding' ? '埋め込み' : '文章生成'}`;
      head.append(name, meta);

      const button = document.createElement('button');
      button.type = 'button';
      button.className = `button${assessment.level === 'yes' ? ' primary' : ''}`;
      button.textContent = '試す';
      button.setAttribute('aria-label', `${model.label}（${variant.dtype}・${variant.device === 'wasm' ? 'WASM' : 'WebGPU'}）を試す`);
      button.addEventListener('click', () => runTest(key));

      const progress = document.createElement('div');
      progress.className = 'progress';
      progress.hidden = true;
      progress.innerHTML = '<div class="progress-track"><div class="progress-fill"></div></div><span class="progress-label"></span>';

      li.append(head, button, verdictLine(assessment), progress);
      rows.push(li);
    }
  }

  const liteRt = document.createElement('li');
  liteRt.className = 'model-row';
  const head = document.createElement('div');
  head.innerHTML = '<p class="model-name"></p><p class="model-meta"></p>';
  head.firstChild.textContent = LITERT_GEMMA4.label;
  head.lastChild.textContent = `Google の公式デモ・${formatSize(LITERT_GEMMA4.sizeMB)}・文章生成（別のページで開きます）`;
  const link = document.createElement('a');
  link.className = 'button';
  link.href = LITERT_GEMMA4.url;
  link.target = '_blank';
  link.rel = 'noopener';
  link.textContent = '開く';
  link.setAttribute('aria-label', `${LITERT_GEMMA4.label}の公式デモを開く`);
  liteRt.append(head, link, verdictLine(assessLiteRt(device)));
  rows.push(liteRt);

  $('model-list').replaceChildren(...rows);
  $('prompt-note').textContent = '文章生成のモデルは、上の「聞くこと」に答えます（毎回同じ答えになる貪欲法）。読み込み時間・最初の 1 トークンまでの時間・1 秒あたりのトークン数も測ります。';
}

function initPrompt() {
  $('prompt').value = GENERATION_PROMPT;
  const choices = [...new Set([...GENERATION_TOKEN_CHOICES, initialTokens])].sort((a, b) => a - b);
  $('max-tokens').replaceChildren(
    ...choices.map((tokens) => Object.assign(document.createElement('option'), { value: String(tokens), textContent: `${tokens} トークン` })),
  );
  $('max-tokens').value = String(initialTokens);
}

function setRunning(key, running) {
  for (const button of document.querySelectorAll('.model-row button')) button.disabled = running;
  const row = document.querySelector(`.model-row[data-key="${key}"]`);
  if (row) row.querySelector('.progress').hidden = !running;
}

function updateProgress(key, message) {
  const row = document.querySelector(`.model-row[data-key="${key}"]`);
  if (!row) return;
  const fill = row.querySelector('.progress-fill');
  const label = row.querySelector('.progress-label');
  if (message.phase === 'download' && message.total) {
    fill.style.width = `${(100 * message.loaded) / message.total}%`;
    label.textContent = `読み込み中 ${formatMB(message.loaded)} / ${formatMB(message.total)}`;
  } else if (message.phase === 'generate') {
    fill.style.width = '100%';
    label.textContent = message.text ? `生成中: ${message.text.slice(-24)}` : '生成を始めています…';
  } else if (message.phase === 'embed') {
    fill.style.width = '100%';
    label.textContent = '埋め込みを計算中…';
  }
}

function renderResult(result) {
  $('results-empty').hidden = true;
  $('copy-results').disabled = false;
  const li = document.createElement('li');
  li.className = 'result-item';
  li.dataset.ok = String(result.ok);
  const title = document.createElement('h3');
  title.textContent = `${result.ok ? '✓' : '✗'} ${result.label}（${result.dtype}・${DEVICE_LABEL[result.device]}）`;
  const numbers = document.createElement('p');
  numbers.className = 'numbers';
  const items = [];
  if (result.ok) {
    const download = result.downloadMs ? `（うちダウンロード ${(result.downloadMs / 1000).toFixed(1)} 秒）` : '';
    items.push(['読み込み', `${(result.loadMs / 1000).toFixed(1)} 秒${download}`]);
    if (result.msPerText !== undefined) {
      items.push(['1 文の埋め込み', `${result.msPerText.toFixed(0)} ms`]);
    } else {
      items.push(['最初の 1 トークン', `${(result.ttftMs / 1000).toFixed(2)} 秒`]);
      items.push(['生成速度', result.decodeTokPerSec ? `${result.decodeTokPerSec.toFixed(1)} トークン/秒` : '—']);
    }
    if (result.downloadedBytes) items.push(['ファイル', formatMB(result.downloadedBytes)]);
  } else {
    items.push(['経過', `${(result.elapsedMs / 1000).toFixed(1)} 秒`]);
  }
  numbers.innerHTML = items.map(([k, v]) => `<span>${k} <strong>${v}</strong></span>`).join('');
  const output = document.createElement('p');
  output.className = 'output';
  output.textContent = result.ok ? result.text : result.error;
  li.append(title, numbers);
  if (result.prompt) {
    const asked = document.createElement('p');
    asked.className = 'muted small';
    asked.textContent = `聞いたこと: ${result.prompt}`;
    li.append(asked);
  }
  li.append(output);
  $('result-list').prepend(li);
}

/** Runs one variant in a new worker; resolves with the result (never rejects). */
function runTest(key) {
  const [modelKey, dtype, device] = key.split(':');
  const model = PHONE_MODELS.find((m) => m.key === modelKey);
  const prompt = model.task === 'embedding' ? null : $('prompt').value.trim() || GENERATION_PROMPT;
  const maxTokens = Number($('max-tokens').value) || GENERATION_MAX_TOKENS;
  setRunning(key, true);
  updateProgress(key, { phase: 'download', loaded: 0, total: 1 });
  const started = performance.now();
  return new Promise((resolve) => {
    const worker = new Worker(new URL('./phone.worker.js', import.meta.url), { type: 'module' });
    const finish = (outcome) => {
      worker.terminate();
      const result = { key, label: model.label, dtype, device, prompt, elapsedMs: performance.now() - started, ...outcome };
      results.push(result);
      renderResult(result);
      setRunning(key, false);
      resolve(result);
    };
    worker.addEventListener('message', ({ data }) => {
      if (data.type === 'progress') updateProgress(key, data);
      else if (data.type === 'done') finish({ ok: true, ...data.result });
      else if (data.type === 'error') finish({ ok: false, error: data.message });
    });
    // A worker that runs out of memory usually dies without a message.
    worker.addEventListener('error', (event) => finish({ ok: false, error: event.message || 'ワーカーが停止しました（メモリ不足の可能性）' }));
    worker.postMessage({ type: 'run', key, useLocal, useCache, maxTokens, prompt });
  });
}

$('copy-results').addEventListener('click', async () => {
  const text = JSON.stringify({ device: window.phoneCheck.device, results }, null, 2);
  try {
    await navigator.clipboard.writeText(text);
    $('copy-results').textContent = 'コピーしました';
  } catch {
    $('copy-results').textContent = 'コピーできませんでした';
  }
  setTimeout(() => {
    $('copy-results').textContent = '結果をコピー';
  }, 2000);
});

initPrompt();
const ready = detectDevice().then((device) => {
  window.phoneCheck.device = device;
  renderDevice(device);
  renderModels(device);
  return device;
});

window.phoneCheck = { ready, device: null, results, run: runTest };
