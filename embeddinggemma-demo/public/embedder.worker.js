// Runs EmbeddingGemma inside a module worker so the page stays responsive while
// onnxruntime-web is busy. Protocol (request -> response, matched by `id`):
//   { type: 'load', model, dtype, device, numThreads, useCache, useLocal } -> { type: 'loaded', info }
//   { type: 'embed', texts, batchSize }                             -> { type: 'embedded', vectors, dim, tokens, timings }
// Progress messages ({ type: 'progress', ... }) are sent while a request runs.
import { MODELS, TRANSFORMERS_URL, textOnlyConfig } from './lib/model-config.js';

const transformersReady = import(TRANSFORMERS_URL);

let tokenizer = null;
let model = null;

async function load({ model: modelKey, dtype, device, useCache, useLocal, numThreads }, report) {
  const { AutoConfig, AutoModel, AutoTokenizer, env } = await transformersReady;
  const { id, revision, dtypes } = MODELS[modelKey];
  // onnxruntime-web defaults to min(4, ceil(cores / 2)) threads; it only takes effect before the first session.
  if (numThreads) env.backends.onnx.wasm.numThreads = numThreads;
  env.useBrowserCache = useCache;
  env.allowLocalModels = useLocal;
  env.localModelPath = '/models/';

  // Aggregate per-file download progress into one number for the UI. `progress_total`
  // events list every model file up front, so the total does not grow mid-download.
  const files = new Map();
  const progress_callback = (event) => {
    if (event.status === 'progress_total') {
      for (const [file, state] of Object.entries(event.files)) files.set(file, state);
    } else if (event.status === 'progress' && event.file && event.total) {
      files.set(event.file, { loaded: event.loaded, total: event.total });
    } else {
      return;
    }
    let loaded = 0;
    let total = 0;
    for (const file of files.values()) {
      loaded += file.loaded;
      total += file.total;
    }
    report({ phase: 'download', file: event.file, loaded, total });
  };

  const started = performance.now();
  tokenizer = await AutoTokenizer.from_pretrained(id, { revision, progress_callback });
  model = await AutoModel.from_pretrained(id, {
    revision,
    config: await textOnlyConfig(AutoConfig, modelKey),
    dtype,
    device,
    model_file_name: dtypes[dtype].modelFileName,
    progress_callback,
  });
  const loadMs = performance.now() - started;

  const wasm = env.backends.onnx?.wasm ?? {};
  return {
    model: modelKey,
    modelId: id,
    revision,
    dtype,
    device,
    loadMs,
    transformersVersion: env.version,
    crossOriginIsolated: self.crossOriginIsolated,
    numThreads: device === 'wasm' ? wasm.numThreads ?? null : null,
    downloadedBytes: [...files.values()].reduce((sum, file) => sum + file.total, 0),
  };
}

async function embed({ texts, batchSize = 8 }, report) {
  if (!model) throw new Error('Model is not loaded yet.');
  if (texts.length === 0) throw new Error('Nothing to embed.');
  let dim = 0;
  let vectors = null;
  const tokens = new Array(texts.length);
  const timings = { tokenizeMs: 0, inferenceMs: 0, totalMs: 0, batchMs: [] };
  const started = performance.now();

  for (let start = 0; start < texts.length; start += batchSize) {
    const batch = texts.slice(start, start + batchSize);

    const t0 = performance.now();
    const inputs = await tokenizer(batch, { padding: true, truncation: true });
    const t1 = performance.now();
    const { sentence_embedding } = await model(inputs);
    const t2 = performance.now();
    timings.tokenizeMs += t1 - t0;
    timings.inferenceMs += t2 - t1;
    timings.batchMs.push(t2 - t0);

    const [rows, cols] = sentence_embedding.dims;
    if (!vectors) {
      dim = cols;
      vectors = new Float32Array(texts.length * dim);
    }
    vectors.set(sentence_embedding.data, start * dim);

    const mask = inputs.attention_mask.tolist();
    for (let row = 0; row < rows; row++) {
      tokens[start + row] = mask[row].reduce((sum, value) => sum + Number(value), 0);
    }
    report({ phase: 'embed', done: Math.min(start + batchSize, texts.length), total: texts.length });
  }

  timings.totalMs = performance.now() - started;
  return { vectors, dim, tokens, timings };
}

self.addEventListener('message', async ({ data }) => {
  const { id, type } = data;
  const report = (progress) => self.postMessage({ id, type: 'progress', ...progress });
  try {
    if (type === 'load') {
      const info = await load(data, report);
      self.postMessage({ id, type: 'loaded', info });
    } else if (type === 'embed') {
      const result = await embed(data, report);
      self.postMessage({ id, type: 'embedded', ...result }, [result.vectors.buffer]);
    } else {
      throw new Error(`Unknown request type: ${type}`);
    }
  } catch (error) {
    self.postMessage({ id, type: 'error', message: error?.message ?? String(error) });
  }
});
