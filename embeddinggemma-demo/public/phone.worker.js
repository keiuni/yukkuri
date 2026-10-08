// Runs one test of the phone check page. The page starts a new worker for every test and
// terminates it afterwards, so the memory a model used is given back before the next one.
// Protocol: { type: 'run', key, useLocal } -> { type: 'progress', ... }* -> { type: 'done', result } | { type: 'error', message }
import { TRANSFORMERS_URL, documentPrompt, queryPrompt, textOnlyConfig } from './lib/model-config.js';
import { GENERATION_MAX_TOKENS, GENERATION_PROMPT, findVariant } from './lib/phone-models.js';

const EMBED_QUERY = '猫に食べさせてはいけないものは？';
const EMBED_DOCS = [
  '玉ねぎやニラに含まれる成分は、猫の赤血球を壊して貧血を起こすことがある。',
  '猫は一日の大半を寝て過ごし、成猫でも12時間以上眠ることが多い。',
  '犬の散歩は朝夕の涼しい時間帯に行うのがよい。',
  'スマートフォンの電池を長持ちさせるには、画面の明るさを下げるとよい。',
  '東京タワーの高さは333メートルで、1958年に完成した。',
  'チョコレートに含まれるテオブロミンは、犬や猫にとって有害である。',
  '観葉植物のユリは、猫が花粉をなめるだけでも腎臓に障害が出ることがある。',
  '日本の首都は東京で、人口は約1400万人である。',
];

function progressTracker(report) {
  const files = new Map();
  let lastProgressAt = null;
  const callback = (event) => {
    if (event.status === 'progress' || event.status === 'done') lastProgressAt = performance.now();
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
    report({ phase: 'download', loaded, total });
  };
  return {
    callback,
    totalBytes: () => [...files.values()].reduce((sum, file) => sum + file.total, 0),
    // Time until the last file finished arriving (from the network or the browser cache).
    downloadMs: (started) => (lastProgressAt === null ? null : lastProgressAt - started),
  };
}

async function runGeneration(lib, model, variant, progress, report, maxTokens) {
  const { AutoModelForCausalLM, AutoTokenizer, TextStreamer } = lib;
  const options = { revision: model.revision, progress_callback: progress.callback };
  const started = performance.now();
  const tokenizer = await AutoTokenizer.from_pretrained(model.id, options);
  const lm = await AutoModelForCausalLM.from_pretrained(model.id, { ...options, dtype: variant.dtype, device: variant.device });
  const loadMs = performance.now() - started;
  const downloadMs = progress.downloadMs(started);
  report({ phase: 'generate', text: '' });

  const inputs = tokenizer.apply_chat_template([{ role: 'user', content: GENERATION_PROMPT }], {
    add_generation_prompt: true,
    return_dict: true,
    enable_thinking: false,
  });
  const tokenTimes = [];
  let streamed = '';
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    token_callback_function: () => tokenTimes.push(performance.now()),
    callback_function: (text) => {
      streamed += text;
      report({ phase: 'generate', text: streamed });
    },
  });
  const generateStarted = performance.now();
  const output = await lm.generate({ ...inputs, max_new_tokens: maxTokens, do_sample: false, streamer });
  const promptTokens = inputs.input_ids.dims.at(-1);
  const text = tokenizer.batch_decode(output.slice(null, [promptTokens, null]), { skip_special_tokens: true })[0];
  await lm.dispose();

  const decodeSeconds = tokenTimes.length > 1 ? (tokenTimes.at(-1) - tokenTimes[0]) / 1000 : null;
  return {
    loadMs,
    downloadMs,
    promptTokens,
    newTokens: tokenTimes.length,
    ttftMs: tokenTimes.length ? tokenTimes[0] - generateStarted : null,
    decodeTokPerSec: decodeSeconds ? (tokenTimes.length - 1) / decodeSeconds : null,
    text: text.trim(),
  };
}

async function runEmbedding(lib, model, variant, progress, report) {
  const { AutoConfig, AutoModel, AutoTokenizer } = lib;
  const options = { revision: model.revision, progress_callback: progress.callback };
  const started = performance.now();
  const tokenizer = await AutoTokenizer.from_pretrained(model.id, options);
  const encoder = await AutoModel.from_pretrained(model.id, {
    ...options,
    config: await textOnlyConfig(AutoConfig, model.embedding),
    dtype: variant.dtype,
    device: variant.device,
    model_file_name: variant.fileName ?? 'model',
  });
  const loadMs = performance.now() - started;
  const downloadMs = progress.downloadMs(started);
  report({ phase: 'embed' });

  const embed = async (texts) => {
    const inputs = await tokenizer(texts, { padding: true, truncation: true });
    const { sentence_embedding } = await encoder(inputs);
    return sentence_embedding.tolist();
  };
  const docs = EMBED_DOCS.map((doc) => documentPrompt(null) + doc);
  await embed(docs.slice(0, 2)); // warm-up
  const embedStarted = performance.now();
  const docVectors = await embed(docs);
  const embedMs = performance.now() - embedStarted;
  const [queryVector] = await embed([queryPrompt('search result') + EMBED_QUERY]);
  await encoder.dispose();

  const scores = docVectors.map((vector) => vector.reduce((sum, value, i) => sum + value * queryVector[i], 0));
  const ranked = scores.map((score, i) => ({ score, i })).sort((a, b) => b.score - a.score);
  return {
    loadMs,
    downloadMs,
    msPerText: embedMs / docs.length,
    text: `「${EMBED_QUERY}」の上位: ${ranked
      .slice(0, 3)
      .map(({ score, i }) => `${EMBED_DOCS[i].slice(0, 18)}…（${score.toFixed(2)}）`)
      .join(' / ')}`,
  };
}

self.addEventListener('message', async ({ data }) => {
  if (data.type !== 'run') return;
  const report = (progress) => self.postMessage({ type: 'progress', ...progress });
  try {
    const found = findVariant(data.key);
    if (!found) throw new Error(`Unknown test: ${data.key}`);
    const { model, variant } = found;
    const lib = await import(TRANSFORMERS_URL);
    lib.env.allowLocalModels = Boolean(data.useLocal);
    lib.env.localModelPath = '/models/';
    lib.env.useBrowserCache = data.useCache !== false;
    const progress = progressTracker(report);
    const run = model.task === 'embedding' ? runEmbedding : runGeneration;
    const result = await run(lib, model, variant, progress, report, data.maxTokens ?? GENERATION_MAX_TOKENS);
    self.postMessage({
      type: 'done',
      result: { ...result, downloadedBytes: progress.totalBytes(), transformersVersion: lib.env.version },
    });
  } catch (error) {
    self.postMessage({ type: 'error', message: error?.message ?? String(error) });
  }
});
