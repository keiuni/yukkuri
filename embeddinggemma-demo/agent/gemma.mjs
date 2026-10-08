// The two Gemma models used by the agent, running in Node with onnxruntime (CPU).
//   - EmbeddingGemma (300M, or EmbeddingGemma 2): the fast "System One" decider that plays Jev's role.
//   - Gemma 4 E2B (instruction tuned): writes the step plan, like the LLM in jev-ultrafast,
//     and can optionally act as the decider for comparison.
import { AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, env } from '@huggingface/transformers';

import { DEFAULT_MODEL, MODELS, documentPrompt, queryPrompt, textOnlyConfig } from '../public/lib/model-config.js';
import { stepQuery } from './actions.mjs';

const MODELS_DIR = new URL('../models/', import.meta.url).pathname;
// Files under ./models are used first; anything missing is downloaded from the Hub into the same folder.
env.localModelPath = MODELS_DIR;
env.cacheDir = MODELS_DIR;

export const GEMMA4_ID = 'onnx-community/gemma-4-E2B-it-ONNX';

export class EmbeddingGemma {
  batchSize = 16;
  #cache = new Map();

  /** `model` is a key of MODELS ('v1' or 'v2'). */
  static async load(model = DEFAULT_MODEL, dtype = 'fp32') {
    const self = new EmbeddingGemma();
    const { id, revision, name, dtypes } = MODELS[model];
    Object.assign(self, { key: model, name: dtype === 'fp32' ? name : `${name} ${dtype}`, revision, dtype });
    self.tokenizer = await AutoTokenizer.from_pretrained(id, { revision });
    self.model = await AutoModel.from_pretrained(id, {
      revision,
      config: await textOnlyConfig(AutoConfig, model),
      dtype,
      device: 'cpu',
      model_file_name: dtypes[dtype].modelFileName,
    });
    return self;
  }

  /** L2-normalized embeddings; texts seen before (e.g. unchanged page elements) come from the cache. */
  async embed(texts) {
    // Batches are padded to their longest text, so embed similar lengths together.
    const missing = [...new Set(texts.filter((text) => !this.#cache.has(text)))].sort((a, b) => a.length - b.length);
    for (let start = 0; start < missing.length; start += this.batchSize) {
      const batch = missing.slice(start, start + this.batchSize);
      const inputs = await this.tokenizer(batch, { padding: true, truncation: true });
      const { sentence_embedding } = await this.model(inputs);
      const [rows, dim] = sentence_embedding.dims;
      for (let row = 0; row < rows; row++) {
        const vector = sentence_embedding.data.slice(row * dim, (row + 1) * dim);
        let norm = 0;
        for (const value of vector) norm += value * value;
        norm = Math.sqrt(norm) || 1;
        for (let i = 0; i < dim; i++) vector[i] /= norm;
        this.#cache.set(batch[row], vector);
      }
    }
    return texts.map((text) => this.#cache.get(text));
  }

  async similarities(query, documents) {
    const [q, ...docs] = await this.embed([queryPrompt('search result') + query, ...documents.map((doc) => documentPrompt(null) + doc)]);
    return docs.map((doc) => doc.reduce((sum, value, i) => sum + value * q[i], 0));
  }
}

export class Gemma4 {
  name = 'Gemma 4 E2B';

  static async load() {
    const self = new Gemma4();
    self.tokenizer = await AutoTokenizer.from_pretrained(GEMMA4_ID);
    self.model = await AutoModelForCausalLM.from_pretrained(GEMMA4_ID, { dtype: 'q4', device: 'cpu' });
    return self;
  }

  async chat(messages, maxNewTokens = 384) {
    const started = performance.now();
    const inputs = this.tokenizer.apply_chat_template(messages, { add_generation_prompt: true, enable_thinking: false, return_dict: true });
    const output = await this.model.generate({ ...inputs, max_new_tokens: maxNewTokens, do_sample: false });
    const promptTokens = inputs.input_ids.dims.at(-1);
    const text = this.tokenizer.batch_decode(output.slice(null, [promptTokens, null]), { skip_special_tokens: true })[0];
    return { text, ms: performance.now() - started, promptTokens, newTokens: output.dims.at(-1) - promptTokens };
  }
}

// ---------------------------------------------------------------- planner (LLM role)

export const PLANNER_PROMPT = `あなたはWebブラウザ操作の手順を書くアシスタントです。目標・これまでの操作・いまのページの要素を見て、いまのページで次に行う操作を、1回の操作（クリック・文字入力・プルダウン選択・スイッチやチェックの切り替え）ごとの短い手順にしてください。
- 出力はJSON配列だけにする。各要素は {"step": "操作の説明", "text": "値"}。
- step には、操作する要素の名前（いまのページの要素の名前をそのまま使う）と、必要な条件（日付・人数・名前など）を書く。
- text には、入力する文字列・プルダウンで選ぶ項目・オンかオフのどれかを書く。値がない操作は空文字にする。
- いまのページでできる操作だけを書く。別のページへ移る操作（検索ボタンやリンク）を書いたら、そこで終える。
- 目標がもう達成されていれば、空の配列 [] だけを出力する。
- 「先月」「来週」などは、今日の日付をもとに具体的な日付に直して考える。`;

function parseSteps(text) {
  const start = text.indexOf('[');
  const end = text.lastIndexOf(']');
  if (start === -1 || end <= start) throw new Error(`No JSON array in planner output: ${text.slice(0, 200)}`);
  const parsed = JSON.parse(text.slice(start, end + 1).replace(/,\s*([\]}])/g, '$1'));
  return parsed
    .filter((item) => item && typeof item.step === 'string' && item.step.trim())
    .map((item) => ({ step: item.step.trim(), text: typeof item.text === 'string' ? item.text.trim() : '' }));
}

/** Calendar facts for relative dates: a 2B model does not reliably turn 「先月」 into a month by itself. */
export function describeToday(date) {
  const month = (offset) => {
    const d = new Date(date.getFullYear(), date.getMonth() + offset, 1);
    return `${d.getFullYear()}年${d.getMonth() + 1}月`;
  };
  const weekday = '日月火水木金土'[date.getDay()];
  return `${date.getFullYear()}年${date.getMonth() + 1}月${date.getDate()}日（${weekday}）／今月は${month(0)}・先月は${month(-1)}・来月は${month(1)}`;
}

/** Plans the next steps on the current page (the agent calls this again after every page change). */
export async function planSteps(gemma4, { goal, site, today, pageTitle, pageElements, history }) {
  const done = history.length ? history.map((line) => `- ${line}`).join('\n') : '（まだ何もしていない）';
  const { text, ms, newTokens, promptTokens } = await gemma4.chat([
    { role: 'system', content: PLANNER_PROMPT },
    {
      role: 'user',
      content: `今日: ${describeToday(today)}\nサイト: ${site}\n目標: ${goal}\n\nこれまでの操作:\n${done}\n\nいまのページ「${pageTitle}」の要素:\n${pageElements}`,
    },
  ]);
  return { steps: parseSteps(text), raw: text, ms, newTokens, promptTokens };
}

export const DONE_PROMPT = `あなたはWebブラウザ操作の結果を確かめる係です。目標・これまでの操作・いまのページを見て、目標がすべて達成されているかを判断してください。
答えは「はい」か「いいえ」の一語だけにしてください。`;

/** Jev's DONE: a separate yes/no question asked whenever the page has changed. */
export async function checkDone(gemma4, { goal, today, site, pageTitle, pageText, pageElements, history }) {
  const done = history.length ? history.map((line) => `- ${line}`).join('\n') : '（まだ何もしていない）';
  const { text, ms } = await gemma4.chat(
    [
      { role: 'system', content: DONE_PROMPT },
      {
        role: 'user',
        content: `今日: ${describeToday(today)}\nサイト: ${site}\n目標: ${goal}\n\nこれまでの操作:\n${done}\n\nいまのページ「${pageTitle}」の本文（冒頭）:\n${pageText}\n\nいまのページの要素:\n${pageElements}\n\n目標はすべて達成されていますか？`,
      },
    ],
    8,
  );
  return { done: /はい|yes/i.test(text) && !/いいえ/.test(text), raw: text.trim(), ms };
}

// ---------------------------------------------------------------- deciders (Jev role)

const TEMPERATURE = 0.02;

function softmax(values, temperature) {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp((value - max) / temperature));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((value) => value / sum);
}

/** EmbeddingGemma: cosine similarity between the step and every candidate action, as probabilities. */
export function embeddingDecider(embedder) {
  return {
    name: embedder.name,
    async decide({ step, candidates }) {
      const started = performance.now();
      const sims = await embedder.similarities(stepQuery(step), candidates.map((candidate) => candidate.text));
      const probs = softmax(sims, TEMPERATURE);
      const index = probs.indexOf(Math.max(...probs));
      return { index, sims, probs, ms: performance.now() - started };
    },
  };
}

const DECIDER_PROMPT = `あなたはブラウザ操作の判断役です。いまの手順に合う操作を、番号つきの候補から1つだけ選んでください。
出力は {"id": 番号} というJSONだけにしてください。`;

/** Gemma 4 E2B as the decider: reads the numbered candidate table and answers with one number.
 *  It gets exactly what EmbeddingGemma gets (the step and the candidates), so the two are comparable. */
export function gemma4Decider(gemma4) {
  return {
    name: gemma4.name,
    async decide({ step, candidates }) {
      const table = candidates.map((candidate, i) => `${i}: ${candidate.text}`).join('\n');
      const { text, ms, promptTokens } = await gemma4.chat(
        [
          { role: 'system', content: DECIDER_PROMPT },
          { role: 'user', content: `いまの手順: ${stepQuery(step)}\n候補:\n${table}` },
        ],
        24,
      );
      const match = text.match(/"id"\s*:\s*(\d+)/) ?? text.match(/(\d+)/);
      const index = match ? Number(match[1]) : -1;
      const valid = index >= 0 && index < candidates.length;
      const probs = candidates.map((_, i) => (i === index ? 1 : 0));
      return { index: valid ? index : -1, probs, ms, raw: text, promptTokens };
    },
  };
}
