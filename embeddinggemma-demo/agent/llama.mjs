// Liquid AI models served by llama.cpp's llama-server (CPU), for comparison with the Gemma models:
//   - d1-3B / d1-omni-600M: decision models. /v1/systemone answers a typed question (a "choice" over
//     named options) in one forward pass, without generating tokens. They play Jev's role, like
//     EmbeddingGemma: pick the candidate action for the current step.
//   - LFM2.5-2.6B: an agentic chat model (OpenAI-compatible /v1/chat/completions). It writes the steps
//     and checks the goal like Gemma 4 E2B, or picks the candidate as a generative decider.
// llama-server comes from a llama.cpp build: set LLAMA_SERVER to its path if it is not on PATH. The GGUF
// files are downloaded into ./models by scripts/download-liquid-models.mjs.
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { stepQuery } from './actions.mjs';

const MODELS_DIR = fileURLToPath(new URL('../models/', import.meta.url));

// Pinned revisions (2026-10-06 for d1, 2026-08-01 for LFM2.5-2.6B).
export const LIQUID_MODELS = {
  'd1-3b': {
    name: 'd1-3B',
    repo: 'LiquidAI/d1-3B-GGUF',
    revision: 'bb1e436ea78eb96a3f1acb6da865f70c2fbeb563',
    file: 'd1-3B-Q8_0.gguf',
    kind: 'decision',
    args: ['-c', '16384'],
  },
  'd1-omni': {
    name: 'd1-omni-600M',
    repo: 'LiquidAI/d1-omni-600M-GGUF',
    revision: '04397145ed8381350403aa556db0ec6a49dd8c07',
    file: 'd1-omni-600M-Q8_0.gguf',
    kind: 'decision',
    // A question is read in one batch (model card), so the batch must hold the longest prompt.
    args: ['-c', '16384', '-b', '16384', '-ub', '16384'],
  },
  lfm: {
    name: 'LFM2.5-2.6B',
    repo: 'LiquidAI/LFM2.5-2.6B-GGUF',
    revision: 'e7caca5d835a3901a8e0d63e94009429bafafdfc',
    file: 'LFM2.5-2.6B-Q4_K_M.gguf',
    kind: 'chat',
    args: ['-c', '16384', '--jinja'],
  },
};

export const ggufPath = (key) => `${MODELS_DIR}${LIQUID_MODELS[key].repo}/${LIQUID_MODELS[key].file}`;

/** One llama-server process per model, on its own port. */
export class LlamaServer {
  static async start(key, port) {
    const model = ggufPath(key);
    if (!existsSync(model)) throw new Error(`${model} is missing: run node scripts/download-liquid-models.mjs`);
    const self = new LlamaServer();
    self.url = `http://127.0.0.1:${port}`;
    const args = ['-m', model, '--host', '127.0.0.1', '--port', String(port), '-np', '1', ...LIQUID_MODELS[key].args];
    self.process = spawn(process.env.LLAMA_SERVER ?? 'llama-server', args, { stdio: ['ignore', 'ignore', 'pipe'] });
    let log = '';
    self.process.stderr.on('data', (chunk) => {
      log = (log + chunk).slice(-4000);
    });
    const exited = new Promise((resolve) => self.process.once('exit', resolve));
    // Up to 10 minutes: reading a 3 GB GGUF that is not in the page cache took over 3 minutes on this disk.
    for (let attempt = 0; attempt < 3000; attempt++) {
      const status = await fetch(`${self.url}/health`).then((response) => response.status, () => 0);
      if (status === 200) return self;
      if (await Promise.race([exited.then(() => true), new Promise((resolve) => setTimeout(() => resolve(false), 200))])) {
        throw new Error(`llama-server exited while loading ${LIQUID_MODELS[key].file}:\n${log}`);
      }
    }
    self.stop();
    throw new Error(`llama-server did not become ready for ${LIQUID_MODELS[key].file}`);
  }

  async post(path, body) {
    const response = await fetch(`${this.url}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const json = await response.json();
    if (!response.ok) throw new Error(`${path}: ${response.status} ${JSON.stringify(json).slice(0, 300)}`);
    return json;
  }

  stop() {
    this.process?.kill();
  }
}

/** A d1 model behind /v1/systemone. */
export class DecisionModel {
  constructor(server, name) {
    this.server = server;
    this.name = name;
  }

  static async load(key, port) {
    return new DecisionModel(await LlamaServer.start(key, port), LIQUID_MODELS[key].name);
  }

  /** Probability of each text in `options` being the answer to `instructions` about `state`, and the prompt size. */
  async choose(state, instructions, options) {
    const criteria = Object.fromEntries(options.map((text, i) => [String(i), text]));
    const { answers, usage } = await this.server.post('/v1/systemone', { state, questions: { pick: { type: 'choice', instructions, criteria } } });
    return { probs: options.map((_, i) => answers.pick.probabilities[String(i)] ?? 0), promptTokens: usage?.input_tokens };
  }

  /** Same interface as EmbeddingGemma.similarities, so it can also pick options in a pull-down. */
  async similarities(query, documents) {
    return (await this.choose(`手順: ${query}`, 'この手順に合う項目はどれですか？', documents)).probs;
  }

  stop() {
    this.server.stop();
  }
}

// The question asked for every step. The page, the step and the candidates stay in Japanese either way;
// `en` only changes the wording around them (the Decision Index questions are in English).
const DECISION_PROMPTS = {
  ja: { state: '手順: ', instructions: 'この手順を実行するには、どの操作をすればよいですか？' },
  en: { state: 'Step: ', instructions: 'Which action carries out this step?' },
};

/** d1 as the decider: the same inputs EmbeddingGemma gets (the step and the candidate actions). */
export function decisionDecider(model, lang = 'ja') {
  const prompt = DECISION_PROMPTS[lang];
  return {
    name: lang === 'ja' ? model.name : `${model.name}（${lang} の指示文）`,
    scorer: model,
    async decide({ step, candidates }) {
      const started = performance.now();
      const { probs, promptTokens } = await model.choose(
        `${prompt.state}${stepQuery(step)}`,
        prompt.instructions,
        candidates.map((candidate) => candidate.text),
      );
      return { index: probs.indexOf(Math.max(...probs)), probs, ms: performance.now() - started, promptTokens };
    },
  };
}

function parseArguments(json) {
  try {
    return typeof json === 'string' ? JSON.parse(json) : json ?? {};
  } catch {
    return {};
  }
}

/**
 * LFM2.5's own tool-call format, a Python list such as [click(element=12), type_text(element=3, text='京都')],
 * in case llama-server leaves it in the text. Only keyword arguments with literal values are read.
 */
export function parsePythonicCalls(text) {
  const body = text.match(/<\|tool_call_start\|>\s*\[([\s\S]*?)\]\s*<\|tool_call_end\|>/)?.[1];
  if (!body) return [];
  const calls = [];
  for (const [, name, rawArgs] of body.matchAll(/(\w+)\(((?:[^()'"]|'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")*)\)/g)) {
    const args = {};
    for (const [, key, value] of rawArgs.matchAll(/(\w+)\s*=\s*('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?|True|False|true|false)/g)) {
      if (/^['"]/.test(value)) args[key] = value.slice(1, -1).replace(/\\(.)/g, '$1');
      else if (/^(true|false)$/i.test(value)) args[key] = value.toLowerCase() === 'true';
      else args[key] = Number(value);
    }
    calls.push({ name, args });
  }
  return calls;
}

/** A chat model behind the OpenAI-compatible API, with the interface of Gemma4 in gemma.mjs. */
export class ChatModel {
  // LFM2.5-2.6B always reasons before answering, so every call gets room for its thinking.
  static MIN_TOKENS = 2048;

  constructor(server, key) {
    this.server = server;
    this.name = LIQUID_MODELS[key].name;
    this.id = `${LIQUID_MODELS[key].repo}/${LIQUID_MODELS[key].file}@${LIQUID_MODELS[key].revision.slice(0, 7)}`;
  }

  static async load(key, port) {
    return new ChatModel(await LlamaServer.start(key, port), key);
  }

  async chat(messages, maxNewTokens = 384) {
    const { calls, ...result } = await this.#complete({ messages }, maxNewTokens);
    return result;
  }

  /** chat() with OpenAI-style tools: also returns the tool calls as [{ name, args }]. */
  chatWithTools(messages, tools, maxNewTokens = ChatModel.MIN_TOKENS) {
    return this.#complete({ messages, tools }, maxNewTokens);
  }

  async #complete(request, maxNewTokens) {
    const started = performance.now();
    // The model card's generation settings, with a fixed seed.
    const json = await this.server.post('/v1/chat/completions', {
      ...request,
      max_tokens: Math.max(maxNewTokens, ChatModel.MIN_TOKENS),
      temperature: 0.1,
      top_k: 50,
      repeat_penalty: 1.1,
      seed: 0,
    });
    const message = json.choices[0].message;
    // llama-server moves the <think> part to reasoning_content; strip it ourselves if it did not.
    const text = (message.content ?? '').replace(/<think>[\s\S]*?(<\/think>|$)/, '').trim();
    const calls = message.tool_calls?.length
      ? message.tool_calls.map((call) => ({ name: call.function.name, args: parseArguments(call.function.arguments) }))
      : parsePythonicCalls(text);
    return {
      text,
      calls,
      ms: performance.now() - started,
      promptTokens: json.usage?.prompt_tokens,
      newTokens: json.usage?.completion_tokens,
      reasoningChars: (message.reasoning_content ?? '').length,
      finish: json.choices[0].finish_reason,
    };
  }

  stop() {
    this.server.stop();
  }
}
