#!/usr/bin/env node
// How well do phone-sized chat models write everyday Japanese? Runs the prompts of
// public/data/ja-chat.json (replies to messages, three-turn conversations, rewrites, a few facts) with
// the models of scripts/lib/chat-models.mjs on llama.cpp's llama-server (CPU), and records every answer
// with its speed and the mechanical checks of public/lib/ja-checks.js.
//
//   npm run download-chat-models
//   LLAMA_SERVER=/path/to/llama-server npm run ja-chat                 # every GGUF model
//   LLAMA_SERVER=/path/to/llama-server npm run ja-chat -- gemma4-e2b   # some of them
//   LLAMA_SERVER=… npm run ja-chat -- --short gemma4-e2b               # with a system prompt asking for short chat replies
//   npm run ja-chat -- --short --data ja-talk gemma4-e4b-litert         # the longer conversations of public/data/ja-talk.json
//
// Each model gets the sampling its card recommends, thinking off, a fixed seed and no system prompt
// (with --short: SHORT_SYSTEM_PROMPT, saved as <model>-short.json).
// In a conversation the model's own replies are fed back. Results go to test-output/ja-chat/<model>.json
// (<model>-talk.json for --data ja-talk). The phone builds of Gemma 4 (`runtime: 'litert-lm'`) run on
// LiteRT-LM's server instead of llama-server (scripts/lib/litert-server.mjs).
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { checkReply } from '../public/lib/ja-checks.js';
import { CHAT_MODELS, GGUF_MODEL_KEYS } from './lib/chat-models.mjs';
import { requestOptions, startChatServer } from './lib/chat-server.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'test-output', 'ja-chat');
const PORT = 8095;
const MAX_TOKENS = 512;
const SEED = 0;

const dataIndex = process.argv.indexOf('--data');
const dataName = dataIndex > 0 ? process.argv[dataIndex + 1] : 'ja-chat';
const data = JSON.parse(await readFile(path.join(ROOT, 'public', 'data', `${dataName}.json`), 'utf8'));
// What a chat app would tell the model to keep replies like messages rather than articles.
const SHORT_SYSTEM_PROMPT = 'あなたはスマートフォンのチャットアプリのアシスタントです。友だちや同僚とのやり取りのように、自然な日本語で短く答えてください。見出しや箇条書きは、頼まれたときだけ使ってください。';
const short = process.argv.includes('--short');
const system = short ? [{ role: 'system', content: SHORT_SYSTEM_PROMPT }] : [];
const keys = process.argv.slice(2).filter((arg, i, args) => !arg.startsWith('-') && args[i - 1] !== '--data');
for (const key of keys) {
  if (!CHAT_MODELS[key]) {
    console.error(`Unknown model "${key}". Choose from: ${Object.keys(CHAT_MODELS).join(', ')}`);
    process.exit(1);
  }
}

/** One chat completion: the reply text (thinking removed, if any), token counts and speeds. */
async function reply(server, model, messages) {
  const started = performance.now();
  const json = await server.post('/v1/chat/completions', { messages, ...requestOptions(model, { maxTokens: MAX_TOKENS, seed: SEED }) });
  const message = json.choices[0].message;
  const text = (message.content ?? '').replace(/<think>[\s\S]*?(<\/think>|$)/, '').trim();
  const timings = json.timings ?? {};
  const ms = performance.now() - started;
  const newTokens = json.usage?.completion_tokens;
  return {
    text,
    // LiteRT-LM's server says "stop" even when the answer was cut at max_completion_tokens.
    finish: newTokens >= MAX_TOKENS ? 'length' : json.choices[0].finish_reason,
    promptTokens: json.usage?.prompt_tokens,
    newTokens,
    ms,
    promptPerSecond: timings.prompt_per_second,
    decodePerSecond: timings.predicted_per_second,
    // Whole request (reading the prompt, then writing): the only speed LiteRT-LM's server lets us see.
    overallPerSecond: newTokens / (ms / 1000),
    reasoningChars: (message.reasoning_content ?? '').length,
  };
}

await mkdir(OUT, { recursive: true });
for (const key of keys.length ? keys : GGUF_MODEL_KEYS) {
  const model = CHAT_MODELS[key];
  process.stdout.write(`${model.name} … `);
  const loadStarted = performance.now();
  const server = await startChatServer(key, PORT);
  const loadMs = performance.now() - loadStarted;
  try {
    await reply(server, model, [{ role: 'user', content: 'こんにちは' }]); // warm-up
    const answers = [];
    for (const item of data.items ?? []) {
      const result = await reply(server, model, [...system, { role: 'user', content: item.prompt }]);
      answers.push({ id: item.id, category: item.category, prompt: item.prompt, ...result, checks: checkReply(result.text, item) });
    }
    for (const conversation of data.conversations) {
      const messages = [...system];
      for (const [turn, spec] of conversation.turns.entries()) {
        messages.push({ role: 'user', content: spec.prompt });
        const result = await reply(server, model, messages);
        messages.push({ role: 'assistant', content: result.text });
        answers.push({ id: `${conversation.id}-${turn + 1}`, conversation: conversation.id, turn: turn + 1, category: conversation.category, prompt: spec.prompt, ...result, checks: checkReply(result.text, spec) });
      }
    }
    const median = (values) => {
      const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
      return sorted[Math.floor(sorted.length / 2)];
    };
    const run = {
      createdAt: new Date().toISOString(),
      model: { key, name: model.name, repo: model.repo, revision: model.revision, file: model.file, quant: model.quant, sampling: model.sampling, templateKwargs: model.templateKwargs ?? null },
      data: dataName,
      device: `cpu (${model.runtime ?? 'llama.cpp'}, 4 cores)`,
      systemPrompt: short ? SHORT_SYSTEM_PROMPT : null,
      loadMs,
      decodePerSecondMedian: median(answers.map((answer) => answer.decodePerSecond)),
      overallPerSecondMedian: median(answers.map((answer) => answer.overallPerSecond)),
      peakMemoryMB: server.peakMemoryMB?.() ?? null,
      answers,
    };
    const suffix = `${short ? '-short' : ''}${dataName === 'ja-chat' ? '' : `-${dataName.replace(/^ja-/, '')}`}`;
    await writeFile(path.join(OUT, `${key}${suffix}.json`), `${JSON.stringify(run, null, 2)}\n`);
    const flagged = answers.filter((answer) => answer.checks.problems.length);
    const speed = run.decodePerSecondMedian ?? run.overallPerSecondMedian;
    console.log(`${answers.length} answers, ${speed?.toFixed(1)} tok/s, ${flagged.length} flagged: ${flagged.map((a) => `${a.id}(${a.checks.problems.join('/')})`).join(', ')}`);
  } finally {
    await server.stop();
  }
}
