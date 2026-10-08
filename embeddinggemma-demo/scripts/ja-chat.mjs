#!/usr/bin/env node
// How well do phone-sized chat models write everyday Japanese? Runs the prompts of
// public/data/ja-chat.json (replies to messages, three-turn conversations, rewrites, a few facts) with
// the models of scripts/lib/chat-models.mjs on llama.cpp's llama-server (CPU), and records every answer
// with its speed and the mechanical checks of public/lib/ja-checks.js.
//
//   npm run download-chat-models
//   LLAMA_SERVER=/path/to/llama-server npm run ja-chat                 # every model
//   LLAMA_SERVER=/path/to/llama-server npm run ja-chat -- gemma4-e2b   # some of them
//   LLAMA_SERVER=… npm run ja-chat -- --short gemma4-e2b               # with a system prompt asking for short chat replies
//
// Each model gets the sampling its card recommends, thinking off, a fixed seed and no system prompt
// (with --short: SHORT_SYSTEM_PROMPT, saved as <model>-short.json).
// In a conversation the model's own replies are fed back. Results go to test-output/ja-chat/<model>.json.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { LlamaServer } from '../agent/llama.mjs';
import { checkReply } from '../public/lib/ja-checks.js';
import { CHAT_MODELS, serverSpec } from './lib/chat-models.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'test-output', 'ja-chat');
const PORT = 8095;
const MAX_TOKENS = 512;
const SEED = 0;

const data = JSON.parse(await readFile(path.join(ROOT, 'public', 'data', 'ja-chat.json'), 'utf8'));
// What a chat app would tell the model to keep replies like messages rather than articles.
const SHORT_SYSTEM_PROMPT = 'あなたはスマートフォンのチャットアプリのアシスタントです。友だちや同僚とのやり取りのように、自然な日本語で短く答えてください。見出しや箇条書きは、頼まれたときだけ使ってください。';
const short = process.argv.includes('--short');
const system = short ? [{ role: 'system', content: SHORT_SYSTEM_PROMPT }] : [];
const keys = process.argv.slice(2).filter((arg) => !arg.startsWith('-'));
for (const key of keys) {
  if (!CHAT_MODELS[key]) {
    console.error(`Unknown model "${key}". Choose from: ${Object.keys(CHAT_MODELS).join(', ')}`);
    process.exit(1);
  }
}

/** One chat completion: the reply text (thinking removed, if any), token counts and speeds. */
async function reply(server, model, messages) {
  const started = performance.now();
  const json = await server.post('/v1/chat/completions', {
    messages,
    max_tokens: MAX_TOKENS,
    seed: SEED,
    ...model.sampling,
    ...(model.templateKwargs ? { chat_template_kwargs: model.templateKwargs } : {}),
  });
  const message = json.choices[0].message;
  const text = (message.content ?? '').replace(/<think>[\s\S]*?(<\/think>|$)/, '').trim();
  const timings = json.timings ?? {};
  return {
    text,
    finish: json.choices[0].finish_reason,
    promptTokens: json.usage?.prompt_tokens,
    newTokens: json.usage?.completion_tokens,
    ms: performance.now() - started,
    promptPerSecond: timings.prompt_per_second,
    decodePerSecond: timings.predicted_per_second,
    reasoningChars: (message.reasoning_content ?? '').length,
  };
}

await mkdir(OUT, { recursive: true });
for (const key of keys.length ? keys : Object.keys(CHAT_MODELS)) {
  const model = CHAT_MODELS[key];
  process.stdout.write(`${model.name} … `);
  const loadStarted = performance.now();
  const server = await LlamaServer.start(serverSpec(key), PORT);
  const loadMs = performance.now() - loadStarted;
  try {
    await reply(server, model, [{ role: 'user', content: 'こんにちは' }]); // warm-up
    const answers = [];
    for (const item of data.items) {
      const result = await reply(server, model, [...system, { role: 'user', content: item.prompt }]);
      answers.push({ id: item.id, category: item.category, prompt: item.prompt, ...result, checks: checkReply(result.text, item) });
    }
    for (const conversation of data.conversations) {
      const messages = [...system];
      for (const [turn, spec] of conversation.turns.entries()) {
        messages.push({ role: 'user', content: spec.prompt });
        const result = await reply(server, model, messages);
        messages.push({ role: 'assistant', content: result.text });
        answers.push({ id: `${conversation.id}-${turn + 1}`, conversation: conversation.id, turn: turn + 1, category: 'conversation', prompt: spec.prompt, ...result, checks: checkReply(result.text, spec) });
      }
    }
    const decode = answers.map((answer) => answer.decodePerSecond).filter(Number.isFinite).sort((a, b) => a - b);
    const run = {
      createdAt: new Date().toISOString(),
      model: { key, name: model.name, repo: model.repo, revision: model.revision, file: model.file, quant: model.quant, sampling: model.sampling, templateKwargs: model.templateKwargs ?? null },
      device: 'cpu (llama.cpp, 4 cores)',
      systemPrompt: short ? SHORT_SYSTEM_PROMPT : null,
      loadMs,
      decodePerSecondMedian: decode[Math.floor(decode.length / 2)],
      answers,
    };
    await writeFile(path.join(OUT, `${key}${short ? '-short' : ''}.json`), `${JSON.stringify(run, null, 2)}\n`);
    const flagged = answers.filter((answer) => answer.checks.problems.length);
    console.log(`${answers.length} answers, ${run.decodePerSecondMedian?.toFixed(1)} tok/s, ${flagged.length} flagged: ${flagged.map((a) => `${a.id}(${a.checks.problems.join('/')})`).join(', ')}`);
  } finally {
    await server.stop();
  }
}
