#!/usr/bin/env node
// Two small bots talk to each other with no human in between: what goes wrong when a phone-sized model is the
// "brain" of an autonomous bot? Each bot gets a name, a character, a situation and a goal (public/data/bot-scenes.json)
// and the conversation so far (the other bot's lines as the user's, its own as the assistant's). The first line is fixed;
// the rest is generated for 30 round trips. Repeats, speaking for the other side, endless goodbyes and broken
// formats are counted by public/lib/bot-metrics.js; whether they get anywhere is judged by reading the transcripts.
//
//   npm run bot-chat -- gemma4-e2b-litert                  # every scene
//   npm run bot-chat -- gemma4-e4b-litert --scene village --rounds 10
//
// Phone builds (`runtime: 'litert-lm'`) need LITERT_LM; GGUF models need LLAMA_SERVER (see scripts/ja-chat.mjs).
// Results go to test-output/bot-chat/<model>-<scene>.json.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { ACTIONS, analyzeConversation } from '../public/lib/bot-metrics.js';
import { CHAT_MODELS } from './lib/chat-models.mjs';
import { requestOptions } from './lib/chat-server.mjs';
import { LiteRtServer } from './lib/litert-server.mjs';
import { LlamaServer } from '../agent/llama.mjs';
import { serverSpec } from './lib/chat-models.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'test-output', 'bot-chat');
const PORT = 8096;
const MAX_TOKENS = 120;

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : fallback;
};
const rounds = Number(option('rounds', 30));
const sceneFilter = option('scene');
const keys = args.filter((arg, i) => !arg.startsWith('-') && !args[i - 1]?.startsWith('--'));
for (const key of keys) {
  if (!CHAT_MODELS[key]) {
    console.error(`Unknown model "${key}". Choose from: ${Object.keys(CHAT_MODELS).join(', ')}`);
    process.exit(1);
  }
}
const { scenes } = JSON.parse(await readFile(path.join(ROOT, 'public', 'data', 'bot-scenes.json'), 'utf8'));

/** The instructions of one bot: who it is, who it talks to, and the rules that keep a bot-to-bot chat on track. */
function systemPrompt(bot, other, scene) {
  const format = scene.actionFormat
    ? `\n- 返事は2行で書く。1行目は${bot.name}のセリフ、2行目は「行動: 」に続けて、次のどれか1つだけ：${ACTIONS.join('、')}`
    : '';
  return [
    `あなたは「${bot.name}」という名前のボットです。${bot.persona}。`,
    `いま、${bot.situation}。相手は「${other.name}」というボットです。`,
    '',
    '決まり：',
    `- 返事は${bot.name}のセリフだけを、1〜2文で書く。`,
    `- ${other.name}のセリフを代わりに言わない。「${bot.name}：」のような名前のラベルも付けない。`,
    '- 同じことをくり返さず、話を少しずつ進める。',
    `- 目的：${bot.goal}。目的が果たせたら、短くあいさつして会話を終える。${format}`,
  ].join('\n');
}

async function chat(server, model, messages, seed) {
  const started = performance.now();
  const json = await server.post('/v1/chat/completions', { messages, ...requestOptions(model, { maxTokens: MAX_TOKENS, seed }) });
  return {
    text: (json.choices[0].message.content ?? '').replace(/<think>[\s\S]*?(<\/think>|$)/, '').trim(),
    ms: performance.now() - started,
    promptTokens: json.usage?.prompt_tokens,
    newTokens: json.usage?.completion_tokens,
  };
}

async function startServer(key) {
  const model = CHAT_MODELS[key];
  // A long chat: 60 lines of 40 tokens or so need more than LiteRT-LM's default 4,096-token window.
  if (model.runtime === 'litert-lm') return LiteRtServer.start({ key, ...model }, PORT, { maxNumTokens: 8192 });
  return LlamaServer.start({ ...serverSpec(key), args: ['-c', '8192', '--jinja'] }, PORT);
}

await mkdir(OUT, { recursive: true });
for (const key of keys.length ? keys : ['gemma4-e2b-litert']) {
  const model = CHAT_MODELS[key];
  const server = await startServer(key);
  try {
    for (const scene of scenes.filter((s) => !sceneFilter || s.id === sceneFilter)) {
      const [a, b] = scene.bots;
      const system = { [a.name]: systemPrompt(a, b, scene), [b.name]: systemPrompt(b, a, scene) };
      const turns = [{ speaker: a.name, text: scene.actionFormat ? `${scene.opener}\n行動: 手を振る` : scene.opener, fixed: true }];
      const started = performance.now();
      process.stdout.write(`${model.name} / ${scene.id}: `);
      while (turns.length < rounds * 2) {
        const speaker = turns.length % 2 === 0 ? a : b;
        // What this bot saw: its own lines as the assistant's, the other bot's as the user's. Bot A starts, so it
        // is first told to begin; the first line was written for it.
        const messages = [{ role: 'system', content: system[speaker.name] }];
        if (speaker === a) messages.push({ role: 'user', content: '（会話を始めてください）' });
        for (const turn of turns) messages.push({ role: turn.speaker === speaker.name ? 'assistant' : 'user', content: turn.text });
        const result = await chat(server, model, messages, turns.length + 1); // LiteRT-LM answers seed 0 like seed 1
        turns.push({ speaker: speaker.name, text: result.text, ms: Math.round(result.ms), promptTokens: result.promptTokens, newTokens: result.newTokens });
        process.stdout.write('.');
      }
      const metrics = analyzeConversation(turns, [a.name, b.name], { actionFormat: Boolean(scene.actionFormat) });
      const run = {
        createdAt: new Date().toISOString(),
        model: { key, name: model.name, revision: model.revision, file: model.file },
        scene: scene.id,
        rounds,
        system,
        elapsedSeconds: Math.round((performance.now() - started) / 1000),
        peakMemoryMB: server.peakMemoryMB?.() ?? null,
        turns,
        metrics,
      };
      await writeFile(path.join(OUT, `${key}-${scene.id}.json`), `${JSON.stringify(run, null, 2)}\n`);
      const { flags, ...summary } = metrics;
      console.log(` ${run.elapsedSeconds}s ${JSON.stringify(summary)}`);
    }
  } finally {
    await server.stop();
  }
}
