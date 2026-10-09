#!/usr/bin/env node
// How often do the phone-sized chat models get everyday Japanese facts right? Asks each question of
// public/data/ja-facts.json several times with different seeds (same settings as scripts/ja-chat.mjs,
// no system prompt) and checks the answer for the required words.
//
//   LLAMA_SERVER=/path/to/llama-server node scripts/ja-facts.mjs                 # every GGUF model, 5 seeds
//   node scripts/ja-facts.mjs gemma4-e4b-litert                                     # a phone build (LiteRT-LM)
//   LLAMA_SERVER=/path/to/llama-server node scripts/ja-facts.mjs gemma4-e2b --seeds 3
//
// Results go to test-output/ja-chat/facts-<model>.json.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { includesAll } from '../public/lib/ja-checks.js';
import { CHAT_MODELS, GGUF_MODEL_KEYS } from './lib/chat-models.mjs';
import { requestOptions, startChatServer } from './lib/chat-server.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'test-output', 'ja-chat');
const PORT = 8097;
const MAX_TOKENS = 256;

const args = process.argv.slice(2);
const seedsIndex = args.indexOf('--seeds');
const seeds = seedsIndex >= 0 ? Number(args[seedsIndex + 1]) : 5;
const keys = args.filter((arg, i) => !arg.startsWith('-') && (seedsIndex < 0 || i !== seedsIndex + 1));
const { items } = JSON.parse(await readFile(path.join(ROOT, 'public', 'data', 'ja-facts.json'), 'utf8'));

await mkdir(OUT, { recursive: true });
for (const key of keys.length ? keys : GGUF_MODEL_KEYS) {
  const model = CHAT_MODELS[key];
  const server = await startChatServer(key, PORT);
  // LiteRT-LM answers seed 0 exactly like seed 1, so the phone builds start at 1 to get as many different samples.
  const firstSeed = model.runtime === 'litert-lm' ? 1 : 0;
  try {
    const answers = [];
    for (const item of items) {
      for (let seed = firstSeed; seed < firstSeed + seeds; seed++) {
        const json = await server.post('/v1/chat/completions', {
          messages: [{ role: 'user', content: item.prompt }],
          ...requestOptions(model, { maxTokens: MAX_TOKENS, seed }),
        });
        const text = (json.choices[0].message.content ?? '').replace(/<think>[\s\S]*?(<\/think>|$)/, '').trim();
        answers.push({ id: item.id, seed, right: includesAll(text, item.includes), text });
      }
    }
    const byItem = Object.fromEntries(items.map((item) => [item.id, answers.filter((a) => a.id === item.id && a.right).length]));
    const right = answers.filter((a) => a.right).length;
    await writeFile(path.join(OUT, `facts-${key}.json`), `${JSON.stringify({ createdAt: new Date().toISOString(), model: model.name, seeds, right, total: answers.length, byItem, answers }, null, 2)}\n`);
    console.log(`${model.name}: ${right}/${answers.length} right  ${Object.entries(byItem).map(([id, n]) => `${id} ${n}/${seeds}`).join('  ')}`);
  } finally {
    await server.stop();
  }
}
