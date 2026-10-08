#!/usr/bin/env node
// How often do the phone-sized chat models get everyday Japanese facts right? Asks each question of
// public/data/ja-facts.json several times with different seeds (same settings as scripts/ja-chat.mjs,
// no system prompt) and checks the answer for the required words.
//
//   LLAMA_SERVER=/path/to/llama-server node scripts/ja-facts.mjs                 # every model, 5 seeds
//   LLAMA_SERVER=/path/to/llama-server node scripts/ja-facts.mjs gemma4-e2b --seeds 3
//
// Results go to test-output/ja-chat/facts-<model>.json.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { LlamaServer } from '../agent/llama.mjs';
import { includesAll } from '../public/lib/ja-checks.js';
import { CHAT_MODELS, serverSpec } from './lib/chat-models.mjs';

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
for (const key of keys.length ? keys : Object.keys(CHAT_MODELS)) {
  const model = CHAT_MODELS[key];
  const server = await LlamaServer.start(serverSpec(key), PORT);
  try {
    const answers = [];
    for (const item of items) {
      for (let seed = 0; seed < seeds; seed++) {
        const json = await server.post('/v1/chat/completions', {
          messages: [{ role: 'user', content: item.prompt }],
          max_tokens: MAX_TOKENS,
          seed,
          ...model.sampling,
          ...(model.templateKwargs ? { chat_template_kwargs: model.templateKwargs } : {}),
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
