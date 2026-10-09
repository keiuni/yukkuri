#!/usr/bin/env node
// Blind judging of the Japanese chat test (scripts/ja-chat.mjs).
//
//   node scripts/ja-chat-judge.mjs pack 1      # write judge packets for pass 1 (answers shuffled, models hidden)
//   node scripts/ja-chat-judge.mjs score       # combine every pass's judgments with the mechanical checks
//   node scripts/ja-chat-judge.mjs pack 1 --short / score --short   # the same for the runs with ja-chat.mjs --short
//   node scripts/ja-chat-judge.mjs pack 4 --tag phone --runs gemma4-e4b-short,gemma4-e4b-litert-short   # exactly these runs
//
// `pack` writes test-output/ja-chat/judge/pass<N>-<part>.md (what a judge reads) and pass<N>-key.json
// (which label is which model). A judge (here: a separate Claude agent per packet, given only the
// packet) writes pass<N>-<part>.json: [{ "item", "label", "natural", "fit", "usable" }], scores 1-5 and
// usable one of "yes" / "edit" / "no". `score` maps the labels back and prints the tables.
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { mean } from '../public/lib/metrics.js';
import { CHAT_MODELS, GGUF_MODEL_KEYS } from './lib/chat-models.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIR = path.join(ROOT, 'test-output', 'ja-chat');
const JUDGE = path.join(DIR, 'judge');
const readData = async (name) => JSON.parse(await readFile(path.join(ROOT, 'public', 'data', `${name}.json`), 'utf8'));
const chat = await readData('ja-chat');
const talk = await readData('ja-talk');
// The longer conversations of ja-talk.json (ja-chat.mjs --data ja-talk) are judged like the three-turn ones.
const data = { categories: [...chat.categories, ...talk.categories], items: chat.items, conversations: [...chat.conversations, ...talk.conversations] };
const talkIds = talk.conversations.map((conversation) => conversation.id);

// What the judge needs to know to check facts (the items' own wording does not give the answers).
const REFERENCE = {
  'knowledge-fuji': '富士山、標高 3,776 m。',
  'knowledge-shinkansen': '東京〜新大阪は、のぞみで約 2 時間 20〜30 分（最速 2 時間 21 分）。',
  'knowledge-proverb': '人に情けをかけておくと、巡り巡って自分によい報いが返ってくる、という意味（「甘やかすと本人のためにならない」は誤用）。',
  'knowledge-mutsuki': '睦月（むつき）。',
  'knowledge-kodomo': '5 月 5 日。',
  'conv-kyoto': '京都の紅葉の見ごろは、例年 11 月中旬〜12 月上旬。',
  'talk-dog': '犬の名前はポチ、12 歳。',
  'talk-meeting': '今日は木曜日なので、来週火曜日の会議までに使える平日は金曜日と月曜日の 2 日（土日を除く）。',
  'talk-hokkaido': '札幌〜小樽は JR 函館本線で、快速なら約 30〜40 分、普通列車で約 50 分。電車で行ける。',
  'talk-english': '「よろしくお願いします」にぴったり重なる英語はなく、場面によって Nice to meet you. / I look forward to working with you. / Thank you in advance. などを使い分ける。',
};

const PARTS = {
  replies: (id) => id.startsWith('reply-'),
  tasks: (id) => id.startsWith('rewrite-') || id.startsWith('knowledge-'),
  conversations: (id) => id.startsWith('conv-'),
  // The ten longer conversations make two packets, so that no judge reads much more than the others.
  talk1: (id) => talkIds.indexOf(id) >= 0 && talkIds.indexOf(id) < talkIds.length / 2,
  talk2: (id) => talkIds.indexOf(id) >= talkIds.length / 2,
};

const RUBRIC = `あなたは日本語の文章の採点者です。小さな言語モデルたちの回答を、モデル名を伏せて並べています（A, B, … はモデルごとにランダムに付けた記号で、問題ごとに付け直しています）。
各回答を、次の 3 つの観点で採点してください。

- natural（日本語の自然さ, 1〜5）: 文法、言葉の選び方、敬語の誤り、不自然な言い回し、他の言語（中国語・英語など）の混入、同じ文の繰り返し。5 = ネイティブが書いたように自然、1 = 意味が通じないほど不自然。
- fit（頼まれたことへの合い方, 1〜5）: 頼まれた内容・口調（くだけた／丁寧）・長さ・形式に合っているか。事実の誤りや、関係のない内容、話のつながりの誤りは大きく減点。5 = 完全に頼まれたとおり、1 = まったく合っていない。
- usable（そのまま使えるか）: "yes" = そのまま送れる・使える、"edit" = 少し直せば使える、"no" = 使えない。

厳しめに、一貫した基準で採点してください。回答の長さそのものでは加点しないでください（短い返信を頼まれているときに長すぎるのは減点）。
出力は JSON の配列だけにしてください。要素は {"item": 問題のID, "label": 記号, "natural": 数, "fit": 数, "usable": "yes"|"edit"|"no"} です。会話は各ターンを別の要素にし、item を "<会話ID>-<ターン番号>"（例: "conv-running-2"）にしてください。`;

/** Fixed pseudo-random shuffle (mulberry32), so a pass can be packed again identically. */
function shuffle(items, seed) {
  let state = seed >>> 0;
  const next = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32;
  };
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

// --short judges the runs made with a system prompt asking for short replies (scripts/ja-chat.mjs --short)
// side by side with the same models' plain runs, so that one judge compares both. --tag <name> --runs a,b
// judges exactly the named runs (test-output/ja-chat/<run>.json and <run>-talk.json, whichever exist).
const option = (name) => {
  const index = process.argv.indexOf(`--${name}`);
  return index > 0 ? process.argv[index + 1] : undefined;
};
const namedRuns = option('runs')?.split(',');
const variant = option('tag') ?? (process.argv.includes('--short') ? 'short' : '');
const prefix = (pass) => `pass${pass}${variant ? `-${variant}` : ''}`;
const modelName = (key) => `${CHAT_MODELS[key.replace(/-short$/, '')].name}${key.endsWith('-short') ? '（短く答えるよう指示）' : ''}`;
const COLUMNS = { reply: '返信', conversation: '会話', rewrite: '書き換え', knowledge: '知識', talk: '長い会話' };

async function loadRuns() {
  const runs = {};
  const read = async (file) => JSON.parse(await readFile(path.join(DIR, file), 'utf8'));
  if (namedRuns) {
    for (const run of namedRuns) {
      const found = (await Promise.all([`${run}.json`, `${run}-talk.json`].map((file) => read(file).catch(() => null)))).filter(Boolean);
      if (!found.length) throw new Error(`No results for ${run} in ${DIR}`);
      runs[run] = { ...found[0], answers: found.flatMap((part) => part.answers) };
    }
    return runs;
  }
  for (const key of GGUF_MODEL_KEYS) {
    try {
      if (!variant) {
        runs[key] = await read(`${key}.json`);
      } else {
        runs[`${key}-short`] = await read(`${key}-short.json`);
        runs[key] = await read(`${key}.json`);
      }
    } catch {
      // not run (yet)
    }
  }
  return runs;
}

async function pack(pass) {
  const runs = await loadRuns();
  const models = Object.keys(runs);
  const key = {};
  const written = [];
  await mkdir(JUDGE, { recursive: true });
  for (const [part, belongs] of Object.entries(PARTS)) {
    const blocks = [];
    const units = [...data.items.map((item) => item.id), ...data.conversations.map((c) => c.id)].filter(belongs);
    for (const [u, unit] of units.entries()) {
      const present = models.filter((model) => runs[model].answers.some((a) => a.id === unit || a.conversation === unit));
      if (!present.length) continue;
      const order = shuffle(present, pass * 1000 + u * 37 + part.length);
      const labels = Object.fromEntries(order.map((model, i) => [String.fromCharCode(65 + i), model]));
      key[unit] = labels;
      const conversation = data.conversations.find((c) => c.id === unit);
      const item = data.items.find((i) => i.id === unit);
      const lines = [`## ${unit}`];
      if (item) lines.push(`問題: ${item.prompt}`);
      if (conversation) lines.push(`会話（${conversation.about}）。ユーザーの発言は全員共通で、回答者ごとに自分の前の返答を覚えたまま続けています。`);
      if (REFERENCE[unit]) lines.push(`参考（事実の確認用）: ${REFERENCE[unit]}`);
      for (const [label, model] of Object.entries(labels)) {
        const answers = runs[model].answers;
        if (item) {
          lines.push(`### 回答 ${label}\n${answers.find((a) => a.id === unit).text || '（空）'}`);
        } else {
          const turns = conversation.turns.map((turn, t) => {
            const answer = answers.find((a) => a.id === `${unit}-${t + 1}`);
            return `ユーザー（ターン ${t + 1}）: ${turn.prompt}\n回答者 ${label}（ターン ${t + 1}）: ${answer.text || '（空）'}`;
          });
          lines.push(`### 回答者 ${label}\n${turns.join('\n')}`);
        }
      }
      blocks.push(lines.join('\n\n'));
    }
    if (!blocks.length) continue;
    await writeFile(path.join(JUDGE, `${prefix(pass)}-${part}.md`), `${RUBRIC}\n\n${blocks.join('\n\n---\n\n')}\n`);
    written.push(`${prefix(pass)}-${part}.md`);
  }
  await writeFile(path.join(JUDGE, `${prefix(pass)}-key.json`), `${JSON.stringify(key, null, 2)}\n`);
  console.log(`Packed pass ${pass} for ${models.length} models: ${written.join(', ')}`);
}

async function score() {
  const runs = await loadRuns();
  const files = await readdir(JUDGE);
  const rows = [];
  const pattern = new RegExp(`^pass\\d+${variant ? `-${variant}` : ''}-(${Object.keys(PARTS).join('|')})\\.json$`);
  for (const file of files.filter((name) => pattern.test(name))) {
    const pass = file.match(/^pass(\d+)/)[1];
    const key = JSON.parse(await readFile(path.join(JUDGE, `${prefix(pass)}-key.json`), 'utf8'));
    for (const judgment of JSON.parse(await readFile(path.join(JUDGE, file), 'utf8'))) {
      const unit = judgment.item.replace(/-\d+$/, '');
      const model = key[judgment.item]?.[judgment.label] ?? key[unit]?.[judgment.label];
      if (!model) throw new Error(`${file}: no model for ${judgment.item} ${judgment.label}`);
      rows.push({ pass, model, ...judgment });
    }
  }
  const summary = {};
  for (const model of Object.keys(runs)) {
    const mine = rows.filter((row) => row.model === model);
    const answers = runs[model].answers;
    const byCategory = {};
    for (const category of data.categories.map((c) => c.id).filter((id) => answers.some((a) => a.category === id))) {
      const ids = new Set(answers.filter((a) => a.category === category).map((a) => a.id));
      const scored = mine.filter((row) => ids.has(row.item));
      byCategory[category] = {
        natural: mean(scored.map((row) => row.natural)),
        fit: mean(scored.map((row) => row.fit)),
        usableYes: mean(scored.map((row) => (row.usable === 'yes' ? 1 : 0))),
        usableAtLeastEdit: mean(scored.map((row) => (row.usable !== 'no' ? 1 : 0))),
      };
    }
    summary[model] = {
      name: modelName(model),
      judgments: mine.length,
      natural: mean(mine.map((row) => row.natural)),
      fit: mean(mine.map((row) => row.fit)),
      usableYes: mean(mine.map((row) => (row.usable === 'yes' ? 1 : 0))),
      usableAtLeastEdit: mean(mine.map((row) => (row.usable !== 'no' ? 1 : 0))),
      byCategory,
      flagged: answers.filter((a) => a.checks.problems.length).map((a) => ({ id: a.id, problems: a.checks.problems })),
      factsRight: answers.filter((a) => a.id.startsWith('knowledge-') || a.id === 'conv-running-3' || a.id === 'conv-kyoto-2').filter((a) => !a.checks.problems.includes('答えに必要な語がない')).length,
      decodePerSecond: runs[model].decodePerSecondMedian,
      meanNewTokens: mean(answers.map((a) => a.newTokens)),
      meanChars: mean(answers.map((a) => a.checks.length)),
    };
  }
  // Agreement between passes: share of (item, model) pairs whose usable verdicts match.
  const pairs = new Map();
  for (const row of rows) pairs.set(`${row.item}|${row.model}`, [...(pairs.get(`${row.item}|${row.model}`) ?? []), row]);
  const both = [...pairs.values()].filter((list) => list.length >= 2);
  const agreement = both.length ? mean(both.map((list) => (list[0].usable === list[1].usable ? 1 : 0))) : null;
  const naturalGap = both.length ? mean(both.map((list) => Math.abs(list[0].natural - list[1].natural))) : null;
  await writeFile(path.join(DIR, `summary${variant ? `-${variant}` : ''}.json`), `${JSON.stringify({ createdAt: new Date().toISOString(), agreement, naturalGap, summary }, null, 2)}\n`);

  const pct = (x) => (Number.isFinite(x) ? `${Math.round(x * 100)}%` : '–');
  const f1 = (x) => (Number.isFinite(x) ? x.toFixed(1) : '–');
  console.log(`passes agree on usable for ${pct(agreement)} of answers; natural differs by ${f1(naturalGap)} on average\n`);
  const categories = Object.keys(COLUMNS).filter((id) => Object.values(summary).some((s) => s.byCategory[id]));
  console.log(`| モデル | 自然さ | 合い方 | そのまま使える | 少し直せば使える（以上） | ${categories.map((id) => COLUMNS[id]).join(' | ')} | 事実（7 問） | 機械的な問題 | 速さ（CPU） |`);
  console.log(`| --- | ---: | ---: | ---: | ---: | ${categories.map(() => '---:').join(' | ')} | ---: | ---: | ---: |`);
  for (const [model, s] of Object.entries(summary).sort((a, b) => b[1].usableYes - a[1].usableYes)) {
    const cat = (id) => pct(s.byCategory[id]?.usableYes);
    console.log(`| ${s.name} | ${f1(s.natural)} | ${f1(s.fit)} | ${pct(s.usableYes)} | ${pct(s.usableAtLeastEdit)} | ${categories.map(cat).join(' | ')} | ${s.factsRight}/7 | ${s.flagged.length} | ${f1(s.decodePerSecond)} トークン/秒 |`);
  }
}

const [command, arg] = process.argv.slice(2);
if (command === 'pack') await pack(Number(arg) || 1);
else if (command === 'score') await score();
else {
  console.error('Usage: node scripts/ja-chat-judge.mjs pack <pass> | score');
  process.exit(1);
}
