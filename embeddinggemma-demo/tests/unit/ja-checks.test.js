import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { checkReply, countBullets, findLoop, includesAll, scriptStats, visibleLength } from '../../public/lib/ja-checks.js';
import { CHAT_MODELS, GGUF_MODEL_KEYS, serverSpec } from '../../scripts/lib/chat-models.mjs';
import { requestOptions } from '../../scripts/lib/chat-server.mjs';

const data = JSON.parse(readFileSync(new URL('../../public/data/ja-chat.json', import.meta.url), 'utf8'));
const talk = JSON.parse(readFileSync(new URL('../../public/data/ja-talk.json', import.meta.url), 'utf8'));

test('script stats count Japanese letters and flag Chinese and Korean', () => {
  assert.equal(scriptStats('了解！少し遅れるね').jaRatio, 1);
  assert.equal(scriptStats('OKです').jaRatio, 2 / 4);
  assert.equal(scriptStats('我们明天见。').foreign, '们见');
  assert.equal(scriptStats('감사합니다').foreign.length, 5);
  assert.equal(scriptStats('ありがとう').foreign, '');
});

test('a loop is a long stretch repeated three times', () => {
  assert.equal(findLoop('今日はいい天気ですね。散歩に行きましょう。'.repeat(3)), '今日はいい天気ですね。散');
  assert.ok(findLoop('ありがとうございます。'.repeat(6)));
  assert.equal(findLoop('ありがとうございます。またね。'), null);
});

test('length, bullets and required words', () => {
  assert.equal(visibleLength('「図書館は本日臨時休館です」'), 12);
  assert.equal(countBullets('- トースト\n- ご飯と味噌汁\n・ヨーグルト'), 3);
  assert.equal(countBullets('1. トースト\n2) ご飯\n③ パン\n説明'), 3);
  assert.ok(includesAll('富士山で、3,776 mです', [['富士山'], ['3776', '3,776']]));
  assert.ok(!includesAll('富士山です', [['富士山'], ['3776']]));
});

test('checkReply reports each problem in words', () => {
  const summary = data.items.find((item) => item.id === 'rewrite-summary');
  assert.deepEqual(checkReply('図書館は今日臨時休館、返却はポストへ。', summary).problems, []);
  assert.match(checkReply('本日、駅前の図書館は設備点検のため臨時休館となります。返却は返却ポストへ。', summary).problems.join(), /30 文字を超えた/);
  assert.match(checkReply('我们明天见', {}).problems.join(), /他の言語/);
  assert.deepEqual(checkReply('', {}).problems.slice(0, 1), ['空の返答']);
});

test('every test item is well formed and every model can be served', () => {
  const ids = new Set();
  const categories = [...data.categories, ...talk.categories];
  for (const item of [...data.items, ...data.conversations, ...talk.conversations]) {
    assert.ok(!ids.has(item.id), item.id);
    ids.add(item.id);
    assert.ok(categories.some((category) => category.id === item.category), item.id);
  }
  for (const conversation of data.conversations) assert.equal(conversation.turns.length, 3, conversation.id);
  for (const conversation of talk.conversations) assert.ok(conversation.turns.length >= 4, conversation.id);
  for (const [key, model] of Object.entries(CHAT_MODELS)) {
    assert.match(model.revision, /^[0-9a-f]{40}$/);
    if (model.runtime === 'litert-lm') {
      assert.match(model.file, /\.litertlm$/);
      assert.throws(() => serverSpec(key), /LiteRT-LM/);
    } else {
      const spec = serverSpec(key);
      assert.match(spec.file, /\.gguf$/);
      assert.ok(spec.args.includes('--jinja'));
    }
  }
  assert.equal(GGUF_MODEL_KEYS.length, 7);
});

test('requests turn thinking off the way each server understands', () => {
  const gguf = requestOptions(CHAT_MODELS['gemma4-e4b'], { maxTokens: 512, seed: 0 });
  assert.deepEqual(gguf.chat_template_kwargs, { enable_thinking: false });
  assert.equal(gguf.max_tokens, 512);
  const phone = requestOptions(CHAT_MODELS['gemma4-e4b-litert'], { maxTokens: 512, seed: 0 });
  assert.equal(phone.reasoning_effort, 'none');
  assert.equal(phone.max_completion_tokens, 512);
  assert.equal(phone.chat_template_kwargs, undefined);
  assert.deepEqual({ temperature: phone.temperature, top_p: phone.top_p, top_k: phone.top_k }, { temperature: 1.0, top_p: 0.95, top_k: 64 });
});
