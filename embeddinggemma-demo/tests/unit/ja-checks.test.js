import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { checkReply, countBullets, findLoop, includesAll, scriptStats, visibleLength } from '../../public/lib/ja-checks.js';
import { CHAT_MODELS, serverSpec } from '../../scripts/lib/chat-models.mjs';

const data = JSON.parse(readFileSync(new URL('../../public/data/ja-chat.json', import.meta.url), 'utf8'));

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
  for (const item of [...data.items, ...data.conversations]) {
    assert.ok(!ids.has(item.id), item.id);
    ids.add(item.id);
    assert.ok(data.categories.some((category) => category.id === item.category), item.id);
  }
  for (const conversation of data.conversations) assert.equal(conversation.turns.length, 3, conversation.id);
  for (const key of Object.keys(CHAT_MODELS)) {
    const spec = serverSpec(key);
    assert.match(spec.file, /\.gguf$/);
    assert.match(spec.revision, /^[0-9a-f]{40}$/);
    assert.ok(spec.args.includes('--jinja'));
  }
});
