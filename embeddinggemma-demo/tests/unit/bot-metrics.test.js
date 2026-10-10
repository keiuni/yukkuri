import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { ACTIONS, analyzeConversation, similarity } from '../../public/lib/bot-metrics.js';

const scenes = JSON.parse(readFileSync(new URL('../../public/data/bot-scenes.json', import.meta.url), 'utf8')).scenes;

test('similarity is 1 for the same words and ignores punctuation and the action line', () => {
  assert.equal(similarity('温泉に行こう！', '温泉に行こう。'), 1);
  assert.equal(similarity('温泉に行こう\n行動: 歩く', '温泉に行こう\n行動: 待つ'), 1);
  assert.ok(similarity('温泉に行こう', '山でキャンプしたい') < 0.2);
});

test('echoes, role slips, own-name labels and goodbyes are counted', () => {
  const turns = [
    { speaker: 'ミナ', text: '今度の週末どこか行かない？' },
    { speaker: 'ケンタ', text: 'ケンタ：静かな温泉がいいな。' },
    { speaker: 'ミナ', text: '温泉いいね！\nケンタ：じゃあ決まりだね。' },
    { speaker: 'ケンタ', text: 'ミナさん、静かな温泉がいいな。' },
    { speaker: 'ミナ', text: 'ケンタさん、またね！' },
    { speaker: 'ケンタ', text: 'ケンタさん、またね！' },
  ];
  const m = analyzeConversation(turns, ['ミナ', 'ケンタ']);
  assert.equal(m.labelsSelf, 1);
  assert.equal(m.roleSlips, 2); // Mina speaks for Kenta; Kenta calls himself "ケンタさん"
  assert.ok(m.echoes >= 1);
  assert.equal(m.farewells, 2);
  assert.equal(m.firstFarewell, 5);
  assert.equal(m.messagesAfterFirstFarewell, 1);
});

test('the action format is checked against the allowed actions', () => {
  const turns = [
    { speaker: 'ノア', text: 'こんにちは。\n行動: 手を振る' },
    { speaker: 'リク', text: '小麦をください。\n行動: 踊る' },
  ];
  assert.equal(analyzeConversation(turns, ['ノア', 'リク'], { actionFormat: true }).actionAdherence, 0.5);
  assert.ok(ACTIONS.includes('手を振る'));
});

test('every scene has two bots with a goal and a first line', () => {
  for (const scene of scenes) {
    assert.equal(scene.bots.length, 2, scene.id);
    for (const bot of scene.bots) assert.ok(bot.name && bot.persona && bot.situation && bot.goal, scene.id);
    assert.ok(scene.opener, scene.id);
  }
});
