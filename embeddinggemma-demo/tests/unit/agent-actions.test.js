import assert from 'node:assert/strict';
import { test } from 'node:test';

import { buildCandidates, chooseOption, desiredChecked, signature, stepQuery, textToType } from '../../agent/actions.mjs';

const snap = {
  url: 'http://127.0.0.1:5175/sites/settings/index.html',
  title: 'アカウント設定',
  heading: 'アカウント設定',
  elements: [
    { id: 1, role: 'textbox', name: '目的地・エリア', context: '', value: '' },
    { id: 2, role: 'combobox', name: '泊数', context: '', options: [{ text: '1泊' }, { text: '2泊', selected: true }] },
    { id: 3, role: 'switch', name: 'メール通知', context: '通知', checked: true },
    { id: 4, role: 'link', name: '注文履歴', context: 'アカウントメニュー' },
    { id: 5, role: 'button', name: '変更を保存', context: '' },
  ],
};

const options = (...texts) => texts.map((text) => ({ text }));

test('buildCandidates gives each element one operation and offers DONE only for steps without a value', () => {
  const candidates = buildCandidates(snap, { step: '設定を保存する', text: '' });
  assert.deepEqual(
    candidates.map((candidate) => candidate.op),
    ['TYPE_TEXT', 'SELECT', 'CHECK', 'CLICK', 'CLICK', 'DONE'],
  );
  assert.equal(candidates[2].text, 'スイッチ「メール通知」を切り替える（いまはオン）（通知）');
  assert.equal(candidates[3].text, 'リンク「注文履歴」を開く（アカウントメニュー）');

  const valued = buildCandidates(snap, { step: '行き先に「京都」と入れる', text: '京都' });
  assert.ok(valued.every((candidate) => candidate.op !== 'DONE'));
});

test('stepQuery appends the value to the step', () => {
  assert.equal(stepQuery({ step: '商品検索', text: 'ワイヤレスイヤホン' }), '商品検索（ワイヤレスイヤホン）');
  assert.equal(stepQuery({ step: '検索を実行する', text: '' }), '検索を実行する');
});

test('desiredChecked reads on or off from the wording and toggles otherwise', () => {
  assert.equal(desiredChecked({ step: 'メールでのお知らせを止める', text: '' }, true), false);
  assert.equal(desiredChecked({ step: '二段階認証を有効にする', text: '' }, false), true);
  assert.equal(desiredChecked({ step: '禁煙の部屋だけに絞り込む', text: '' }, false), true);
  assert.equal(desiredChecked({ step: 'ニュースレター', text: 'off' }, true), false);
  assert.equal(desiredChecked({ step: 'ダークモード', text: '' }, false), true);
});

test('textToType uses the planner value, then the first quoted part of the step', () => {
  assert.equal(textToType({ step: '検索ボックスに入れる', text: '東京タワー' }), '東京タワー');
  assert.equal(textToType({ step: '行き先に「京都」と入れる', text: '' }), '京都');
  assert.equal(textToType({ step: '検索を実行する', text: '' }), '');
});

test('chooseOption matches the option text before asking EmbeddingGemma', async () => {
  const checkin = { name: 'チェックイン', options: options('12月18日（金）', '12月19日（土）', '12月20日（日）') };
  const unused = { similarities: async () => assert.fail('a string match was expected') };
  assert.deepEqual(await chooseOption(checkin, { step: 'チェックイン日を12月19日にする', text: '' }, unused), { index: 1, how: '文字列一致' });

  // 「2人」 does not appear in 「2名」, so the embedding similarity decides.
  const adults = { name: '大人', options: options('1名', '2名', '3名') };
  const embedder = { similarities: async (query, docs) => docs.map((doc) => (doc.includes('2名') ? 0.8 : 0.2)) };
  assert.deepEqual(await chooseOption(adults, { step: '大人の人数を2人にする', text: '' }, embedder), { index: 1, how: 'EmbeddingGemma' });
});

test('signature identifies an element across snapshots by page, role, name and context', () => {
  assert.equal(
    signature('http://127.0.0.1:5175/sites/settings/index.html?saved=1', snap.elements[2]),
    '/sites/settings/index.html|switch|メール通知|通知',
  );
});
