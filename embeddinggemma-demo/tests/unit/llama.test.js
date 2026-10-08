import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ChatModel, DecisionModel, LIQUID_MODELS, decisionDecider } from '../../agent/llama.mjs';

/** Stands in for LlamaServer: records the requests and answers with `reply(path, body)`. */
function fakeServer(reply) {
  const requests = [];
  return {
    requests,
    async post(path, body) {
      requests.push({ path, body });
      return reply(path, body);
    },
    stop() {},
  };
}

test('the d1 decider asks one choice question over the candidates and picks the most probable', async () => {
  const server = fakeServer(() => ({
    answers: { pick: { type: 'choice', choice: '1', probabilities: { 0: 0.2, 1: 0.7, 2: 0.1 } } },
    usage: { input_tokens: 321, output_tokens: 0 },
  }));
  const decider = decisionDecider(new DecisionModel(server, 'd1-3B'));
  const candidates = [{ text: 'リンク「ホーム」を開く' }, { text: 'ボタン「変更を保存」を押す' }, { text: 'ボタン「キャンセル」を押す' }];
  const decision = await decider.decide({ step: { step: '設定を保存する', text: '' }, candidates });

  assert.equal(decision.index, 1);
  assert.deepEqual(decision.probs, [0.2, 0.7, 0.1]);
  assert.equal(decision.promptTokens, 321);
  const { path, body } = server.requests[0];
  assert.equal(path, '/v1/systemone');
  assert.equal(body.state, '手順: 設定を保存する');
  assert.equal(body.questions.pick.type, 'choice');
  assert.deepEqual(Object.values(body.questions.pick.criteria), candidates.map((candidate) => candidate.text));
});

test('the English wording only changes the words around the step', async () => {
  const server = fakeServer(() => ({ answers: { pick: { probabilities: { 0: 1 } } } }));
  const decider = decisionDecider(new DecisionModel(server, 'd1-omni-600M'), 'en');
  await decider.decide({ step: { step: '行き先に「京都」と入れる', text: '京都' }, candidates: [{ text: '入力欄「目的地」に入力する' }] });
  assert.equal(server.requests[0].body.state, 'Step: 行き先に「京都」と入れる（京都）');
  assert.match(decider.name, /en/);
});

test('a decision model can pick pull-down options like EmbeddingGemma does', async () => {
  const server = fakeServer(() => ({ answers: { pick: { probabilities: { 0: 0.1, 1: 0.9 } } } }));
  const scores = await new DecisionModel(server, 'd1-3B').similarities('大人の人数を2人にする', ['「大人」で「1名」を選ぶ', '「大人」で「2名」を選ぶ']);
  assert.deepEqual(scores, [0.1, 0.9]);
});

test('the chat model leaves room for thinking and returns only the answer', async () => {
  const server = fakeServer(() => ({
    choices: [{ message: { content: '<think>まず考える</think>\n["a"]', reasoning_content: '' }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 50, completion_tokens: 600 },
  }));
  const model = new ChatModel(server, 'lfm');
  const result = await model.chat([{ role: 'user', content: 'こんにちは' }], 8);

  assert.equal(result.text, '["a"]');
  assert.equal(result.newTokens, 600);
  assert.equal(server.requests[0].body.max_tokens, ChatModel.MIN_TOKENS);
  assert.equal(model.name, LIQUID_MODELS.lfm.name);
  assert.match(model.id, /LFM2\.5-2\.6B-Q4_K_M\.gguf@e7caca5/);
});
