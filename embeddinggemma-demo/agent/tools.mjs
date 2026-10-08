// Tool-calling agent, the way LFM2.5-2.6B is meant to work: the model reads the page's numbered
// elements and calls tools that act on them directly. There is no separate decider, unlike the
// planner + decider split of the other modes (agent/run.mjs --planner lfm-tools).
import { describeToday } from './gemma.mjs';

export const TOOL_PROMPT = `あなたはWebブラウザを操作するエージェントです。目標を達成するために、いまのページの要素を道具で操作してください。
- 要素は番号で指定する。いまのページの一覧にある番号だけを使う。
- 一度に複数の道具を呼んでよい。ページが変わる操作（検索ボタンやリンク）は最後に呼ぶ。
- 目標がすべて達成されたら finish だけを呼ぶ。
- 「先月」「来週」などは、今日の日付をもとに具体的な日付に直して考える。`;

const element = { type: 'integer', description: '操作する要素の番号' };
const tool = (name, description, properties = {}) => ({
  type: 'function',
  function: { name, description, parameters: { type: 'object', properties, required: Object.keys(properties) } },
});

export const TOOLS = [
  tool('type_text', '入力欄に文字を入力する（いまの値は置き換える）', { element, text: { type: 'string', description: '入力する文字列' } }),
  tool('select_option', 'プルダウンで項目を選ぶ', { element, option: { type: 'string', description: '選ぶ項目の表示名' } }),
  tool('set_switch', 'スイッチやチェックボックスをオンまたはオフにする', { element, on: { type: 'boolean', description: 'オンにするなら true' } }),
  tool('click', 'ボタンやリンクを押す', { element }),
  tool('finish', '目標がすべて達成されたので終える'),
];

const KINDS = { textbox: '入力欄', searchbox: '入力欄', combobox: 'プルダウン', checkbox: 'チェックボックス', switch: 'スイッチ', radio: 'ラジオボタン', link: 'リンク' };

/** The page's elements with their numbers, current state and every option of a pull-down. */
export function describeForTools(snap) {
  return snap.elements
    .map((el) => {
      let detail = '';
      if (el.options) {
        const current = el.options.find((option) => option.selected)?.text;
        detail = `（選択中: ${current}／項目: ${el.options.map((option) => option.text).join('、')}）`;
      } else if (el.checked !== undefined) {
        detail = `（いまは${el.checked ? 'オン' : 'オフ'}）`;
      } else if (el.value) {
        detail = `（入力済み: ${el.value}）`;
      }
      const context = el.context && el.context !== el.name ? `［${el.context}］` : '';
      return `[${el.id}] ${KINDS[el.role] ?? 'ボタン'}「${el.name}」${detail}${context}`;
    })
    .join('\n');
}

/** Asks the model for the next tool calls on the current page. */
export async function nextToolCalls(model, { goal, site, today, pageTitle, pageElements, history }) {
  const done = history.length ? history.map((line) => `- ${line}`).join('\n') : '（まだ何もしていない）';
  const result = await model.chatWithTools(
    [
      { role: 'system', content: TOOL_PROMPT },
      {
        role: 'user',
        content: `今日: ${describeToday(today)}\nサイト: ${site}\n目標: ${goal}\n\nこれまでの操作:\n${done}\n\nいまのページ「${pageTitle}」の要素:\n${pageElements}`,
      },
    ],
    TOOLS,
  );
  return result;
}

const normalize = (text) => (text ?? '').normalize('NFKC').replace(/\s+/g, '').toLowerCase();

/** Turns one tool call into an action for execute() in actions.mjs, or { op: 'BLOCKED', reason }. */
export function toolAction(call, snap) {
  if (call.name === 'finish') return { op: 'FINISH' };
  const target = snap.elements.find((el) => el.id === Number(call.args.element));
  if (!target) return { op: 'BLOCKED', reason: `番号 ${call.args.element} の要素がない` };
  switch (call.name) {
    case 'type_text':
      if (!['textbox', 'searchbox'].includes(target.role)) return { op: 'BLOCKED', reason: `[${target.id}] は入力欄ではない` };
      return { op: 'TYPE_TEXT', element: target, value: String(call.args.text ?? '') };
    case 'select_option': {
      if (!target.options) return { op: 'BLOCKED', reason: `[${target.id}] はプルダウンではない` };
      const wanted = normalize(call.args.option);
      const index = target.options.findIndex((option) => normalize(option.text) === wanted);
      const loose = index >= 0 ? index : target.options.findIndex((option) => normalize(option.text).includes(wanted) || wanted.includes(normalize(option.text)));
      if (loose < 0) return { op: 'BLOCKED', reason: `[${target.id}] に「${call.args.option}」という項目がない` };
      return { op: 'SELECT', element: target, optionIndex: loose, optionText: target.options[loose].text };
    }
    case 'set_switch':
      if (target.checked === undefined) return { op: 'BLOCKED', reason: `[${target.id}] はスイッチではない` };
      return { op: 'CHECK', element: target, checked: call.args.on === true || call.args.on === 'true' };
    case 'click':
      return { op: 'CLICK', element: target };
    default:
      return { op: 'BLOCKED', reason: `知らない道具 ${call.name}` };
  }
}
