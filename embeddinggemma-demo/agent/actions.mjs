// Turns the element table into candidate actions (operation + target, like Jev's heads),
// fills in the value for TYPE_TEXT / SELECT / CHECK, and executes the chosen action.

/** The text a decider compares with the candidates: the step, plus its value if it has one. */
export const stepQuery = (step) => (step.text ? `${step.step}（${step.text}）` : step.step);

const kindLabel = { checkbox: 'チェックボックス', switch: 'スイッチ', radio: 'ラジオボタン' };

function withContext(element) {
  return element.context && element.context !== element.name ? `（${element.context}）` : '';
}

/** One candidate per element; DONE ("this step is already satisfied") when the step carries no value. */
export function buildCandidates(snap, step) {
  const candidates = [];
  for (const element of snap.elements) {
    const context = withContext(element);
    switch (element.role) {
      case 'textbox':
      case 'searchbox':
        candidates.push({ op: 'TYPE_TEXT', element, text: `入力欄「${element.name}」に入力する${element.value ? `（いまの値「${element.value}」）` : ''}${context}` });
        break;
      case 'combobox':
        candidates.push({
          op: 'SELECT',
          element,
          text: `プルダウン「${element.name}」で選ぶ（${element.options.slice(0, 8).map((option) => option.text).join('／')}）`,
        });
        break;
      case 'checkbox':
      case 'switch':
        candidates.push({ op: 'CHECK', element, text: `${kindLabel[element.role]}「${element.name}」を切り替える（いまは${element.checked ? 'オン' : 'オフ'}）${context}` });
        break;
      case 'radio':
        candidates.push({ op: 'CLICK', element, text: `${kindLabel.radio}「${element.name}」を選ぶ${context}` });
        break;
      case 'link':
        candidates.push({ op: 'CLICK', element, text: `リンク「${element.name}」を開く${context}` });
        break;
      default:
        candidates.push({ op: 'CLICK', element, text: `ボタン「${element.name}」を押す${context}` });
    }
  }
  if (!step.text) {
    candidates.push({ op: 'DONE', element: null, text: `この手順はもう済んでいる（いまのページは「${snap.heading || snap.title}」）` });
  }
  return candidates;
}

/** Compact list of the page's elements for the planner: names, context and current state, no numbers. */
export function describeForPlanner(snap, limit = 50) {
  const kinds = { textbox: '入力欄', searchbox: '入力欄', combobox: 'プルダウン', checkbox: 'チェックボックス', switch: 'スイッチ', radio: 'ラジオボタン', link: 'リンク' };
  return snap.elements
    .slice(0, limit)
    .map((element) => {
      let detail = '';
      if (element.options) {
        const current = element.options.find((option) => option.selected)?.text;
        detail = `（選択中: ${current}／選べる項目: ${element.options.slice(0, 6).map((option) => option.text).join('、')}${element.options.length > 6 ? ' など' : ''}）`;
      } else if (element.checked !== undefined) {
        detail = `（いまは${element.checked ? 'オン' : 'オフ'}）`;
      } else if (element.value) {
        detail = `（入力済み: ${element.value}）`;
      }
      const context = element.context && element.context !== element.name ? `［${element.context}］` : '';
      return `- ${kinds[element.role] ?? 'ボタン'}「${element.name}」${detail}${context}`;
    })
    .join('\n');
}

/** Operations that consume the step's value (text to type, option to pick, on/off). */
export const VALUE_OPS = new Set(['TYPE_TEXT', 'SELECT', 'CHECK']);

/** Identifies an element across snapshots (ids are renumbered every time). */
export const signature = (url, element) => `${new URL(url).pathname}|${element.role}|${element.name}|${element.context}`;

const normalize = (text) => (text ?? '').normalize('NFKC').replace(/\s+/g, '').toLowerCase();
const stripParens = (text) => text.replace(/（[^）]*）|\([^)]*\)/g, '');

/** Picks the option named in the step (string match first; then the scores of `embedder`, EmbeddingGemma or a decision model). */
export async function chooseOption(element, step, embedder) {
  const value = normalize(step.text);
  const haystack = normalize(`${step.step}${step.text}`);
  let best = null;
  element.options.forEach((option, index) => {
    const text = normalize(stripParens(option.text));
    if (!text) return;
    const hit = (value && (text === value || value.includes(text) || text.includes(value))) || haystack.includes(text);
    if (hit && (!best || text.length > best.length)) best = { index, length: text.length };
  });
  if (best) return { index: best.index, how: '文字列一致' };
  const sims = await embedder.similarities(
    stepQuery(step),
    element.options.map((option) => `「${element.name}」で「${option.text}」を選ぶ`),
  );
  return { index: sims.indexOf(Math.max(...sims)), how: embedder.name ?? 'EmbeddingGemma' };
}

/** Desired switch state from the step wording; toggles when the step does not say. */
export function desiredChecked(step, current) {
  const text = `${step.step} ${step.text}`;
  if (/オフ|無効|外す|解除|やめ|止め|停止|\boff\b/i.test(text)) return false;
  if (/オン|有効|つける|付ける|入れる|チェック|絞り込|\bon\b/i.test(text)) return true;
  return !current;
}

/** Text to type: the planner's value, else the first 「quoted」 part of the step. */
export function textToType(step) {
  return step.text || step.step.match(/「([^」]+)」/)?.[1] || '';
}

export async function execute(page, action) {
  const target = action.element ? page.locator(`[data-gj-id="${action.element.id}"]`) : null;
  const before = page.url();
  switch (action.op) {
    case 'CLICK':
      await target.click({ timeout: 5000 });
      break;
    case 'TYPE_TEXT':
      await target.fill(action.value, { timeout: 5000 });
      break;
    case 'SELECT':
      await target.selectOption({ index: action.optionIndex }, { timeout: 5000 });
      break;
    case 'CHECK':
      await target.setChecked(action.checked, { timeout: 5000 });
      break;
    default:
      return;
  }
  // Wait for a navigation the action may have started, then for the page to settle.
  await page.waitForTimeout(300);
  await page.waitForLoadState('domcontentloaded', { timeout: 15000 }).catch(() => {});
  if (page.url() !== before) await page.waitForLoadState('load', { timeout: 15000 }).catch(() => {});
  await page.waitForTimeout(300);
}

/** One-line description of an action for logs and the inspector. */
export function describe(action) {
  const name = action.element ? `[${action.element.id}] ${action.element.name}` : '';
  switch (action.op) {
    case 'TYPE_TEXT':
      return `${name} に「${action.value}」と入力`;
    case 'SELECT':
      return `${name} で「${action.optionText}」を選択`;
    case 'CHECK':
      return `${name} を${action.checked ? 'オン' : 'オフ'}`;
    case 'CLICK':
      return `${name} をクリック`;
    case 'DONE':
      return action.element ? `${name} は設定済みなので何もしない` : 'この手順は完了済みとして次へ';
    default:
      return action.op;
  }
}
