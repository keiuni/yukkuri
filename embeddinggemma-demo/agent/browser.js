// Code that runs inside the page. Both functions are passed to Playwright as-is, so they
// must not reference anything outside their own body.

/**
 * Atomic snapshot of the rendered interactive elements, like jev-ultrafast's snapshot.js:
 * every element gets a number (data-gj-id) and a role, an accessible name, its current
 * value/state and the heading of the card or section it sits in.
 */
export function snapshot() {
  const SELECTOR =
    'a[href], button, input:not([type=hidden]), select, textarea, summary, [role=button], [role=link], [role=tab], [role=switch], [role=checkbox], [role=menuitem]';
  const clean = (text) => (text ?? '').replace(/\s+/g, ' ').trim();
  const ownText = (node) =>
    clean([...node.childNodes].filter((child) => child.nodeType === Node.TEXT_NODE).map((child) => child.textContent).join(' '));

  const isRendered = (el) => {
    if (el.closest('#gj-panel')) return false;
    const rect = el.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return false;
    const style = getComputedStyle(el);
    return style.visibility !== 'hidden' && style.display !== 'none' && Number(style.opacity) > 0.05;
  };
  const inViewport = (el) => {
    const rect = el.getBoundingClientRect();
    return rect.bottom > 0 && rect.top < innerHeight && rect.right > 0 && rect.left < innerWidth;
  };

  const nameOf = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria) return clean(aria);
    const labelledBy = el.getAttribute('aria-labelledby');
    if (labelledBy) return clean(labelledBy.split(/\s+/).map((id) => document.getElementById(id)?.textContent ?? '').join(' '));
    if (el.id) {
      const label = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (label) return clean(label.textContent);
    }
    const wrapper = el.closest('label');
    if (wrapper && wrapper !== el) {
      const text = ownText(wrapper);
      if (text) return text;
    }
    if (el.matches('input, textarea')) return clean(el.placeholder || el.title || el.name);
    const text = clean(el.innerText || el.textContent);
    if (text) return text.slice(0, 80);
    return clean(el.title || el.querySelector('img[alt]')?.alt || el.value || '');
  };

  const CONTAINERS = 'article, li, section, aside, nav, form, header, footer, tr, [role=dialog]';
  const contextOf = (el) => {
    const container = el.closest(CONTAINERS);
    if (!container) return '';
    // The container's own heading, not one inside a nested card (a sort menu above a list of hotels
    // would otherwise be labelled with the first hotel's name).
    const heading = [...container.querySelectorAll('h1, h2, h3, h4, legend, caption')].find((h) => h.closest(CONTAINERS) === container);
    return clean(heading?.textContent || container.getAttribute('aria-label') || '').slice(0, 60);
  };

  const roleOf = (el) => {
    const explicit = el.getAttribute('role');
    if (el.tagName === 'SELECT') return 'combobox';
    if (el.tagName === 'TEXTAREA') return 'textbox';
    if (el.tagName === 'INPUT') {
      const type = (el.getAttribute('type') || 'text').toLowerCase();
      if (type === 'checkbox') return explicit === 'switch' ? 'switch' : 'checkbox';
      if (type === 'radio') return 'radio';
      if (['submit', 'button', 'reset', 'image'].includes(type)) return 'button';
      return type === 'search' ? 'searchbox' : 'textbox';
    }
    if (explicit) return explicit;
    if (el.tagName === 'A') return 'link';
    return 'button';
  };

  for (const old of document.querySelectorAll('[data-gj-id]')) old.removeAttribute('data-gj-id');
  // Elements on screen first, then the rest of the page in document order (Playwright scrolls when acting).
  const rendered = [...document.querySelectorAll(SELECTOR)].filter((el) => isRendered(el) && !el.disabled);
  const ordered = [...rendered.filter(inViewport), ...rendered.filter((el) => !inViewport(el))];
  const elements = [];
  for (const el of ordered) {
    const id = elements.length + 1;
    el.setAttribute('data-gj-id', String(id));
    const role = roleOf(el);
    const item = { id, role, name: nameOf(el), context: contextOf(el), inView: inViewport(el) };
    if (role === 'textbox' || role === 'searchbox') item.value = el.value;
    if (role === 'checkbox' || role === 'switch' || role === 'radio') item.checked = el.checked;
    if (role === 'combobox') item.options = [...el.options].map((option) => ({ text: clean(option.textContent), selected: option.selected }));
    elements.push(item);
    if (elements.length >= 150) break;
  }
  const heading = document.querySelector('h1');
  // A little of the visible text (notices, result counts, the article lead) for the completion check.
  const root = document.querySelector('main, [role=main]') ?? document.body;
  const panelText = document.getElementById('gj-panel')?.innerText ?? '';
  const text = clean((root.innerText ?? '').replace(panelText, '')).slice(0, 400);
  return { url: location.href, title: document.title, heading: clean(heading?.textContent ?? ''), text, elements };
}

/** Installs the inspector panel (decisions, plan, probabilities) and the element markers. */
export function installInspector() {
  const install = () => {
    if (document.getElementById('gj-panel')) return;
    const style = document.createElement('style');
    style.textContent = `
      html { margin-right: 430px !important; }
      #gj-panel { position: fixed; top: 0; right: 0; bottom: 0; width: 430px; z-index: 2147483646; overflow: hidden;
        background: #111214; color: #eceae4; font: 13px/1.5 system-ui, "Noto Sans JP", "Hiragino Sans", sans-serif;
        display: flex; flex-direction: column; border-left: 1px solid #2c2c2a; }
      #gj-panel header { padding: 12px 16px; border-bottom: 1px solid #2c2c2a; }
      #gj-panel header strong { font-size: 15px; }
      #gj-panel header .who { color: #9a9890; font-size: 12px; }
      #gj-panel section { padding: 10px 16px; border-bottom: 1px solid #2c2c2a; }
      #gj-panel h4 { margin: 0 0 6px; font-size: 11px; letter-spacing: 0.04em; color: #9a9890; font-weight: 600; }
      #gj-panel .goal { font-size: 14px; font-weight: 600; }
      #gj-panel ol { margin: 0; padding-left: 20px; }
      #gj-panel li { margin: 2px 0; color: #9a9890; }
      #gj-panel li.done { color: #6fcf8f; }
      #gj-panel li.current { color: #fff; font-weight: 600; }
      #gj-panel li.skipped { color: #9a9890; text-decoration: line-through; }
      #gj-panel .value { color: #f0b47a; }
      #gj-panel .op { display: inline-block; padding: 0 6px; border-radius: 4px; background: #eb6834; color: #fff; font-weight: 700; font-size: 11px; margin-right: 6px; }
      #gj-panel .cand { display: grid; grid-template-columns: 34px minmax(0, 1fr) 46px; gap: 6px; align-items: center; margin: 4px 0; }
      #gj-panel .cand .id { color: #9a9890; font-variant-numeric: tabular-nums; }
      #gj-panel .cand .label { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      #gj-panel .cand .p { text-align: right; font-variant-numeric: tabular-nums; }
      #gj-panel .bar { grid-column: 2 / 4; height: 4px; border-radius: 2px; background: #2c2c2a; }
      #gj-panel .bar i { display: block; height: 100%; border-radius: 2px; background: #3987e5; }
      #gj-panel .cand.chosen .label { color: #fff; font-weight: 700; }
      #gj-panel .cand.chosen .bar i { background: #eb6834; }
      #gj-panel .meta { color: #9a9890; font-size: 12px; }
      #gj-panel .status { margin-top: auto; padding: 12px 16px; font-weight: 700; font-size: 14px; }
      #gj-panel .status.ok { background: #0f3d24; color: #8ff0b0; }
      #gj-panel .status.ng { background: #4a1616; color: #ffb4b4; }
      #gj-panel .ff { display: inline-block; padding: 0 8px; border-radius: 4px; background: #eb6834; color: #fff; font-size: 12px; margin-right: 6px; }
      .gj-mark { position: absolute; z-index: 2147483645; pointer-events: none; font: 700 10px/14px system-ui, sans-serif;
        padding: 0 3px; border-radius: 3px; background: rgba(57, 135, 229, 0.85); color: #fff; }
      .gj-mark.chosen { background: #eb6834; font-size: 12px; line-height: 16px; }
      .gj-outline { position: absolute; z-index: 2147483644; pointer-events: none; border: 3px solid #eb6834; border-radius: 6px;
        box-shadow: 0 0 0 4px rgba(235, 104, 52, 0.25); }
    `;
    document.head.append(style);
    const panel = document.createElement('aside');
    panel.id = 'gj-panel';
    document.body.append(panel);

    const layer = document.createElement('div');
    layer.id = 'gj-layer';
    document.body.append(layer);

    const text = (tag, className, value) => Object.assign(document.createElement(tag), { className, textContent: value });

    window.__gj = {
      render(state) {
        panel.replaceChildren();
        const header = document.createElement('header');
        header.append(text('strong', '', 'Gemma ブラウザエージェント'), document.createElement('br'));
        header.append(text('span', 'who', `計画: ${state.plannerName}　判断（Jev の役）: ${state.deciderName}`));
        panel.append(header);

        const goal = document.createElement('section');
        goal.append(text('h4', '', '目標'), text('div', 'goal', state.goal));
        if (state.expect) goal.append(text('div', 'meta', `正しい動き: ${state.expect}`));
        panel.append(goal);

        const plan = document.createElement('section');
        plan.append(text('h4', '', state.planMs ? `計画（${(state.planMs / 1000).toFixed(1)} 秒）` : '計画'));
        if (state.planning) plan.append(text('div', 'meta', state.planning));
        const list = document.createElement('ol');
        (state.steps ?? []).forEach((step, index) => {
          const item = document.createElement('li');
          item.className = state.stepStatus?.[index] ?? '';
          item.append(step.step);
          if (step.text) item.append(text('span', 'value', `「${step.text}」`));
          list.append(item);
        });
        plan.append(list);
        panel.append(plan);

        if (state.decision) {
          const decision = document.createElement('section');
          decision.append(text('h4', '', `判断（${state.decision.ms.toFixed(0)} ms・候補 ${state.decision.candidates} 件）`));
          const chosen = document.createElement('div');
          chosen.append(text('span', 'op', state.decision.op), state.decision.summary);
          decision.append(chosen);
          for (const candidate of state.decision.top) {
            const row = document.createElement('div');
            row.className = `cand${candidate.chosen ? ' chosen' : ''}`;
            const bar = document.createElement('div');
            bar.className = 'bar';
            const fill = document.createElement('i');
            fill.style.width = `${Math.round(candidate.p * 100)}%`;
            bar.append(fill);
            // EmbeddingGemma shows its cosine similarity: the probabilities are a softmax of it.
            const score = candidate.sim === undefined ? `${(candidate.p * 100).toFixed(0)}%` : candidate.sim.toFixed(3);
            row.append(text('span', 'id', candidate.id ? `[${candidate.id}]` : '—'), text('span', 'label', candidate.label), text('span', 'p', score), bar);
            decision.append(row);
          }
          panel.append(decision);
        }
        if (state.note) {
          const note = document.createElement('section');
          note.append(text('div', 'meta', state.note));
          panel.append(note);
        }
        if (state.result) panel.append(text('div', `status ${state.result.ok ? 'ok' : 'ng'}`, state.result.text));
        if (state.fastForward) {
          const ff = document.createElement('section');
          ff.append(text('span', 'ff', '▶▶ 早送り'), state.fastForward);
          panel.append(ff);
        }
      },
      marks(ids, chosenId) {
        layer.replaceChildren();
        for (const id of ids) {
          const el = document.querySelector(`[data-gj-id="${id}"]`);
          if (!el) continue;
          const rect = el.getBoundingClientRect();
          const chosen = id === chosenId;
          if (chosen) {
            const outline = document.createElement('div');
            outline.className = 'gj-outline';
            Object.assign(outline.style, { left: `${rect.left + scrollX - 4}px`, top: `${rect.top + scrollY - 4}px`, width: `${rect.width + 8}px`, height: `${rect.height + 8}px` });
            layer.append(outline);
          }
          const mark = text('div', `gj-mark${chosen ? ' chosen' : ''}`, String(id));
          Object.assign(mark.style, { left: `${rect.left + scrollX - 6}px`, top: `${rect.top + scrollY - (chosen ? 18 : 12)}px` });
          layer.append(mark);
        }
      },
    };
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', install);
  else install();
}
