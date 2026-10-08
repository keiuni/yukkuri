#!/usr/bin/env node
// Gemma browser agent: the jev-ultrafast loop with Gemma models in place of Jev.
// For every step, EmbeddingGemma picks the operation and the target from the numbered
// element table (Jev's job); Playwright executes it; a task-specific check verifies the
// outcome at the end. Where the steps come from:
//   --planner script (default)  sub-goals written in the task, the way a coding agent such
//                               as Claude Code hands them to Jev through jev-mcp
//   --planner gemma4            everything local: Gemma 4 E2B plans each page and decides
//                               when the goal is met (Jev's DONE)
//
//   npm run agent                               # every task
//   npm run agent -- hotel wikipedia            # only these tasks
//   npm run agent -- --embedding v2             # EmbeddingGemma 2 instead of the first model
//   npm run agent -- --decider gemma4           # Gemma 4 E2B picks the targets instead (comparison)
//   npm run agent -- --planner gemma4           # Gemma 4 E2B also writes the steps
//   npm run agent -- --video                    # record test-output/agent/videos/*.mp4
//   npm run agent -- --replan                   # ignore cached Gemma 4 answers
import { createHash } from 'node:crypto';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { chromium } from '@playwright/test';

import { startServer, toMp4 } from '../scripts/lib/video.mjs';
import { VALUE_OPS, buildCandidates, chooseOption, describe, describeForPlanner, desiredChecked, execute, signature, textToType } from './actions.mjs';
import { installInspector, snapshot } from './browser.js';
import { DEFAULT_MODEL, MODELS } from '../public/lib/model-config.js';
import { DONE_PROMPT, EmbeddingGemma, GEMMA4_ID, Gemma4, PLANNER_PROMPT, checkDone, embeddingDecider, gemma4Decider, planSteps } from './gemma.mjs';
import { TASKS, TODAY } from './tasks.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'test-output', 'agent');
const PORT = Number(process.env.PORT ?? 5175);
const BASE_URL = `http://127.0.0.1:${PORT}`;
const VIEWPORT = { width: 1600, height: 900 };
// Below this cosine similarity the best candidate is not trusted and the step is skipped.
const MIN_SIMILARITY = 0.3;
// Safety limits for --planner gemma4: planner calls and executed steps per task.
const MAX_ROUNDS = 6;
const MAX_STEPS = 16;

const args = process.argv.slice(2);
const flag = (name) => args.includes(`--${name}`);
const option = (name, fallback) => {
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : fallback;
};
const deciderName = option('decider', 'embedding');
const plannerName = option('planner', 'script');
const embeddingModel = option('embedding', DEFAULT_MODEL);
const video = flag('video');
const wanted = args.filter((arg, i) => !arg.startsWith('--') && !['--decider', '--planner', '--embedding'].includes(args[i - 1]));
if (!MODELS[embeddingModel]) {
  console.error(`Unknown embedding model. Choose from: ${Object.keys(MODELS).join(', ')}`);
  process.exit(1);
}
const tasks = wanted.length ? TASKS.filter((task) => wanted.includes(task.id)) : TASKS;
if (tasks.length === 0) {
  console.error(`Unknown task. Choose from: ${TASKS.map((task) => task.id).join(', ')}`);
  process.exit(1);
}
const runName = `${plannerName}-${deciderName}${embeddingModel === DEFAULT_MODEL ? '' : `-${embeddingModel}`}`;

await mkdir(OUT, { recursive: true });
const cacheFile = path.join(OUT, 'gemma4-cache.json');
const gemma4Cache = JSON.parse(await readFile(cacheFile, 'utf8').catch(() => '{}'));

const needsGemma4 = plannerName === 'gemma4' || deciderName === 'gemma4';
console.log(`Loading ${MODELS[embeddingModel].name}${needsGemma4 ? ' and Gemma 4 E2B' : ''}…`);
const [embedder, gemma4] = await Promise.all([EmbeddingGemma.load(embeddingModel), needsGemma4 ? Gemma4.load() : null]);
const decider = deciderName === 'gemma4' ? gemma4Decider(gemma4) : embeddingDecider(embedder);
const plannerLabel = plannerName === 'gemma4' ? gemma4.name : '手順書（Claude が作成）';
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, video ? ms : 0));

/** Gemma 4 output is deterministic (greedy decoding), so the same model, prompts and input reuse the stored answer. */
async function cachedCall(kind, fn, input) {
  const key = createHash('sha1').update(JSON.stringify([kind, GEMMA4_ID, PLANNER_PROMPT, DONE_PROMPT, input])).digest('hex');
  if (!video && !flag('replan') && gemma4Cache[key]) return { ...gemma4Cache[key], cached: true };
  const result = await fn(gemma4, input);
  gemma4Cache[key] = result;
  await writeFile(cacheFile, `${JSON.stringify(gemma4Cache, null, 2)}\n`);
  return result;
}

/** Turns the decider's choice into a concrete action with its value. */
async function toAction(chosen, similarity, step) {
  if (!chosen || (similarity !== undefined && similarity < MIN_SIMILARITY)) return { op: 'BLOCKED', reason: '確信の持てる候補がない' };
  switch (chosen.op) {
    case 'TYPE_TEXT': {
      const value = textToType(step);
      return value ? { op: 'TYPE_TEXT', element: chosen.element, value } : { op: 'BLOCKED', reason: '入力する文字列がない' };
    }
    case 'SELECT': {
      const picked = await chooseOption(chosen.element, step, embedder);
      return { op: 'SELECT', element: chosen.element, optionIndex: picked.index, optionText: chosen.element.options[picked.index].text, optionHow: picked.how };
    }
    case 'CHECK':
      return { op: 'CHECK', element: chosen.element, checked: desiredChecked(step, chosen.element.checked) };
    default:
      return { op: chosen.op, element: chosen.element };
  }
}

async function runTask(browser, task, number) {
  const rawDir = path.join(OUT, 'videos', 'raw');
  const context = await browser.newContext({ viewport: VIEWPORT, locale: 'ja-JP', ...(video ? { recordVideo: { dir: rawDir, size: VIEWPORT } } : {}) });
  await context.addInitScript(installInspector);
  const page = await context.newPage();
  const started = Date.now();
  const seconds = () => (Date.now() - started) / 1000;
  const edits = [];

  const state = { goal: task.goal, plannerName: plannerLabel, deciderName: decider.name, steps: [], stepStatus: [], planMs: 0 };
  const render = (extra = {}) => page.evaluate((s) => window.__gj?.render(s), { ...state, ...extra }).catch(() => {});

  await page.goto(task.start.startsWith('http') ? task.start : BASE_URL + task.start, { waitUntil: 'load' });
  await render();
  await pause(1200);

  const history = []; // what has been done, in words, for the planner
  const changed = new Set(); // fields set by an earlier step are not set twice
  const log = [];
  const rounds = [];

  /** One Jev-style decision: element table -> candidates -> decider -> action -> execute. */
  async function performStep(step, index, round) {
    state.stepStatus[index] = 'current';
    await render();
    const snap = await page.evaluate(snapshot);
    const all = buildCandidates(snap, step);
    // Like Jev's per-operation heads: a step that carries a value is an input, a selection or a switch.
    // Only the explicit `text` counts as a value; 「quoted」 words in a step are often element names.
    const valued = step.text ? all.filter((candidate) => VALUE_OPS.has(candidate.op)) : [];
    let candidates = valued.length ? valued : all;
    let decision = await decider.decide({ goal: task.goal, step, candidates });
    if (valued.length && decision.sims && decision.sims[decision.index] < MIN_SIMILARITY) {
      candidates = all;
      decision = await decider.decide({ goal: task.goal, step, candidates });
    }
    const similarity = decision.sims?.[decision.index];
    const chosen = candidates[decision.index];
    // A field an earlier step already set is left alone rather than set twice.
    const alreadySet = chosen && VALUE_OPS.has(chosen.op) && changed.has(signature(snap.url, chosen.element));
    const action = alreadySet ? { op: 'DONE', element: chosen.element, reason: '設定済み' } : await toAction(chosen, similarity, step);

    const top = decision.probs
      .map((p, k) => ({ p, k }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 5)
      .map(({ p, k }) => ({ id: candidates[k].element?.id ?? null, label: candidates[k].text, p, chosen: k === decision.index }));
    const summary = action.op === 'BLOCKED' ? `スキップ: ${action.reason}` : describe(action);
    await page.evaluate(([ids, chosenId]) => window.__gj?.marks(ids, chosenId), [snap.elements.filter((e) => e.inView).map((e) => e.id), action.element?.id ?? null]).catch(() => {});
    await render({ decision: { op: action.op, summary, top, ms: decision.ms, candidates: candidates.length } });
    await pause(1800);

    const executedAt = performance.now();
    let error = null;
    try {
      await execute(page, action);
    } catch (caught) {
      error = caught.message.split('\n')[0];
    }
    state.stepStatus[index] = error || action.op === 'BLOCKED' || action.op === 'DONE' ? 'skipped' : 'done';
    if (!error && VALUE_OPS.has(action.op)) changed.add(signature(snap.url, action.element));
    if (!error && action.op !== 'BLOCKED') history.push(action.op === 'DONE' ? `（${step.step}：済んでいるので何もしない）` : describe(action).replace(/^\[\d+\] /, ''));
    log.push({
      round,
      step,
      url: snap.url,
      elements: snap.elements.length,
      candidates: candidates.length,
      decisionMs: decision.ms,
      executionMs: performance.now() - executedAt,
      action: {
        op: action.op,
        target: action.element ? { id: action.element.id, role: action.element.role, name: action.element.name, context: action.element.context } : null,
        value: action.value ?? action.optionText ?? action.checked,
        optionHow: action.optionHow,
        reason: action.reason,
      },
      similarity,
      top: top.map(({ id, label, p }) => ({ id, label, p })),
      raw: decision.raw,
      error,
    });
    await page.evaluate(() => window.__gj?.marks([], null)).catch(() => {});
    await render();
    await pause(500);
  }

  if (plannerName === 'script') {
    state.steps = task.steps;
    state.stepStatus = task.steps.map(() => '');
    await render();
    await pause(2200);
    for (const [index, step] of task.steps.entries()) await performStep(step, index, 0);
  } else {
    for (let round = 0; round < MAX_ROUNDS && log.length < MAX_STEPS; round++) {
      const pageSnap = await page.evaluate(snapshot);
      const pageInput = { goal: task.goal, site: task.site, today: TODAY, pageTitle: pageSnap.title, pageElements: describeForPlanner(pageSnap), history: [...history] };

      // Jev's DONE: once something has been done, Gemma 4 E2B checks whether the goal is met.
      if (round > 0) {
        await render({ planning: 'Gemma 4 E2B が目標を達成できたか確かめています…', fastForward: 'Gemma 4 E2B が達成を確認中' });
        const checkStart = seconds();
        const verdict = await cachedCall('done', checkDone, { ...pageInput, pageText: pageSnap.text });
        if (!verdict.cached) edits.push({ start: checkStart, end: seconds(), factor: 'fast' });
        rounds.push({ url: pageSnap.url, doneCheck: verdict.raw, ms: verdict.ms });
        state.planMs += verdict.ms;
        if (verdict.done) {
          await render({ note: 'Gemma 4 E2B: 目標は達成済み（DONE）と判断' });
          await pause(1500);
          break;
        }
      }

      // Gemma 4 E2B plans the steps for the current page.
      await render({ planning: `Gemma 4 E2B がこのページの手順を考えています（${round + 1}回目）…`, fastForward: 'Gemma 4 E2B が手順を作成中' });
      const planStart = seconds();
      const planned = await cachedCall('plan', planSteps, pageInput);
      if (!planned.cached) edits.push({ start: planStart, end: seconds(), factor: 'fast' });
      const planSignature = `${new URL(pageSnap.url).pathname}|${JSON.stringify(planned.steps)}`;
      const repeated = rounds.some((previous) => previous.signature === planSignature);
      rounds.push({ url: pageSnap.url, steps: planned.steps, ms: planned.ms, raw: planned.raw, signature: planSignature });
      state.planMs += planned.ms;
      if (planned.steps.length === 0 || repeated) {
        await render({ note: planned.steps.length === 0 ? 'Gemma 4 E2B: これ以上の手順はないと判断' : '同じ計画の繰り返しになったので終了' });
        await pause(1500);
        break;
      }
      const offset = state.steps.length;
      state.steps = [...state.steps, ...planned.steps];
      state.stepStatus = [...state.stepStatus, ...planned.steps.map(() => '')];
      await render();
      await pause(2000);

      const pathBefore = new URL(page.url()).pathname;
      for (const [i, step] of planned.steps.entries()) {
        await performStep(step, offset + i, round);
        if (new URL(page.url()).pathname !== pathBefore || log.length >= MAX_STEPS) break; // new page: plan again
      }
    }
  }

  // Independent outcome check.
  const checks = await task.verify(page);
  const ok = checks.every((item) => item.ok);
  await render({ result: { ok, text: ok ? '✓ 目標を達成（タスクごとの検証で確認）' : `✗ 未達成: ${checks.filter((item) => !item.ok).map((item) => item.name).join('、')}` } });
  await pause(3500);
  await context.close();

  let videoPath = null;
  if (video) {
    const webm = path.join(rawDir, `${number}-${task.id}.webm`);
    await rename(await page.video().path(), webm);
    const suffix = runName === 'script-embedding' ? '' : `-${runName}`;
    videoPath = await toMp4(webm, path.join(OUT, 'videos', `${number}-${task.id}${suffix}.mp4`), edits);
  }
  return { id: task.id, goal: task.goal, ok, checks, rounds, log, planMs: state.planMs, videoPath };
}

const server = await startServer(ROOT, PORT);
const browser = await chromium.launch();
const results = [];
try {
  for (const task of tasks) {
    const number = String(TASKS.indexOf(task) + 1).padStart(2, '0');
    process.stdout.write(`${number} ${task.id} … `);
    const result = await runTask(browser, task, number);
    results.push(result);
    const decisions = result.log.map((entry) => entry.decisionMs);
    const avg = decisions.reduce((a, b) => a + b, 0) / Math.max(1, decisions.length);
    console.log(
      `${result.ok ? 'OK ' : 'NG '} steps=${result.log.length} ops=${result.log.map((entry) => entry.action.op[0]).join('')} decision avg=${avg.toFixed(0)}ms` +
        (plannerName === 'gemma4' ? ` plan=${(result.planMs / 1000).toFixed(1)}s` : '') +
        (result.ok ? '' : ` failed: ${result.checks.filter((c) => !c.ok).map((c) => `${c.name}(${c.detail})`).join(', ')}`) +
        (result.videoPath ? ` → ${path.relative(ROOT, result.videoPath)}` : ''),
    );
  }
} finally {
  await browser.close();
  server.kill();
  await rm(path.join(OUT, 'videos', 'raw'), { recursive: true, force: true }).catch(() => {});
}

const summary = {
  createdAt: new Date().toISOString(),
  planner: plannerLabel,
  decider: decider.name,
  passed: results.filter((result) => result.ok).length,
  total: results.length,
  results,
};
await writeFile(path.join(OUT, `summary-${runName}${video ? '-video' : ''}.json`), `${JSON.stringify(summary, null, 2)}\n`);
console.log(`\n${summary.passed}/${summary.total} tasks passed (planner: ${plannerLabel}, decider: ${decider.name}).`);
