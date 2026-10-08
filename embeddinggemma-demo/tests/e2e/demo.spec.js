// Drives the demo the way a person would (pick options, click buttons, read what is
// rendered) and records screenshots plus the measured numbers in test-output/.
//
//   npm run test:e2e                         # q8 only
//   DTYPES=fp32,q8,q4 THREADS=4 npm run test:e2e
//   MODEL=v2 DTYPES=fp32 THREADS=4 npm run test:e2e        # EmbeddingGemma 2 on WASM
//   MODEL=v2 DEVICE=webgpu DTYPES=q8,q4 npm run test:e2e   # its quantized exports need WebGPU
//   REMOTE=1 npm run test:e2e                # also load the model from the Hugging Face Hub
//   npm run report                           # summarize test-output/e2e-*.json
import { mkdir, writeFile } from 'node:fs/promises';

import { expect, test } from '@playwright/test';

import { SEARCH_PRESETS } from '../../public/data/presets.js';
import { MODELS } from '../../public/lib/model-config.js';

const OUTPUT_DIR = 'test-output';
const MODEL = process.env.MODEL ?? 'v1';
const DEVICE = process.env.DEVICE ?? 'wasm';
const DTYPES = (process.env.DTYPES ?? 'q8')
  .split(',')
  .map((dtype) => dtype.trim())
  .filter(Boolean);
const THREADS = process.env.THREADS ?? 'auto';
const SLOW = 10 * 60 * 1000;

// Scores printed in the onnx-community model card for the "Red Planet" example: fp32 for the first
// model, q4 for EmbeddingGemma 2. The published dtype must match closely; the others drift more
// the fewer bits they keep (and, for EmbeddingGemma 2, are compared with q4 numbers).
const MODEL_CARD = SEARCH_PRESETS.find((preset) => preset.id === 'model-card').referenceScores[MODEL];
const MODEL_CARD_TOLERANCE = {
  v1: { fp32: 0.005, q8: 0.03, q4: 0.05 },
  v2: { fp32: 0.02, q8: 0.02, q4: 0.005 },
}[MODEL];

const screenshotPath = (dtype, name) => `${OUTPUT_DIR}/screenshots/${MODEL === 'v1' ? '' : `${MODEL}-`}${dtype}-${name}.png`;

async function writeJson(file, data) {
  await mkdir(OUTPUT_DIR, { recursive: true });
  await writeFile(`${OUTPUT_DIR}/${file}`, `${JSON.stringify(data, null, 2)}\n`);
}

/**
 * Clicks a run button and waits until its results container reports a new run, so
 * assertions never read the previous run's output that is still on screen.
 */
async function runAndWait(page, buttonName, resultsId, timeout = 60_000) {
  const results = page.locator(`#${resultsId}`);
  const previous = Number((await results.getAttribute('data-run')) ?? 0);
  await page.getByRole('button', { name: buttonName }).click();
  await expect(results).toHaveAttribute('data-run', String(previous + 1), { timeout });
}

test('boots without a model, cross-origin isolated, with working tabs', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  await expect(page.locator('#model-status')).toHaveAttribute('data-state', 'idle');
  await expect(page.getByRole('button', { name: '検索する' })).toBeDisabled();
  expect(await page.evaluate(() => self.crossOriginIsolated)).toBe(true);

  await page.getByRole('tab', { name: 'ミニベンチマーク' }).click();
  await expect(page.locator('#panel-benchmark')).toBeVisible();
  await expect(page.locator('#panel-search')).toBeHidden();
  await expect(page.locator('#benchmark-intro')).toContainText('文書63件・クエリ56件');

  await page.getByRole('tab', { name: 'ミニベンチマーク' }).press('ArrowLeft');
  await expect(page.getByRole('tab', { name: '類似度マトリクス' })).toHaveAttribute('aria-selected', 'true');
  await page.screenshot({ path: `${OUTPUT_DIR}/screenshots/initial.png`, fullPage: true });
});

test('fits a phone-sized screen without horizontal scrolling', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  await page.screenshot({ path: `${OUTPUT_DIR}/screenshots/mobile.png`, fullPage: true });
});

test('loads the model straight from the Hugging Face Hub', async ({ page }) => {
  test.skip(!process.env.REMOTE, 'downloads about 220 MB; run with REMOTE=1');
  test.setTimeout(SLOW);
  // local=0 skips /models/, so this is the path a fresh checkout takes (CORS under COEP included).
  await page.goto('/?cache=0&local=0&dtype=q4&autoload=1');
  await expect(page.locator('#model-status')).toHaveAttribute('data-state', /ready|error/, { timeout: SLOW });
  await expect(page.locator('#model-status')).toHaveAttribute('data-state', 'ready');
  await runAndWait(page, '検索する', 'search-results');
  await expect(page.locator('#search-results .ranking li').first()).toContainText('Mars');
});

for (const dtype of DTYPES) {
  test.describe(`${MODELS[MODEL].name} ${dtype} (${DEVICE})`, () => {
    test.describe.configure({ mode: 'serial' });

    /** @type {import('@playwright/test').Page} */
    let page;
    const record = { model: MODEL, dtype, device: DEVICE, threads: THREADS };

    test.beforeAll(async ({ browser }) => {
      page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
      // cache=0: the model is served by the local server; do not copy it into Cache Storage on every run.
      await page.goto('/?cache=0');
      await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
      record.userAgent = await page.evaluate(() => navigator.userAgent);
    });

    test.afterAll(async () => {
      await writeJson(`e2e-${MODEL}-${dtype}-${DEVICE}.json`, record);
      await page?.close();
    });

    test('loads the model from the UI', async () => {
      test.setTimeout(SLOW);
      // By id: the label's accessible name also contains the option texts, and サンプル's options mention モデル.
      await page.locator('#model').selectOption(MODEL);
      await page.getByLabel('実行環境').selectOption(DEVICE);
      await page.getByLabel('精度').selectOption(dtype);
      if (DEVICE === 'wasm') await page.getByLabel('スレッド数').selectOption(THREADS);
      await page.getByRole('button', { name: 'モデルを読み込む' }).click();

      await expect(page.locator('#model-status')).toHaveAttribute('data-state', /ready|error/, { timeout: SLOW });
      await expect(page.locator('#model-status')).toHaveAttribute('data-state', 'ready');
      await expect(page.locator('#model-facts')).toContainText(DEVICE === 'wasm' ? 'WASM' : 'WebGPU');
      await expect(page.locator('#model-facts')).toContainText(MODELS[MODEL].name);
      await expect(page.getByRole('button', { name: '検索する' })).toBeEnabled();

      record.loaded = await page.evaluate(() => window.__demo.model);
      expect(record.loaded.model).toBe(MODEL);
      expect(record.loaded.dtype).toBe(dtype);
      expect(record.loaded.crossOriginIsolated).toBe(true);
    });

    test('reproduces the model card example', async () => {
      await page.getByRole('tab', { name: '検索' }).click();
      await page.getByLabel('サンプル').selectOption('model-card');
      await runAndWait(page, '検索する', 'search-results');
      await expect(page.locator('#search-results .ranking li').first()).toContainText('Mars');
      await expect(page.locator('#search-results')).toContainText('モデルカード');

      const { ranking } = await page.evaluate(() => window.__demo.last.search);
      // Mars first; the order of the others is the one the model card shows for each model.
      const expectedOrder = MODEL_CARD.scores.map((score, index) => ({ score, index })).sort((a, b) => b.score - a.score);
      expect(ranking.map((row) => row.index)).toEqual(expectedOrder.map((row) => row.index));
      const maxDiff = Math.max(...ranking.map((row) => Math.abs(row.score - MODEL_CARD.scores[row.index])));
      record.modelCard = { reference: MODEL_CARD, scores: ranking.sort((a, b) => a.index - b.index).map((row) => row.score), maxDiff };
      expect(maxDiff).toBeLessThan(MODEL_CARD_TOLERANCE[dtype]);
      await page.screenshot({ path: screenshotPath(dtype, 'search'), fullPage: true });
    });

    test('ranks the Japanese and cross-lingual samples', async () => {
      record.presets = {};
      // anime-ja is a known miss (the model puts 作画監督 above 動画 at every precision):
      // its rank is recorded for the report instead of being asserted.
      for (const [preset, mustBeTop] of [
        ['faq-ja', true],
        ['cross-lingual', true],
        ['anime-ja', false],
      ]) {
        await page.getByLabel('サンプル').selectOption(preset);
        await runAndWait(page, '検索する', 'search-results');
        await expect(page.locator('#search-results .ranking .badge')).toHaveText('想定解');

        const search = await page.evaluate(() => window.__demo.last.search);
        await expect(page.locator('#search-results .ranking li')).toHaveCount(search.ranking.length);
        record.presets[preset] = {
          expectedRank: search.expectedRank,
          top: search.ranking[0].text,
          scores: search.ranking.map((row) => row.score),
          elapsedMs: search.elapsedMs,
        };
        if (mustBeTop) {
          expect(search.expectedRank).toBe(1);
          await expect(page.locator('#search-results .ranking li').first().locator('.badge')).toBeVisible();
        }
      }
      await page.screenshot({ path: screenshotPath(dtype, 'search-anime'), fullPage: true });
    });

    test('keeps the right answer on top for every MRL dimension', async () => {
      await page.getByLabel('サンプル').selectOption('faq-ja');
      for (const dims of ['768', '512', '256', '128']) {
        await page.locator('#search-dims').selectOption(dims);
        await runAndWait(page, '検索する', 'search-results');
        await expect(page.locator('#search-results .result-meta')).toContainText(`${dims}次元`);
        await expect(page.locator('#search-results .ranking li').first()).toContainText('領収書はマイページ');
      }
      // Same texts as before, so nothing is embedded again: only the truncation changes.
      await expect(page.locator('#search-results .result-meta')).toContainText('キャッシュを再利用');
      await page.locator('#search-dims').selectOption('768');
    });

    test('groups paraphrases across languages in the similarity matrix', async () => {
      await page.getByRole('tab', { name: '類似度マトリクス' }).click();
      await runAndWait(page, '類似度を計算', 'similarity-results');
      await expect(page.locator('#similarity-results table td')).toHaveCount(64);

      const { matrix, sentences } = await page.evaluate(() => window.__demo.last.similarity);
      const n = sentences.length;
      const groups = [[0, 1, 2], [3, 4], [5, 6], [7]]; // hot weather, ramen, meeting, stocks
      const groupOf = (i) => groups.findIndex((group) => group.includes(i));
      let minSame = Infinity;
      let maxDifferent = -Infinity;
      for (let i = 0; i < n; i++) {
        expect(matrix[i * n + i]).toBeCloseTo(1, 3);
        for (let j = i + 1; j < n; j++) {
          const score = matrix[i * n + j];
          if (groupOf(i) === groupOf(j)) minSame = Math.min(minSame, score);
          else maxDifferent = Math.max(maxDifferent, score);
        }
      }
      record.similarity = { minSameMeaning: minSame, maxDifferentMeaning: maxDifferent, matrix };
      expect(minSame).toBeGreaterThan(maxDifferent);

      await page.locator('#similarity-results td[data-i="0"][data-j="2"]').hover();
      await expect(page.locator('#tooltip')).toBeVisible();
      await page.screenshot({ path: screenshotPath(dtype, 'similarity'), fullPage: true });
    });

    test('runs the mini benchmark and exports the results', async () => {
      test.setTimeout(SLOW);
      await page.getByRole('tab', { name: 'ミニベンチマーク' }).click();
      await page.getByLabel('バッチサイズ（文書）').selectOption('8');
      await runAndWait(page, 'ベンチマークを実行', 'benchmark-results', SLOW);
      const save = page.getByRole('button', { name: '結果をJSONで保存' });
      await expect(save).toBeVisible();
      await expect(page.locator('#benchmark-results table')).toHaveCount(3);

      const result = await page.evaluate(() => window.__demo.last.benchmark);
      record.benchmark = result;
      const main = result.evaluations.find((evaluation) => evaluation.mode === 'prompt' && evaluation.dims === 768);
      // Sanity floors, far below what a working setup scores; they catch broken prompts or tokenization.
      expect(main.overall.acc1).toBeGreaterThan(0.6);
      expect(main.overall.mrr10).toBeGreaterThan(0.7);

      const [file] = await Promise.all([page.waitForEvent('download'), save.click()]);
      expect(file.suggestedFilename()).toBe(`embeddinggemma-${MODEL}-benchmark-${dtype}-${DEVICE}.json`);
      await page.screenshot({ path: screenshotPath(dtype, 'benchmark'), fullPage: true });
    });

    test('writes text one character at a time towards a sentence embedding', async () => {
      test.setTimeout(SLOW);
      await page.getByRole('tab', { name: '文字生成' }).click();
      await page.getByLabel('漢字の候補').selectOption('16');
      await page.getByLabel('最大文字数').selectOption('12');
      // Without kana a step tries about 30 characters instead of 200, so even q4 on WASM finishes in minutes.
      await page.locator('#chargen-kana').selectOption('0');
      await page.getByRole('button', { name: '生成する' }).click();
      await expect(page.locator('#chargen-results')).toHaveAttribute('data-done', /true|error/, { timeout: SLOW });
      await expect(page.locator('#chargen-results')).toHaveAttribute('data-done', 'true');

      const text = await page.locator('#chargen-results .chargen-text').textContent();
      const steps = await page.locator('#chargen-results .chargen-steps li').count();
      record.chargen = { text, steps, note: await page.locator('#chargen-note').textContent() };
      expect(steps).toBeGreaterThan(0);
      // Content characters of the target sentence come first; the order is not expected to survive.
      expect([...'東京都港区電波塔'].some((char) => text.includes(char))).toBe(true);
      await page.screenshot({ path: screenshotPath(dtype, 'chargen'), fullPage: true });
    });
  });
}

test('phone check page rates every model for this browser without loading any', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/phone.html');
  await expect(page.locator('#device-status')).toHaveAttribute('data-state', 'ready');
  await expect(page.locator('#device-facts div')).toHaveCount(8);
  const levels = await page.locator('.model-row .chip').evaluateAll((chips) => chips.map((chip) => chip.dataset.level));
  expect(levels.length).toBeGreaterThan(5);
  expect(levels.every((level) => ['yes', 'maybe', 'no'].includes(level))).toBe(true);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  await page.screenshot({ path: `${OUTPUT_DIR}/screenshots/phone-check.png`, fullPage: true });
});
