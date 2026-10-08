#!/usr/bin/env node
// Runs the phone check page (public/phone.html) in Chromium the way a phone would see it:
// Pixel 7 screen and user agent, and a WebGPU adapter with phone-like limits. On a machine
// without a GPU, Chromium's software WebGPU (SwiftShader) has no shader-f16 and a 1 GiB
// buffer limit, so the f16 variants are skipped and the limit checks are real.
// It records each test's result and the peak memory (RSS) of all browser processes.
//
//   npm run download-phone-models        # once, so the models come from localhost
//   npm run phone-check                  # every non-f16 variant
//   npm run phone-check -- gemma-3-270m:q4:wasm --video
//   npm run phone-check -- --tokens 16   # shorter generations (software WebGPU is slow)
import { execFileSync } from 'node:child_process';
import { mkdir, rename, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { chromium, devices } from '@playwright/test';

import { PHONE_MODELS, variantKey } from '../public/lib/phone-models.js';
import { startServer, toMp4 } from './lib/video.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'test-output', 'phone-check');
const PORT = Number(process.env.PORT ?? 5176);
const TEST_TIMEOUT_MS = 30 * 60 * 1000;

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : fallback;
};
const video = args.includes('--video');
const tokens = Number(option('tokens', 32));
const cache = option('cache', '1');
const repeat = Number(option('repeat', 1));
const optionValues = new Set([option('tokens'), option('cache'), option('repeat')].filter(Boolean));
const requested = args.filter((arg) => !arg.startsWith('--') && !optionValues.has(arg));
const keys = requested.length
  ? requested
  : PHONE_MODELS.flatMap((model) => model.variants.filter((variant) => !variant.f16).map((variant) => variantKey(model, variant)));

/**
 * Resident memory (MB) of the processes this script started: the whole browser, and the largest
 * renderer (the tab, which also hosts the worker; on a phone this is what gets killed).
 */
function browserRssMB() {
  const rows = execFileSync('ps', ['-eo', 'pid=,ppid=,rss=,args='], { encoding: 'utf8' })
    .trim()
    .split('\n')
    .map((line) => {
      const [pid, ppid, rss, ...rest] = line.trim().split(/\s+/);
      return { pid: Number(pid), ppid: Number(ppid), mb: Number(rss) / 1024, renderer: rest.join(' ').includes('--type=renderer') };
    });
  const children = new Map();
  for (const row of rows) children.set(row.ppid, [...(children.get(row.ppid) ?? []), row]);
  let total = 0;
  let renderer = 0;
  const stack = [...(children.get(process.pid) ?? [])];
  while (stack.length) {
    const row = stack.pop();
    if (row.pid === server.pid) continue;
    total += row.mb;
    if (row.renderer) renderer = Math.max(renderer, row.mb);
    stack.push(...(children.get(row.pid) ?? []));
  }
  return { total, renderer };
}

await mkdir(OUT, { recursive: true });
const server = await startServer(ROOT, PORT);
const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu'] });
const results = [];
let device = null;
try {
  for (const key of keys) {
    const rawDir = path.join(OUT, 'raw');
    const context = await browser.newContext({
      ...devices['Pixel 7'],
      locale: 'ja-JP',
      ...(video ? { recordVideo: { dir: rawDir, size: { width: 412, height: 915 } } } : {}),
    });
    const page = await context.newPage();
    page.on('console', (message) => {
      if (message.type() === 'error') console.log(`   [console] ${message.text().slice(0, 200)}`);
    });
    await page.goto(`http://127.0.0.1:${PORT}/phone.html?tokens=${tokens}&cache=${cache}`);
    device = await page.evaluate(() => window.phoneCheck.ready);

    const row = page.locator(`.model-row[data-key="${key}"]`);
    await row.scrollIntoViewIfNeeded();
    const verdict = await row.getAttribute('data-level');
    // The second and later runs find the model files in the browser's Cache Storage.
    for (let attempt = 1; attempt <= repeat; attempt++) {
      const baseline = browserRssMB();
      const peak = { ...baseline };
      const sample = () => {
        const now = browserRssMB();
        peak.total = Math.max(peak.total, now.total);
        peak.renderer = Math.max(peak.renderer, now.renderer);
      };
      const sampler = setInterval(sample, 500);
      process.stdout.write(`${key}${repeat > 1 ? ` #${attempt}` : ''} … `);
      const result = await page.evaluate((k) => window.phoneCheck.run(k), key).catch((error) => ({ ok: false, error: error.message }));
      clearInterval(sampler);
      sample();
      await page.locator('#result-list li').first().scrollIntoViewIfNeeded().catch(() => {});
      await page.waitForTimeout(video ? 2500 : 200);

      const entry = {
        key,
        attempt,
        verdict,
        ...result,
        rssMB: { baseline: Math.round(baseline.total), peak: Math.round(peak.total), rendererBaseline: Math.round(baseline.renderer), rendererPeak: Math.round(peak.renderer) },
      };
      results.push(entry);
      console.log(
        result.ok
          ? `OK   load=${(result.loadMs / 1000).toFixed(1)}s ` +
              (result.msPerText !== undefined
                ? `embed=${result.msPerText.toFixed(0)}ms/text`
                : `ttft=${(result.ttftMs / 1000).toFixed(2)}s decode=${result.decodeTokPerSec?.toFixed(1)}tok/s`) +
              ` tab=${entry.rssMB.rendererPeak}MB browser=${entry.rssMB.peak}MB  ${result.text.replace(/\s+/g, ' ').slice(0, 60)}`
          : `FAIL (${result.error?.slice(0, 160)}) tab=${entry.rssMB.rendererPeak}MB browser=${entry.rssMB.peak}MB`,
      );
      if (!result.ok) break;
    }

    await context.close();
    if (video) {
      const webm = path.join(rawDir, `${key.replaceAll(':', '-')}.webm`);
      await rename(await page.video().path(), webm);
      await toMp4(webm, path.join(OUT, `${key.replaceAll(':', '-')}.mp4`), []);
    }
  }
} finally {
  await browser.close();
  server.kill();
  await rm(path.join(OUT, 'raw'), { recursive: true, force: true }).catch(() => {});
}

const summary = { createdAt: new Date().toISOString(), profile: 'Pixel 7 emulation + software WebGPU', tokens, browserCache: cache !== '0', device, results };
const file = path.join(OUT, `results-${new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '')}.json`);
await writeFile(file, `${JSON.stringify(summary, null, 2)}\n`);
console.log(`\nSaved ${path.relative(ROOT, file)}`);
