// LiteRT-LM is the runtime Google ships for running Gemma on phones: Android and iOS apps embed it through
// its Kotlin and Swift APIs, and the Google AI Edge Gallery app uses it. Its command-line tool also has an
// OpenAI-compatible server, which scripts/ja-chat.mjs uses to run the phone builds of Gemma 4 (.litertlm
// files, scripts/lib/chat-models.mjs) on this machine's CPU, with the same requests as llama-server.
//
// Install the tool with `pip install litert-lm` (or `uv tool install litert-lm`) and set LITERT_LM to its
// path if it is not on PATH.
import { execFile, spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const MODELS_DIR = fileURLToPath(new URL('../../models/', import.meta.url));
// The tool keeps its imported models (hard links to the downloaded files) and its settings here.
const HOME = path.join(MODELS_DIR, '.litert-lm');
const CLI = process.env.LITERT_LM ?? 'litert-lm';

/** Peak resident memory (VmHWM) of a process and its descendants, in MB, or null where /proc is missing. */
function peakMemoryMB(pid) {
  try {
    const status = readFileSync(`/proc/${pid}/status`, 'utf8');
    const own = Number(status.match(/VmHWM:\s+(\d+) kB/)?.[1] ?? 0) / 1024;
    const children = readFileSync(`/proc/${pid}/task/${pid}/children`, 'utf8').trim().split(/\s+/).filter(Boolean);
    return Math.max(own, ...children.map((child) => peakMemoryMB(child) ?? 0));
  } catch {
    return null;
  }
}

/** One `litert-lm serve` process for one model, on its own port. */
export class LiteRtServer {
  /**
   * `spec` is { key, repo, file } (scripts/lib/chat-models.mjs). This machine has no GPU, so the model runs
   * on the CPU with 4 threads, like the CPU rows of the model card's benchmarks. The first start writes the
   * repacked weights next to the model file (2.5 GB for E4B) and later starts map them; without that cache
   * the E4B engine took 13.6 GB of memory here instead of 3.4 GB.
   */
  static async start(spec, port, { threads = 4, maxNumTokens = 4096 } = {}) {
    const file = path.join(MODELS_DIR, spec.repo, spec.file);
    if (!existsSync(file)) throw new Error(`${file} is missing: download it first (npm run download-chat-models)`);
    await mkdir(HOME, { recursive: true });
    const env = { ...process.env, LITERT_LM_DIR: HOME };
    const settings = { backend: 'cpu', cpu_thread_count: threads, max_num_tokens: maxNumTokens, cache: 'disk', thinking: false };
    await writeFile(path.join(HOME, 'config.json'), `${JSON.stringify({ models: { [spec.key]: settings } }, null, 2)}\n`);
    await promisify(execFile)(CLI, ['import', file, spec.key], { env });

    const self = new LiteRtServer();
    self.model = spec.key;
    self.url = `http://127.0.0.1:${port}`;
    self.process = spawn(CLI, ['serve', '--host', '127.0.0.1', '--port', String(port)], { env, stdio: ['ignore', 'ignore', 'pipe'] });
    let log = '';
    self.process.stderr.on('data', (chunk) => {
      log = (log + chunk).slice(-4000);
    });
    const exited = new Promise((resolve) => self.process.once('exit', resolve));
    for (let attempt = 0; attempt < 600; attempt++) {
      const status = await fetch(`${self.url}/v1/models`).then((response) => response.status, () => 0);
      if (status === 200) return self;
      if (await Promise.race([exited.then(() => true), new Promise((resolve) => setTimeout(() => resolve(false), 200))])) {
        throw new Error(`litert-lm serve exited:\n${log}`);
      }
    }
    await self.stop();
    throw new Error('litert-lm serve did not become ready');
  }

  /** Same as LlamaServer.post; the model is named in every request. The engine loads on the first one. */
  async post(urlPath, body) {
    const response = await fetch(`${this.url}${urlPath}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...body, model: this.model }),
    });
    const text = await response.text();
    if (!response.ok) throw new Error(`${urlPath}: ${response.status} ${text.slice(0, 300)}`);
    return JSON.parse(text);
  }

  /** Peak memory of the server so far (model, packed weights, KV cache and the Python server), in MB. */
  peakMemoryMB() {
    return peakMemoryMB(this.process.pid);
  }

  /** Stops the server; resolves once the process has exited (its port is free again). */
  stop() {
    if (!this.process || this.process.exitCode !== null || this.process.signalCode !== null) return Promise.resolve();
    const exited = new Promise((resolve) => this.process.once('exit', resolve));
    this.process.kill();
    return exited;
  }
}
