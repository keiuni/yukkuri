#!/usr/bin/env node
// Zero-dependency static server for the demo.
//
// It sends COOP/COEP headers so the page is cross-origin isolated, which lets
// onnxruntime-web use SharedArrayBuffer and run WASM inference on several threads.
// Files under ./models (see scripts/download-model.mjs) are served at /models/ so the
// worker can load the model from localhost before falling back to the Hugging Face Hub.
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = path.join(ROOT, 'public');
const MODELS_DIR = path.join(ROOT, 'models');
const HOST = process.env.HOST ?? '127.0.0.1';
const PORT = Number(process.env.PORT ?? 5173);

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.wasm': 'application/wasm',
};

const BASE_HEADERS = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'X-Content-Type-Options': 'nosniff',
};

function resolvePath(pathname) {
  const [baseDir, relative] = pathname.startsWith('/models/')
    ? [MODELS_DIR, pathname.slice('/models/'.length)]
    : [PUBLIC_DIR, pathname === '/' ? 'index.html' : pathname.slice(1)];
  const file = path.resolve(baseDir, relative);
  // Reject anything that escapes the served directory (e.g. "/../server.mjs").
  return file.startsWith(baseDir + path.sep) ? file : null;
}

function send(res, status, body) {
  res.writeHead(status, { ...BASE_HEADERS, 'Content-Type': 'text/plain; charset=utf-8' });
  res.end(body);
}

const server = createServer(async (req, res) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    send(res, 405, 'Method not allowed');
    return;
  }

  let file;
  try {
    file = resolvePath(decodeURIComponent(new URL(req.url, 'http://localhost').pathname));
  } catch {
    send(res, 400, 'Bad request');
    return;
  }

  const info = file && (await stat(file).catch(() => null));
  if (!info?.isFile()) {
    send(res, 404, 'Not found');
    return;
  }

  res.writeHead(200, {
    ...BASE_HEADERS,
    'Content-Type': MIME_TYPES[path.extname(file)] ?? 'application/octet-stream',
    'Content-Length': info.size,
    // transformers.js keeps its own Cache Storage copy of model files, so plain revalidation is enough.
    'Cache-Control': 'no-cache',
  });
  if (req.method === 'HEAD') {
    res.end();
    return;
  }
  createReadStream(file).pipe(res);
});

server.listen(PORT, HOST, () => {
  console.log(`EmbeddingGemma demo: http://${HOST}:${PORT}/`);
});
