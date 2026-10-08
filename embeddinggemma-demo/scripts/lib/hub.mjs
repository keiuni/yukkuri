// Downloads files from the Hugging Face Hub into ./models/<model id>/ so that server.mjs can serve
// them at /models/. Behind an HTTP proxy, run the scripts with NODE_USE_ENV_PROXY=1 (Node >= 22.21).
import { createWriteStream } from 'node:fs';
import { mkdir, rename, stat } from 'node:fs/promises';
import path from 'node:path';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';

export const formatMB = (bytes) => `${(bytes / 1e6).toFixed(1)} MB`;

async function exists(file) {
  try {
    return (await stat(file)).isFile();
  } catch {
    return false;
  }
}

/** Downloads `file` of `modelId` at `revision` into `targetDir` unless it is already there. */
export async function downloadFile({ modelId, revision, file, targetDir, force = false }) {
  const destination = path.join(targetDir, file);
  if (!force && (await exists(destination))) {
    console.log(`skip  ${file} (already downloaded)`);
    return;
  }
  await mkdir(path.dirname(destination), { recursive: true });

  const url = `https://huggingface.co/${modelId}/resolve/${revision}/${file}`;
  const response = await fetch(url);
  if (!response.ok || !response.body) {
    throw new Error(`GET ${url} failed: ${response.status} ${response.statusText}`);
  }

  const total = Number(response.headers.get('content-length')) || 0;
  let received = 0;
  let lastReport = 0;
  const progress = new Transform({
    transform(chunk, _encoding, callback) {
      received += chunk.length;
      if (total > 50e6 && received - lastReport > 100e6) {
        lastReport = received;
        console.log(`      ${file}: ${formatMB(received)} / ${formatMB(total)}`);
      }
      callback(null, chunk);
    },
  });

  // Write to a temporary name first so an interrupted download is never mistaken for a complete one.
  const partial = `${destination}.part`;
  await pipeline(Readable.fromWeb(response.body), progress, createWriteStream(partial));
  await rename(partial, destination);
  console.log(`done  ${file} (${formatMB(received)})`);
}
