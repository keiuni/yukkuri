// Embeddings of single characters (with the document prompt), cached on disk under
// test-output/char-gen because they only depend on the model and the character list.
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { MODEL_REVISION, documentPrompt } from '../../public/lib/model-config.js';

const CACHE_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', 'test-output', 'char-gen');

/** Unit vectors for `chars`, from the cache or computed with `embedder` (agent/gemma.mjs). */
export async function charVectorsFor(embedder, chars) {
  const key = createHash('sha1').update(MODEL_REVISION + chars.join('')).digest('hex').slice(0, 12);
  const file = path.join(CACHE_DIR, `char-vectors-${key}.bin`);
  try {
    const data = new Float32Array((await readFile(file)).buffer.slice(0));
    const dim = data.length / chars.length;
    return chars.map((_, i) => data.subarray(i * dim, (i + 1) * dim));
  } catch {
    const started = Date.now();
    const vectors = await embedder.embed(chars.map((char) => documentPrompt(null) + char));
    const dim = vectors[0].length;
    const data = new Float32Array(chars.length * dim);
    vectors.forEach((vector, i) => data.set(vector, i * dim));
    await mkdir(CACHE_DIR, { recursive: true });
    await writeFile(file, Buffer.from(data.buffer));
    console.log(`embedded ${chars.length} single characters in ${((Date.now() - started) / 1000).toFixed(1)}s`);
    return vectors;
  }
}
