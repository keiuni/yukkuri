// Shared helpers for the screen recordings: start the demo server, and turn Playwright's
// WebM into an MP4 where some spans are cut or fast-forwarded.
import { execFile, spawn } from 'node:child_process';
import { rename } from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';

// Fast-forwarded spans play at least FAST times faster and never take more than MAX_FAST_SECONDS of video.
const FAST = 4;
const MAX_FAST_SECONDS = 5;

const run = promisify(execFile);

/** Spans are { start, end, factor } in seconds; factor Infinity removes the span, 'fast' speeds it up. */
export function editFilter(edits) {
  const pieces = [];
  let cursor = 0;
  for (const { start, end, factor } of [...edits].sort((a, b) => a.start - b.start)) {
    if (start - cursor > 0.05) pieces.push({ start: cursor, end: start, factor: 1 });
    const speed = factor === 'fast' ? Math.max(FAST, (end - start) / MAX_FAST_SECONDS) : factor;
    if (Number.isFinite(speed) && end - start > 0.05) pieces.push({ start, end, factor: speed.toFixed(3) });
    cursor = Math.max(cursor, end);
  }
  pieces.push({ start: cursor, end: null, factor: 1 });

  const split = `[0:v]split=${pieces.length}${pieces.map((_, i) => `[in${i}]`).join('')}`;
  const segments = pieces.map(({ start, end, factor }, i) => {
    const trim = end === null ? `trim=start=${start.toFixed(3)}` : `trim=start=${start.toFixed(3)}:end=${end.toFixed(3)}`;
    return `[in${i}]${trim},setpts=(PTS-STARTPTS)/${factor}[seg${i}]`;
  });
  const concat = `${pieces.map((_, i) => `[seg${i}]`).join('')}concat=n=${pieces.length}:v=1:a=0,fps=25,format=yuv420p[out]`;
  return [split, ...segments, concat].join(';');
}

/** Converts to H.264 MP4 with the edits applied; keeps the WebM next to `mp4` if ffmpeg fails. */
export async function toMp4(webm, mp4, edits) {
  try {
    await run('ffmpeg', [
      '-y', '-loglevel', 'error', '-i', webm,
      '-filter_complex', editFilter(edits), '-map', '[out]',
      '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-movflags', '+faststart',
      mp4,
    ]);
    return mp4;
  } catch (error) {
    const kept = path.join(path.dirname(mp4), path.basename(webm));
    await rename(webm, kept);
    console.warn(`ffmpeg failed (${error.message.split('\n')[0]}); kept the WebM instead`);
    return kept;
  }
}

/** Starts server.mjs on `port` and resolves once it answers. */
export async function startServer(root, port) {
  const server = spawn(process.execPath, ['server.mjs'], { cwd: root, env: { ...process.env, PORT: String(port) }, stdio: 'ignore' });
  for (let attempt = 0; attempt < 50; attempt++) {
    try {
      if ((await fetch(`http://127.0.0.1:${port}/`)).ok) return server;
    } catch {
      // not listening yet
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  server.kill();
  throw new Error(`The demo server did not start on port ${port}`);
}
