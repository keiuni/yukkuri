// Shared helpers for the screen recordings: start the demo server, and turn Playwright's
// WebM into an MP4 where some spans are cut or fast-forwarded.
import { execFile, spawn } from 'node:child_process';
import { rename } from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';

// Fast-forwarded spans play at least FAST times faster and never take more than MAX_FAST_SECONDS of video.
const FAST = 4;
const MAX_FAST_SECONDS = 5;
// Playwright's recordings are 25 frames per second.
const FPS = 25;

const run = promisify(execFile);

// Sync marker. Playwright writes every screencast frame for at least 1/FPS second, so while the page
// repaints faster than that (a progress bar, a smooth scroll) the video runs ahead of the wall clock,
// by several seconds over a long recording. Cuts and fast-forwards placed by wall-clock time then land
// on the wrong frames. The recorded page shows a small square that takes the next of these colors at
// every edit boundary, and markTimes finds those changes in the video.
export const MARK_COLORS = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff'];
export const MARK_BOX = { x: 2, y: 2, size: 8 };
const MARK_RGB = MARK_COLORS.map((hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)));

/** Video time (seconds) at which each of the first `count` marks appears, or null if it was not found. */
export async function markTimes(video, count) {
  const { x, y } = MARK_BOX;
  const { stdout } = await run(
    'ffmpeg',
    ['-loglevel', 'error', '-i', video, '-vf', `crop=4:4:${x + 2}:${y + 2},scale=1:1:flags=area`, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
    { encoding: 'buffer', maxBuffer: 1 << 26 },
  );
  const frames = stdout.length / 3;
  const colorAt = (frame) => {
    const rgb = [0, 1, 2].map((c) => stdout[frame * 3 + c]);
    const distances = MARK_RGB.map((mark) => Math.hypot(...mark.map((value, c) => value - rgb[c])));
    const best = distances.indexOf(Math.min(...distances));
    return distances[best] < 90 ? best : -1;
  };
  const times = [];
  let from = 0;
  for (let mark = 0; mark < count; mark++) {
    let frame = from;
    while (frame < frames && colorAt(frame) !== mark % MARK_COLORS.length) frame++;
    if (frame < frames) {
      times.push(frame / FPS);
      from = frame;
    } else {
      times.push(null);
    }
  }
  return times;
}

/**
 * Moves edit boundaries from wall-clock time to video time. Each edit names the marks set at its start
 * and end (startMark / endMark, indexes into `marks`, the wall-clock time of every mark); a mark that was
 * not found keeps its wall-clock time shifted by the drift measured at the last mark that was.
 */
export function alignEdits(edits, marks, times) {
  const toVideo = (index) => {
    if (times[index] !== null) return times[index];
    for (let i = index - 1; i >= 0; i--) if (times[i] !== null) return marks[index] + times[i] - marks[i];
    return marks[index];
  };
  return edits.map((edit) => ({ ...edit, start: toVideo(edit.startMark), end: toVideo(edit.endMark) }));
}

/**
 * Spans are { start, end, factor } in seconds; factor Infinity removes the span, 'fast' speeds it up.
 * `post` is appended to the filter chain of the joined video.
 */
export function editFilter(edits, post = '') {
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
  const concat = `${pieces.map((_, i) => `[seg${i}]`).join('')}concat=n=${pieces.length}:v=1:a=0,fps=${FPS},format=yuv420p${post}[out]`;
  return [split, ...segments, concat].join(';');
}

/**
 * Converts to H.264 MP4 with the edits applied; keeps the WebM next to `mp4` if ffmpeg fails.
 * `hideMarker` paints over the sync marker with the pixels around it.
 */
export async function toMp4(webm, mp4, edits, { hideMarker = false } = {}) {
  const { x, y, size } = MARK_BOX;
  const filter = editFilter(edits, hideMarker ? `,delogo=x=${x - 1}:y=${y - 1}:w=${size + 2}:h=${size + 2}` : '');
  try {
    await run('ffmpeg', [
      '-y', '-loglevel', 'error', '-i', webm,
      '-filter_complex', filter, '-map', '[out]',
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
