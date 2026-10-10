// Mechanical checks of a conversation between two small bots (scripts/bot-chat.mjs): do they repeat each
// other, speak for the wrong side, keep saying goodbye, break the format they were given, drift into
// another script. Whether they get anywhere is judged separately.
import { findLoop, scriptStats } from './ja-checks.js';

/** The set of two-character pieces of `text`, ignoring spaces, punctuation and the action line. */
function bigrams(text) {
  const flat = text.replace(/行動[:：].*$/m, '').replace(/[\s、。！？!?…「」『』（）()・,.~〜ー]/g, '');
  const set = new Set();
  for (let i = 0; i + 2 <= flat.length; i++) set.add(flat.slice(i, i + 2));
  return set;
}

/** Overlap of the two-character pieces of two texts, 0 (nothing shared) to 1 (the same). */
export function similarity(a, b) {
  const x = bigrams(a);
  const y = bigrams(b);
  if (!x.size || !y.size) return 0;
  let shared = 0;
  for (const piece of x) if (y.has(piece)) shared++;
  return shared / (x.size + y.size - shared);
}

const FAREWELL = /(さようなら|さよなら|またね|じゃあね|じゃあまた|バイバイ|おやすみ|失礼します|また会おう|また後で|ではまた|それでは、?また|またお会い|行ってきます|いってらっしゃい)/;
export const ACTIONS = ['待つ', '歩く', '渡す', '受け取る', '笑う', '手を振る'];

/**
 * Per-message flags for a conversation. `turns` is [{ speaker, text }] in order, `names` the two speakers.
 * A message is an "echo" when it is almost the same as an earlier one (0.7 and above), a "role slip" when it
 * puts words in the other bot's mouth, labels itself or calls itself by its own name.
 */
export function analyzeConversation(turns, names, { actionFormat = false } = {}) {
  const flags = turns.map((turn, i) => {
    const [self, other] = turn.speaker === names[0] ? names : [names[1], names[0]];
    let nearest = 0;
    let nearestOwn = 0;
    for (let j = 0; j < i; j++) {
      const s = similarity(turn.text, turns[j].text);
      nearest = Math.max(nearest, s);
      if (turns[j].speaker === turn.speaker) nearestOwn = Math.max(nearestOwn, s);
    }
    const labels = [...turn.text.matchAll(/(?:^|\n)\s*[「『（(]?([^\s:：「」『』（）()]{1,8})[」』）)]?\s*[:：]/g)].map((m) => m[1]);
    const speaksForOther = labels.includes(other);
    const labelsSelf = labels.includes(self);
    const callsSelf = new RegExp(`${self}(さん|くん|君|ちゃん|様)`).test(turn.text);
    return {
      turn: i + 1,
      speaker: turn.speaker,
      echo: nearest >= 0.7,
      ownEcho: nearestOwn >= 0.7,
      nearest: Number(nearest.toFixed(2)),
      loop: Boolean(findLoop(turn.text, 10, 3)),
      roleSlip: speaksForOther || callsSelf,
      labelsSelf,
      foreign: scriptStats(turn.text).foreign,
      farewell: FAREWELL.test(turn.text),
      chars: turn.text.replace(/\s/g, '').length,
      action: actionFormat ? ACTIONS.find((a) => new RegExp(`行動[:：]\\s*${a}`).test(turn.text)) ?? null : undefined,
    };
  });
  const firstFarewell = flags.findIndex((f) => f.farewell);
  const half = Math.floor(flags.length / 2);
  const mean = (xs) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0);
  const share = (xs, key) => (xs.length ? xs.filter((f) => f[key]).length / xs.length : 0);
  return {
    messages: flags.length,
    echoes: flags.filter((f) => f.echo).length,
    echoShareFirstHalf: share(flags.slice(0, half), 'echo'),
    echoShareSecondHalf: share(flags.slice(half), 'echo'),
    firstEcho: flags.find((f) => f.echo)?.turn ?? null,
    roleSlips: flags.filter((f) => f.roleSlip).length,
    labelsSelf: flags.filter((f) => f.labelsSelf).length,
    loops: flags.filter((f) => f.loop).length,
    foreign: flags.filter((f) => f.foreign).length,
    farewells: flags.filter((f) => f.farewell).length,
    firstFarewell: firstFarewell < 0 ? null : firstFarewell + 1,
    messagesAfterFirstFarewell: firstFarewell < 0 ? 0 : flags.length - firstFarewell - 1,
    charsFirstTen: mean(flags.slice(0, 10).map((f) => f.chars)),
    charsLastTen: mean(flags.slice(-10).map((f) => f.chars)),
    actionAdherence: actionFormat ? share(flags, 'action') : undefined,
    flags,
  };
}
