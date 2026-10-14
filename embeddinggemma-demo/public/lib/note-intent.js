// Sorting a short Japanese sentence into 予定 / ToDo / メモ with embeddings, and reading the date and time out of it
// with rules (scripts/note-intents.mjs). An embedding says what kind of note a sentence is; it cannot say that
// 「来週の火曜」 is the 20th. That part is plain code.

/** Dot product of two vectors (cosine similarity when both are L2-normalized). */
export function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

/** The normalized mean of some normalized vectors: one "prototype" vector for a label. */
export function centroid(vectors) {
  const mean = new Float32Array(vectors[0].length);
  for (const vector of vectors) for (let i = 0; i < mean.length; i++) mean[i] += vector[i];
  let norm = 0;
  for (const value of mean) norm += value * value;
  norm = Math.sqrt(norm) || 1;
  return mean.map((value) => value / norm);
}

/** Scores per label (highest first) and the gap between the best and the second best. */
function ranked(scores) {
  const list = Object.entries(scores).map(([label, score]) => ({ label, score })).sort((a, b) => b.score - a.score);
  return { label: list[0].label, margin: list[0].score - (list[1]?.score ?? 0), scores: list };
}

/** Nearest prototype: `prototypes` is { label: vector }. */
export function classifyByPrototype(vector, prototypes) {
  return ranked(Object.fromEntries(Object.entries(prototypes).map(([label, p]) => [label, dot(vector, p)])));
}

/** The k most similar labelled examples vote with their similarity: `examples` is [{ label, vector }]. */
export function classifyByNeighbors(vector, examples, k = 3) {
  const top = examples.map((e) => ({ label: e.label, sim: dot(vector, e.vector) })).sort((a, b) => b.sim - a.sim).slice(0, k);
  const scores = {};
  for (const label of new Set(examples.map((e) => e.label))) scores[label] = 0;
  for (const { label, sim } of top) scores[label] += sim;
  return ranked(scores);
}

/** Accuracy, the confusion counts ({ truth: { predicted: n } }) and per-label recall of [{ truth, predicted }]. */
export function summarize(rows) {
  const confusion = {};
  for (const { truth, predicted } of rows) {
    confusion[truth] ??= {};
    confusion[truth][predicted] = (confusion[truth][predicted] ?? 0) + 1;
  }
  const recall = Object.fromEntries(Object.entries(confusion).map(([truth, row]) => [truth, (row[truth] ?? 0) / Object.values(row).reduce((s, n) => s + n, 0)]));
  return { accuracy: rows.filter((r) => r.truth === r.predicted).length / rows.length, confusion, recall };
}

/**
 * What happens if the app asks the user whenever the margin is below a threshold: for each threshold the share of
 * sentences it answers by itself and how many of those are right. `rows` is [{ truth, predicted, margin }].
 */
export function abstainCurve(rows, thresholds) {
  return thresholds.map((threshold) => {
    const answered = rows.filter((r) => r.margin >= threshold);
    return {
      threshold,
      coverage: answered.length / rows.length,
      accuracy: answered.length ? answered.filter((r) => r.truth === r.predicted).length / answered.length : null,
      wrongKept: answered.filter((r) => r.truth !== r.predicted).length,
    };
  });
}

// ---- date and time ----

const DAY_MS = 86400000;
const WEEKDAYS = { 日: 0, 月: 1, 火: 2, 水: 3, 木: 4, 金: 5, 土: 6 };
const toDate = (iso) => new Date(`${iso}T00:00:00Z`);
const toIso = (date) => date.toISOString().slice(0, 10);
const addDays = (iso, days) => toIso(new Date(toDate(iso).getTime() + days * DAY_MS));
const weekday = (iso) => toDate(iso).getUTCDay();
const mondayOf = (iso) => addDays(iso, -((weekday(iso) + 6) % 7));
const pad = (n) => String(n).padStart(2, '0');

function monthDay(now, month, day) {
  const [year] = now.split('-').map(Number);
  const iso = `${year}-${pad(month)}-${pad(day)}`;
  return iso < now ? `${year + 1}-${pad(month)}-${pad(day)}` : iso;
}

/** The n-th given weekday (0 = Sunday) of a month, as YYYY-MM-DD. */
function nthWeekday(year, month, n, wd) {
  const first = `${year}-${pad(month)}-01`;
  return addDays(first, (wd - weekday(first) + 7) % 7 + (n - 1) * 7);
}

function dateOf(text, now) {
  let match = text.match(/(\d{1,2})月(\d{1,2})日/) ?? text.match(/(?<![\d/])(\d{1,2})\/(\d{1,2})(?![\d/])/);
  if (match) return monthDay(now, Number(match[1]), Number(match[2]));
  const [year, month] = now.split('-').map(Number);
  // 「来月の第2土曜日」「第3水曜」: the n-th weekday of next month, or of this month (next month's if it has passed).
  match = text.match(/(来月|今月)?(?:の)?第(\d)([日月火水木金土])曜/);
  if (match) {
    const [nextYear, nextMonth] = month === 12 ? [year + 1, 1] : [year, month + 1];
    if (match[1] === '来月') return nthWeekday(nextYear, nextMonth, Number(match[2]), WEEKDAYS[match[3]]);
    const thisMonth = nthWeekday(year, month, Number(match[2]), WEEKDAYS[match[3]]);
    return thisMonth >= now ? thisMonth : nthWeekday(nextYear, nextMonth, Number(match[2]), WEEKDAYS[match[3]]);
  }
  match = text.match(/来月(?:の)?(\d{1,2})日/);
  if (match) return month === 12 ? `${year + 1}-01-${pad(match[1])}` : `${year}-${pad(month + 1)}-${pad(match[1])}`;
  match = text.match(/(\d{1,2})日後/);
  if (match) return addDays(now, Number(match[1]));
  if (/明後日|あさって/.test(text)) return addDays(now, 2);
  if (/明日|あした/.test(text)) return addDays(now, 1);
  if (/今日|今夜|今晩|今朝|本日/.test(text)) return now;
  match = text.match(/(再来週|来週|今週|今度|次)(?:の)?([日月火水木金土])曜/);
  if (match) {
    const offset = (WEEKDAYS[match[2]] + 6) % 7;
    if (match[1] === '再来週') return addDays(mondayOf(now), 14 + offset);
    if (match[1] === '来週') return addDays(mondayOf(now), 7 + offset);
    if (match[1] === '今週') return addDays(mondayOf(now), offset);
    return addDays(now, (WEEKDAYS[match[2]] - weekday(now) + 7) % 7 || 7); // 今度 / 次: the next one
  }
  match = text.match(/([日月火水木金土])曜/);
  if (match) return addDays(now, (WEEKDAYS[match[1]] - weekday(now) + 7) % 7 || 7);
  match = text.match(/(\d{1,2})日(?![間後])/);
  if (match) {
    // A bare day of the month: this month if it has not passed yet, else next month.
    const day = Number(match[1]);
    const thisMonth = `${now.slice(0, 7)}-${pad(day)}`;
    if (thisMonth >= now) return thisMonth;
    return month === 12 ? `${year + 1}-01-${pad(day)}` : `${year}-${pad(month + 1)}-${pad(day)}`;
  }
  return null;
}

function timeOf(text) {
  if (/正午/.test(text)) return '12:00';
  let match = text.match(/(\d{1,2}):(\d{2})/);
  let hour;
  let minute;
  if (match) {
    [hour, minute] = [Number(match[1]), Number(match[2])];
  } else {
    match = text.match(/(\d{1,2})時(?:(\d{1,2})分|(半))?/);
    if (!match) return null;
    hour = Number(match[1]);
    minute = match[3] ? 30 : Number(match[2] ?? 0);
  }
  if (hour < 12 && hour > 0 && /午後|夜|夕方|今夜|今晩|晩/.test(text) && !/午前|朝/.test(text)) hour += 12;
  return `${pad(hour)}:${pad(minute)}`;
}

/**
 * The date and time a sentence talks about, relative to `now` (YYYY-MM-DD). Weeks run Monday to Sunday: 「来週の火曜」 is
 * the Tuesday of next week, a bare 「金曜」 or 「今度の日曜」 the next such day after today. Returns nulls for what the
 * sentence does not say; 「午前中」「お昼に」「夜に」 give a date but no clock time.
 */
export function parseWhen(text, now) {
  const date = dateOf(text, now);
  let endDate = null;
  const range = text.match(/から(\d{1,2})日まで/);
  const days = text.match(/(\d{1,2})日間/);
  if (date && range) endDate = `${date.slice(0, 8)}${pad(range[1])}`;
  else if (date && days) endDate = addDays(date, Number(days[1]) - 1);
  return { date, time: timeOf(text), endDate, allDay: Boolean(endDate || /終日|一日中|丸一日/.test(text)) };
}
