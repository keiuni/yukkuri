#!/usr/bin/env node
// Builds public/data/char-vocab.json: the characters the character-by-character generator may
// choose from. Kanji are the most frequent ones in a fixed set of general Japanese Wikipedia
// articles (so the list does not lean towards any test sentence); kana, digits, Latin letters and
// common punctuation are always included.
//
//   NODE_USE_ENV_PROXY=1 node scripts/build-char-vocab.mjs [kanjiCount=2000]
import { writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'public', 'data', 'char-vocab.json');
const KANJI_COUNT = Number(process.argv[2] ?? 2000);

const ARTICLES = [
  '日本', '東京都', '大阪府', '北海道', '沖縄県', '富士山', '日本の歴史', '江戸時代', '明治維新', '第二次世界大戦',
  '経済', '政治', '日本国憲法', '科学', '物理学', '化学', '生物学', '数学', '医学', '健康',
  '音楽', '美術', '文学', '映画', 'アニメ', '漫画', 'テレビゲーム', '野球', 'サッカー', 'オリンピック',
  '日本料理', '寿司', 'ラーメン', '米', 'ネコ', 'イヌ', '鉄道', '新幹線', '自動車', '航空機',
  'コンピュータ', 'インターネット', 'スマートフォン', '人工知能', '宇宙', '地球', '気候', '地震', '仏教', '神道',
  '教育', '大学', '日本語', '英語', '家族', '仕事', '天気', '季節', '観光',
];

const HIRAGANA = 'ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわをん';
const KATAKANA = 'ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヲンヴー';
const SYMBOLS = '、。・「」（）？！〜々0123456789';
const LATIN = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,\'-';

function textOf(html) {
  return html
    .replace(/<(style|script)[\s\S]*?<\/\1>/g, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&[a-z#0-9]+;/gi, ' ');
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** Article HTML through the Action API, one request per second (Wikimedia's etiquette for bots). */
async function articleHtml(title) {
  const url = `https://ja.wikipedia.org/w/api.php?action=parse&format=json&formatversion=2&prop=text&redirects=1&page=${encodeURIComponent(title)}`;
  for (let attempt = 0; attempt < 4; attempt++) {
    const response = await fetch(url, { headers: { 'User-Agent': 'embeddinggemma-demo/0.1 (character vocabulary builder)' } });
    if (response.status === 429) {
      await sleep(5000 * (attempt + 1));
      continue;
    }
    if (!response.ok) throw new Error(`${response.status}`);
    const data = await response.json();
    if (data.error) throw new Error(data.error.info);
    return data.parse.text;
  }
  throw new Error('rate limited');
}

const counts = new Map();
const used = [];
for (const title of ARTICLES) {
  try {
    const text = textOf(await articleHtml(title));
    for (const char of text) {
      if (/\p{Script=Han}/u.test(char)) counts.set(char, (counts.get(char) ?? 0) + 1);
    }
    used.push(title);
    process.stdout.write('.');
  } catch (error) {
    console.warn(`skip ${title}: ${error.message}`);
  }
  await sleep(1000);
}
console.log();

const kanji = [...counts.entries()]
  .sort((a, b) => b[1] - a[1])
  .slice(0, KANJI_COUNT)
  .map(([char]) => char)
  .join('');
const vocab = {
  description: 'Characters for scripts/char-generate.mjs and the 文字生成 tab. Kanji: the most frequent in the listed Japanese Wikipedia articles.',
  createdAt: new Date().toISOString().slice(0, 10),
  articles: used,
  kanji,
  hiragana: HIRAGANA,
  katakana: KATAKANA,
  symbols: SYMBOLS,
  latin: LATIN,
};
await writeFile(OUT, `${JSON.stringify(vocab, null, 1)}\n`);
console.log(`${used.length} articles, ${counts.size} distinct kanji, kept ${kanji.length} -> ${path.relative(ROOT, OUT)}`);
console.log(`most frequent: ${kanji.slice(0, 40)}`);
console.log(`least frequent kept: ${kanji.slice(-20)}`);
