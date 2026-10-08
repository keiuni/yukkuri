// Mechanical checks of Japanese replies (scripts/ja-chat.mjs). They catch what small models get
// wrong most visibly: drifting into Chinese or Korean, looping, ignoring a length or format, getting a
// fact wrong. How natural or fitting a reply is, is judged separately.

// Simplified Chinese characters that Japanese does not use (none of them is among the 2,000 kanji of
// data/char-vocab.json). Small multilingual models sometimes slip into Chinese mid-sentence.
const SIMPLIFIED =
  '们这说么吗个为东车门问间关开对发经现实长还进过动种样让给买卖应该认识语话请谢见电脑网页视频务单达运输选择设计记录读习题级组织约纸线红绿蓝吧呢啊哪怎谁您很专业两严丽举乐书亚产亲从仓价众优伟传伤兴养军农决况净减击则刚创别剧劝办劳势华协历压厅县变启员响团园围图圆场坏块坚坛处备复够头夺奋妈审宽宾寻导尔层岁岛师广库废张弹归录总恶战执扩扫扬护报择换摄敌无时显术杀杂权极构标树检欢气汉汤沟泽洁测济浓润涨满滨灭灵灾烟热爱环疗盐监盖盘矿础确离积稳穷竞笔签简类粮紧编缘罗职联艺节药获营虑补观规览觉讨训议讯讲许论访证评诉词试诗详误诸课调谈谋负财责败货质购贵费贸资赏赛趋跃转轮软轻较辆辑边迁远违连迟适递遗邮邻释针钟钢钱铁银锁错键镜闭闲闻阅队阳阴阵阶际陆陈险隐难雾韩顶项顺须顾顿预领颜额风飞饭饮馆马驾验骑鱼鲜鸟鸡齐齿龙龟';
const SIMPLIFIED_SET = new Set(SIMPLIFIED);

const KANA = /[぀-ヿ]/u;
const KANJI = /[㐀-鿿]/u;
const LATIN = /[A-Za-z]/;
const HANGUL = /[가-힯ᄀ-ᇿ]/u;

/**
 * Share of Japanese letters (kana and kanji) among all letters, and characters from other scripts:
 * simplified Chinese and Hangul.
 */
export function scriptStats(text) {
  let japanese = 0;
  let latin = 0;
  const foreign = [];
  for (const char of text) {
    if (SIMPLIFIED_SET.has(char) || HANGUL.test(char)) foreign.push(char);
    else if (KANA.test(char) || KANJI.test(char)) japanese++;
    else if (LATIN.test(char)) latin++;
  }
  const letters = japanese + latin + foreign.length;
  return { jaRatio: letters ? japanese / letters : 0, foreign: [...new Set(foreign)].join('') };
}

/** A stretch of `size` characters that occurs `times` times or more: the model is going round in circles. */
export function findLoop(text, size = 12, times = 3) {
  const flat = text.replace(/\s+/g, '');
  const counts = new Map();
  for (let i = 0; i + size <= flat.length; i++) {
    const piece = flat.slice(i, i + size);
    const seen = counts.get(piece) ?? { count: 0, end: -1 };
    // Count non-overlapping occurrences only.
    if (i >= seen.end) counts.set(piece, { count: seen.count + 1, end: i + size });
    if ((counts.get(piece)?.count ?? 0) >= times) return piece;
  }
  return null;
}

/** Length in characters, without whitespace and without quotes the model may have put around the text. */
export function visibleLength(text) {
  return [...text.replace(/\s+/g, '').replace(/^[「『"“]+|[」』"”]+$/g, '')].length;
}

/** Lines that are list items (・ - * 1. ①). */
export function countBullets(text) {
  return text.split('\n').filter((line) => /^\s*(?:[-*・•●◦]|\d+[.)．）]|[①-⑩])\s*\S/.test(line)).length;
}

/** `groups` is a list of alternatives; every group must have at least one string in `text`. */
export function includesAll(text, groups) {
  const flat = text.replace(/\s+/g, '');
  return groups.every((group) => group.some((needle) => flat.includes(needle.replace(/\s+/g, ''))));
}

/** Every mechanical check for one reply; `spec` is the item (or the turn) of data/ja-chat.json. */
export function checkReply(text, spec = {}) {
  const { jaRatio, foreign } = scriptStats(text);
  const problems = [];
  if (!text.trim()) problems.push('空の返答');
  if (jaRatio < 0.6) problems.push('日本語が少ない');
  if (foreign) problems.push(`他の言語の文字（${foreign}）`);
  const loop = findLoop(text);
  if (loop) problems.push(`繰り返し（${loop}…）`);
  if (spec.maxChars && visibleLength(text) > spec.maxChars) problems.push(`${spec.maxChars} 文字を超えた（${visibleLength(text)} 文字）`);
  if (spec.bullets && countBullets(text) !== spec.bullets) problems.push(`箇条書きが ${spec.bullets} つでない（${countBullets(text)} つ）`);
  if (spec.includes && !includesAll(text, spec.includes)) problems.push('答えに必要な語がない');
  return { jaRatio, foreign, loop, length: visibleLength(text), problems };
}
