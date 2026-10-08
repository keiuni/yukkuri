#!/usr/bin/env node
// Records captioned videos of the demo being operated in Chromium, for watching the
// browser automation instead of reading test logs. A fake cursor, click ripples and
// captions are drawn on top of the page; waiting for model loads is cut or fast-forwarded.
//
//   npm run record                                  # every scenario
//   npm run record -- similarity benchmark          # only the named ones
//   RECORD_DEBUG=1 npm run record                   # also print how long each cut/fast-forward lasted
//
// Output: test-output/videos/<nn>-<scenario>.mp4 (H.264 via ffmpeg; WebM is kept if
// ffmpeg is missing). Needs the q8, q4 and fp32 models in ./models:
//   npm run download-model -- fp32 q8 q4
import { mkdir, rename, rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { chromium } from '@playwright/test';

import { startServer, toMp4 } from './lib/video.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUTPUT_DIR = path.join(ROOT, 'test-output', 'videos');
const RAW_DIR = path.join(OUTPUT_DIR, 'raw');
const PORT = Number(process.env.PORT ?? 5174);
const BASE_URL = `http://127.0.0.1:${PORT}`;
const VIEWPORT = { width: 1280, height: 720 };

// ---------------------------------------------------------------- overlay drawn into the page

function installOverlay() {
  const install = () => {
    if (document.getElementById('rec-cursor')) return;
    const style = document.createElement('style');
    style.textContent = `
      #rec-cursor { position: fixed; left: -40px; top: -40px; z-index: 2147483647; width: 24px; height: 24px; pointer-events: none; }
      .rec-ripple { position: fixed; z-index: 2147483646; width: 40px; height: 40px; margin: -20px 0 0 -20px; border-radius: 50%;
        border: 3px solid #eb6834; pointer-events: none; animation: rec-ripple 0.6s ease-out forwards; }
      @keyframes rec-ripple { from { transform: scale(0.3); opacity: 1; } to { transform: scale(1.5); opacity: 0; } }
      #rec-caption { position: fixed; left: 50%; bottom: 22px; z-index: 2147483645; transform: translateX(-50%);
        width: max-content; max-width: calc(100% - 64px); padding: 10px 20px; border-radius: 12px; pointer-events: none;
        background: rgba(11, 11, 11, 0.84); color: #fff; text-align: center; box-shadow: 0 8px 28px rgba(0, 0, 0, 0.28);
        font: 600 19px/1.5 system-ui, "Noto Sans JP", "Hiragino Sans", sans-serif; }
      #rec-caption small { display: block; font-size: 15px; font-weight: 400; opacity: 0.88; }
      #rec-caption .speed { display: inline-block; margin-right: 10px; padding: 0 8px; border-radius: 6px; background: #eb6834; font-size: 15px; }
      #rec-caption:empty { display: none; }
      #rec-title { position: fixed; inset: 0; z-index: 2147483644; display: flex; flex-direction: column; align-items: center;
        justify-content: center; gap: 12px; padding: 40px; background: rgba(13, 13, 13, 0.92); color: #fff; text-align: center;
        font-family: system-ui, "Noto Sans JP", "Hiragino Sans", sans-serif; transition: opacity 0.5s; }
      #rec-title h1 { margin: 0; font-size: 40px; line-height: 1.3; }
      #rec-title p { margin: 0; font-size: 21px; opacity: 0.85; }
      #rec-title.hide { opacity: 0; }
    `;
    document.head.append(style);

    const cursor = document.createElement('div');
    cursor.id = 'rec-cursor';
    cursor.innerHTML =
      '<svg width="24" height="24" viewBox="0 0 24 24"><path d="M3 2l16 9-7 1.6L8.6 20z" fill="#fff" stroke="#111" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    const caption = document.createElement('div');
    caption.id = 'rec-caption';
    document.body.append(cursor, caption);

    document.addEventListener(
      'mousemove',
      (event) => {
        cursor.style.left = `${event.clientX - 3}px`;
        cursor.style.top = `${event.clientY - 2}px`;
      },
      true,
    );
    const ripple = (x, y) => {
      const ring = document.createElement('div');
      ring.className = 'rec-ripple';
      ring.style.left = `${x}px`;
      ring.style.top = `${y}px`;
      document.body.append(ring);
      setTimeout(() => ring.remove(), 700);
    };
    document.addEventListener('mousedown', (event) => ripple(event.clientX, event.clientY), true);

    window.__rec = {
      ripple,
      caption(text, sub = '', speed = '') {
        caption.replaceChildren();
        if (!text) return;
        if (speed) {
          const badge = document.createElement('span');
          badge.className = 'speed';
          badge.textContent = speed;
          caption.append(badge);
        }
        caption.append(text);
        if (sub) {
          const small = document.createElement('small');
          small.textContent = sub;
          caption.append(small);
        }
      },
      title(text, sub) {
        const card = document.createElement('div');
        card.id = 'rec-title';
        const heading = document.createElement('h1');
        heading.textContent = text;
        const line = document.createElement('p');
        line.textContent = sub;
        card.append(heading, line);
        document.body.append(card);
        return new Promise((resolve) =>
          setTimeout(() => {
            card.classList.add('hide');
            setTimeout(() => {
              card.remove();
              resolve();
            }, 500);
          }, 2600),
        );
      },
    };
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', install);
  else install();
}

// ---------------------------------------------------------------- recording helpers

class Recording {
  constructor(page) {
    this.page = page;
    this.started = Date.now();
    this.edits = []; // { start, end, factor } in seconds; factor Infinity removes the span, 'fast' speeds it up
  }

  seconds() {
    return (Date.now() - this.started) / 1000;
  }

  async span(factor, action) {
    const start = this.seconds();
    const result = await action();
    this.edits.push({ start, end: this.seconds(), factor });
    return result;
  }

  cut(action) {
    return this.span(Infinity, action);
  }

  async fastForward(text, action) {
    await this.caption(text, '', '▶▶ 早送り');
    return this.span('fast', action);
  }

  async caption(text, sub = '', speed = '') {
    await this.page.evaluate(([t, s, f]) => window.__rec.caption(t, s, f), [text, sub, speed]);
  }

  async say(text, sub = '', pause = 2600) {
    await this.caption(text, sub);
    await this.page.waitForTimeout(pause);
  }

  /** Shows a full-screen title card; resolves once it has faded out. */
  async title(text, sub) {
    await this.page.evaluate(([t, s]) => window.__rec.title(t, s), [text, sub]);
  }

  async moveTo(locator) {
    await locator.scrollIntoViewIfNeeded();
    const box = await locator.boundingBox();
    const x = box.x + Math.min(box.width / 2, 120);
    const y = box.y + box.height / 2;
    await this.page.mouse.move(x, y, { steps: 18 });
    await this.page.waitForTimeout(200);
    return { x, y };
  }

  async click(locator) {
    await this.moveTo(locator);
    await locator.click();
    await this.page.waitForTimeout(350);
  }

  /** Native <select> popups are not captured on video, so show the click and set the value. */
  async select(locator, value) {
    const { x, y } = await this.moveTo(locator);
    await this.page.evaluate(([px, py]) => window.__rec.ripple(px, py), [x, y]);
    await this.page.waitForTimeout(350);
    await locator.selectOption(value);
    await this.page.waitForTimeout(500);
  }

  async type(locator, text) {
    await this.click(locator);
    await this.page.keyboard.press('Control+A');
    await this.page.keyboard.press('Backspace');
    await locator.pressSequentially(text, { delay: 110 });
    await this.page.waitForTimeout(400);
  }

  async scrollTo(locator, offset = 90) {
    await locator.evaluate((el, top) => window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - top, behavior: 'smooth' }), offset);
    await this.page.waitForTimeout(900);
  }

  /** Clicks a run button and waits for its results container to report a new run. */
  async run(buttonName, resultsId) {
    const previous = Number((await this.page.locator(`#${resultsId}`).getAttribute('data-run')) ?? 0);
    await this.click(this.page.getByRole('button', { name: buttonName }));
    await this.page.locator(`#${resultsId}[data-run="${previous + 1}"]`).waitFor({ timeout: 10 * 60 * 1000 });
  }

  async waitForModel() {
    await this.page.locator('#model-status[data-state="ready"]').waitFor({ timeout: 10 * 60 * 1000 });
  }

  /** Opens the demo with the given model already loaded; the loading itself is cut from the video. */
  async openLoaded(dtype = 'q8') {
    await this.cut(async () => {
      await this.page.goto(`${BASE_URL}/?cache=0&threads=4&dtype=${dtype}&autoload=1`);
      await this.waitForModel();
      await this.page.mouse.move(640, 420);
    });
  }
}

const searchState = (page) => page.evaluate(() => window.__demo.last.search);
const rankOf = (search, index) => search.ranking.findIndex((row) => row.index === index) + 1;
const score = (x) => x.toFixed(3);

// ---------------------------------------------------------------- scenarios

const SCENARIOS = [
  {
    id: 'load-and-search',
    title: '① モデルを読み込んで検索する',
    subtitle: 'q8（331 MB）を WASM で読み込み、モデルカードの例を再現',
    async run(rec, page) {
      await page.goto(`${BASE_URL}/?cache=0`);
      await page.locator('body[data-ready="true"]').waitFor();
      await rec.title(this.title, this.subtitle);

      await rec.caption('精度・実行環境・スレッド数を選ぶ');
      await rec.select(page.getByLabel('精度'), 'q8');
      await rec.select(page.getByLabel('スレッド数'), '4');
      await rec.caption('「モデルを読み込む」をクリック');
      await rec.click(page.getByRole('button', { name: 'モデルを読み込む' }));
      await rec.fastForward('モデルを読み込み中（ダウンロード → 初期化 → 初回推論）', () => rec.waitForModel());
      const model = await page.evaluate(() => window.__demo.model);
      await rec.moveTo(page.locator('#model-facts'));
      await rec.say('準備完了', `読み込み ${(model.loadMs / 1000).toFixed(1)} 秒・WASM ${model.numThreads} スレッド・クロスオリジン分離が有効`, 3000);

      await rec.caption('サンプル「惑星（モデルカードの例）」で検索');
      await rec.run('検索する', 'search-results');
      await rec.scrollTo(page.locator('#search-results'));
      let search = await searchState(page);
      await rec.say(
        search.ranking[0].index === 1 ? `Mars の文が1位（${score(search.ranking[0].score)}）` : `1位は「${search.ranking[0].text.slice(0, 30)}…」`,
        '括弧内はモデルカードに載っている fp32 の公式スコア',
        4200,
      );

      await rec.scrollTo(page.locator('#search-dims'), 200);
      await rec.caption('次元を 768 → 128 に減らして再検索（Matryoshka 表現）');
      await rec.select(page.locator('#search-dims'), '128');
      await rec.run('検索する', 'search-results');
      await rec.scrollTo(page.locator('#search-results'));
      search = await searchState(page);
      await rec.say(
        search.ranking[0].index === 1 ? '128次元でも Mars が1位のまま' : '128次元では順位が変わった',
        '埋め込みは計算済みなので再利用し、先頭128次元だけで比べ直している',
        4000,
      );
    },
  },
  {
    id: 'japanese-search',
    title: '② 日本語で自由に検索する',
    subtitle: 'ECサイトのFAQに、言い回しを変えた質問を入力',
    async run(rec, page) {
      await rec.openLoaded('q8');
      await rec.title(this.title, this.subtitle);

      await rec.caption('サンプル「ECサイトFAQ（日本語）」を選ぶ');
      await rec.select(page.getByLabel('サンプル'), 'faq-ja');
      await rec.moveTo(page.locator('#search-docs'));
      await rec.say('この6件の文書から探す', 'パスワード／領収書／配送状況／返品／退会とポイント／カード情報の変更', 3000);

      const queries = [
        {
          text: '買った服を送り返したい',
          expected: 3,
          ok: '「返品」という言葉を使っていないのに、返品の案内が1位',
        },
        {
          text: '荷物がまだ届かない',
          expected: 2,
          ok: '「配送状況」の案内が1位',
        },
        {
          text: 'カードの有効期限が切れた',
          expected: 5,
          ok: 'カード情報の変更方法が1位',
          miss: '「期限切れ」が「ポイント失効」に引っ張られた。言葉の連想で外れることもある',
        },
      ];
      for (const query of queries) {
        await rec.scrollTo(page.locator('#search-query'), 160);
        await rec.caption('クエリを入力');
        await rec.type(page.locator('#search-query'), query.text);
        await rec.caption(`「${query.text}」で検索`);
        await rec.run('検索する', 'search-results');
        await rec.scrollTo(page.locator('#search-results'));
        const search = await searchState(page);
        const rank = rankOf(search, query.expected);
        const target = search.ranking.find((row) => row.index === query.expected);
        if (rank === 1) {
          await rec.say(`○ ${query.ok}`, `スコア ${score(target.score)}`, 3800);
        } else {
          await rec.say(
            `× 狙った文書（カード情報の変更）は${rank}位`,
            query.miss ?? `1位は「${search.ranking[0].text.slice(0, 24)}…」`,
            5000,
          );
        }
      }
    },
  },
  {
    id: 'cross-lingual-and-miss',
    title: '③ 日本語→英語の検索と、苦手なケース',
    subtitle: '言語をまたぐ検索は得意。アニメ制作の専門用語は苦手',
    async run(rec, page) {
      await rec.openLoaded('q8');
      await rec.title(this.title, this.subtitle);

      await rec.caption('サンプル「日本語クエリ × 英語文書」を選んで検索');
      await rec.select(page.getByLabel('サンプル'), 'cross-lingual');
      await rec.run('検索する', 'search-results');
      await rec.scrollTo(page.locator('#search-results'));
      let search = await searchState(page);
      const onionRings = rankOf(search, 3);
      await rec.say(
        `「猫に玉ねぎをあげても大丈夫？」→ 猫と玉ねぎの英文が${rankOf(search, 0)}位（${score(search.ranking.find((row) => row.index === 0).score)}）`,
        onionRings > 1 ? `単語が似ているだけの “Onion rings” の文は${onionRings}位` : '“Onion rings” の文が1位になってしまった',
        4500,
      );

      await rec.scrollTo(page.locator('#search-preset'), 200);
      await rec.caption('サンプル「アニメ制作の工程」で検索');
      await rec.select(page.getByLabel('サンプル'), 'anime-ja');
      await rec.run('検索する', 'search-results');
      await rec.scrollTo(page.locator('#search-results'));
      search = await searchState(page);
      await rec.say(
        `想定解の「動画」は${search.expectedRank}位`,
        `「原画と原画のあいだの絵を描く担当」に対し、1位は「${search.ranking[0].text.split('：')[0]}」。専門用語の区別は苦手`,
        5200,
      );
    },
  },
  {
    id: 'similarity',
    title: '④ 類似度マトリクス',
    subtitle: '日本語と英語の言い換えが、同じ意味どうしでまとまるか',
    async run(rec, page) {
      await rec.openLoaded('q8');
      await rec.title(this.title, this.subtitle);

      await rec.click(page.getByRole('tab', { name: '類似度マトリクス' }));
      await rec.moveTo(page.locator('#similarity-sentences'));
      await rec.say('日英の言い換えを含む8文', '暑い（日・日・英）／ラーメン（日・英）／会議（日・英）／株価（日）', 3000);
      await rec.caption('「類似度を計算」をクリック');
      await rec.run('類似度を計算', 'similarity-results');
      await rec.scrollTo(page.locator('#similarity-results'), 40);
      await rec.say('同じ意味の文どうしが濃い青のブロックになる', '', 2600);

      const { matrix, sentences } = await page.evaluate(() => window.__demo.last.similarity);
      const cellScore = (i, j) => matrix[i * sentences.length + j].toFixed(2);
      const groups = [[0, 1, 2], [3, 4], [5, 6], [7]]; // hot weather, ramen, meeting, stocks
      const groupOf = (i) => groups.findIndex((group) => group.includes(i));
      const different = [];
      for (let i = 0; i < sentences.length; i++) {
        for (let j = i + 1; j < sentences.length; j++) if (groupOf(i) !== groupOf(j)) different.push(matrix[i * sentences.length + j]);
      }
      const hover = async (i, j, text, sub) => {
        await rec.moveTo(page.locator(`#similarity-results td[data-i="${i}"][data-j="${j}"]`));
        await rec.say(text, sub, 3400);
      };
      await hover(0, 2, `「今日はとても暑いですね。」× “It's really hot today.” = ${cellScore(0, 2)}`, '言語が違っても意味が同じなら高い');
      await hover(3, 4, `「このラーメンはとても美味しい。」× “This ramen is delicious.” = ${cellScore(3, 4)}`, '');
      await hover(
        0,
        3,
        `「今日はとても暑いですね。」× 「このラーメンはとても美味しい。」= ${cellScore(0, 3)}`,
        `意味が違う組は ${Math.min(...different).toFixed(2)}〜${Math.max(...different).toFixed(2)}`,
      );

      await page.mouse.move(640, 300, { steps: 10 });
      await rec.scrollTo(page.locator('#similarity-sentences'), 120);
      await rec.caption('英文を1行追加してみる');
      const textarea = page.locator('#similarity-sentences');
      await rec.click(textarea);
      await page.keyboard.press('Control+End');
      await page.keyboard.press('Enter');
      await page.keyboard.type('Stock prices fell sharply.', { delay: 90 });
      await page.waitForTimeout(500);
      await rec.run('類似度を計算', 'similarity-results');
      await rec.scrollTo(page.locator('#similarity-results'), 40);
      const updated = await page.evaluate(() => window.__demo.last.similarity);
      const n = updated.sentences.length;
      await rec.moveTo(page.locator(`#similarity-results td[data-i="${n - 1}"][data-j="${n - 2}"]`));
      await rec.say(
        `追加した “Stock prices fell sharply.” は「株価が大きく下落した。」と ${updated.matrix[(n - 1) * n + (n - 2)].toFixed(2)}`,
        `${n}行目で濃いのは、同じ意味の ${n - 1}列目（と自分自身）だけ`,
        4200,
      );
    },
  },
  {
    id: 'precision',
    title: '⑤ 精度を切り替えて比べる',
    subtitle: '同じ例を q8 → fp32 → q4 で検索し、モデルカードの値と比べる',
    async run(rec, page) {
      await rec.openLoaded('q8');
      await rec.title(this.title, this.subtitle);

      const MODEL_CARD = [0.30109718441963196, 0.6358831524848938, 0.4930494725704193, 0.48887503147125244];
      const searchAndReport = async (label) => {
        await rec.caption(`${label}：モデルカードの例を検索`);
        await rec.run('検索する', 'search-results');
        await rec.scrollTo(page.locator('#search-results'));
        const search = await searchState(page);
        const maxDiff = Math.max(...search.ranking.map((row) => Math.abs(row.score - MODEL_CARD[row.index])));
        const sameOrder = search.ranking.map((row) => row.index).join() === '1,2,3,0';
        return { search, maxDiff, sameOrder };
      };
      const switchTo = async (dtype, label) => {
        await page.mouse.move(640, 300, { steps: 8 });
        await rec.scrollTo(page.locator('#model-heading'), 30);
        await rec.caption(`精度を ${label} に切り替えて読み込み直す`);
        await rec.select(page.getByLabel('精度'), dtype);
        await rec.click(page.getByRole('button', { name: '別の設定で読み込み直す' }));
        await rec.fastForward(`${label} を読み込み中`, () => rec.waitForModel());
        await rec.caption('');
      };

      const verdict = ({ maxDiff, sameOrder }) =>
        !sameOrder ? 'モデルカードと順位が変わった' : maxDiff < 0.001 ? 'モデルカードの値を再現できている' : maxDiff < 0.01 ? '順位は同じで、スコアもほぼ同じ' : '順位は同じだが、スコアは少しずれる';

      let result = await searchAndReport('q8（int8・331 MB）');
      await rec.say(`q8：モデルカードの値との差は最大 ${result.maxDiff.toFixed(4)}`, verdict(result), 4000);

      await switchTo('fp32', 'fp32（1.26 GB）');
      result = await searchAndReport('fp32');
      await rec.say(`fp32：差は最大 ${result.maxDiff.toFixed(4)}`, verdict(result), 4000);

      await switchTo('q4', 'q4（4bit・217 MB）');
      result = await searchAndReport('q4');
      await rec.say(`q4：差は最大 ${result.maxDiff.toFixed(4)}`, verdict(result), 4500);
    },
  },
  {
    id: 'benchmark',
    title: '⑥ ミニベンチマーク',
    subtitle: '日英の検索データ（文書63件・クエリ56件）で精度と速度を測る',
    async run(rec, page) {
      await rec.openLoaded('q8');
      await rec.title(this.title, this.subtitle);

      await rec.click(page.getByRole('tab', { name: 'ミニベンチマーク' }));
      await rec.moveTo(page.locator('#benchmark-intro'));
      await rec.say('各話題に「同じ話題だが答えではない」紛らわしい文書を入れてある', 'プロンプトの有無 × 次元（768/512/256/128）も比較する', 3400);
      await rec.scrollTo(page.getByRole('button', { name: 'ベンチマークを実行' }), 260);
      await rec.caption('「ベンチマークを実行」をクリック');
      const previous = Number((await page.locator('#benchmark-results').getAttribute('data-run')) ?? 0);
      await rec.click(page.getByRole('button', { name: 'ベンチマークを実行' }));
      await rec.fastForward('文書とクエリを順に埋め込み中（約240件）', () =>
        page.locator(`#benchmark-results[data-run="${previous + 1}"]`).waitFor({ timeout: 10 * 60 * 1000 }),
      );

      const result = await page.evaluate(() => window.__demo.last.benchmark);
      const main = result.evaluations.find((row) => row.mode === 'prompt' && row.dims === 768).overall;
      await rec.scrollTo(page.locator('#benchmark-results .tiles'), 120);
      await rec.say(
        `Top-1 正解率 ${(main.acc1 * 100).toFixed(1)}%（${Math.round(main.acc1 * main.count)}/${main.count}問）・MRR@10 ${main.mrr10.toFixed(3)}`,
        `文書 ${Math.round(result.speed.documents.msPerText)} ms/件・検索 ${Math.round(result.speed.queries.p50Ms)} ms（中央値、この録画環境での値）`,
        4200,
      );
      const mrr = (mode, dims) => result.evaluations.find((row) => row.mode === mode && row.dims === dims).overall.mrr10.toFixed(3);
      const byType = result.evaluations.find((row) => row.mode === 'prompt' && row.dims === 768).byType;
      const pct = (type) => `${((byType[type]?.acc1 ?? 0) * 100).toFixed(1)}%`;
      const firstMiss = result.perQuery.find((row) => row.rank !== 1);
      const missTop = firstMiss && (await page.evaluate((id) => window.__demo.dataset.docs.find((doc) => doc.id === id).text, firstMiss.top[0].id));

      const headings = page.locator('#benchmark-results h3');
      await rec.scrollTo(headings.nth(0), 70);
      await rec.say(
        `MRR@10：プロンプトあり ${mrr('prompt', 768)} / なし ${mrr('raw', 768)}（768次元）`,
        `次元を減らすと 512: ${mrr('prompt', 512)} → 256: ${mrr('prompt', 256)} → 128: ${mrr('prompt', 128)}`,
        3800,
      );
      await rec.scrollTo(headings.nth(1), 70);
      const acc = (type) => byType[type]?.acc1 ?? 0;
      const crossHolds = Math.min(acc('ja→en'), acc('en→ja')) >= Math.min(acc('ja→ja'), acc('en→en'));
      await rec.say(
        `Top-1：日→英 ${pct('ja→en')}・英→日 ${pct('en→ja')}・英→英 ${pct('en→en')}・日→日 ${pct('ja→ja')}`,
        crossHolds ? '言語をまたぐ検索も、同じ言語どうしの検索と遜色ない' : '言語をまたぐ検索の方が弱い',
        3800,
      );
      await rec.scrollTo(headings.nth(3), 70);
      await rec.say(
        '外したクエリと、代わりに1位になった文書',
        firstMiss ? `例：「${firstMiss.text}」→ 1位は「${missTop.slice(0, 26)}…」` : '外したクエリはなし',
        4500,
      );
    },
  },
];

// ---------------------------------------------------------------- main

const wanted = process.argv.slice(2);
const scenarios = wanted.length > 0 ? SCENARIOS.filter((scenario) => wanted.includes(scenario.id)) : SCENARIOS;
if (scenarios.length === 0) {
  console.error(`Unknown scenario. Choose from: ${SCENARIOS.map((scenario) => scenario.id).join(', ')}`);
  process.exit(1);
}

await mkdir(RAW_DIR, { recursive: true });
const server = await startServer(ROOT, PORT);
const browser = await chromium.launch();
try {
  for (const scenario of scenarios) {
    const number = String(SCENARIOS.indexOf(scenario) + 1).padStart(2, '0');
    const context = await browser.newContext({ viewport: VIEWPORT, recordVideo: { dir: RAW_DIR, size: VIEWPORT } });
    await context.addInitScript(installOverlay);
    const page = await context.newPage();
    const rec = new Recording(page);
    process.stdout.write(`${number} ${scenario.id} … `);
    try {
      await scenario.run(rec, page);
      if (process.env.RECORD_DEBUG) {
        const spans = rec.edits.map(({ start, end, factor }) => `${factor === Infinity ? 'cut' : factor} ${(end - start).toFixed(1)}s`);
        process.stdout.write(`[${spans.join(', ')}; model loadMs=${Math.round(await page.evaluate(() => window.__demo.model?.loadMs ?? 0))}] `);
      }
      await page.evaluate(() => window.__rec.caption(''));
      await page.waitForTimeout(800);
    } finally {
      await context.close();
    }
    const webm = path.join(RAW_DIR, `${number}-${scenario.id}.webm`);
    await rename(await page.video().path(), webm);
    const output = await toMp4(webm, path.join(OUTPUT_DIR, `${number}-${scenario.id}.mp4`), rec.edits);
    console.log(path.relative(ROOT, output));
  }
} finally {
  await browser.close();
  server.kill();
}
await rm(RAW_DIR, { recursive: true, force: true }).catch(() => {});
