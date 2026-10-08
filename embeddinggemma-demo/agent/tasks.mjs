// Tasks for the Gemma browser agent. Each has a natural-language goal, the sub-goals a
// coding agent would hand to a Jev-style decider (written in everyday words on purpose, not
// copied from the UI labels), and an independent outcome check: the agent's own "done" is
// never trusted, as in jev-ultrafast.

const params = (url) => new URL(url).searchParams;

function check(name, ok, detail) {
  return { name, ok: Boolean(ok), detail };
}

// Fixed so that relative dates in the goals ("先月") always point at the same orders.
export const TODAY = new Date(2026, 9, 8);

export const TASKS = [
  {
    id: 'hotel',
    site: 'たびのやど（宿泊予約サイト）',
    start: '/sites/hotel/index.html',
    goal: '京都で12月19日から1泊、大人2名で泊まれる宿を探して、禁煙で朝食付きの部屋に絞り込み、評価の高い順に並べて',
    steps: [
      { step: '行き先に「京都」と入れる', text: '京都' },
      { step: 'チェックイン日を12月19日にする', text: '' },
      { step: '宿泊数を1泊にする', text: '' },
      { step: '大人の人数を2人にする', text: '' },
      { step: '検索を実行する', text: '' },
      { step: '禁煙の部屋だけに絞り込む', text: '' },
      { step: '朝食付きのプランだけに絞り込む', text: '' },
      { step: '評価が高い順に並べ替える', text: '' },
    ],
    verify: async (page) => {
      const p = params(page.url());
      const filters = (p.get('filters') ?? '').split(',');
      return [
        check('検索結果ページ', page.url().includes('/results.html'), new URL(page.url()).pathname),
        check('目的地=京都', p.get('city') === '京都', p.get('city')),
        check('チェックイン=12/19', p.get('checkin') === '2026-12-19', p.get('checkin')),
        check('1泊', p.get('nights') === '1', p.get('nights')),
        check('大人2名', p.get('adults') === '2', p.get('adults')),
        check('禁煙', filters.includes('nonsmoking'), p.get('filters')),
        check('朝食付き', filters.includes('breakfast'), p.get('filters')),
        check('評価が高い順', p.get('sort') === 'rating', p.get('sort')),
      ];
    },
  },
  {
    id: 'shop-receipt',
    site: 'みどりマート（通販サイト）',
    start: '/sites/shop/index.html',
    goal: '先月注文した商品の領収書を発行して',
    steps: [
      { step: '注文履歴を開く', text: '' },
      { step: '2026年9月の注文の領収書を出す', text: '' },
    ],
    verify: async (page) => [
      check('領収書ページ', page.url().includes('/receipt.html'), new URL(page.url()).pathname),
      check('9月の注文（503-2026091402）', params(page.url()).get('order') === '503-2026091402', params(page.url()).get('order')),
    ],
  },
  {
    id: 'shop-search',
    site: 'みどりマート（通販サイト）',
    start: '/sites/shop/index.html',
    goal: 'ワイヤレスイヤホンを検索して、価格の安い順に並べて',
    steps: [
      { step: '商品検索に「ワイヤレスイヤホン」と入れる', text: 'ワイヤレスイヤホン' },
      { step: '検索を実行する', text: '' },
      { step: '値段が安い順に並べ替える', text: '' },
    ],
    verify: async (page) => [
      check('検索結果ページ', page.url().includes('/search.html'), new URL(page.url()).pathname),
      check('キーワード=ワイヤレスイヤホン', (params(page.url()).get('q') ?? '').trim() === 'ワイヤレスイヤホン', params(page.url()).get('q')),
      check('価格の安い順', params(page.url()).get('sort') === 'price_asc', params(page.url()).get('sort')),
    ],
  },
  {
    id: 'settings-mail',
    site: 'ノートクラウド（アカウント設定）',
    start: '/sites/settings/index.html',
    goal: 'メール通知をオフにして、表示言語を英語にして保存して',
    steps: [
      { step: 'メールでのお知らせを止める', text: '' },
      { step: '画面の言語を英語にする', text: '' },
      { step: '設定を保存する', text: '' },
    ],
    verify: async (page) => {
      const p = params(page.url());
      return [
        check('保存された', p.get('saved') === '1', p.get('saved')),
        check('メール通知オフ', p.get('email') === 'false', p.get('email')),
        check('表示言語=English', p.get('lang') === 'en', p.get('lang')),
        check('他の通知はそのまま', p.get('push') === 'true' && p.get('newsletter') === 'false', `${p.get('push')}/${p.get('newsletter')}`),
        check('アカウントは削除していない', p.get('deleted') !== '1', p.get('deleted')),
      ];
    },
  },
  {
    id: 'settings-security',
    site: 'ノートクラウド（アカウント設定）',
    start: '/sites/settings/index.html',
    goal: 'テーマをダークにして、二段階認証をオンにしてから保存して',
    steps: [
      { step: '画面を暗い配色にする', text: '' },
      { step: 'ログイン時に確認コードを求める設定（二段階認証）を有効にする', text: '' },
      { step: '設定を保存する', text: '' },
    ],
    verify: async (page) => {
      const p = params(page.url());
      return [
        check('保存された', p.get('saved') === '1', p.get('saved')),
        check('テーマ=ダーク', p.get('theme') === 'dark', p.get('theme')),
        check('二段階認証オン', p.get('twofactor') === 'true', p.get('twofactor')),
        check('アカウントは削除していない', p.get('deleted') !== '1', p.get('deleted')),
      ];
    },
  },
  {
    id: 'wikipedia',
    site: 'ウィキペディア日本語版',
    start: 'https://ja.wikipedia.org/',
    goal: 'ウィキペディアで東京タワーの記事を開いて',
    steps: [
      { step: 'ウィキペディアの検索ボックスに「東京タワー」と入れる', text: '東京タワー' },
      { step: '検索を実行する', text: '' },
    ],
    verify: async (page) => {
      const title = await page.title();
      return [check('「東京タワー」の記事', /^東京タワー\s*-\s*Wikipedia/.test(title), title)];
    },
  },
];

// Tasks built around the kinds of steps the failure probes found hard for EmbeddingGemma
// (scripts/failure-probes.mjs). The steps are still ones a person would follow without trouble.
// `kind` names the difficulty and `expect` the right action (the runner shows it in the panel).
// `plainSteps` are the same steps as a planner that has read the page would write them (the page's own
// words, no negation, no comparison, no conversion); agent/run.mjs --plain uses them.
export const HARD_TASKS = [
  {
    id: 'hard-cancel',
    kind: '否定',
    site: 'ノートクラウド（アカウント設定）',
    start: '/sites/settings/index.html',
    goal: 'テーマをダークにしてみたけれど、やっぱり保存しないで取り消して',
    expect: 'テーマを変えたあと「キャンセル」を押す（「変更を保存」は押さない）',
    steps: [
      { step: '画面を暗い配色にする', text: '' },
      { step: '変更を保存しないで取り消す', text: '' },
    ],
    plainSteps: [
      { step: 'テーマを「ダーク」にする', text: '' },
      { step: '「キャンセル」を押す', text: '' },
    ],
    verify: async (page) => {
      const p = params(page.url());
      const theme = await page.locator('#theme').inputValue().catch(() => null);
      return [
        check('保存していない', p.get('saved') !== '1', p.get('saved')),
        check('アカウントは削除していない', p.get('deleted') !== '1', p.get('deleted')),
        check('テーマが元に戻った（キャンセルした）', theme === 'system', theme),
      ];
    },
  },
  {
    id: 'hard-push-only',
    kind: '否定（〜はそのまま）',
    site: 'ノートクラウド（アカウント設定）',
    start: '/sites/settings/index.html',
    goal: 'メール通知はそのままにして、プッシュ通知だけ止めて保存して',
    expect: '「プッシュ通知」だけをオフにして保存する',
    steps: [
      { step: 'メール通知はそのままにして、プッシュ通知だけを止める', text: '' },
      { step: '設定を保存する', text: '' },
    ],
    plainSteps: [
      { step: '「プッシュ通知」をオフにする', text: '' },
      { step: '設定を保存する', text: '' },
    ],
    verify: async (page) => {
      const p = params(page.url());
      return [
        check('保存された', p.get('saved') === '1', p.get('saved')),
        check('プッシュ通知オフ', p.get('push') === 'false', p.get('push')),
        check('メール通知はオンのまま', p.get('email') === 'true', p.get('email')),
      ];
    },
  },
  {
    id: 'hard-sort',
    kind: '反対の意味（言い換え）',
    site: 'みどりマート（通販サイト）',
    start: '/sites/shop/index.html',
    goal: 'ワイヤレスイヤホンを検索して、値段が張るものから順に並べて',
    expect: '表示順で「価格の高い順」を選ぶ',
    steps: [
      { step: '商品検索に「ワイヤレスイヤホン」と入れる', text: 'ワイヤレスイヤホン' },
      { step: '検索を実行する', text: '' },
      { step: '値段が張るものから順に並べる', text: '' },
    ],
    plainSteps: [
      { step: '商品検索に「ワイヤレスイヤホン」と入れる', text: 'ワイヤレスイヤホン' },
      { step: '検索を実行する', text: '' },
      { step: '表示順を「価格の高い順」にする', text: '' },
    ],
    verify: async (page) => [
      check('検索結果ページ', page.url().includes('/search.html'), new URL(page.url()).pathname),
      check('キーワード=ワイヤレスイヤホン', (params(page.url()).get('q') ?? '').trim() === 'ワイヤレスイヤホン', params(page.url()).get('q')),
      check('価格の高い順', params(page.url()).get('sort') === 'price_desc', params(page.url()).get('sort')),
    ],
  },
  {
    id: 'hard-route',
    kind: '向き・時刻の書き換え',
    site: 'のりかえナビ（経路検索）',
    start: '/sites/transit/index.html',
    goal: '博多から名古屋まで、午後3時ごろに出発する経路を調べて',
    expect: '到着地に「名古屋」、出発地に「博多」、出発時刻は「15時」',
    steps: [
      { step: '行き先を「名古屋」にする', text: '名古屋' },
      { step: '出発する駅を「博多」にする', text: '博多' },
      { step: '出発時刻を午後3時にする', text: '' },
      { step: '経路を検索する', text: '' },
    ],
    plainSteps: [
      { step: '到着地に「名古屋」と入れる', text: '名古屋' },
      { step: '出発地に「博多」と入れる', text: '博多' },
      { step: '出発時刻を「15時」にする', text: '' },
      { step: '経路を検索する', text: '' },
    ],
    verify: async (page) => {
      const p = params(page.url());
      return [
        check('検索結果ページ', page.url().includes('/transit/results.html'), new URL(page.url()).pathname),
        check('出発地=博多', p.get('from') === '博多', p.get('from')),
        check('到着地=名古屋', p.get('to') === '名古屋', p.get('to')),
        check('出発時刻=15時', p.get('time') === '15', p.get('time')),
      ];
    },
  },
  {
    id: 'hard-oldest',
    kind: '最上級（いちばん古い）',
    site: 'みどりマート（通販サイト）',
    start: '/sites/shop/index.html',
    goal: 'いちばん古い注文の領収書を発行して',
    expect: '2026年7月3日の注文の「領収書を発行する」を押す',
    steps: [
      { step: '注文履歴を開く', text: '' },
      { step: 'いちばん古い注文の領収書を出す', text: '' },
    ],
    plainSteps: [
      { step: '注文履歴を開く', text: '' },
      { step: '2026年7月3日の注文の領収書を発行する', text: '' },
    ],
    verify: async (page) => [
      check('領収書ページ', page.url().includes('/receipt.html'), new URL(page.url()).pathname),
      check('7月の注文（503-2026070304）', params(page.url()).get('order') === '503-2026070304', params(page.url()).get('order')),
    ],
  },
  {
    id: 'hard-icon',
    kind: '記号だけのボタン',
    site: 'みどりマート（通販サイト）',
    start: '/sites/shop/index.html?banner=1',
    goal: 'タイムセールのお知らせを閉じて',
    expect: 'お知らせの「×」を押す（タイムセールのページは開かない）',
    steps: [{ step: 'タイムセールのお知らせを閉じる', text: '' }],
    plainSteps: [{ step: 'お知らせの「×」ボタンを押す', text: '' }],
    verify: async (page) => {
      const hidden = await page.locator('#sale-banner').isHidden().catch(() => null);
      return [
        check('お知らせが閉じた', hidden === true && params(page.url()).get('banner') === 'closed', `${hidden}/${params(page.url()).get('banner')}`),
      ];
    },
  },
  {
    id: 'hard-missing',
    kind: 'できない手順',
    site: 'みどりマート（通販サイト）',
    start: '/sites/shop/index.html',
    goal: 'クーポンコード「AUTUMN10」を入力して',
    expect: 'このページにクーポンの入力欄はないので、何もしない',
    steps: [{ step: 'クーポンコードの入力欄に「AUTUMN10」と入れる', text: 'AUTUMN10' }],
    // A planner that has read the page sees there is no coupon field and hands over no step.
    plainSteps: [],
    verify: async (page) => {
      const value = await page.locator('input[name=q]').inputValue().catch(() => null);
      return [
        check('商品検索に入力していない', value === '', value),
        check('ページを移動していない', new URL(page.url()).pathname.endsWith('/shop/index.html'), new URL(page.url()).pathname),
      ];
    },
  },
];
