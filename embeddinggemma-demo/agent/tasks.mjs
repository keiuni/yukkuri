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
