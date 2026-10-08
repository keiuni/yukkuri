// Sample inputs for the search and similarity tabs. `expectedTop` is the index of the
// document a person would pick; the UI marks it so misses are easy to spot.

export const SEARCH_PRESETS = [
  {
    id: 'model-card',
    label: '惑星（モデルカードの例・英語）',
    task: 'search result',
    query: 'Which planet is known as the Red Planet?',
    docs: [
      "Venus is often called Earth's twin because of its similar size and proximity.",
      'Mars, known for its reddish appearance, is often referred to as the Red Planet.',
      'Jupiter, the largest planet in our solar system, has a prominent red spot.',
      'Saturn, famous for its rings, is sometimes mistaken for the Red Planet.',
    ],
    expectedTop: 1,
    // Scores published in the onnx-community model cards, for comparison: fp32 for the first model,
    // q4 (three decimals) for EmbeddingGemma 2, whose card runs the example with dtype "q4".
    referenceScores: {
      v1: { dtype: 'fp32', scores: [0.30109718441963196, 0.6358831524848938, 0.4930494725704193, 0.48887503147125244] },
      v2: { dtype: 'q4', scores: [0.684, 0.854, 0.752, 0.783] },
    },
  },
  {
    id: 'faq-ja',
    label: 'ECサイトFAQ（日本語）',
    task: 'search result',
    query: '先月買った物の領収書がほしい',
    docs: [
      'パスワードを忘れた場合は、ログイン画面の「パスワードを再設定」から手続きしてください。',
      '領収書はマイページの「注文履歴」から注文ごとにPDFで発行できます。',
      '配送状況は、注文履歴の「配送状況を確認する」ボタンから確認できます。',
      '商品到着後8日以内で未使用の場合に限り、返品を受け付けています。',
      '退会すると保有しているポイントはすべて失効します。',
      'クレジットカードの情報は、マイページの「お支払い方法」から変更できます。',
    ],
    expectedTop: 1,
  },
  {
    id: 'anime-ja',
    label: 'アニメ制作の工程（日本語）',
    task: 'search result',
    query: '原画と原画のあいだの絵を描く担当',
    docs: [
      '原画：レイアウトをもとに、動きの起点と終点になる重要なポーズを描く。',
      '動画：原画の間を埋める中割りを描き、線をクリーンアップする。',
      '仕上げ：スキャンした動画に、色指定に従ってデジタルで彩色する。',
      '撮影：彩色したセルと背景を合成し、カメラワークや特殊効果を加える。',
      '作画監督：原画の絵柄を統一し、キャラクターの崩れを修正する。',
      '美術：美術設定やボードをもとに背景画を描く。',
    ],
    expectedTop: 1,
  },
  {
    id: 'cross-lingual',
    label: '日本語クエリ × 英語文書',
    task: 'search result',
    query: '猫に玉ねぎをあげても大丈夫？',
    docs: [
      "Onions, garlic, and chives can damage a cat's red blood cells and should never be fed to cats.",
      'Dogs should not eat chocolate because it contains theobromine.',
      'Cats are obligate carnivores and need taurine in their diet.',
      'Onion rings are a popular side dish made by deep-frying battered onion slices.',
      'Most adult cats are lactose intolerant, so milk can upset their stomachs.',
    ],
    expectedTop: 0,
  },
  // Weak spots found by the failure probes (scripts/failure-probes.mjs, docs/embeddinggemma-2-failures-*.md).
  // EmbeddingGemma 2 puts a wrong document first in every one of them, at fp32, q8 and q4. `alsoV1`
  // tells whether the first model misses it too.
  ...[
    {
      id: 'weak-negation',
      label: '苦手な例：否定「タバコが吸えない宿」',
      query: 'タバコが吸えない宿',
      docs: ['このホテルは全館禁煙で、喫煙所もありません。', 'このホテルには喫煙できる客室があります。', 'このホテルは駅から徒歩3分の場所にあります。', 'このホテルの大浴場は24時間利用できます。'],
      expectedTop: 0,
      alsoV1: false,
    },
    {
      id: 'weak-direction',
      label: '苦手な例：向き「大阪から東京へ」',
      query: '大阪から東京へ向かう新幹線',
      docs: ['東京発・新大阪行きの新幹線', '新大阪発・東京行きの新幹線', '東京発・金沢行きの新幹線'],
      expectedTop: 1,
      alsoV1: false,
    },
    {
      id: 'weak-role',
      label: '苦手な例：役割「お金を受け取ったのは誰」',
      query: '佐藤さんがお金を受け取った取引',
      docs: ['田中さんが佐藤さんに1万円を送金した。', '佐藤さんが田中さんに1万円を送金した。', '鈴木さんが高橋さんに3千円を送金した。'],
      expectedTop: 0,
      alsoV1: true,
    },
    {
      id: 'weak-time',
      label: '苦手な例：時刻の書き換え「午後3時」',
      query: '午後3時に始まる会議',
      docs: ['会議は15時に始まります。', '会議は13時に始まります。', '会議は10時に始まります。'],
      expectedTop: 0,
      alsoV1: true,
    },
    {
      id: 'weak-relative',
      label: '苦手な例：相対的な日付「明日出せるごみ」',
      query: '今日は水曜日。明日出せるごみは？',
      docs: ['燃えるごみの収集日は月曜日と木曜日です。', '資源ごみの収集日は水曜日です。', '粗大ごみの収集日は金曜日です。'],
      expectedTop: 0,
      alsoV1: true,
    },
    {
      id: 'weak-trap',
      label: '苦手な例：同じ言葉の罠「パソコンが立ち上がらない」',
      query: 'パソコンが立ち上がらない',
      docs: ['電源ボタンを押しても画面が真っ暗なままのときの対処法', 'パソコンの立ち上がりを速くする設定', 'プリンターが印刷できないときの対処法'],
      expectedTop: 0,
      alsoV1: true,
    },
    {
      id: 'weak-superlative',
      label: '苦手な例：比較「いちばん低い山」',
      query: 'この中でいちばん低い山',
      docs: ['北岳の標高は3193メートルです。', '富士山の標高は3776メートルです。', '奥穂高岳の標高は3190メートルです。'],
      expectedTop: 2,
      alsoV1: true,
    },
  ].map((preset) => ({ task: 'search result', ...preset })),
];

// Pairs that mean the same thing (across languages) should light up as blocks.
export const SIMILARITY_SENTENCES = [
  '今日はとても暑いですね。',
  '本日は気温がかなり高いです。',
  "It's really hot today.",
  'このラーメンはとても美味しい。',
  'This ramen is delicious.',
  '会議は午後3時から始まります。',
  'The meeting starts at 3 p.m.',
  '株価が大きく下落した。',
];
