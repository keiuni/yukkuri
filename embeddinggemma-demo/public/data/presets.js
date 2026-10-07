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
    // fp32 scores published in the onnx-community model card, for comparison.
    referenceScores: [0.30109718441963196, 0.6358831524848938, 0.4930494725704193, 0.48887503147125244],
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
