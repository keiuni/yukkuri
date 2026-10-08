# EmbeddingGemma Playground

[google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)（初代）と
[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) の ONNX 版
（[onnx-community/embeddinggemma-300m-ONNX](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX)、
[onnx-community/embeddinggemma-2-ONNX](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX) のテキストモデル）を
[Transformers.js](https://huggingface.co/docs/transformers.js) でブラウザ内実行し、性能を確かめるためのデモです。
推論はすべてブラウザ内で行われ、入力テキストは外部に送信されません。

Playwright で画面を実際に操作する E2E 検証も同梱しています。測定結果は
[docs/results-2026-10-07.md](docs/results-2026-10-07.md)（初代）と
[docs/embeddinggemma-2-2026-10-08.md](docs/embeddinggemma-2-2026-10-08.md)（2 との比較）にまとめました。

jev-ultrafast の Jev（次の操作を選ぶ判断役）を EmbeddingGemma に置き換えたブラウザエージェントも入っています
（下の「Gemma ブラウザエージェント」）。Liquid AI の判断モデル d1 と LFM2.5-2.6B も同じエージェントで比べました
（[docs/liquid-ai-2026-10-08.md](docs/liquid-ai-2026-10-08.md)）。スマホのブラウザで Gemma 系のモデルが動くかを調べるページと、
EmbeddingGemma に 1 文字ずつ文字を選ばせる実験もあります（[docs/phone-and-chargen-2026-10-08.md](docs/phone-and-chargen-2026-10-08.md)）。

![検索タブ](docs/screenshots/search.png)

## できること

| タブ | 内容 |
| --- | --- |
| 検索 | クエリに近い文書を順位付け。タスク別プロンプト、MRL 次元（768/512/256/128）を切り替え可能。サンプルは想定解に「想定解」バッジが付くので外れがすぐ分かる |
| 類似度マトリクス | 複数文のコサイン類似度をヒートマップ表示。日英の言い換えがまとまるかを確認 |
| ミニベンチマーク | 日英の小さな検索データセット（文書63件・クエリ56件、紛らわしい文書入り）で Top-1 / Recall@5 / MRR@10 / nDCG@10 と速度を測定。プロンプト有無 × 次元の比較、言語ペア別、外したクエリの一覧、JSON 保存 |
| 文字生成 | 文字列に 1 文字ずつ足しては埋め込み、目標（文や質問の埋め込み）にいちばん近づく文字を選び続ける。各ステップの上位候補と類似度を表示 |

モデルパネルでは、モデル（初代 / 2）、精度（fp32 / q8 / q4）、実行環境（WASM / WebGPU）、WASM のスレッド数を選べます。

## 使い方

Node.js 20 以上が必要です。デモ本体は依存パッケージなしで動きます（Transformers.js は jsDelivr から読み込みます）。

```bash
cd embeddinggemma-demo
npm start            # http://127.0.0.1:5173/
```

ブラウザで開いて「モデルを読み込む」を押すと、Hugging Face Hub からモデルを取得します（初回のみ。以降はブラウザのキャッシュを使います）。

`npm start` のサーバーは COOP/COEP ヘッダーを付けて配信するため、WASM がマルチスレッドで動きます。
`python -m http.server` などで `public/` を配信しても動きますが、その場合は 1 スレッドになり遅くなります。

### モデルを手元に置く（任意）

```bash
npm run download-model                              # 初代の q8 + q4（約 530 MB）
npm run download-model -- fp32 q8 q4                # 初代の全部（約 1.8 GB）
npm run download-model -- --model v2 fp32 q8 q4     # 2 のテキストモデル（約 1.6 GB）
```

`models/` に保存したファイルはサーバーが `/models/` で配信し、ブラウザはそちらを優先して読み込みます（無ければ Hub から取得）。
プロキシ環境では `NODE_USE_ENV_PROXY=1 npm run download-model` のように実行してください（Node 22.21 以降）。

### モデル・精度・実行環境の選び方

| モデル・精度 | サイズ | 実行環境 | 傾向 |
| --- | ---: | --- | --- |
| 初代 fp32 | 1.26 GB | WASM / WebGPU | 精度はモデルカード通り。WASM では一番速い。ダウンロードが重い |
| 初代 q8 | 331 MB | WASM / WebGPU | fp32 とほぼ同じ精度。単発クエリは fp32 の約 2 倍の時間 |
| 初代 q4 | 217 MB | WASM / WebGPU | 一番小さいが WASM では一番遅く、精度もわずかに落ちる |
| 2 fp32 | 1.12 GB | WASM / WebGPU | ミニベンチマークの精度は初代より少し高い。WASM で動く 2 はこれだけで、初代 fp32 の 1.2〜1.3 倍の時間 |
| 2 q8 | 346 MB | WebGPU のみ | 精度は 2 fp32 と同じ（ミニベンチマーク） |
| 2 q4 | 207 MB | WebGPU のみ | 精度は 2 fp32 と同じ（ミニベンチマーク）。一番軽く、スマホ向き |

- どちらのモデルカードも、活性値が fp16 の範囲を超えると注意しているので、fp16 / q4f16 は選択肢から外しています。
- 初代の q4 は `model_no_gather_q4.onnx` を使います（通常の `model_q4.onnx` は WASM が未対応の `GatherBlockQuantized` を含むため）。
  2 には no_gather 版がなく、q8・q4 とも `GatherBlockQuantized` を使うので、WASM では選べません。
- 2 は画像・音声のエンコーダーを外して、テキストモデルだけを読み込みます（`model-config.js` の `textOnlyConfig`）。
- 2 は類似度が全体に高め（無関係な文どうしでも 0.7 前後）に出ます。順位は問題ありませんが、しきい値は初代と別に決める必要があります。
- WebGPU は対応ブラウザでのみ選択できます。この環境（GPU なしのヘッドレス Chromium）ではソフトウェア実装の WebGPU で動作だけ確認しました。

### URL パラメータ

| パラメータ | 例 | 意味 |
| --- | --- | --- |
| `model` | `?model=v2` | モデルの初期値（`v1` = 初代、`v2` = EmbeddingGemma 2） |
| `dtype` | `?dtype=fp32` | 精度の初期値 |
| `device` | `?device=webgpu` | 実行環境の初期値 |
| `threads` | `?threads=4` | WASM スレッド数の初期値 |
| `autoload` | `?autoload=1` | 開いたらすぐ読み込む |
| `cache` | `?cache=0` | Cache Storage にモデルを保存しない |
| `local` | `?local=0` | `/models/` を見ずに Hub から取得する |

## E2E 検証（Playwright）

```bash
npm install                      # @playwright/test のみ
npx playwright install chromium  # 初回のみ
npm run download-model -- fp32 q8 q4

npm test                                        # 評価指標の単体テスト（node:test）
DTYPES=fp32,q8,q4 THREADS=4 npm run test:e2e    # 既定は DTYPES=q8, THREADS=auto
MODEL=v2 DTYPES=fp32 THREADS=4 npm run test:e2e # EmbeddingGemma 2（WASM では fp32 のみ）
MODEL=v2 DEVICE=webgpu DTYPES=q8,q4 npm run test:e2e   # 2 の q8・q4（WebGPU）
npm run report                                  # test-output/REPORT.md を生成（全モデル・全精度をまとめる）
npm run benchmark -- --model v2 --dtype q8,q4   # ミニベンチマークの精度だけを Node（CPU）で測る
```

精度ごとに 1 つのページでモデルを読み込み、ユーザーと同じ操作で次を確認します。

1. 初期表示（モデル未読み込みでボタンが無効、クロスオリジン分離が有効、タブのキーボード操作）とスマホ幅で横スクロールが出ないこと
2. UI からモデルを読み込めること
3. モデルカードの例を再現できること（順位が一致し、スコア差が初代は fp32 0.005 / q8 0.03 / q4 0.05 未満。2 はモデルカードが q4 の値なので q4 0.005 / fp32・q8 0.02 未満）
4. 日本語 FAQ と日→英の検索で想定解が 1 位になること（アニメ制作の例は全精度で 3 位になる既知の外れのため、順位を記録のみ）
5. 768 / 512 / 256 / 128 次元のどれでも FAQ の想定解が 1 位で、2 回目以降は埋め込みキャッシュが再利用されること
6. 類似度マトリクスで「同じ意味の組の最小値 > 違う意味の組の最大値」になること、ツールチップが出ること
7. ミニベンチマークが完走し（Top-1 > 0.6、MRR@10 > 0.7 の下限チェック）、JSON を保存できること
8. 文字生成タブで、文の復元を 12 文字まで生成できること

スクリーンショットと数値は `test-output/`（git 管理外）に出力されます。
ソフトウェア実装の WebGPU では 1 文に数秒かかるので、WebGPU の実行では `-- --grep-invert "mini benchmark|one character"` で 7・8 を外し、
精度は `npm run benchmark` で測ってください（Node 版はブラウザ版と同じ評価コードで、ブラウザで測れる組み合わせでは値が一致します）。

## 操作動画の録画

E2E とは別に、人が見る用の操作動画（字幕・カーソル・クリック表示つき、1280×720 の MP4）を録画できます。
モデルの読み込み待ちはカットするか早送り（4 倍速以上、1 回あたり最大 5 秒）にします。MP4 への変換には libx264 入りの ffmpeg が必要です（無ければ WebM のまま残します）。

```bash
npm run download-model -- fp32 q8 q4
npm run record                            # 全シナリオ → test-output/videos/*.mp4
npm run record -- similarity benchmark    # 指定したシナリオだけ
```

| シナリオ | 内容 |
| --- | --- |
| `load-and-search` | モデルを UI から読み込み、モデルカードの例を検索。768 → 128 次元に減らして再検索 |
| `japanese-search` | FAQ に言い回しを変えた質問を入力（当たる例と外れる例） |
| `cross-lingual-and-miss` | 日本語クエリで英語文書を検索。アニメ制作用語で外れる例 |
| `similarity` | 類似度マトリクスを計算し、セルをホバー。英文を 1 行足して再計算 |
| `precision` | q8 → fp32 → q4 と読み込み直し、モデルカードの値との差を比べる |
| `benchmark` | ミニベンチマークを実行し、結果を順に見る |
| `chargen` | 文字生成タブで、文の復元と質問への「回答」を 1 文字ずつ生成する（fp32） |
| `phone-check` | スマホ実行チェックを Pixel 7 の画面で開き、EmbeddingGemma と Gemma 3 270M を試す（要 `npm run download-phone-models`） |
| `v2-compare` | 同じ検索を初代（q8）と EmbeddingGemma 2（fp32）で実行して比べ、2 でミニベンチマークを回す |

字幕の数値はその場の結果から作るので、結果が変わっても字幕と画面が食い違いません。録画中は CPU を録画にも使うため、速度の数値は参考程度にしてください。
Playwright の録画は、画面の描き替えが多い場面（プログレスバーなど）で実時間より長くなります。そのため、カットや早送りの位置は
画面の左上に映し込んだ目印（色が変わる 8px の四角。最終の動画では周りの画素で塗りつぶします）を動画の中で探して合わせています。

## スマホ実行チェック

`phone.html` は、開いた端末のブラウザで次のモデルが動くかを調べるページです。

- EmbeddingGemma 300M（q4 / q8）
- EmbeddingGemma 2 のテキストモデル（q4 / q8 は WebGPU、fp32 は WASM）
- LFM2.5-2.6B（Liquid AI。q4f16 / q4、WebGPU のみ）
- Gemma 3 270M（q4f16 / q4 / fp32）
- Gemma 3 1B（q4f16 / q4）
- Gemma 4 E2B（q4f16 / q4）

ページは WebGPU の有無、f16 対応、GPU の 1 バッファの上限、メモリを調べて、モデルごとに「動きそう／微妙／動かない見込み」を出します。
「試す」を押すとモデルを読み込み、読み込み時間・最初の 1 トークンまでの時間・生成速度を測ります。
Google の LiteRT-LM Web 版の Gemma 4 E2B は、公式デモへのリンクで試せます。

```bash
npm run download-phone-models     # 計測用のモデルを ./models に置く（WebGPU の f16 版以外）
npm run phone-check               # Pixel 7 の画面 + ソフトウェア WebGPU（f16 なし・1 GiB 上限）で全モデルを試す
npm run phone-check -- gemma-3-270m:fp32:wasm --tokens 64 --repeat 2
```

このクラウド環境（CPU 4 コア・GPU なし）での結果は次のとおりです。

| モデル | 実行 | 結果 |
| --- | --- | --- |
| EmbeddingGemma q8 | WASM | ✓ 1 文 0.24 秒、タブのメモリ 2.0 GB |
| EmbeddingGemma 2 fp32 | WASM | ✓ 1 文 0.33 秒、タブのメモリ 2.8 GB（WebGPU がない端末での唯一の選択肢） |
| EmbeddingGemma 2 q4 / q8 | WebGPU | ✓ 読み込みと検索ができる（175 MB / 314 MB、最大の重み 67 MB / 134 MB） |
| Gemma 3 270M fp32 | WASM | ✓ 10 トークン/秒、タブのメモリ 4.1 GB |
| Gemma 3 270M / 1B q4 | WebGPU | ✓ 読み込みと生成ができる（ソフト GPU なので速度は参考外） |
| Gemma 3 270M q4 | WASM | ✗ 埋め込み表の GatherBlockQuantized が WASM 版 onnxruntime にない（q8 版も同じ演算を使う） |
| Gemma 4 E2B q4 | WebGPU | ✗ 1,174 MB の重みが 1 GiB のバッファ上限に入らず、最初の推論で失敗 |

スマホ実機で試すには、HTTPS で配信する必要があります（WebGPU は HTTPS か localhost でしか使えません）。
このリポジトリは公開されているので、GitHub Pages（Settings → Pages、Branch: `claude/wizardly-clarke-gcgyaw`、`/ (root)`）を有効にすると、
`https://keiuni.github.io/yukkuri/embeddinggemma-demo/public/phone.html` で開けます。
詳しい結果と考察は [docs/phone-and-chargen-2026-10-08.md](docs/phone-and-chargen-2026-10-08.md) にあります。

## 文字生成（EmbeddingGemma を 1 文字ずつ回す）

EmbeddingGemma は文章を書くモデルではありません。そこで、いまの文字列の後ろに候補の文字を 1 つずつ足して埋め込み、
目標の埋め込みにいちばん近づいた文字を残す、を繰り返します（貪欲法、またはビームサーチ）。目標は次の 2 通りです。

- 文の復元: 文の埋め込みを目標にして、元の文を書き戻せるかを見る
- 質問への回答: 質問の埋め込みを目標にして、モデルがどんな文書を「答え」とみなすかを見る

候補はかな・句読点・数字と、目標に近い漢字（1 文字だけの埋め込みで選んだ上位）です。
漢字は日本語版ウィキペディアの一般的な記事でよく使われる 2,000 字です（`scripts/build-char-vocab.mjs`）。

```bash
npm run char-generate                       # 7 つの実験を貪欲法とビーム幅 4 で → test-output/char-gen/results.json
npm run char-generate -- query-mountain --beams 1
npm run char-generate -- --model v2         # EmbeddingGemma 2 → test-output/char-gen/results-v2.json
```

結果の一部です（fp32・ビーム幅 4）。文章にはならず、意味の濃い漢字を並べた文字列になります。

| 目標 | 生成された文字列 | 類似度 |
| --- | --- | ---: |
| 文「東京タワーは東京都港区にある電波塔です。」 | 塔は都電港区都塔で、都の塔・、 | 0.893 |
| 文「猫に玉ねぎを食べさせてはいけない。」 | 猫餌禁菜・猫、オめぼをと禁きつの猫やのうき | 0.882 |
| 質問「猫に食べさせてはいけないものは？」 | 猫食忌だ餌。菜菓毒禁味ゾ。禁ンよりぐ。 | 0.690（正しい答えの文は 0.620） |

単語の途中（「タ」「タワ」）では類似度が上がらないので、「タワー」のような語は組み立てられません。質問に対しては答えではなく話題の字を並べ、
意味の通らない文字列が正しい答えの文より高い類似度になることもあります。詳しくは [docs/phone-and-chargen-2026-10-08.md](docs/phone-and-chargen-2026-10-08.md) にあります。

## Gemma ブラウザエージェント（Jev の代わりに Gemma）

[jev-ultrafast](https://github.com/browser-use/jev-ultrafast)（Browser Use）と同じ流れのブラウザエージェントです。
TypeSafe の Jev が担う「次に操作する要素と、操作の種類を選ぶ」役を EmbeddingGemma に置き換えています。

```
ページ ─▶ 番号付きの要素表（agent/browser.js の snapshot）
      ─▶ 候補: CLICK / TYPE_TEXT / SELECT / CHECK / DONE（agent/actions.mjs）
      ─▶ EmbeddingGemma が「手順の文」と各候補の類似度を確率にして 1 つ選ぶ（Jev の役）
      ─▶ Playwright が実行 ─▶ 最後にタスクごとの検証で結果を確かめる
```

| 役割 | jev-ultrafast | このデモ |
| --- | --- | --- |
| 次の操作と対象を選ぶ | Jev（TypeSafe） | EmbeddingGemma 300M（Node + onnxruntime、CPU） |
| 手順（サブゴール） | Jev がゴールから直接判断。jev-mcp では Claude Code などが指示 | `--planner script`（既定）: タスクに書いた手順（Claude が作成）<br>`--planner gemma4`: Gemma 4 E2B がページごとに作成 |
| 入力する文字列 | 小型 LLM が書く | 手順中の「」の文字列（`gemma4` では Gemma 4 が書く） |
| 完了（DONE） | Jev の DONE ＋ 独立した検証 | タスクごとの検証（`gemma4` では Gemma 4 に「はい/いいえ」を聞く） |
| ブラウザ操作 | Browser Harness（CDP） | Playwright |

画面の右側に、計画・各ステップの候補と確率・判断時間を出すインスペクターが付きます。

```bash
npm install                        # .npmrc で onnxruntime-node の CUDA 用ダウンロードを省略
npm run download-model -- fp32     # EmbeddingGemma（fp32）
npm run agent                      # 6 タスクを実行（手順書 + EmbeddingGemma）
npm run agent -- --video           # test-output/agent/videos/*.mp4 も録画
npm run agent -- --embedding v2    # 判断役を EmbeddingGemma 2 にする（要 npm run download-model -- --model v2 fp32）
npm run agent -- --decider gemma4  # 判断役を Gemma 4 E2B にして比較（初回に約 3.6 GB をダウンロード）
npm run agent -- --planner gemma4  # 手順も Gemma 4 E2B が作る（完全ローカル）
```

Liquid AI のモデルは llama.cpp の `llama-server` で動かします（`/v1/systemone` のある 2026-10 以降の版。ビルド手順は
`scripts/download-liquid-models.mjs` の先頭にあります）。

```bash
export LLAMA_SERVER=/path/to/llama.cpp/build/bin/llama-server
npm run download-liquid-models                 # d1-3B・d1-omni-600M・LFM2.5-2.6B の GGUF（約 5 GB）
npm run agent -- --decider d1-3b               # 判断役を d1-3B に（d1-omni、lfm も可。--d1-lang en で指示文を英語に）
npm run agent -- --planner lfm                 # LFM2.5-2.6B が手順を JSON で書く
npm run agent -- --planner lfm-tools           # LFM2.5-2.6B が道具呼び出しで要素を直接操作する
```

タスクは、宿泊予約（検索 → 絞り込み → 並べ替え）、通販（先月の注文の領収書、商品検索 → 並べ替え）、
アカウント設定（スイッチとプルダウンを変えて保存、「アカウントを削除」ボタンあり）をモックサイト（`public/sites/`）で 5 つ、
実サイトの日本語版ウィキペディアで 1 つです。手順書はあえて画面のラベルと違う言い方で書いています
（例: 「画面を暗い配色にする」→ テーマ「ダーク」、「大人の人数を2人にする」→ 大人「2名」）。

### 結果（4 コア CPU のクラウド環境）

| 手順を書く役 | 判断役（Jev の役） | 成功 | 1 回の判断（中央値 / 最大） | 計画・完了判定（Gemma 4 E2B） |
| --- | --- | --- | --- | --- |
| 手順書（Claude が作成） | **EmbeddingGemma 300M** | **6/6** | 0.1 秒 / 3.1 秒 | — |
| 手順書（Claude が作成） | EmbeddingGemma 2 | 6/6（21 手すべて初代と同じ選択） | 0.13 秒 / 4.6 秒 | — |
| 手順書（Claude が作成） | d1-3B（Liquid AI の判断モデル） | 5/6 | 12 秒 / 66 秒 | — |
| 手順書（Claude が作成） | Gemma 4 E2B | 3/6 | 7.4 秒 / 59 秒 | — |
| 手順書（Claude が作成） | LFM2.5-2.6B | 2/6 | 78 秒 / 229 秒 | — |
| 手順書（Claude が作成） | d1-omni-600M | 1/6 | 2.7 秒 / 28 秒 | — |
| Gemma 4 E2B | EmbeddingGemma 300M | 3/6 | 0.15 秒 / 2.9 秒 | 計画 1 回 16〜44 秒（中央値 24 秒）<br>完了判定 1 回 5〜16 秒（中央値 9 秒） |
| LFM2.5-2.6B（JSON で手順） | EmbeddingGemma 300M | 2/6 | — | 計画 1 回 31〜244 秒 |
| LFM2.5-2.6B（道具呼び出しで直接操作） | （同じモデル） | 2/6 | — | 1 回 23〜193 秒 |

Liquid AI のモデルは llama.cpp（CPU）、EmbeddingGemma と Gemma 4 E2B は onnxruntime（CPU）で動かしています。

- 判断の最大値は、要素が約 150 個あるウィキペディアのページを初めて見たときです（候補をすべて埋め込むため。同じ文はキャッシュします）。
- 判断役を Gemma 4 E2B にした行では、EmbeddingGemma と同じ情報（手順と候補の表）だけを渡しています。
  「検索を実行する」で検索ボタンではなく入力欄を選ぶ（宿泊予約・ウィキペディア）、
  「2026年9月の注文の領収書を出す」で先頭の 10 月の注文を選ぶ、の 3 つで失敗しました。
  目標の文も一緒に渡すと 0/6 でした（検索や保存の手順でも、目標に出てくる入力欄やスイッチを選び続けた）。
- Gemma 4 E2B が手順も作る行では、失敗はどれも計画か完了判定によるものでした。
  宿泊予約は最初のページの計画で「1泊」を落とし、領収書は 9 月の領収書ページまで行けたのに「達成していない」と判断して注文履歴へ戻り、そこで「達成した」と判断し、商品検索は並べ替える前に「達成した」と判断しました。
  EmbeddingGemma は、Gemma 4 E2B が書いた手順どおりの要素を毎回選んでいます（間違った手順にもそのまま従います）。
- 手順書は、jev-mcp で Claude Code などが Jev に渡すサブゴールに相当します。目標から手順を作る部分（計画）と、
  終わったかどうかの判断は、2B クラスの生成モデルには荷が重いという結果です。

### 調整の経緯

最初は Gemma 4 E2B に目標から手順をまとめて作らせる構成で 2/6 でした。そこから次の設計変更をしています（特定のタスク向けの調整はしていません）。

- 画面外の要素も候補に入れる（「変更を保存」が画面外にあり、別のボタンを押していた）
- 値を持つ手順は入力・選択・切り替えだけを候補にする（Jev の操作ごとの head と同じ考え方）
- 一度設定した要素は二度設定しない
- Gemma 4 にはページが変わるたびに、いまのページの要素とこれまでの操作を見せて計画し直させ、完了は別の「はい/いいえ」質問にする
- 「先月」を正しく月に直せなかったため、プロンプトに今日・今月・先月・来月を明記する
- 埋め込みを文字数順にまとめて計算する（パディングの無駄が減り、判断が約 5 倍速くなった）

## 構成

```
server.mjs                  静的サーバー（COOP/COEP、/models/ の配信、パストラバーサル防止）
public/index.html           画面
public/app.js               UI ロジック（検索・類似度・ベンチマーク）
public/embedder.worker.js   Web Worker 内で Transformers.js を実行
public/lib/model-config.js  初代と 2 のモデル ID・リビジョン固定・精度、プロンプト、2 をテキストだけで読む設定
public/lib/metrics.js       MRL 切り詰め、コサイン類似度、Top-1 / MRR / nDCG
public/data/benchmark.json  ミニベンチマーク用データセット（手作り）
public/data/presets.js      検索・類似度のサンプル
scripts/download-model.mjs  モデルのダウンロード
scripts/make-report.mjs     E2E 結果の集計
scripts/benchmark.mjs       ミニベンチマークの精度を Node（CPU）で測る
scripts/record-demo.mjs     字幕つき操作動画の録画
scripts/lib/video.mjs       録画の後処理（カット・早送り・MP4 変換）とサーバー起動
agent/run.mjs               Gemma ブラウザエージェント（ループ・インスペクター・録画）
agent/browser.js            ページ内で動く要素表の作成とインスペクター
agent/actions.mjs           候補の作成、値の決め方、Playwright での実行
agent/gemma.mjs             EmbeddingGemma と Gemma 4 E2B（計画・完了判定・比較用の判断役）
agent/llama.mjs             llama-server で動かす Liquid AI のモデル（d1 の /v1/systemone、LFM2.5 のチャット）
agent/tools.mjs             道具呼び出しで要素を直接操作するエージェント（LFM2.5-2.6B 用）
agent/tasks.mjs             6 つのタスクと、それぞれの結果の検証
public/sites/               エージェントが操作するモックサイト（宿泊予約・通販・アカウント設定）
public/phone.html, phone.js スマホ実行チェック（端末の WebGPU・メモリの確認と、モデルごとの試行）
public/phone.worker.js      スマホ実行チェックの 1 回の試行（試行ごとに新しいワーカー）
public/lib/phone-models.js  候補モデルのサイズ・最大の重み・リビジョンと、動くかどうかの判定
public/chargen-tab.js       文字生成タブ
public/lib/char-gen.js      1 文字ずつの生成（候補の選び方、貪欲法・ビームサーチ、評価）
public/data/char-vocab.json 文字生成の候補文字（かな・記号と、よく使う漢字 2,000 字）
public/data/char-vectors.json 候補文字 1 文字ずつの埋め込み（int8、漢字の絞り込み用。2 用は char-vectors-v2.json）
scripts/phone-check.mjs     スマホ相当の条件で phone.html を動かし、結果とメモリを記録
scripts/download-phone-models.mjs  スマホ実行チェック用のモデルのダウンロード
scripts/download-liquid-models.mjs  Liquid AI のモデル（GGUF）のダウンロード
scripts/char-generate.mjs   文字生成の実験
scripts/build-char-vocab.mjs, export-char-vectors.mjs  候補文字とその埋め込みの作成
scripts/lib/hub.mjs         Hugging Face Hub からのダウンロード
scripts/lib/char-vectors.mjs  候補文字の埋め込みの計算とキャッシュ
tests/unit, tests/e2e       単体テスト、Playwright テスト
```

## ライセンスについて

初代の重みは [Gemma Terms of Use](https://ai.google.dev/gemma/terms)、EmbeddingGemma 2 の重みは Apache 2.0 に従います。
Liquid AI のモデル（d1、LFM2.5）は LFM Open License v1.0 です（年間売上 1,000 万ドル以上の組織の商用利用は対象外）。
ミニベンチマークのデータセットはこのデモ用に書いたもので、MTEB / JMTEB の代わりにはなりません。
