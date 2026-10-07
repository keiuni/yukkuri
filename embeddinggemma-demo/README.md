# EmbeddingGemma Playground

[google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) の ONNX 版
（[onnx-community/embeddinggemma-300m-ONNX](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX)）を
[Transformers.js](https://huggingface.co/docs/transformers.js) でブラウザ内実行し、性能を確かめるためのデモです。
推論はすべてブラウザ内で行われ、入力テキストは外部に送信されません。

Playwright で画面を実際に操作する E2E 検証も同梱しています。測定結果は
[docs/results-2026-10-07.md](docs/results-2026-10-07.md) にまとめました。

![検索タブ](docs/screenshots/search.png)

## できること

| タブ | 内容 |
| --- | --- |
| 検索 | クエリに近い文書を順位付け。タスク別プロンプト、MRL 次元（768/512/256/128）を切り替え可能。サンプルは想定解に「想定解」バッジが付くので外れがすぐ分かる |
| 類似度マトリクス | 複数文のコサイン類似度をヒートマップ表示。日英の言い換えがまとまるかを確認 |
| ミニベンチマーク | 日英の小さな検索データセット（文書63件・クエリ56件、紛らわしい文書入り）で Top-1 / Recall@5 / MRR@10 / nDCG@10 と速度を測定。プロンプト有無 × 次元の比較、言語ペア別、外したクエリの一覧、JSON 保存 |

モデルパネルでは精度（fp32 / q8 / q4）、実行環境（WASM / WebGPU）、WASM のスレッド数を選べます。

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
npm run download-model                 # q8 + q4（約 530 MB）
npm run download-model -- fp32 q8 q4   # 全部（約 1.8 GB）
```

`models/` に保存したファイルはサーバーが `/models/` で配信し、ブラウザはそちらを優先して読み込みます（無ければ Hub から取得）。
プロキシ環境では `NODE_USE_ENV_PROXY=1 npm run download-model` のように実行してください（Node 22.21 以降）。

### 精度と実行環境の選び方

| 精度 | サイズ | WASM（CPU）での傾向 |
| --- | ---: | --- |
| fp32 | 1.26 GB | 精度はモデルカード通り。WASM では一番速い。ダウンロードが重い |
| q8 | 331 MB | fp32 とほぼ同じ精度。単発クエリは fp32 の約2倍の時間 |
| q4 | 217 MB | 一番小さいが WASM では一番遅く、精度もわずかに落ちる |

- モデルカードの注意書きどおり、EmbeddingGemma は fp16 の活性値に対応していないため fp16 / q4f16 は選択肢から外しています。
- q4 は `model_no_gather_q4.onnx` を使います（通常の `model_q4.onnx` は WASM が未対応の `GatherBlockQuantized` を含むため）。
- WebGPU は対応ブラウザでのみ選択できます。この環境（GPU なしのヘッドレス Chromium）では未検証です。

### URL パラメータ

| パラメータ | 例 | 意味 |
| --- | --- | --- |
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
npm run report                                  # test-output/REPORT.md を生成
```

精度ごとに 1 つのページでモデルを読み込み、ユーザーと同じ操作で次を確認します。

1. 初期表示（モデル未読み込みでボタンが無効、クロスオリジン分離が有効、タブのキーボード操作）とスマホ幅で横スクロールが出ないこと
2. UI からモデルを読み込めること
3. モデルカードの例を再現できること（順位が一致し、スコア差が fp32 0.005 / q8 0.03 / q4 0.05 未満）
4. 日本語 FAQ と日→英の検索で想定解が 1 位になること（アニメ制作の例は全精度で 3 位になる既知の外れのため、順位を記録のみ）
5. 768 / 512 / 256 / 128 次元のどれでも FAQ の想定解が 1 位で、2 回目以降は埋め込みキャッシュが再利用されること
6. 類似度マトリクスで「同じ意味の組の最小値 > 違う意味の組の最大値」になること、ツールチップが出ること
7. ミニベンチマークが完走し（Top-1 > 0.6、MRR@10 > 0.7 の下限チェック）、JSON を保存できること

スクリーンショットと数値は `test-output/`（git 管理外）に出力されます。

## 構成

```
server.mjs                  静的サーバー（COOP/COEP、/models/ の配信、パストラバーサル防止）
public/index.html           画面
public/app.js               UI ロジック（検索・類似度・ベンチマーク）
public/embedder.worker.js   Web Worker 内で Transformers.js を実行
public/lib/model-config.js  モデル ID・リビジョン固定・プロンプト・精度の定義
public/lib/metrics.js       MRL 切り詰め、コサイン類似度、Top-1 / MRR / nDCG
public/data/benchmark.json  ミニベンチマーク用データセット（手作り）
public/data/presets.js      検索・類似度のサンプル
scripts/download-model.mjs  モデルのダウンロード
scripts/make-report.mjs     E2E 結果の集計
tests/unit, tests/e2e       単体テスト、Playwright テスト
```

## ライセンスについて

モデルの重みは [Gemma Terms of Use](https://ai.google.dev/gemma/terms) に従います。
ミニベンチマークのデータセットはこのデモ用に書いたもので、MTEB / JMTEB の代わりにはなりません。
