# 分類ベンチマーク環境（実 Jev ⇔ ローカル JevBERT の比較）

| 項目 | 内容 |
| --- | --- |
| 文書バージョン | 0.2.0（2026-09-22。レビュー指摘を反映） |
| 対象 | `bench/` パッケージ、`bench/experiments/*.yaml` |
| 段階 | **環境構築のみ**。実 Jev に対する実験は一度も実行していない（課金が発生するため。§6） |

本書は、公開の自然言語分類データセットを使って **実 Jev の API** と **このリポジトリのローカル Jev 互換 API** を
同じ条件で叩き、結果を比べるための環境の設計書である。どこまで同じ条件にでき、どこからは同じにできないか
（留保）を §4・§5 に書く。

---

## 1. 何ができるか

```bash
uv sync                                                                   # bench グループ（pyarrow）も既定で入る
uv run python -m bench prepare bench/experiments/jev-vs-local.yaml        # データ取得と標本抽出（1 回だけ）
uv run python -m bench plan    bench/experiments/jev-vs-local.yaml        # 送る件数・1 件目の要求本文を表示（送信しない）
uv run python -m bench run     bench/experiments/jev-vs-local.yaml --target local
uv run python -m bench run     bench/experiments/jev-vs-local.yaml --target jev --allow-billable
uv run python -m bench report  bench/experiments/jev-vs-local.yaml        # 比較表を report.md / report.json に書く
```

実験条件は **YAML（または JSON / TOML。拡張子で判定）1 ファイル**に書く。コードを変えずに、データセット・
ラベル集合・標本数・指示文・接続先・同時実行数・retry・timeout を変えられる。本番の設定は
[`bench/experiments/jev-vs-local.yaml`](../bench/experiments/jev-vs-local.yaml)（各項目にコメントあり）。
設定ファイルの綴り間違い（`max_retry` など）は既定値に黙って戻さず、読み込み時にエラーにする。

動作環境は Windows ネイティブ（本リポジトリの `.venv`）で確認した。WSL2 固有の設定は無い。

- 途中で止めても `run` を再実行すれば**決着済みの標本は飛ばして続きから**送る（結果は追記型 JSONL）。
  再送するのは「送り直せば直り得る失敗」（接続失敗・408・429・5xx）だけ。422 や choice の無い応答のように
  **相手が答えた失敗**は、課金済みの可能性があり同じ答えが返るだけなので再送しない（`--retry-failed` で再送する）。
- Ctrl+C で止めると、新しい標本は取らず、送信中の要求だけ終えて記録してから終了する（終了コード 130）。
- `--limit N`（1 以上）で各タスクの先頭 N 件だけ送れる（標本の順序は固定なので、両ターゲットで同じ N 件になる）。
- 課金対象のターゲットは `--allow-billable` が無いと**送信せずに件数（retry 込みの最大試行回数も）を表示して終了**する。
  `billable` を書き忘れても安全側に倒れるよう、**`127.0.0.1`・`localhost`・`::1` 以外のホストは既定で課金対象**とする
  （明示的に `billable: false` と書いたときだけ外れる）。ローカル以外への `http://` は API key が平文で流れるため読み込み時に拒否する。

終了コード: 0 完了 / 1 失敗した標本がある / 2 設定・標本・key の不備 / 3 課金対象への送信を拒否 / 130 中断。

## 2. データセット

3 種類。いずれも HF Hub の **commit SHA を固定**して取得する（同じ設定ファイルなら誰がいつ `prepare` しても同じ標本）。

| タスク ID | データセット | 言語 | 入力（state） | ラベル | 既定の標本数 |
| --- | --- | --- | --- | --- | --- |
| `arxiv-primary` | [`librarian-bots/arxiv-metadata-snapshot`](https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot)（Cornell の arXiv メタデータの週次ミラー、CC0。固定 revision は 2026-09-21 同期分） | 英 | `{"title", "abstract"}` | **primary category**（`cs.CL` など）16 種 | 各 15 件 = 240 |
| `ag-news` | [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) test split | 英 | `{"text"}` | World / Sports / Business / Sci/Tech の 4 種 | 各 50 件 = 200 |
| `livedoor` | [`llm-book/livedoor-news-corpus`](https://huggingface.co/datasets/llm-book/livedoor-news-corpus) test split（HF の parquet 変換版） | 日 | `{"title", "body"}` | 9 媒体（`sports-watch` など） | 各 20 件 = 180 |

### 2.1 arXiv タスクの作り方

- **「最近の論文」に限定**する: 新形式 ID の年月（`YYMM.NNNNN` の `YYMM`）が **2606 以降**（2026 年 6 月〜
  固定 snapshot の 2026-09-21 まで）の論文だけを使う。該当は 113,723 件、そのうち primary が 16 ラベルの
  いずれかであるものが 55,790 件（2026-09-22 に固定 revision で実測）。
  学習データに入っている可能性（contamination）を下げるための選択である（§5 C1）。
- 正解は `categories` の**先頭**のカテゴリ（arXiv の慣例では先頭が primary）。
- 16 ラベルは、該当期間で件数が多いものから、紛らわしい組（`cs.LG`/`stat.ML`/`cs.AI`、`cs.CV`/`eess.SP`）と
  離れた分野（`quant-ph`・`astro-ph.GA`・`cond-mat.mtrl-sci`）が両方入るように選んだ。
  **正解が 16 ラベル外の論文は標本に入れない**（閉集合の分類。§5 C6）。
- ラベルの説明文（Choice の `criteria`）はカテゴリ名と対象範囲を 1 行で書いた英文。
- 補助指標として「予測が cross-list を含むいずれかのカテゴリに当たった率」（`any_category_hit`）も出す。

### 2.2 livedoor の前処理

タイトルの `【Sports Watch】`・`【デジ通】` のような**コーナー名は媒体をほぼ言い当てる**（test split 738 件中
302 件のタイトルに `【…】` がある）。既定設定では `【…】` を丸ごと除去してから送る
（`remove_patterns`。設定で外せる）。本文は先頭 500 文字で切る。

### 2.3 標本抽出

- ラベルごとに、`sha256(seed + ":" + 標本ID)` の昇順で先頭 N 件を取る（層化抽出）。行の並び順・ファイル分割に依存しない。
- 並び順は「各ラベルの 1 番目 → 各ラベルの 2 番目 → …」の巡回にする。`--limit N` で先頭だけ送っても偏らない。
- 抽出結果は `bench_runs/<実験名>/samples/<タスク>.jsonl` に **送信する state そのもの**として保存し、
  取得元（repo・revision・ファイル）と抽出条件、ファイルの SHA-256 を `…manifest.json` に記録する。
- 送信側はこのファイルだけを読む。**データセットの取得と送信は分離**してあり、両ターゲットは同じファイルから送る。
- 標本ファイルが manifest の SHA-256 と合わない（後から編集された）場合、または設定ファイルの取得元・state・
  ラベル集合・標本数が manifest 作成時と変わった場合は、`run` が送信を拒否して `prepare` のやり直しを求める。
  ラベルの**説明文**と指示文は標本に影響しないので、変えても `prepare` は要らない。
- 結果の各行には、**標本ファイルの SHA-256** と **質問条件の SHA-256**（model・質問 ID・指示文・候補とその説明。
  state 以外の本文すべて）を記録する。今の設定と合わない行は `run` の再開判定でも `report` の集計でも**使わない**
  （`report` には「stale record(s) ignored」として件数を出す）。指示文を変えて再開しても、古い文言の結果が混ざらない。

### 2.4 データに残っている癖（手を加えていない）

- AG News の本文には HTML 実体参照の残骸（`quot;`・`#36;` など）が元データの時点で含まれる。除去せずそのまま送る。
- livedoor はタイトルの `【…】` 以外は加工していない。記事末尾の関連リンク文言などもそのまま（先頭 500 字に入れば送る）。

## 3. 1 件の要求の形

1 標本 = 1 リクエスト = 1 つの Choice 質問。

```json
{
  "state": {"title": "…", "abstract": "…"},
  "model": "jev-latest",
  "questions": {
    "label": {
      "type": "choice",
      "instructions": "Which arXiv category is the primary category of this paper?",
      "criteria": {"cs.CL": "Computation and Language: …", "cs.CV": "…", "…": "…"}
    }
  }
}
```

- 送信は**公式 SDK `typesafe-sdk==0.7.0` の `TypeSafeClient.system_one`**。両ターゲットで同じクライアント・同じ
  retry 方針・同じ timeout を使う。違うのは `base_url` と API key だけである。
- SDK が実際に送った**本文のバイト列の SHA-256** を標本ごとに記録する（transport 層で捕捉）。`report` は
  両ターゲットで本文ハッシュが一致した標本数を表示する。一致しない標本があれば apple-to-apple は崩れている。
- 応答は SDK が型付きで復元したものではなく、**wire の JSON をそのまま**結果ファイルに残す。

## 4. apple-to-apple にしたもの

| 項目 | どう揃えたか |
| --- | --- |
| 入力データ | 同じ標本ファイル（同じ ID・同じ state 文字列）。前処理（切り詰め・除去）は `prepare` で 1 回だけ行う |
| 要求本文 | 同じ関数が組み立て、送信バイト列の SHA-256 を両方で記録・照合する |
| 質問の型・指示文・候補とその説明 | 設定ファイルの 1 か所に書き、両ターゲットで共有する（ターゲット別に上書きできない） |
| model 名 | 既定は両方 `jev-latest`（SDK 既定）。応答の `model` を標本ごとに記録する |
| クライアント | 公式 SDK 0.7.0 の同期クライアント。retry（既定 `max_retries: 2`）・timeout（既定 60 秒）・同時実行数（既定 1）は `client:` で共有 |
| 評価する標本 | 対比較の表は**両ターゲットが両方とも成功した標本の共通部分**で計算する。ターゲット別の表はそのターゲットが成功した標本で計算し、失敗は「カバレッジ」と状態別件数（`not_sent` を含む）で出す |
| 条件の一致 | 結果の各行に標本と質問条件のハッシュを持たせ、今の設定と一致する行だけを集計する（§2.3） |
| 指標 | 同じ関数（`bench/metrics.py`）で計算する。対応のある比較（McNemar の正確検定、正解率差の paired bootstrap 95% 区間）を出す |

## 5. apple-to-apple に**できない**もの（留保）

比較結果を読むときは、以下を必ず併記すること。

| ID | 留保 | 影響 |
| --- | --- | --- |
| C1 | **学習データへの混入（contamination）を確認できない。** 実 Jev の学習データ・知識カットオフは非公開。AG News・livedoor は 2000 年代〜2012 年の古い公開データで、どちらのモデルが見ていても不思議はない。arXiv は 2026-06 以降に限定して下げたが、ゼロにはできない | 公開データでの正解率は、未知データでの性能より高く出る可能性がある。どちらに有利かも分からない |
| C2 | **`jev-latest` は動く alias。** 実 Jev 側の実体は予告なく変わり得る。ローカルは不変 ID（`jevbert-poc-nli-ja-en-0.2.0`） | 実 Jev の結果は実行日に依存する。応答の `model` を記録してあるので、途中で変わったら検出できる。長時間にわたる実行は同一モデルとは限らない |
| C3 | **`confidence`・確率の意味が違う。** ローカルは未校正の NLI 分布（T = 1.0）。実 Jev の `confidence` の式は非公開（`compat/differences.md`） | NLL・Brier・ECE は「それぞれの出力確率をそのまま評価した値」であって、同じ尺度の比較ではない。**正解率・macro-F1 を主指標**とする |
| C4 | **レイテンシーは比較できない。** 実 Jev はインターネット越し・共有基盤・rate limit あり。ローカルは loopback・RTX 5090・単一利用者 | レイテンシーは参考値として記録するだけ。性能比較の結論に使わない |
| C5 | **入力長の上限が違う。** ローカルは 1 系列 2,048 token（超過は 422 `context_length_exceeded`、切り詰めない）。実 Jev の上限は資料上の token 予算のみで、実挙動は未確認 | `prepare` で文字数を切り詰めて両者の上限内に収める（arXiv 要旨 2,000 字・livedoor 本文 500 字・AG News 1,000 字）。それでも片方だけが 422 を返した標本は共通部分から外れ、カバレッジに出る |
| C6 | **閉集合の分類である。** arXiv の 16 ラベル外の論文は除いた。実際の arXiv の分布（150 以上のカテゴリ、裾が長い）とは違う | 数値は「16 択の primary category 当て」の性能であり、arXiv 分類一般の性能ではない |
| C7 | **arXiv の primary は `categories` の先頭という慣例に依拠。** snapshot 側での保証は確認していない。また cross-list があるので、別カテゴリの予測が「間違い」とは限らない | 補助指標 `any_category_hit` を併記する |
| C8 | **ラベルの説明文・指示文は私たちが書いた。** 両ターゲットに同じ文を送るので公平ではあるが、どちらかの流儀（プロンプト感度）に偶然合っている可能性はある | 指示文を変えた実験は設定ファイルの差し替えで回せる。1 種類の文言での結果を一般化しない |
| C9 | **決定性。** ローカルは同じ入力に同じ出力（GPU の浮動小数点誤差を除く）。実 Jev が決定的かは未確認 | 実 Jev は同じ標本を 2 回送った一致率を別途測らないと、差のどこまでが揺らぎか分からない（この環境では既定で 1 回ずつ） |
| C10 | **retry の効き方が違う。** 両者に同じ retry 方針を渡すが、429 を返すのは実 Jev だけ。試行回数は標本ごとに記録する | 失敗率・所要時間の差は基盤の差で、モデルの差ではない |
| C11 | **token 数・費用は比較できない。** ローカルの `usage.input_tokens` は自前の tokenizer での数で、Jev の課金 token とは定義が違う（仕様書 §5.8、`compat/differences.md` D03。候補ごとに state を繰り返すので候補数に比例して大きくなる） | `usage` は記録するが比較表に出さない |
| C12 | **データの利用条件。** livedoor は CC BY-ND 2.1 JP、AG News は配布元のライセンス表記なし（学術利用が慣行）、arXiv メタデータは CC0（要旨の著作権は著者）。データはリポジトリに含めず、利用者の手元で取得する | **実 Jev に送る = 第三者のサービスにテキストを送る**ことになる。データの利用条件と実 Jev の利用規約の確認は利用者の責任で行う |
| C13 | **ローカル側は学習済み JevBERT ではない。** 中身は zero-shot NLI 分類器（README「ではない」） | 比較は「現在のローカル実装」と「実 Jev」の比較であり、JevBERT という手法の上限の評価ではない |
| C14 | **正規表現の除去（`remove_patterns`）は切り詰めの前に全文へかかる。** 設定ファイルに病的な正規表現を書くと `prepare` が遅くなり得る | 設定ファイルは自分で書くものとして許容した（第三者の入力ではない） |

## 6. 実 Jev を叩くときの手順と注意

実 Jev の API key はリポジトリにも `.env` の既定値にも**無い**。叩く人が自分で設定する。

1. `TYPESAFE_API_KEY` を環境変数に設定する（設定ファイルの `targets.jev.api_key_env` で名前を変えられる）。
2. `plan` で件数（既定 620 リクエスト）と 1 件目の本文を確認する。課金額はこの環境では見積もらない（Jev の単価表を持っていない）。
3. まず `--limit 5` などで少数を送り、応答が記録されることを確かめる。
4. 本番の実行。`--allow-billable` を付けたときだけ送る。

## 7. 出力

```
bench_runs/<実験名>/            ← gitignore 対象
  samples/<タスク>.jsonl         送信する state・正解・メタデータ（prepare が書く）
  samples/<タスク>.manifest.json 取得元 revision・抽出条件・SHA-256
  runs/<ターゲット>/<タスク>.jsonl  1 標本 1 行: 状態・HTTP status・choice・確率・confidence・応答 model・
                                  request_id・usage・所要時間・試行回数・送信本文の SHA-256・wire の応答 JSON・
                                  標本と質問条件の SHA-256
  report.md / report.json        比較表（report が書く）
```

### 7.1 状態（`status`）

| 値 | 意味 | 再実行で再送するか |
| --- | --- | --- |
| `ok` | choice のある応答を得た | しない |
| `api_error` | HTTP エラー（`http_status` に番号） | 408・429・5xx は再送。それ以外（422 など）は `--retry-failed` のときだけ |
| `connection_error` | 接続できない・timeout | 再送 |
| `no_choice` | 200 だが choice が無い | `--retry-failed` のときだけ |
| `response_invalid` / `sdk_error` | SDK が応答を検証できなかった等 | `--retry-failed` のときだけ |
| `request_changed` | retry の間に送信本文が変わった（SDK の挙動が変わった兆候。apple-to-apple の前提が崩れる） | `--retry-failed` のときだけ |

## 8. 環境構築時の動作確認（2026-09-22、Windows ネイティブ）

**品質の主張ではない。** 環境が端から端まで動くことの確認である。実 Jev には 1 件も送っていない。

| 確認 | 結果 |
| --- | --- |
| 単体テスト（`tests/bench/`、108 件） | すべて成功。リポジトリ全体は 1159 passed / 136 skipped（skip はモデル重みの無い worktree での `model` マーカー） |
| `prepare`（3 タスク） | arXiv 240 件（該当 55,790 件から）・AG News 200 件（7,600 件から）・livedoor 180 件（734 件から） |
| 別ディレクトリへの 2 回目の `prepare` | 標本ファイルが**バイト単位で一致**（再現性） |
| `run --target local`（620 件） | 620 / 620 が `ok`、所要 38 秒（同時実行 1）。再実行すると 620 件すべて skip（再開動作） |
| 同じ標本でローカルを 2 回実行 | 3 タスクとも正解率が完全一致（ローカル側の決定性。C9） |
| 条件ハッシュ導入前の記録 620 件が残った状態で `run` | 620 件を古い記録として無視して送り直し、`report` に「stale record(s) ignored」と表示 |
| `run --target jev`（`--allow-billable` なし） | 送信せず終了コード 3（「620 requests would be sent (up to 1860 HTTP attempts with retries)」） |
| `run --target jev --allow-billable`（key 未設定） | 送信せず終了コード 2（「set TYPESAFE_API_KEY」） |
| 2 ターゲットの対比較（両方ともローカルを指す一時設定、各タスク 10 件） | 送信本文の SHA-256 一致 **10 / 10**、予測の一致率 1.000、McNemar p = 1.0（同じサーバーなので当然の結果。比較経路が正しく動くことの確認） |

参考として、ローカル単独の結果（`report` の出力。**留保 §5 を読まずに引用しないこと**）:

| タスク | ラベル数 | 正解率 | macro-F1 | p50 ms |
| --- | --- | --- | --- | --- |
| arxiv-primary | 16 | 0.412 | 0.406 | 107.3 |
| ag-news | 4 | 0.885 | 0.882 | 14.6 |
| livedoor | 9 | 0.350 | 0.320 | 55.3 |

## 9. 未決・今後

- 実 Jev での実行（課金。PO 判断）。
- 実 Jev の決定性の確認（C9）: 同じ標本を 2 回送る設定を追加するかどうか。
- 指示文の感度分析（C8）。
- 多ラベル（cross-list 全体）を Noul の並びで問う設計は、リクエスト数がラベル数倍になるため見送った。

## 変更履歴

| 版 | 日付 | 内容 |
| --- | --- | --- |
| 0.2.0 | 2026-09-22 | レビュー（セキュリティ・設計・QA）の指摘を反映: 結果に標本・質問条件のハッシュを持たせ不一致の行を無視、非ローカルホストは既定で課金対象・`http://` 拒否、再送は再試行可能な失敗のみ（`--retry-failed`）、Ctrl+C で送信停止、request-id ヘッダー欠落を失敗扱いしない、`no_choice` 状態、壊れた JSONL 行の位置を表示、表のセル中の応答文字列をエスケープ |
| 0.1.0 | 2026-09-22 | 初版。環境構築のみ（実 Jev 未実行） |
