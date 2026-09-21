# JevBERT

BERT 系 encoder を用いた、Jev 互換 API を目指す指示条件付き意思決定エンジン。
本リポジトリは **P0.5（PoC）段階**である。

- 仕様・設計書: [JevBERT_spec_design.md](JevBERT_spec_design.md)（v0.3.0）
- PoC 実装設計書: [docs/POC_DESIGN.md](docs/POC_DESIGN.md)
- **実測結果**: [docs/POC_RESULTS.md](docs/POC_RESULTS.md)
- Jev との差分・未確認事項: [compat/differences.md](compat/differences.md)

---

## これが何で、何でないか

**である**

- Jev の `POST /v1/systemone` と同じ形の JSON を受け取り、同じ形の JSON を返す**ローカル HTTP サーバー**。
- 公式 Python SDK `typesafe-sdk==0.7.0` から、**接続先と API key を差し替えるだけ**で動く（[実演](scripts/sdk_demo.py)）。
- 構造・数値の不変条件（I01〜I09）をモデルの精度と独立に検証する層。

**ではない**

- **学習済みの JevBERT ではない。** 中身は公開の zero-shot NLI 分類器
  （`MoritzLaurer/bge-m3-zeroshot-v2.0`）を A0 の形に差し込んだ暫定 backend である（仕様書 §7.7）。
  JevBERT の学習は行っていない。
- **校正されていない。** 確率は `uncalibrated`（T = 1.0）。返る数値は「正しい確率」ではない。
  Jev で使っていた `confidence >= 0.8` のような閾値を移植してはならない。
- **品質は未評価。** [smoke 評価](docs/POC_RESULTS.md#4-smoke-評価傾向の記録品質の主張ではない)は
  52 件の自作例に対する傾向の記録であり、ゲート G2 の根拠ではない。業務での判断品質・指示追従は測っていない。
- **実 Jev と照合していない。** 下記「互換性の現状」を読むこと。

---

## 互換性の現状

Jev 互換を 3 区分で正直に書く。**「Jev 互換」と言い切れる段階ではない。**

### A. 公式 SDK 0.7.0 に対して実証済み（実ソケットで試験）

| 事項 | 試験 |
| --- | --- |
| `TypeSafeClient` / `AsyncTypeSafeClient` から 3 型（Noul / Choice / Score）の送受信と型付き復元 | `tests/sdk/test_sdk_socket.py` |
| **model を指定しない**（SDK 既定の `jev-latest`）呼び出しが、PoC 設定の alias で解決する | `TestJevCompatibilityOverTheSocket::test_the_sdk_default_model_resolves_through_the_alias` |
| `response.request_id` の復元（`x-typesafe-request-id` ヘッダー） | 同上 |
| 例外の `status` 属性、`error.message` からの例外メッセージ抽出 | `TestSyncErrors` |
| `extra_body` による未知トップレベルフィールドが既定で受理され、`reject` 設定で 422 になること | `test_extra_body_unknown_field_is_accepted_by_default` / `..._is_refused_when_configured` |
| 422 本文の `detail` 配列（`loc` が `"body"` 始まり、質問 ID の後に型 tag）を SDK 経由で読めること | `test_a_422_body_carries_jev_s_detail_array` |
| `detail` の併記が例外メッセージを変えないこと | `test_the_detail_array_does_not_change_the_exception_message` |
| 256 質問のリクエスト | `test_many_questions_are_accepted` |
| **日本語入力に現れる全角記号（`＜/s＞` 等）が 200 で返ること** | `test_a_reserved_string_in_any_field_is_answered_not_refused` |
| `strict=True` の応答検証が通ること（float の扱いを含む） | `TestStrictValidationObservations` |
| Score の `legend` が object / array の型を保って復元されること | `tests/sdk/` |

### B. 公開資料・SDK の wire schema から推定して合わせた（実 Jev では未確認）

| 事項 | 根拠 | 典拠 |
| --- | --- | --- |
| 422 本文の `detail` 配列の**外形** | SDK の wire schema（Jev の OpenAPI 由来）が `HTTPValidationError = {"detail":[{"loc","msg","type","input"?,"ctx"?}]}` と定義し、`loc` の例が `["body","questions","urgency","score","criteria"]` | 仕様書 §3.4・§5.9、ADR-016 |
| トップレベル未知フィールドを**無視**する | SDK が `extra_body` で将来のフィールドを送れる。実 Jev の扱いは資料から確定できないため、受理側に倒した | 仕様書 §5.3、ADR-016 |
| 質問数上限 **256** | Jev に質問数上限の記載がなく（token 上限のみ）、件数だけを理由に拒否しないための値 | 仕様書 §4.2 |
| `jev-latest` / `jev-preview` を PoC 設定で受理 | SDK の既定 model が `jev-latest`。Jev の schema も応答 `model` が要求の alias と異なり得るとする | 仕様書 §18.3 |
| 未定義パス 404 / 非対応メソッド 405 のエラー本文 | JevBERT の設計。実 Jev の本文は未確認 | 仕様書 §5.9、`compat/differences.md` D07 |

### C. 実 Jev サーバーに対して**未検証**（G5 未実施）

**実 API key を用いた実 Jev との照合は一度も行っていない。** 以下はすべて「一致する」とも「一致しない」とも書けない。

- 401 / 429 / 5xx のエラー本文の外形（422 のみ SDK schema から寄せた。他は JevBERT 形式のまま）
- 同率時の候補選択規則、token 計数の定義、`confidence` の具体式
- 未知フィールドの実際の扱い、入力長上限の実挙動
- 同一入力に対する実 Jev と JevBERT の判断一致率

さらに、`confidence`・`usage.input_tokens`・`response.model` は**意図的に一致しない**
（モデルが違う以上一致させられない）。詳細は [compat/differences.md](compat/differences.md)。

### D. Jev が受理する要求のうち、JevBERT が**形式上の理由で**拒否しうるもの

[compat/differences.md の R1〜R13](compat/differences.md) に、影響と回避策つきで一覧してある。
要約すると、1 系列 2,048 token・1 リクエスト 131,072 token・展開後 26,214 系列・524,288 文字が上限で、
そのほかは Choice/Score の件数範囲と厳格 JSON の規則である。**ここに無いものを形式で拒否することはない。**

**利用者が書いたテキストそのものが拒否の理由になることはない。**
`＜/s＞` のような全角の予約文字列は、モデル入力の直前で無害化して 200 を返す
（[differences.md L08・L09](compat/differences.md)）。フェーズ2 ではここが 500 になっていた。

---

## セットアップ

```bash
uv sync
uv run python -m jevbert init-env            # .env に API key を生成（32文字未満の鍵では起動しない）
uv run python -m jevbert fetch-model --trust-first-fetch   # 固定 revision を取得し manifest へ SHA-256 を記録（約 1.1 GB）
uv run python -m jevbert serve --config configs/jevbert.poc.yaml   # 127.0.0.1:8765
```

Windows では `./scripts/run_server.ps1` が上記を順に確認して起動する。

```bash
uv run python scripts/sdk_demo.py            # 公式 SDK から §5.6 の例を送る（AC2）
uv run python scripts/bench_latency.py       # §14.3 の条件でレイテンシーを測る（AC6）
uv run python scripts/compare_templates.py   # K4 のテンプレート比較を再実行する
uv run pytest                                # 全テスト（モデル未取得なら model マーカーは skip）
uv run ruff check
```

`fetch-model` は冪等である。2 回目以降はローカルファイルの SHA-256 を manifest と照合し、
一致すれば何も取得しない。一致しなければ**上書きせずエラーで知らせる**。2 回目以降に
`--trust-first-fetch` は要らない。

`--trust-first-fetch` は初回だけ必要である。manifest にまだ hash が無い状態では、
**取得したものを検証する材料が無く、取得結果がそのまま bundle の身元になる**（trust on first use）。
これは運用者が明示的に選ぶ操作であり、フィールドが空だったから起きる操作ではない。
manifest が指す repo・revision がこのビルドの固定値と違えば、`fetch-model` は**取得せずに拒否する**。

```bash
uv run pytest                                # モデル未取得なら model マーカー 140 件は skip
JEVBERT_REQUIRE_MODEL=1 uv run pytest        # その skip を失敗にする（検証時はこちら）
```

`model` マーカーの 140 件が skip されたときのサマリ行は、それらが通ったときと見分けがつかない。
実モデルまで含めて確認したい実行では `JEVBERT_REQUIRE_MODEL=1` を付ける。

## API の使い方

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient(base_url="http://127.0.0.1:8765", api_key="<.env の鍵>") as client:
    response = client.system_one(                      # model を指定しなければ jev-latest
        state={"message": "同じ利用料金が二重に引き落とされました。今日中に返金してください。"},
        questions={
            "refund_requested": Noul(instructions="顧客は明示的に返金を求めていますか。"),
            "department": Choice(
                instructions="担当部署を選んでください。",
                criteria={"billing": "請求・返金", "technical": "不具合", "other": "その他"},
            ),
            "urgency": Score(
                instructions="対応期限の切迫度を評価してください。",
                criteria=["指定なし", "数日以内", "当日中"],
            ),
        },
    )
print(response.model)                       # jevbert-poc-nli-ja-en-0.2.0（alias ではなく不変 ID）
print(response.answers["department"].choice)
```

| エンドポイント | 用途 | 認証 |
| --- | --- | --- |
| `POST /v1/systemone` | 型付き判定（Jev 互換対象の中心） | Bearer |
| `GET /v1/models` | 登録モデルと有効な alias の一覧 | Bearer |
| `GET /jevbert/v1/capabilities` | 実効制限・定義 ID・既知差分（**JevBERT 固有**。Jev の API ではない） | Bearer |
| `GET /healthz` / `GET /readyz` | liveness / readiness（**JevBERT 固有**） | なし |

認証は default-deny である。未認証の呼び出し側には、未定義パスでも 404 ではなく **401** を返す
（パスの存在を列挙させないため）。認証済みなら 404 / 405 は通常どおり返る。

---

## 非機能要件の充足状況（AC5）

[POC_DESIGN 第9章](docs/POC_DESIGN.md)の一覧に対する、**実装・実測の結果**。見込みではない。

**充足 13 / 部分充足 2 / 未充足 8。**（N20 は「一部のみ実装」だが未実施の項目が残るため未充足に数える）

| ID | 要件 | 状況 | 根拠（テスト名・計測値・未実装の理由） |
| --- | --- | --- | --- |
| N01 | 出力の型・有限性・候補対応をモデル精度から独立に検証 | **充足** | `contracts/response.py` が送信前に毎回 I01〜I09 を検査し違反は 500。`tests/unit/test_response.py`、`tests/property/test_response_properties.py`（PT02）、実モデルでは `test_the_invariants_hold_on_a_real_distribution`。`confidence` は分布から再計算して照合する。AC2 の根拠を作る `scripts/sdk_demo.py` の検査器自体も `tests/unit/test_scripts.py` で不変条件ごとに検査する |
| N02 | 数値異常時に一様分布・0.5 で成功扱いしない | **充足** | NaN・inf logit と候補数不一致は 500 で、本文に確率を含めない（CT08、`tests/contract/test_numeric_contract.py`）。backend 側でも非有限 logit は例外（`nli.py`） |
| N03 | 生入力・認証情報を標準ログに保存しない | **充足** | 1 リクエスト 1 行の JSON に state・instructions・criteria・候補キー・質問 ID・鍵を含めない。例外メッセージと `exc_info` も出さない。**422 の `message` も利用者のフィールド名を引用しない**（場所は `path` / `detail[].loc` が示す。`TestSL3UnknownFieldNameIsNotReflected`）。`tests/contract/test_logging.py` が sentinel 文字列で検査。実サーバーの出力は POC_RESULTS §7.4 |
| N04 | batch 化してもattention・回答対応・認可境界を混ぜない | **部分充足** | リクエスト間 batching を実装していない（N23）ので混同は構造的に起きない。並行リクエストの回答が混ざらないことは `TestCT12RequestIsolation` で確認。**単一 tenant のみ**（N19） |
| N05 | 無断 truncation なし、overflow は reject | **充足** | `truncation=False`。token 上限超過は 422 `context_length_exceeded`。CT06、実モデルでは `test_nothing_is_truncated` と `test_an_oversized_request_is_refused_not_truncated` |
| N06 | 原子性：部分 200 なし | **充足** | 1 件でも無効な質問があれば推論前に全体を拒否。CT05（`tests/contract/test_request_contract.py`） |
| N07 | ローカル推論、外部への無断転送・無断学習なし | **充足** | サーバーは `local_files_only=True` でのみロードし、ネットワークに出ない。`allow_remote_model_download: false` は設定で固定（`Literal[False]`）。取得は `fetch-model` のみ |
| N08 | supply chain：依存 lock、revision・hash 固定、任意コード実行なし、safetensors | **充足** | `uv.lock`、revision をコミット SHA で固定、`trust_remote_code=False`、`*.bin` を取得しない allow-list、manifest の 6 ファイル SHA-256 をロード時に照合（`tests/unit/test_fetch.py`、`TestWeightVerification`）。**記録済み hash はロードするファイルすべてを覆っていなければ起動しない**、モデルディレクトリ直下の allow-list 外ファイルは拒否、manifest は repo・revision を差し替えられない、初回取得は `--trust-first-fetch` が要る。**既知の限界**: hash 照合と `from_pretrained` の間に TOCTOU の窓があり、PoC では許容している |
| N09 | readiness/liveness 分離、hash 整合、warmup | **充足** | lifespan で hash 照合 → ロード → warmup → 最小 fixture。全て成功するまで `/readyz` は 503。失敗してもプロセスは落とさない。`TestReadiness`、実測は POC_RESULTS §7.1 |
| N10 | 過負荷 529・deadline 504、無制限に積まない | **充足** | admission（既定 8、encode も枠を消費）超過で 529、deadline 30 秒超過で 504。`TestOverloadAndDeadline`、`TestEncodingIsAdmissionControlled` |
| N11 | bundle 不変 ID・digest、版の追跡 | **充足** | bundle digest = manifest の canonical JSON の SHA-256。重み hash・dtype・温度・serializer のどれが変わっても digest が変わる（`TestTheDigestCoversTheWeights` が実 bundle の manifest で検査）。template を変えた manifest は起動を拒否。**escape 規則を変えたフェーズ2.5 では bundle ID を `0.2.0` へ上げた**（仕様書 §5.7、POC_DESIGN §12.4） |
| N12 | 推論 cache 既定無効 | **充足** | cache を実装していない。`result_cache_enabled` は `Literal[False]`。premise の token 化 map は 1 呼び出し内のみ |
| N13 | 入力を権限にしない | **充足** | ツール実行・外部アクセス・モデル選択権限のいずれも入力から変更できない。`TestFakeInjectionIsUnreachable`（fake backend の注入口が request / manifest / 設定のどれからも届かないこと） |
| N14 | レイテンシー目標 p95 ≦ 250 ms（§14.3） | **充足** | **実測 p95 = 34.2 ms**（p50 33.4 / p99 34.5、Q=4・K=8・同時実行 1・warm・100 回）。POC_RESULTS 第6章。「各系列 512 token 以下」はフェーズ2.5 から**最長系列の上界**で確認する（平均ではない。POC_RESULTS §6.1）。**ただし測ったのはこの 1 点のみ** |
| N15 | 観測項目・metrics・drift | **部分充足** | 構造化ログ（段階別所要時間・系列数・token 数・`error_code`）のみ。**metrics endpoint・drift 監視は未実装** |
| N16 | 校正済み確率（G3） | **未充足** | `uncalibrated`、T = 1.0 固定。校正データも校正器もない。capabilities と `X-JevBERT-Calibration` で明示 |
| N17 | 業務品質・指示追従の評価（G2） | **未充足** | 52 件の smoke 評価のみ（傾向の記録）。Noul accuracy 0.650・Choice 1.000・Score 平均誤差 0.261。統計設計・アノテーター複数・区間推定のいずれもない。**Noul の否定例は「別の話題を尋ねる」型しかなく、極性反転の hard negative が無い**ので、測れているのは実質「話題一致」である（POC_RESULTS §4.3）。回帰の番人としての下限は無情報予測器より厳しい側へ引き上げた（同 §4.2） |
| N18 | 利用者別 rate limit 429 | **未充足** | 個人 PoC の単一利用者を想定（ADR-015）。`RateLimitExceededError` は定義のみで誰も送出しない。**したがって 429 に対する SDK の挙動も試験できていない**（POC_DESIGN §8.3） |
| N19 | tenant 分離 | **未充足** | 単一 tenant。CT12 は「並行リクエストの回答が混ざらない」ことに縮小 |
| N20 | 運用：負荷試験、OOM 試験、graceful shutdown、rollback、監視、データ保持 | **未充足（一部のみ）** | deadline / 529 / engine の graceful shutdown は実装済み。**負荷試験（同時実行 8/32）・OOM 試験・rollback 手順・監視は未実施**。CUDA OOM の処理経路は実装済みだが本環境で一度も発火しておらず実地未検証 |
| N21 | 通信路保護（TLS） | **未充足** | TLS なし。既定 bind を `127.0.0.1` にして緩和。`scripts/run_server.ps1` は loopback 以外の `-ServerHost` に警告を出す |
| N22 | 実 Jev との照合（G5）、JavaScript SDK | **未実施** | 実 API key での照合を一度も行っていない。JavaScript SDK は未試験 |
| N23 | GPU batch 待機によるリクエスト間 batching | **未実装** | PoC では意図的に実装しない（N04 の構造的保証と引き換え） |

---

## ライセンス

- 本リポジトリ: MIT
- バックボーン `MoritzLaurer/bge-m3-zeroshot-v2.0`: MIT（配布物には含めない。`fetch-model` で取得する）
