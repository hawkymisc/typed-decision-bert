# Jev との差分・未確認事項

| 項目 | 内容 |
| --- | --- |
| 版 | 0.2.0（2026-09-21）。P0.5 フェーズ2（実モデル backend・互換性優先の反映）時点 |
| 典拠 | [仕様書](../JevBERT_spec_design.md) §2.3, §3.3, §3.4, §5, §6, §18.3, ADR-016；[POC_DESIGN](../docs/POC_DESIGN.md) §5.1, §12.3 |
| 参照した上流の版 | [compat/upstream/README.md](upstream/README.md) |

**本書の最重要点**: 以下の「差分」は、実 Jev API と照合して確認したものでは**ない**。JevBERT 側の
決定と、公開資料・SDK 実装から読み取れた範囲を記録したものである。実 Jev の挙動が本書と一致する保証は
なく、一致しない保証もない（仕様書 §13.6、OPEN-01）。**「Jev 互換」と表記してはならない段階である。**

差分は 3 種類に分けて記す。

- **D**: 意図的に異なる（JevBERT の設計判断）
- **U**: 公開資料だけでは確定できず、JevBERT 側で決めた（実 Jev 照合が未了）
- **L**: PoC 限りの実装上の制限（学習済み A1 bundle で解消する想定）

---

## 0. 最初に読む表：Jev で通る要求が JevBERT で**形式上の理由により**拒否されうる箇所

ADR-016 の方針は「Jev が受理し得る要求を、形式だけを理由に拒否しない」である。
それでもなお残る拒否要因を、影響と回避策つきで列挙する。**ここに無いものは形式で拒否しない。**

| # | 拒否されうる要求 | status / code | 根拠 | 回避策 |
| --- | --- | --- | --- | --- |
| R1 | **1 系列の token 数が 2,048 を超える**（state＋指示＋1 候補＋制御 token） | 422 `context_length_exceeded` | モデル上限は 8,192 だが PoC の実効上限を 2,048 に絞っている（仕様書 §4.2）。切り詰めは禁止（`overflow_policy: reject`） | 設定 `limits.max_sequence_tokens` を上げる（manifest の `model_max_sequence_tokens` = 8192 が天井）。長文は呼び出し側で分割する |
| R2 | **リクエスト内の全系列 token 合計が 131,072 を超える** | 422 `context_length_exceeded` | A0 系は候補ごとに state を繰り返すため、同じ body でも A1 の数倍の token を数える（L04） | 質問数・候補数を減らす。A1 bundle では増幅が消える |
| R3 | **展開後の文字数合計が 524,288 を超える** | 422 `context_length_exceeded` | token 化の前に置いた安価な門（L07）。係数 4 は意図的に緩く、token 上限で通る要求がここで落ちることはほぼ無い | `limits.max_request_chars` を上げる |
| R4 | **Choice の候補が 1 個**、または 256 個以上 | 422 `validation_error` | 仕様書 §5.3 の 2〜255。候補 1 個の分布は常に `{k: 1.0}` で情報が無い。**Jev が 1 候補を受理するかは未確認** | 候補を 2 個以上にする |
| R5 | **Score の段階が 1 個**、または 11 個以上 | 422 `validation_error` | 仕様書 §5.3 の 2〜10 | 段階を 2〜10 にする |
| R6 | **Score の段階に `null`** | 422 `validation_error` | U02。Advanced docs は許容、HTTP reference と SDK は非 null | 段階を文字列・配列・objectにする |
| R7 | **Question の未知フィールド** | 422 `validation_error` | 質問の意味を変え得るフィールドを黙って無視しない（ADR-016）。**トップレベルの未知フィールドは無視する**ので混同しないこと | 該当フィールドを外す |
| R8 | **厳格 JSON 違反**：重複キー、`NaN`/`Infinity`/`1e999`、BOM、孤立 surrogate、入れ子深さ 33 以上 | 400 `invalid_json` / 422 `validation_error` | 仕様書 §5.2。parser に許した後で schema だけに頼らない | 正しい JSON を送る |
| R9 | **`state` のトップレベルが `null`・boolean・number** | 422 `validation_error` | 仕様書 §5.2（object・array の**内部**では使える） | 文字列・配列・object で包む |
| R10 | **body が 2 MiB 超**、`Content-Type` が `application/json` 以外、`Content-Encoding` が `identity` 以外 | 413 / 415 | 仕様書 §4.2 | 圧縮せずに送る、分割する |
| R11 | **未登録の `jev-*` モデル名**（`jev-1.13.0` など） | 422 `model_not_found` | ワイルドカードで受理すると、その版の Jev を実行したと誤認させる（仕様書 §18.3） | `jev-latest`・`jev-preview`、または不変 ID を使う |
| R12 | **同時実行 9 件以上 / 30 秒超過** | 529 `overloaded` / 504 `deadline_exceeded` | 単一 GPU worker の PoC 構成。Jev 側の閾値は未確認 | `max_pending_requests`・`request_deadline_seconds` を上げる |

### エラー本文の外形について

- **422 のみ**、Jev の schema と同形の `detail` 配列を `error` と併記する（仕様書 §5.9、ADR-016）。
  `loc` は `["body", ...]` で始まり、path が質問の内部を指し型が判別できた場合のみ質問 ID の直後に型 tag が入る。
  `input`・`ctx` は**付けない**（入力値を反射しないため。仕様書 §15.2）。
- **401・429・5xx の Jev 本文の外形は、公開資料からも SDK schema からも確定できない。**
  これらは JevBERT 形式（`{"error":{...},"request_id":...}`）のままである。一致は**未確認**。

### エンドポイントの範囲について

- Jev の API として照合対象にしているのは **`POST /v1/systemone` と `GET /v1/models` の 2 つ**である。
- `GET /healthz`・`GET /readyz`・`GET /jevbert/v1/capabilities` は **JevBERT 固有**であり、Jev の API ではない。
  これらの有無・本文は互換性の議論に含めない。
- 未認証の呼び出し側には未定義パスでも 401 を返す（D09）。これは上記 2 エンドポイントへの
  **認証済みの**呼び出しの挙動には影響しない（認証済みなら 404・405 は D07 のとおり）。

---

## D: 意図的に異なる

| ID | 差分 | 理由・典拠 |
| --- | --- | --- |
| D01 | `response.model` は常に JevBERT の不変 bundle ID を返す。`jev-1.13.0` 等は返さない | Jev を実行していないため（仕様書 §2.3）。`jev-latest` 等の alias は `allow_jev_aliases: true` と明示的な対応表がある場合のみ受理し、応答は実 ID |
| D02 | `confidence` は正規化エントロピー由来の JevBERT 独自定義（`normalized-entropy-v1`） | Jev の具体式が公開資料から確定できない（仕様書 §8.2、ADR-005）。**フィールド名と型が同じでも数値は一致しない。Jev で使っていた閾値をそのまま移植してはならない** |
| D03 | `usage.input_tokens` は JevBERT の tokenizer と入力展開方式による計数（`expanded-input-a0-v1`） | Jev の課金 token 数の再現ではない（仕様書 §5.8）。A0 系は候補ごとに state を繰り返すため、A1 系（`expanded-input-v1`）より大きな値になる。**backend の異なる bundle 間で比較してはならない** |
| D04 | 確率は未校正（`uncalibrated`、T=1.0） | 校正は G3。capabilities と `X-JevBERT-Calibration` で明示する（仕様書 §11.3） |
| D05 | JevBERT 固有の応答ヘッダーを付与する（`X-JevBERT-Contract` / `-Confidence` / `-Usage` / `-Bundle` / `-Calibration`） | 互換本文を汚さずに版と定義を追跡する（仕様書 §5.7）。`x-typesafe-request-id` は SDK の `request_id` 復元のために `X-Request-ID` と同値で付ける（ADR-013） |
| D06 | Choice の同率は Unicode code point 順で最小のキーを選ぶ | JevBERT の決定論的規則（仕様書 §5.4）。**Jev の同率処理との一致は未確認** |
| D07 | 未定義パスは 404 `not_found`、非対応メソッドは 405 `method_not_allowed` を JevBERT のエラー本文で返す | フレームワーク既定の本文を露出させない（仕様書 §5.9）。実 Jev がこの 2 つを返すかは未確認。なお **422 だけは Jev の schema と同形の `detail` を併記する**（D11） |
| D08 | 利用者別 rate limit（429）、tenant 分離、リクエスト間 batching、TLS を実装しない | 個人 PoC の範囲（ADR-015、N18・N19・N21・N23） |
| D09 | 認証が default-deny であり、**未認証の呼び出し側には 404・405 の代わりに 401 を返す**（`/healthz`・`/readyz` のみ無認証） | 未認証でパスの存在を列挙させない（POC_DESIGN §12.2 P-2）。認証済みなら 404・405 は D07 のとおり。実 Jev の挙動は未確認 |
| D10 | API key は 32 文字以上でなければサーバーが起動しない | 鍵の強度を運用者の裁量に委ねない（POC_DESIGN §12.2 P-3）。JevBERT 側の運用上の決定であり、Jev の鍵形式とは無関係 |
| D11 | **422 の本文に `error` と `detail` の両方を載せる** | `detail` は Jev の schema と同形（`[{"loc","msg","type"}]`、`loc` は `"body"` 始まり）にして、`detail` を直接読むクライアントが動くようにする（仕様書 §5.9、ADR-016）。`msg` は `error.message`、`type` は `error.code` と同値。**`input`・`ctx` は付けない**（入力値を反射しない。仕様書 §15.2）。401・429・5xx には `detail` を付けない |
| D12 | **トップレベルの未知フィールドを既定で無視する**（モデル入力にも応答にも使わない） | SDK の `extra_body` が将来のトップレベルフィールドを送れるため、形式だけを理由に拒否しない（仕様書 §5.3、ADR-016）。**代償：フィールド名の誤記に気付けない。** 設定 `compat.unknown_top_level_fields: reject` で 422 に戻せる。Question の未知フィールドは従来どおり 422（R7） |
| D13 | **質問数の上限を 256 とする** | Jev に質問数上限の記載がなく token 上限のみであるため、件数だけを理由に拒否しない（仕様書 §4.2、ADR-016）。実効的な上限は token 予算（R2）が決める |

## U: 公開資料の不一致に対する JevBERT の決定

| ID | 不一致 | 決定 | 状態 |
| --- | --- | --- | --- |
| U01 | HTTP reference は `instructions` 必須・非 null、Python SDK は省略・null 可 | 省略と null を同一視し「追加指示なし」と扱う | SDK 0.7.0 が未設定フィールドを wire から省略することを実装で確認済み。**実 Jev への送信結果は未確認** |
| U02 | Advanced は Score 段階の null を許容、HTTP reference と SDK は非 null | Score 段階の null は 422 で拒否 | SDK 0.7.0 の `ScoreAnswer.legend` が `dict[int, str \| dict \| list]` で null を受理しないため、この決定は SDK 復元と整合する。実 Jev は未確認 |
| U03 | HTTP reference の `legend` は文字列 map、SDK 応答型は object・array も許容 | 元の rubric の型・値を保持して返す（勝手に文字列化しない） | SDK 実ソケット試験で object・array の復元を確認済み（`tests/sdk/`）。実 Jev は未確認 |
| U04 | エラー本文、未知フィールド、同率処理が公開資料だけでは確定しない | **422 は SDK の wire schema に合わせて `detail` を併記**（D11）。トップレベルの未知フィールドは既定で無視（D12）、Question の未知フィールドは 422。401・429・5xx は JevBERT 形式のまま | **未照合**。SDK 経由で `detail` を読めること・`extra_body` が既定で受理されることは試験で確認済みだが、これは JevBERT 側の挙動 |
| U05 | `confidence` の具体式、token 計数の完全な再現条件が未確認 | 数値互換の対象から除外（D02・D03） | 解消見込みなし。定義 ID を版管理して差分を可視化する |
| U06 | SDK docs の retry 対象 status と実装既定が食い違う | 実装を正とする（408・429・500〜599 を既定で最大 2 回再試行） | SDK 0.7.0 実装で確認済み。**`inference_error`（500）のような再試行しても回復しない失敗も既定で再試行される**。推論は副作用を持たないため安全だが、利用者には差分として明示する |

## L: PoC 限りの実装制限（`serializer-nli-v1`）

| ID | 制限 | 影響 | 解消予定 |
| --- | --- | --- | --- |
| L01 | `render()` が、トップレベルの文字列 `"[1]"` と配列 `[1]` を同じテキストに落とす | state や criteria がこの 2 つで意味的に異なる場合、モデルは区別できない。API の型保存（応答の `legend` 等）には影響しない | `serializer-v1`（A1 用）には持ち込まない（POC_DESIGN §5.1） |
| L02 | Choice 候補を `"<key>: <description>"` の形で描画するため、候補 `"a"`（説明 `"b"`）と候補 `"a: b"`（説明 null）が同一の hypothesis になる | 両候補のモデル入力が同一になり、同一 logit → 同率になる。API は与えられたキー集合に対する分布を返すので構造は壊れないが、**モデルはその 2 候補を区別できない**。フェーズ1で発見（`tests/contract/test_markers.py`） | 区切りを制御 ID 側へ移す、または key と description を別フィールドとして渡す設計で解消する。A1 の `serializer-v1` ではキーと説明を `typed_json({name, description})` として分離する（仕様書 §6.2）ため、この衝突は起きない |
| L03 | zero-shot NLI backend（`a0-nli-zeroshot-v1`）は JevBERT の学習成果物ではない。指示追従・Score の順序性・日本語品質はいずれも**未評価** | 品質主張に使えない。G2・G3 を通過したものとして扱わない | 学習済み A1 bundle への置き換え（仕様書 §7.7、ADR-011） |
| L04 | 候補ごとに state を繰り返すため、`usage.input_tokens` と計算量が候補数に比例する | `max_request_tokens` の実効値を 131,072 に引き上げている（仕様書 §4.2 の 32,768 は A1 の 1 質問 1 系列を前提とした値）。capabilities で公開する | A1 bundle で `expanded-input-v1` に戻る |
| L05 | `fake-deterministic-v1` bundle は入力のハッシュを返すだけで、回答に意味はない | 設定で明示的に有効化したときだけ登録される（既定は無効）。**その出力をモデル出力として提示してはならない** | PoC 限り。contract 試験専用 |
| L06 | `instructions` が空文字列 `""` のとき、`nli-template-v1` の「指示なし」側を使う（`"{C}"`）。`[]` と `{}` は `"[]"` / `"{}"` に描画されるので従来どおり指示として扱う | `"{I} — {C}"` に空の `I` を入れると全候補が `" — "` で始まり、モデルから見れば意味のある区切りに見える。API の構造は変わらない。**空文字列と「指示なし」を区別したい呼び出し側は区別できない**（実 Jev がどう扱うかは未確認） | A1 の `serializer-v1` は指示を `typed_json` の独立フィールドとして渡すため、空文字列がテンプレート構造に漏れない |
| L08 | **予約文字列を escape するため、モデルが見るテキストが元の入力と一致しない。** `</s>` → `< /s>`、`<s>` → `< s>`、`<pad>`・`<unk>`・`<mask>` も同様に `<` の直後へ空白が 1 つ入る | tokenizer の `split_special_tokens=True` が実測で効かず（制御 token ID が 6 個データ中に残った。POC_RESULTS §5.2）、データと制御 ID を分離する（仕様書 §6.2、CT11）ために採った手段。**API の型保存・応答の `legend`・候補キー集合には影響しない**（escape はモデル入力の直前にだけ適用され、応答は元の値を返す）。影響するのは「モデルが読む文字列」だけで、`<s>` を含む文章の意味がわずかに変わりうる。大文字（`<S>` 等）は tokenizer が特殊 token として扱わないため escape しない | 制御 token を語彙に持たない backbone、または offset ベースで marker を挿入する `serializer-v1`（仕様書 §6.2）で不要になる |
| L07 | 1リクエストあたりの展開後**文字数**に上限 `max_request_chars`（既定 `4 × max_request_tokens`）がある。超過は 422 `context_length_exceeded` | A0 系は候補ごとに state と instructions を繰り返すため、小さな body が数百万文字のモデル入力に膨らむ。token 上限より手前の安価な門であり、**token 上限では受理されうるリクエストが文字数で拒否されることはほぼ無い**（係数 4 は意図的に緩い）。実 Jev にこの上限は無いと思われるが未確認 | A1 の 1質問1系列では増幅が起きないため不要になる見込み |

## 未確認事項（実 Jev 照合が必要なもの）

以下は本プロジェクトが**一度も確認していない**。「一致する」とも「一致しない」とも書けない。

- 実 Jev のエラー本文の全文、`error` オブジェクトのフィールド構成
- 実 Jev の同率時の候補選択規則
- 実 Jev の未知フィールドの扱い（無視するのか 422 なのか）
- 実 Jev の token 計数の定義と `usage` の数値
- 実 Jev の `confidence` の具体式
- 実 Jev の入力長上限（リクエスト 64k / state と最長質問 32k という記載の実挙動）
- 同一入力に対する実 Jev と JevBERT の判断一致率（仕様書 §13.6 の比較指標）
- JavaScript SDK での動作（C2 の後続）

これらは G5（実 Jev 置換検証）の対象であり、許可された認証情報・データ・予算が用意できた時点で
`compat/fixtures/` に観測を残したうえで本書を更新する。
