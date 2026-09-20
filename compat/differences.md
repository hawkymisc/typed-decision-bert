# Jev との差分・未確認事項

| 項目 | 内容 |
| --- | --- |
| 版 | 0.1.0（2026-09-21）。P0.5 フェーズ1（API層・contract・compiler・fake backend）時点 |
| 典拠 | [仕様書](../JevBERT_spec_design.md) §2.3, §3.3, §3.4, §5, §6；[POC_DESIGN](../docs/POC_DESIGN.md) §5.1 |
| 参照した上流の版 | [compat/upstream/README.md](upstream/README.md) |

**本書の最重要点**: 以下の「差分」は、実 Jev API と照合して確認したものでは**ない**。JevBERT 側の
決定と、公開資料・SDK 実装から読み取れた範囲を記録したものである。実 Jev の挙動が本書と一致する保証は
なく、一致しない保証もない（仕様書 §13.6、OPEN-01）。**「Jev 互換」と表記してはならない段階である。**

差分は 3 種類に分けて記す。

- **D**: 意図的に異なる（JevBERT の設計判断）
- **U**: 公開資料だけでは確定できず、JevBERT 側で決めた（実 Jev 照合が未了）
- **L**: PoC 限りの実装上の制限（学習済み A1 bundle で解消する想定）

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
| D07 | 未定義パスは 404 `not_found`、非対応メソッドは 405 `method_not_allowed` を JevBERT のエラー本文で返す | フレームワーク既定の `{"detail": ...}` を露出させない（仕様書 §5.9）。実 Jev がこの 2 つを返すかは未確認 |
| D08 | 利用者別 rate limit（429）、tenant 分離、リクエスト間 batching、TLS を実装しない | 個人 PoC の範囲（ADR-015、N18・N19・N21・N23） |

## U: 公開資料の不一致に対する JevBERT の決定

| ID | 不一致 | 決定 | 状態 |
| --- | --- | --- | --- |
| U01 | HTTP reference は `instructions` 必須・非 null、Python SDK は省略・null 可 | 省略と null を同一視し「追加指示なし」と扱う | SDK 0.7.0 が未設定フィールドを wire から省略することを実装で確認済み。**実 Jev への送信結果は未確認** |
| U02 | Advanced は Score 段階の null を許容、HTTP reference と SDK は非 null | Score 段階の null は 422 で拒否 | SDK 0.7.0 の `ScoreAnswer.legend` が `dict[int, str \| dict \| list]` で null を受理しないため、この決定は SDK 復元と整合する。実 Jev は未確認 |
| U03 | HTTP reference の `legend` は文字列 map、SDK 応答型は object・array も許容 | 元の rubric の型・値を保持して返す（勝手に文字列化しない） | SDK 実ソケット試験で object・array の復元を確認済み（`tests/sdk/`）。実 Jev は未確認 |
| U04 | エラー本文、未知フィールド、同率処理が公開資料だけでは確定しない | 仕様書 §5.9 の形式で固定。トップレベルと Question の未知フィールドは 422 | **未照合**。SDK の `extra_body` で未知フィールドを送ると 422 になることを試験で確認済みだが、これは JevBERT 側の挙動 |
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
