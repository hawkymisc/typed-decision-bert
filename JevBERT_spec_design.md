# JevBERT 仕様・設計書

> **JevBERT — BERT系encoderを用いた、Jev互換APIを目指す指示条件付き意思決定エンジン。**

| 項目 | 内容 |
| --- | --- |
| 文書バージョン | 0.2.0 |
| 作成・公開資料確認日 | 2026-09-21（Asia/Tokyo） |
| ステータス | P0.5（PoC）実装に向けて詳細化した設計。モデル学習、校正、性能測定、実Jev APIとの互換性試験は未実施。PoCの実装設計は[docs/POC_DESIGN.md](docs/POC_DESIGN.md)を参照 |
| プロジェクト名 | **JevBERT** |
| Pythonパッケージ名案 | `jevbert`。配布レジストリ上の利用可否は未確認 |
| 主な対象読者 | API実装者、機械学習エンジニア、評価・運用担当者 |
| 互換対象 | 確認日時点のTypeSafe System One HTTP APIおよび公式SDKの公開契約 |
| 本文の規範 | MUST＝必須、SHOULD＝原則実施、MAY＝任意。いずれもJevBERT側の要求であり、実装済みを意味しない |

**本書の中心的な区別は、APIの形を合わせること、指示に従って正しく判定すること、確率やconfidenceを使う既存処理を安全に置換することは、それぞれ別の課題だという点である。**

外部仕様・先行実装についての記述には参考資料番号を付す。根拠を付していない要件、数式の適用方法、構成、制限値、受入基準は、本プロジェクトの設計判断または明示した暫定案である。参考資料は末尾にまとめる。

## 目次

1. [目的と製品境界](#sec-01)
2. [互換性の定義と達成条件](#sec-02)
3. [外部仕様の確認結果と不一致](#sec-03)
4. [機能要件・非機能要件](#sec-04)
5. [HTTP API契約](#sec-05)
6. [入力正規化・シリアライズ](#sec-06)
7. [モデルアーキテクチャ](#sec-07)
8. [確率・confidence・判断保留](#sec-08)
9. [学習データ設計](#sec-09)
10. [学習方法・損失関数](#sec-10)
11. [キャリブレーション](#sec-11)
12. [推論サーバー・内部境界](#sec-12)
13. [評価・互換性試験](#sec-13)
14. [受入基準・リリースゲート](#sec-14)
15. [セキュリティ・データ管理](#sec-15)
16. [デプロイ・観測・再現性](#sec-16)
17. [実装構成・設定](#sec-17)
18. [Jevからの移行](#sec-18)
19. [開発段階・成果物](#sec-19)
20. [設計判断記録](#sec-20)
21. [未確認事項・リスク](#sec-21)
22. [参考資料](#sec-22)
23. [付録A：リクエストJSON Schema](#sec-23)
24. [付録B：成功レスポンスJSON Schema](#sec-24)
25. [付録C：数値処理の参照実装](#sec-25)

---

<a id="sec-01"></a>

## 1. 目的と製品境界

### 1.1 目的

JevBERTは、`state`と実行時に渡される`instructions`・`criteria`を入力し、`noul`、`choice`、`score`の型付き回答を返す。中心となるモデルは、BERT系の双方向encoderと学習済みの判定ヘッドとする。自由文やJSON文字列の自己回帰生成は行わず、APIサーバーがスコアからレスポンスを構築する。

目標とする条件付き分布は次のとおりである。

$$
p_\theta(y\mid s,i,c,t)
$$

ここで、`s`はstate、`i`は指示、`c`は候補・評価基準、`t`は質問型である。固定クラスの分類器とは異なり、指示、選択肢、rubricがリクエストごとに変わる。入力として受理できるだけでなく、それらの変更を判定に反映する能力を学習・評価する。

### 1.2 MVPの対象

初期対象は**問い合わせ・サポート文書の判定**とする。具体的には、返金要求の有無、担当部署へのルーティング、緊急度のrubric評価を共通のAPIで扱う。これは要件具体化のための暫定ドメインであり、ユーザーから業務ドメインが指定されたものではない。

開発時の評価言語は日本語と英語とする。ただし、両言語の品質を無条件に保証するものではない。公開する各モデルは、受入基準を通過した言語・業務・入力長・候補数の範囲を明記する。日本語が未達なら、日本語対応をうたわず限定プレビューに留める。

### 1.3 対象外

v0.1では、自由文生成、説明文生成、画像・音声入力、外部検索、ツール実行、業務上の副作用の実行を提供しない。任意業務での汎用ゼロショット性能、Jevと同じ重み・内部構造、Jevと同一の全出力・速度・課金単位も保証しない。

複数質問間の推論依存はサーバー内で解決しない。ある回答を次の質問の入力に使う場合は、呼び出し側が次のリクエストを組み立てる。同一リクエスト内の質問は独立に評価する。これは公開されているJevの質問モデルとも整合する。[S02](#source-s02)

---

<a id="sec-02"></a>

## 2. 互換性の定義と達成条件

### 2.1 四つの検証面

| 検証面 | 達成したと判断する条件 | v0.1での位置づけ |
| --- | --- | --- |
| C1：HTTP・構造互換 | 対象範囲でエンドポイント、認証方式、入力型、回答フィールドを扱える | 必須目標。未試験 |
| C2：SDK互換 | 固定した公式SDKで、接続先・認証・モデル設定の変更により要求送信と応答復元ができる | Pythonを必須、JavaScriptを後続。未試験 |
| C3：意味・判断互換 | 型の意味を保ち、対象業務で品質・確率・下流判断への影響が許容される | 数値定義は規定。品質・閾値互換は別途実測 |
| C4：制約・運用互換 | 入力長、候補数、障害応答、レート、レイテンシー等が移行先の要件を満たす | 差分を公開。Jevと同一の制約・性能は目標にしない |

これらは単純な段階や総合点ではない。例えばC1を満たしていても、長文が収まらなければ対象ワークロードを置換できない。

### 2.2 公開時の表現

受入試験前は「**Jev互換APIを目指す実装**」と表記する。試験後も、契約プロファイル、SDKバージョン、対応制限、`confidence`・`usage`の相違を併記した「対象範囲でのAPI互換」とする。根拠なく「完全互換」「drop-inで出力が同じ」「既存閾値がそのまま使える」と表記してはならない。

本書の契約プロファイル名は、JevBERT独自の`jevbert-core-2026-09-21`とする。これはTypeSafeが公開したバージョン番号ではない。

### 2.3 意図的に異なる事項

`response.model`は実際に使用したJevBERTの不変バージョンIDを返す。Jevを実行していないのに`jev-1.13.0`等を返してはならない。`jev-latest`等の入力名を受け付ける移行用aliasは明示設定によってのみ有効化する。

`confidence`は第8章のJevBERT独自定義とする。`usage`もJevBERTのtokenizer・入力展開方式に基づく計数とする。これらはフィールドの存在・型が同じでも、Jevと数値的に同一ではない。

---

<a id="sec-03"></a>

## 3. 外部仕様の確認結果と不一致

### 3.1 確認できた契約の骨格

確認した公式APIは`POST /v1/systemone`であり、Bearer認証、`model`・`state`・`questions`を受け、`model`・`answers`・`usage`を返す。質問IDは回答の対応付けに使われ、モデルへの入力には使われない。[S01](#source-s01)

`noul`はyesの確率に相当する0〜1の数値で、独立した`confidence`フィールドは持たない。`choice`は全候補の分布とその最大確率候補を返す。`score`は0始まりの段階番号の期待値を返す。[S03](#source-s03)[S04](#source-s04)[S05](#source-s05)[S07](#source-s07)

### 3.2 外部仕様とJevBERTの対応

| 項目 | 確認した公開資料 | JevBERTの決定 |
| --- | --- | --- |
| Choiceの候補数 | 公式APIは最大255候補。[S03](#source-s03) | 構造上2〜255。1候補の拒否はJevBERTの暫定制限 |
| Scoreの段階数 | 公式APIは2〜10段階。[S04](#source-s04) | 2〜10。数字だけのrubricも構造上は受理するが品質は保証しない |
| 構造化入力 | state、指示、評価基準にobject・arrayを使える。[S01](#source-s01)[S06](#source-s06)[S08](#source-s08) | JSON構造を保持して正規化する |
| モデル一覧 | `GET /v1/models`は`models`配列と`name`・`description`・`release_date`を使用。[S09](#source-s09) | 同じ外形を採用。OpenAI形式の`data`配列にはしない |
| Jevの入力長 | 確認時点でリクエスト64k、stateと最長質問32kと記載。[S09](#source-s09) | 同じ上限は約束しない。第4章の制限を公開 |
| Confidence | 分布由来の0〜1の統計量。確認資料に具体式は見当たらない。[S07](#source-s07) | 正規化エントロピー由来の独自定義を明示 |

### 3.3 公開資料の不一致と採用方針

| ID | 不一致・未確定点 | v0.1の決定と残す検証 |
| --- | --- | --- |
| U01 | HTTP referenceは`instructions`必須・非nullとしているが、Python SDKは省略・nullを許容。[S01](#source-s01)[S10](#source-s10) | SDK側に合わせ、省略・nullを受理して「追加指示なし」と扱う。SDK 0.7.0は未設定の`instructions`・`criteria`をwireから**省略**する（nullは送らない）ことを実装で確認。[S18](#source-s18) 実Jevへの送信結果は未確認 |
| U02 | AdvancedはScore段階のnullを許容する一方、HTTP referenceとSDKのScore型は非null。[S01](#source-s01)[S06](#source-s06)[S10](#source-s10) | Score段階のnullは422で拒否する。差分fixtureを残す。SDK 0.7.0の応答型`legend`もnull値を受理しないため、この決定はSDK復元とも整合する。[S18](#source-s18) |
| U03 | HTTP referenceのScore `legend`は文字列mapだが、SDK応答型はobject・arrayも許容。[S01](#source-s01)[S11](#source-s11) | 元のrubricの型・値を保持して返す。勝手に文字列化しない。実API照合を残す |
| U04 | エラー本文、未知フィールド、同率時の選択等は公開資料だけでは完全に確定しない | 第5章で独自挙動を固定し、実API照合まで「完全一致」としない。SDKがエラー本文から読む箇所は3.4節で確認済み |
| U05 | `confidence`の具体式、token計数の完全な再現条件が未確認 | 数値互換対象から除外。独自の定義をversion管理する |
| U06 | SDK docsのretry対象statusは`{408, 429, 500, 501, 502, 503, 504, 599}`だが、SDK 0.7.0実装の既定は`{408, 429, 500〜599の全て}`。[S18](#source-s18)[S20](#source-s20) | 実装を正とする。529・500も既定で最大2回再試行されることを前提に、サーバー側の副作用なし・冪等を保つ |

**未知仕様を推測してJevの仕様として記述しない。** API、SDK、実際の動作に差がある場合は、それぞれの観測と採用判断を分けて記録する。

### 3.4 公式Python SDK 0.7.0の実装から確認した事項

2026-09-21にPyPI配布物`typesafe-sdk==0.7.0`のソースを読んで確認した。docsの記述ではなく**SDK実装の観測**であり、SDKの版が変われば再確認する。[S18](#source-s18)

| 項目 | 観測 | JevBERTへの影響 |
| --- | --- | --- |
| URL結合 | `base_url.rstrip("/") + "/v1/systemone"`、`+ "/v1/models"` | `base_url`は`/v1`を含まないサーバールートを指定する。サーバーは`/v1/...`で待ち受ける |
| 送信ヘッダー | `Authorization: Bearer <key>`、`Accept: application/json`、`Content-Type: application/json`、`User-Agent`・`X-TypeSafe-SDK`（`typesafe-sdk/<ver>`）、`X-TypeSafe-Runtime`、再試行時`X-TypeSafe-Retry-Count` | 未知の要求ヘッダーで拒否しない |
| request ID | 応答ヘッダー`x-typesafe-request-id`を読む。無いと`response.request_id`が例外になる | 全応答（成功・エラー）に`x-typesafe-request-id`を付与する（5.7節） |
| 未設定フィールド | `Noul`/`Choice`/`Score`オブジェクトの`None`フィールドはwireから省略。criteria内部のnullは保持 | 省略とnullの両方を受理する |
| `extra_body` | トップレベルbodyへshallow merge | 未知トップレベルフィールドは5.3節どおり422。SDK利用者には差分として明示 |
| 応答の検証 | `extra="ignore"`、`strict=True`。`model`・`usage`・`answers`が対象。未知のanswer typeは無視 | 数値はJSON上もfloatとして出力する。余分なフィールドは無視されるが、I09により追加しない |
| Scoreのキー | wireは文字列キー。SDKが`dict[int, ...]`へ変換 | wireでは`"0"`〜`"K-1"`の文字列キーを返す |
| エラー本文 | `error`（文字列）→`error.message`→`message`→`detail`（文字列／`detail.message`／FastAPI形式の配列）の順でメッセージ抽出 | 5.9節の`{"error":{"code","message",...}}`形式で`error.message`が例外メッセージになる |
| 例外の対応 | 400/401/403/404/422/429は専用例外。500以上（529含む）は`TypeSafeInternalServerError` | 529・503・504はSDKから同じ例外型に見える。`status`属性で区別する |
| Retry既定 | `max_retries=2`、対象`{408, 429, 500〜599}`、`Retry-After`・`retry-after-ms`を尊重、全体予算30秒、既定timeout 10秒 | 504 deadline（30秒）より先にSDK側が10秒でtimeoutし得る。PoCではクライアント側timeoutの明示を推奨 |
| models一覧 | `{"models":[{"name","description","release_date"}]}`。`release_date`は`YYYY-MM-DD`文字列 | 同じ外形で返す |

---

<a id="sec-04"></a>

## 4. 機能要件・非機能要件

### 4.1 機能要件

| ID | 要件 | 優先度 |
| --- | --- | --- |
| F01 | `noul / choice / score`を同じリクエストで混在させられる | MUST |
| F02 | 指示・候補・rubricがリクエストごとに変わっても固定ラベル数のヘッドに依存しない | MUST |
| F03 | JSON文字列を生成せず、有限候補の分布から型付き応答を構築する | MUST |
| F04 | 質問IDの変更、質問の追加・順序変更で既存質問の意味入力を変えない | MUST |
| F05 | 全候補・全段階を応答分布に含める。候補を無断で追加・削除しない | MUST |
| F06 | 構造化state、構造化instructions、構造化criteriaを扱う | MUST |
| F07 | 指示変更、否定、条件追加、基準差し替えに対する挙動を評価する | MUST |
| F08 | モデル、tokenizer、serializer、calibrator、応答定義の版を追跡できる | MUST |
| F09 | Python SDKで正常応答を復元できる | MUST（リリースゲート） |
| F10 | 外部サービスへの無断転送や無断学習を行わず、ローカル推論できる | MUST |

### 4.2 初期制限値

以下は**JevBERTの暫定デフォルト**であり、Jevの制限値でも、学習済み能力の測定値でもない。公開前に負荷・品質試験を行い、モデルmanifestとcapabilitiesに実効値を記録する。

| パラメーター | v0.1案 | 超過時 |
| --- | --- | --- |
| HTTP body | 展開後2 MiB | 413 |
| JSON入れ子深さ | 32 | 422 |
| 質問数 | 1〜32 | 422 |
| Choice候補数 | 2〜255 | 422 |
| Score段階数 | 2〜10 | 422 |
| 質問ごとの総token数 | 2,048以下。state・指示・全候補・制御tokenを含む | 422 |
| リクエスト内総token数 | 各質問に展開した実長の合計32,768以下 | 422 |
| サーバー処理deadline | 30秒。運用設定値 | 504 |
| GPU batch待機の上限案 | 5 ms | 上限到達でdispatch。性能は要測定 |

学習・検証済み長が2,048より短いモデルは、その短い長さを実効上限とする。バックボーンが8,192 tokenを扱えても、判定ヘッドやfine-tuning済みモデルの品質をその長さまで確認したことにはならない。

255候補が構造上許容されても、全候補とstateがtoken予算に収まるとは限らない。**候補数の上限と文脈長の上限は同時に適用する。** 大きい候補集合を黙って分割し、部分softmaxを結合して全体分布と称してはならない。

### 4.3 非機能要件

出力の型・有限性・候補対応はモデル精度から独立して検証する。数値異常時に一様分布や0.5を返して成功扱いしてはならない。生の入力や認証情報を標準ログに保存しない。異なるリクエストをbatch化してもattention・回答対応・認可境界を混ぜない。

レイテンシー目標は第14章の限定した測定条件で扱う。encoder-onlyであるという理由だけで、Jevより高速・安価であるとは主張しない。

---

<a id="sec-05"></a>

## 5. HTTP API契約

### 5.1 エンドポイント

| Method / Path | 用途 | 認証 |
| --- | --- | --- |
| `POST /v1/systemone` | 型付き判定。Jev互換対象の中心 | Bearer必須 |
| `GET /v1/models` | 許可されたモデルID・aliasの一覧 | Bearer必須 |
| `GET /healthz` | プロセス生存確認。モデル詳細を返さない | インフラ側で制御 |
| `GET /readyz` | モデルとcalibratorが読み込み済みかを確認 | インフラ側で制御 |
| `GET /jevbert/v1/capabilities` | JevBERT固有の制限・定義・対応範囲 | Bearer必須 |

`/v1/classifier`、`/v1/chat/completions`、streamingはv0.1の契約に含めない。simple-jevには独自のAPI差分があるため、その契約をJevの契約として流用しない。[S15](#source-s15)

### 5.2 共通入力型

```text
JSONValue = null | boolean | finite number | string
          | array<JSONValue> | map<string, JSONValue>
Content   = string | array<JSONValue> | map<string, JSONValue>
Entry     = Content | null
```

`state`のトップレベルは`Content`とする。したがってトップレベルの`null`・boolean・numberは拒否するが、objectやarrayの内部では利用できる。NaN、Infinity、重複JSONキー、不正なUnicodeを拒否する。検証前にbooleanをnumberへ、numberをstringへ暗黙変換してはならない。

### 5.3 リクエスト

| フィールド | 型 | 必須 | JevBERTのルール |
| --- | --- | --- | --- |
| `model` | string | 必須 | 登録された不変IDまたはalias。空文字列は不可 |
| `state` | Content | 必須 | 空文字列・空配列・空objectは構造上受理する |
| `questions` | map<string, Question> | 必須 | 1件以上。キーは対応付けのみ。モデルに送らない |

Questionは`type`で識別する。トップレベルとQuestionの未知フィールドは422で拒否する。stateや構造化説明の内部キーは業務データとして保持し、schemaの未知フィールド規則を適用しない。

| 質問型 | `instructions` | `criteria` |
| --- | --- | --- |
| `noul` | Entry。省略時null | 省略・null、または`true`・`false`だけを許すobject。片側のみも可。各値はEntry |
| `choice` | Entry。省略時null | 2〜255件の`map<string, Entry>`。キーも候補の意味の一部 |
| `score` | Entry。省略時null | 2〜10個のContentの配列。順番が段階番号を決める |

HTTP JSON内の`noul.criteria`キーは文字列の`"true"`・`"false"`である。Choiceの`null`説明は「キーだけで候補を表す」ことを意味し、候補の削除を意味しない。

### 5.4 回答の定義

#### Noul

```json
{"type":"noul","noul":0.93}
```

`noul = p(yes)`とする。0.5はyes/noに近い確率を与えた状態であり、第三のクラス、欠損値、強度の中間値ではない。レスポンスに`confidence`やbooleanの判定結果を追加しない。行動閾値は呼び出し側で定める。[S05](#source-s05)

criteriaの片側が省略・nullなら、その側にはserializerで固定した汎用yes/no説明を使う。`serializer-v1`ではtrue側を`The answer to the question is yes.`、false側を`The answer to the question is no.`とする。省略と明示nullは同じ正規形にする。両側が曖昧でも質問IDから意味を補ってはならない。

#### Choice

`probabilities`はcriteriaの全キーと同じキー集合を持つ。`choice`は最大確率の候補とする。同率時は**Unicode code point順で最小の候補キー**を返す。これはJevBERTの決定論的規則で、Jevとの同率処理一致は未確認である。

「どれでもない」を表したい場合は呼び出し側がその候補を含める。サーバーが`other`や`unknown`を追加してはならない。返す分布は与えられた候補集合に条件付けられた分布であり、その集合が網羅的であることを保証しない。

#### Score

段階数をK、分布を`p_0, ..., p_(K-1)`とすると、次を返す。[S04](#source-s04)

$$
\mathrm{score}=\sum_{k=0}^{K-1}k\,p_k,\qquad 0\le\mathrm{score}\le K-1
$$

HTTPの`probabilities`と`legend`のキーは`"0"`から`"K-1"`までの文字列とする。`legend[str(k)]`は`criteria[k]`のJSON型・値を保持する。`score`を0〜1へ正規化せず、1始まりにせず、argmax段階や丸め値にも置き換えない。

段階間の数値距離はAPI上のインデックス距離であって、業務上の等間隔の効用を保証しない。業務固有の重みや損失が必要なら、呼び出し側が分布から別途計算する。

### 5.5 成功応答の不変条件

| ID | MUSTとなる条件 |
| --- | --- |
| I01 | 入力の質問ID集合と出力`answers`のID集合が一致する |
| I02 | 各回答の`type`が対応する質問の`type`と一致する |
| I03 | 全数値が有限であり、確率・confidenceは0〜1に収まる |
| I04 | Choice・Scoreの分布和の誤差は`abs(sum(p)-1) <= 1e-6` |
| I05 | `choice`は出力分布のargmax。同率時は規定のtie-breakを使用 |
| I06 | `score`と出力分布から計算した期待値の差は`<= 1e-6` |
| I07 | Scoreの`legend`に元の段階説明が型を保って格納される |
| I08 | Noulに`confidence`は付かない |
| I09 | 本文に未規定の`reasoning`・`explanation`・`routing`等を追加しない |

API出力を小数点2桁などに一律丸めない。表示用丸めとwire上の数値を分離する。サーバーが出力する分布・score・confidenceは同じ最終確率ベクトルから計算する。

### 5.6 リクエスト・レスポンス例

以下は設計説明用のfixtureであり、モデル推論の実測結果ではない。モデルIDも将来の命名例である。

```json
{
  "model": "jevbert-support-ja-en-0.1.0",
  "state": {
    "message": "同じ利用料金が二重に引き落とされました。今日中に確認して、重複分を返金してください。"
  },
  "questions": {
    "refund_requested": {
      "type": "noul",
      "instructions": "顧客は明示的に返金を求めていますか。"
    },
    "department": {
      "type": "choice",
      "instructions": "この問い合わせを最初に担当すべき部署を選んでください。",
      "criteria": {
        "billing": "請求、支払い、返金の問い合わせ",
        "technical": "ソフトウェアの不具合や接続障害",
        "other": "上記に該当しない問い合わせ"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "顧客が表明している対応期限の切迫度を評価してください。",
      "criteria": [
        "対応期限の指定がない",
        "数日以内の対応を求めている",
        "当日中または直ちに対応することを求めている"
      ]
    }
  }
}
```

```json
{
  "model": "jevbert-support-ja-en-0.1.0",
  "answers": {
    "refund_requested": {"type": "noul", "noul": 0.93},
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.8, "technical": 0.15, "other": 0.05},
      "confidence": 0.44214218356782187
    },
    "urgency": {
      "type": "score",
      "score": 1.6,
      "legend": {
        "0": "対応期限の指定がない",
        "1": "数日以内の対応を求めている",
        "2": "当日中または直ちに対応することを求めている"
      },
      "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
      "confidence": 0.27015330083790245
    }
  },
  "usage": {"input_tokens": 500, "output_tokens": 0}
}
```

`input_tokens: 500`は計数フィールドの例示値であり、この文章を実tokenizerで計測した値ではない。一方、confidenceとscoreは、記載した分布から本書の式で計算した値である。

### 5.7 モデルID・ヘッダー・capabilities

モデルの不変IDは、重みだけでなくtokenizer、serializer、校正器、数値処理定義を含む**推論bundle**を識別する。校正器だけの差し替えでも出力は変わるため、既存の不変IDのまま内容を更新してはならない。

互換本文を汚さず、次のJevBERT固有ヘッダーを返す。

```text
X-Request-ID: <server-generated-id>
x-typesafe-request-id: <same-server-generated-id>
X-JevBERT-Contract: jevbert-core-2026-09-21
X-JevBERT-Confidence: normalized-entropy-v1
X-JevBERT-Usage: <usage-semantics-id>
X-JevBERT-Bundle: <immutable-bundle-digest>
X-JevBERT-Calibration: calibrated | uncalibrated
```

`x-typesafe-request-id`は公式SDKが`response.request_id`・例外の`request_id`として読む応答ヘッダーであり、SDK互換（C2）のために`X-Request-ID`と同じ値で付与する（3.4節）。request IDはサーバー生成とし、クライアント指定値を信頼して採用しない。エラー応答本文の`request_id`も同じ値とする。`X-JevBERT-Usage`はbackendの入力展開方式ごとのID（A1は`expanded-input-v1`、A0系は`expanded-input-a0-v1`）を返す（5.8節）。

`GET /v1/models`では、登録・認可されたモデルだけを`models`配列に含める。`name`・`description`・`release_date`は実際のmanifestから取得する。本書作成日をモデルのリリース日として流用しない。[S09](#source-s09)

`GET /jevbert/v1/capabilities`は、contract ID、利用可能bundle、実効制限、検証済み言語・業務・候補数、confidence定義、usage定義、calibration状態、既知の互換差分を機械可読で返す。未評価範囲は`unknown`または明示した未検証状態とし、検証済みと見せない。

### 5.8 Usage

`usage.input_tokens`は、質問ごとに組み立てたモデル入力の非padding token数の合計とする。特殊tokenを含む。同じstateをQ個の質問へ展開した場合はQ回分を数える。batch padding、再試行、サーバー内部の追加計算は含めない。

上記はA1（1質問1系列）の定義`expanded-input-v1`である。A0系backend（候補ごとに1系列）では、`usage.input_tokens`を**全質問・全候補の系列の非padding token数の合計**とし、定義IDを`expanded-input-a0-v1`とする。同じ入力でもA1より大きな値になるため、backendの異なるbundle間で比較しない。4.2節の「質問ごとの総token数」はA0系では「1系列（state＋指示＋1候補）のtoken数」に、「リクエスト内総token数」は全系列の合計に適用し、実効値はbundleのmanifestとcapabilitiesで公開する。

`usage.output_tokens = 0`とする。自己回帰出力tokenを生成していないためであり、レスポンスJSONをtokenizeした数ではない。この計数はJevの請求用token数の再現ではない。バックボーンやserializerが異なるモデル間の課金・コスト比較にそのまま使わない。

### 5.9 エラーと原子性

| HTTP | JevBERTエラーcode例 | 方針 |
| --- | --- | --- |
| 400 | `invalid_json` | 不正JSON、重複キー、非有限数値表現等 |
| 401 | `unauthorized` | 認証情報の欠落・不正 |
| 413 | `request_too_large` | 展開後body上限超過 |
| 415 | `unsupported_media_type` | 非対応Content-Type・Content-Encoding |
| 422 | `validation_error` / `context_length_exceeded` / `model_not_found` | 型・候補数・長さ・認可されたモデル登録等の検証エラー |
| 429 | `rate_limit_exceeded` | 利用者別制限。可能な場合`Retry-After`を返す |
| 529 | `overloaded` | 推論queueが満杯。短いbackoff後の再試行を許可 |
| 503 | `model_unavailable` | warmup中、モデル読み込み失敗、サービス準備未完了 |
| 504 | `deadline_exceeded` | サーバー処理deadline超過 |
| 500 | `inference_error` | NaN logits、不変条件違反、回復不能な内部異常 |

公式referenceには401・422・429・529が記載されている。上表のエラー本文や追加statusはJevBERTの設計であり、Jevとの全文一致は未確認である。[S01](#source-s01)

未定義のパスは404（`not_found`）、定義済みパスへの非対応メソッドは405（`method_not_allowed`）を同じエラー本文形式で返す。フレームワーク既定のエラー本文（FastAPIの`{"detail": ...}`等）をそのまま露出させない。

公式SDK 0.7.0は`error.message`を例外メッセージとして抽出し、408・429・500番台（529を含む）を既定で再試行する（3.4節）。`retryable`フィールドはJevBERT独自の補助情報で、SDKの再試行判定には使われない。推論は副作用を持たないため再試行は安全だが、`inference_error`（500）のような再試行しても回復しない失敗もSDK既定では再試行される点を差分として公開する。

```json
{
  "error": {
    "code": "context_length_exceeded",
    "message": "The encoded question exceeds the model input limit.",
    "path": ["questions", "department"],
    "retryable": false
  },
  "request_id": "example-request-id"
}
```

1件でも無効な質問があれば、GPU推論前にリクエスト全体を拒否する。実行中に1件でも失敗した場合も成功回答との混在や部分200を返さない。`200`は全質問が完了し、不変条件を満たした場合だけ返す。

---

<a id="sec-06"></a>

## 6. 入力正規化・シリアライズ

### 6.1 正規化規則

serializerを`serializer-v1`として版管理し、学習と推論で同一実装を使う。

| 対象 | 規則 |
| --- | --- |
| Question ID | 入力tokenに含めない。ルーティング・教師信号の抜け道にも使わない |
| Choiceキー | 候補名として入力する。null説明でもキーは残す |
| Choice候補順 | Unicode code point順でソートし、HTTP objectの挿入順に依存させない |
| Score段階順 | 配列順を保持する。ソートしない |
| Score段階番号 | 外部の対応付け・lossに使用。v0.1では独立した数値特徴として入力に追加しない |
| state・構造化説明内のobject | 再帰的にキーをソートする |
| array | 順序を保持する。会話順・業務順序を壊さない |
| 文字列 | 内容の空白・改行・大文字小文字を勝手に正規化しない |
| JSON表現 | UTF-8、固定のコンパクト表記。数値・null・booleanと文字列を区別する |
| Unicode | 意味が変わり得るNFKC変換などを既定で行わない |

文字列`"1"`と数値`1`、文字列`"null"`とnullを別物として保持する。文書のバイト完全一致ではなく、JSONの値と型の保存を保証する。空object等は意味のある判定材料を保証しないが、構造だけを理由にエラーにはしない。

一般のobjectのキー順は意味を持たないとする。順序そのものを評価したい業務データはarrayで表す。これはJevBERTの入力意味論であり、Jevの内部serializer再現を意味しない。

### 6.2 制御tokenとデータの分離

内部表現の概念形は次のとおりとする。表記中の制御tokenは設計上の記号であり、既存tokenizerにそのまま存在するという意味ではない。

```text
<JB_TYPE_CHOICE>
<JB_STATE>        typed_json(state)
<JB_INSTRUCTION>  typed_json(instructions)
<JB_OPTION>       typed_json({name: option_key, description: criterion})
<JB_MARK>
<JB_OPTION>       typed_json({name: option_key, description: criterion})
<JB_MARK>
<JB_END>
```

制御tokenを追加する場合はembeddingを初期化し、fine-tuningに含める。marker位置はserializerが生成したoffsetから取得し、ユーザー文字列から検索して決めない。

ユーザーが文字列中に`<JB_MARK>`等を含めても制御tokenとして解釈されないよう、データtokenizationと制御IDの挿入を分離する。reserved文字列のエスケープ・通常token化の規則を固定し、tokenizerごとに回帰試験する。これは境界混同対策であり、自然言語による指示注入を完全に防ぐものではない。

### 6.3 長文方針

v0.1の`overflow_policy`は`reject`に固定する。state、指示、候補説明のどれも黙って切り詰めない。特に、否定語・例外・候補間の差分が消える切り詰めを正常処理として扱わない。

chunking、retrieval、要約、階層分類は後続の明示的なpipeline機能とする。単一文書の分布は一般にchunkの分布の単純平均では再現できないため、別モデル・別校正・別評価対象として扱う。

---

<a id="sec-07"></a>

## 7. モデルアーキテクチャ

### 7.1 バックボーン選定

| 候補 | 本プロジェクトでの役割 | 注意点 |
| --- | --- | --- |
| `jhu-clsp/mmBERT-base` | 日本語・英語の共通バックボーンの第一実験候補 | ModernBERT系の多言語encoder。下流の指示追従能力と日本語品質は別途学習・評価する。[S14](#source-s14) |
| `answerdotai/ModernBERT-base` | 英語の速度・精度・実装比較用 | 英語・code中心のBERT系encoder。分類にはfine-tuningを要する。日本語品質を推定しない。[S13](#source-s13) |
| ModernBERT-large系 | baseで不足した場合の容量比較 | 大きいモデルが業務効用でも優れるか実測して判断する |
| Layaの公開チェックポイント | 近接する先行実装との比較用 | encoderとoption markerによる判定実装がある。採用するrevisionと用途を固定する。[S16](#source-s16) |
| `MoritzLaurer/bge-m3-zeroshot-v2.0` | **P0.5（PoC）の暫定serving backend**。学習済みJevBERTが存在しない段階で、任意の指示・候補に意味のある分布を返すため | XLM-RoBERTa-large系（BGE-M3）の多言語zero-shot NLI分類器。2クラス（entailment / not_entailment）、MIT、safetensors、標準の`transformers`クラスでロードでき任意コード実行が不要。JevBERTの学習成果物ではなく、指示追従・校正・日本語品質は未評価。[S19](#source-s19) |

v0.1の最終バックボーンは第14章のgateで決める。P0.5のbackendはこの選定の対象外であり、学習済みbundleができた時点で置き換える（7.7節）。JevBERTという名前は特定のバックボーンを固定しない。classic BERTそのものに限定せず、BERT系encoder-only Transformerを対象とする。

### 7.2 A0：候補ごとのcross-encoderベースライン

A0は学習・オフライン比較用とし、v0.1の公開推論backendはA1とする。例外として、学習済みbundleが存在しないP0.5（PoC）に限り、A0派生のzero-shot NLI backendをservingに使う（7.7節）。A0では各候補について、state・指示・候補名・説明を入力し、共有scalar headでlogitを得る。

$$
z_{q,k}=f_\theta(s,i_q,c_{q,k},t_q)
$$

各質問内で候補logitを集めてsoftmaxを取る。Noulもyes/noの2候補として扱える。学習は候補集合全体に対するcross-entropyで行う。

長所は、候補ごとの入力とスコア対応が明確で、正しさの基準実装にしやすいことである。短所はstateを候補ごとに再処理することである。単純化したdense attentionの計算イメージは、質問数Q、候補数K、state長Ls、指示長Li、候補長Lcについて`O(Q K (Ls+Li+Lc)^2)`となる。実際の計算量はバックボーンのattention方式で異なる。

既存のNLIモデルのentailment確率をそのまま候補間softmaxへ流し込めば目的の分布になる、とは仮定しない。使用するlogitと学習目的を合わせ、候補集合が変わる条件で検証する。

### 7.3 A1：候補一括・option-marker scorer（v0.1本命）

1質問につき、state・指示・全候補を1系列に組み立てる。各候補の説明に対応するmarker位置のhidden stateを取得し、全候補で共有するscalar headを適用する。

$$
H_q=\mathrm{Encoder}_\theta(x_q),\qquad
z_{q,k}=w^\top\mathrm{LN}(H_q[m_{q,k}])+b
$$

候補数に比例して出力次元を固定した`Linear(hidden, num_labels)`は使わない。出力は動的なK個のscalarである。質問型は制御tokenで与え、基本形では共有scorerを使う。型別headの追加はablationの結果で判断する。

```text
Request
  -> Validate / normalize
  -> Q個の「state + 指示 + 全候補」系列
  -> 長さ別microbatch
  -> BERT系encoder
  -> 候補markerのhidden state
  -> 共有scalar scorer
  -> 質問内softmax・校正
  -> 型ごとの決定論的response adapter
```

バッチtensorの概念は`input_ids[B,L]`、`hidden[B,L,D]`、`option_positions[B,Kmax]`、`option_mask[B,Kmax]`、`logits[B,Kmax]`とする。無効候補をsoftmaxの分母へ含めない。マスク対象の内部`-inf`は許すが、有効候補logitは有限でなければならない。全候補が無効な行はエラーとする。

A1では候補間の相互参照が可能になる。これは候補を個別評価するモデルと同一の帰納バイアスではない。候補集合変更への頑健性や相対的な説明への依存を別途評価する。API外形の一致から内部計算の一致は導かれない。

### 7.4 「1 forward pass」の意味

A1が削減するのは**同一質問内で候補ごとにstateを繰り返す計算**である。Q質問をbatchに入れても、v0.1ではstateのencoder計算は原則Q回分存在する。tokenizationの再利用は可能だが、state hidden stateの共通化とは異なる。

また、全候補を詰めると系列長が伸びる。単純化したdense attentionでは`O(Q (Ls+Li+sum Lc)^2)`となり、大きいKでA0より常に安いとは限らない。batch1回、GPU呼び出し1回、state処理1回を同義に扱わない。

### 7.5 A2：共有state encoder＋質問別head（後続実験）

後続案では、`H_s = Encoder(s)`を1回計算し、質問・候補表現から`H_s`へのcross-attentionを持つ小さなheadで各候補を評価する。これにより質問間のstate計算共有を目指す。

ただし、A1の全双方向相互作用と同じ関数ではない。再学習と再校正が必要である。A1のstate部分のhidden stateを単にキャッシュして別質問へ再利用してはならない。A1ではそのhidden state自体が質問・候補に依存するためである。

A2はv0.1のクリティカルパスに入れず、Qの大きいワークロードで品質・計算量の両面の改善を確認して採用する。

### 7.6 先行実装の扱い

Layaは近接するencoder型のtyped-decision実装であり、simple-jevにもnative encoder backendとしての起動例がある。したがって「encoder-onlyであること」だけを新規性とは位置付けない。[S15](#source-s15)[S16](#source-s16)

Layaのモデルカードには、typed-decisionsに対するベースモデルの限界、benchmark訓練分割でfine-tuningしたモデルとの違い、過信や候補数・token予算の制約が記載されている。著者の性能値は本プロジェクトで再現した値ではなく、JevBERTの性能根拠や学習法の正しさの保証には使わない。[S16](#source-s16)

2026-09-21の追加調査では、Layaの利用には独自の`laya` pipパッケージが必要で、公式のソースリポジトリは確認できなかった。入力は`[CLS] 型 指示 [SEP] ([MASK] 候補)... [SEP] state [SEP]`の形で、各候補直前の`[MASK]`位置のhidden stateを2層Transformer head＋scalar scorerへ通す構成であり、A1と近い。一方、モデルカード自身がbase checkpointのzero-shot typed-decisions精度をほぼ偶然水準と記載し、日本語の評価値も確認できなかった。simple-jevにはLICENSEファイルがなく、コードの流用可否は未確定である。これらにより、Laya・simple-jevはP0.5のbackend・コード流用元には採用せず、比較対象としての位置づけを維持する（ADR-012）。

### 7.7 P0.5：zero-shot NLI backend（PoC用のA0派生）

学習済みJevBERT bundleが存在しない段階でAPIサーバーをend-to-endで成立させるため、A0（候補ごとのcross-encoder）の形を保ったまま、共有scalar headを**公開zero-shot NLI分類器のentailment log-odds**で代用するbackendを`a0-nli-zeroshot-v1`として定義する。

$$
z_{q,k}=\log p_{\mathrm{ent}}(x_{q,k})-\log\bigl(1-p_{\mathrm{ent}}(x_{q,k})\bigr)
$$

ここで`x_{q,k}`は、premise＝正規化したstate、hypothesis＝指示と候補`k`から決定論的テンプレートで組み立てた文である。候補間の分布は第8章どおり質問内softmaxで得る。Noulは`false, true`の2候補、Scoreは段階ごとの候補として同じ経路を通す。

これは7.2節が戒める「既存NLIのentailment確率を候補間softmaxへ流せば目的の分布になる」という仮定を**置かない**ための位置づけを要する。すなわち、このbackendの出力は構造・数値の不変条件（5.5節）は満たすが、確率としての校正は未実施（`uncalibrated`、T=1）であり、指示追従・Scoreの順序性・日本語品質はいずれも未評価である。capabilitiesとREADMEにその旨を明示し、G2・G3を通過したものとして扱わない。backendの内部契約（12.2節）は共通とし、学習済みA1 bundleへの置き換えでAPI層を変更しない。テンプレート・token予算・評価方法の詳細は[docs/POC_DESIGN.md](docs/POC_DESIGN.md)に置く。

---

<a id="sec-08"></a>

## 8. 確率・confidence・判断保留

### 8.1 共通の確率計算

有効候補logitを`z_k`、校正温度を`T > 0`とし、次を計算する。

$$
p_k=\frac{\exp(z_k/T-a)}{\sum_j\exp(z_j/T-a)},\qquad a=\max_j(z_j/T)
$$

Noulは内部順を`false, true`に固定し、`p_true`を返す。これは差分logitに対するsigmoidと等価である。ChoiceとScoreでは、その質問の有効候補だけを正規化する。

logitは少なくともFP32へ変換して数値処理する。最終出力段階では有限性を確認し、probabilityを再正規化した場合はscore・confidenceも同じ分布から再計算する。

### 8.2 JevBERTのconfidence定義

定義IDは`normalized-entropy-v1`とする。

$$
H(p)=-\sum_{k=0}^{K-1}p_k\log p_k,\qquad
\mathrm{confidence}=\mathrm{clip}\left(1-\frac{H(p)}{\log K},0,1\right)
$$

`0 log 0 = 0`とする。Kは実際の有効候補数で、v0.1ではK≧2である。一様分布なら0、1候補への完全な集中なら1となる。校正後の確率から算出し、ChoiceとScoreで同じ定義を使う。

これは**分布の集中度**であって、「回答が正しい確率」の定義ではない。Jevの具体式を再現したものでもない。公式資料でもconfidenceは分布由来の統計量として説明されているが、確認できた資料から同一式を確定できない。[S07](#source-s07)

最大確率と区別する理由は、一様分布の基準が候補数によって変わるためである。ただし正規化エントロピーでも、Kが違うタスク間で判断リスクが同等になるわけではない。候補追加による値の変化を評価する。

### 8.3 Scoreでの限界

同じconfidenceでも、確率が隣接段階に分かれる場合と両端に分かれる場合では業務リスクが異なる。標準confidenceは順序距離を表さない。

policy評価では、必要に応じて次のような別指標を使う。

$$
V=\frac{\sum_k p_k(k-\mu)^2}{(K-1)^2},\qquad \mu=\sum_k k p_k
$$

この分散指標を標準レスポンスへ無断追加しない。評価レポート、または別の明示的な診断機能で扱う。

### 8.4 判断保留

Jev互換本文には第三の回答型や`abstain`フィールドを追加しない。呼び出し側のpolicy layerが、確率・confidence・業務ルール・入力の対応範囲に基づき、人手確認や別システムへ回す。

入力に根拠がないこと、候補が網羅的でないこと、未知言語・未知業務であることは、softmaxだけでは確実に検出できない。高confidenceの誤答も存在し得るため、閾値は対応範囲の検証の代わりにならない。

Jevで使っていた`confidence >= 0.8`等の閾値をそのまま移植しない。v0.1では「正しい確率」専用headも必須としない。正誤予測器を追加する場合は別の教師信号と独立した検証を必要とする。

---

<a id="sec-09"></a>

## 9. 学習データ設計

### 9.1 学習単位

学習の基本単位は`state + 1 question + target`とする。APIの多質問リクエストから分解してよいが、元文書・会話・顧客等のgroup IDを保持し、split leakageを防ぐ。学習時にcandidate集合は保持し、候補を独立サンプルへ分解して質問内正規化を失わない。

```json
{
  "schema_version": "jevbert-training-v1",
  "example_id": "support-ja-0001-refund",
  "group_id": "conversation-0001",
  "task_family": "support_triage",
  "language": "ja",
  "state": {"message": "重複した引き落とし分を返金してください。"},
  "question": {
    "type": "noul",
    "instructions": "顧客は明示的に返金を求めていますか。"
  },
  "target": {
    "kind": "hard_label",
    "label": true
  },
  "provenance": {
    "source_kind": "human_authored",
    "annotation_policy_version": "support-v1",
    "training_use_approved": true
  }
}
```

`example_id`・`group_id`・`task_family`・`language`・provenanceは管理・評価に使う。業務依存のtask IDやquestion IDをモデル入力に注入して、指示理解の代わりに暗記させてはならない。言語は原則として入力本文から学習し、将来明示特徴を追加する場合は仕様を版上げする。

### 9.2 Targetの型

| kind | Noul | Choice | Score |
| --- | --- | --- | --- |
| `hard_label` | JSON boolean | criteriaに存在するキー | 0〜K-1の整数 |
| `distribution` | `{"false": p0, "true": p1}` | 全候補キーの分布 | `"0"`〜`"K-1"`の分布 |
| `scalar_score` | 使用しない | 使用しない | 0〜K-1の連続値。補助学習用途に限定 |

分布の全キー・和・有限性を検証する。教師の数値だけでなく、人手合議、観測頻度、teacher model等の由来を保存する。

連続scoreだけから一意の確率分布は復元できない。例えば平均1は`[0,1,0]`でも`[0.5,0,0.5]`でも実現できる。`scalar_score`だけを持つ例に架空の確率分布を割り当て、校正された教師として扱ってはならない。

### 9.3 必要なデータ変動

| 変動軸 | 含める内容 |
| --- | --- |
| 指示差 | 同じstateで「要求の有無」「承認要否」「審査への回付」を区別 |
| 条件・否定 | 否定文、例外、閾値、条件の有無により正解が変わる対 |
| 候補差 | 近い候補、説明の境界、候補の追加・削除、null説明、未見の候補名 |
| 構造差 | 文字列、nested JSON、会話array、構造化instructions・criteria |
| 曖昧性 | 情報不足、複数候補の競合、曖昧な期限、矛盾する記録 |
| 言語差 | 日本語、英語、混在入力、指示とstateの言語が異なる例 |
| 頑健性 | 長い前置き、根拠位置の変化、表記揺れ、データ中の偽指示 |

未見の候補集合に対応する能力を評価するため、labelの固定IDだけで解けるデータに偏らせない。Choiceキーは意味の一部なので、その名前を変えた場合に常に同じ結果を要求するのではなく、説明との意味が保たれているかを管理する。

### 9.4 Split

`train`、`model-dev`、`calibration`、`policy-dev`、`locked-test`を分離する。サイズは利用可能データと必要な統計精度に応じて決め、固定テストの評価前に確定する。

元文書・会話・顧客・テンプレート系列でgroup splitし、言い換えやsynthetic augmentationはsplit決定後に生成する。タスク横断の汎化評価には未見instruction template、未見rubric、未見業務のholdoutを別に設ける。時間変化を見るtime-based holdoutも用意する。

公開benchmarkの訓練分割でfine-tuningした結果は、そのbenchmarkのゼロショット結果とは区別する。Jevの回答に一致することを、人手正解への正しさと同一視しない。

### 9.5 データ調達

人手ラベルと利用許可のある業務データを主とする。生成モデルによるsyntheticデータやJev出力の蒸留は任意とし、使用条件・情報送信・保存・学習利用の承認を確認してから行う。既存APIへ顧客データを自動送信する機能は既定で無効にする。

---

<a id="sec-10"></a>

## 10. 学習方法・損失関数

### 10.1 最初に成立させる学習

最初は教師ありの質問内cross-entropyを使い、A0とA1を同じsplit・データ・予算で比較する。RLは必須にしない。適切な学習目的を選ぶことと、有限データで未見業務まで校正されることは別である。

hard labelまたは教師分布を`q_k`とすると、

$$
\mathcal{L}_{CE}=-\sum_k q_k\log p_k
$$

とする。Noulは2クラス、ChoiceとScoreは動的なKクラスである。lossは候補数で重みが変わりすぎないよう質問単位に集約する。

### 10.2 Scoreの順序loss

補助lossとして、累積分布間の距離を使う。

$$
\mathcal{L}_{RPS}=\frac{1}{K-1}\sum_{m=0}^{K-2}
\left(\sum_{k=0}^{m}p_k-\sum_{k=0}^{m}q_k\right)^2
$$

`L_score = L_CE + lambda_RPS * L_RPS`とし、`lambda_RPS`はmodel-devで選ぶ。初期候補を0と0.1とするが、これは最適値の主張ではない。累積lossだけでなく分布・期待値・業務閾値への影響を測る。

連続scoreしかない補助例では、`(E_p[k] - target_score)^2 / (K-1)^2`等を使える。ただし、このlossだけで出力分布を識別できないため、分布品質の根拠に使わない。

### 10.3 蒸留

教師分布を利用する場合、同じ候補集合・順序・指示に対する分布を用いる。教師logitが取得でき、温度付き蒸留を行う場合は、教師と生徒の両方へ同じ蒸留温度を適用し、温度補正を含めて目的を固定する。

教師の確率しか得られない場合は、その分布へのcross-entropyまたはKLを使う。logitがあるかのような処理をしない。教師から誤りや過信も移る可能性があるため、人手正解のlocked-testを別に維持する。

Jevとの判断一致率を高めることと業務上の正確さが競合する場合は、両方を報告し、無条件に一致率を優先しない。

### 10.4 学習手順

1. serializer、tokenizer、特殊token、候補mappingのunit testを先に通す。
2. 少数例へのoverfitで、markerの抽出とlossの対応が正しいことを確認する。
3. A0とA1を共通の教師あり設定で学習し、指示変更テストを含めて比較する。
4. 必要に応じてdomain追加、hard negative、順序loss、蒸留を一つずつ追加する。
5. モデル選定後にcalibrationを行い、別のpolicy-devで行動閾値を選ぶ。
6. 全てを固定してからlocked-testを一度評価し、未達なら新しい実験版として管理する。

初期探索案は、encoder学習率`1e-5 / 2e-5`、head学習率`5e-5 / 1e-4`、weight decay `0.01`、warmup比率`0.05`、最大3 epoch、3 seedとする。batchは例数だけでなくtoken数で制御する。これらは出発点であり、データ・GPU・収束を見て変更し、変更履歴を残す。

---

<a id="sec-11"></a>

## 11. キャリブレーション

### 11.1 初期方式

モデル選定後、独立したcalibration splitのNLLを最小化する正の温度Tをfitする。Temperature scalingは既存の校正手法として参照するが、適用による改善は当該データで測定する。[S17](#source-s17)

初期は質問型ごとの3温度とする。候補数bucket、言語、業務ごとの温度は十分な件数がある場合に限って比較する。細分化でデータ不足になる場合は、型単位または全体温度へ戻す。少数例のECEだけを最適化しない。

### 11.2 再校正の条件

重み、tokenizer、入力serializer、候補数分布、入力長、量子化、attention backendなどを変更した場合は、確率への影響を評価する。影響が無視できない場合は再校正し、bundle IDを変更する。新しいタスクに古いTが有効であるとは仮定しない。

キャリブレーション対象は`p(y | input)`であって、Jevの`confidence`への数値合わせではない。JevBERTのconfidenceは校正後の分布から計算する独立した定義である。

### 11.3 配布物

calibratorには方式、型別T、fit対象のsplit hash、件数、対象範囲、fit目的、校正前後の指標、対応bundleを記録する。本番用bundleに適合するcalibratorがない場合はready状態にしない。研究用にT=1を使うモードは、`uncalibrated`として明示し、本番合格とは区別する。

---
<a id="sec-12"></a>

## 12. 推論サーバー・内部境界

### 12.1 コンポーネント

| コンポーネント | 責務 | 行ってはならないこと |
| --- | --- | --- |
| API gateway | 認証、body制限、request ID、deadline | 未認証リクエストをGPU queueへ流す |
| Contract validator | schema、候補数、深さ、モデル認可の検証 | データを黙って型変換する |
| Model registry | 名前から不変bundleへ解決 | 任意の利用者指定HF ID・URLを直接downloadする |
| Question compiler | 正規形、token IDs、marker位置、候補mappingの生成 | 質問IDをモデルへ入力する |
| Batch scheduler | 長さ・token予算・deadlineに基づくdispatch | 別リクエストを同じattention系列として連結する |
| Encoder backend | 有効候補のlogitを返す | APIのconfidenceを独自に決定する |
| Probability adapter | 校正、softmax、confidence、期待値 | 他backendの既存confidenceを無検証で透過する |
| Response adapter | 元のID・候補キー・legendへ復元 | 候補・質問を欠落させる |
| Policy layer | 呼び出し側で行動・確認・保留を決める | JevBERTのAPIと業務上の権限を同一視する |

### 12.2 Backendの内部契約

内部表現は、少なくとも次を保持する。

```text
CompiledQuestion:
  request_id
  question_id            # モデル外メタデータ
  question_type
  input_ids
  attention_mask
  option_positions
  option_keys            # choice keys / score indices / false,true
  original_criteria      # legend復元用
  logical_input_tokens

RawQuestionResult:
  request_id
  question_id
  valid_option_logits    # 校正前の有限logit
  bundle_id
```

`valid_option_logits`はcandidate mappingと1対1に対応する。学習時と推論時で候補の順序・yes/no順を変えない。候補数分のlogitが得られなければ内部エラーとする。

比較対象の既存実装が確率しか返さない場合は、独立した評価runnerとして扱う。丸め・clip済み確率から元logitを復元したと偽って共通backendへ渡さない。確率に対する再校正を試す場合は、その変換と情報損失を別の方式として記録する。

### 12.3 Scheduling

推論batchはbundle ID、device、dtype、実装backendの一致を条件とし、系列長bucketでpaddingを抑える。batch内の質問数上限と総token数上限の両方を使う。必要なら1 API requestを複数microbatchへ分割するが、全質問が成功するまで応答を確定しない。

queue満杯では529を返し、無制限にメモリへ蓄積しない。client切断やdeadline超過後は未dispatchの処理を取り消す。実行済みGPU計算を途中停止できない場合でも、不要になった結果を別リクエストへ返してはならない。

### 12.4 Cache

v0.1では、明示的なtokenization cache以外の推論cacheを既定で無効にする。後続で有効化する場合は、tenant、bundle、serializer、正規化したstate・指示・候補集合をcache keyへ含める。機密入力のハッシュや埋め込みも保護対象として扱う。

A1ではquestionに依存するstate hidden stateを、別questionの推論へ流用しない。共有表現のcacheを導入する場合はA2等の専用設計が必要である。

---

<a id="sec-13"></a>

## 13. 評価・互換性試験

### 13.1 三種類の正解を分ける

**構造の正解**は契約・不変条件で決まる。**業務の正解**は人手ラベル・明示した業務基準で決まる。**Jevとの一致**は、同じ入力に対する実Jevの観測結果で決まる。

Jevと同じ誤りをしたことを品質向上として扱わず、Jevと異なる正しい回答を互換性失敗だけで捨てない。移行判断では業務品質、動作差分、誤りのコストを並べて判断する。

### 13.2 Contract・数値試験

| 試験ID | 内容 | 期待結果 |
| --- | --- | --- |
| CT01 | 3型の単独・混在、1質問・32質問 | ID・型・件数・応答schemaが正しい |
| CT02 | string、array、nested objectのstate・instructions・criteria | 型と値を保持し、正常にcompileできる |
| CT03 | Choiceのnull説明、Noulの片側criteria、null・省略instructions | 第5章どおりに処理 |
| CT04 | 候補数の境界：Choice 1/2/255/256、Score 1/2/10/11 | 規定どおり受理・拒否 |
| CT05 | 空questions、未知type、未知フィールド、NaN、重複キー | 規定の4xx。部分200を返さない |
| CT06 | token上限の直前・一致・超過、body上限、深さ上限 | 無断truncationなし |
| CT07 | 同率、極端logit、一様logit、1候補以外がpadding | 正規化、tie-break、maskが正しい |
| CT08 | all-masked、NaN logit、候補mapping欠落 | 500等で失敗し、架空の確率を返さない |
| CT09 | Scoreの期待値・legend、Noulのyes/no方向 | 計算・復元が仕様に一致 |
| CT10 | alias、認証、models一覧、readiness、過負荷、deadline | 正しいmodel・status・retry情報 |
| CT11 | reserved文字列、quoted instructions、長いデータ中の偽marker | 制御tokenの個数・位置を乗っ取れない |
| CT12 | 別tenantを混在batch化 | attention、回答mapping、cache、認証情報が分離 |

付録のschemaで確認できない候補集合の一致、期待値、合計、token上限等は、runtime validatorとproperty-based testで補う。

### 13.3 不変性・指示追従の試験

| 試験 | 求める性質 |
| --- | --- |
| Question IDを意味の異なる文字列へ変更 | 対応キー以外の意味入力・分布は変わらない |
| 質問の順序変更、無関係な質問の追加 | 上限内で既存質問の分布が実装差分許容誤差内に収まる |
| Choice objectの挿入順変更 | 正規化後の入力tokenが完全一致する |
| Choice名を含む意味変更 | 不変性は要求しない。新しい候補意味への追従を評価する |
| 指示に否定・条件を追加 | 正解が変わる対で、必要な方向に回答が変わる |
| stateは同じ、rubricの定義が異なる | rubricを無視した固定分類になっていない |
| Score順を変更 | legendと出力indexの対応を保つ。scoreの不変性は要求しない |
| 根拠位置を先頭・中間・末尾へ移動 | 長さ別の性能劣化を測定する |
| 未見rubric・未知業務 | API受理率と判断品質を別々に測る |

数値再現の初期許容誤差は、同一FP32参照backendで確率差`1e-6`、GPUのbatch形状・低精度演算の比較では`1e-3`を候補とする。実装に合わせて公開前に固定し、誤差を超えた差分を無条件に許容しない。候補の最大差だけでなく、実際の行動分岐が変わった件数も測る。

### 13.4 品質指標

| 対象 | 主指標 | 補足 |
| --- | --- | --- |
| Noul | NLL、binary Brier、AUROC、AUPRC、固定閾値でのprecision/recall | クラス不均衡とfalse positive/negativeのコストを併記 |
| Choice | Accuracy、macro-F1、NLL、multiclass Brier | 候補数、未見候補集合、頻出・稀少クラス別に報告 |
| Score | MAE、正規化MAE、RPS、分布NLL | 期待値品質と段階分布品質を分離 |
| 指示追従 | 反実仮想対の両方正解率、条件追加への追従率 | 片側だけの正解で成功としない |
| 校正 | NLL、Brier、ECE、reliability diagram | 型・言語・候補数・長さで分解 |
| Policy | coverage、selective risk、業務期待損失 | confidenceの高さだけを成果にしない |

Scoreの正規化MAEは`mean(abs(pred_score - target_level)/(K-1))`とする。段階数の違うタスクを比較するときは、元の単位のMAEも残す。

binary Brierは`mean((p_yes-y)^2)`、multiclass Brierは`mean(sum_k((p_k-y_k)^2))`とする。この二つを定義を示さず同じスケールで比較しない。

top-label ECEでは`max(p)`とargmax正誤を使用し、JevBERTのエントロピーconfidenceを正解確率として代入しない。初期レポートは15-binのequal-width ECEを基本とし、件数・binごとの支持数・必要な感度分析を併記する。ECEの単一値だけで校正良好とは判断しない。

### 13.5 統計設計

比較は同じlocked-testの同じ入力で行い、元文書・会話groupを単位とするpaired bootstrap等で95%区間を出す。同一stateから作った多質問を独立標本として扱い、見かけの標本数を水増ししない。

policy閾値はpolicy-devだけで選び、locked-testでcoverageとriskを評価する。稀少な重大誤りは、点推定だけでなく片側上限と必要なサンプル量を検討する。データが足りない場合は「合格」ではなく「評価不足」とする。

### 13.6 Jev・先行実装との比較

実Jevへの照合は、許可された認証情報・データ・予算がある場合に実行する。本書作成では行っていない。比較時は可能な限り不変のJevモデルIDを指定し、リクエスト、観測日時、返却model、SDK版を記録する。[S09](#source-s09)

A0、A1、Layaの固定revision、許可されたJevを同一の評価集合で比較する。異なるデータセット、異なる候補数、異なるprompt、校正前後が混在した公開数値を並べて優劣を断定しない。

Jevとの比較指標は、Choice一致率、Noul絶対差、Scoreの正規化差、分布間の距離、既存policyでの分岐変更率とする。これらは同一出力を強制するlossではなく、移行時の影響を可視化する指標である。

### 13.7 公式SDK試験

固定版のPython SDKで、sync・async、`system_one`、`models.list`、型付きQuestion、辞書Question、応答復元、422・429・529での例外・retryを確認する。SDKがScoreのキーを整数へ変換する箇所も含め、HTTPの文字列キーと混同しない。[S04](#source-s04)[S11](#source-s11)[S12](#source-s12)

`base_url`指定に対応していることは公式資料で確認できる。[S12](#source-s12) URL結合・ヘッダー・例外対応・retry既定はSDK 0.7.0の実装で確認した（3.4節）。JevBERTへ向けた実接続はP0.5のSDK試験で確認し、結果を`compat/differences.md`へ記録する。SDK試験は実サーバープロセス（実ソケット）に対して行い、少なくとも次を含める：`system_one`の3型混在の復元、`response.request_id`、`models.list`、Scoreのintキー復元、401→`TypeSafeAuthenticationError`、422→`TypeSafeUnprocessableEntityError`、5xx→`TypeSafeInternalServerError`、`AsyncTypeSafeClient`での同等の正常系。SDKの版は`typesafe-sdk==0.7.0`に固定する。

---

<a id="sec-14"></a>

## 14. 受入基準・リリースゲート

### 14.1 Gate

| Gate | 条件 | 未達時 |
| --- | --- | --- |
| G0：契約 | 付録schema、CT01〜CT12、数値不変条件を全件通過 | HTTP互換版として公開しない |
| G1：SDK | 固定版Python SDKの正常・異常系を通過。資料の不一致を差分表で管理 | SDK互換を表記しない |
| G2：業務品質 | 宣言する言語・業務・候補数・長さの全必須sliceで、事前固定した品質基準を満たす | 対応範囲を狭めるか研究previewに留める |
| G3：確率・policy | 独立calibration、固定閾値でriskとcoverageを評価。既存Jev閾値の移植なし | 自動処理用途のリリースを止める |
| G4：運用 | 認証、負荷、OOM、shutdown、rollback、監視、データ保持の試験を通過 | 本番提供しない |
| G5：Jev置換 | 実Jevとのcontract照合と対象アプリのshadow試験を通過 | 「Jev置換検証済み」と表記しない |

### 14.2 暫定品質目標

次の数値は**低リスクなサポート仕分けを想定した初期提案**である。ユーザーが承認したSLA、既存性能、達成済み結果ではない。業務責任者が誤りコストを確認し、locked-testを見る前に採用・変更する。

| 指標 | 初期提案 |
| --- | --- |
| Choice・Noulのaccuracy差 | 比較対象に対する差の95%区間下限が`-0.02`以上 |
| Score正規化MAE差 | 比較対象との差の95%区間上限が`+0.02`以下 |
| 校正後NLL | 校正前に対して劣化しないことを確認 |
| Top-label ECE | 対象sliceで0.05以下を目安とし、件数・区間を確認 |
| 自動仕分けの誤り率 | 固定policyの対象群で、片側95%上限が1%以下 |
| 自動仕分けのcoverage | 上記risk条件を満たしたうえで50%以上 |

比較対象は開発ではA0、実置換の判定では現在利用するモデル・Jevの固定版とする。比較対象が未計測なら絶対品質とA0比較だけを報告し、Jevとの非劣性が成立したとはしない。

accuracy等の非劣性に加えて、稀少カテゴリのrecall、業務損失、重大誤りも確認する。全体平均だけでsliceの失敗を隠してはならない。上記risk目標は、返金実行等の不可逆な処理の承認基準として流用しない。

### 14.3 性能測定

初期ベンチマークは、Q=`1/4/16/32`、Choice K=`2/8/32/255`、系列長=`128/512/2048`、同時実行数=`1/8/32`を組み合わせる。token上限を超える組合せは推論速度測定から分離し、拒否処理として測る。

CPUのFP32参照実装と、本番候補GPUの固定dtype・固定backendを別々に測る。tokenization、queue待ち、GPU実行、数値処理、serialization、end-to-endを分け、warm/cold、p50/p95/p99、requests/s、questions/s、peak memoryを記録する。

暫定工学目標は、**warm、Q=4、K≦8、各系列512 token以下、同時実行1の固定GPU環境でend-to-end p95≦250 ms**とする。GPU型・runtime・電力設定・測定方法をG4前に固定する。現時点で達成可否は未測定であり、ハードウェア未指定のSLAにはしない。

A1の採用は速度だけで決めない。同じrisk条件でのcoverageと、実運用の稼働率・待機費用を含むコストを比較する。self-hostedを計算資源費用ゼロとして扱わない。

---

<a id="sec-15"></a>

## 15. セキュリティ・データ管理

### 15.1 入力を権限にしない

stateは評価対象データであり、サーバー設定・ネットワークアクセス・モデル選択権限を変更できない。instructionsも判定内容の指定であって、コード実行やファイル読み取りの指示にはしない。モデル出力は業務上の許可・本人確認・認証の代わりにしない。

自然言語による偽指示にencoderが影響される可能性は残る。制御token分離、訓練例、red-team試験により対処するが、モデルをセキュリティ境界として信頼しない。業務の実行権限・入力検証は通常のコードで管理する。

### 15.2 データ保護

標準ログにはrequest ID、bundle ID、型別質問数、長さ、status、処理時間だけを記録する。state、instructions、criteria、認証token、個人識別可能なquestion IDは既定で記録しない。question IDは利用者が個人情報を含め得るため、生でmetric labelにもしない。

顧客リクエストを学習データへ自動転用しない。外部teacher APIへの送信、評価fixtureへの保存、障害解析のbody保存は別の明示承認と保持期間を必要とする。推論サーバーは、modelの事前配備後は外向きネットワークなしで稼働できる構成を目標にする。

### 15.3 Supply chain

依存packageをlockし、model・tokenizer・adapterのrevisionとhashを固定する。`trust_remote_code`相当の任意コード実行を既定で無効にし、必要な場合はコードをレビューして固定する。安全なweight形式を優先し、runtimeが利用者指定の任意URLを取得しないようにする。

バックボーン・既存実装・学習データの利用条件は個別に確認し、JevBERT全体の配布条件とは分ける。名称・商標・配布レジストリの確認は公開前のチェック項目とし、本書で利用可能と断定しない。

---

<a id="sec-16"></a>

## 16. デプロイ・観測・再現性

### 16.1 初期構成

API層とモデルworkerを論理的に分ける。単一マシンでは同一deployment内でもよい。GPUごとにモデル常駐workerを用意し、モデルを各APIプロセスで重複ロードしてメモリを圧迫しない。

推論時はモデルをevaluation modeに固定し、dropoutを無効化し、勾配計算を行わない。起動時に重み・tokenizer・serializer・calibrator・manifestのhash整合を確認し、warmupと最小fixture試験を行う。全て成功するまで`readyz`を503とする。readiness失敗をliveness失敗と混同して再起動loopを発生させない。

### 16.2 観測項目

| 種別 | 記録する値 |
| --- | --- |
| API | request数、status、validation理由、p50/p95/p99、timeout、rate limit |
| Inference | 実系列長、候補数、質問数、queue時間、batch fill、GPU時間、OOM |
| Numerical | 非有限logit、分布和の異常、期待値不一致、mask異常 |
| Model | bundle・serializer・calibrator・confidence定義のID |
| Drift | 入力長、質問型、候補数、確率分布の集計変化。ラベルがある場合の品質変化 |

confidenceの平均低下・上昇だけを精度変化と解釈しない。対応範囲外入力は高confidenceになり得る。運用の正解ラベルを得られる範囲で、遅延評価を実施する。

### 16.3 再現性・更新

同一bundleに対してinput正規化は決定論的にする。GPU kernel、dtype、batch形状による数値差は第13章の許容範囲で測定する。乱数seedだけで全GPU環境のbitwise一致を保証したとはしない。

重み・校正器更新は新しいbundleとしてshadow評価し、policy閾値も対応版を固定する。aliasは新旧bundleのroutingだけを変え、旧bundleをrollback用に保持する。影響がある更新を同じ不変IDで置き換えない。

---

<a id="sec-17"></a>

## 17. 実装構成・設定

### 17.1 リポジトリ案

```text
jevbert/
  src/jevbert/
    api/                 # HTTP、認証、エラー、capabilities
    contracts/           # request/response schema、runtime invariants
    compiler/            # JSON正規化、serializer、token budget
    models/              # A0、A1、共通backend契約
    inference/           # scheduler、registry、worker
    scoring/             # softmax、expectation、confidence
    calibration/         # temperature fit、artifact reader
    training/            # datasets、loss、train loop
    evaluation/          # quality、robustness、policy、benchmark
  tests/
    unit/
    property/
    contract/
    sdk/
    integration/
  compat/
    upstream/            # 確認した公式資料の版・取得情報
    fixtures/            # 許可されたAPI観測、golden fixtures
    differences.md       # Jevとの差分・未確認事項
  configs/
  manifests/
  docs/
  pyproject.toml
  dependency-lockfile
```

API層はFastAPI/Pydantic等、モデル層はPyTorch/Transformers等を候補とする。バージョン番号は本書で未検証のまま「最新」へ固定せず、最初の動作確認で選定しlockfileへ記録する。JSON Schemaと実際のPydantic型・validationが一致することをテストする。

### 17.2 設定例

以下は未学習の開発設定案である。`null`のrevision等を埋めるまで本番bundleとして登録しない。

```yaml
project: JevBERT
contract_profile: jevbert-core-2026-09-21

model:
  public_id: jevbert-support-ja-en-0.1.0
  backbone: jhu-clsp/mmBERT-base
  backbone_revision: null
  tokenizer_revision: null
  architecture: joint-option-marker-v1
  serializer_version: serializer-v1
  validated_max_sequence_tokens: null
  supported_domains: []
  validated_languages: []

limits:
  max_body_bytes: 2097152
  max_json_depth: 32
  max_questions: 32
  min_choice_options: 2
  max_choice_options: 255
  min_score_levels: 2
  max_score_levels: 10
  max_sequence_tokens: 2048
  max_request_tokens: 32768
  overflow_policy: reject

scoring:
  confidence: normalized-entropy-v1
  probability_math_dtype: float32
  calibration_artifact: null
  require_calibration_for_production: true
  usage_semantics: expanded-input-v1

serving:
  max_queue_wait_ms_for_batch: 5
  request_deadline_seconds: 30
  result_cache_enabled: false
  allow_remote_model_download: false
  allow_jev_aliases: false
  raw_request_logging: false
```

model manifestは上記に加えて、重みhash、bundle digest、依存lockのhash、学習データ版、split hash、seed、学習設定、評価結果の場所、release_dateを持つ。未検証の言語・業務を自動で`validated_*`へ登録しない。

---

<a id="sec-18"></a>

## 18. Jevからの移行

### 18.1 変更点

正常系のアプリケーションでは、接続先、API key、model IDをJevBERT向けへ変更することを目標とする。ただし、固定したSDKでの結合試験と対象入力の制限確認を先に行う。

公式Python SDKは`base_url`、`api_key`、model設定を持つ。[S12](#source-s12) 次は**JevBERT実装後の結合試験用コード案**であり、本書作成時点で実行したコードではない。

```python
import os
from typesafe_sdk import Noul, TypeSafeClient

with TypeSafeClient(
    base_url=os.environ["JEVBERT_BASE_URL"],
    api_key=os.environ["JEVBERT_API_KEY"],
    model=os.environ["JEVBERT_MODEL"],
) as client:
    response = client.system_one(
        state="重複した引き落とし分を返金してください。",
        questions={
            "refund_requested": Noul(
                instructions="顧客は明示的に返金を求めていますか。"
            )
        },
    )
    value = response.answers["refund_requested"].noul
    assert 0.0 <= value <= 1.0
    print(response.model, value)
```

base URLのroot・path結合、環境変数の優先順位、モデル一覧の復元は固定SDKで確認する。TypeSafeの本物のAPI keyをJevBERTサーバーへ流用させない。

### 18.2 段階的切替

最初に、許可された代表入力をshadowで両方へ送り、業務動作は旧システムのままとする。差分、長さ超過、未知field、SDK例外、分岐変更を記録する。その後、JevBERTの確率でpolicy閾値を作り直し、低リスクな限定trafficへcanary適用する。

問題があれば旧bundle・旧policyへ戻せる状態を保つ。旧JevとJevBERTを暗黙に混ぜた回答を返さない。fallbackを呼び出し側で設ける場合は、実際に判定したprovider・modelを監査できるようにする。

### 18.3 Alias

`jev-latest`の受理は、既存コードの設定変更を少なくするための任意機能とする。既定は無効。有効化してもresponse.modelは実際のJevBERT IDとし、capabilitiesにaliasの対応を表示する。

未登録の`jev-*`をワイルドカードで何でも受理しない。Jevのバージョンを指定した利用者が同じモデルを実行したと誤認する挙動を避ける。

---

<a id="sec-19"></a>

## 19. 開発段階・成果物

| 段階 | 主な作業 | 成果物・終了条件 |
| --- | --- | --- |
| P0：契約固定 | schema、validator、固定logitのfake backend、SDK fixture、外部仕様の不一致整理 | 数値・構造試験が通り、ML未実装でも入出力を検証できる |
| P0.5：PoCサーバー | P0の成果に、zero-shot NLI backend（7.7節）、model registry・manifest、SDK実接続試験、PoC品質smoke評価を加え、単一マシンで常駐させる | 個人PoC用途でJev互換APIがローカルで動作する。G0と、G1のうち正常系・主要異常系を通過。G2〜G5は未達として明示。非機能要件の充足・未充足をREADMEに記載 |
| P1：学習ベースライン | pilotデータ、人手基準、A0/A1、少数例overfit、指示変更試験 | 学習・推論の対応が確認でき、少なくとも固定分類以上の追従性を測れる |
| P2：限定業務版 | データ拡充、model選定、校正、policy-dev、locked-test | G0〜G4を通過した範囲だけを明示したpreview |
| P3：移行検証 | 許可された実Jev比較、SDK結合、shadow、canary | G5通過。対象アプリでの置換条件を文書化 |
| P4：最適化・拡張 | A2、量子化、長文、多言語・業務拡張 | 新bundleごとに再学習・再校正・回帰評価 |

最初から汎用Jevの完全再現をP1の終了条件にしない。一方、固定ドメインの成功を未知業務の汎用性能へ拡張解釈しない。RL、長文自動要約、複数モデルrouterは、ベースラインが成立してから個別に判断する。

---

<a id="sec-20"></a>

## 20. 設計判断記録

| ADR | 決定 | 理由・代償 |
| --- | --- | --- |
| ADR-001 | 名称はJevBERT | BERT由来とJev互換を意図した独立プロジェクト名として使用 |
| ADR-002 | encoder-onlyの判定モデル | 生成JSONのparseに依存しない。ただし正しい判断は学習が必要 |
| ADR-003 | A0を比較基準、A1を本命 | 実装の検証可能性と候補内の計算重複削減を両立して検証 |
| ADR-004 | A2の共有state計算は後続 | 学習・実装の難度とA1との関数差を分離 |
| ADR-005 | confidenceは独自式を公開 | Jevの未確認式を推測で再現しない。既存閾値の移行は追加作業になる |
| ADR-006 | overflowは拒否 | 情報欠落・候補削除を隠さない。長文の受理範囲は狭くなる |
| ADR-007 | API名・意味・運用の互換性を別評価 | JSONが通ることだけでdrop-in互換と誤認しない |
| ADR-008 | supervised CEを最初に使用 | RLの導入前に指示条件付き分類と校正の基準を成立させる |
| ADR-009 | 質問IDはモデルへ渡さない | 呼び出し側のID変更に依存しない契約を保つ |
| ADR-010 | 新版ごとにbundle・policyを固定 | 校正やserializer変更による無言の挙動変更を防ぐ |
| ADR-011 | P0.5のserving backendは公開zero-shot NLI分類器（`bge-m3-zeroshot-v2.0`）によるA0派生 | 学習データ・学習済みモデルがない状態でend-to-endを成立させる。標準`transformers`クラス・safetensors・MITで、任意コード実行が不要。代償：候補数に比例する計算、未校正、指示追従・日本語品質は未評価。fake backendのみ（意味のある回答を返さない）と、JevBERTの学習を先行させる案（データがなくPoCの完了条件に届かない）は不採用 |
| ADR-012 | Laya・simple-jevをP0.5のbackend・コード流用元にしない | Layaは独自pipパッケージ経由でしかロードできず公式ソースが確認できない（15.3節のsupply chain方針と衝突）。モデルカード記載のzero-shot精度もほぼ偶然水準。simple-jevはLICENSE不在。比較対象としての位置づけは維持し、内部backend契約を共通にして後から追加可能にする |
| ADR-013 | 応答に`x-typesafe-request-id`を付与 | 公式SDKの`request_id`復元に必要。互換本文は変えずヘッダーのみ追加 |
| ADR-014 | モデル取得は明示的な取得スクリプトでrevision固定のうえ事前に行い、サーバーはoffline（`local_files_only`）でロード | `allow_remote_model_download: false`（17.2節）と15.3節を満たす。代償：初回セットアップが1手順増える |
| ADR-015 | P0.5の認証は設定された静的Bearer keyの定数時間比較 | 個人PoCの単一利用者を想定。利用者別rate limit（429）・tenant分離は実装しない。未充足としてREADMEに明示 |

---

<a id="sec-21"></a>

## 21. 未確認事項・リスク

### 21.1 リリース前に解決・明示する事項

| ID | 項目 | 解決方法 |
| --- | --- | --- |
| OPEN-01 | U01〜U04の実Jev挙動 | 許可されたAPI観測と固定SDKでfixture化。未確認のままなら互換対象から明示除外 |
| OPEN-02 | `confidence`具体式の公式定義 | 追加資料が得られた場合に別定義として検討。現行独自式を無断変更しない |
| OPEN-03 | 最終backbone・checkpoint・runtime | 共通データでA0/A1比較。実行できたrevisionと依存を固定 |
| OPEN-04 | 対象業務の誤りコストと品質閾値 | 低リスクな仕分けから定義し、テスト評価前に固定 |
| OPEN-05 | 校正・policyの十分な標本数 | 対象sliceごとの必要精度と誤り率に応じてデータを設計 |
| OPEN-06 | 高候補数と長文の品質 | 受理上限と検証済み品質範囲を別々に測る |
| OPEN-07 | 名前・package・配布条件 | 公開前にregistry、依存・データの条件を確認 |
| OPEN-08 | 本番ハードウェア・SLO・運用コスト | 固定環境の負荷試験と実稼働率で見積もる |
| OPEN-09 | P0.5のNLI backendにおけるhypothesisテンプレートの妥当性、Noulの2候補softmaxとScoreの順序性の品質 | PoC品質smoke評価（日本語・英語の少数fixture）で傾向を記録する。統計的な品質主張はしない。学習済みA1 bundleで解消する |
| OPEN-10 | simple-jevのライセンス、Layaの公式ソース・依存・日本語品質 | 比較対象として使う時点で再確認する |

U01（SDKが未設定`instructions`を省略すること）、SDKのURL結合・request IDヘッダー・例外対応・retry既定は、SDK 0.7.0の実装観測により解決済み（3.4節）。実Jevサーバー側の挙動はOPEN-01のまま残る。

### 21.2 主要リスク

| リスク | 兆候 | 対応 |
| --- | --- | --- |
| 指示を無視する固定分類化 | 指示を変えても回答が変わらない | 反実仮想対、未見rubric、ID除去、データ見直し |
| 未知業務での過信 | 高confidenceだが人手正解と不一致 | 対応範囲制限、domain holdout、人手回付 |
| 位置・候補数への依存 | 候補追加で大きく崩れる | 候補数別評価、完全説明の保持、再学習 |
| 校正の移転失敗 | 新言語・新業務・量子化後にNLL悪化 | 再校正、bundle更新、policyの再評価 |
| API資料の更新 | SDK型・limit・errorの変化 | 参照版を固定し、新しいcompat profileで差分試験 |
| benchmark leakage | 公開benchmarkだけ良い | group split、未見テンプレート、人手locked-test |
| 実装差分による誤判定 | yes/no逆転、Score indexずれ、候補欠落 | MLとは独立したcontract・数値試験 |

本書は、上記の不確実性を残しながら実装へ進めるための初期設計である。実装済みAPI、学習済みJevBERTモデル、Jevとの実測比較結果を提供する文書ではない。

---

<a id="sec-22"></a>

## 22. 参考資料

全て2026-09-21に公開内容を確認。API・モデルカード・リポジトリの内容は更新され得る。本書作成ではコミット単位の保存や有料APIの実行は行っていない。実装着手時に必要な資料・SDK・モデルrevisionを固定し、取得日時とhashを`compat/upstream`等へ記録する。

| ID | 一次資料 | 本書で参照した範囲 |
| --- | --- | --- |
| <a id="source-s01"></a>S01 | [TypeSafe API reference](https://docs.typesafe.ai/api) | endpoint、共通request/response、エラー、型定義 |
| <a id="source-s02"></a>S02 | [Primitives (Questions)](https://docs.typesafe.ai/primitives) | 質問型、質問ID、質問の独立性 |
| <a id="source-s03"></a>S03 | [Choice](https://docs.typesafe.ai/primitives/choice) | 候補キー・説明、全候補分布、候補上限 |
| <a id="source-s04"></a>S04 | [Score](https://docs.typesafe.ai/primitives/score) | 0始まり段階、期待値、rubric、SDKでのキー |
| <a id="source-s05"></a>S05 | [Noul](https://docs.typesafe.ai/primitives/noul) | yes/no確率、criteria、Noulの意味 |
| <a id="source-s06"></a>S06 | [Advanced: structure](https://docs.typesafe.ai/primitives/advanced) | 構造化説明、Entryのnull許容記述 |
| <a id="source-s07"></a>S07 | [Confidence](https://docs.typesafe.ai/confidence) | 分布由来の統計量、Noulには付かない点 |
| <a id="source-s08"></a>S08 | [State](https://docs.typesafe.ai/concepts/state) | stateの入力構造 |
| <a id="source-s09"></a>S09 | [Models](https://docs.typesafe.ai/models) | モデルID・alias・上限・models一覧 |
| <a id="source-s10"></a>S10 | [Python SDK: Questions](https://docs.typesafe.ai/sdk/python/api/types/questions) | instructions省略・null、NoulCriteria、Scoreの型 |
| <a id="source-s11"></a>S11 | [Python SDK: Answers and responses](https://docs.typesafe.ai/sdk/python/api/types/responses) | legendの構造化型、応答model |
| <a id="source-s12"></a>S12 | [Python SDK: Sync client](https://docs.typesafe.ai/sdk/python/api/clients/sync) | base_url等の設定、呼び出し・応答復元 |
| <a id="source-s13"></a>S13 | [ModernBERT-base model card](https://huggingface.co/answerdotai/ModernBERT-base) | BERT系encoder、英語/code、文脈長、fine-tuning |
| <a id="source-s14"></a>S14 | [mmBERT-base model card](https://huggingface.co/jhu-clsp/mmBERT-base) | ModernBERT系多言語encoderとしての候補 |
| <a id="source-s15"></a>S15 | [featherless-ai/simple-jev](https://github.com/featherless-ai/simple-jev) | 独自API差分、Laya backendの存在 |
| <a id="source-s16"></a>S16 | [Laya model card](https://huggingface.co/convaiinnovations/laya) | option-marker型実装、学習済みcheckpointの限界 |
| <a id="source-s17"></a>S17 | [Guo et al., On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) | Temperature scalingの参照 |
| <a id="source-s18"></a>S18 | [typesafe-sdk 0.7.0（PyPI配布物のソース）](https://pypi.org/project/typesafe-sdk/0.7.0/)、[typesafe-sdk-python](https://github.com/typesafe-ai/typesafe-sdk-python) | URL結合、送信ヘッダー、request ID、wire serialize、応答検証、例外対応、retry既定。2026-09-21にwheelのソースを直接確認 |
| <a id="source-s19"></a>S19 | [bge-m3-zeroshot-v2.0 model card](https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0) | P0.5 backend。revision `9abf1c8aaeb82a2447809c20753ed0b106b76652`、`XLMRobertaForSequenceClassification`、`id2label = {0: entailment, 1: not_entailment}`、MIT |
| <a id="source-s20"></a>S20 | [Python SDK: Exceptions](https://docs.typesafe.ai/sdk/python/api/exceptions)、[Retries](https://docs.typesafe.ai/sdk/python/api/retries)、[Changelog](https://docs.typesafe.ai/sdk/python/changelog) | 例外階層、RetryPolicy、0.6.0でのScore criteriaのSequence化、0.7.0でのpydantic化 |

---
<a id="sec-23"></a>

## 23. 付録A：リクエストJSON Schema

JSON Schema Draft 2020-12によるJevBERT側の構造契約である。第3章の公開資料の不一致については、第5章で決定した挙動を採用している。Jev公式schemaの転載ではない。

このschemaは最大の構造上限を表す。モデル登録、認可、bodyサイズ、入れ子深さ、token予算、実効モデル制限、重複JSONキー、非有限数値は別のvalidationで検証する。JSON parserに非有限数値や重複キーを許可した後でschemaだけに依存してはならない。

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "urn:jevbert:request:jevbert-core-2026-09-21",
  "title": "JevBERT SystemOne Request",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "model",
    "state",
    "questions"
  ],
  "properties": {
    "model": {
      "type": "string",
      "minLength": 1
    },
    "state": {
      "$ref": "#/$defs/Content"
    },
    "questions": {
      "type": "object",
      "minProperties": 1,
      "maxProperties": 32,
      "additionalProperties": {
        "$ref": "#/$defs/Question"
      }
    }
  },
  "$defs": {
    "JSONValue": {
      "anyOf": [
        {
          "type": [
            "null",
            "boolean",
            "number",
            "string"
          ]
        },
        {
          "type": "array",
          "items": {
            "$ref": "#/$defs/JSONValue"
          }
        },
        {
          "type": "object",
          "additionalProperties": {
            "$ref": "#/$defs/JSONValue"
          }
        }
      ]
    },
    "Content": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "array",
          "items": {
            "$ref": "#/$defs/JSONValue"
          }
        },
        {
          "type": "object",
          "additionalProperties": {
            "$ref": "#/$defs/JSONValue"
          }
        }
      ]
    },
    "Entry": {
      "anyOf": [
        {
          "$ref": "#/$defs/Content"
        },
        {
          "type": "null"
        }
      ]
    },
    "Noul": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "type"
      ],
      "properties": {
        "type": {
          "const": "noul"
        },
        "instructions": {
          "anyOf": [
            {
              "$ref": "#/$defs/Content"
            },
            {
              "type": "null"
            }
          ],
          "default": null
        },
        "criteria": {
          "anyOf": [
            {
              "type": "null"
            },
            {
              "type": "object",
              "additionalProperties": false,
              "properties": {
                "true": {
                  "$ref": "#/$defs/Entry"
                },
                "false": {
                  "$ref": "#/$defs/Entry"
                }
              }
            }
          ]
        }
      }
    },
    "Choice": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "type",
        "criteria"
      ],
      "properties": {
        "type": {
          "const": "choice"
        },
        "instructions": {
          "anyOf": [
            {
              "$ref": "#/$defs/Content"
            },
            {
              "type": "null"
            }
          ],
          "default": null
        },
        "criteria": {
          "type": "object",
          "minProperties": 2,
          "maxProperties": 255,
          "additionalProperties": {
            "$ref": "#/$defs/Entry"
          }
        }
      }
    },
    "Score": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "type",
        "criteria"
      ],
      "properties": {
        "type": {
          "const": "score"
        },
        "instructions": {
          "anyOf": [
            {
              "$ref": "#/$defs/Content"
            },
            {
              "type": "null"
            }
          ],
          "default": null
        },
        "criteria": {
          "type": "array",
          "minItems": 2,
          "maxItems": 10,
          "items": {
            "$ref": "#/$defs/Content"
          }
        }
      }
    },
    "Question": {
      "oneOf": [
        {
          "$ref": "#/$defs/Noul"
        },
        {
          "$ref": "#/$defs/Choice"
        },
        {
          "$ref": "#/$defs/Score"
        }
      ]
    }
  }
}
```

---

<a id="sec-24"></a>

## 24. 付録B：成功レスポンスJSON Schema

`score`の上限9は最大10段階に由来する静的上限である。実際にはその質問の`K-1`を上限として検証する。Choiceの候補集合、Scoreの連続したindex、元criteriaとの一致、分布和、期待値、confidenceの式はschema外の不変条件である。

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "urn:jevbert:response:jevbert-core-2026-09-21",
  "title": "JevBERT SystemOne Success Response",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "model",
    "answers",
    "usage"
  ],
  "properties": {
    "model": {
      "type": "string",
      "minLength": 1
    },
    "answers": {
      "type": "object",
      "minProperties": 1,
      "maxProperties": 32,
      "additionalProperties": {
        "$ref": "#/$defs/Answer"
      }
    },
    "usage": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "input_tokens",
        "output_tokens"
      ],
      "properties": {
        "input_tokens": {
          "type": "integer",
          "minimum": 0
        },
        "output_tokens": {
          "type": "integer",
          "const": 0
        }
      }
    }
  },
  "$defs": {
    "JSONValue": {
      "anyOf": [
        {
          "type": [
            "null",
            "boolean",
            "number",
            "string"
          ]
        },
        {
          "type": "array",
          "items": {
            "$ref": "#/$defs/JSONValue"
          }
        },
        {
          "type": "object",
          "additionalProperties": {
            "$ref": "#/$defs/JSONValue"
          }
        }
      ]
    },
    "Content": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "array",
          "items": {
            "$ref": "#/$defs/JSONValue"
          }
        },
        {
          "type": "object",
          "additionalProperties": {
            "$ref": "#/$defs/JSONValue"
          }
        }
      ]
    },
    "Probability": {
      "type": "number",
      "minimum": 0,
      "maximum": 1
    },
    "NoulAnswer": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "type",
        "noul"
      ],
      "properties": {
        "type": {
          "const": "noul"
        },
        "noul": {
          "$ref": "#/$defs/Probability"
        }
      }
    },
    "ChoiceAnswer": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "type",
        "choice",
        "probabilities",
        "confidence"
      ],
      "properties": {
        "type": {
          "const": "choice"
        },
        "choice": {
          "type": "string"
        },
        "probabilities": {
          "type": "object",
          "minProperties": 2,
          "maxProperties": 255,
          "additionalProperties": {
            "$ref": "#/$defs/Probability"
          }
        },
        "confidence": {
          "$ref": "#/$defs/Probability"
        }
      }
    },
    "ScoreAnswer": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "type",
        "score",
        "legend",
        "probabilities",
        "confidence"
      ],
      "properties": {
        "type": {
          "const": "score"
        },
        "score": {
          "type": "number",
          "minimum": 0,
          "maximum": 9
        },
        "legend": {
          "type": "object",
          "minProperties": 2,
          "maxProperties": 10,
          "propertyNames": {
            "pattern": "^[0-9]$"
          },
          "additionalProperties": {
            "$ref": "#/$defs/Content"
          }
        },
        "probabilities": {
          "type": "object",
          "minProperties": 2,
          "maxProperties": 10,
          "propertyNames": {
            "pattern": "^[0-9]$"
          },
          "additionalProperties": {
            "$ref": "#/$defs/Probability"
          }
        },
        "confidence": {
          "$ref": "#/$defs/Probability"
        }
      }
    },
    "Answer": {
      "oneOf": [
        {
          "$ref": "#/$defs/NoulAnswer"
        },
        {
          "$ref": "#/$defs/ChoiceAnswer"
        },
        {
          "$ref": "#/$defs/ScoreAnswer"
        }
      ]
    }
  }
}
```

---

<a id="sec-25"></a>

## 25. 付録C：数値処理の参照実装

次のコードはPython標準ライブラリのみを使う、**有効候補を抽出した後の数値処理の参照実装**である。APIサーバーや学習済みモデルの実装ではない。入力validation済みの候補mappingと組み合わせて使い、本番tensor実装はこの結果との誤差を試験する。

```python
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real


def _finite_real(value: Real, name: str) -> float:
    """Reject booleans, implicit string conversions, NaN and infinity."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def probabilities_from_logits(
    logits: Sequence[float], temperature: float = 1.0
) -> tuple[float, ...]:
    """Normalize valid-option logits only; padded options must be removed."""
    if len(logits) < 2:
        raise ValueError("At least two valid options are required")
    t = _finite_real(temperature, "temperature")
    if t <= 0.0:
        raise ValueError("temperature must be positive")
    values = tuple(_finite_real(v, "logit") for v in logits)
    maximum = max(values)
    # Center before division to avoid overflow from large positive logits.
    weights = tuple(math.exp((v - maximum) / t) for v in values)
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Invalid softmax normalizer")
    return tuple(w / total for w in weights)


def _checked_probabilities(
    probabilities: Sequence[float], tolerance: float = 1e-6
) -> tuple[float, ...]:
    if len(probabilities) < 2:
        raise ValueError("At least two probabilities are required")
    p = tuple(_finite_real(v, "probability") for v in probabilities)
    if any(v < 0.0 or v > 1.0 for v in p):
        raise ValueError("Probabilities must be in [0, 1]")
    total = math.fsum(p)
    if abs(total - 1.0) > tolerance:
        raise ValueError("Probabilities must sum to one")
    return tuple(v / total for v in p)


def normalized_entropy_confidence(probabilities: Sequence[float]) -> float:
    """Concentration statistic, not a calibrated probability of correctness."""
    p = _checked_probabilities(probabilities)
    entropy = -math.fsum(v * math.log(v) for v in p if v > 0.0)
    return min(1.0, max(0.0, 1.0 - entropy / math.log(len(p))))


def expected_score(probabilities: Sequence[float]) -> float:
    """Return the expectation of zero-based rubric indices."""
    p = _checked_probabilities(probabilities)
    if len(p) > 10:
        raise ValueError("Score supports at most ten levels")
    return math.fsum(index * value for index, value in enumerate(p))


def select_choice(probabilities: Mapping[str, float]) -> str:
    """Resolve exact ties by Unicode code point order of option keys."""
    if not all(isinstance(key, str) for key in probabilities):
        raise ValueError("Option keys must be strings")
    if not 2 <= len(probabilities) <= 255:
        raise ValueError("Choice supports two to 255 options")
    keys = sorted(probabilities)
    p = _checked_probabilities([probabilities[key] for key in keys])
    index = max(range(len(keys)), key=p.__getitem__)
    return keys[index]

```

Noulには`probabilities_from_logits([false_logit, true_logit], T)[1]`を使う。Choiceは入力候補に対応する分布を構築して`select_choice`を呼び、Scoreは段階順の分布を`expected_score`へ渡す。Noulにentropy confidenceを追加しない。

### 文書内fixtureの検算値

| 入力 | 期待値 |
| --- | --- |
| Choice分布`[0.8, 0.15, 0.05]` | confidence ≈ `0.44214218356782187` |
| Score分布`[0.1, 0.2, 0.7]` | score = `1.6`、confidence ≈ `0.27015330083790245` |
| 一様分布`[0.5, 0.5]` | confidence = `0.0` |
| 集中分布`[0.0, 1.0]` | confidence = `1.0` |
| 同率Choice `{"b":0.5,"a":0.5}` | 選択キー`"a"` |

---

## 変更履歴

| 日付 | 版 | 内容 |
| --- | --- | --- |
| 2026-09-21 | 0.1.0 | JevBERTとして初版作成。互換範囲、モデル構成、学習、校正、評価、運用、移行、schema、数値参照実装を定義 |
| 2026-09-21 | 0.2.0 | 公式SDK 0.7.0の実装観測を反映（3.3節U01・U02・U06、3.4節新設、5.7節`x-typesafe-request-id`・`X-JevBERT-Calibration`、5.9節404/405とSDK retry、13.7節SDK試験項目）。A0系のusage定義`expanded-input-a0-v1`を追加（5.8節）。P0.5（PoC）段階とzero-shot NLI backendを定義（7.1・7.2・7.7節、第19章）。Laya・simple-jevの追加調査結果と不採用理由を記録（7.6節、ADR-012）。ADR-011〜015、OPEN-09・10、S18〜S20を追加。PoCの実装設計を`docs/POC_DESIGN.md`へ分離 |
