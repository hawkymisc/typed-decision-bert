# JevBERT P0.5（PoC）実装設計書

| 項目 | 内容 |
| --- | --- |
| 文書バージョン | 0.3.0（2026-09-21。フェーズ2の実測と互換性優先（ADR-016）を§12.3に反映し、本文の該当箇所を追従させた） |
| 上位文書 | [JevBERT 仕様・設計書 v0.3.0](../JevBERT_spec_design.md)（以下「仕様書」。`§5.3`のように節を参照する） |
| 目的 | 個人PoC用途で、Jev互換APIサーバーをこのマシン上で動作させる |
| ステータス | 実装前に固定した設計（基準線）。実装・検証で判明した差分は第12章に追記する。**実測値は[docs/POC_RESULTS.md](POC_RESULTS.md)** |

本書は仕様書の**P0.5段階（§19）**の実装設計である。仕様書と本書が矛盾する場合は仕様書を正とし、本書を直す。本書にしかない決定（テンプレート、モジュール構成、PoC制限値など）はPoC限りの決定であり、仕様書のbundle契約を変更しない。

---

## 1. ゴールと完了条件

### 1.1 完了条件（受入基準）

| ID | 基準 | 確認方法 |
| --- | --- | --- |
| AC1 | 実モデルbackendでサーバーがこのマシン上で起動し、`/readyz`が200を返す | 実プロセス起動＋HTTP確認 |
| AC2 | 公式Python SDK `typesafe-sdk==0.7.0`から仕様書§5.6の例（3型混在・日本語）を送り、I01〜I09（§5.5）を満たす応答を復元できる | `scripts/sdk_demo.py`の実行結果 |
| AC3 | 第8章のテスト一式がGREEN（fake backendの全テスト＋実モデルのintegration） | `pytest`の件数・結果 |
| AC4 | PoC品質smoke評価の結果が記録されている（合否ではなく傾向の記録） | `docs/POC_RESULTS.md` |
| AC5 | READMEに、第9章の非機能要件ごとの充足・部分充足・未充足と根拠が記載されている | README |
| AC6 | §14.3の暫定工学目標の条件でのレイテンシー実測値が記録されている（達成可否は問わない。未達なら未達と書く） | `docs/POC_RESULTS.md` |

### 1.2 スコープ

**含む**：§5の5エンドポイント、厳格なJSON検証、Bearer認証、model registry・manifest・bundle digest、fake backend、zero-shot NLI backend（§7.7）、付録Cの数値処理、応答の不変条件検査、エラー契約、過負荷（529）・deadline（504）、readiness、入力を含まない構造化ログ、capabilities、テスト、README。

**含まない**（未充足として明示する）：JevBERTの学習・A0/A1の学習済みbundle、校正（T=1固定、`uncalibrated`）、利用者別rate limit（429）、tenant分離、リクエスト間batching、metrics endpoint、TLS、JavaScript SDK試験、実Jevとの照合（G5）、G2〜G4の品質・運用ゲート。

### 1.3 前提を疑った結果（答え空間の棚卸し）

「Jev互換APIサーバーが動く」ために**JevBERTの学習は必要か**を最初に問うた。学習データも学習済みモデルも存在しないため、学習を完了条件に含めるとPoCは成立しない。一方でfake backendだけでは「APIの形」しか確認できず、個人利用のPoCとして意味のある回答が返らない。そこで、内部backend契約（§12.2）を固定したうえで、公開zero-shot NLI分類器をA0の形で差し込む案を採った（ADR-011）。学習済みA1 bundleができたらbackendを差し替えるだけでAPI層は変わらない。検討して採らなかった案はADR-011・ADR-012に理由付きで記録した。

---

## 2. 実行環境（2026-09-21に実機で確認）

| 項目 | 値 |
| --- | --- |
| OS | Windows 11 Home 10.0.26200 |
| CPU / RAM | Intel Core Ultra 7 265KF / 約64 GB |
| GPU | NVIDIA GeForce RTX 5090 32 GB（driver 595.79、確認時点で他プロセスが約6 GB使用） |
| Python | 3.13.14（`py -3.13`）、3.11も利用可 |
| パッケージ管理 | uv 0.11.15 |
| レビュー | coderabbit CLIは未インストール → サブエージェントレビュー経路 |
| git | リモートなしのローカルリポジトリ。`feature/poc-api-server`で作業 |

RTX 5090（Blackwell, sm_120）はCUDA 12.8以降でビルドされたPyTorchが必要である。PyTorchはPyTorch公式indexのCUDA 12.8以降のwheelを使う（`cu128`を第一候補とし、動かなければ新しいCUDA版indexを試す）。採用した版はlockfileで固定する。

### 2.1 技術スタック

- Python 3.13、uvプロジェクト（`pyproject.toml` + `uv.lock`、src layout、パッケージ名`jevbert`）
- API：FastAPI + uvicorn。pydantic v2、pydantic-settings
- 推論：PyTorch（CUDA）、transformers、safetensors、huggingface_hub（取得スクリプトのみ）
- 設定：YAML（PyYAML）＋環境変数
- 開発：pytest、pytest-asyncio、hypothesis、jsonschema、`typesafe-sdk==0.7.0`、ruff（`PLC0415`を有効化）

バージョンは最初に動作確認できた組み合わせを`uv.lock`で固定する（§17.1）。

---

## 3. アーキテクチャ

```text
HTTP request
  -> middleware: request ID発行、応答ヘッダー付与、構造化ログ
  -> auth: Bearer（定数時間比較、default-deny。§12.2 P-2）  401
  -> routing（/healthz・/readyz 以外は認証済みのみ到達）     404 / 405
  -> body gate: Content-Type / Content-Encoding / サイズ  415 / 413
  -> strict JSON parse: 重複キー・非有限数・不正Unicode    400
  -> contract validator: 付録A相当 + 深さ                 422 validation_error
  -> registry: model名 -> bundle（alias解決）             422 model_not_found
  -> readiness                                           503 model_unavailable
  -> encoder thread（単一。deadline・admissionの対象）
       compiler: 正規化 -> 文字数上限の事前判定           422 context_length_exceeded
                 -> 候補ごとの(premise, hypothesis)
       backend:  token IDs（制御IDとデータの分離）        422 context_length_exceeded
  -> engine: 有界queue -> 単一worker thread -> microbatch 529 / 504
  -> backend: 候補ごとの有限logit                          500 inference_error
  -> scoring: 付録C（softmax, confidence, 期待値, tie-break）
  -> response adapter: ID・候補キー・legend復元 + 不変条件検査  500
  -> 200
```

検証順序は上から固定する。認証前にbodyを解釈せず、認証前にパスの存在も明かさない。1件でも無効な質問があれば推論前に全体を拒否し、部分200を返さない（§5.9）。

### 3.1 モジュール構成

```text
src/jevbert/
  __init__.py
  __main__.py              # CLI: `python -m jevbert serve|fetch-model|init-env`
  config.py                # 設定の読み込みと検証（fail-fast）
  api/
    app.py                 # create_app(settings, registry) factory、lifespan
    routes.py              # 5エンドポイント
    auth.py
    errors.py              # JevBERTError階層、エラー本文、例外ハンドラ
    middleware.py          # request ID、応答ヘッダー、ログ
  contracts/
    strict_json.py         # 厳格JSON parser
    validator.py           # 構造検証 -> 型付きRequest（dataclass）
    response.py            # 応答構築、不変条件 I01〜I09 の検査
    schemas/request.schema.json, response.schema.json   # 付録A・Bそのもの
  compiler/
    normalize.py           # 正規JSON表現（§6.1）
    serializer_nli.py      # serializer-nli-v1（premise/hypothesisテンプレート）
    compiled.py            # CompiledQuestion等のdataclass
  backends/
    base.py                # Backend Protocol（§12.2）
    fake.py                # fake-deterministic-v1
    nli.py                 # a0-nli-zeroshot-v1
  inference/
    registry.py            # manifest、bundle digest、alias
    engine.py              # queue、worker、deadline、microbatch
  scoring/
    numeric.py             # 付録Cの参照実装（そのまま）と型別adapter
  evaluation/
    smoke.py               # PoC品質smoke評価
tests/{unit,property,contract,sdk,integration}/
compat/upstream/           # 参照したSDK版・モデルrevisionの記録
compat/differences.md      # Jevとの差分・未確認事項
configs/jevbert.poc.yaml
manifests/*.json
scripts/                   # fetch_model / run_server.ps1 / sdk_demo.py / bench_latency.py
docs/POC_DESIGN.md, POC_RESULTS.md
README.md
```

ユーザー規約により、importは全てモジュールtop-levelに置く（lazy import禁止）。`catch`での握り潰しを禁止し、例外は文脈付きでログするか`raise ... from`で伝播する。

---

## 4. API層の詳細

### 4.1 厳格JSON parser（`contracts/strict_json.py`）

入力はbytes。次の順で処理し、違反は400 `invalid_json`とする。

1. UTF-8としてstrictにdecode（BOM付きは拒否）。
2. 生テキストを1回走査し、文字列リテラル外の`[` `{`の入れ子深さを数える。上限（32）超過は**422 `validation_error`**（§4.2の表に従う。parse前に判定し、深い入れ子でPythonの再帰上限に触れない）。深さの定義：body全体のobjectを深さ1とする。
3. `json.loads`に`object_pairs_hook`（重複キー検出）、`parse_constant`（`NaN`・`Infinity`・`-Infinity`拒否）、`parse_float`（`1e999`のような非有限化する表現を拒否）を渡す。
4. 文字列（キーを含む）に孤立surrogateが含まれれば拒否する。

boolean・number・stringの間の暗黙変換はしない。整数はPython `int`、小数は`float`のまま保持する。

### 4.2 Contract validator（`contracts/validator.py`）

付録Aと同じ構造規則を、parse済みのPython値に対して手続き的に検証し、最初の違反の`path`（例：`["questions","department","criteria"]`）を返す。pydanticの暗黙変換に依存しない。付録Aのschemaファイルと実装の判定が一致することをproperty-based testで確認する（第8章 PT03）。

- **トップレベルの未知フィールドは既定で無視する**（設定`compat.unknown_top_level_fields: reject`で422に戻せる）。**Questionの未知フィールドは422**。stateや構造化説明の内部キーには未知フィールド規則を適用しない（仕様書§5.3、ADR-016）。
- `instructions`は省略とnullを同一視（SDK 0.7.0は省略する。§3.4）。
- Noul `criteria`：省略・null・`{}`・片側のみを受理。キーは`"true"`・`"false"`のみ。
- Choice `criteria`：2〜255件。値はEntry（nullは「キーだけで候補を表す」）。空文字列のキーも構造上は受理する。
- Score `criteria`：2〜10件のContent。nullの段階は422（U02）。
- 質問数1〜256（仕様書§4.2。Jevに件数の上限の記載がないため、件数だけを理由に拒否しない）。

### 4.3 認証

`Authorization: Bearer <key>`を、設定された鍵集合と`hmac.compare_digest`で比較する。比較はbytesで行う（非ASCIIのヘッダーやASCIIでない鍵で例外にならないようにするため。§12.2 P-6）。鍵は環境変数`JEVBERT_API_KEYS`（カンマ区切り）から読む。**鍵が1つも設定されていなければサーバーは起動を拒否する**。**32文字未満の鍵でも起動を拒否する**（§12.2 P-3）。無認証モードは設けない。認証はrouting の前に立つmiddlewareで全パスに適用し、`/healthz`・`/readyz`だけを明示的なallow-listとする（default-deny、§12.2 P-2）。ログ・エラー本文・設定エラー文言に鍵を出さない。鍵は設定ファイルではなく環境変数からのみ読む。

`python -m jevbert init-env`は、ランダムな鍵を生成して`.env`（gitignore対象）に書き出す。既存の`.env`は上書きしない（`O_EXCL`での排他作成。可能なら`0o600`）。

### 4.4 エンドポイント

| Path | 応答 |
| --- | --- |
| `POST /v1/systemone` | §5.4〜5.5。`Content-Type: application/json`（`charset=utf-8`可）以外は415。`Content-Encoding`が`identity`以外は415。bodyは実受信byte数で2 MiBを判定（`Content-Length`の事前判定＋読み込み中の累積判定） |
| `GET /v1/models` | `{"models":[{"name","description","release_date"}]}`。不変IDと、有効化されたaliasを含む。値はmanifest由来 |
| `GET /healthz` | 常に200 `{"status":"ok"}`。モデル詳細を返さない |
| `GET /readyz` | 全bundleがready（4.6節）なら200、それ以外は503 |
| `GET /jevbert/v1/capabilities` | 4.7節 |

**認証済みの呼び出し側に対しては**、未定義パスは404 `not_found`、非対応メソッドは405 `method_not_allowed`（Starletteの`Allow`ヘッダーを透過する）。未認証の呼び出し側にはいずれも401であり、パスの存在は明かさない（§12.2 P-2）。全ての応答（エラー含む）に§5.7のヘッダーを付ける。ただしbundleが決まる前に失敗した場合、`X-JevBERT-Bundle`・`X-JevBERT-Usage`・`X-JevBERT-Calibration`は省略する。`/readyz`の503本文は`{"status":"not_ready"}`だけで、bundleの詳細は認証付きの capabilities にのみ出す（§12.2 P-6）。

### 4.5 エラー

本文は§5.9の形式に固定する。**422に限り、Jevのschemaと同形の`detail`配列を`error`と併記する**（`[{"loc","msg","type"}]`、`loc`は`["body"] + path`で、pathが質問の内部を指し型が判別できた場合のみ質問IDの直後に型tagを挿入する。`input`・`ctx`は入力値を反射するため付けない。仕様書§5.9、ADR-016）。`path`は該当箇所が特定できる場合のみ含める。`retryable`は429・503・504・529でtrue、それ以外はfalse。529と503には`Retry-After: 1`を付ける。FastAPI・Starlette既定のエラー本文を露出させない（`RequestValidationError`・`HTTPException`・未捕捉例外の全ハンドラを登録）。未捕捉例外は500 `internal_error`とし、詳細はログにのみ出す（入力値を含めない）。

### 4.6 Readiness

起動時（lifespan）にバックグラウンドでbundleをロードする：manifest読み込み → 重み・tokenizerファイルのSHA-256照合 → モデルロード（eval mode）→ warmup推論 → 最小fixture（固定入力に対し、有限なlogitが候補数分返ること）。全て成功するまで`/readyz`は503、`/v1/systemone`は503 `model_unavailable`。失敗した場合はプロセスを落とさず503のまま原因をログする（liveness失敗と混同しない。§16.1）。

### 4.7 Capabilities

```json
{
  "contract": "jevbert-core-2026-09-21",
  "stage": "P0.5-poc",
  "bundles": [{
    "id": "jevbert-poc-nli-ja-en-0.1.0",
    "digest": "sha256:…",
    "backend": "a0-nli-zeroshot-v1",
    "serializer": "serializer-nli-v1",
    "source_model": {"repo": "MoritzLaurer/bge-m3-zeroshot-v2.0", "revision": "9abf1c8a…"},
    "calibration": {"state": "uncalibrated", "temperature": {"noul": 1.0, "choice": 1.0, "score": 1.0}},
    "confidence": "normalized-entropy-v1",
    "usage": "expanded-input-a0-v1",
    "limits": {"max_sequence_tokens": 2048, "max_request_tokens": 131072, "…": "…"},
    "validated": {"languages": [], "domains": [], "max_choice_options": "unknown", "quality": "unevaluated"}
  }],
  "aliases": {},
  "limits": {"max_body_bytes": 2097152, "max_json_depth": 32, "max_questions": 32,
             "max_request_chars": 524288, "…": "…"},
  "known_differences": ["confidence is JevBERT-specific", "usage is not Jev billing tokens", "…"]
}
```

未評価の範囲を検証済みに見せない（`validated_*`は空、`quality: "unevaluated"`）。

---

## 5. Compiler：`serializer-nli-v1`

### 5.1 正規JSON表現（`normalize.py`）

`canonical_json(value)`：objectは再帰的にキーをUnicode code point順でソート、arrayは順序保持、区切りは`","`・`":"`のコンパクト表記、`ensure_ascii=False`、文字列内容は変更しない（NFKC等なし）。数値・null・booleanと文字列の区別はJSON表記で保つ。

`render(content)`：Contentがstringならそのまま、array・objectなら`canonical_json`。

**既知の制限**：トップレベルの文字列`"[1]"`と配列`[1]`は同じ表現になる。NLIモデルへ自然文をそのまま渡す品質上の利点を優先したPoC限りの決定であり、`serializer-v1`（A1用、§6.2）には持ち込まない。`compat/differences.md`に記録する。

### 5.2 候補の展開

| 型 | 候補の内部順 | 候補テキスト |
| --- | --- | --- |
| noul | `false`, `true`（固定） | criteriaの該当側。省略・nullなら`The answer to the question is no.` / `The answer to the question is yes.`（§5.4） |
| choice | キーのUnicode code point順 | 説明がnullなら`<key>`、それ以外は`<key>: <render(説明)>` |
| score | 配列順（ソートしない） | `render(段階説明)`。段階番号は入力に含めない（§6.1） |

質問IDはどの入力にも含めない。応答のキー順は、`answers`はリクエストの質問順、Choiceの`probabilities`はリクエストのcriteria順、Scoreは`"0"`〜`"K-1"`順とする（モデル入力の順序とは独立）。

### 5.3 premise / hypothesis

- premise＝`render(state)`
- hypothesis＝指示`I = render(instructions)`（nullなら空）と候補テキスト`C`から、テンプレートで組み立てる。

初期テンプレート（`nli-template-v1`）：

```text
指示あり: "{I} — {C}"
指示なし: "{C}"
```

テンプレートは品質に直結するが事前に最適形を決められない（既知の未知数K4）。実装時に3案をsmoke評価で比較した（`scripts/compare_templates.py`）。

| ID | 指示あり | 指示なし |
| --- | --- | --- |
| `nli-template-v1`（**採用**） | `{I} — {C}` | `{C}` |
| `nli-template-v2` | `Question: {I} Answer: {C}` | `Answer: {C}` |
| `nli-template-v3` | `「{I}」の答えは「{C}」である。` / `The answer to "{I}" is "{C}".` | 同様の自然文 |

**結果：3案は二値36件中32件で一致し、fixture（n=52）では精度で優劣を決められなかった。** したがって決め手を構造的コストとし、文字種判定を必要とせず区切りが言語中立な`nli-template-v1`を維持した。比較結果と判断理由の全文はPOC_RESULTS 第3章。不採用案も`TEMPLATES`に残し、再実行可能にしてある。

採用案は`serializer_version`（`serializer-nli-v1+nli-template-v1`）として固定し、一致しないmanifestは起動を拒否する。テンプレートを変えたらbundle IDを変える（§5.7）。

### 5.4 tokenizeと制御IDの分離（§6.2、CT11）

premiseとhypothesisは**別々に**`add_special_tokens=False`でtoken化する。系列はcompilerがIDで組み立てる：

```text
[bos] + premise_ids + [eos, eos] + hypothesis_ids + [eos]      # XLM-RoBERTaのpair形式
```

**`split_special_tokens=True`は実測で効かなかった**（K2。制御token IDが6個データ中に残った。§12.3、POC_RESULTS §5.2）。代替として、token化の直前に**予約文字列（`</s>` `<s>` `<pad>` `<unk>` `<mask>`）の1文字目`<`の直後へ空白を1つ挿入する**escapeをpremiseとhypothesisの両方に適用する（`"a<s>b"` → `"a< s>b"`）。大文字（`<S>`等）はtokenizerが特殊tokenとして照合しないためescapeしない。

これにより、ユーザーデータ中の`</s>`・`<s>`・`<mask>`・`<pad>`等が制御tokenとして解釈されない。**系列中の制御token IDの個数が常にちょうど4であること**をbackend内部でassertし、違反は例外にする（CT11）。ここで数える制御tokenに`<unk>`は含めない：語彙にない文字を表すデータtokenであり、数えると珍しいが正当な入力が500になるためである。escapeによってモデルが読むテキストが元の入力と一致しなくなる点は`compat/differences.md` L08に記録した。

### 5.5 token予算

- 1系列（1候補）のtoken数 ≦ `max_sequence_tokens`（PoC既定2,048。モデル上限8,192を超える設定は起動時に拒否。manifestの`model_max_sequence_tokens`との突合で実装する。§12.2 P-5）
- リクエスト内の全系列の合計 ≦ `max_request_tokens`（PoC既定131,072。§4.2の32,768はA1の1質問1系列を前提とした値であり、候補ごとにstateを繰り返すA0系では同じ値だと実用的な質問数・候補数を受理できないため、PoC bundleの実効値として別に定める。capabilitiesで公開する）
- **tokenizeの前**に文字数の上限 `max_request_chars`（既定 `4 × max_request_tokens`）を適用する。展開後の文字数合計＝`(premise長 + instructions長) × 候補数 + Σ候補長` を質問ごとに積み上げ、超過で422 `context_length_exceeded`。長さの算術だけで判定し、巨大な文字列を組み立てない（§12.2 P-1）。capabilitiesで公開する
- manifestの上限が設定の上限を**上回る**ことは許さない（下回る＝絞り込むのは可）。上回る設定は起動時に拒否する（§12.2 P-5）
- 超過は422 `context_length_exceeded`。**切り詰めない**（`truncation=False`）。`path`は該当質問まで。
- `usage.input_tokens`＝全系列の実長（特殊token込み、padding除く）の合計（`expanded-input-a0-v1`）。

---

## 6. Backendとengine

### 6.1 Backend契約（`backends/base.py`）

```python
class Backend(Protocol):
    backend_id: str
    def load(self) -> None: ...                       # 失敗は例外。engineがready状態を管理
    def count_and_encode(self, pairs: Sequence[TextPair]) -> list[EncodedSequence]: ...
    def score(self, sequences: Sequence[EncodedSequence], cancel: CancelToken) -> list[float]:
        """系列ごとに1つの有限logitを返す。非有限値は呼び出し側が500にする。"""
```

backendは確率・confidenceを決めない（§12.1）。logit→確率は`scoring`だけが行う。

`count_and_encode`は engine の encoder スレッド（単一）から呼ばれる。A0系では1リクエストの全pairが**同一のpremise**を持つ（最大 32×255＝8,160 個）ため、**同一呼び出し内で同一premiseのtoken化は1回だけ行うべき（SHOULD）**。キャッシュは呼び出しを越えて保持しない（§15.2）。

backendの生成は factory に `BackendContext`（`models_dir`・`max_batch_tokens`・`max_batch_sequences`・`device`）を渡す形にする（§12.2 P-4）。

### 6.2 `fake-deterministic-v1`

logit＝`sha256(premise ‖ 0x1f ‖ hypothesis)`の先頭8バイトを`[-4, 4]`へ線形写像した値。質問ID・順序に依存しないので、不変性テスト（§13.3）がcompilerの欠陥を検出できる。token数は「UTF-8バイト数＋4」で代用する。テスト用に`logit_fn`をコンストラクタで差し替え可能にし、同率・極端値・NaN・個数不一致を注入する（API経由では注入できない）。fake bundleは設定で明示的に有効化したときだけ登録する（既定は無効）。

### 6.3 `a0-nli-zeroshot-v1`

- モデル：`MoritzLaurer/bge-m3-zeroshot-v2.0` @ `9abf1c8aaeb82a2447809c20753ed0b106b76652`、`AutoModelForSequenceClassification`、`trust_remote_code=False`、`local_files_only=True`、safetensorsのみ。
- entailmentのindexは`config.id2label`から名前で解決し、`entailment`が見つからなければロード失敗とする（index 0を決め打ちしない）。
- logit：2クラスのlogitを`z_ent, z_not`として`z = z_ent − z_not`（§7.7のlog-oddsと等価）。FP32へ変換してから計算する。
- 実行：`model.eval()`、`torch.inference_mode()`。deviceは`auto`（CUDAがあればCUDA）。**dtypeはmanifestの`dtype`が決める**（`float32`/`float16`/`bfloat16`/`auto`。`auto`がCUDAでFP16という当初の既定）。CPUは常にFP32。**PoC bundleは`float32`を宣言する**：FP16はK3の許容誤差を満たさなかった（§12.3、POC_RESULTS §5.3）。
- microbatch：系列を長さ順に並べ、「batch内最大長×件数 ≦ `max_batch_tokens`（既定16,384）」かつ「件数 ≦ 64」で区切る。結果は元の順へ戻す。microbatchの間で`cancel`を確認する。
- CUDA OOMは500 `inference_error`として文脈付きでログし、キャッシュを解放して次のリクエストを処理可能に保つ。

### 6.4 Engine（`inference/engine.py`）

- GPU推論は単一の専用worker thread（`ThreadPoolExecutor(max_workers=1)`）で直列実行し、event loopを塞がない。
- compileとtokenizeも**別の**単一スレッド（encoder）で実行する。tokenizerへの同時アクセスを避けるため単一であり、GPU workerと分けるのはtokenizeがGPU待ちの後ろに並ばないようにするため（§12.2 P-1）。
- 受付中＋実行中のリクエスト数が`max_pending_requests`（既定8）に達していたら529 `overloaded`。無制限に積まない。encode待ちもこの枠で数える。
- リクエストごとにdeadline（既定30秒）を設け、超過で504 `deadline_exceeded`。**encodeもこのdeadlineの対象**である。超過・切断時はcancel tokenを立て、未dispatchのmicrobatchを実行しない。実行済みの結果を別リクエストへ返さない（結果はリクエストごとのfutureにのみ結び付く）。
- warmup（§4.6）も同じ2つのworkerを経由する。loader threadから直接backendを呼ぶと、実GPU backendではserve中のリクエストと並行してdeviceに触りうる（§12.2 P-8）。
- リクエスト間のbatchingはしない（PoC）。したがって異なるリクエストが同じtensorに入ることはなく、attention・回答対応の混同は構造的に起きない。

### 6.5 Registry・manifest・bundle digest

`manifests/<bundle-id>.json`に、`public_id`、`description`、`release_date`（bundleを作成した実日付）、`backend`、`source_model`（repo・revision・ファイルごとのSHA-256）、`serializer_version`（テンプレートIDを含む）、`calibration`、`confidence`、`usage_semantics`、`dtype`、`limits`、`model_max_sequence_tokens`（省略可）、`validated_*`を持つ。**bundle digest＝manifestの`canonical_json`のSHA-256**。manifestが重み・tokenizerのhashを含むので、どれが変わってもdigestが変わる。

起動時に、manifestの`limits.*`が`settings.limits`の同名の値を上回らないこと、`limits.max_sequence_tokens`が`model_max_sequence_tokens`を上回らないことを検査し、違反は起動を拒否する（§12.2 P-5）。

`python -m jevbert fetch-model`が、固定revisionを`models/`（gitignore対象）へ取得し、ファイルhashをmanifestへ書き込む（ADR-014）。サーバーはネットワークからモデルを取得しない（`HF_HUB_OFFLINE=1`相当）。利用者が指定した任意のHF ID・URLはロードしない。

alias（例：`jev-latest`）は`allow_jev_aliases: true`かつ明示的な対応表があるときだけ有効。**ソフトウェアの既定は無効のままだが、`configs/jevbert.poc.yaml`では`jev-latest`・`jev-preview`を明示的に有効化する**（公式SDKの既定modelが`jev-latest`であり、接続先とAPI keyの変更だけで動くこと（C2）を実演するため。仕様書§18.3、ADR-016）。`response.model`は常に不変ID（§2.3）。未登録の`jev-*`をワイルドカードで受理しない。

manifestの`serializer_version`はテンプレートIDを含む（`serializer-nli-v1+nli-template-v1`）。**この build がコンパイルするテンプレートと一致しないmanifestは起動を拒否する**（bundle IDが約束するモデル入力と実際のモデル入力が食い違わないようにするため。§5.7）。

---

## 7. Scoringと応答

`scoring/numeric.py`は付録Cの関数をそのまま収録し、改変しない。型別adapter：

- Noul：`probabilities_from_logits([z_false, z_true], T)[1]`
- Choice：ソート済みキー順の分布 → `select_choice`（同率はcode point最小キー）→ `normalized_entropy_confidence`
- Score：段階順の分布 → `expected_score` → `normalized_entropy_confidence`

T（温度）はmanifestの型別値（PoCは全て1.0）。分布・score・confidenceは同じ最終確率ベクトルから計算する。wire上の数値は丸めない。JSON出力ではfloatを必ず小数表記にする（SDKは`strict=True`で検証する。§3.4）。

`contracts/response.py`は、送信前に毎回I01〜I09を検査し、違反は500 `inference_error`とする（架空の分布で成功扱いしない）。付録Bのschemaにも適合することをテストする。

### 7.1 ログ

1リクエスト1行のJSON（stderr）：`request_id`、`status`、`error_code`、`bundle`、型別質問数、系列数、`input_tokens`、各段階の所要時間（parse / compile / inference / total）。**フィールドは常に全て出力し、その段階に到達しなかったものは`null`にする**（消費側が形を1つだけ読めばよいようにするため）。`error_code`は全ての`JevBERTError`で記録し、5xxでは別途operator向けの行も出す（§12.2 P-6）。**state・instructions・criteria・候補キー・質問ID・認証情報は記録しない**（§15.2）。**例外メッセージと`exc_info`も出さない**（実tokenizer・torchの例外文言は入力テキストを含みうる。型名と`file:line`だけを出す）。uvicornのaccess logはbodyを含まないことを確認する。

---

## 8. テスト計画

TDDで進める：仕様（仕様書＋本書）からテストを先に書き、実装に合わせてテストを曲げない。fake backendのテストはGPU・モデルなしで走る。実モデルのテストは`@pytest.mark.model`とし、モデル未取得時はskip理由を明示する。

### 8.1 仕様書§13.2との対応

| ID | 内容 | PoCでの扱い |
| --- | --- | --- |
| CT01 | 3型の単独・混在、1質問・256質問・257質問 | 実施 |
| CT02 | string・array・nested objectのstate・instructions・criteria、`legend`の型保持 | 実施 |
| CT03 | Choiceのnull説明、Noulの片側・空・null criteria、instructionsの省略・null同値 | 実施 |
| CT04 | Choice 1/2/255/256、Score 1/2/10/11 | 実施（255はfake backendで） |
| CT05 | 空questions、未知type、Questionの未知フィールド、トップレベル未知フィールド（既定で無視・`reject`設定で422）、NaN、重複キー、BOM、孤立surrogate、top-levelがnull/number/booleanのstate、boolean→numberの非変換 | 実施。部分200が無いことを確認（`tests/contract/test_compat.py`） |
| CT06 | token上限の直前・一致・超過、body 2 MiBの境界、深さ32/33 | 実施。切り詰めが無いこと（受理時のtoken数＝全文のtoken数） |
| CT07 | 同率・極端logit（±1e4）・一様logit | 実施（`logit_fn`注入） |
| CT08 | NaN・inf logit、候補数とlogit数の不一致 | 実施。500で、確率を含む本文を返さない |
| CT09 | Score期待値・legend、Noulのyes/no方向 | 実施。付録Cの検算値fixtureを含む |
| CT10 | alias有効・無効、認証（欠落・不正・別scheme）、models一覧、readiness（ロード前503）、過負荷529、deadline 504 | 実施（遅いfake backendで529・504を再現） |
| CT11 | reserved文字列・偽marker | 実施（実tokenizerが必要な部分は`model`マーク） |
| CT12 | 別tenantの混在batch | **縮小**：単一tenant・リクエスト間batchingなし。並行リクエストの回答が混ざらないことのみ確認 |

### 8.2 Property-based（hypothesis）

| ID | 性質 |
| --- | --- |
| PT01 | 任意の有限logit列（K=2〜255）に対し、分布和・範囲・confidence範囲・score範囲・argmax整合（I03〜I06） |
| PT02 | 任意の有効リクエストに対する応答が付録B schemaとI01〜I09を満たす |
| PT03 | 生成したJSON値に対し、付録A schema（jsonschema）の判定とruntime validatorの構造判定が一致する |
| PT04 | 質問IDの改名、質問順の入れ替え、無関係な質問の追加、Choice criteriaの挿入順変更で、対象質問のcompiled系列と分布が完全一致する（§13.3） |
| PT05 | `canonical_json`が決定論的で、parseし直すと元の値・型に戻る |

### 8.3 SDK試験（実ソケット）

uvicornを別スレッドまたは別プロセスで空きポートに起動し、`TypeSafeClient`・`AsyncTypeSafeClient`（0.7.0）で確認する項目は仕様書§13.7のとおり。加えて、`Noul(criteria={"true": ...})`の片側指定、`Score`の構造化criteriaと`legend`の型保持、**`extra_body`で未知トップレベルフィールドを送ると既定では200で無視され`reject`設定では422になること**、**modelを指定しない（SDK既定の`jev-latest`）呼び出しがaliasで解決すること**、**422本文の`detail`配列をSDK経由で読めること（`loc[0] == "body"`）と、その併記が例外メッセージを変えないこと**、429・5xxでのSDK再試行を止めるため試験では`RetryPolicy(max_retries=0)`を指定することを含める。

### 8.4 実モデルintegrationとsmoke評価

- 実モデル・実GPUで、§5.6の例を送りI01〜I09を満たすこと、同一入力の再実行で分布が許容誤差（§13.3の`1e-3`）内であること、CT11の特殊token個数を確認する。
- **smoke評価**（`evaluation/smoke.py`、fixtureは`tests/fixtures/smoke/*.jsonl`）：自作の日本語・英語の例を型ごとに15件以上（計45件以上）。Noulは肯定・否定の対、Choiceは近い候補を含む3〜6択、Scoreは3〜5段階。指標：Noulは0.5閾値のaccuracy、Choiceはaccuracy、Scoreは`|score − 正解段階| / (K−1)`の平均。結果は傾向の記録であり、品質主張・G2の根拠にしない。
- 回帰の番人として、初回計測値より十分低い下限（例：初回値−0.15）をintegration testに置き、初回値と下限を`POC_RESULTS.md`に記録する。**下限をテストが通るように後から下げない**。下回ったら原因を調べて記録する。

### 8.5 レイテンシー計測

`scripts/bench_latency.py`：warm状態、Q=4、Choice K=8、1系列512 token以下、同時実行1で、HTTP end-to-endのp50/p95/p99を100回計測（§14.3）。A0系ではこの条件で32系列を処理するため、A1を想定した目標250 msに届かない可能性がある。結果はそのまま記録する。

---

## 9. 非機能要件の一覧（READMEの充足表の元）

READMEでは各項目を**充足／部分充足／未充足**に分類し、根拠（テスト名・計測値・未実装理由）を添える。下表の「見込み」は実装前の想定であり、READMEには実測・実装結果を書く。

> **実装後の結果は[README](../README.md)の「非機能要件の充足状況」が正である**（充足13／部分充足2／未充足8）。下表と食い違う場合はREADMEを正とし、差は§12.3に記す。実測との差はN08（充足見込み→**充足**、hash照合を実装）とN14（実測して記載→**充足**、p95 34.2 ms）の2件のみで、他は見込みどおりである。

| ID | 要件（出典） | 見込み |
| --- | --- | --- |
| N01 | 出力の型・有限性・候補対応をモデル精度から独立に検証（§4.3, I01〜I09） | 充足 |
| N02 | 数値異常時に一様分布・0.5で成功扱いしない（§4.3） | 充足 |
| N03 | 生入力・認証情報を標準ログに保存しない（§4.3, §15.2） | 充足 |
| N04 | batch化してもattention・回答対応・認可境界を混ぜない（§4.3） | 部分充足（リクエスト間batchingなし・単一tenant） |
| N05 | 無断truncationなし、overflowはreject（§6.3） | 充足 |
| N06 | 原子性：部分200なし（§5.9） | 充足 |
| N07 | ローカル推論、外部への無断転送・無断学習なし（F10, §15.2） | 充足 |
| N08 | supply chain：依存lock、revision・hash固定、任意コード実行なし、safetensors（§15.3） | 充足見込み |
| N09 | readiness/liveness分離、hash整合、warmup（§16.1） | 充足 |
| N10 | 過負荷529・deadline 504、無制限に積まない（§12.3） | 充足 |
| N11 | bundle不変ID・digest、版の追跡（F08, §5.7, §16.3） | 充足 |
| N12 | 推論cache既定無効（§12.4） | 充足（cacheなし） |
| N13 | 入力を権限にしない（§15.1） | 充足（ツール実行・外部アクセスなし） |
| N14 | レイテンシー目標 p95≦250 ms（§14.3） | 実測して記載 |
| N15 | 観測項目・metrics・drift（§16.2） | 部分充足（構造化ログのみ） |
| N16 | 校正済み確率（第11章, G3） | 未充足（`uncalibrated`） |
| N17 | 業務品質・指示追従の評価（G2） | 未充足（smoke評価のみ） |
| N18 | 利用者別rate limit 429（§5.9） | 未充足 |
| N19 | tenant分離（CT12） | 未充足（単一tenant） |
| N20 | 運用：負荷試験、OOM試験、graceful shutdown、rollback、監視、データ保持（G4） | 未充足（一部のみ） |
| N21 | 通信路保護（TLS） | 未充足（127.0.0.1 bindを既定にして緩和） |
| N22 | 実Jevとの照合・shadow（G5）、JavaScript SDK（C2後続） | 未実施 |
| N23 | GPU batch待機によるリクエスト間batching（§4.2） | 未実装 |

---

## 10. 実行方法（目標とする手順）

```text
uv sync
uv run python -m jevbert init-env            # .env にAPI keyを生成
uv run python -m jevbert fetch-model         # 固定revisionを取得しmanifestへhashを記録
uv run python -m jevbert serve --config configs/jevbert.poc.yaml   # 127.0.0.1:8765
uv run python scripts/sdk_demo.py            # 公式SDKから§5.6の例を送る
uv run pytest                                # 全テスト
```

既定のbindは`127.0.0.1`。外部公開は想定しない。

---

## 11. 既知の未知数（検証で想定外が起きうる点）

| ID | 未知数 | 起きたときの切り分け |
| --- | --- | --- |
| K1 | Python 3.13 + Windows + RTX 5090で動くPyTorch wheelの組み合わせ | `torch.cuda.is_available()`とsm_120対応を最初に確認。だめならCUDA版index・Python 3.11を順に試す |
| K2 | XLM-RoBERTa fast tokenizerで`split_special_tokens`が期待どおり働くか | 5.4節の「特殊token IDがちょうど4個」テストで検出。だめならエスケープ方式へ → **働かなかった。エスケープ方式を採用（§12.3）** |
| K3 | FP16でのlogitの安定性、batch形状による数値差 | FP32 CPU結果との差を計測。`1e-3`を超えるならBF16/FP32へ → **確率差4.2e-3で超過。FP32へ切り替えた（§12.3）** |
| K4 | hypothesisテンプレートの品質、日本語の指示＋英語の既定Noul文の混在 | smoke評価でテンプレート比較 → **3案がほぼ同等。`nli-template-v1`を維持（§12.3）** |
| K5 | SDKの`strict=True`検証が、整数値に見えるfloat（`1.0`）や指数表記を受理するか | SDK実ソケット試験で確認 |
| K6 | 他プロセスが使用中のVRAM（約6 GB）との共存 | モデルはFP16で約1.2 GB。OOM時はbatch token上限を下げる → **FP32約2.2 GBで共存。OOMは一度も発生せず（§12.3）** |
| K7 | uvicornのbody受信上限・切断検知の挙動 | CT06・CT10で確認 |

---

## 12. 実装・検証で判明した差分（実装後に追記）

### 12.1 フェーズ1（API層・contract・compiler・fake backend、2026-09-21）

実装したのは実モデル backend 以外の全部である。`backends/nli.py`、`evaluation/smoke.py`、
`scripts/`、READMEの充足表、`docs/POC_RESULTS.md` はフェーズ2に残した。

#### D1. 付録Cの検算値 `score = 1.6` は参照実装からは出ない（仕様書側の不正確）

仕様書付録Cの検算表は Score 分布 `[0.1, 0.2, 0.7]` に対し `score = 1.6` としているが、
**同じ付録Cの参照実装はこの入力に対して `1.5999999999999999`（1.6 の 1 ulp 下）を返す**。
`math.fsum(0*0.1 + 1*0.2 + 2*0.7)` の丸めによるもので、`_checked_probabilities` の再正規化とは
無関係である（`math.fsum([0.1,0.2,0.7])` はちょうど `1.0` になるため、再正規化は値を変えない）。

`1.6` は数学的な値としては正しく、参照実装は「改変禁止」であるため、**どちらも変更していない**。
テスト（`tests/unit/test_numeric.py`）は参照実装の実際の出力を固定値で pin したうえで、
1 ulp 以内であることを併記している。I06 の許容誤差は `1e-6` なので不変条件には影響しない。
仕様書 §5.6 の応答例が `"score": 1.6` である点も同様（表示上の値であり、実出力とは 1 ulp 異なりうる）。

#### D2. `serializer-nli-v1` に第2の描画衝突がある（L02、`compat/differences.md`）

POC_DESIGN §5.1 は `"[1]"` と `[1]` の衝突（L01）を既知の制限として挙げているが、
実装とテスト中に**同じ family の衝突をもう1つ**発見した。Choice 候補を `"<key>: <description>"`
で描画するため、候補 `"a"`（説明 `"b"`）と候補 `"a: b"`（説明 null）の hypothesis が一致する。
結果として両候補は同一 logit を受け取り、同率として扱われる。API の構造（キー集合・分布）は
壊れないが、モデルはこの2候補を区別できない。`compat/differences.md` の L02 に記録し、
`tests/contract/test_markers.py` で挙動を固定した（黙って変わらないようにするため）。
A1 の `serializer-v1` はキーと説明を `typed_json({name, description})` で分離するため発生しない。

#### D3. 検証順序に readiness（503）の位置を明記

§3 のパイプラインは 422 `model_not_found` の次を compiler としているが、
bundle が決まってからでないと readiness を判定できない。実装の順序は

`401 → 415 → 413 → 400 → 422(構造/深さ) → 422(model_not_found) → 503(model_unavailable) → 422(context_length_exceeded) → 推論`

とした。未登録モデルは readiness と無関係に 422 になり、登録済みだが未ロードなら 503 になる。

#### D4. モジュールを2つ追加

§3.1 のモジュール構成に対し、次の2つを追加した。いずれも責務の切り出しであり、設計変更ではない。

- `api/encoding.py`：応答 JSON の書き出し（float を必ず小数表記にする、キー順を保つ）。
  `api/errors.py` と `api/routes.py` の両方から使うため、`contracts/response.py` には置けない。
- `contracts/__init__.py` の `load_schema()`：付録A・Bの schema ファイル読み込み。

`compiler/serializer_nli.py` に `encode_request()`（tokenize と token 予算の適用）を置いた。
compiler が (premise, hypothesis) までを担当し、token 数の計数と ID 組み立ては Backend の
`count_and_encode` が持つという分担は §6.1 のとおりである。

#### D5. engine の admission 枠は worker thread 側で解放する

「`max_pending_requests` に達していたら529」を素直に `future.add_done_callback` で実装すると、
**コールバックが await の再開より後に走る**ため、逐次的な呼び出しでもキューが空なのに 529 を返す。
実測で確認したうえで、枠の解放を worker thread 内の `finally` へ移した（`threading.Lock` で保護）。
これにより「deadline で待つのをやめたリクエストの枠が、worker が実際に終わるまで解放されない」という
本来の意味（§6.4「受付中＋実行中」）も正しくなる。

#### D6. `count_and_encode` を event loop 上で呼んでいる（**フェーズ1.5 で撤回**）

token 予算の判定（422）は engine への投入（529/504）より前でなければならないため、
`count_and_encode` はリクエストハンドラ内、すなわち event loop 上で同期的に呼んでいる。
tokenizer に触るのが常に単一スレッドになるので thread safety の問題は起きない一方、
**実 tokenizer で長い入力を処理する間 event loop が塞がる**。PoC の同時実行数では許容するが、
フェーズ2でレイテンシーを実測し、必要なら専用スレッドへ移す（その場合も 422 の判定順序は保つこと）。

> **この決定はフェーズ1.5（§12.2 P-1）で撤回した。** 「PoC の同時実行数では許容する」という
> 前提が誤りで、fake backend ですら 1.6 MB body 1 件で server 全体が約3秒止まることが
> 実測で分かった（候補数×state の増幅、S-H2）。compile と encode は専用の単一スレッド
> executor へ移し、deadline と admission の対象に含めた。判定順序は §12.2 P-1 の形で保っている。

#### D7. K1（PyTorch × RTX 5090）は解決済み

`torch==2.11.0+cu128` を PyTorch 公式 index の explicit index 指定（`[[tool.uv.index]]` +
`[tool.uv.sources]`）で導入し、`uv sync` が一度で通った。実測:

```text
2.11.0+cu128 True NVIDIA GeForce RTX 5090 (12, 0)
```

compute capability `(12, 0)` = sm_120 を認識している。§11 の切り分け（別 CUDA 版 index、Python 3.11）は
不要だった。`transformers==5.17.0` も同時に導入済みで、`uv.lock` で固定した。

#### D8. K5（SDK の strict 検証と float）は「整数でも通る」

§11 K5 は「`strict=True` が整数値に見える float（`1.0`）や指数表記を受理するか」を未知数としていたが、
実測の答えは**受理する**。SDK 0.7.0 の応答モデルは `ConfigDict(extra="ignore", frozen=True, strict=True)`
だが、pydantic v2 の strict モードは float 型フィールドに対して int を許容する（python モード・
JSON モードとも）。したがって `{"confidence": 1}` でも復元できる。

とはいえ本実装は §7 の指示どおり float を必ず小数表記で出力する（指数表記は `Decimal` で位取り記法へ
展開し、丸めない）。これは SDK 0.7.0 に対しては保険であり、要求ではない。

その他 SDK 実挙動の観測は `compat/upstream/README.md` に記録した。特に、例外の status 属性は
**`status` であって `status_code` ではない**（仕様書 §3.4 の記載どおり）。

#### D9. `GET /v1/models` での alias の出し方

§4.4 は「不変IDと、有効化されたaliasを含む」とだけ書いている。実装では alias を独立した行として
並べ、`description` を `"Alias for <target>. <元の説明>"`、`release_date` を対象 bundle の値とした。
実 Jev がどう返すかは未確認である。

#### D10. fake bundle の登録方法

§6.2 の「設定で明示的に有効化したときだけ登録する（既定は無効）」を、
`enable_fake_bundle: true` のときだけ `manifests/jevbert-fake-0.0.0.json` を読み込む形で実装した。
加えて、**`bundles:` に fake の manifest を書いても `enable_fake_bundle` が false なら起動を拒否する**
（設定ミスで意味のない回答を返すサーバーが上がらないようにするため）。

---

### 12.2 フェーズ1.5（レビュー指摘の反映、2026-09-21）

フェーズ1の成果物に対し security / architecture / QA の3系統のレビューを行い、その指摘を反映した。
本節は**設計（本書 §3〜§9）から変えた点と、その理由**だけを記す。テストの増強（`verify_invariants` の
分岐テスト、構造化ログの sentinel 試験、実ソケットでの body 上限試験など）は設計を変えていないので
ここには書かない。

#### P-1. token 化を event loop から外し、安価な文字数上限を前段に置いた（D6 を撤回）

**変えた点**:

1. `limits.max_request_chars` を新設した（既定 `4 × max_request_tokens`）。backend を呼ぶ前に、
   全候補の「premise + instructions + 候補テキスト」の**文字数合計**が上限を超えたら
   422 `context_length_exceeded` で拒否する。合計は `(premise長 + instructions長) × 候補数 + Σ候補長`
   という長さの算術で求め、**巨大な文字列を作らずに判定する**。capabilities の `limits` で公開する。
2. compile と encode を**専用の単一スレッド executor**（`jevbert-encoder`）で実行する。
   GPU worker（`jevbert-inference`）とは別スレッドだが、いずれも単一である
   （tokenizer への同時アクセスを避けるため）。encode もリクエスト deadline の対象であり、
   admission（`max_pending_requests`）の枠も消費する。
3. `backends/base.py` の `count_and_encode` docstring に、**同一呼び出し内で同一 premise の
   token 化は1回だけ行うべき（SHOULD）**と明記した。fake backend は呼び出しごとの map で準拠する
   （キャッシュは呼び出しを越えて保持しない。§15.2 の記録禁止と同じ理由）。

**理由**: §5.2 の A0 展開は候補ごとに state を繰り返すため、最大 32 質問 × 255 候補 = 8,160 系列になる。
D6 はこれを「PoC の同時実行数では許容する」としていたが、fake backend（`len()` を数えるだけ）ですら
1.6 MB body 1 件で server 全体が約3秒止まった。実 tokenizer では分単位になる。さらに、テンプレートが
instructions も候補ごとに繰り返すため、**hypothesis の構築そのもの**が候補数×instructions長 の
メモリを要求する。したがって文字数判定は「hypothesis を組み立てる前」でなければ意味がない。

**検証順序**: §3 と D3 の順序は保つ。ただし encode が admission を取るようになったため、
529 が 422 `context_length_exceeded` より先に返ることはありうる。**「encode が終わった時点で
token 超過なら 422」**という意味論は変えていない（token 上限は推論より前に判定される）。
文字数上限は encode の内側（compile の直後、tokenize の前）で判定する。

#### P-2. 認証を default-deny にした

**変えた点**: 認証をハンドラごとの手動呼び出しから、routing の**前**に立つ middleware へ移した。
`/healthz` と `/readyz` だけを明示的な allow-list とする。結果として **401 が 404・405 より先に返る**。

**理由**: 手動呼び出しは、新しいルートを足したときに fail-open する。また 404/405 が認証より先に返ると、
未認証の呼び出し側がパスの存在を列挙できる。§4.4 の「未定義パスは404、非対応メソッドは405」は
**認証済みの呼び出し側に対して**の話であると読み替える。

#### P-3. API key の最小長を 32 文字にした

**変えた点**: 32 文字未満の鍵は起動を拒否する（`init-env` が生成する `secrets.token_urlsafe(32)` は
43 文字なので余裕がある）。鍵の検証は pydantic の外で行い、`Settings.api_keys` は `repr=False` にした。
YAML に `api_keys` を書くことも拒否する（鍵は環境変数のみ）。

**理由**: §4.3 は「鍵が1つも設定されていなければ起動を拒否する」としか書いていなかったため、
`k` 1文字でも起動できた。また pydantic の検証エラーメッセージは**入力値をそのまま引用する**ので、
鍵の検証を pydantic に任せると起動失敗時のログに鍵が出る。

#### P-4. backend factory に `BackendContext` を渡す

**変えた点**: factory の型を `(BundleManifest, Path) -> Backend` から
`(BundleManifest, BackendContext) -> Backend` に変えた。`BackendContext` は
`models_dir` / `max_batch_tokens` / `max_batch_sequences` / `device` を持つ。
`serving.device`（`auto` / `cpu` / `cuda`）を新設した。

**理由**: §6.3 の microbatch 上限（`max_batch_tokens`、`max_batch_sequences`）は設定に存在したが
**誰も読んでいなかった**。フェーズ2が `backends/nli.py` と registry の1エントリ追加だけで済むように、
backend が必要とする設定の受け渡し口を先に作った。

#### P-5. manifest は設定の上限を広げられない

**変えた点**: `manifest.limits.max_sequence_tokens` / `max_request_tokens` が
`settings.limits` の同名の値を超えていたら起動を拒否する。manifest に
`model_max_sequence_tokens`（省略可）を追加し、`limits.max_sequence_tokens` がそれを超えていたら
起動を拒否する。

**理由**: リクエスト経路が実際に適用するのは bundle 側の上限なので、manifest が設定より大きい値を
宣言すると、運用者が設定した上限を黙って引き上げられる。後者は §5.5 の
「モデル上限（8,192）を超える設定は起動時に拒否」の実装であり、超過をリクエストごとの実行時失敗ではなく
起動時の拒否にするためのもの。

#### P-6. エラーと観測の細かい修正

| 項目 | 変えた点 | 理由 |
| --- | --- | --- |
| `InferenceCancelled` | 503 ではなく **504 `deadline_exceeded`** にした | ジョブを取り消すのは deadline か切断だけであり、§5.9 の対応する code は 504 |
| 405 | Starlette の `Allow` ヘッダーを透過する | 405 が伝えるべき唯一の情報だった |
| 契約外の HTTP status | 素通しせず 500 `internal_error` にしてログする | §5.9 に無い code を wire に発明しない |
| access ログ | 全フィールドを常に出力し（未到達の段階は `null`）、`error_code` を必ず含める。5xx では別途 operator 向けの行を出す | NaN logit や不変条件違反の 500 は本文が意図的に空なので、ログが唯一の証跡 |
| 例外ログ | 例外メッセージと `exc_info` を出さず、型名と `file:line` を出す | 実 tokenizer / torch の例外文言は入力テキストを含みうる（§15.2） |
| `/readyz` の 503 本文 | `{"status":"not_ready"}` のみ | 無認証で bundle ID と状態を開示していた。詳細は認証付き capabilities へ |
| 重複キーの 400 本文 | キー名を引用しない | 呼び出し側のデータの反射 |
| `.env` 生成 | `O_EXCL` + `0o600` で排他作成 | `exists()` 確認と書き込みの間の競合で運用者の鍵を上書きしうる |

#### P-7. 空文字列の `instructions` は「指示なし」

**変えた点**: `render(instructions)` が空文字列になる場合、テンプレートの「指示なし」側を使う。
`[]` / `{}` はそれぞれ `"[]"` / `"{}"` に描画されるので従来どおり指示として扱う。

**理由**: §5.3 のテンプレート `"{I} — {C}"` に空の `I` を入れると、全候補が `" — "` で始まる。
モデルから見ればこれは意味のある区切りに見える。`compat/differences.md` の L06 に記録した。

#### P-8. warmup を engine の worker 経由にした

**変えた点**: `Bundle.load()` は engine を受け取り、warmup の `count_and_encode` を encoder スレッド、
`score` を inference worker で実行する。fake backend は warmup の premise に対して
`delay_seconds` を適用しない。

**理由**: §4.6 の warmup は loader スレッドから直接 backend を呼んでいた。実 GPU backend では
serve 中のリクエストと並行して device に触りうる（§6.4 の「単一 worker で直列実行」の目的に反する）。
後者は副産物で、テストスイートの実行時間が約9秒縮んだ。

#### P-9. 応答の不変条件の穴を塞いだ

**変えた点**: `verify_invariants` に次を追加した。(a) `confidence` を分布から再計算して照合する、
(b) `score` の float 性・有限性・`0 ≤ score ≤ K-1` を検査する、
(c) トップレベルのキー集合が `{"model","answers","usage"}` であることと `model` が非空文字列であること、
`usage` のキー集合を検査する。

**理由**: `verify_invariants` を no-op にしてもフェーズ1の 499 件が全て通った。
特に `confidence` は**他のどの不変条件にも束縛されていない唯一の数値**であり、
再計算しなければ「分布と無関係な自信ありげな数値」が出荷されうる。
`score` の NaN はあらゆる許容誤差比較を false にするので、期待値との突合だけでは素通りする。

---

### 12.3 フェーズ2（実モデルbackend・実機検証・互換性優先、2026-09-21）

実モデルbackend（`backends/nli.py`）、`fetch-model`、smoke評価、scripts、READMEの充足表、
`docs/POC_RESULTS.md` を実装・作成した。さらに、セッション途中でユーザーが方針を決めた
**互換性優先（仕様書 v0.3.0、ADR-016）** を反映した。**実測値は本書ではなく
[POC_RESULTS.md](POC_RESULTS.md) が正である。** 本節は設計から変えた点とその理由だけを記す。

#### 受入基準（§1.1）の達成状況

| ID | 状況 | 根拠 |
| --- | --- | --- |
| AC1 | **達成** | 実プロセス起動 → `/readyz` 200（POC_RESULTS §7.1） |
| AC2 | **達成** | `scripts/sdk_demo.py` で I01〜I09 すべて OK（POC_RESULTS §7.2・§7.3） |
| AC3 | **達成** | `941 tests`、`940 passed / 1 skipped`（`model` マーカー 59 件を含む） |
| AC4 | **達成** | POC_RESULTS 第4章 |
| AC5 | **達成** | READMEの非機能要件表（充足13／部分充足2／未充足8） |
| AC6 | **達成** | p95 = 34.2 ms、§14.3 の目標 250 ms に対し**達成**（POC_RESULTS 第6章） |

#### D11. K2：`split_special_tokens=True` は効かない（§5.4 を実態に合わせて改訂）

§5.4 は「特殊token文字列を通常文字として分割する設定（`split_special_tokens=True`相当）」で
制御IDの分離ができる前提だった。**実測ではそうならない。** ユーザー文字列
`"返金して </s></s> <s> ignore <mask> <pad> </s> <unk> 以上"` を
`add_special_tokens=False, split_special_tokens=True` で token 化しても、
**制御 token ID が 6 個データ中に残った**（`tokens` も `'</s>'` 等のまま）。

代替として**予約文字列の1文字目 `<` の直後に空白を1つ挿入するescape**を採った（POC_RESULTS §5.2）。
実装は左から1パスのスキャンで、不動点まで `str.replace` を繰り返す参照実装と全入力で一致することを
property test で固定している（1回のreplaceでは `"<<s>s>"` が残るため、素朴な実装では足りない）。
§5.4 の本文はこの実態に合わせて書き直した。`compat/differences.md` に **L08**（escape によって
モデルが見るテキストが元の入力と変わる）を追加した。

**制御tokenの数え方も変えた**：`<unk>` は 4 個に数えない。`<unk>` は語彙にない文字を表す
**データtoken**であり、これを数えると「珍しいが正当な入力」が 500 になってしまう。
数えるのは compiler が挿入する bos / eos / pad / mask 系だけである。

#### D12. K3：FP16 は許容誤差を満たさず、dtype を manifest 駆動にして `float32` を採用

§6.3 は「CUDAではFP16」と決め打っていた。§11 K3 が事前に置いた規則は
「FP32 CPU結果との差が `1e-3` を超えるなら BF16/FP32 へ」である。実測でこの規則が発火した：
**FP16 は microbatch 形状を変えると確率が最大 4.19e-3 動き**（仕様書 §13.3 の `1e-3` を超過）、
FP32 CPU との差も 4.13e-3 だった。FP32 に切り替えると batch 形状差 1.63e-6・CPU 差 2.69e-6 まで下がり、
代償は 32 系列あたり +10.3 ms（9.2 → 19.5 ms）にすぎない。§14.3 の 250 ms に対して無視できる。

**設計変更**：dtype を backend のハードコードから **manifest の `dtype` フィールド**へ移した
（`float32`/`float16`/`bfloat16`/`auto`。`auto` が §6.3 当初の「CUDAでFP16」）。
bundle digest は manifest 全体のハッシュなので、dtype を変えれば bundle ID が変わる（§5.7 の要求どおり）。
CPUは宣言に関わらず常に FP32（半精度CPU推論は速くも参照にもならないため）。

#### D13. K4：テンプレートは精度で決められなかった（§5.3 に結果を反映）

3案の比較で**二値36件中32件が一致**した。差は最大で1件、Score誤差で 0.021 であり、
n=52 の自作 fixture ではノイズと区別できない。したがって採用は精度ではなく**構造的コスト**で決め、
`nli-template-v1` を維持した。`nli-template-v3`（僅差で最良）を採らなかったのは、
serializer に文字種判定が入り「英語の質問に日本語の文字が1つ混ざるだけで全候補の枠が切り替わる」
不連続を恒久的に抱えるためである。判断の全文と不採用理由は POC_RESULTS 第3章。

副産物として分かったこと：**Noul の誤答は3案でほぼ共通**しており、原因はテンプレートではなく
既定の英文候補（`The answer to the question is yes. / no.`）と未校正のバックボーンにある可能性が高い。

#### D14. K6：VRAM 共存は問題なし。ただし CUDA OOM 経路は実地未検証

他プロセスが約6 GBを使用したまま、FP32（重み約2.2 GB）でロード・warmup・smoke 52件・
ベンチ110リクエスト・実モデルテスト59件をすべて完走した。`max_batch_tokens` を下げる必要はなかった。
**CUDA OOM の処理経路（文脈付きログ → `empty_cache()` → 例外 → 500）は実装済みだが、
本環境では一度も発火していない。** 実地未検証として README N20 に記した。

#### D15. D6（`count_and_encode` を event loop 上で呼ぶ）は撤回済みのままで正しかった

フェーズ1.5（§12.2 P-1）で encode を専用スレッドへ移した判断は、実 tokenizer で裏付けられた。
実測の compile+encode は Q=4・K=8 で約1.0 ms と軽いが、これは escape と token 化が
**同一 premise を1回だけ**処理しているからである（`backends/base.py` の SHOULD）。
32系列すべてで state を token 化していれば同じ仕事を32回していた。

#### D16. 互換性優先（ADR-016）で変えた4点

ユーザーの方針決定（「実ユースケースが未定の現段階では、Jev API との互換性が最も強く問われる」）に伴い、
仕様書が v0.3.0 へ改訂された。本書の §4.2・§4.5・§6.5・§8.1・§8.3 を追従させたうえで、次を実装した。

| # | 変えた点 | 理由と代償 |
| --- | --- | --- |
| 1 | **トップレベルの未知フィールドを既定で無視**する（`compat.unknown_top_level_fields: ignore`、`reject`で従来どおり422） | SDK の `extra_body` が将来のフィールドを送れるため、形式だけを理由に拒否しない。**代償：フィールド名の誤記に気付けない。** 付録Aのトップレベルを `additionalProperties: true` にし、PT03（schemaとvalidatorの一致）は既定の `ignore` で一致させた。`reject` は「公開schemaより狭い」方向なので、運用者が明示的に選んだときだけ起きる |
| 2 | **422本文に `detail` 配列を併記** | SDK の wire schema（Jev の OpenAPI 由来）が 422 を `{"detail":[{"loc","msg","type"}]}` と定義するため、`detail` を直接読むクライアントが動く。`loc` は `["body"] + path`、質問の内部を指し型が判別できた場合のみ質問IDの直後に型tagを挿入。**`input`・`ctx` は付けない**（入力値の反射になる。§15.2）。`msg` は `error.message` と同値なので SDK の例外メッセージは変わらない（実ソケットで確認） |
| 3 | **質問数上限 32 → 256** | Jev に件数上限の記載がない。実効上限は token 予算が決める。付録A・B の `maxProperties`、response schema、CT01 の境界（256/257）、capabilities を同期した |
| 4 | **PoC設定で `jev-latest`・`jev-preview` を有効化** | SDK の既定 model が `jev-latest` であり、**接続先とAPI keyだけ差し替えて動く**ことがC2の実演になる。ソフトウェア既定は無効のまま。`response.model` は不変ID（D01） |

`compat/differences.md` は互換性の観点で構成し直し、冒頭に
**「Jev で通る要求が JevBERT で形式理由により拒否されうる箇所」（R1〜R12）**の一覧を置いた。

#### D17. 設計に無かった追加

- **`src/jevbert/fetch.py`**：§3.1 のモジュール構成にない。`fetch-model` の実装（固定revisionの取得、
  safetensorsのみのallow-list、SHA-256のmanifestへの記録）を CLI から分離した。
  記録済みhashは**上書きしない**。ローカルファイルが manifest と食い違う場合は
  **再取得せずエラーで知らせる**（黙って直すと、hashが存在する理由そのものを潰すため）。
- **`src/jevbert/evaluation/smoke.py` に `pipeline_answerer`**：smoke 評価をサーバー自身の
  compile → encode → score → build 経路で走らせる。HTTPだけが無い。
  smoke の数値が endpoint の返す値より良くなり得ないようにするため。
- **capabilities のバグ修正**：`serializer_version` がテンプレートIDを含むようになった結果、
  `manifest.serializer_version == SERIALIZER_VERSION` という比較が成立しなくなり
  `"template": null` を公開していた。`template_id_of()` に置き換えた。

#### D18. 残った懸念

POC_RESULTS 第9章に記載した。要点は、Noul精度が低いこと（原因は既定候補文の可能性が高い）、
CUDA OOM 経路が実地未検証であること、escape がモデル入力のテキストを変えること、
**実 Jev との照合（G5）を一度も行っていないこと**、レイテンシーを1点しか測っていないこと。
