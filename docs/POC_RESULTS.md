# JevBERT P0.5（PoC）実測結果

| 項目 | 内容 |
| --- | --- |
| 文書バージョン | 1.1.0（2026-09-21。フェーズ2.5の実測を §4.2・§4.4・§5.8〜§5.10・§6.1・§7.4 に追記した） |
| 上位文書 | [仕様書 v0.3.0](../JevBERT_spec_design.md)、[POC_DESIGN v0.4.0](POC_DESIGN.md) |
| 対象 | フェーズ2（実モデル backend の実機検証）とフェーズ2.5（レビュー指摘の反映）で得た**実測値** |
| 計測日 | 2026-09-21 |

本書は**実測の記録**である。見込み・期待値は書かない。未達は未達と書く。

> **本書の数値を品質の主張に使ってはならない。** 第4章の smoke 評価は 52 件の自作例に対する
> 傾向の記録であり、校正の確認でもゲート G2 の根拠でもない（仕様書 §7.7・§14.1）。

---

## 1. 実行環境（実測）

| 項目 | 値 |
| --- | --- |
| OS | Windows 11 Home 10.0.26200 |
| GPU | NVIDIA GeForce RTX 5090 32 GB（sm_120 = compute capability `(12, 0)`） |
| PyTorch | `2.11.0+cu128`、`torch.cuda.is_available() == True` |
| transformers / tokenizers | `5.17.0` / `0.23.2` |
| huggingface_hub | `1.32.0` |
| Python / uv | 3.13.14 / uv 0.11.15 |
| 他プロセスの VRAM 使用 | 約 6 GB（計測中も並存。K6 参照） |

## 2. モデルと bundle

| 項目 | 値 |
| --- | --- |
| repo | `MoritzLaurer/bge-m3-zeroshot-v2.0` |
| revision（固定） | `9abf1c8aaeb82a2447809c20753ed0b106b76652` |
| 取得物 | safetensors のみ 6 ファイル、合計 **1104.1 MiB**（`*.bin` は取得しない） |
| 配置 | `models/MoritzLaurer--bge-m3-zeroshot-v2.0/`（gitignore 対象） |
| ラベル | `id2label = {0: "entailment", 1: "not_entailment"}`（名前で解決。index 0 を決め打ちしない） |
| bundle ID | `jevbert-poc-nli-ja-en-0.2.0`（フェーズ2は `…-0.1.0`。escape 規則が変わったため ID を上げた。POC_DESIGN §12.4） |
| bundle digest | `sha256:61dbb2190c473fa8925a523e28f32a1ec83df1dcbb2f52f832fdbdea0cb1d494`（フェーズ2は `sha256:85cc98bf…`） |
| backend | `a0-nli-zeroshot-v2`（フェーズ2は `…-v1`） |
| serializer | `serializer-nli-v1+nli-template-v1`（compiler は無変更なので据え置き） |
| dtype | `float32`（K3 の結果による。§5.3） |
| calibration | `uncalibrated`、T = 1.0（全型） |
| `validated_*` | 空 / `quality: "unevaluated"` |

`source_model.files` に記録した SHA-256（manifest に格納、bundle digest に含まれる）:

| ファイル | SHA-256 |
| --- | --- |
| `config.json` | `5449085fe904cdb0715398c7abf7fd049c3fc057e06353aa6a4701a6180ae4a1` |
| `model.safetensors` | `142bc710004b7552bef46b36a2f749eeb0ca198db2608445841926f78202d12c` |
| `sentencepiece.bpe.model` | `cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865` |
| `special_tokens_map.json` | `8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835` |
| `tokenizer.json` | `6710678b12670bc442b99edc952c4d996ae309a7020c1fa0096dd245c2faf790` |
| `tokenizer_config.json` | `f90024142df07163e5e6c5b9a6ad7c8c68b22a9112af11e3db4559a9ff90f737` |

バックエンドはロード時にこの 6 ファイルを照合し、1 つでも一致しなければロードに失敗する（`/readyz` は 503 のまま）。

---

## 3. K4：hypothesis テンプレートの比較と採用

`scripts/compare_templates.py` で、同一の smoke fixture 52 件を 3 案すべてに通した（実モデル、float32）。

### 3.1 候補

| ID | 指示あり | 指示なし |
| --- | --- | --- |
| `nli-template-v1` | `{I} — {C}` | `{C}` |
| `nli-template-v2` | `Question: {I} Answer: {C}` | `Answer: {C}` |
| `nli-template-v3` | `「{I}」の答えは「{C}」である。` / `The answer to "{I}" is "{C}".` | 同様の自然文 |

`nli-template-v3` の言語選択は、**その質問の指示と候補テキストに日本語の文字（ひらがな・カタカナ・CJK 統合漢字）が
1 文字でも含まれれば日本語の枠、そうでなければ英語の枠**という文字範囲判定である（言語判定モデルではない）。
決定論的で失敗しない代わりに、**コンパイル後のモデル入力が文字種に依存する**。

### 3.2 結果（実測）

| テンプレート | Noul accuracy (all / ja / en) | Choice accuracy (all / ja / en) | Score 平均誤差 (all / ja / en) | 二値誤答数 |
| --- | --- | --- | --- | --- |
| **`nli-template-v1`（採用）** | **0.650** / 0.700 / 0.600 | **1.000** / 1.000 / 1.000 | 0.261 / 0.250 / 0.271 | 7 / 36 |
| `nli-template-v2` | 0.700 / 0.700 / 0.700 | 0.938 / 0.875 / 1.000 | 0.257 / 0.259 / 0.255 | 7 / 36 |
| `nli-template-v3` | 0.750 / 0.700 / 0.800 | 0.938 / 1.000 / 0.875 | **0.240** / 0.241 / 0.239 | 6 / 36 |

Score は**誤差**なので小さいほうがよい。Noul 20 件・Choice 16 件・Score 16 件。

### 3.3 比較で判明した最も重要なこと

**3 案は二値 36 件中 32 件で完全に一致した。** 誤答が分かれたのは次の 4 件だけである。

| ケース | v1 | v2 | v3 |
| --- | --- | --- | --- |
| `noul-en-refund-no` | × | × | ○ |
| `noul-en-contact-no` | × | ○ | ○ |
| `choice-ja-route-technical` | ○ | × | ○ |
| `choice-en-feature` | ○ | ○ | × |

残りの誤答（`noul-{ja,en}-booking-no`、`noul-{ja,en}-crash-yes`、`noul-ja-refund-no`）は**3 案すべてで同じように外れた**。
つまりこの fixture では、**誤答の大半はテンプレートではなくバックボーン側の性質**である。
さらに、Noul の全ケースは `criteria` を省略しているため候補が既定の英文
（`The answer to the question is yes. / no.`）になる。Noul の弱さ（v1 で 0.650、英語 0.600）は
**テンプレートではなくこの既定候補文と未校正のバックボーン**に帰属する可能性が高い。

### 3.4 採用判断：`nli-template-v1` を維持

1. **精度では決められない。** 差は最大で二値 36 件中 1 件、Score 誤差で 0.021。n = 52 の自作 fixture で
   この差を採用根拠にするのはノイズへの当てはめである。
2. **したがって決め手は構造的コスト**とした。`nli-template-v3` は僅差で最良だが、代償として
   serializer に文字種判定が入り、**英語の質問に日本語の文字が 1 つ混ざるだけで全候補の枠が切り替わる**
   不連続を永久に抱える。解像できない差のために恒久的な構造コストを払うのは割に合わない。
3. `nli-template-v2` は文字種判定こそ不要だが、日本語の本文に英語のラベル（`Question:` / `Answer:`）を
   常に被せる。ja Choice が 3 案中唯一 0.875 に落ちたのもこの案である。
4. `nli-template-v1` は区切りが**記号のみで言語中立**であり、Choice で 16/16 と最良。

**不採用案は削除していない。** `jevbert.compiler.serializer_nli.TEMPLATES` に 3 案とも残し、
`scripts/compare_templates.py` で再実行できる。採用案は `DEFAULT_TEMPLATE` が指し、
manifest の `serializer_version`（`serializer-nli-v1+nli-template-v1`）と一致しない manifest は**起動を拒否**する。

### 3.5 今後の打ち手（本 PoC では未実施）

Noul の既定候補文（`The answer to the question is yes. / no.`）は、指示が日本語のときに
仮説が言語混在になる。§5.4 が `serializer-v1` の固定値として定めたものなので PoC では変えていないが、
Noul 精度を上げる最も筋のよいレバーはテンプレートではなくここである。

---

## 4. Smoke 評価（傾向の記録。品質の主張ではない）

`tests/fixtures/smoke/*.jsonl`、52 件（自作）。実行は `jevbert.evaluation.smoke`。

| 型 | 件数 | 構成 |
| --- | --- | --- |
| Noul | 20（ja 10 / en 10） | 同一 state で指示を変えると正解が反転する**対**のみ（10 対） |
| Choice | 16（ja 8 / en 8） | 3〜4 択。近い候補を必ず含む（`cancellation` と `plan_change` 等） |
| Score | 16（ja 8 / en 8） | 3〜5 段階（緊急度・満足度・深刻度） |

### 4.1 採用テンプレート（`nli-template-v1`、float32）での初回実測値

| 型 | 指標 | all | ja | en |
| --- | --- | --- | --- | --- |
| Noul | accuracy（閾値 0.5、高いほどよい） | **0.650** | 0.700 | 0.600 |
| Choice | accuracy（高いほどよい） | **1.000** | 1.000 | 1.000 |
| Score | 平均 `|score − 正解段階| / (K−1)`（**低い**ほどよい） | **0.261** | 0.250 | 0.271 |

### 4.2 回帰の番人（下限）— フェーズ2.5 で**厳しくした**

`tests/integration/test_smoke_quality.py`（`@pytest.mark.model`）に固定してある。

**フェーズ2 の下限（初回実測値 ∓ 0.15）は無情報予測器を検出できなかった。** 変異試験で実証:

- Noul を**定数 0.99** に固定 → fixture が 10:10 均衡なので accuracy ちょうど **0.500**、下限 `≧ 0.50` を等号で通過。
- Score を**常に中央段階**に固定 → 平均誤差 **0.375**、上限 `≦ 0.41` を通過。
- 番人として機能していたのは Choice だけだった。

そこで「質問を読まない予測器のスコア」を fixture から計算し（`uninformative_baselines()`）、
下限をその値と初回実測値の**間**に置き直した。

| 型 | 無情報基準線（実測） | 初回実測値 | 下限 / 上限（フェーズ2.5） | フェーズ2 | 向き |
| --- | --- | --- | --- | --- | --- |
| Noul accuracy | 0.500（定数回答の最良） | 0.650 | **≧ 0.55** | ≧ 0.50 | **厳しくした** |
| Choice accuracy | 0.260（一様推測と定数キーの最良） | 1.000 | **≧ 0.85** | ≧ 0.85 | 据え置き |
| Score 平均誤差 | 0.375（定数位置の最良＝中央） | 0.261 | **≦ 0.36** | ≦ 0.41 | **厳しくした** |

無情報基準線の定義: Noul は always-yes / always-no の良いほう、Choice は「一様推測 `mean 1/K`」と
「最も当たる単一キーを常に答える」の良いほう、Score は rubric 内の**固定位置**を常に答えたときの平均誤差の最小値
（`K` が case ごとに違うので正規化位置で取り、最小は目標値の中央値の位置）。
**数値は fixture から計算しており、書き写していない** — case を足せば基準線も動く。

言語別の下限は「全体の下限 − 0.1」で、Score にもフェーズ2.5 で追加した（従来は Noul と Choice だけ）。
**言語別ガードは「片方の言語だけ崩れる」ことを見るためのものであり、無情報予測器は捕まえない**
（定数予測は言語別でも 0.500 で、0.55 − 0.1 = 0.45 を通る）。それを捕まえるのは全体の下限である。

> **この下限は、テストを通すために後から下げてはならない。** 下回った場合は原因を調べ、
> 本書に追記する。下限は品質の保証ではなく、「コンパイラ・テンプレート・escape・dtype を変えたときに
> 回答が静かに悪くなっていないか」を検出するためだけのものである。構造テストは**整形式の誤答**をすべて通してしまう。
> フェーズ2.5 の変更は**厳しくする方向のみ**で、緩めた項目は無い。

### 4.3 この数値で言えないこと

- 校正されているか（していない。`uncalibrated`、T = 1.0）
- 実業務で使えるか（未評価。G2 未通過）
- 実 Jev との判断一致率（**一度も測っていない**。G5 未実施）
- 統計的な有意差（n = 52、区間推定なし、アノテーター 1 名、テンプレートを選んだ本人が作成）
- **指示追従**: Noul の否定例は 10 対すべてが「同じ state に対して**別の話題を尋ねる**」型であり、
  同じ話題のまま答えが No になる**極性反転型の hard negative が 1 件も無い**。
  したがってこの 20 件が測れているのは実質「話題が一致しているか」であって、
  「指示が問うている命題が成り立つか」ではない。Noul 0.650 という数字はこの限界の上に乗っている。

### 4.4 escape の再設計（フェーズ2.5）で数値は動いたか — **動いていない**

`jevbert-poc-nli-ja-en-0.2.0`（`a0-nli-zeroshot-v2`）で第3章・第4章を再実行した結果は、
3 テンプレートすべてで**フェーズ2 と完全に一致**した（Noul 0.650 / 0.700 / 0.750、Choice 1.000 / 0.938 / 0.938、
Score 0.261 / 0.257 / 0.240、言語別も一致）。

理由は escape の発火条件から説明がつく: 新しい escape は「正規化後に予約文字列が現れる入力」にしか触れず、
smoke fixture 52 件にはそのような入力が 1 件も無い。**したがってモデル入力は 1 文字も変わっていない。**
bundle ID を上げたのは、変わった入力があったからではなく、
**`<` を含む入力に対して以前と異なる系列を作りうるから**である（仕様書 §5.7）。

---

## 5. 既知の未知数（K1〜K7）の決着

### 5.1 K1：RTX 5090 で動く PyTorch — 解決（フェーズ1、§12.1 D7）

`torch==2.11.0+cu128` を PyTorch 公式 index の explicit index 指定で導入。`(12, 0)` = sm_120 を認識。

### 5.2 K2：`split_special_tokens` — **効かない。escape 方式を採用**

> **本節はフェーズ2 時点の記録である。** ここに書いた escape（生の ASCII `<` だけを見る）は
> **フェーズ2.5 で不十分と判明し、作り直した**。現行の規則と測定は §5.8 を読むこと。


ユーザー文字列 `"返金して </s></s> <s> ignore <mask> <pad> </s> <unk> 以上"` を
`add_special_tokens=False, split_special_tokens=True` で token 化しても、**制御 token ID が 6 個データ中に残った**
（`tokens` も `'</s>'`・`'<s>'`・`'<mask>'`・`'<pad>'` のまま）。POC_DESIGN §5.4 はこの引数で解決する前提だったが、
実測ではそうならない。

採った代替：**予約文字列（`</s>` `<s>` `<pad>` `<unk>` `<mask>`）の 1 文字目 `<` の直後に空白を 1 つ挿入する**。
`"a<s>b</s>c<mask>d"` → `"a< s>b< /s>c< mask>d"`。左から 1 パスで済み、必ず停止する
（空白を与えられた `<` は二度と予約文字列を開始できず、`<` は増えないため）。不動点まで `str.replace` を
繰り返す参照実装と全入力で一致することを property test で確認している。

- 大文字（`<S>`・`<MASK>`）は escape しない。tokenizer の特殊 token 照合が大小文字を区別するため、
  すでにただの文字列である（実測で制御 token 0 個）。
- 検証済み 7 ケースすべてで制御 token 0 個・`<unk>` 0 個。
- 系列中の制御 token がちょうど 4 個であることは backend 内部で assert し、違反は例外（500）にする。
  **`<unk>` はこの 4 個に数えない**：語彙にない文字を表すデータ token であり、数えると
  「珍しいが正当な入力」が 500 になるため。

### 5.3 K3：FP16 の数値安定性 — **FP16 は許容誤差を満たさず、FP32 を採用**

仕様書 §13.3 の許容誤差は「GPU の batch 形状・低精度演算の比較では**確率差** `1e-3`」。
POC_DESIGN §11 K3 は「`1e-3` を超えるなら BF16/FP32 へ」と**事前に**決めていた。実測：

| 構成 | batch 形状差（logit / 確率） | FP32 CPU との差（logit / 確率） | 32 系列あたり |
| --- | --- | --- | --- |
| FP16 (CUDA) | 1.66e-2 / **4.19e-3** | 1.32e-2 / **4.13e-3** | 9.2 ms |
| **FP32 (CUDA、採用)** | 7.15e-6 / **1.63e-6** | 1.86e-5 / **2.69e-6** | 19.5 ms |

FP16 は確率差 4.2e-3 で `1e-3` を**超過**した（Noul の 2 候補だけでも 3.2e-3）。FP32 は
`1e-3` どころか「同一 FP32 参照 backend」用の `1e-6` にほぼ届く水準で、代償は 32 系列あたり +10 ms。
§14.3 の目標 250 ms に対して無視できるため、**事前に決めた規則どおり FP32 に切り替えた**。

dtype は **manifest の `dtype` フィールドが決める**（`float32` / `float16` / `bfloat16` / `auto`）。
bundle digest は manifest 全体のハッシュなので、dtype を変えれば bundle ID が変わる。
CPU は常に FP32（半精度 CPU 推論は速くも参照にもならないため）。

### 5.4 K4：テンプレート — 第3章のとおり `nli-template-v1` を維持

### 5.5 K5：SDK の `strict=True` と float — 解決（フェーズ1、§12.1 D8）

`strict=True` でも float 型フィールドに int を受理する。本実装は float を必ず小数表記で出力する（保険）。

### 5.6 K6：他プロセスの VRAM（約 6 GB）との共存 — **問題なし**

FP32 で重みは約 2.2 GB。他プロセスが約 6 GB を使用したまま、モデルロード・warmup・
smoke 52 件・ベンチ 110 リクエスト・`model` マーカーのテスト 59 件をすべて OOM なしで完走した。
`max_batch_tokens` を下げる必要はなかった。CUDA OOM の経路（文脈付きログ → `empty_cache()` → 例外）は
実装済みだが、**本環境では一度も発火していないため実地未検証**である。

### 5.7 K7：uvicorn の body 上限・切断検知 — 解決（フェーズ1、CT06・CT10）

---

## 5A. フェーズ2.5 の実測

### 5.8 tokenizer の正規化と escape（S-H1）

`§5.2` の escape はフェーズ2 の時点で**不完全だった**。tokenizer は特殊 token を照合する**前に**
normalizer（`Precompiled` charsmap）を通すため、生の ASCII だけを見る escape は迂回される。
稼働中のサーバーに `state` として `＜s＞` を送ると **500 `inference_error`**
（`the encoded sequence carries 5 control tokens, expected 4`）が返っていた。

実測（実 tokenizer、`backend_tokenizer.normalizer.normalize_str`）:

| 入力 | 正規化後 | `add_special_tokens=False` の ID |
| --- | --- | --- |
| `＜s＞`（U+FF1C s U+FF1E） | `<s>` | `[6, 0]` |
| `＜/s＞` | `</s>` | `[6, 2]` |
| `＜pad＞` | `<pad>` | `[6, 1]` |
| `＜mask＞` | `<mask>` | `[6, 250001]` |
| `＜unk＞` | `<unk>` | `[6, 3]`（`<unk>` は個数検査の対象外なので**検知もされずに**モデル入力へ入っていた） |
| `<ｓ>`（全角 s） | `<s>` | `[6, 0]` |
| `<﹤s﹥>`（U+FE64 / U+FE65） | `<<s>>` | `[4426, 0, 2740]` |
| `<\x01s>`（C0 制御文字） | `<s>` | `[6, 0]` |
| `<` + `\x01`×50 + `s>` | `<s>` | `[6, 0]` |
| `<​s>`（ゼロ幅） | `< s>` | `[4426, 91, 2740]`（**元から無害**。削除ではなく空白になる） |
| `<\xads>`（soft hyphen） | `<\xads>` | 変化なし（無害） |

設計判断に使った測定:

| 測定 | 結果 |
| --- | --- |
| 正規化すると `<` を含む code point の数（全 1,112,064 点を走査） | **3 点のみ**: `<` U+003C、`﹤` U+FE64、`＜` U+FF1C。いずれも結果はちょうど `"<"` |
| 正規化して空文字列になる code point | **30 点**（C0 制御文字から `\t\n\f\r` を除いたもの＋ U+007F・U+008F・U+009F）。これが窓付き照合を不可能にする |
| charsmap と `unicodedata.NFKC` が食い違う code point | **181 点**。NFKC は C0 制御文字を削除しないため `<\x01s>` を**見逃す** |
| 1 文字ずつ正規化して連結したものと全体正規化の一致 | 結合文字で**不一致**（`<ś>`: 全体 `<ś>` / 連結 `<s` + `́` + `>`）。添字対応を取る設計は成立しない |
| normalizer の冪等性（`norm(norm(x)) == norm(x)`） | 検査した全ケースで成立。強い fallback が正規化出力を入力に使える根拠 |
| `escape_reserved` の所要時間 | 予約文字列を含まない 80 文字の文字列 1 万回で **20.9 ms**（正規化 1 回ぶんより安い。角括弧が無ければ正規化もしない）。20 万文字・角括弧のみの入力 1 本で **6.8 ms** |

採用した設計と不採用案は POC_DESIGN §12.4 に記録した。

### 5.9 `-m model` の 20 回連続実行（A-H1）

backend の共有をやめたあと、静止した作業ツリーで `uv run --no-sync pytest -q -m model` を 20 回連続実行した。

```text
run 1:  140 passed, 1047 deselected, 1 warning in 24.67s
run 2:  140 passed, 1047 deselected, 1 warning in 24.17s
...
run 19: 140 passed, 1047 deselected, 1 warning in 24.56s
run 20: 140 passed, 1047 deselected, 1 warning in 24.43s

20 回 / 20 回とも 140 passed（1047 deselected）。失敗・エラーは 0。1 回あたり 24.1〜26.2 秒。
```

（フェーズ2 の 59 件から 140 件に増えたのは、S-H1 の回帰表・fuzz・HTTP 試験と Q-M1 の順序試験を足したため。）

### 5.10 最悪系列数（S-L1）

| 項目 | 値 |
| --- | --- |
| 受理しうる最大の系列数 | 256 質問 × 255 候補 = **65,280 系列** |
| 事前上限（`max_request_tokens ÷ 5`、1 系列 = 制御 4 ＋ データ 1） | **26,214 系列** |
| フェーズ2 の挙動 | 65,280 系列すべてを token 化してから 422（約 0.44 秒を費やしていた） |
| フェーズ2.5 の挙動 | token 化の前に件数だけで 422（`compat/differences.md` R13） |

---

## 6. レイテンシー実測（AC6）

`scripts/bench_latency.py`。仕様書 §14.3 の暫定工学目標の条件に正確に合わせた：
**warm、Q=4、Choice K=8、各系列 512 token 以下、同時実行 1、HTTP end-to-end を 100 回**。

```text
warm: 10 requests; measuring 100, concurrency 1
Q=4 K=8 -> 32 sequences per request, 2096 input tokens (66 per sequence, limit 512)

  p50_ms         33.4
  p95_ms         34.2
  p99_ms         34.5
  min_ms         32.6
  max_ms         37.0
  mean_ms        33.5

spec 14.3 target p95 <= 250 ms: MET (measured p95 34.2 ms)
```

| 指標 | 実測 | §14.3 目標 | 判定 |
| --- | --- | --- | --- |
| p50 | 33.4 ms | — | — |
| **p95** | **34.2 ms** | p95 ≦ 250 ms | **達成** |
| p99 | 34.5 ms | — | — |

別プロセスで起動し直した確認実行（同条件・同 100 回）では p50 32.0 / **p95 32.8** / p99 33.2 ms だった。
上表は1回目の計測値であり、2回目も目標を達成している。**この 1.4 ms の差は実行間のばらつきであり、
コードの差ではない**（どちらも同一コミットの同一 bundle digest に対する計測）。

内訳（サーバーの構造化ログより、代表的な 1 リクエスト）:
`parse 0.13 ms` / `compile+encode 1.0 ms` / `inference 30.7 ms` / `total 32.6 ms`。
A0 系はこの条件で **32 系列**（Q=4 × K=8）を処理しているため、A1（1 質問 1 系列 = 4 系列）を
想定した目標を FP32 のまま 7 倍以上の余裕で満たしたことになる。

**この数値の限界**：同時実行 1・単一プロセス・ウォーム状態のみ。§14.3 が挙げる他の組合せ
（Q=256、K=255、系列長 2048、同時実行 8/32）は測っていない。

### 6.1 「各系列 512 token 以下」の確認方法（フェーズ2.5 で訂正）

上の実行ログにある `66 per sequence` は `usage.input_tokens ÷ 系列数`、すなわち**平均**である。
§14.3 の条件は**系列ごと**に述べられているので、平均が 512 を下回っても条件を満たしたことにならない（Q-M4）。

`scripts/bench_latency.py` は probe リクエストを 1 本足して、**最長系列の上界**を計算するようにした。
probe は候補が両方とも空文字列の Noul 1 問なので、`usage.input_tokens ÷ 2` がちょうど `premise + 4` になる。
本番 body は全系列が同じ premise を共有し候補だけが異なるので

```text
per_question  = 候補数 × (premise + 4) + Σ(候補の token 数)
最長系列       = (premise + 4) + max(候補)  ≦  (premise + 4) + Σ(候補)
```

この上界を 512 と比較する。上界が通れば最長も通る。**平均では言えないことが、上界では言える。**
計算関数は `tests/unit/test_scripts.py` で検査してある。

---

## 7. サーバー実機検証（AC1・AC2）

### 7.1 起動と readiness

```text
uv run python -m jevbert serve --config configs/jevbert.poc.yaml
GET /readyz -> 503 {"status":"not_ready"}     # ロード中
GET /readyz -> 200 {"status":"ready"}         # 重みの SHA-256 照合 + ロード + warmup 完了後
```

### 7.2 公式 SDK からの応答（`scripts/sdk_demo.py`、抜粋）

> フェーズ2（bundle `…-0.1.0`）の記録。新 bundle での再確認は §7.4。

drop-in（`base_url` と `api_key` だけ差し替え、model は SDK 既定の `jev-latest`）:

```text
  requested model : (not set -> the SDK sends its default, 'jev-latest')
  answered model  : jevbert-poc-nli-ja-en-0.1.0
```

仕様書 §5.6 の 3 型混在・日本語リクエストへの応答:

```json
{
  "model": "jevbert-poc-nli-ja-en-0.1.0",
  "answers": {
    "refund_requested": {"type": "noul", "noul": 0.6689449430817159},
    "department": {
      "type": "choice", "choice": "billing",
      "probabilities": {"billing": 0.9508810154463128, "technical": 0.006338046369070595,
                        "other": 0.04278093818461668},
      "confidence": 0.8044792449987631
    },
    "urgency": {
      "type": "score", "score": 1.9057055815509552,
      "legend": {"0": "対応期限の指定がない", "1": "数日以内の対応を求めている",
                 "2": "当日中または直ちに対応することを求めている"},
      "probabilities": {"0": 0.005583996977809088, "1": 0.0831264244934268,
                        "2": 0.9112895785287641},
      "confidence": 0.7083676845534974
    }
  },
  "usage": {"input_tokens": 476, "output_tokens": 0}
}
```

I01〜I09 はすべて OK（生出力は §7.3）。SDK 側では `legend` が `dict[int, str]` に、
`probabilities` のキーが int に復元される（SDK 0.7.0 の仕様）。

### 7.3 不変条件の検査結果（生出力）

```text
  I01 question IDs round-trip                                OK
  I02 answer types match the questions                       OK
  I03 every number finite, probabilities and confidence in [0,1] OK
  I04 distributions sum to 1 within 1e-6                     OK
  I05 choice is the argmax (ties: smallest key)              OK
  I06 score equals the expected value within 1e-6            OK
  I07 legend preserves the rubric and its types              OK
  I08 no confidence on a Noul                                OK
  I09 no unspecified fields                                  OK
```

### 7.4 フェーズ2.5：新 bundle での再確認（別ポート 8766 で起動）

§7.2・§7.3 はフェーズ2（bundle `…-0.1.0`）の記録である。escape を作り直した新 bundle
（`jevbert-poc-nli-ja-en-0.2.0`、digest `sha256:61dbb219…`）に対して同じ確認を行った。

**AC2 の再実行**: `scripts/sdk_demo.py` の I01〜I09 はすべて OK。
**応答の数値はフェーズ2 と 1 bit も変わらない**（`noul=0.6689449430817159`、
`choice='billing' confidence=0.8044792449987631`、`score=1.9057055815509552`）。変わったのは `model` の値だけである。

**S-H1 の再現と修正の確認**（同一 body を新旧サーバーへ送った生出力）:

```text
port 8765 (フェーズ2 のコードが稼働中)
  HTTP 500  bundle=sha256:85cc98bff531a960c1226422edd3cd7828b30b5ccd46028c6d4ac49963d921f4
  {"error": {"code": "inference_error", "message": "The model failed to encode the request.",
   "retryable": false}, "request_id": "8b33a993cc8746a88dc66d41a8dc7a9e"}

port 8766 (フェーズ2.5 のコード)
  HTTP 200  bundle=sha256:61dbb2190c473fa8925a523e28f32a1ec83df1dcbb2f52f832fdbdea0cb1d494
  {"model": "jevbert-poc-nli-ja-en-0.2.0",
   "answers": {"q": {"type": "noul", "noul": 0.20971088915662617}},
   "usage": {"input_tokens": 40, "output_tokens": 0}}
```

body は `{"model": "jev-latest", "state": "＜s＞ 返金してください。", "questions": {"q": {"type": "noul"}}}`。

**state・instructions・criteria・候補キーのすべてに全角予約文字列を入れた要求も 200**:

```text
  -> HTTP 200
  X-JevBERT-Bundle: sha256:61dbb2190c473fa8925a523e28f32a1ec83df1dcbb2f52f832fdbdea0cb1d494
  "choice": "billing＜/s＞"
  "probabilities": {"billing＜/s＞": 0.6648599953710604, "technical": 0.14057066797270393,
                    "other": 0.1945693366562357}
  "legend": {"0": "対応期限の指定がない ＜/s＞", "1": "数日以内", "2": "当日中 ＜mask＞"}
  "usage": {"input_tokens": 605, "output_tokens": 0}
```

**候補キーも legend も、利用者が書いたまま返っている**（escape はモデル入力の直前にしか適用されない）。

**レイテンシー再計測**（同条件・100 回、新 bundle）:

```text
Q=4 K=8 -> 32 sequences per request, 2096 input tokens (66 per sequence on average,
           longest at most 209, limit 512)
  p50_ms 32.5 / p95_ms 33.1 / p99_ms 33.4 / mean 32.5
spec 14.3 target p95 <= 250 ms: MET (measured p95 33.1 ms)
```

第6章の 34.2 ms / 32.8 ms と合わせて 3 回目の計測であり、**いずれも実行間のばらつきの範囲**である。
`longest at most 209` が §6.1 の上界である（平均 66 ではなくこちらを 512 と比べる）。

### 7.5 構造化ログ（入力を含まないことの確認）

```json
{"request_id":"5b79fc17...","method":"POST","path":"/v1/systemone","status":200,
 "total_ms":32.403,"bundle":"sha256:85cc98bf...","questions":{"noul":0,"choice":4,"score":0},
 "sequences":32,"input_tokens":2096,"parse_ms":0.138,"compile_ms":1.011,
 "inference_ms":30.666,"error_code":null}
```

state・instructions・criteria・候補キー・質問 ID・認証情報のいずれも含まない。
`tests/contract/test_logging.py` が sentinel 文字列で継続的に検査する。

---

## 8. 受入基準（§1.1 AC1〜AC6）の達成状況

| ID | 基準 | 状況 | 根拠 |
| --- | --- | --- | --- |
| AC1 | 実モデル backend でサーバーが起動し `/readyz` が 200 | **達成** | §7.1（新 bundle での再確認は §7.4）。`tests/integration/test_nli_model.py::TestSpecExampleOverHttp::test_readyz_is_two_hundred` |
| AC2 | 公式 SDK から §5.6 の例を送り I01〜I09 を満たす応答を復元 | **達成** | §7.2・§7.3、新 bundle での再実行は §7.4 |
| AC3 | 第8章のテスト一式が GREEN | **達成** | フェーズ2.5 時点で `1187 tests` 収集、`1186 passed, 1 skipped`（`model` マーカー 140 件を含む）。フェーズ2 時点は `941 tests` / `940 passed, 1 skipped`（`model` 59 件） |
| AC4 | smoke 評価の結果が記録されている | **達成** | 第4章 |
| AC5 | README に非機能要件ごとの充足・部分充足・未充足と根拠 | **達成** | [README](../README.md) の「非機能要件の充足状況」（充足 13 / 部分充足 2 / 未充足 8） |
| AC6 | §14.3 の条件でのレイテンシー実測値が記録されている | **達成** | 第6章（p95 34.2 ms、目標達成）。新 bundle での 3 回目は p95 33.1 ms（§7.4）。「各系列 512 token 以下」の確認方法は §6.1 で訂正した |

---

## 9. 残った懸念（フェーズ2.5 時点）

1. **Noul の精度が低い（0.650、英語 0.600）。** §3.3 のとおり原因はテンプレートではなく、
   既定の英文候補と未校正のバックボーンにある可能性が高い。校正（G3）も指示追従の評価（G2）も未実施。
   さらに §4.3 のとおり、**否定例が「別の話題を尋ねる」型しかないので、この 20 件は実質「話題一致」しか測れていない。**
2. **CUDA OOM 経路が実地未検証。** 実装はあるが本環境では一度も発火していない（§5.6）。
3. **escape がモデルの見るテキストを変える。** ただしフェーズ2.5 以降、変わるのは
   **正規化後に予約文字列が現れる入力だけ**である（`compat/differences.md` L08）。
   `a < b` のような普通の文は 1 文字も変わらない。API の型保存には影響しない。
4. **強い fallback escape（L09）は実入力で一度も発火していない。** 発火させるには escape を無効化する必要があり、
   テストはそうして経路を通している。**実運用で発火したときの挙動は試験でしか確認していない。**
5. **実 Jev との照合（G5）は一度も行っていない。** 互換性に関する本書・README の記述はすべて
   公開資料と SDK 実装から読み取れた範囲に基づく。
6. **レイテンシーは 1 点しか測っていない。** 同時実行・大きな Q/K・長い系列は未測定（§6）。
   tokenizer 自身は `model_max_length: 512` を宣言しており、**512 超の系列での品質は未測定**である
   （実効上限は 2,048。`compat/differences.md` R1）。
7. **smoke fixture は作成者が 1 名で、テンプレートを選んだ本人と同一。** 選択バイアスを排除していない。
   下限は無情報予測器より厳しくしたが（§4.2）、それは「番人として機能する」ことの保証であって
   「品質が十分」の保証ではない。
8. **429（rate limit）の SDK 試験は未実施。** N18 未実装で `RateLimitExceededError` を誰も送出しないため、
   観測手段が無い（POC_DESIGN §8.3）。CT12 と同種の**縮小**であり、達成ではない。
9. **escape の正しさは「`<` を生む文字は 3 つだけ」という測定に依存している。** 別の checkpoint では成り立ちうる。
   測定はテストで固定し、成り立たなかった場合は位置検査 → 強い fallback → 例外の順で受け止める設計にしてあるが、
   **その checkpoint では escape が最小変更でなくなる**（fallback が正規化形を渡すため）。
