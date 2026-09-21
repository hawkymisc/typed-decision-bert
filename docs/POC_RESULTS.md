# JevBERT P0.5（PoC）実測結果

| 項目 | 内容 |
| --- | --- |
| 文書バージョン | 1.0.0（2026-09-21） |
| 上位文書 | [仕様書 v0.3.0](../JevBERT_spec_design.md)、[POC_DESIGN v0.3.0](POC_DESIGN.md) |
| 対象 | フェーズ2（実モデル backend の実機検証）で得た**実測値** |
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
| bundle ID | `jevbert-poc-nli-ja-en-0.1.0` |
| bundle digest | `sha256:85cc98bff531a960c1226422edd3cd7828b30b5ccd46028c6d4ac49963d921f4` |
| serializer | `serializer-nli-v1+nli-template-v1` |
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

### 4.2 回帰の番人（下限）

`tests/integration/test_smoke_quality.py`（`@pytest.mark.model`）に固定した。初回実測値 ∓ 0.15。

| 型 | 初回値 | 下限 / 上限 |
| --- | --- | --- |
| Noul accuracy | 0.650 | **≧ 0.50** |
| Choice accuracy | 1.000 | **≧ 0.85** |
| Score 平均誤差 | 0.261 | **≦ 0.41** |

言語別にも下限の −0.1 を置いた（片方の言語だけ崩れる変更が平均に隠れないようにするため）。

> **この下限は、テストを通すために後から下げてはならない。** 下回った場合は原因を調べ、
> 本書に追記する。下限は品質の保証ではなく、「コンパイラ・テンプレート・escape・dtype を変えたときに
> 回答が静かに悪くなっていないか」を検出するためだけのものである。構造テストは**整形式の誤答**をすべて通してしまう。

### 4.3 この数値で言えないこと

- 校正されているか（していない。`uncalibrated`、T = 1.0）
- 実業務で使えるか（未評価。G2 未通過）
- 実 Jev との判断一致率（**一度も測っていない**。G5 未実施）
- 統計的な有意差（n = 52、区間推定なし、アノテーター 1 名、テンプレートを選んだ本人が作成）

---

## 5. 既知の未知数（K1〜K7）の決着

### 5.1 K1：RTX 5090 で動く PyTorch — 解決（フェーズ1、§12.1 D7）

`torch==2.11.0+cu128` を PyTorch 公式 index の explicit index 指定で導入。`(12, 0)` = sm_120 を認識。

### 5.2 K2：`split_special_tokens` — **効かない。escape 方式を採用**

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

---

## 7. サーバー実機検証（AC1・AC2）

### 7.1 起動と readiness

```text
uv run python -m jevbert serve --config configs/jevbert.poc.yaml
GET /readyz -> 503 {"status":"not_ready"}     # ロード中
GET /readyz -> 200 {"status":"ready"}         # 重みの SHA-256 照合 + ロード + warmup 完了後
```

### 7.2 公式 SDK からの応答（`scripts/sdk_demo.py`、抜粋）

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

### 7.4 構造化ログ（入力を含まないことの確認）

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
| AC1 | 実モデル backend でサーバーが起動し `/readyz` が 200 | **達成** | §7.1。`tests/integration/test_nli_model.py::TestSpecExampleOverHttp::test_readyz_is_two_hundred` |
| AC2 | 公式 SDK から §5.6 の例を送り I01〜I09 を満たす応答を復元 | **達成** | §7.2・§7.3 |
| AC3 | 第8章のテスト一式が GREEN | **達成** | `941 tests` 収集、`940 passed, 1 skipped`（`model` マーカー 59 件を含む） |
| AC4 | smoke 評価の結果が記録されている | **達成** | 第4章 |
| AC5 | README に非機能要件ごとの充足・部分充足・未充足と根拠 | **達成** | [README](../README.md) の「非機能要件の充足状況」（充足 13 / 部分充足 2 / 未充足 8） |
| AC6 | §14.3 の条件でのレイテンシー実測値が記録されている | **達成** | 第6章（p95 34.2 ms、目標達成） |

---

## 9. 残った懸念

1. **Noul の精度が低い（0.650、英語 0.600）。** §3.3 のとおり原因はテンプレートではなく、
   既定の英文候補と未校正のバックボーンにある可能性が高い。校正（G3）も指示追従の評価（G2）も未実施。
2. **CUDA OOM 経路が実地未検証。** 実装はあるが本環境では一度も発火していない（§5.6）。
3. **escape がモデルの見るテキストを変える。** `</s>` → `< /s>`（`compat/differences.md` L08）。
   API の型保存には影響しないが、モデル入力は元のテキストと完全一致しない。
4. **実 Jev との照合（G5）は一度も行っていない。** 互換性に関する本書・README の記述はすべて
   公開資料と SDK 実装から読み取れた範囲に基づく。
5. **レイテンシーは 1 点しか測っていない。** 同時実行・大きな Q/K・長い系列は未測定（§6）。
6. **smoke fixture は作成者が 1 名で、テンプレートを選んだ本人と同一。** 選択バイアスを排除していない。
