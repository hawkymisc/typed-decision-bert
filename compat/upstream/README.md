# 参照した上流成果物の記録

本ファイルは、JevBERT の実装が**どの版の外部成果物を見て書かれたか**を固定する。版が変わったら
再確認し、本ファイルと `compat/differences.md` を更新する（仕様書 §15.3、§21.2「API資料の更新」）。

| 項目 | 値 |
| --- | --- |
| 記録日 | 2026-09-21（Asia/Tokyo） |
| 記録した段階 | P0.5（PoC）フェーズ1（API層・contract・compiler・fake backend） |
| 契約プロファイル | `jevbert-core-2026-09-21`（JevBERT 独自。TypeSafe の版番号ではない） |

## 公式 Python SDK

| 項目 | 値 |
| --- | --- |
| パッケージ | `typesafe-sdk` |
| 版 | **0.7.0**（`pyproject.toml` の dev 依存で `==` 固定、`uv.lock` でハッシュ固定） |
| 確認方法 | PyPI 配布物を `uv sync` で取得し、インストール済みソースを直接読んだうえで、実ソケット試験（`tests/sdk/`）で挙動を観測 |
| 参照範囲 | URL 結合、送信ヘッダー、request ID、wire serialize、応答検証、例外階層、retry 既定 |

SDK 0.7.0 の実装から**このフェーズで観測した**事項（仕様書 §3.4 の追補。観測であって推測ではない）:

| 観測 | 内容 |
| --- | --- |
| 応答検証 | `ConfigDict(extra="ignore", frozen=True, strict=True)` |
| 例外の status 属性 | `TypeSafeAPIError.status`。**`status_code` は存在しない**（仕様書 §3.4 の記述どおり `status`） |
| 例外の request_id | `headers["x-typesafe-request-id"]` を読む property。ヘッダー欠落時は `None` |
| 例外メッセージ | `extract_message(body)` が `error.message` を取り出し、`__str__` が `"<status> <message> (request_id=...)"` を組み立てる |
| Score の応答型 | `legend: dict[int, str \| dict \| list]`、`probabilities: dict[int, float]`。wire の文字列キーを int へ変換する |
| Noul の応答型 | `confidence` フィールドを持たない（I08 と整合） |
| RetryPolicy 既定 | `max_retries=2`、`http_statuses` に 408・429・500〜599、全体予算 `timeout=30.0` |
| クライアント引数 | `TypeSafeClient(base_url=, api_key=, model=, retry=, timeout=, headers=, transport=, http_client=)` |

## モデル

| 項目 | 値 |
| --- | --- |
| P0.5 backend のモデル | `MoritzLaurer/bge-m3-zeroshot-v2.0` |
| 固定 revision | `9abf1c8aaeb82a2447809c20753ed0b106b76652`（仕様書 S19） |
| 本フェーズでの取得 | **未取得**。フェーズ1は実モデルを使わない。`python -m jevbert fetch-model`（フェーズ2）が固定 revision を `models/` へ取得し、ファイルごとの SHA-256 を manifest に記録する（ADR-014） |
| ライセンス | MIT（モデルカード記載。取得時に再確認すること） |

**未確認**: 上表のモデル行は仕様書の記載を引き写したものであり、本フェーズでファイルを取得してハッシュを
確認してはいない。フェーズ2が実取得したうえで、revision・ファイルハッシュ・取得日時を本ファイルへ追記する。

## 実行環境（フェーズ1で実測）

| 項目 | 値 |
| --- | --- |
| OS | Windows 11 Home 10.0.26200 |
| Python | 3.13.14 |
| uv | 0.11.15 |
| PyTorch | **2.11.0+cu128**（PyTorch 公式 index `https://download.pytorch.org/whl/cu128` を explicit index 指定） |
| GPU 確認 | `torch.cuda.is_available() == True` / `NVIDIA GeForce RTX 5090` / compute capability `(12, 0)` = sm_120 |
| transformers | 5.17.0 |
| fastapi / starlette / uvicorn | 0.121.3 / 1.6.0 / 0.53.0 |
| pydantic | 2.12.5 |

## 実 Jev API

**未照合**。本プロジェクトは実 TypeSafe API へ一度も接続していない（仕様書 §13.6、OPEN-01）。
したがって、エラー本文・同率処理・未知フィールド・token 計数の**実 Jev との一致は検証されていない**。
`compat/fixtures/` に実 API 観測を置くのは、許可された認証情報とデータが用意できた段階（G5）とする。
