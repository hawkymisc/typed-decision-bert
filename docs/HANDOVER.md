# Handover — JevBERT P0.5（PoC）

| 項目 | 内容 |
| --- | --- |
| 日付 | 2026-09-21 |
| ブランチ | `feature/poc-api-server`（`main` は原本仕様書の1 commitのみ。未マージ） |
| リモート | なし。そのため GitHub Issue ではなく本ファイルを引き継ぎ文書とする |
| 状態 | P0.5 完了。サーバーは `127.0.0.1:8765` で稼働（セッションから切り離して起動。OS 再起動では戻らない） |

## 1. 到達点

完了条件3つを実地で確認済み。

1. **Jev 互換 API サーバーがこのマシンで動作** — 公式 SDK `typesafe-sdk==0.7.0` から、接続先と API key の差し替えだけで動く（`model` 無指定＝SDK 既定の `jev-latest` のまま）。3型混在・日本語・全角の予約文字列入りの入力・`extra_body`・422 の `detail`・401・models 一覧を確認。
2. **機能試験** — `pytest` 1186 passed / 1 skipped（実モデル 140 件込み、`JEVBERT_REQUIRE_MODEL=1`）。`ruff check` clean。実モデルテストは 20 回連続で全パス。レビューは security / architecture / QA の3観点を2巡し、変異試験で「歯のないテスト」を潰した。
3. **README** — 非機能要件 N01〜N23 を充足13 / 部分充足2 / 未充足8 に分類し根拠を記載。「互換性の現状」は SDK 実証済み / schema からの推定 / 実 Jev 未検証の3区分。

詳細：[POC_DESIGN.md](POC_DESIGN.md)（設計と §12 の経緯）、[POC_RESULTS.md](POC_RESULTS.md)（実測値）、[../compat/differences.md](../compat/differences.md)（Jev との差分）。

## 2. これは何でないか

- backend は学習済み JevBERT ではなく、公開 zero-shot NLI 分類器（`MoritzLaurer/bge-m3-zeroshot-v2.0`、revision 固定、FP32）を A0 形式で使う**暫定版**（ADR-011）。
- 確率は**未校正**（`uncalibrated`、T=1）。品質は smoke 評価 52 件のみ：Choice 1.00 / Noul 0.65 / Score 誤差 0.26。Noul は実質「話題一致」しか測れておらず、否定・条件付きの指示には弱い見込み。
- **実 Jev サーバーとの照合（G5）は一度も行っていない。** 互換性の根拠は公式 docs と SDK 0.7.0 の実装・wire schema だけである。

## 3. 方針（PO 判断）

実ユースケースが未定のあいだは **Jev API との互換性が最優先**（ADR-016）。「Jev が受理し得る要求を形式理由で拒否しない」「エラー外形は Jev に寄せる」。外部から見える挙動を変える提案は、まず Jev の2エンドポイント（`/v1/systemone`・`/v1/models`）の見え方が変わるかを確認する。

## 4. 次の一手（未着手・PO の GO 待ち）

| 候補 | 内容 | 備考 |
| --- | --- | --- |
| A：お手軽な追加学習 | 公開データセットの Jev 形式変換＋LLM 合成データ（否定・条件・Score の順序・日本語）で、NLI checkpoint を A0 のまま継続 fine-tune。あわせて温度校正 | 推奨。半日規模の見込み。現在の smoke 値が改善前の基準になる。合成データの利用規約と仕様書 §9.5 の承認が前提 |
| B：実 Jev との照合（G5） | TypeSafe の API key で十数リクエストを投げ、401/429/5xx の本文外形、未知フィールドの扱い、同率時の選択などを fixture 化 | key の発行・設定は PO の作業。課金はごく少額（入力 $0.042/1M token） |
| C：リモート作成と PR | GitHub リポジトリを作り `feature/poc-api-server` を PR にする | 公開範囲（private/public）の判断が必要。public 化は不可逆 |
| D：`-c` 版モデルへの差し替え検討 | 現モデルは学習データに非商用ライセンスのものを含む。外へ出す可能性があるなら商用可データのみの `…-v2.0-c` を検討 | 個人 PoC のあいだは不要 |

## 5. 既知の懸念（deferred）

- 512 token 超の系列での品質は未測定（tokenizer 自身の宣言は 512、構造上の上限は 8,192）。
- CUDA OOM の処理経路、強い fallback escape は、実装・テスト済みだが実入力で一度も発火していない。
- レイテンシーは1条件のみ計測（Q=4・K=8・同時実行1で p95 34 ms）。同時実行・大きな Q/K・長系列は未計測。
- 利用者別 rate limit（429）、tenant 分離、TLS、metrics、リクエスト間 batching は未実装（README の未充足項目）。

## 6. 運用メモ

- 起動・常駐・停止の手順は README の「セットアップ」を参照。稼働確認は記憶ではなく `/readyz` を実際に叩く。
- API key は `.env`（gitignore）。モデルは `models/`（gitignore、`fetch-model` で再取得でき、manifest の SHA-256 と照合される）。
- レビューで変異試験をさせるときは、他のレビュアーと同じ作業ツリーで並行させない（変異中のコードを他方がテストし、偽の flaky 報告が出た実績がある）。
