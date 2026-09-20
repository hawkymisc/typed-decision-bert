# JevBERT

BERT系encoderを用いた、Jev互換APIを目指す指示条件付き意思決定エンジン。

- 仕様・設計書: [JevBERT_spec_design.md](JevBERT_spec_design.md)
- P0.5（PoC）実装設計書: [docs/POC_DESIGN.md](docs/POC_DESIGN.md)
- Jevとの差分・未確認事項: [compat/differences.md](compat/differences.md)

## 状態

P0.5（PoC）実装中。フェーズ1（API層・contract・compiler・fake backend・テスト）と
フェーズ1.5（3系統のレビュー指摘の反映。[POC_DESIGN §12.2](docs/POC_DESIGN.md)）が完了し、
フェーズ2（zero-shot NLI backend、実機検証、非機能要件の充足表）が未了である。

本READMEは、POC_DESIGN 第9章の非機能要件ごとの充足・部分充足・未充足と根拠（AC5）を
フェーズ2で記載する。現時点でJev互換を主張しない。

## 実行方法

```bash
uv sync
uv run python -m jevbert init-env                                  # .env にAPI keyを生成
uv run python -m jevbert fetch-model                               # フェーズ2で実装
uv run python -m jevbert serve --config configs/jevbert.poc.yaml   # 127.0.0.1:8765
uv run pytest
```
