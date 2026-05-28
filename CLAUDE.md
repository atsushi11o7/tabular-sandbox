# tabular-sandbox

テーブルデータ系の実験を放り込むサンドボックス。手軽さと再現性のバランスを取りつつ、固まった処理は再利用できる形に育てていく。

## 環境・依存管理

- 実行環境は devcontainer (CUDA 12.6 + Python 3.12)。venv は `/opt/venv`（`UV_PROJECT_ENVIRONMENT` 指定、`.venv` は使わない）。
- パッケージ管理は **uv**。`pip install` は使わず、依存の増減は次の手順で行う:
  - 追加: `uv add <pkg>`（dev 依存は `uv add --dev <pkg>`）
  - 削除: `uv remove <pkg>`
  - これらは `pyproject.toml` と `uv.lock` を自動更新する。手で `uv.lock` を編集しない。
- `uv.lock` は **コミット対象**。再現性のため依存を変えたら必ず lock も一緒にコミットする。
- torch は CUDA 12.6 wheel を `pytorch-cu126` index から取得する設定（`[tool.uv.sources]`）。バージョン変更時はこの制約を壊さない。

## ディレクトリ構成

`PYTHONPATH=/workspace` が通っているので、`src/` 配下はそのまま `import` できる。

- `notebooks/` — 探索・可視化用の `.ipynb`。試行錯誤はここで自由に。
- `src/` — 固まった共通処理・再利用するモジュール。Notebook で安定したコードはここへ切り出す。
- `experiments/` — 実験の実行スクリプト（エントリポイント）。
- `configs/` — hydra 設定（使う場合のみ）。
- `data/` `outputs/` `results/` — 入出力。すべて gitignore 済み（コミットしない）。

橋渡しルール: Notebook で探索 → 安定したら `src/` の関数/クラスへ抽出 → `experiments/` のスクリプトから呼ぶ。Notebook に長いロジックを残し続けない。

## 実験管理

hydra / optuna / wandb は導入済みだが **強制しない**。サンドボックスなので手軽さ優先で、必要になったときに使う。使う場合の目安:

- hydra/omegaconf: 設定の出し入れが増えてきたら `configs/` で管理。
- optuna: ハイパラ探索が必要なときだけ。
- wandb: ログを残したい実験で。API キーは `.env`（gitignore 済み）に置く。

## コーディング・Lint

- Lint/format は **ruff**（`line-length=100`, `target=py312`, ルール: `E,F,I,B,UP`）。
- コミット前に `ruff check --fix .` と `ruff format .` を通す。
- データ列名やマジックナンバーなど、後で読んで非自明な箇所のみコメントを残す。

## Git・コミット

- コミットメッセージ: **タイトル1行のみ・英語の命令形・〜72文字・本文なし・`Co-Authored-By` フッターなし**。例: `Add target encoding utility`, `Fix CV split leakage`。
- 1コミット = 1論理変更。`git add -A` ではなく対象ファイルを明示的に stage する。
- `.env` や `data/`・モデル重みなど gitignore 対象は絶対にコミットしない。
- ブランチ運用: `feature/<topic>` を切って作業し、PR 経由でレビュー後 `main` へマージする。`main` への直接コミットはしない。
