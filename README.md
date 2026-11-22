# tabular-sandbox

テーブルデータ系の実験を放り込むサンドボックス。

## 環境変数

秘密情報は `.env`（gitignore 済み）に置き、devcontainer 起動時に `--env-file` でコンテナへ渡す。

1. 雛形をコピーして値を埋める

   ```bash
   cp .env.example .env
   ```

2. devcontainer を起動（または再起動）する

- `.env` が無いと devcontainer が起動しないため、使わない値も雛形のまま置いておく。
- `.env` は起動時にしか読まれない。値を変えたらコンテナを再起動する。
- 変数を追加したときは `.env.example` にも追記する。

| 変数 | 用途 |
|---|---|
| `GH_TOKEN` | GitHub CLI（`gh`）の認証 |
| `TABPFN_TOKEN` | TabPFN-3 のモデル重み取得（[Prior Labs](https://ux.priorlabs.ai) で発行） |
