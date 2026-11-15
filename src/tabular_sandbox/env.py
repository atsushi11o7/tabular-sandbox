import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None) -> None:
    """`.env` の KEY=VALUE を環境変数へ読み込む（既存の値は上書きしない）。

    uv run は既定で .env を読まないため、TABPFN_TOKEN などを手動でプロセスへ渡す。
    """
    path = path or REPO_ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
