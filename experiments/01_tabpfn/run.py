"""TabPFN-3 単体検証 — UCI Adult (Census Income)。

固定の層化分割 (test=20%) に対し、学習データ量を変えて TabPFN-3 を学習・評価し、
ROC-AUC と fit/predict 時間が学習サイズでどう変わるかを見る。
前処理はせず、生の DataFrame（カテゴリ列・欠損を含む）をそのまま渡す。

実行: uv run python experiments/01_tabpfn/run.py
出力: results/01_tabpfn/sweep.json
"""

from __future__ import annotations

import json
import time

from tabular_sandbox.env import REPO_ROOT, load_dotenv

load_dotenv()  # TABPFN_TOKEN を tabpfn の import 前に環境変数へ載せる

import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from tabpfn import TabPFNClassifier  # noqa: E402
from tabpfn.model_loading import ModelSource  # noqa: E402

from tabular_sandbox.data import load_adult, stratified_subsample  # noqa: E402

# None = 学習プール全量。それ以外はその件数へ層化サブサンプリングする
TRAIN_SIZES: list[int | None] = [1000, 5000, 10000, 20000, None]
TEST_SIZE = 0.2
RANDOM_STATE = 0
DEVICE = "cuda"


def main() -> None:
    X, y = load_adult()
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(
        f"data: {X.shape[0]} rows x {X.shape[1]} features | "
        f"positive_rate={y.mean():.3f} | test={len(y_test)}"
    )

    rng = np.random.default_rng(RANDOM_STATE)
    model_file = ModelSource.get_classifier_v3().default_filename

    results: list[dict] = []
    for size in TRAIN_SIZES:
        n = size if size is not None else len(y_pool)
        idx = stratified_subsample(y_pool, n, rng)
        X_tr, y_tr = X_pool.iloc[idx], y_pool[idx]

        clf = TabPFNClassifier(device=DEVICE, model_path=model_file, random_state=RANDOM_STATE)
        t0 = time.perf_counter()
        clf.fit(X_tr, y_tr)
        fit_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        proba = clf.predict_proba(X_test)[:, 1]
        predict_sec = time.perf_counter() - t0

        auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, (proba >= 0.5).astype(int))
        results.append(
            {
                "n_train": int(len(y_tr)),
                "auc": float(auc),
                "accuracy": float(acc),
                "fit_sec": fit_sec,
                "predict_sec": predict_sec,
            }
        )
        print(
            f"n_train={len(y_tr):>6}: AUC={auc:.4f} acc={acc:.4f} "
            f"fit={fit_sec:.2f}s predict={predict_sec:.2f}s"
        )

    summary = {
        "model": "TabPFN-3 (v3 default)",
        "model_file": model_file,
        "dataset": "UCI Adult (Census Income), OpenML version=2",
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "split": f"stratified holdout, test_size={TEST_SIZE}, seed={RANDOM_STATE}",
        "n_test": int(len(y_test)),
        "device": DEVICE,
        "preprocessing": "none (raw DataFrame with categorical + missing passed directly)",
        "results": results,
    }

    out_dir = REPO_ROOT / "results" / "01_tabpfn"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sweep.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
