"""Kumo Tabular 単体検証 — UCI Adult (Census Income)。

experiments/01_tabpfn と同じ条件（固定の層化分割 test=20%・同じ学習サイズ・同じサブサンプル）で
NVIDIA Kumo Tabular を評価し、ROC-AUC と fit/predict 時間が学習サイズでどう変わるかを見る。
前処理はせず、生の DataFrame を TableTensor に変換して渡す。

実行: uv run python experiments/02_kumo_tabular/run.py
出力: results/02_kumo_tabular/sweep.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sdm
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabular_sandbox.data import load_adult, stratified_subsample

ROOT = Path(__file__).resolve().parents[2]  # /workspace

# None = 学習プール全量。それ以外はその件数へ層化サブサンプリングする
# 全量 (None) は手元の環境 (VRAM 16GB / RAM 15GB) で途中終了したため除外
TRAIN_SIZES: list[int | None] = [1000, 5000, 10000, 20000]
TEST_SIZE = 0.2
RANDOM_STATE = 0
DEVICE = "cuda"
MODEL_SIZE = "large"  # KumoTabular の既定
NUM_ESTIMATORS = 8  # 公式 README の例に合わせる
TARGET = "target"
# 16GB GPU ではテスト 9769 行を一括で predict すると VRAM からあふれて極端に遅くなるため分割する
PREDICT_CHUNK = 2000
WARMUP_ROWS = 500


def to_table(df: pd.DataFrame, stypes: dict) -> sdm.TableTensor:
    return sdm.TableTensor.from_pandas(df=df, stypes=stypes, device=DEVICE)


def main() -> None:
    X, y = load_adult()
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(
        f"data: {X.shape[0]} rows x {X.shape[1]} features | "
        f"positive_rate={y.mean():.3f} | test={len(y_test)}"
    )

    x_stypes = sdm.infer_stypes(X)
    train_stypes = x_stypes | {TARGET: "categorical"}
    x_test = to_table(X_test, x_stypes)

    t0 = time.perf_counter()
    model = sdm.models.KumoTabular(size=MODEL_SIZE, device=DEVICE)
    load_sec = time.perf_counter() - t0
    print(f"model load: {load_sec:.1f}s", flush=True)

    # 初回だけ CUDA の初期化などが入るので、計測前に1回空打ちして除外する
    t0 = time.perf_counter()
    warm = to_table(
        X_pool.iloc[:WARMUP_ROWS].assign(**{TARGET: y_pool[:WARMUP_ROWS]}), train_stypes
    )
    model.fit(x=warm.drop_columns(TARGET), y=warm[:, TARGET], num_estimators=NUM_ESTIMATORS)
    model.predict(x_test[:WARMUP_ROWS])
    torch.cuda.synchronize()
    model.clear()
    torch.cuda.reset_peak_memory_stats()
    warmup_sec = time.perf_counter() - t0
    print(f"warmup: {warmup_sec:.1f}s", flush=True)

    rng = np.random.default_rng(RANDOM_STATE)
    results: list[dict] = []
    summary = {
        "model": f"NVIDIA Kumo Tabular ({MODEL_SIZE})",
        "library": f"structured-data-models {sdm.__version__}",
        "num_estimators": NUM_ESTIMATORS,
        "predict_chunk": PREDICT_CHUNK,
        "dataset": "UCI Adult (Census Income), OpenML version=2",
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "split": f"stratified holdout, test_size={TEST_SIZE}, seed={RANDOM_STATE}",
        "n_test": int(len(y_test)),
        "device": f"{DEVICE} ({torch.cuda.get_device_name()})",
        "model_load_sec": load_sec,
        "warmup_sec": warmup_sec,
        "preprocessing": "none (raw DataFrame -> TableTensor with inferred stypes)",
        "results": results,
    }

    out_dir = ROOT / "results" / "02_kumo_tabular"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sweep.json"
    for size in TRAIN_SIZES:
        n = size if size is not None else len(y_pool)
        print(f"[{time.strftime('%H:%M:%S')}] start n_train={n}", flush=True)
        idx = stratified_subsample(y_pool, n, rng)
        train = X_pool.iloc[idx].assign(**{TARGET: y_pool[idx]})
        table = to_table(train, train_stypes)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.fit(
            x=table.drop_columns(TARGET),
            y=table[:, TARGET],
            num_estimators=NUM_ESTIMATORS,
            generator=torch.Generator(device=DEVICE).manual_seed(RANDOM_STATE),
        )
        torch.cuda.synchronize()
        fit_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        # 予測対象の行は学習行だけを参照するので、分割しても予測は変わらない
        pred = pd.concat(
            [
                model.predict(x_test[i : i + PREDICT_CHUNK]).to_pandas()
                for i in range(0, len(y_test), PREDICT_CHUNK)
            ],
            ignore_index=True,
        )
        torch.cuda.synchronize()
        predict_sec = time.perf_counter() - t0
        model.clear()

        # 出力列はクラスラベルの文字列で、並び順は保証されない（['1', '0'] の順で返ることがある）
        proba = pred["1"].to_numpy()
        auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, (proba >= 0.5).astype(int))
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.reset_peak_memory_stats()
        results.append(
            {
                "n_train": int(len(idx)),
                "auc": float(auc),
                "accuracy": float(acc),
                "fit_sec": fit_sec,
                "predict_sec": predict_sec,
                "peak_gpu_mem_gb": peak_mem_gb,
            }
        )
        print(
            f"n_train={len(idx):>6}: AUC={auc:.4f} acc={acc:.4f} "
            f"fit={fit_sec:.2f}s predict={predict_sec:.2f}s mem={peak_mem_gb:.1f}GB",
            flush=True,
        )
        # 途中で止めても完了済みの条件が残るよう、毎回保存し直す
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
