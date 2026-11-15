import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


def load_adult() -> tuple[pd.DataFrame, np.ndarray]:
    """UCI Adult (OpenML version=2) を生のまま返す。y は `>50K` を 1 とした二値。"""
    ds = fetch_openml("adult", version=2, as_frame=True)
    X = ds.data  # カテゴリ列・欠損を含む（前処理しない）
    y = (ds.target == ">50K").astype(int).to_numpy()
    return X, y


def stratified_subsample(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """二値ラベル y の陽性比率を保ったまま n 件抽出するインデックスを返す。"""
    idx = np.arange(len(y))
    if n >= len(y):
        return idx
    pos, neg = idx[y == 1], idx[y == 0]
    n_pos = round(n * len(pos) / len(y))
    n_neg = n - n_pos
    take = np.concatenate(
        [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
    )
    rng.shuffle(take)
    return take
