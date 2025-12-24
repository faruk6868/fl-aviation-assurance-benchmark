"""Utilities for loading and preprocessing the NASA C-MAPSS dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - torch optional for type checkers
    torch = None  # type: ignore
    Dataset = object  # type: ignore


C_MAPSS_COLUMNS = [
    "engine_id",
    "cycle",
    *[f"setting_{i}" for i in range(1, 4)],
    *[f"sensor_{i:02d}" for i in range(1, 21 + 1)],
]


def load_cmapss_raw(dataset_id: str, root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load raw train/test dataframes and ground-truth RUL for a dataset split.

    Parameters
    ----------
    dataset_id:
        One of ``"FD001"`` ... ``"FD004"``.
    root:
        Directory containing the ``train_*.txt`` etc. files.

    Returns
    -------
    tuple
        ``(train_df, test_df, test_rul)`` where ``test_rul`` is a series of
        per-engine remaining useful life values for the test set.
    """

    dataset_id = dataset_id.upper()
    train_path = root / f"train_{dataset_id}.txt"
    test_path = root / f"test_{dataset_id}.txt"
    rul_path = root / f"RUL_{dataset_id}.txt"

    for file_path in (train_path, test_path, rul_path):
        if not file_path.exists():
            raise FileNotFoundError(f"Missing C-MAPSS file: {file_path}")

    train_df = _read_cmapss_file(train_path)
    test_df = _read_cmapss_file(test_path)
    test_rul = pd.read_csv(rul_path, header=None, names=["RUL"])["RUL"]

    return train_df, test_df, test_rul


def _read_cmapss_file(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep=" ", header=None)
    data = data.drop(columns=[26, 27], errors="ignore")  # trailing blanks
    data.columns = C_MAPSS_COLUMNS
    return data


def compute_rul(df: pd.DataFrame, clip: int | None = 125) -> pd.Series:
    """Compute capped RUL for a training dataframe."""

    grouped = df.groupby("engine_id")["cycle"].max()
    rul = (
        grouped.loc[df["engine_id"].values].values
        - df["cycle"].values
        + 1
    )
    if clip is not None:
        rul = np.minimum(rul, clip)
    return pd.Series(rul, index=df.index, name="RUL")


def append_test_rul(test_df: pd.DataFrame, test_rul: pd.Series, clip: int | None = 125) -> pd.Series:
    """Expand per-engine RUL values to per-record labels for test set."""

    engines = test_df["engine_id"].unique()
    expanded: list[np.ndarray] = []
    for engine_id, base_rul in zip(engines, test_rul, strict=True):
        engine_cycles = test_df[test_df["engine_id"] == engine_id]["cycle"]
        rul = base_rul + engine_cycles.max() - engine_cycles
        expanded.append(rul.values)

    concat = np.concatenate(expanded)
    if clip is not None:
        concat = np.minimum(concat, clip)
    return pd.Series(concat, index=test_df.index, name="RUL")


def prepare_cmapss_dataset(
    dataset_id: str,
    root: Path,
    rul_clip: int | None = 125,
    normalize: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    """Return processed train/test dataframes with RUL column and scalers.

    The returned metadata dictionary contains fitted scaler parameters to be
    reused for client-specific processing.
    """

    train_df, test_df, test_rul = load_cmapss_raw(dataset_id, root)

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["RUL"] = compute_rul(train_df, clip=rul_clip)
    test_df["RUL"] = append_test_rul(test_df, test_rul, clip=rul_clip)

    features = [col for col in train_df.columns if col not in {"engine_id", "RUL"}]

    metadata: Dict[str, np.ndarray] = {}
    if normalize:
        scaler = StandardScaler()
        train_df[features] = scaler.fit_transform(train_df[features])
        test_df[features] = scaler.transform(test_df[features])
        metadata["scaler_mean"] = scaler.mean_.astype(np.float32)
        metadata["scaler_scale"] = scaler.scale_.astype(np.float32)

    return train_df, test_df, metadata


def add_failure_label(df: pd.DataFrame, lead_time_cycles: int) -> pd.DataFrame:
    """Annotate dataframe with binary ``is_event`` column using lead time."""

    df = df.copy()
    df["is_event"] = (df["RUL"] <= lead_time_cycles).astype(int)
    return df


def split_train_validation(
    df: pd.DataFrame,
    validation_size: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/validation sets stratified by engine id."""

    engine_ids = df["engine_id"].unique()
    train_ids, val_ids = train_test_split(
        engine_ids,
        test_size=validation_size,
        random_state=seed,
    )
    train_df = df[df["engine_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["engine_id"].isin(val_ids)].reset_index(drop=True)
    return train_df, val_df


@dataclass
class CmapssDataset(Dataset):
    """Torch dataset for C-MAPSS features and RUL targets."""

    features: np.ndarray
    targets: np.ndarray

    def __post_init__(self) -> None:
        if self.features.shape[0] != self.targets.shape[0]:  # pragma: no cover
            raise ValueError("Features and targets must have same length")

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int):  # type: ignore[override]
        x = self.features[index]
        y = self.targets[index]
        if torch is not None:
            return torch.from_numpy(x).float(), torch.tensor(y).float()
        return x, y


def to_numpy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract numpy arrays for features and target from processed dataframe."""

    feature_cols = [col for col in df.columns if col not in {"engine_id", "RUL"}]
    features = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df["RUL"].to_numpy(dtype=np.float32)
    return features, targets


def to_dataset(df: pd.DataFrame) -> CmapssDataset:
    features, targets = to_numpy(df)
    return CmapssDataset(features, targets)

