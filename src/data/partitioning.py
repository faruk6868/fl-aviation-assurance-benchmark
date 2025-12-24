"""Client data partitioning strategies for federated experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


def iid_partition(df: pd.DataFrame, num_clients: int, seed: int = 42) -> Dict[int, pd.DataFrame]:
    """Evenly distribute engines across clients (IID)."""

    rng = np.random.default_rng(seed)
    engine_ids = df["engine_id"].unique()
    rng.shuffle(engine_ids)
    splits = np.array_split(engine_ids, num_clients)
    partitions: Dict[int, pd.DataFrame] = {}
    for client_id, engine_subset in enumerate(splits):
        partitions[client_id] = df[df["engine_id"].isin(engine_subset)].reset_index(drop=True)
    return partitions


def dirichlet_quantity_skew(
    df: pd.DataFrame,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> Dict[int, pd.DataFrame]:
    """Partition engines using a Dirichlet distribution (quantity skew)."""

    rng = np.random.default_rng(seed)
    engine_ids = df["engine_id"].unique()
    rng.shuffle(engine_ids)
    proportions = rng.dirichlet(alpha=[alpha] * num_clients)
    counts = np.maximum((proportions * len(engine_ids)).astype(int), 1)

    # adjust to ensure sum equals total engines
    while counts.sum() < len(engine_ids):
        counts[rng.integers(0, num_clients)] += 1
    while counts.sum() > len(engine_ids):
        idx = rng.integers(0, num_clients)
        if counts[idx] > 1:
            counts[idx] -= 1

    partitions: Dict[int, pd.DataFrame] = {}
    start = 0
    for client_id, count in enumerate(counts):
        subset = engine_ids[start : start + count]
        partitions[client_id] = df[df["engine_id"].isin(subset)].reset_index(drop=True)
        start += count
    return partitions


def label_skew_partition(
    df: pd.DataFrame,
    num_clients: int,
    lead_time_cycles: int,
    seed: int = 42,
) -> Dict[int, pd.DataFrame]:
    """Create label-skewed partitions by grouping engines by positive rate."""

    labeled = df.copy()
    labeled["is_event"] = (labeled["RUL"] <= lead_time_cycles).astype(int)
    engine_stats = (
        labeled.groupby("engine_id")["is_event"]
        .mean()
        .sort_values()
        .reset_index()
    )

    rng = np.random.default_rng(seed)
    partitions: Dict[int, List[int]] = defaultdict(list)

    # Assign engines alternating between high and low event rates to amplify skew
    low_engines = engine_stats.iloc[: len(engine_stats) // 2]["engine_id"].tolist()
    high_engines = engine_stats.iloc[len(engine_stats) // 2 :]["engine_id"].tolist()
    rng.shuffle(low_engines)
    rng.shuffle(high_engines)

    for idx, engine_id in enumerate(low_engines):
        partitions[idx % num_clients].append(engine_id)
    for idx, engine_id in enumerate(high_engines):
        partitions[(idx + num_clients // 2) % num_clients].append(engine_id)

    data_partitions: Dict[int, pd.DataFrame] = {}
    for client_id in range(num_clients):
        engines = partitions[client_id]
        subset = labeled[labeled["engine_id"].isin(engines)].reset_index(drop=True)
        if "is_event" in subset.columns:
            subset = subset.drop(columns=["is_event"])
        data_partitions[client_id] = subset
    return data_partitions


def merge_partitions(partitions: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate client partitions into a single dataframe."""

    return pd.concat(partitions, axis=0).reset_index(drop=True)

