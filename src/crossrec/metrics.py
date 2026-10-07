"""Ranking and rating metrics, plus a seeded bootstrap confidence interval."""

from __future__ import annotations

import math

import numpy as np


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("relevant set is empty")
    return len(set(ranked[:k]) & relevant) / min(len(relevant), k)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / k


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("relevant set is empty")
    dcg = sum(1.0 / math.log2(n + 2) for n, it in enumerate(ranked[:k]) if it in relevant)
    ideal = sum(1.0 / math.log2(n + 2) for n in range(min(len(relevant), k)))
    return dcg / ideal


def average_precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("relevant set is empty")
    hits, total = 0, 0.0
    for n, it in enumerate(ranked[:k]):
        if it in relevant:
            hits += 1
            total += hits / (n + 1)
    return total / min(len(relevant), k)


def hit_rate_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return float(bool(set(ranked[:k]) & relevant))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def bootstrap_ci(values, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile interval of the mean over users."""
    v = np.asarray(values, float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
