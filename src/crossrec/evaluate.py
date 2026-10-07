"""The cold-start evaluation protocol.

For each test user, the model sees only the source-domain history. The model ranks all target
items that it knows. The held-out target ratings with a rating at or above the relevance
threshold are the relevant items. The protocol also measures RMSE and MAE on all held-out ratings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics
from .models import Recommender
from .split import ColdStartSplit

RANK_METRICS = {
    "recall": metrics.recall_at_k,
    "ndcg": metrics.ndcg_at_k,
    "map": metrics.average_precision_at_k,
    "precision": metrics.precision_at_k,
    "hit_rate": metrics.hit_rate_at_k,
}


@dataclass
class ModelResult:
    model: str
    k: int
    users_ranked: int
    users_without_relevant: int
    unreachable_relevant: int
    means: dict[str, float]
    ci95: dict[str, tuple[float, float]]
    rmse: float
    mae: float
    per_user: dict[str, list[float]] = field(default_factory=dict, repr=False)

    def as_dict(self) -> dict:
        return {
            "model": self.model, "k": self.k, "users_ranked": self.users_ranked,
            "users_without_relevant": self.users_without_relevant,
            "unreachable_relevant": self.unreachable_relevant,
            **{f"{m}@{self.k}": round(v, 4) for m, v in self.means.items()},
            **{f"{m}@{self.k}_ci95": [round(a, 4), round(b, 4)] for m, (a, b) in self.ci95.items()},
            "rmse": round(self.rmse, 4), "mae": round(self.mae, 4),
        }


def evaluate_model(model: Recommender, split: ColdStartSplit, k: int = 10,
                   relevant_threshold: float = 4.0, seed: int = 0) -> ModelResult:
    split.check_no_leakage()
    src = split.train.source_ratings
    histories = {u: g for u, g in src[src["user_id"].isin(split.test_users)].groupby("user_id")}
    per_user: dict[str, list[float]] = {m: [] for m in RANK_METRICS}
    y_true, y_pred = [], []
    no_relevant = unreachable = 0
    for user, held in split.test_target.groupby("user_id"):
        history = histories.get(user, src.iloc[0:0])
        # The model must not use any stored state for a test user: score from the source history.
        y_true.extend(held["rating"].tolist())
        y_pred.extend(model.predict(_anon(user), held["item_id"].tolist(), history).tolist())
        relevant = set(held.loc[held["rating"] >= relevant_threshold, "item_id"])
        if not relevant:
            no_relevant += 1
            continue
        unreachable += len(relevant - set(model.target_items.ids))
        ranked = [it for it, _ in model.recommend(_anon(user), k=k, source_history=history)]
        for name, fn in RANK_METRICS.items():
            per_user[name].append(fn(ranked, relevant, k))
    means = {m: float(np.mean(v)) if v else float("nan") for m, v in per_user.items()}
    ci = {m: metrics.bootstrap_ci(v, seed=seed) for m, v in per_user.items()}
    return ModelResult(
        model=model.name, k=k, users_ranked=len(per_user["recall"]), users_without_relevant=no_relevant,
        unreachable_relevant=unreachable, means=means, ci95=ci,
        rmse=metrics.rmse(y_true, y_pred) if y_true else float("nan"),
        mae=metrics.mae(y_true, y_pred) if y_true else float("nan"), per_user=per_user,
    )


def _anon(user_id: str) -> str:
    """A key that no model knows. It forces the fold-in path from the source history."""
    return f"__cold_start__::{user_id}"


def results_table(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "model": r.model, f"recall@{r.k}": r.means["recall"], f"ndcg@{r.k}": r.means["ndcg"],
            f"map@{r.k}": r.means["map"], f"hit@{r.k}": r.means["hit_rate"], "rmse": r.rmse, "mae": r.mae,
            "ndcg_ci95": f"[{r.ci95['ndcg'][0]:.3f}, {r.ci95['ndcg'][1]:.3f}]",
        })
    return pd.DataFrame(rows).set_index("model")
