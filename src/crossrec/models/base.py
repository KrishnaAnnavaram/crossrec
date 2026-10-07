"""The one model interface: fit, score, predict, recommend, save and load.

Users and items are addressed by their real string IDs. Already rated items are excluded from a
recommendation list. Model files are a JSON metadata file plus a NumPy ``.npz`` file (no pickle).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from ..split import TrainData
from .als import Biases, IdIndex, fit_biases

META_FILE = "model.json"
ARRAYS_FILE = "arrays.npz"


class NotFittedError(RuntimeError):
    pass


class Recommender(ABC):
    name = "base"

    def __init__(self, **params):
        self.params = params
        self.fitted = False

    # ----- fit -------------------------------------------------------------------------------
    def fit(self, train: TrainData) -> "Recommender":
        self.source, self.target = train.source, train.target
        tgt = train.target_ratings
        self.target_items = IdIndex(tgt["item_id"])
        self.target_users = IdIndex(tgt["user_id"])
        self.target_bias = fit_biases(
            self.target_users.encode(tgt["user_id"]), self.target_items.encode(tgt["item_id"]),
            tgt["rating"].to_numpy(float), len(self.target_users), len(self.target_items),
        )
        self._fit(train)
        self.fitted = True
        return self

    @abstractmethod
    def _fit(self, train: TrainData) -> None: ...

    # ----- score -----------------------------------------------------------------------------
    @abstractmethod
    def score_target(self, user_id: str, source_history: pd.DataFrame | None = None) -> np.ndarray:
        """Return one score for each item in ``self.target_items`` (higher is better)."""

    def predict(self, user_id: str, item_ids, source_history: pd.DataFrame | None = None) -> np.ndarray:
        """Predicted star rating. Items not seen at fit time get the target mean."""
        self._check()
        scores = self.predict_known(user_id, source_history)
        out = np.full(len(item_ids), self.target_bias.mu)
        for n, it in enumerate(item_ids):
            pos = self.target_items.pos.get(str(it))
            if pos is not None:
                out[n] = scores[pos]
        return np.clip(out, 1.0, 5.0)

    def predict_known(self, user_id: str, source_history: pd.DataFrame | None) -> np.ndarray:
        """Rating predictions for all known target items. Default: the item-bias rating."""
        return self.target_bias.mu + self.target_bias.item

    def recommend(self, user_id: str, k: int = 10, source_history: pd.DataFrame | None = None,
                  exclude=()) -> list[tuple[str, float]]:
        """Top-k target items for one user, without the items in ``exclude``."""
        self._check()
        scores = self.score_target(user_id, source_history).astype(float).copy()
        for it in exclude:
            pos = self.target_items.pos.get(str(it))
            if pos is not None:
                scores[pos] = -np.inf
        k = min(k, int(np.isfinite(scores).sum()))
        top = np.argpartition(-scores, k - 1)[:k] if k > 0 else np.array([], dtype=int)
        top = top[np.lexsort((top, -scores[top]))]
        return [(self.target_items.ids[j], float(scores[j])) for j in top]

    def _check(self):
        if not self.fitted:
            raise NotFittedError(f"{self.name}: call fit() first")

    # ----- persistence -----------------------------------------------------------------------
    def _arrays(self) -> dict[str, np.ndarray]:
        return {}

    def _meta(self) -> dict:
        return {}

    def _restore(self, arrays: dict[str, np.ndarray], meta: dict) -> None:
        pass

    def save(self, directory: str | Path) -> Path:
        self._check()
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        arrays = {
            "target_bias_user": self.target_bias.user,
            "target_bias_item": self.target_bias.item,
            **self._arrays(),
        }
        np.savez(d / ARRAYS_FILE, **arrays)
        meta = {
            "model": self.name,
            "params": self.params,
            "source": self.source,
            "target": self.target,
            "target_mu": self.target_bias.mu,
            "target_items": self.target_items.ids,
            "target_users": self.target_users.ids,
            **self._meta(),
        }
        (d / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
        return d

    @classmethod
    def restore(cls, meta: dict, arrays: dict[str, np.ndarray]) -> "Recommender":
        model = cls(**meta["params"])
        model.source, model.target = meta["source"], meta["target"]
        model.target_items = IdIndex(meta["target_items"])
        model.target_users = IdIndex(meta["target_users"])
        model.target_bias = Biases(meta["target_mu"], arrays["target_bias_user"], arrays["target_bias_item"])
        model._restore(arrays, meta)
        model.fitted = True
        return model


def history_arrays(history: pd.DataFrame | None) -> tuple[list[str], np.ndarray]:
    if history is None or history.empty:
        return [], np.array([])
    return history["item_id"].astype(str).tolist(), history["rating"].to_numpy(float)
