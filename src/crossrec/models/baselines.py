"""Two single-domain baselines. Neither model uses the source domain.

``popularity`` ranks target items by the number of positive target ratings.
``itembias`` ranks by the regularised mean rating (global mean + item bias). A target-only matrix
factorisation has no user vector for a cold-start user, so it reduces to this model.
"""

from __future__ import annotations

import numpy as np

from ..split import TrainData
from .base import Recommender


class PopularityRecommender(Recommender):
    name = "popularity"

    def __init__(self, relevant_threshold: float = 4.0, **params):
        super().__init__(relevant_threshold=relevant_threshold, **params)
        self.relevant_threshold = relevant_threshold

    def _fit(self, train: TrainData) -> None:
        tgt = train.target_ratings
        idx = self.target_items.encode(tgt["item_id"])
        positive = (tgt["rating"].to_numpy(float) >= self.relevant_threshold).astype(float)
        n = len(self.target_items)
        # Positive count first, total count as a small tie-break.
        self.counts = np.bincount(idx, weights=positive, minlength=n) + 1e-3 * np.bincount(idx, minlength=n)

    def score_target(self, user_id, source_history=None):
        self._check()
        return self.counts

    def _arrays(self):
        return {"counts": self.counts}

    def _restore(self, arrays, meta):
        self.counts = arrays["counts"]


class ItemBiasRecommender(Recommender):
    name = "itembias"

    def _fit(self, train: TrainData) -> None:
        pass

    def score_target(self, user_id, source_history=None):
        self._check()
        return self.target_bias.mu + self.target_bias.item
