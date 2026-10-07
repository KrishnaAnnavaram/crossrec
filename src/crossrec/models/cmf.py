"""Collective matrix factorisation (CMF): ONE user-factor matrix for both domains.

The user step solves each user vector from the ratings of both domains together. The item step
solves the food items and the book items against the same user vectors. Thus a factor has the
same meaning in both domains, and a user vector learned from food ratings can score books.
Each domain keeps its own global mean, user biases and item biases.
"""

from __future__ import annotations

import numpy as np

from ..split import TrainData
from .als import IdIndex, als, fit_biases, fold_in, user_bias
from .base import Recommender, history_arrays


class CMFRecommender(Recommender):
    name = "cmf"

    def __init__(self, factors: int = 8, reg: float = 0.1, iterations: int = 12,
                 source_weight: float = 1.0, seed: int = 42):
        super().__init__(factors=factors, reg=reg, iterations=iterations,
                         source_weight=source_weight, seed=seed)
        self.factors, self.reg, self.iterations = factors, reg, iterations
        self.source_weight, self.seed = source_weight, seed

    def _fit(self, train: TrainData) -> None:
        src, tgt = train.source_ratings, train.target_ratings
        self.users = IdIndex(list(src["user_id"]) + list(tgt["user_id"]))
        self.source_items = IdIndex(src["item_id"])
        n_s, n_t, n_u = len(self.source_items), len(self.target_items), len(self.users)

        us, is_, rs = self.users.encode(src["user_id"]), self.source_items.encode(src["item_id"]), \
            src["rating"].to_numpy(float)
        ut, it, rt = self.users.encode(tgt["user_id"]), self.target_items.encode(tgt["item_id"]), \
            tgt["rating"].to_numpy(float)

        self.source_bias = fit_biases(us, is_, rs, n_u, n_s)
        tb = fit_biases(ut, it, rt, n_u, n_t)
        self.target_user_bias = tb.user  # indexed by self.users
        res_s = rs - self.source_bias.mu - self.source_bias.user[us] - self.source_bias.item[is_]
        res_t = rt - tb.mu - tb.user[ut] - tb.item[it]

        u = np.concatenate([us, ut])
        i = np.concatenate([is_, it + n_s])  # one item space: source items first, then target items
        res = np.concatenate([res_s, res_t])
        w = np.concatenate([np.full(len(res_s), self.source_weight), np.ones(len(res_t))])
        self.user_f, item_f, self.history = als(
            u, i, res, n_u, n_s + n_t, factors=self.factors, reg=self.reg,
            iterations=self.iterations, seed=self.seed, weights=w,
        )
        self.source_item_f, self.target_item_f = item_f[:n_s], item_f[n_s:]

    def _user_vector(self, user_id, source_history):
        if user_id in self.users:
            pos = self.users.pos[user_id]
            return self.user_f[pos], float(self.target_user_bias[pos])
        items, ratings = history_arrays(source_history)
        known = [(a, b) for a, b in zip(items, ratings) if a in self.source_items]
        if not known:
            return np.zeros(self.factors), 0.0
        idx = self.source_items.encode([a for a, _ in known])
        r = np.array([b for _, b in known])
        bu = user_bias(idx, r, self.source_bias)
        res = r - self.source_bias.mu - bu - self.source_bias.item[idx]
        return fold_in(self.source_item_f, idx, res, self.reg, self.source_weight), 0.0

    def predict_known(self, user_id, source_history):
        vec, bu = self._user_vector(user_id, source_history)
        return self.target_bias.mu + bu + self.target_bias.item + self.target_item_f @ vec

    def score_target(self, user_id, source_history=None):
        self._check()
        return self.predict_known(user_id, source_history)

    def _arrays(self):
        return {
            "user_f": self.user_f, "source_item_f": self.source_item_f, "target_item_f": self.target_item_f,
            "target_user_bias": self.target_user_bias, "source_bias_user": self.source_bias.user,
            "source_bias_item": self.source_bias.item,
        }

    def _meta(self):
        return {"users": self.users.ids, "source_items": self.source_items.ids, "source_mu": self.source_bias.mu}

    def _restore(self, arrays, meta):
        from .als import Biases

        self.users = IdIndex(meta["users"])
        self.source_items = IdIndex(meta["source_items"])
        self.user_f, self.source_item_f = arrays["user_f"], arrays["source_item_f"]
        self.target_item_f, self.target_user_bias = arrays["target_item_f"], arrays["target_user_bias"]
        self.source_bias = Biases(meta["source_mu"], arrays["source_bias_user"], arrays["source_bias_item"])
