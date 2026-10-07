"""Mapping-based transfer (EMCDR style).

1. Fit a biased MF on the source domain and a separate biased MF on the target domain.
2. On users with ratings in both domains, learn a map f(source user vector) -> target user vector.
3. For a cold-start user, map the source vector into the target space and score the target items.

The two latent spaces are never multiplied directly. The learned map connects them. The default
map is closed-form ridge regression. ``mapping="mlp"`` uses scikit-learn (extra ``mlp``).
"""

from __future__ import annotations

import numpy as np

from ..split import TrainData
from .als import Biases, BiasedMF, IdIndex
from .base import Recommender, history_arrays


class EMCDRRecommender(Recommender):
    name = "emcdr"

    def __init__(self, factors: int = 8, reg: float = 0.1, iterations: int = 12,
                 map_reg: float = 1.0, mapping: str = "linear", seed: int = 42):
        if mapping not in ("linear", "mlp"):
            raise ValueError("mapping must be 'linear' or 'mlp'")
        super().__init__(factors=factors, reg=reg, iterations=iterations, map_reg=map_reg,
                         mapping=mapping, seed=seed)
        self.factors, self.reg, self.iterations = factors, reg, iterations
        self.map_reg, self.mapping, self.seed = map_reg, mapping, seed

    def _fit(self, train: TrainData) -> None:
        src, tgt = train.source_ratings, train.target_ratings
        kw = dict(factors=self.factors, reg=self.reg, iterations=self.iterations, seed=self.seed)
        self.src_mf = BiasedMF(**kw).fit(src["user_id"], src["item_id"], src["rating"])
        self.tgt_mf = BiasedMF(**kw).fit(tgt["user_id"], tgt["item_id"], tgt["rating"])
        # Align the target MF item order with self.target_items (both come from tgt in order).
        if self.tgt_mf.items.ids != self.target_items.ids:
            raise RuntimeError("target item order mismatch")
        bridge = [u for u in self.tgt_mf.users.ids if u in self.src_mf.users]
        if len(bridge) < 2:
            raise ValueError("EMCDR needs at least 2 users with ratings in both domains")
        x = self.src_mf.user_f[self.src_mf.users.encode(bridge)]
        y = self.tgt_mf.user_f[self.tgt_mf.users.encode(bridge)]
        self.n_bridge = len(bridge)
        if self.mapping == "linear":
            x1 = np.hstack([x, np.ones((len(x), 1))])
            penalty = self.map_reg * np.eye(x1.shape[1])
            penalty[-1, -1] = 0.0  # do not shrink the intercept
            self.map_w = np.linalg.solve(x1.T @ x1 + penalty, x1.T @ y)
        else:
            from sklearn.neural_network import MLPRegressor

            self.map_mlp = MLPRegressor(hidden_layer_sizes=(2 * self.factors,), alpha=self.map_reg,
                                        max_iter=2000, random_state=self.seed).fit(x, y)

    def map_vector(self, src_vec: np.ndarray) -> np.ndarray:
        if self.mapping == "linear":
            return np.append(src_vec, 1.0) @ self.map_w
        return self.map_mlp.predict(src_vec[None, :])[0]

    def _target_vector(self, user_id, source_history):
        if user_id in self.tgt_mf.users:
            pos = self.tgt_mf.users.pos[user_id]
            return self.tgt_mf.user_f[pos], float(self.tgt_mf.biases.user[pos])
        if user_id in self.src_mf.users:
            src_vec = self.src_mf.user_f[self.src_mf.users.pos[user_id]]
        else:
            items, ratings = history_arrays(source_history)
            src_vec, _ = self.src_mf.fold_in(items, ratings)
        return self.map_vector(src_vec), 0.0

    def predict_known(self, user_id, source_history):
        vec, bu = self._target_vector(user_id, source_history)
        b = self.tgt_mf.biases
        return b.mu + bu + b.item + self.tgt_mf.item_f @ vec

    def score_target(self, user_id, source_history=None):
        self._check()
        return self.predict_known(user_id, source_history)

    def _arrays(self):
        if self.mapping != "linear":
            raise NotImplementedError("save() supports the linear map only")
        out = {"map_w": self.map_w}
        for tag, mf in (("src", self.src_mf), ("tgt", self.tgt_mf)):
            out.update({f"{tag}_user_f": mf.user_f, f"{tag}_item_f": mf.item_f,
                        f"{tag}_bu": mf.biases.user, f"{tag}_bi": mf.biases.item})
        return out

    def _meta(self):
        return {
            "src_users": self.src_mf.users.ids, "src_items": self.src_mf.items.ids, "src_mu": self.src_mf.biases.mu,
            "tgt_users": self.tgt_mf.users.ids, "tgt_mu": self.tgt_mf.biases.mu, "n_bridge": self.n_bridge,
        }

    def _restore(self, arrays, meta):
        self.map_w = arrays["map_w"]
        self.n_bridge = meta["n_bridge"]
        kw = dict(factors=self.factors, reg=self.reg, iterations=self.iterations, seed=self.seed)
        self.src_mf, self.tgt_mf = BiasedMF(**kw), BiasedMF(**kw)
        self.src_mf.users, self.src_mf.items = IdIndex(meta["src_users"]), IdIndex(meta["src_items"])
        self.tgt_mf.users, self.tgt_mf.items = IdIndex(meta["tgt_users"]), IdIndex(meta["target_items"])
        for tag, mf, mu in (("src", self.src_mf, meta["src_mu"]), ("tgt", self.tgt_mf, meta["tgt_mu"])):
            mf.user_f, mf.item_f = arrays[f"{tag}_user_f"], arrays[f"{tag}_item_f"]
            mf.biases = Biases(mu, arrays[f"{tag}_bu"], arrays[f"{tag}_bi"])
