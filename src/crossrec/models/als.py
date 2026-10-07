"""Biased matrix factorisation fit by alternating least squares on OBSERVED ratings only.

A missing rating is not a zero rating. The fit uses only the (user, item, rating) triples that
exist. Ratings are first explained by a global mean, a user bias and an item bias. The factors
then explain the residual. Regularisation is weighted by the number of ratings (ALS-WR).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class IdIndex:
    """A stable map between string IDs and row positions."""

    def __init__(self, ids):
        self.ids: list[str] = list(dict.fromkeys(str(i) for i in ids))
        self.pos: dict[str, int] = {i: n for n, i in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, key) -> bool:
        return key in self.pos

    def encode(self, values) -> np.ndarray:
        return np.fromiter((self.pos[str(v)] for v in values), dtype=np.int64, count=len(values))


@dataclass
class Biases:
    mu: float
    user: np.ndarray
    item: np.ndarray


def fit_biases(u: np.ndarray, i: np.ndarray, r: np.ndarray, n_users: int, n_items: int,
               reg_user: float = 5.0, reg_item: float = 5.0, n_iter: int = 5) -> Biases:
    """Regularised user and item biases by coordinate descent."""
    mu = float(r.mean()) if r.size else 0.0
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)
    cnt_u = np.bincount(u, minlength=n_users)
    cnt_i = np.bincount(i, minlength=n_items)
    for _ in range(n_iter):
        bi = np.bincount(i, weights=r - mu - bu[u], minlength=n_items) / (cnt_i + reg_item)
        bu = np.bincount(u, weights=r - mu - bi[i], minlength=n_users) / (cnt_u + reg_user)
    return Biases(mu, bu, bi)


def group(idx: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR-style grouping: rows of group ``a`` are ``order[indptr[a]:indptr[a+1]]``."""
    order = np.argsort(idx, kind="stable")
    indptr = np.concatenate([[0], np.cumsum(np.bincount(idx, minlength=n))])
    return order, indptr


def solve_rows(fixed: np.ndarray, other: np.ndarray, values: np.ndarray, weights: np.ndarray,
               order: np.ndarray, indptr: np.ndarray, n: int, reg: float) -> np.ndarray:
    """Solve one ridge problem per row group. Rows with no ratings get a zero vector."""
    k = fixed.shape[1]
    out = np.zeros((n, k))
    eye = np.eye(k)
    for a in range(n):
        rows = order[indptr[a]:indptr[a + 1]]
        if rows.size == 0:
            continue
        f = fixed[other[rows]]
        w = weights[rows]
        lhs = (f * w[:, None]).T @ f + reg * max(1.0, float(w.sum())) * eye
        out[a] = np.linalg.solve(lhs, (f * w[:, None]).T @ values[rows])
    return out


def als(u: np.ndarray, i: np.ndarray, residual: np.ndarray, n_users: int, n_items: int, *,
        factors: int, reg: float, iterations: int, seed: int,
        weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Factorise the residual. Return (user factors, item factors, train RMSE per iteration)."""
    rng = np.random.default_rng(seed)
    weights = np.ones_like(residual) if weights is None else weights
    user_f = rng.normal(0, 0.1, size=(n_users, factors))
    item_f = rng.normal(0, 0.1, size=(n_items, factors))
    by_user = group(u, n_users)
    by_item = group(i, n_items)
    history = []
    for _ in range(iterations):
        user_f = solve_rows(item_f, i, residual, weights, *by_user, n_users, reg)
        item_f = solve_rows(user_f, u, residual, weights, *by_item, n_items, reg)
        pred = np.einsum("ij,ij->i", user_f[u], item_f[i])
        history.append(float(np.sqrt(np.average((residual - pred) ** 2, weights=weights))))
    return user_f, item_f, history


def fold_in(item_f: np.ndarray, items: np.ndarray, residual: np.ndarray, reg: float,
            weight: float = 1.0) -> np.ndarray:
    """Solve the factor vector of one new user from the item factors of the items it rated."""
    k = item_f.shape[1]
    if items.size == 0:
        return np.zeros(k)
    f = item_f[items]
    lhs = weight * f.T @ f + reg * max(1.0, weight * items.size) * np.eye(k)
    return np.linalg.solve(lhs, weight * f.T @ residual)


def user_bias(items: np.ndarray, ratings: np.ndarray, b: Biases, reg_user: float = 5.0) -> float:
    if items.size == 0:
        return 0.0
    return float((ratings - b.mu - b.item[items]).sum() / (items.size + reg_user))


@dataclass
class BiasedMF:
    """One-domain biased MF. Used by EMCDR for the source and the target domain."""

    factors: int = 8
    reg: float = 0.1
    iterations: int = 12
    seed: int = 42

    def fit(self, users, items, ratings) -> "BiasedMF":
        self.users = IdIndex(users)
        self.items = IdIndex(items)
        u = self.users.encode(users)
        i = self.items.encode(items)
        r = np.asarray(ratings, dtype=float)
        self.biases = fit_biases(u, i, r, len(self.users), len(self.items))
        res = r - self.biases.mu - self.biases.user[u] - self.biases.item[i]
        self.user_f, self.item_f, self.history = als(
            u, i, res, len(self.users), len(self.items),
            factors=self.factors, reg=self.reg, iterations=self.iterations, seed=self.seed,
        )
        return self

    def fold_in(self, items, ratings) -> tuple[np.ndarray, float]:
        known = [(it, r) for it, r in zip(items, ratings) if str(it) in self.items]
        if not known:
            return np.zeros(self.factors), 0.0
        idx = self.items.encode([k for k, _ in known])
        r = np.array([v for _, v in known], dtype=float)
        bu = user_bias(idx, r, self.biases)
        res = r - self.biases.mu - bu - self.biases.item[idx]
        return fold_in(self.item_f, idx, res, self.reg), bu
