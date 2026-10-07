import numpy as np
import pytest

from crossrec.models.als import BiasedMF, IdIndex, als, fit_biases, fold_in, group


def test_group_returns_rows_of_each_index():
    idx = np.array([2, 0, 2, 1])
    order, indptr = group(idx, 4)
    assert sorted(order[indptr[2]:indptr[3]].tolist()) == [0, 2]
    assert indptr[-1] == 4 and indptr[3] == indptr[4]  # index 3 has no rows


def test_id_index_is_stable_and_unique():
    ix = IdIndex(["b", "a", "b", "c"])
    assert ix.ids == ["b", "a", "c"]
    assert ix.encode(["c", "b"]).tolist() == [2, 0]
    assert "a" in ix and "z" not in ix


def test_missing_ratings_are_not_zero_stars():
    """Problem 4: a sparse matrix of 5-star ratings must predict near 5, not near 0."""
    rng = np.random.default_rng(0)
    users = np.repeat(np.arange(30), 4)
    items = rng.integers(0, 40, size=users.size)
    mf = BiasedMF(factors=3, iterations=5, seed=0).fit(
        users.astype(str), items.astype(str), np.full(users.size, 5.0))
    pred = mf.biases.mu + mf.biases.item + mf.item_f @ mf.user_f[0]
    assert pred.min() > 4.5


def test_biases_recover_a_harsh_user():
    u = np.array([0, 0, 1, 1, 2, 2])
    i = np.array([0, 1, 0, 1, 0, 1])
    r = np.array([2.0, 2.0, 4.0, 4.0, 4.0, 4.0])
    b = fit_biases(u, i, r, 3, 2, reg_user=0.0, reg_item=0.0, n_iter=20)
    assert b.user[0] < 0 < b.user[1]


def test_als_reduces_the_train_error():
    rng = np.random.default_rng(1)
    uf, vf = rng.normal(size=(50, 3)), rng.normal(size=(40, 3))
    u = rng.integers(0, 50, 900)
    i = rng.integers(0, 40, 900)
    res = np.einsum("ij,ij->i", uf[u], vf[i])
    _, _, hist = als(u, i, res, 50, 40, factors=3, reg=0.01, iterations=8, seed=0)
    assert hist[-1] < 0.5 * hist[0]
    assert all(b <= a + 1e-9 for a, b in zip(hist, hist[1:]))


def test_fold_in_recovers_a_known_user_vector():
    rng = np.random.default_rng(2)
    item_f = rng.normal(size=(30, 4))
    true = np.array([0.5, -1.0, 0.2, 0.8])
    items = np.arange(30)
    est = fold_in(item_f, items, item_f @ true, reg=1e-4)
    assert np.allclose(est, true, atol=0.05)
    assert np.all(fold_in(item_f, np.array([], dtype=int), np.array([]), reg=0.1) == 0)


def test_biased_mf_fold_in_ignores_unknown_items():
    mf = BiasedMF(factors=2, iterations=2).fit(["a", "b"], ["x", "y"], [4.0, 2.0])
    vec, bu = mf.fold_in(["nope"], [5.0])
    assert vec.shape == (2,) and bu == 0.0
    with pytest.raises(KeyError):
        mf.items.encode(["nope"])
