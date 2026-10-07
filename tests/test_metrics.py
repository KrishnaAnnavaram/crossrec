import math

import pytest

from crossrec import metrics as m


def test_ranking_metrics_known_values():
    ranked = ["a", "b", "c", "d"]
    rel = {"b", "d", "z"}
    assert m.recall_at_k(ranked, rel, 4) == pytest.approx(2 / 3)
    assert m.precision_at_k(ranked, rel, 4) == pytest.approx(0.5)
    assert m.hit_rate_at_k(ranked, rel, 1) == 0.0
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert m.ndcg_at_k(ranked, rel, 4) == pytest.approx(dcg / idcg)
    assert m.average_precision_at_k(ranked, rel, 4) == pytest.approx((1 / 2 + 2 / 4) / 3)


def test_perfect_ranking_scores_one():
    assert m.ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)
    assert m.recall_at_k(["a", "b"], {"a", "b"}, 2) == 1.0


def test_empty_relevant_set_raises():
    with pytest.raises(ValueError):
        m.ndcg_at_k(["a"], set(), 1)


def test_rating_metrics_and_bootstrap():
    assert m.rmse([1, 2], [1, 4]) == pytest.approx(math.sqrt(2))
    assert m.mae([1, 2], [1, 4]) == pytest.approx(1.0)
    lo, hi = m.bootstrap_ci([0.2] * 10 + [0.4] * 10, seed=1)
    assert lo <= 0.3 <= hi
    assert m.bootstrap_ci([0.2, 0.4], seed=1) == m.bootstrap_ci([0.2, 0.4], seed=1)
