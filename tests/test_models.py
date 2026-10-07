import numpy as np
import pandas as pd
import pytest

from crossrec.config import Settings
from crossrec.data import CrossDomainData
from crossrec.evaluate import evaluate_model
from crossrec.models import MODELS, NotFittedError, build, load
from crossrec.split import ColdStartSplit, TrainData, cold_start_split, full_train

S = Settings(seed=7)


@pytest.fixture(scope="module")
def split(core):
    return cold_start_split(core, "food", "books", test_fraction=0.2, seed=7)


@pytest.fixture(scope="module")
def results(split):
    out = {}
    for name in MODELS:
        model = build(name, S).fit(split.train)
        out[name] = evaluate_model(model, split, k=10, seed=7)
    return out


def test_transfer_models_beat_the_single_domain_baselines(results):
    """Problems 1 and 3: shared or mapped user factors must carry taste across domains."""
    for name in ("cmf", "emcdr"):
        assert results[name].means["ndcg"] > results["itembias"].means["ndcg"] + 0.03
        assert results[name].means["ndcg"] > results["popularity"].means["ndcg"]
        assert results[name].rmse < results["itembias"].rmse - 0.03


def test_cmf_needs_a_real_source_history(split):
    """Problem 1: if the source ratings carry no taste, the transfer advantage goes away."""
    real = evaluate_model(build("cmf", S).fit(split.train), split, seed=7).means["ndcg"]
    shuffled = split.train.source_ratings.copy()
    shuffled["rating"] = np.random.default_rng(0).permutation(shuffled["rating"].to_numpy())
    noisy = ColdStartSplit(TrainData("food", "books", shuffled, split.train.target_ratings),
                           split.test_target, split.train_users, split.test_users, split.seed)
    fake = evaluate_model(build("cmf", S).fit(noisy.train), noisy, seed=7).means["ndcg"]
    assert real > fake + 0.03


def test_more_items_than_users_does_not_fail():
    """Problem 2: item and user axes are never swapped, also when items outnumber users."""
    rng = np.random.default_rng(0)
    users = [f"u{n}" for n in range(6)]
    rows = {d: [] for d in ("food", "books")}
    for d, n_items in (("food", 40), ("books", 60)):
        for u in users:
            for it in rng.choice(n_items, 12, replace=False):
                rows[d].append((u, f"{d}{it}", float(rng.integers(1, 6)), 0))
    frames = {d: pd.DataFrame(r, columns=["user_id", "item_id", "rating", "timestamp"])
              for d, r in rows.items()}
    data = CrossDomainData(ratings=frames)
    for name in ("cmf", "emcdr"):
        model = build(name, factors=3, iterations=3).fit(full_train(data, "food", "books"))
        assert model.score_target("u0").shape == (len(model.target_items),)
        recs = model.recommend("u0", k=5)
        assert len(recs) == 5
        assert all(it.startswith("books") for it, _ in recs)


def test_recommend_excludes_rated_items_and_uses_real_ids(core):
    """Problem 6: real user IDs in, no already rated items out."""
    model = build("cmf", S).fit(full_train(core, "food", "books"))
    user = sorted(core.users("books"))[0]
    books = core.ratings["books"]
    seen = set(books.loc[books["user_id"] == user, "item_id"])
    recs = model.recommend(user, k=20, exclude=seen)
    assert len(recs) == 20
    assert not seen & {it for it, _ in recs}
    scores = [s for _, s in recs]
    assert scores == sorted(scores, reverse=True)


def test_stored_vector_and_fold_in_agree(split):
    model = build("emcdr", S).fit(split.train)
    user = split.test_users[0]
    src = split.train.source_ratings
    hist = src[src["user_id"] == user]
    stored = model.score_target(user)
    folded = model.score_target("someone-new", hist)
    assert np.corrcoef(stored, folded)[0, 1] > 0.8


def test_unknown_user_without_history_gets_the_baseline_order(split):
    model = build("cmf", S).fit(split.train)
    base = model.target_bias.mu + model.target_bias.item
    assert np.allclose(model.score_target("nobody"), base)


def test_predictions_are_clipped_and_unknown_items_get_the_mean(split):
    model = build("cmf", S).fit(split.train)
    pred = model.predict("nobody", ["not-an-item", split.train.target_ratings.item_id.iloc[0]])
    assert pred[0] == pytest.approx(model.target_bias.mu)
    assert 1.0 <= pred.min() and pred.max() <= 5.0


@pytest.mark.parametrize("name", sorted(MODELS))
def test_save_and_load_give_the_same_scores(name, split, tmp_path):
    model = build(name, S).fit(split.train)
    user = split.test_users[0]
    src = split.train.source_ratings
    hist = src[src["user_id"] == user]
    model.save(tmp_path / name)
    again = load(tmp_path / name)
    assert again.name == name
    assert np.allclose(model.score_target("x", hist), again.score_target("x", hist))
    assert model.recommend(user, 5, hist) == again.recommend(user, 5, hist)


def test_unfitted_model_raises():
    with pytest.raises(NotFittedError):
        build("cmf").recommend("u", 3)
    with pytest.raises(KeyError):
        build("nope")


def test_emcdr_mlp_mapping(split):
    pytest.importorskip("sklearn")
    model = build("emcdr", S, mapping="mlp").fit(split.train)
    res = evaluate_model(model, split, seed=7)
    assert res.users_ranked > 0 and np.isfinite(res.means["ndcg"])
