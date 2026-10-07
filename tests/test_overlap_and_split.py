import pytest

from crossrec.overlap import kcore, overlap_users, prepare, sample_then_intersect, subsample_users
from crossrec.split import LeakageError, cold_start_split


def test_kcore_meets_both_minimums(synth):
    core, rounds = kcore(synth, min_user=10, min_item=15)
    assert rounds >= 1
    users = None
    for d, df in core.ratings.items():
        per_user = df.groupby("user_id").size()
        per_item = df.groupby("item_id").size()
        assert per_user.min() >= 10
        assert per_item.min() >= 15
        users = set(per_user.index) if users is None else users
        assert set(per_user.index) == users  # every user is in both domains


def test_overlap_first_keeps_more_shared_users_than_sample_first(synth):
    """Problem 5: an independent sample in each domain loses most shared users."""
    shared = len(overlap_users(synth))
    old = sample_then_intersect(synth, {"food": 0.05, "books": 0.005}, seed=1)
    _, rep = prepare(synth, 3, 3, max_users=0, seed=1)
    assert rep.overlap_users == shared
    assert old < shared / 5


def test_subsample_is_seeded_and_after_overlap(core):
    a = subsample_users(core, 50, seed=3)
    b = subsample_users(core, 50, seed=3)
    assert a.users("food") == b.users("food")
    assert len(a.users("food") | a.users("books")) == 50
    assert subsample_users(core, 0, seed=3) is core


def test_cold_start_split_hides_all_target_ratings_of_test_users(core):
    split = cold_start_split(core, "food", "books", test_fraction=0.25, seed=5)
    test = set(split.test_users)
    assert test.isdisjoint(split.train.target_ratings["user_id"])
    assert set(split.test_target["user_id"]) == test
    # Test users keep their source history.
    assert test <= set(split.train.source_ratings["user_id"])
    total = len(split.train.target_ratings) + len(split.test_target)
    assert total == len(core.ratings["books"])


def test_split_is_reproducible_and_seed_dependent(core):
    a = cold_start_split(core, "food", "books", seed=1).test_users
    b = cold_start_split(core, "food", "books", seed=1).test_users
    c = cold_start_split(core, "food", "books", seed=2).test_users
    assert a == b and a != c


def test_leakage_check_detects_a_leak(core):
    split = cold_start_split(core, "food", "books", seed=1)
    import pandas as pd

    split.train.target_ratings = pd.concat([split.train.target_ratings, split.test_target.head(1)])
    with pytest.raises(LeakageError):
        split.check_no_leakage()


def test_split_rejects_same_domain(core):
    with pytest.raises(ValueError):
        cold_start_split(core, "food", "food")
