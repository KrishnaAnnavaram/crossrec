"""The cold-start split for cross-domain evaluation.

Test users keep all their source-domain ratings. All their target-domain ratings are held out.
Train users keep the ratings of both domains. No test-user target rating is in the train data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import CrossDomainData


class LeakageError(AssertionError):
    """A held-out rating is visible to the model."""


@dataclass
class TrainData:
    """What a model can see at fit time."""

    source: str
    target: str
    source_ratings: pd.DataFrame
    target_ratings: pd.DataFrame


@dataclass
class ColdStartSplit:
    train: TrainData
    test_target: pd.DataFrame
    train_users: list[str]
    test_users: list[str]
    seed: int

    def check_no_leakage(self) -> None:
        test = set(self.test_users)
        seen = set(self.train.target_ratings["user_id"])
        if seen & test:
            raise LeakageError(f"{len(seen & test)} test users have target ratings in the train data")
        if set(self.test_target["user_id"]) - test:
            raise LeakageError("held-out ratings belong to users who are not test users")
        if set(self.train_users) & test:
            raise LeakageError("a user is in the train set and the test set")


def cold_start_split(data: CrossDomainData, source: str, target: str,
                     test_fraction: float = 0.2, seed: int = 42) -> ColdStartSplit:
    """Select test users from the users of both domains (seeded) and hide their target ratings."""
    if source == target:
        raise ValueError("source and target must be different domains")
    for d in (source, target):
        if d not in data.ratings:
            raise KeyError(f"unknown domain {d!r}. Known: {data.domains()}")
    shared = sorted(data.users(source) & data.users(target))
    if len(shared) < 2:
        raise ValueError("need at least 2 users with ratings in both domains")
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(test_fraction * len(shared))))
    test_users = sorted(rng.choice(shared, size=n_test, replace=False).tolist())
    test_set = set(test_users)
    tgt = data.ratings[target]
    is_test = tgt["user_id"].isin(test_set)
    split = ColdStartSplit(
        train=TrainData(
            source=source,
            target=target,
            source_ratings=data.ratings[source].reset_index(drop=True),
            target_ratings=tgt[~is_test].reset_index(drop=True),
        ),
        test_target=tgt[is_test].reset_index(drop=True),
        train_users=sorted(set(shared) - test_set),
        test_users=test_users,
        seed=seed,
    )
    split.check_no_leakage()
    return split


def full_train(data: CrossDomainData, source: str, target: str) -> TrainData:
    """All ratings of both domains, for the final model that serves recommendations."""
    return TrainData(source, target, data.ratings[source], data.ratings[target])
