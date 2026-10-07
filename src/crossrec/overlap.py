"""Overlap-first filters: shared users, a joint k-core and a seeded user subsample.

The order is fixed: find all users in both domains, apply the k-core filter, and only then
subsample users. A subsample before the overlap step loses most shared users.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import CrossDomainData


@dataclass
class OverlapReport:
    users_per_domain: dict[str, int]
    overlap_users: int
    after_kcore: dict
    kcore_rounds: int


def overlap_users(data: CrossDomainData, domains: tuple[str, str] | None = None) -> set[str]:
    a, b = domains or tuple(data.domains()[:2])
    return data.users(a) & data.users(b)


def kcore(data: CrossDomainData, min_user: int, min_item: int, max_rounds: int = 100):
    """Keep users with >= ``min_user`` ratings in EACH domain and items with >= ``min_item`` ratings.

    Repeat until nothing changes. Return the filtered data and the number of rounds.
    """
    frames = {d: df.copy() for d, df in data.ratings.items()}
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        before = sum(len(f) for f in frames.values())
        keep = None
        for df in frames.values():
            counts = df.groupby("user_id").size()
            ok = set(counts[counts >= min_user].index)
            keep = ok if keep is None else keep & ok
        for d, df in frames.items():
            df = df[df["user_id"].isin(keep)]
            item_counts = df.groupby("item_id")["user_id"].transform("size")
            frames[d] = df[item_counts >= min_item]
        if sum(len(f) for f in frames.values()) == before:
            break
    frames = {d: f.reset_index(drop=True) for d, f in frames.items()}
    return CrossDomainData(ratings=frames, titles=data.titles, reports=data.reports), rounds


def subsample_users(data: CrossDomainData, max_users: int, seed: int) -> CrossDomainData:
    """Keep a seeded random subset of ``max_users`` users. 0 keeps every user."""
    users = sorted(set().union(*(data.users(d) for d in data.domains())))
    if max_users <= 0 or max_users >= len(users):
        return data
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(users, size=max_users, replace=False))
    frames = {d: df[df["user_id"].isin(chosen)].reset_index(drop=True) for d, df in data.ratings.items()}
    return CrossDomainData(ratings=frames, titles=data.titles, reports=data.reports)


def prepare(data: CrossDomainData, min_user: int, min_item: int, max_users: int, seed: int):
    """Run overlap -> k-core -> subsample -> k-core. Return (data, OverlapReport)."""
    per_domain = {d: len(data.users(d)) for d in data.domains()}
    shared = overlap_users(data)
    core, rounds = kcore(data, min_user, min_item)
    if max_users > 0:
        core = subsample_users(core, max_users, seed)
        core, extra = kcore(core, min_user, min_item)
        rounds += extra
    return core, OverlapReport(per_domain, len(shared), core.summary(), rounds)


def sample_then_intersect(data: CrossDomainData, fractions: dict[str, float], seed: int) -> int:
    """Count shared users after an independent row sample in each domain (the old approach)."""
    sampled = {}
    for d, df in data.ratings.items():
        frac = fractions.get(d, 1.0)
        sampled[d] = set(df.sample(frac=frac, random_state=seed)["user_id"]) if frac < 1 else set(df["user_id"])
    sets = list(sampled.values())
    return len(sets[0] & sets[1])

