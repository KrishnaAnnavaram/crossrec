"""Synthetic two-domain review data with a known shared user taste.

Each user has one taste vector. The same vector drives the food ratings and the book ratings, so a
model that transfers taste across domains can beat popularity. Some users rate only one domain.
Two different books share one title, to test that items are keyed by ID and not by title.
The writer produces CSV files with the same column names as the real public files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import BOOKS_FILE, FOOD_FILE, CrossDomainData
from .schema import validate_ratings

SHARED_TITLE = "Collected Poems"


@dataclass(frozen=True)
class SyntheticSpec:
    n_users: int = 600
    n_food: int = 120
    n_books: int = 160
    dim: int = 4
    ratings_per_user: int = 18
    only_food_fraction: float = 0.15
    only_books_fraction: float = 0.15
    noise: float = 0.5
    seed: int = 42


def _domain_ratings(rng, taste, items, item_bias, users, n_per_user, noise):
    rows = []
    n_items = items.shape[0]
    popularity = 1.0 / np.arange(1, n_items + 1) ** 0.2
    popularity = popularity[rng.permutation(n_items)]
    for u in users:
        affinity = items @ taste[u]
        weights = popularity * np.exp(0.6 * affinity)
        weights /= weights.sum()
        n = int(np.clip(rng.poisson(n_per_user), 4, n_items))
        chosen = rng.choice(n_items, size=n, replace=False, p=weights)
        raw = 3.6 + 0.9 * affinity[chosen] + item_bias[chosen] + rng.normal(0, noise, size=n)
        stars = np.clip(np.rint(raw), 1, 5)
        for j, s in zip(chosen, stars):
            rows.append((u, j, float(s)))
    return rows


def generate(spec: SyntheticSpec = SyntheticSpec()) -> CrossDomainData:
    """Make validated ratings for the domains ``food`` and ``books``."""
    rng = np.random.default_rng(spec.seed)
    taste = rng.normal(0, 1, size=(spec.n_users, spec.dim)) / np.sqrt(spec.dim)
    food_items = rng.normal(0, 1, size=(spec.n_food, spec.dim))
    book_items = rng.normal(0, 1, size=(spec.n_books, spec.dim))
    food_bias = rng.normal(0, 0.3, size=spec.n_food)
    book_bias = rng.normal(0, 0.3, size=spec.n_books)

    order = rng.permutation(spec.n_users)
    n_food_only = int(spec.only_food_fraction * spec.n_users)
    n_books_only = int(spec.only_books_fraction * spec.n_users)
    food_users = np.concatenate([order[:n_food_only], order[n_food_only + n_books_only:]])
    book_users = order[n_food_only:]

    user_ids = [f"U{u:05d}" for u in range(spec.n_users)]
    food_ids = [f"F{j:05d}" for j in range(spec.n_food)]
    book_ids = [f"{1000000000 + 7919 * j}" for j in range(spec.n_books)]
    t0 = 1_300_000_000

    frames, titles = {}, {"food": {}, "books": {}}
    for domain, items, bias, users, ids in (
        ("food", food_items, food_bias, food_users, food_ids),
        ("books", book_items, book_bias, book_users, book_ids),
    ):
        rows = _domain_ratings(rng, taste, items, bias, users, spec.ratings_per_user, spec.noise)
        df = pd.DataFrame(
            {
                "user_id": [user_ids[u] for u, _, _ in rows],
                "item_id": [ids[j] for _, j, _ in rows],
                "rating": [r for _, _, r in rows],
                "timestamp": t0 + rng.integers(0, 10_000_000, size=len(rows)),
            }
        )
        frames[domain], _ = validate_ratings(df, domain)
        for j, item in enumerate(ids):
            titles[domain][item] = f"{'Pantry item' if domain == 'food' else 'Book'} {j:03d}"
    # Two different books with one title: they must stay two items.
    titles["books"][book_ids[0]] = SHARED_TITLE
    titles["books"][book_ids[1]] = SHARED_TITLE
    return CrossDomainData(ratings=frames, titles=titles)


def write_raw_csv(data: CrossDomainData, out_dir: str | Path) -> dict[str, Path]:
    """Write the synthetic data with the column names of the real public files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    food = data.ratings["food"]
    food_raw = pd.DataFrame(
        {
            "Id": np.arange(1, len(food) + 1),
            "ProductId": food["item_id"],
            "UserId": food["user_id"],
            "Score": food["rating"].round().astype(int),
            "Time": food["timestamp"],
            "Summary": "synthetic",
            "Text": "synthetic review",
        }
    )
    books = data.ratings["books"]
    books_raw = pd.DataFrame(
        {
            "Id": books["item_id"],
            "Title": [data.title("books", i) for i in books["item_id"]],
            "Price": "",
            "User_id": books["user_id"],
            "profileName": "synthetic",
            "review/helpfulness": "0/0",
            "review/score": books["rating"],
            "review/time": books["timestamp"],
            "review/summary": "synthetic",
            "review/text": "synthetic review",
        }
    )
    paths = {"food": out / FOOD_FILE, "books": out / BOOKS_FILE}
    food_raw.to_csv(paths["food"], index=False)
    books_raw.to_csv(paths["books"], index=False)
    return paths
