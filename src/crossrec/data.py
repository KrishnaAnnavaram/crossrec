"""Loaders for the two public review files and the in-memory cross-domain container.

The loaders read only the columns that the pipeline uses, in chunks. The books file is about 2.9 GB,
so the books loader keeps only users that also rated food. This builds the user overlap first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .schema import SchemaError, ValidationReport, merge_duplicates, validate_ratings

FOOD_FILE = "Reviews.csv"
BOOKS_FILE = "Books_rating.csv"

REQUIRED_CANONICAL = ("user_id", "item_id", "rating")

# Raw column name -> canonical column name.
FOOD_COLUMNS = {"UserId": "user_id", "ProductId": "item_id", "Score": "rating", "Time": "timestamp"}
BOOKS_COLUMNS = {
    "User_id": "user_id",
    "Id": "item_id",
    "review/score": "rating",
    "review/time": "timestamp",
    "Title": "title",
}


@dataclass
class CrossDomainData:
    """Validated ratings for each domain, plus display titles keyed by item ID."""

    ratings: dict[str, pd.DataFrame]
    titles: dict[str, dict[str, str]] = field(default_factory=dict)
    reports: dict[str, ValidationReport] = field(default_factory=dict)

    def domains(self) -> list[str]:
        return list(self.ratings)

    def users(self, domain: str) -> set[str]:
        return set(self.ratings[domain]["user_id"].unique())

    def title(self, domain: str, item_id: str) -> str:
        return self.titles.get(domain, {}).get(item_id, item_id)

    def summary(self) -> dict:
        out = {}
        for d, df in self.ratings.items():
            out[d] = {"ratings": len(df), "users": df["user_id"].nunique(), "items": df["item_id"].nunique()}
        return out


def _read_chunks(path: Path, columns: dict[str, str], chunksize: int, users: set[str] | None):
    header = pd.read_csv(path, nrows=0).columns
    required = [raw for raw, canon in columns.items() if canon in REQUIRED_CANONICAL]
    missing = [c for c in required if c not in header]
    if missing:
        raise SchemaError(f"{path.name}: missing columns {missing}")
    use = [c for c in columns if c in header]
    for chunk in pd.read_csv(path, usecols=use, chunksize=chunksize, dtype=str):
        chunk = chunk.rename(columns=columns)
        if users is not None:
            chunk = chunk[chunk["user_id"].isin(users)]
        yield chunk


def load_domain(path: str | Path, columns: dict[str, str], domain: str, *,
                users: set[str] | None = None, chunksize: int = 200_000):
    """Load one raw file into the canonical schema. Return (ratings, titles, report)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md for the download steps.")
    parts, titles, report = [], {}, ValidationReport(domain=domain)
    for chunk in _read_chunks(path, columns, chunksize, users):
        if "title" in chunk.columns:
            named = chunk.dropna(subset=["item_id", "title"])
            titles.update(zip(named["item_id"].str.strip(), named["title"].str.strip()))
        clean, part = validate_ratings(chunk, domain)
        report.add(part)
        parts.append(clean)
    df = pd.concat(parts, ignore_index=True) if parts else validate_ratings(
        pd.DataFrame(columns=["user_id", "item_id", "rating"]), domain)[0]
    # Duplicates can sit in two different chunks, so merge again over the full domain.
    df, merged = merge_duplicates(df)
    report.duplicates_merged += merged
    report.rows_out = len(df)
    return df, titles, report


def load_reviews(data_dir: str | Path, chunksize: int = 200_000) -> CrossDomainData:
    """Load both domains from ``data_dir``. Books rows are kept only for users who rated food."""
    data_dir = Path(data_dir)
    food, food_titles, food_rep = load_domain(data_dir / FOOD_FILE, FOOD_COLUMNS, "food", chunksize=chunksize)
    food_users = set(food["user_id"].unique())
    books, book_titles, books_rep = load_domain(
        data_dir / BOOKS_FILE, BOOKS_COLUMNS, "books", users=food_users, chunksize=chunksize
    )
    books_rep.notes.append("rows kept only for users who also rated food")
    return CrossDomainData(
        ratings={"food": food, "books": books},
        titles={"food": food_titles, "books": book_titles},
        reports={"food": food_rep, "books": books_rep},
    )
