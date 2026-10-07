"""Canonical rating schema and its validation.

Every loader returns a frame with the columns in RATING_COLUMNS. Items are keyed by a stable
product ID (ASIN), never by a display title, so two books with the same title stay two items.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

RATING_COLUMNS = ("user_id", "item_id", "rating", "timestamp")
MIN_RATING = 1.0
MAX_RATING = 5.0


class SchemaError(ValueError):
    """The input frame does not have the expected columns."""


@dataclass
class ValidationReport:
    domain: str
    rows_in: int = 0
    missing_id_or_rating: int = 0
    rating_out_of_range: int = 0
    duplicates_merged: int = 0
    rows_out: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "domain": self.domain,
            "rows_in": self.rows_in,
            "missing_id_or_rating": self.missing_id_or_rating,
            "rating_out_of_range": self.rating_out_of_range,
            "duplicates_merged": self.duplicates_merged,
            "rows_out": self.rows_out,
        }

    def add(self, other: "ValidationReport") -> None:
        self.rows_in += other.rows_in
        self.missing_id_or_rating += other.missing_id_or_rating
        self.rating_out_of_range += other.rating_out_of_range
        self.duplicates_merged += other.duplicates_merged


def validate_ratings(frame: pd.DataFrame, domain: str) -> tuple[pd.DataFrame, ValidationReport]:
    """Clean one domain: drop bad rows, average duplicate (user, item) ratings, keep the last time."""
    report = ValidationReport(domain=domain, rows_in=len(frame))
    missing = [c for c in ("user_id", "item_id", "rating") if c not in frame.columns]
    if missing:
        raise SchemaError(f"{domain}: missing columns {missing}")
    df = frame.copy()
    if "timestamp" not in df.columns:
        df["timestamp"] = 0
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype("int64")
    for col in ("user_id", "item_id"):
        df[col] = df[col].astype("string").str.strip()
    bad = df["user_id"].isna() | df["item_id"].isna() | df["rating"].isna()
    bad |= (df["user_id"] == "") | (df["item_id"] == "")
    report.missing_id_or_rating = int(bad.sum())
    df = df[~bad]
    out_of_range = (df["rating"] < MIN_RATING) | (df["rating"] > MAX_RATING)
    report.rating_out_of_range = int(out_of_range.sum())
    df = df[~out_of_range]
    clean, merged = merge_duplicates(df)
    report.duplicates_merged = merged
    report.rows_out = len(clean)
    return clean, report


def merge_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Average repeated ratings of one user for one item. Return the frame and the merged count."""
    if df.empty:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in
                             zip(RATING_COLUMNS, ("string", "string", "float64", "int64"))}), 0
    grouped = (
        df.groupby(["user_id", "item_id"], sort=True, observed=True)
        .agg(rating=("rating", "mean"), timestamp=("timestamp", "max"))
        .reset_index()
    )
    grouped["user_id"] = grouped["user_id"].astype(str)
    grouped["item_id"] = grouped["item_id"].astype(str)
    return grouped[list(RATING_COLUMNS)], len(df) - len(grouped)
