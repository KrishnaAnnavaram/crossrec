import pandas as pd
import pytest

from crossrec.data import load_domain, load_reviews, BOOKS_COLUMNS
from crossrec.schema import SchemaError, validate_ratings
from crossrec.synthetic import SHARED_TITLE, write_raw_csv


def test_validation_drops_bad_rows_and_merges_duplicates():
    df = pd.DataFrame({
        "user_id": ["a", "a", "b", None, "c", " "],
        "item_id": ["x", "x", "y", "z", "z", "q"],
        "rating": [4, 2, 9, 5, "bad", 3],
        "timestamp": [1, 5, 2, 3, 4, 6],
    })
    clean, rep = validate_ratings(df, "food")
    assert rep.missing_id_or_rating == 3  # None user, non-numeric rating, blank user
    assert rep.rating_out_of_range == 1
    assert rep.duplicates_merged == 1
    assert len(clean) == 1
    row = clean.iloc[0]
    assert (row.user_id, row.item_id, row.rating, row.timestamp) == ("a", "x", 3.0, 5)


def test_missing_column_raises():
    with pytest.raises(SchemaError):
        validate_ratings(pd.DataFrame({"user_id": ["a"], "rating": [3]}), "food")


def test_round_trip_through_real_column_names(synth, tmp_path):
    write_raw_csv(synth, tmp_path)
    data = load_reviews(tmp_path, chunksize=500)  # small chunks: duplicates can cross chunks
    assert len(data.ratings["food"]) == len(synth.ratings["food"])
    books = synth.ratings["books"]
    assert len(data.ratings["books"]) == books["user_id"].isin(synth.users("food")).sum()
    assert data.ratings["food"]["rating"].sum() == synth.ratings["food"]["rating"].sum()


def test_books_loader_keeps_only_users_who_rated_food(synth, tmp_path):
    write_raw_csv(synth, tmp_path)
    data = load_reviews(tmp_path)
    assert data.users("books") <= data.users("food")
    assert len(synth.users("books") - synth.users("food")) > 0  # some users were dropped


def test_books_are_keyed_by_id_not_title(synth, tmp_path):
    """Problem 8: two different books with one title must stay two items."""
    write_raw_csv(synth, tmp_path)
    data = load_reviews(tmp_path)
    same = [i for i, t in data.titles["books"].items() if t == SHARED_TITLE]
    assert len(same) == 2
    items = set(data.ratings["books"]["item_id"])
    assert set(same) <= items | set(synth.ratings["books"]["item_id"])


def test_duplicates_in_different_chunks_are_merged(tmp_path):
    raw = pd.DataFrame({
        "Id": ["B1", "B2", "B3", "B1"], "Title": ["t1", "t2", "t3", "t1"],
        "User_id": ["u", "v", "w", "u"], "review/score": [5, 3, 4, 3], "review/time": [1, 2, 3, 4],
    })
    path = tmp_path / "Books_rating.csv"
    raw.to_csv(path, index=False)
    df, titles, rep = load_domain(path, BOOKS_COLUMNS, "books", chunksize=2)
    assert len(df) == 3
    assert df.loc[(df.user_id == "u") & (df.item_id == "B1"), "rating"].item() == 4.0
    assert rep.duplicates_merged == 1
    assert titles["B1"] == "t1"


def test_missing_raw_column_and_missing_file(tmp_path):
    path = tmp_path / "Books_rating.csv"
    pd.DataFrame({"Id": ["B1"], "review/score": [5]}).to_csv(path, index=False)
    with pytest.raises(SchemaError):
        load_domain(path, BOOKS_COLUMNS, "books")
    with pytest.raises(FileNotFoundError):
        load_reviews(tmp_path / "nothing")
