"""Command line: ``crossrec synth | stats | evaluate | train | recommend``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import DOMAINS
from .config import Settings
from .data import CrossDomainData, load_reviews
from .evaluate import evaluate_model, results_table
from .models import MODELS, build, load
from .overlap import prepare, sample_then_intersect
from .split import cold_start_split, full_train
from .synthetic import SyntheticSpec, generate, write_raw_csv

# The independent row fractions of the old notebook, used only for the comparison in ``stats``.
OLD_FRACTIONS = {"food": 0.05, "books": 0.005}


def _settings(args) -> Settings:
    s = Settings.from_env()
    if getattr(args, "config", None):
        s = s.merge_toml(args.config)
    over = {k: getattr(args, k, None) for k in ("data_dir", "seed", "factors", "reg", "iterations", "k",
                                                 "min_user_ratings", "min_item_ratings", "max_users")}
    return s.merge(**over).validate()


def _load(args, s: Settings) -> CrossDomainData:
    if getattr(args, "synthetic", False):
        return generate(SyntheticSpec(seed=s.seed))
    return load_reviews(s.data_dir)


def _prepared(args, s: Settings):
    data = _load(args, s)
    return prepare(data, s.min_user_ratings, s.min_item_ratings, s.max_users, s.seed)


def cmd_synth(args) -> int:
    data = generate(SyntheticSpec(n_users=args.users, seed=args.seed))
    paths = write_raw_csv(data, args.out)
    for d, p in paths.items():
        print(f"{d}: {p} ({len(data.ratings[d])} ratings)")
    return 0


def cmd_stats(args) -> int:
    s = _settings(args)
    data = _load(args, s)
    for d, rep in data.reports.items():
        print(f"validation {d}: {rep.as_dict()}")
    print(f"before filters: {data.summary()}")
    old = sample_then_intersect(data, OLD_FRACTIONS, s.seed)
    core, rep = prepare(data, s.min_user_ratings, s.min_item_ratings, s.max_users, s.seed)
    print(f"users in both domains: {rep.overlap_users}")
    print(f"shared users after an independent 5% / 0.5% row sample: {old}")
    print(f"after k-core (user>={s.min_user_ratings} in each domain, item>={s.min_item_ratings}), "
          f"{rep.kcore_rounds} rounds: {rep.after_kcore}")
    return 0


def cmd_evaluate(args) -> int:
    s = _settings(args)
    core, rep = _prepared(args, s)
    split = cold_start_split(core, args.source, args.target, s.test_fraction, s.seed)
    print(f"data after k-core: {rep.after_kcore}")
    print(f"cold-start split: {len(split.train_users)} train users, {len(split.test_users)} test users "
          f"({args.source} -> {args.target}), seed {s.seed}")
    results = []
    for name in args.models.split(","):
        model = build(name.strip(), s).fit(split.train)
        results.append(evaluate_model(model, split, k=s.k, relevant_threshold=s.relevant_threshold, seed=s.seed))
    with pd.option_context("display.width", 140, "display.precision", 4):
        print(results_table(results))
    if args.out:
        payload = {
            "source": args.source, "target": args.target, "seed": s.seed, "data": rep.after_kcore,
            "synthetic": bool(args.synthetic), "test_users": len(split.test_users),
            "results": [r.as_dict() for r in results],
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"report: {args.out}")
    return 0


def cmd_train(args) -> int:
    s = _settings(args)
    core, rep = _prepared(args, s)
    model = build(args.model, s).fit(full_train(core, args.source, args.target))
    out = model.save(args.out)
    print(f"{args.model} ({args.source} -> {args.target}) trained on {rep.after_kcore}. Saved to {out}")
    return 0


def cmd_recommend(args) -> int:
    s = _settings(args)
    model = load(args.model_dir)
    data = _load(args, s)
    src = data.ratings[model.source]
    tgt = data.ratings[model.target]
    history = src[src["user_id"] == args.user]
    seen = set(tgt.loc[tgt["user_id"] == args.user, "item_id"])
    if history.empty and args.user not in model.target_users:
        print(f"user {args.user!r} has no {model.source} ratings and is not in the model", file=sys.stderr)
        return 2
    recs = model.recommend(args.user, k=s.k, source_history=history, exclude=seen)
    print(f"user {args.user}: {len(history)} {model.source} ratings, {len(seen)} {model.target} ratings "
          f"(excluded). Model: {model.name}")
    for rank, (item, score) in enumerate(recs, 1):
        print(f"{rank:>2}. {item}  {data.title(model.target, item)}  score={score:.3f}")
    return 0


def _data_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", help="folder with Reviews.csv and Books_rating.csv (CROSSREC_DATA_DIR)")
    p.add_argument("--synthetic", action="store_true", help="use the built-in synthetic data")
    p.add_argument("--config", help="TOML file with a [crossrec] table")
    p.add_argument("--seed", type=int)
    p.add_argument("--k", type=int)
    p.add_argument("--min-user-ratings", type=int)
    p.add_argument("--min-item-ratings", type=int)
    p.add_argument("--max-users", type=int)


def _model_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--factors", type=int)
    p.add_argument("--reg", type=float)
    p.add_argument("--iterations", type=int)
    p.add_argument("--source", choices=DOMAINS, default="food")
    p.add_argument("--target", choices=DOMAINS, default="books")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="crossrec", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="write synthetic CSV files with the real column names")
    p.add_argument("--out", required=True)
    p.add_argument("--users", type=int, default=SyntheticSpec.n_users)
    p.add_argument("--seed", type=int, default=SyntheticSpec.seed)
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("stats", help="validation counts, overlap and k-core sizes")
    _data_flags(p)
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("evaluate", help="cold-start evaluation of one or more models")
    _data_flags(p)
    _model_flags(p)
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--out", help="write a JSON report")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("train", help="fit one model on all data and save it")
    _data_flags(p)
    _model_flags(p)
    p.add_argument("--model", choices=sorted(MODELS), default="cmf")
    p.add_argument("--out", required=True, help="output folder (for example models/cmf)")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("recommend", help="top-k target items for one user ID")
    _data_flags(p)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--user", required=True, help="the real user ID, for example A3SGXH7AUHU8GW")
    p.set_defaults(fn=cmd_recommend)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if getattr(args, "source", None) and args.source == args.target:
        print("error: --source and --target must be different", file=sys.stderr)
        return 2
    try:
        return args.fn(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
