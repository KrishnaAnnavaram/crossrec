"""Model registry: build a model by name, or load a saved model directory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import ARRAYS_FILE, META_FILE, NotFittedError, Recommender
from .baselines import ItemBiasRecommender, PopularityRecommender
from .cmf import CMFRecommender
from .emcdr import EMCDRRecommender

MODELS: dict[str, type[Recommender]] = {
    "popularity": PopularityRecommender,
    "itembias": ItemBiasRecommender,
    "cmf": CMFRecommender,
    "emcdr": EMCDRRecommender,
}


def build(name: str, settings=None, **extra) -> Recommender:
    """Make an unfitted model. ``settings`` is a ``crossrec.config.Settings``."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}. Known: {sorted(MODELS)}")
    kw: dict = {}
    if settings is not None:
        if name in ("cmf", "emcdr"):
            kw = dict(factors=settings.factors, reg=settings.reg, iterations=settings.iterations,
                      seed=settings.seed)
            if name == "cmf":
                kw["source_weight"] = settings.source_weight
        elif name == "popularity":
            kw = dict(relevant_threshold=settings.relevant_threshold)
    kw.update(extra)
    return MODELS[name](**kw)


def load(directory: str | Path) -> Recommender:
    d = Path(directory)
    meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
    with np.load(d / ARRAYS_FILE, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    return MODELS[meta["model"]].restore(meta, arrays)


__all__ = ["MODELS", "NotFittedError", "Recommender", "build", "load"]
