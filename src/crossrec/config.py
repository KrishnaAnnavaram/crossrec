"""Run settings. Order of precedence: CLI flag > TOML file > environment variable > default."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path

ENV_PREFIX = "CROSSREC_"


@dataclass(frozen=True)
class Settings:
    data_dir: str = "data"
    seed: int = 42
    factors: int = 8
    reg: float = 0.1
    iterations: int = 12
    source_weight: float = 1.0
    min_user_ratings: int = 3
    min_item_ratings: int = 3
    max_users: int = 0  # 0 = keep every overlapping user after the k-core filter
    test_fraction: float = 0.2
    k: int = 10
    relevant_threshold: float = 4.0

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        values = {}
        for f in fields(cls):
            raw = env.get(ENV_PREFIX + f.name.upper())
            if raw is None or raw == "":
                continue
            values[f.name] = _coerce(f.name, raw, cls)
        return cls(**values)

    def merge(self, **overrides) -> "Settings":
        clean = {k: v for k, v in overrides.items() if v is not None}
        unknown = set(clean) - {f.name for f in fields(self)}
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        return replace(self, **{k: _coerce(k, v, type(self)) for k, v in clean.items()})

    def merge_toml(self, path: str | Path) -> "Settings":
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        return self.merge(**data.get("crossrec", data))

    def validate(self) -> "Settings":
        if self.factors < 1:
            raise ValueError("factors must be >= 1")
        if self.reg < 0:
            raise ValueError("reg must be >= 0")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1")
        if self.min_user_ratings < 1 or self.min_item_ratings < 1:
            raise ValueError("k-core minimums must be >= 1")
        if self.k < 1:
            raise ValueError("k must be >= 1")
        return self


def _coerce(name: str, raw, cls) -> object:
    target = {f.name: f.type for f in fields(cls)}[name]
    if target in ("int", int):
        return int(raw)
    if target in ("float", float):
        return float(raw)
    return str(raw)
