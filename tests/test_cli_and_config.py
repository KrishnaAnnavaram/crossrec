import json

import pytest

from crossrec.cli import main
from crossrec.config import Settings


def test_settings_precedence(tmp_path):
    env = {"CROSSREC_FACTORS": "5", "CROSSREC_REG": "0.3", "CROSSREC_DATA_DIR": "x"}
    s = Settings.from_env(env)
    assert (s.factors, s.reg, s.data_dir) == (5, 0.3, "x")
    toml = tmp_path / "c.toml"
    toml.write_text("[crossrec]\nfactors = 9\nk = 5\n", encoding="utf-8")
    s = s.merge_toml(toml)
    assert (s.factors, s.k, s.reg) == (9, 5, 0.3)
    assert s.merge(factors=2, k=None).factors == 2
    with pytest.raises(ValueError):
        s.merge(nonsense=1)
    with pytest.raises(ValueError):
        Settings(test_fraction=1.5).validate()


def test_cli_end_to_end(tmp_path, capsys):
    data = tmp_path / "data"
    assert main(["synth", "--out", str(data), "--users", "300", "--seed", "3"]) == 0
    assert main(["stats", "--data-dir", str(data)]) == 0
    assert "users in both domains" in capsys.readouterr().out
    report = tmp_path / "r.json"
    assert main(["evaluate", "--data-dir", str(data), "--models", "popularity,cmf",
                 "--out", str(report)]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert [r["model"] for r in payload["results"]] == ["popularity", "cmf"]
    assert payload["test_users"] > 0
    assert main(["train", "--data-dir", str(data), "--model", "cmf", "--out", str(tmp_path / "m")]) == 0
    capsys.readouterr()
    assert main(["recommend", "--data-dir", str(data), "--model-dir", str(tmp_path / "m"),
                 "--user", "U00010", "--k", "3"]) == 0
    out = capsys.readouterr().out.splitlines()
    lines = [ln for ln in out if ln.strip().split(".")[0].isdigit()]
    assert len(lines) == 3
    # Problem 7: each title is a whole string, not one character per entry.
    assert all("Book " in ln or "Collected Poems" in ln for ln in lines)


def test_cli_errors(tmp_path, capsys):
    assert main(["stats", "--data-dir", str(tmp_path / "missing")]) == 2
    assert "not found" in capsys.readouterr().err
    assert main(["evaluate", "--synthetic", "--source", "food", "--target", "food"]) == 2
    assert main(["recommend", "--synthetic", "--model-dir", str(tmp_path), "--user", "x"]) == 2
