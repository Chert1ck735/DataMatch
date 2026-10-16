"""Карточка игрока: вбрасывания — Bradley–Terry с поправкой на соперника (METHODOLOGY 11.6).

Запуск из Solve (кэш L0 только читается):

    PYTHONPATH=src .venv/bin/python scripts/card_faceoffs.py

τ² (кросс-валидация по матчам) и контекст β — только на разработке; надёжность — половины матчей разработки
через один; test — оценки при замороженных τ² и β. Выход: outputs/card/faceoffs.csv, faceoffs_report.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import validation_report as vr

from datamatch import card_faceoffs as cf
from datamatch import io
from datamatch.config import load_config, setup_logging
from datamatch.splits import load_splits, matches_for

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "card"
# CONFIG: METHODOLOGY 11.6 (записано до расчёта)
TAU2_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
N_FOLDS = 5
MIN_FACEOFFS = 20
RELIABILITY_GATE = 0.3
COLS = ["match_id", "period", "actor_id", "other_id", "actor_team_id", "home_team_id", "zone"]


def load_faceoffs(l0: Path, part: str) -> pd.DataFrame:
    e = pq.read_table(l0 / f"{part}_events.parquet", columns=COLS, filters=[("event_type", "==", "faceoff")]).to_pandas()
    e = e[(e.period.astype(int) >= 1) & e.actor_id.notna() & e.other_id.notna()]
    return cf.home_perspective(e.reset_index(drop=True))


def counts(df: pd.DataFrame) -> pd.DataFrame:
    """Число вбрасываний и побед игрока (в любой роли — хозяев или гостей)."""
    won_h = df.groupby("home_pid").y.agg(["sum", "size"])
    won_a = df.assign(w=1 - df.y).groupby("away_pid").w.agg(["sum", "size"])
    t = won_h.add(won_a, fill_value=0)
    return pd.DataFrame({"n_faceoffs": t["size"].astype(int), "wins": t["sum"].astype(int)})


def split_half(dev: pd.DataFrame, tau2: float) -> dict:
    matches = sorted(dev.match_id.unique())
    half = {m: i % 2 for i, m in enumerate(matches)}
    parts = [dev[dev.match_id.map(half) == k] for k in (0, 1)]
    fits = [cf.fit_bt(p, tau2) for p in parts]
    n = [counts(p).n_faceoffs for p in parts]
    ok = sorted(set(n[0].index[n[0] >= MIN_FACEOFFS]) & set(n[1].index[n[1] >= MIN_FACEOFFS]))
    r = float(np.corrcoef(fits[0].theta[ok], fits[1].theta[ok])[0, 1]) if len(ok) > 2 else float("nan")
    sb = 2 * r / (1 + r) if np.isfinite(r) and r > -1 else float("nan")
    return {"r": r, "r_sb": sb, "n_players": len(ok), "gate": RELIABILITY_GATE,
            "passed": bool(np.isfinite(sb) and sb >= RELIABILITY_GATE)}


def run(args) -> dict:
    cfg = load_config()
    setup_logging(cfg, "card_faceoffs")
    l0 = vr.l0_dir(vr.ReadOnlyCache(cfg, io.data_version(cfg)))
    dev_ids = set(matches_for(load_splits(cfg), "dev"))
    tr = load_faceoffs(l0, "train")
    dev = tr[tr.match_id.isin(dev_ids)].reset_index(drop=True)
    te = load_faceoffs(l0, "test")
    tau2, scores = cf.choose_tau2(dev, TAU2_GRID, N_FOLDS, int(cfg["seed"]))
    beta = cf.fit_bt(dev, tau2).beta
    rel = split_half(dev, tau2)
    fit = cf.fit_bt(te, tau2, beta=beta)
    c = counts(te)
    th, sd = fit.theta.reindex(c.index), fit.sd.reindex(c.index)
    w = 1.0 - sd ** 2 / tau2
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))   # noqa: E731
    out = c.assign(win_share_raw=c.wins / c.n_faceoffs, skill_pct=100 * sig(th),
                   skill_lo95=100 * sig(th - 1.96 * sd), skill_hi95=100 * sig(th + 1.96 * sd), w=w,
                   reliability=np.where(w >= 0.7, "HIGH", np.where(w >= 0.4, "MEDIUM", "LOW")),
                   flag=np.where(c.n_faceoffs < MIN_FACEOFFS, "мало вбрасываний", ""))
    if not rel["passed"]:
        out = out.assign(skill_pct=np.nan, skill_lo95=np.nan, skill_hi95=np.nan, reliability="описательно")
    out = out.rename_axis("player_id").reset_index()
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out / "faceoffs.csv", index=False)
    report = {"dev_matches": len(dev_ids), "n_faceoffs_dev": int(len(dev)), "n_faceoffs_test": int(len(te)),
              "tau2": tau2, "tau2_cv_logloss": scores, "beta_dev": beta, "split_half_dev": rel,
              "test_players": int(len(out)), "test_labels": out.reliability.value_counts().to_dict()}
    (args.out / "faceoffs_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    rep = run(ap.parse_args(argv))
    print(json.dumps({k: rep[k] for k in ("tau2", "beta_dev", "split_half_dev", "test_labels")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
