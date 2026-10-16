"""Карточка для клуба: понятная шкала и текст по шаблону без LLM (METHODOLOGY 11.7).

Запуск из Solve после карточки (кэши и выходы только читаются):

    PYTHONPATH=src .venv/bin/python scripts/card_club.py

Перцентиль — относительно распределения истинных уровней амплуа N(0, 1ᵀT_r1) (L5, train); вывод — по 95%-интервалу
(«выше / ниже среднего амплуа» или «отличие от среднего амплуа не подтверждено»); надёжность — словами; сильные и
слабые стороны — только не для LOW. WAR за сезон — WAR/60 × типичное время амплуа на матч (медиана train) × 82 —
условный пересчёт: только числовые колонки, в текст не входит. Выход: outputs/card/club_ratings.csv (все игроки test).
test_ratings.csv не меняется.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import validation_report as vr

from datamatch import card_club as cc
from datamatch import card_io, io
from datamatch.config import load_config, setup_logging

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "outputs" / "card"
MIN_GAMES_TYPICAL_TOI = 5
PROFILE = ("shot", "creation", "recovery", "loss", "faceoff")


def typical_toi_per_game(pm_train: pd.DataFrame) -> dict:
    """Медиана среднего TOI игрока за матч (мин) по амплуа, игроки train с ≥ 5 матчами."""
    g = pm_train.groupby("player_id").agg(role=("role", "first"), toi_s=("toi_s", "sum"), n=("match_id", "nunique"))
    g = g[g.n >= MIN_GAMES_TYPICAL_TOI]
    return (g.toi_s / g.n / 60.0).groupby(g.role).median().to_dict()


def build(detail: pd.DataFrame, war: pd.DataFrame, roles: pd.DataFrame, one_t1: dict, toi_game: dict) -> pd.DataFrame:
    d = detail.merge(war[["player_id", "team_id", "is_avangard", "role", "toi_min", "war", "war_lo95", "war_hi95"]],
                     on="player_id", how="left", validate="one_to_one")
    d = d.merge(roles[["player_id"] + [f"z_{p}" for p in PROFILE]], on="player_id", how="left", validate="one_to_one")
    rows = []
    for r in d.itertuples(index=False):
        tau = math.sqrt(one_t1[r.role])
        hours = r.toi_min / 60.0 if r.toi_min and r.toi_min > 0 else np.nan
        per60 = [x / hours if hours == hours else np.nan for x in (r.war, r.war_lo95, r.war_hi95)]
        season = [cc.war_per_season(x, toi_game[r.role]) if x == x else np.nan for x in per60]
        flag = r.display_flag if isinstance(r.display_flag, str) else ""
        rel = cc.reliability_words(r.reliability_display, flag)
        s, w = (cc.strengths({p: getattr(r, f"z_{p}") for p in PROFILE}) if cc.show_profile(r.reliability_display)
                else ([], []))
        pct, pct_lo, pct_hi = (cc.percentile(x, tau) for x in (r.rating, r.lo95, r.hi95))
        ver = cc.verdict(r.lo95, r.hi95)
        rows.append({"player_id": r.player_id, "team_id": r.team_id, "is_avangard": r.is_avangard, "role": r.role,
                     "rating": r.rating, "lo95": r.lo95, "hi95": r.hi95, "percentile": pct,
                     "percentile_lo95": pct_lo, "percentile_hi95": pct_hi,
                     "verdict": ver, "reliability_words": rel, "war_per60": per60[0], "war_season": season[0],
                     "war_season_lo95": season[1], "war_season_hi95": season[2],
                     "strengths": ", ".join(s), "weaknesses": ", ".join(w),
                     "text": cc.player_text(r.role, pct, pct_lo, pct_hi, ver, r.reliability_display, flag, s, w)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=CARD)
    args = ap.parse_args(argv)
    cfg = load_config()
    setup_logging(cfg, "card_club")
    tables = card_io.load_card_tables(cfg, vr.ReadOnlyCache(cfg, io.data_version(cfg)))
    one_t1 = tables["l5"]["report"]["moments"]["oneT1"]
    toi_game = typical_toi_per_game(tables["pm_train"])
    out = build(tables["detail"], pd.read_csv(CARD / "war.csv"), pd.read_csv(CARD / "roles.csv"), one_t1, toi_game)
    out.to_csv(args.out / "club_ratings.csv", index=False)
    print(f"club_ratings.csv: {len(out)} игроков; типичное время на матч {({k: round(v, 1) for k, v in toi_game.items()})}")
    print(out[out.is_avangard].sort_values("rating", ascending=False).text.head(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
