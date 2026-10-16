"""Карточка игрока, пункт 1: WAR = xGAR / G_win (METHODOLOGY 11.1).

Запуск из Solve после `datamatch.cli shrink` и L6 full (кэши только читаются, слои не пересчитываются):

    PYTHONPATH=src .venv/bin/python scripts/card_war.py [--n-boot 2000] [--n-draws 20000]

G_win — по счёту матчей части разработки train (внешний блок не открывается), счёт восстановлен из
событий goal кэша L0. Выход: outputs/card/war.csv и outputs/card/war_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import validation_report as vr

from datamatch import card_io, card_war as cw
from datamatch import io
from datamatch.config import load_config, setup_logging
from datamatch.splits import load_splits, matches_for

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "card"
DEFAULT_N_BOOT = 2000
DEFAULT_N_DRAWS = 20000
PYTHAGOREAN_K = 2.0          # литературное значение для хоккея; только чувствительность
TRUNCATED_STATUS = "truncated"


def load_scores(l0: Path, cfg, dev: set) -> tuple[pd.DataFrame, dict]:
    """Счёт матчей разработки (без обрезанных) и сверка с известным счётом games.csv.

    Внешний блок не открывается: события, периоды и протоколы фильтруются по матчам разработки
    до восстановления счёта; сверка — на матчах разработки с известным счётом (ValueError при расхождении).
    """
    ids = sorted(dev)
    goals = pq.read_table(l0 / "train_events.parquet", columns=["match_id", "period", "actor_team_id"],
                          filters=[("event_type", "==", "goal"), ("match_id", "in", ids)]).to_pandas()
    periods = pd.read_parquet(l0 / "train_periods.parquet", columns=["match_id", "status"])
    periods = periods[periods.match_id.isin(dev)]
    games = pd.read_parquet(l0 / "train_games.parquet", columns=["match_id", "home_team_id", "away_team_id"])
    games = games[games.match_id.isin(dev)]
    bad = set(periods.match_id[periods.status == TRUNCATED_STATUS])
    scores = cw.match_scores(goals, games, bad)
    known = io.load_games(cfg, "train").dropna(subset=["home_goals", "away_goals"])
    info = {"matches_total": int(len(games)), "scope": "матчи разработки train",
            "excluded_truncated": sorted(bad),
            "goals_by_period": {str(k): int(v) for k, v in goals.period.value_counts().sort_index().items()},
            **cw.known_score_check(scores, known[known.match_id.isin(dev)]),
            "score_rule": "итог = основное время + ОТ, победа по буллитам = +1 гол (как в протоколе)"}
    return scores, info


def g_win_block(rows: pd.DataFrame, n_boot: int, seed: int) -> tuple[dict, np.ndarray]:
    d = rows.d_reg.to_numpy()
    g_hat = cw.g_win_marginal(d)
    stat = lambda r: cw.g_win_marginal(r.d_reg.to_numpy())   # noqa: E731
    by_team = cw.cluster_bootstrap(rows, "team_id", stat, n_boot, seed)
    by_match = cw.cluster_bootstrap(rows, "match_id", stat, n_boot, seed + 1)
    lam = float(rows.gf_reg.mean())
    decided = rows[rows.win.notna()]
    res = {
        "method": "предельная ценность гола: ΔW = ½·P(D=0) + ½·P(D=−1), D — разница основного времени",
        "n_matches": int(rows.match_id.nunique()), "n_team_games": int(len(rows)),
        "n_teams": int(rows.team_id.nunique()),
        "p_d0": float(np.mean(d == 0)), "p_d_minus1": float(np.mean(d == -1)),
        "delta_w": cw.marginal_win_per_goal(d), "g_win": g_hat,
        "ci95_team_bootstrap": np.percentile(by_team, [2.5, 97.5]).tolist(),
        "ci95_match_bootstrap": np.percentile(by_match, [2.5, 97.5]).tolist(),
        "sensitivity": {"skellam": cw.g_win_skellam(lam, lam), "regulation_goals_per_team_game": lam,
                        f"pythagorean_k{PYTHAGOREAN_K:g}": cw.g_win_pythagorean(lam, PYTHAGOREAN_K),
                        "ols_win_on_reg_margin": cw.g_win_ols(decided.d_reg.to_numpy(), decided.win.to_numpy()),
                        "share_undecided": float(rows.win.isna().mean())},
        "n_boot": n_boot}
    return res, by_team


def war_frame(tables: dict, detail: pd.DataFrame, avangard: str) -> pd.DataFrame:
    """Игроки test: xGAR из L5 (O1), v_tot и метка из L6; игроки без времени — xGAR 0."""
    o1 = tables["l5"]["O1"]["test"][["player_id", "role", "xgar", "toi_min", "theta"]]
    d = detail[["player_id", "rating", "v_tot", "reliability_display", "toi_min", "role"]]
    df = cw.attach_xgar(d, o1)
    pm = tables["pm_test"]
    team = pm.groupby("player_id").team_id.agg(lambda s: s.mode().iat[0])
    return df.assign(team_id=df.player_id.map(team), is_avangard=df.player_id.map(team).eq(avangard))


def run(args) -> dict:
    cfg = load_config()
    setup_logging(cfg, "card_war")
    cache = vr.ReadOnlyCache(cfg, io.data_version(cfg))
    l0 = vr.l0_dir(cache)
    splits = load_splits(cfg)
    dev = set(matches_for(splits, "dev"))
    scores, score_info = load_scores(l0, cfg, dev)
    rows = cw.team_game_rows(scores)
    seed = int(cfg["seed"])
    gwin, g_draws = g_win_block(rows, args.n_boot, seed)

    tables = card_io.load_card_tables(cfg, cache)
    rep = tables["l5"]["report"]["replacement"]["O1"]
    R_hat = rep["R"]
    R_draws = cw.replacement_bootstrap(tables["l5"]["O1"]["train"], args.n_boot, seed + 2)
    detail = tables["detail"].merge(pd.read_csv(cfg.paths.outputs / "test_ratings.csv")[["player_id", "role", "toi_min"]],
                                    on="player_id")
    df = war_frame(tables, detail, str(cfg["avangard_team_id"]))
    war = cw.war_table(df, gwin["g_win"], g_draws, R_hat, R_draws, args.n_draws, seed + 3)
    out = df[["player_id", "team_id", "is_avangard", "role", "rating", "reliability_display"]].merge(
        war.drop(columns=["role"]), on="player_id").rename(columns={"reliability_display": "reliability"})
    out["war_per60"] = np.where(out.toi_min > 0, out.war / out.toi_min * 60.0, 0.0)
    order = {r: float(g.war_per60.rank().corr(g.rating.rank())) for r, g in out[out.toi_min > 0].groupby("role")}
    args.out.mkdir(parents=True, exist_ok=True)
    out.sort_values("war", ascending=False).to_csv(args.out / "war.csv", index=False)
    report = {"l5_cache": tables["l5"]["key"], "g_win": gwin, "scores": score_info, "dev_matches": len(dev),
              "replacement": {"R": R_hat, "ci95": {r: (np.percentile(v, [2.5, 97.5]).tolist() if len(v) else None)
                                                   for r, v in R_draws.items()},
                              "source": "L5 O1 train (как в stage_l5), бутстрэп игроков амплуа"},
              "war_interval": f"Монте-Карло {args.n_draws} draws: θ̂ ~ N(·, v_tot L6), R — бутстрэп, G_win — бутстрэп по командам",
              "spearman_war_per60_vs_rating_within_role": order,
              "players": int(len(out)), "avangard_players": int(out.is_avangard.sum()),
              "war_sum_by_team_top": out.groupby("team_id").war.sum().sort_values(ascending=False).head(5).to_dict()}
    (args.out / "war_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--n-draws", type=int, default=DEFAULT_N_DRAWS)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    try:
        rep = run(args)
    except (vr.CachesNotReady, card_io.CardInputMismatch) as ex:
        print(f"кэши не готовы: {ex}", file=sys.stderr)
        return 3
    g = rep["g_win"]
    print(f"G_win = {g['g_win']:.3f} [{g['ci95_team_bootstrap'][0]:.3f}; {g['ci95_team_bootstrap'][1]:.3f}] "
          f"(команды), матчей {g['n_matches']}; сверка счёта {rep['scores']['known_scores_match']}/"
          f"{rep['scores']['known_scores_checked']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
