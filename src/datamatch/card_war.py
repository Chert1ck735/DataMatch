"""Карточка игрока, WAR: перевод xGAR в победы одной константой G_win (METHODOLOGY 11.1).

G_win — «голов разницы на одну победу» по предельной ценности гола: лишний гол переводит ничью
основного времени в победу (+½ ожидаемой победы при равных шансах в овертайме и буллитах), а
проигрыш в один гол — в ничью (+½), поэтому ΔW = ½·P(D = 0) + ½·P(D = −1), G_win = 1/ΔW, где D —
разница голов команды в основное время. Счёт восстанавливается из событий goal: периоды 1–3 —
основное время, ≥ 4 — овертайм, период 0 — буллиты.

WAR_i = xGAR_i / G_win; xGAR — из L5 (θ̂ − R_r)·TOI/60. Интервал — Монте-Карло: θ̂ ~ N(·, v_tot)
(апостериорная дисперсия + разброс моделей ценности L6), R_r — бутстрэп игроков train, G_win —
кластерный бутстрэп по командам. Новой оценки игрока здесь нет: только смена единиц.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import skellam

from .l5_shrink import ROLES
from .stage_l5 import replacement_level

REGULATION_PERIODS = (1, 2, 3)
SHOOTOUT_PERIOD = 0
CI_LOW, CI_HIGH = 2.5, 97.5


# ---------------------------------------------------------------------------------------
# Счёт матча

def match_scores(goals: pd.DataFrame, games: pd.DataFrame, bad_matches: set) -> pd.DataFrame:
    """Счёт основного времени, овертайма и буллитов по событиям goal; матчи из bad_matches выбрасываются.

    goals: match_id, period, actor_team_id (команда, забившая гол); games: match_id, home_team_id,
    away_team_id. Гол команды, не игравшей в матче, — нарушение контракта (ValueError).
    home_win: 1/0 по итогу матча; NaN — если ничья и буллитов нет (исход по данным не определён).
    """
    g = games.loc[~games.match_id.isin(bad_matches), ["match_id", "home_team_id", "away_team_id"]]
    gl = goals[goals.match_id.isin(g.match_id)].merge(g, on="match_id", how="left")
    is_home = gl.actor_team_id.to_numpy() == gl.home_team_id.to_numpy()
    is_away = gl.actor_team_id.to_numpy() == gl.away_team_id.to_numpy()
    stray = gl.loc[~(is_home | is_away), "actor_team_id"].astype(str).unique().tolist()
    if stray:
        raise ValueError(f"match_scores: голы команд не из матча: {stray[:5]}")
    period = gl.period.astype(int).to_numpy()
    kind = np.where(np.isin(period, REGULATION_PERIODS), "reg",
                    np.where(period == SHOOTOUT_PERIOD, "so", "ot"))
    side = np.where(is_home, "home", "away")
    counts = (pd.DataFrame({"match_id": gl.match_id.to_numpy(), "col": np.char.add(np.char.add(side, "_"), kind)})
              .groupby(["match_id", "col"]).size().unstack(fill_value=0))
    cols = [f"{s}_{k}" for s in ("home", "away") for k in ("reg", "ot", "so")]
    out = g.set_index("match_id").join(counts.reindex(columns=cols, fill_value=0)).fillna(
        {c: 0 for c in cols})
    out[cols] = out[cols].astype(int)
    diff = (out.home_reg + out.home_ot) - (out.away_reg + out.away_ot)
    so = out.home_so - out.away_so
    win = np.where(diff > 0, 1.0, np.where(diff < 0, 0.0,
                   np.where(so > 0, 1.0, np.where(so < 0, 0.0, np.nan))))
    return out.assign(home_win=win).reset_index()


def team_game_rows(scores: pd.DataFrame) -> pd.DataFrame:
    """Две строки на матч (хозяева и гости): d_reg — разница голов основного времени, win — исход."""
    home = pd.DataFrame({"match_id": scores.match_id, "team_id": scores.home_team_id, "is_home": True,
                         "gf_reg": scores.home_reg, "ga_reg": scores.away_reg, "win": scores.home_win})
    away = pd.DataFrame({"match_id": scores.match_id, "team_id": scores.away_team_id, "is_home": False,
                         "gf_reg": scores.away_reg, "ga_reg": scores.home_reg, "win": 1.0 - scores.home_win})
    rows = pd.concat([home, away], ignore_index=True)
    return rows.assign(d_reg=rows.gf_reg - rows.ga_reg).sort_values(["match_id", "is_home"],
                                                                     ascending=[True, False]).reset_index(drop=True)


def known_score_check(scores: pd.DataFrame, known: pd.DataFrame) -> dict:
    """Сверка восстановленного счёта с протоколом (основное + ОТ, победа по буллитам = +1 гол).

    Критерий METHODOLOGY 11.1: все матчи с известным счётом совпадают; иначе ValueError.
    """
    chk = known[["match_id", "home_goals", "away_goals"]].merge(scores, on="match_id", how="inner")
    if chk.empty:
        raise ValueError("known_score_check: нет матчей с известным счётом для сверки")
    total_h = chk.home_reg + chk.home_ot + (chk.home_so > chk.away_so)
    total_a = chk.away_reg + chk.away_ot + (chk.away_so > chk.home_so)
    exact = (total_h == chk.home_goals) & (total_a == chk.away_goals)
    bad = chk.loc[~exact, "match_id"].tolist()
    if bad:
        raise ValueError(f"known_score_check: счёт не совпал с протоколом в матчах {bad}")
    return {"known_scores_checked": int(len(chk)), "known_scores_match": int(exact.sum()),
            "known_scores_mismatch": bad}


def attach_xgar(detail: pd.DataFrame, o1: pd.DataFrame) -> pd.DataFrame:
    """xGAR и θ̂ из O1; xGAR = 0 только у игроков, которых нет в O1 (нет времени на льду)."""
    df = detail.merge(o1[["player_id", "xgar", "theta"]], on="player_id", how="left", indicator=True)
    nan_in_o1 = (df["_merge"] == "both") & df["xgar"].isna()
    if nan_in_o1.any():
        raise ValueError(f"attach_xgar: xGAR = NaN у игроков O1 {df.loc[nan_in_o1, 'player_id'].tolist()}")
    xgar = df["xgar"].where(df["_merge"] == "both", 0.0)
    return df.drop(columns="_merge").assign(xgar=xgar)


# ---------------------------------------------------------------------------------------
# G_win: основной метод и проверки чувствительности

def marginal_win_per_goal(d_reg: np.ndarray) -> float:
    """ΔW = ½·P(D = 0) + ½·P(D = −1) по строкам «команда × матч»."""
    d = np.asarray(d_reg)
    if d.size == 0:
        raise ValueError("marginal_win_per_goal: пустая выборка")
    return float(0.5 * np.mean(d == 0) + 0.5 * np.mean(d == -1))


def _inverse(dw: float, what: str) -> float:
    if not np.isfinite(dw) or dw <= 0:
        raise ValueError(f"{what}: ценность гола ΔW = {dw} не положительна — G_win не определён")
    return 1.0 / dw


def g_win_marginal(d_reg: np.ndarray) -> float:
    return _inverse(marginal_win_per_goal(d_reg), "g_win_marginal")


def skellam_marginal(lam_for: float, lam_against: float) -> float:
    """ΔW при пуассоновском счёте: dP(победа)/dλ_свои = ½·P(D = 0) + ½·P(D = −1), D ~ Скеллам."""
    return float(0.5 * skellam.pmf(0, lam_for, lam_against) + 0.5 * skellam.pmf(-1, lam_for, lam_against))


def g_win_skellam(lam_for: float, lam_against: float) -> float:
    return _inverse(skellam_marginal(lam_for, lam_against), "g_win_skellam")


def g_win_pythagorean(goals_per_team_game: float, k: float) -> float:
    """Пифагорово ожидание при GF = GA = g: dW/dGD = k/(4g) → G_win = 4g/k."""
    return float(4.0 * goals_per_team_game / k)


def g_win_ols(d: np.ndarray, win: np.ndarray) -> float:
    """1 / наклон МНК исхода на разницу голов (средний наклон ступенчатой функции, чувствительность)."""
    d, w = np.asarray(d, dtype=float), np.asarray(win, dtype=float)
    ok = np.isfinite(d) & np.isfinite(w)
    d, w = d[ok], w[ok]
    slope = float(np.mean((d - d.mean()) * (w - w.mean())) / np.var(d))
    return _inverse(slope, "g_win_ols")


def cluster_bootstrap(rows: pd.DataFrame, cluster_col: str, stat: Callable[[pd.DataFrame], float],
                      B: int, seed: int) -> np.ndarray:
    """Кластерный бутстрэп: кластеры с возвращением, все строки кластера целиком."""
    pos = rows.groupby(cluster_col, sort=True).indices
    keys = list(pos)
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([pos[keys[i]] for i in pick])
        out[b] = stat(rows.iloc[idx])
    return out


# ---------------------------------------------------------------------------------------
# Уровень замены и WAR

def replacement_bootstrap(o1_train: pd.DataFrame, B: int, seed: int) -> dict[str, np.ndarray]:
    """Бутстрэп R_r: игроки амплуа с возвращением, тот же `stage_l5.replacement_level`."""
    rng = np.random.default_rng(seed)
    out = {}
    for r in ROLES:
        g = o1_train[o1_train.role == r].reset_index(drop=True)
        if g.empty:
            out[r] = np.empty(0)
            continue
        out[r] = np.array([replacement_level(g.iloc[rng.integers(0, len(g), len(g))])[r] for _ in range(B)])
    return out


def war_table(df: pd.DataFrame, g_hat: float, g_draws: np.ndarray, R_hat: dict, R_draws: dict,
              n_draws: int, seed: int) -> pd.DataFrame:
    """WAR и его 95%-интервал. df: player_id, role, xgar, toi_min, v_tot (дисперсия θ̂ на 60).

    Точка: xGAR/G_win. Draw: (xGAR·60/TOI + √v_tot·z − (R_draw − R̂))·TOI/60 / G_draw. При TOI = 0
    xGAR = 0 и интервал вырожден (времени на льду нет).
    """
    missing = sorted(set(df.role) - set(R_hat))
    if missing:
        raise KeyError(f"war_table: нет уровня замены для амплуа {missing}")
    rng = np.random.default_rng(seed)
    toi = df.toi_min.to_numpy(dtype=float)
    hours = toi / 60.0
    rate = np.divide(df.xgar.to_numpy(dtype=float), hours, out=np.zeros_like(hours), where=hours > 0)
    sd = np.sqrt(np.clip(df.v_tot.to_numpy(dtype=float), 0.0, None))
    g = np.asarray(g_draws, dtype=float)[rng.integers(0, len(g_draws), n_draws)]
    shift = np.zeros((n_draws, len(df)))
    for r in sorted(set(df.role)):
        cols = np.flatnonzero(df.role.to_numpy() == r)
        rd = np.asarray(R_draws[r], dtype=float)
        shift[:, cols] = (rd[rng.integers(0, len(rd), n_draws)] - R_hat[r])[:, None]
    z = rng.standard_normal((n_draws, len(df)))
    draws = (rate[None, :] + sd[None, :] * z - shift) * hours[None, :] / g[:, None]
    lo, hi = np.percentile(draws, [CI_LOW, CI_HIGH], axis=0)
    xgar_draws = draws * g[:, None]
    xlo, xhi = np.percentile(xgar_draws, [CI_LOW, CI_HIGH], axis=0)
    return pd.DataFrame({"player_id": df.player_id.to_numpy(), "role": df.role.to_numpy(),
                         "toi_min": toi, "xgar": df.xgar.to_numpy(dtype=float), "xgar_lo95": xlo,
                         "xgar_hi95": xhi, "war": df.xgar.to_numpy(dtype=float) / g_hat,
                         "war_lo95": lo, "war_hi95": hi})
