"""L4 RAPM, этап 1: правило приёма O2 вместо O1 (METHODOLOGY 4.3а, решение пользователя 07.10.2026).

Всё на train, случайные половины матчей с seed L7. Каждая оценка O2 подгоняется целиком на своей
части матчей: λ ridge-к-нулю и ridge-к-prior (кросс-валидация по матчам части), веса SPM-prior,
O1 для признаков prior. Условия:
1. split-half r_SB(O2) > r_SB(сырой on-ice xGD/60 5×5) и > r_SB(один prior): нижняя граница ДИ > 0
   (игроки с ≥ 5 матчами в каждой половине; бутстрэп игроков со стратификацией по амплуа);
2. ΔSS = SS(O2) − SS(O1) через общий слой L7, направления объединены, бутстрэп игроков: нижняя граница > 0;
3. наклон слоя для O2 > 0 в обоих направлениях (нижняя граница ДИ > 0) и отличается меньше чем вдвое;
4. Спирмен O2 (весь train) и O2 (без случайных 10% матчей) ≥ 0,9 у игроков с ≥ 10 матчами.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from . import l4_rapm as rp
from . import l7_calibration as cal
from . import l7_forecast as fc
from . import validation as va
from .stage_l5 import player_roles
from .validation import _boot_indices, _ci

MIN_GAMES_HALF = 5
MIN_GAMES_PRIOR = 10          # на весь train; для части матчей — пропорционально её доле
MIN_GAMES_PRIOR_FLOOR = 2
MIN_GAMES_ROBUST = 10
DROP_SHARE = 0.10
ROBUST_MIN_SPEARMAN = 0.9
MAX_SLOPE_RATIO = 2.0
# Этап 2: разброс O2 у игроков с ≥ 10 матчами не меньше половины разброса prior
MIN_SPREAD_RATIO = 0.5
REF = "O2"


def _games(pm: pd.DataFrame) -> pd.Series:
    return pm[pm.toi_rate_s > 0].groupby("player_id").match_id.nunique()


def prior_min_games(n_part_matches: int, n_all_matches: int) -> int:
    """Порог матчей для весов SPM-prior в части данных: 10 на весь train, пропорционально доле части.

    Без масштаба на половине train порогу «≥ 10» удовлетворяют единицы игроков (3 и 7 в первом прогоне
    07.10) — веса prior фактически не обучаются.
    """
    share = n_part_matches / max(n_all_matches, 1)
    return max(MIN_GAMES_PRIOR_FLOOR, int(round(MIN_GAMES_PRIOR * share)))


class RapmRatings:
    """O2, RAPM к нулю и один prior для части матчей `pm_part`; кэш по набору матчей."""

    def __init__(self, rows: pd.DataFrame, shrink: Mapping, grid=rp.LAMBDA_GRID, k: int = rp.CV_FOLDS,
                 seed: int = va.SEED):
        self.rows, self.shrink, self.grid, self.k, self.seed = rows, dict(shrink), tuple(grid), k, seed
        self.memo: dict[frozenset, dict] = {}
        self.n_all = int(rows.match_id.nunique())

    def compute(self, pm_part: pd.DataFrame) -> dict:
        key = frozenset(pm_part.match_id.unique())
        if key in self.memo:
            return self.memo[key]
        rows = self.rows[self.rows.match_id.isin(key)]
        lam0, cv0 = rp.cv_lambda(rows, self.grid, self.k, self.seed)
        fit0 = rp.fit_rapm(rows, lam0)
        det, _ = cal.o1_detail(pm_part, self.shrink)
        comps = [c for c in det.columns if c.startswith("comp_")]
        feats = det.set_index("player_id")[["role"] + comps]
        g = _games(pm_part)
        thr = prior_min_games(len(key), self.n_all)
        prior = rp.spm_prior({"o": fit0.beta_o, "d": fit0.beta_d}, feats, fit0.toi_h,
                             g.index[g >= thr], self.k, self.seed)
        lam1, cv1 = rp.cv_lambda(rows, self.grid, self.k, self.seed, prior["o"], prior["d"])
        fit1 = rp.fit_rapm(rows, lam1, prior["o"], prior["d"])
        players = fit1.o2.index
        out = {"O2": fit1.o2, "RAPM0": fit0.o2,
               "PRIOR": (prior["o"] - prior["d"]).reindex(players).rename("PRIOR"),
               "info": {"n_matches": len(key), "n_rows": int(len(rows)), "lambda_zero": lam0, "cv_zero": cv0,
                        "lambda_prior": lam1, "cv_prior": cv1, "controls": fit1.controls,
                        "prior_min_games": thr, "n_prior_eligible": int((g >= thr).sum())}}
        self.memo[key] = out
        return out

    def fn(self, name: str) -> Callable[[pd.DataFrame], pd.Series]:
        def rating(pm_part: pd.DataFrame) -> pd.Series:
            return self.compute(pm_part)[name].rename(name)
        return rating


# --- 1. надёжность --------------------------------------------------------------------------

def _sb_pooled(a: np.ndarray, b: np.ndarray, groups: np.ndarray) -> float:
    return va.spearman_brown(va._group_stats(a, b, None, groups)[va.POOLED].get("spearman", np.nan))


def reliability(pm: pd.DataFrame, fns: Mapping[str, Callable], seed: int = va.SEED,
                n_boot: int = va.N_BOOT, ref: str = REF) -> dict:
    halves = va.assign_halves(pm.match_id, seed)
    h = pm.match_id.map(halves).to_numpy(dtype=int)
    pa, pb = pm[h == 0], pm[h == 1]
    ga, gb = _games(pa), _games(pb)
    ids = ga.index[ga >= MIN_GAMES_HALF].intersection(gb.index[gb >= MIN_GAMES_HALF])
    df = pd.DataFrame({"role": player_roles(pm).reindex(ids)}, index=ids)
    for n, fn in fns.items():
        df[f"a::{n}"] = fn(pa).reindex(ids).to_numpy(float)
        df[f"b::{n}"] = fn(pb).reindex(ids).to_numpy(float)
    df = df.dropna()
    groups = df.role.to_numpy().astype(str)

    def stats(ix):
        return {n: _sb_pooled(df[f"a::{n}"].to_numpy()[ix], df[f"b::{n}"].to_numpy()[ix], groups[ix]) for n in fns}

    point = stats(np.arange(len(df)))
    rng = np.random.default_rng(seed + 31)
    boots = [stats(_boot_indices(groups, rng)) for _ in range(n_boot)]
    out = {n: {"r_sb": point[n], "r_sb_ci": _ci([b[n] for b in boots])} for n in fns}
    vs = {n: {"diff": point[ref] - point[n], "diff_ci": _ci([b[ref] - b[n] for b in boots])}
          for n in fns if n != ref}
    return {"n_players": int(len(df)), "min_games_half": MIN_GAMES_HALF, "ratings": out, "vs": vs}


# --- 2. прогноз и 3. наклон ------------------------------------------------------------------

def _pooled_ss(p: pd.DataFrame, names) -> dict:
    y, w = p.target.to_numpy(float), p.toi_s.to_numpy(float)
    mse = {n: float((w * (y - p[f"p::{n}"].to_numpy(float)) ** 2).sum() / w.sum()) for n in names}
    return {n: 1.0 - mse[n] / mse[fc.ROLE_ONLY] for n in names}


def pooled_delta_ss(preds: pd.DataFrame, a: str, b: str, n_boot: int = va.N_BOOT, seed: int = va.SEED) -> dict:
    """SS(a) − SS(b) по объединённым направлениям; бутстрэп игроков (все строки игрока вместе)."""
    names = [fc.ROLE_ONLY, a, b]
    point = _pooled_ss(preds, names)
    boots = [_pooled_ss(preds.iloc[ix], names) for ix in cal._player_boot(preds, n_boot, seed)]
    return {"ss": {a: point[a], b: point[b]}, "delta_ss": point[a] - point[b],
            "delta_ss_ci": _ci([x[a] - x[b] for x in boots]), "n_rows": int(len(preds)),
            "n_players": int(preds.player_id.nunique())}


def forecast(pm: pd.DataFrame, fns: Mapping[str, Callable], seed: int = va.SEED,
             min_toi_s: float = va.MIN_TOI_PRED_S, n_boot: int = va.N_BOOT, ref: str = REF,
             base: str = "O1") -> tuple[dict, dict]:
    frames = fc.train_half_frames(pm, fns, seed, min_toi_s)
    names = list(fns)
    preds = fc.crossfit_predict(frames, names, fc.player_folds(frames, fc.N_FOLDS, seed))
    long = pd.concat([p.assign(direction=d, player_id=p.index) for d, p in preds.items()], ignore_index=True)
    out = pooled_delta_ss(long, ref, base, n_boot, seed)
    out["ss_all"] = {n: float(v) for n, v in _pooled_ss(long, [fc.ROLE_ONLY] + names).items()}
    return out, frames


def slopes(frames: Mapping[str, pd.DataFrame], name: str = REF, n_boot: int = va.N_BOOT,
           seed: int = va.SEED) -> dict:
    out = {}
    rng = np.random.default_rng(seed + 41)
    for d, f in frames.items():
        groups = f.role.to_numpy().astype(str)
        b = fc.fit_layer(f, name)["b"]
        boots = [fc.fit_layer(f.iloc[_boot_indices(groups, rng)], name)["b"] for _ in range(n_boot)]
        out[d] = {"b": b, "b_ci": _ci(boots), "n": int(len(f))}
    return out


# --- 4. устойчивость -------------------------------------------------------------------------

def robustness(pm: pd.DataFrame, ratings: RapmRatings, seed: int = va.SEED) -> dict:
    full = ratings.compute(pm)["O2"]
    mids = np.array(sorted(pm.match_id.unique()), dtype=object)
    drop = set(np.random.default_rng(seed + 53).choice(mids, int(round(DROP_SHARE * len(mids))), replace=False))
    part = ratings.compute(pm[~pm.match_id.isin(drop)])["O2"]
    g = _games(pm)
    ids = g.index[g >= MIN_GAMES_ROBUST].intersection(full.index).intersection(part.index)
    rho = float(full.reindex(ids).corr(part.reindex(ids), method="spearman"))
    return {"spearman": rho, "n_players": int(len(ids)), "dropped_matches": len(drop)}


# --- вердикт ---------------------------------------------------------------------------------

def _lo(ci) -> float:
    return float(ci[0]) if ci is not None and np.isfinite(ci[0]) else float("nan")


def verdict(res: Mapping) -> dict:
    vs = res["reliability"]["vs"]
    c1 = all(_lo(vs[n]["diff_ci"]) > 0 for n in ("onice_xgd_raw", "prior_only"))
    c2 = _lo(res["forecast"]["delta_ss_ci"]) > 0
    sl = res["slope"]
    bs = [sl[d]["b"] for d in sl]
    c3 = (all(_lo(sl[d]["b_ci"]) > 0 for d in sl) and min(bs) > 0
          and max(bs) / min(bs) < MAX_SLOPE_RATIO)
    c4 = bool(res["robustness"]["spearman"] >= ROBUST_MIN_SPEARMAN)
    cond = {"1_reliability": bool(c1), "2_forecast": bool(c2), "3_slope": bool(c3), "4_robustness": c4}
    if "spread" in res:      # этап 2 (METHODOLOGY 4.3б): защита от почти постоянного рейтинга
        ratio = res["spread"].get("ratio")
        cond["5_spread"] = bool(ratio is not None and np.isfinite(ratio) and ratio >= MIN_SPREAD_RATIO)
    rule = ("METHODOLOGY 4.3б: все пять условий" if "5_spread" in cond else "METHODOLOGY 4.3а: все четыре условия")
    return {"conditions": cond, "accepted": all(cond.values()),
            "rule": f"{rule}; иначе O1 остаётся итогом, O2 — диагностика"}


# --- этап 2: λ по эмпирическому Байесу (METHODOLOGY 4.3б) ------------------------------------

class EbRatings:
    """O2 этапа 2 (prior внутри регрессии, λ = σ²/τ² по ЭМ) для части матчей; кэш по набору матчей."""

    def __init__(self, rows: pd.DataFrame, shrink: Mapping, seed: int = va.SEED):
        self.rows, self.shrink, self.seed = rows, dict(shrink), seed
        self.memo: dict[tuple, dict] = {}

    def compute(self, pm_part: pd.DataFrame, lam_scale: float = 1.0, with_teams: bool = False) -> dict:
        from . import l4_rapm_eb as eb
        mkey = frozenset(pm_part.match_id.unique())
        key = (mkey, float(lam_scale), bool(with_teams))
        if key in self.memo:
            return self.memo[key]
        base = None if (lam_scale == 1.0 and not with_teams) else self.compute(pm_part)
        rows = self.rows[self.rows.match_id.isin(mkey)]
        det, _ = cal.o1_detail(pm_part, self.shrink)
        comps = [c for c in det.columns if c.startswith("comp_")]
        feats = det.set_index("player_id")[["role"] + comps]
        fixed = base["fit"].tau2 if base is not None and not with_teams else None
        fit = eb.fit_eb(rows, feats, lam_scale=lam_scale, with_teams=with_teams, tau2_fixed=fixed)
        out = {"O2": fit.o2, "PRIOR": fit.prior, "fit": fit,
               "info": {"n_matches": len(mkey), "n_rows": int(len(rows)), "lam_scale": lam_scale,
                        "with_teams": with_teams, "sigma2": fit.sigma2, "sigma2_resid": fit.sigma2_resid,
                        "sigma2_poisson": fit.sigma2_poisson, "tau2": fit.tau2, "tau2_team": fit.tau2_team,
                        "lambda": fit.lam, "converged": fit.converged, "n_iter": fit.n_iter,
                        "n_trace_iter": fit.n_trace_iter,
                        "gamma_o": fit.gamma["o"].to_dict(), "gamma_d": fit.gamma["d"].to_dict()}}
        self.memo[key] = out
        return out

    def fn(self, name: str, lam_scale: float = 1.0, with_teams: bool = False) -> Callable[[pd.DataFrame], pd.Series]:
        def rating(pm_part: pd.DataFrame) -> pd.Series:
            return self.compute(pm_part, lam_scale, with_teams)[name].rename(name)
        return rating


def spread(pm: pd.DataFrame, ratings: EbRatings) -> dict:
    """Условие 5: sd(O2) / sd(prior) у игроков с ≥ 10 матчами (подгонка на всём train)."""
    full = ratings.compute(pm)
    g = _games(pm)
    ids = g.index[g >= MIN_GAMES_ROBUST].intersection(full["O2"].index)
    s_o2, s_pr = float(full["O2"].reindex(ids).std()), float(full["PRIOR"].reindex(ids).std())
    return {"sd_o2": s_o2, "sd_prior": s_pr, "ratio": s_o2 / s_pr if s_pr > 0 else float("nan"),
            "n_players": int(len(ids)), "min_ratio": MIN_SPREAD_RATIO}
