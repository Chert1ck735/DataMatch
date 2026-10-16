"""L7: общий линейный прогнозный слой и ошибка прогноза будущей игры (архитектура, раздел L7).

Смысл. Рейтинги разных величин (O1 — xG/60 вклада, очки/60, on-ice xGD/60…) нельзя напрямую
сравнивать с целью по ошибке: у них разные шкалы. Каждый рейтинг z^(m) переводится в прогноз
общей цели одним и тем же слоем

    ŷ_i^(m) = a_{r(i)}^(m) + b^(m) · z_i^(m),

a_r — свободный член амплуа, b — общий наклон; WLS с весом будущего TOI. «Только амплуа» —
частный случай z = 0 (ŷ = a_r = взвешенное среднее цели амплуа), «только команда» — z = on-ice
xGD/60 команды игрока. Цель — on-ice xGD/60 при 5 на 5 (замороженный xG L1), вес — будущий TOI
при 5 на 5.

Метрики: MSE, RMSE и MAE с весом будущего TOI внутри амплуа и объединённо (объединённый MSE —
TOI-взвешенное среднее MSE амплуа) и skill score SS = 1 − MSE_модели / MSE_только амплуа.
Уровень ошибки включает шум будущей цели, который не уменьшит никакая модель, поэтому
интерпретируются SS и разности, а не абсолютные RMSE.

Данные. Train: случайные половины матчей (A→B и B→A, рейтинг по одной половине, цель по другой).
Коэффициенты слоя обучаются перекрёстно по игрокам (K фолдов со стратификацией по амплуа):
прогноз игрока строится коэффициентами, подогнанными без него, по обоим направлениям, и
коэффициенты усредняются по направлениям. Для диагностики вне train (train → test,
хронология «Авангарда») коэффициенты подгоняются на всех игроках половин train.

CI — бутстрэп игроков со стратификацией по амплуа на фиксированных прогнозах (парный для
разностей). Отклонение от архитектуры (там — бутстрэп матчей): повторная подгонка O1 на каждой
выборке матчей дорога; CI не учитывает общие матчи партнёров и потому несколько узок.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from . import baselines as bl
from . import validation as va
from .validation import _boot_indices, _ci

RatingFn = Callable[[pd.DataFrame], pd.Series]

ROLE_ONLY = "role_only"
TEAM_ONLY = "team_only"
REFERENCE = "O1"
TARGET = "xgd_5v5_60"
N_FOLDS = 5
# Рейтинг вырожден, если хотя бы в одном амплуа одной из половин его разброс меньше порога (единицы
# рейтинга): там он константа, наклон слоя неустойчив (пример — стянутый on-ice GD/60 с τ² ≈ 0).
DEGENERATE_Z_SD = 1e-3
# Минимум матчей игрока в train и в test для диагностики train → test (архитектура: ≥ 5 в обоих).
MIN_GAMES_CROSS = 5
ROLES = tuple(bl.FIELD_ROLES)
TEAM_COLS = ["match_id", "player_id", "team_id", "xgf_5v5", "xga_5v5", "toi_rate_5v5_s"]


# --- базовая строка «только команда» --------------------------------------------------------

def team_rating(pm_half: pd.DataFrame) -> pd.Series:
    """On-ice xGD/60 при 5 на 5 команды игрока по его матчам: Σ (xGF − xGA) / Σ TOI команды × 3600.

    Командные суммы матча — по всем её игрокам: время каждого игрока на льду входит один раз,
    поэтому отношение — ставка команды при 5 на 5 (знаменатель и числитель кратны числу полевых).
    """
    va._require(pm_half, TEAM_COLS, "team_rating")
    d = pm_half[TEAM_COLS]
    team = d.assign(num=d.xgf_5v5 - d.xga_5v5).groupby(["team_id", "match_id"]).agg(
        num=("num", "sum"), toi=("toi_rate_5v5_s", "sum"))
    rows = d[["player_id", "team_id", "match_id"]].drop_duplicates().join(team, on=["team_id", "match_id"])
    g = rows.groupby("player_id")[["num", "toi"]].sum()
    rate = np.where(g.toi > 0, g.num / g.toi.where(g.toi > 0) * bl.SECONDS_PER_HOUR, np.nan)
    return pd.Series(rate, index=pd.Index(g.index, name="player_id"), name=TEAM_ONLY)


# --- прогнозный слой ------------------------------------------------------------------------

def _z(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name == ROLE_ONLY:
        return np.zeros(len(frame))
    return frame[f"r::{name}"].to_numpy(dtype=float)


def fit_layer(frame: pd.DataFrame, name: str) -> dict:
    """WLS y = a_r + b·z с весом будущего TOI; для «только амплуа» b = 0 (a_r — среднее амплуа).

    Амплуа без строк получает общий взвешенный средний уровень (прогноз не теряется).
    """
    va._require(frame, ["role", "target", "toi_s"], "fit_layer")
    if not len(frame):
        raise ValueError("fit_layer: пустая выборка")
    roles = frame.role.to_numpy().astype(str)
    present = [r for r in ROLES if (roles == r).any()]
    y, w = frame.target.to_numpy(dtype=float), frame.toi_s.to_numpy(dtype=float)
    X = np.column_stack([(roles == r).astype(float) for r in present])
    if name != ROLE_ONLY:
        X = np.column_stack([X, _z(frame, name)])
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    overall = float(np.average(y, weights=w))
    a = {r: float(beta[present.index(r)]) if r in present else overall for r in ROLES}
    return {"a": a, "b": 0.0 if name == ROLE_ONLY else float(beta[-1])}


def predict(frame: pd.DataFrame, name: str, coef: Mapping) -> np.ndarray:
    a = pd.Series(coef["a"]).reindex(frame.role.to_numpy().astype(str)).to_numpy(dtype=float)
    return a + float(coef["b"]) * _z(frame, name)


def _mean_coef(coefs: list[dict]) -> dict:
    return {"a": {r: float(np.mean([c["a"][r] for c in coefs])) for r in ROLES},
            "b": float(np.mean([c["b"] for c in coefs]))}


def _with_role_only(names: Iterable[str]) -> list[str]:
    return [ROLE_ONLY] + [n for n in names if n != ROLE_ONLY]


def within_role_sd(frames: Mapping[str, pd.DataFrame], name: str) -> float:
    """Наименьший по направлениям и амплуа TOI-взвешенный разброс рейтинга (диагностика вырождения).

    Если в каком-то амплуа одной половины рейтинг — константа (стягивание с τ² ≈ 0), наклон этого
    направления подгоняется по остальным амплуа и при усреднении направлений может быть произвольным.
    """
    out = []
    for f in frames.values():
        z, w = _z(f, name), f.toi_s.to_numpy(dtype=float)
        roles = f.role.to_numpy().astype(str)
        for r in np.unique(roles):
            m = roles == r
            if w[m].sum() > 0:
                mean = np.average(z[m], weights=w[m])
                out.append(float(np.sqrt(np.average((z[m] - mean) ** 2, weights=w[m]))))
    return float(min(out)) if out else float("nan")


def fit_coefs(frames: Mapping[str, pd.DataFrame], names: Iterable[str]) -> dict:
    """Коэффициенты каждого рейтинга, усреднённые по направлениям (все игроки)."""
    return {n: _mean_coef([fit_layer(f, n) for f in frames.values()]) for n in _with_role_only(names)}


def player_folds(frames: Mapping[str, pd.DataFrame], k: int = N_FOLDS, seed: int = va.SEED) -> pd.Series:
    """Фолд игрока 0..k−1: внутри амплуа случайная перестановка, номер = позиция mod k."""
    roles = pd.concat([f.role for f in frames.values()])
    roles = roles[~roles.index.duplicated()].sort_index()
    rng = np.random.default_rng(seed + 11)
    fold = pd.Series(0, index=roles.index, dtype=int)
    for r in sorted(roles.unique()):
        ids = roles.index[roles.to_numpy() == r]
        fold[ids[rng.permutation(len(ids))]] = np.arange(len(ids)) % k
    return fold


def crossfit_predict(frames: Mapping[str, pd.DataFrame], names: Iterable[str], folds: pd.Series) -> dict:
    """Прогноз каждого игрока коэффициентами, подогнанными без его фолда (по всем направлениям)."""
    names = _with_role_only(names)
    out = {d: f[["role", "target", "toi_s"]].copy() for d, f in frames.items()}
    for d in out:
        for n in names:
            out[d][f"p::{n}"] = np.nan
    for k in sorted(folds.unique()):
        test_ids = set(folds.index[folds.to_numpy() == k])
        train = {d: f[~f.index.isin(test_ids)] for d, f in frames.items()}
        train = {d: f for d, f in train.items() if len(f)}
        for n in names:
            coef = _mean_coef([fit_layer(f, n) for f in train.values()])
            for d, f in frames.items():
                m = f.index.isin(test_ids)
                if m.any():
                    out[d].loc[m, f"p::{n}"] = predict(f[m], n, coef)
    return out


# --- метрики ошибки -------------------------------------------------------------------------

def _err(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict:
    sw = w.sum()
    if sw <= 0 or not len(y):
        return {"n": int(len(y)), "mse": float("nan"), "rmse": float("nan"), "mae": float("nan")}
    e = y - p
    mse = float((w * e * e).sum() / sw)
    return {"n": int(len(y)), "mse": mse, "rmse": float(np.sqrt(mse)), "mae": float((w * np.abs(e)).sum() / sw)}


def _ss(mse: float, mse_base: float) -> float:
    return float(1.0 - mse / mse_base) if np.isfinite(mse_base) and mse_base > 0 else float("nan")


def error_metrics(pred: pd.DataFrame, names: Iterable[str]) -> dict:
    """MSE/RMSE/MAE с весом TOI внутри амплуа и объединённо; SS против «только амплуа»."""
    names = _with_role_only(names)
    roles = pred.role.to_numpy().astype(str)
    y, w = pred.target.to_numpy(dtype=float), pred.toi_s.to_numpy(dtype=float)
    groups = {r: roles == r for r in ROLES if (roles == r).any()}
    base_p = pred[f"p::{ROLE_ONLY}"].to_numpy(dtype=float)
    base = {g: _err(y[m], base_p[m], w[m]) for g, m in groups.items()}
    base_pooled = _err(y, base_p, w)
    out = {}
    for n in names:
        p = pred[f"p::{n}"].to_numpy(dtype=float)
        by_role = {}
        for g, m in groups.items():
            e = _err(y[m], p[m], w[m])
            by_role[g] = {**e, "ss": _ss(e["mse"], base[g]["mse"])}
        pooled = _err(y, p, w)
        out[n] = {"by_role": by_role, "pooled": {**pooled, "ss": _ss(pooled["mse"], base_pooled["mse"])}}
    return out


def _pooled_point(y, w, preds: dict, base: str = ROLE_ONLY) -> dict:
    mse = {n: _err(y, p, w)["mse"] for n, p in preds.items()}
    return {n: {"mse": m, "rmse": float(np.sqrt(m)), "ss": _ss(m, mse[base])} for n, m in mse.items()}


def error_report(pred: pd.DataFrame, names: Iterable[str], n_boot: int = va.N_BOOT, seed: int = va.SEED,
                 ref: str = REFERENCE) -> dict:
    """Метрики + CI бутстрэпа игроков (стратификация по амплуа) и парные разности «ref − X»."""
    names = _with_role_only(names)
    pred = pred.dropna(subset=[f"p::{n}" for n in names])
    metrics = error_metrics(pred, names)
    y, w = pred.target.to_numpy(dtype=float), pred.toi_s.to_numpy(dtype=float)
    preds = {n: pred[f"p::{n}"].to_numpy(dtype=float) for n in names}
    groups = pred.role.to_numpy().astype(str)
    rng = np.random.default_rng(seed + 5)
    boots = []
    for _ in range(n_boot if len(pred) else 0):
        ix = _boot_indices(groups, rng)
        boots.append(_pooled_point(y[ix], w[ix], {n: p[ix] for n, p in preds.items()}))
    for n in names:
        metrics[n]["pooled"]["ss_ci"] = _ci([b[n]["ss"] for b in boots])
        metrics[n]["pooled"]["rmse_ci"] = _ci([b[n]["rmse"] for b in boots])
    diffs = {}
    if ref in names:
        for n in names:
            if n == ref:
                continue
            diffs[n] = {"ss_diff": metrics[ref]["pooled"]["ss"] - metrics[n]["pooled"]["ss"],
                        "ss_diff_ci": _ci([b[ref]["ss"] - b[n]["ss"] for b in boots]),
                        "rmse_diff": metrics[ref]["pooled"]["rmse"] - metrics[n]["pooled"]["rmse"],
                        "rmse_diff_ci": _ci([b[ref]["rmse"] - b[n]["rmse"] for b in boots])}
    return {"n_players": int(len(pred)), "n_boot": n_boot, "metrics": metrics, "o1_minus": diffs,
            "bootstrap": "players, stratified by role (paired, predictions fixed)"}


# --- выборки «рейтинг по прошлому → цель по будущему» ---------------------------------------

def target_table(pm: pd.DataFrame) -> pd.DataFrame:
    """Таблица общей цели (on-ice xGD при 5 на 5) в формате MF_COLS."""
    return bl.metric_frame(pm, bl.METRICS[TARGET])


def forecast_frame(ratings: Mapping[str, pd.Series], target: pd.DataFrame) -> pd.DataFrame:
    """Игроки цели с конечными значениями всех рейтингов: role, target, toi_s, r::<имя>."""
    return va._paired_frame(ratings, target, 0.0)


def half_frame(pm: pd.DataFrame, rating_fns: Mapping[str, RatingFn], past: set, future: set,
               min_toi_s: float) -> pd.DataFrame:
    """Рейтинги по матчам `past`, цель по `future`; TOI цели ≥ порога в обеих частях."""
    if past & future:
        raise ValueError("half_frame: прошлые и будущие матчи пересекаются")
    mf = target_table(pm)
    t_past = va.target_rates(mf[mf.match_id.isin(past)])
    t_fut = va.target_rates(mf[mf.match_id.isin(future)])
    ids = t_past.index[t_past.toi_s >= min_toi_s].intersection(t_fut.index[t_fut.toi_s >= min_toi_s])
    pm_past = pm[pm.match_id.isin(past)]
    ratings = {n: fn(pm_past) for n, fn in rating_fns.items()}
    return forecast_frame(ratings, t_fut.loc[ids])


def train_half_frames(pm: pd.DataFrame, rating_fns: Mapping[str, RatingFn], seed: int = va.SEED,
                      min_toi_s: float = va.MIN_TOI_PRED_S) -> dict:
    halves = va.assign_halves(pm.match_id, seed)
    a, b = set(halves.index[halves == 0]), set(halves.index[halves == 1])
    return {"A->B": half_frame(pm, rating_fns, a, b, min_toi_s),
            "B->A": half_frame(pm, rating_fns, b, a, min_toi_s)}


def train_forecast(pm: pd.DataFrame, rating_fns: Mapping[str, RatingFn], seed: int = va.SEED,
                   min_toi_s: float = va.MIN_TOI_PRED_S, n_boot: int = va.N_BOOT, k_folds: int = N_FOLDS,
                   ref: str = REFERENCE, frames: Mapping[str, pd.DataFrame] | None = None) -> dict:
    """Train: перекрёстно подогнанный слой на случайных половинах матчей; коэффициенты для вне-train."""
    frames = dict(frames) if frames is not None else train_half_frames(pm, rating_fns, seed, min_toi_s)
    names = list(rating_fns)
    folds = player_folds(frames, k_folds, seed)
    preds = crossfit_predict(frames, names, folds)
    directions = {d: error_report(p, names, n_boot, seed, ref) for d, p in preds.items()}
    mean_ss = {n: float(np.nanmean([directions[d]["metrics"][n]["pooled"]["ss"] for d in directions]))
               for n in _with_role_only(names)}
    return {"target": TARGET, "split": "random match halves", "seed": seed, "min_toi_s": min_toi_s,
            "k_folds": k_folds, "layer": "y = a_role + b·z, WLS с весом будущего TOI; коэффициенты — "
            "перекрёстно по игрокам, среднее по двум направлениям",
            "directions": directions, "mean_ss": mean_ss, "coefficients": fit_coefs(frames, names),
            "z_within_role_sd": (sd := {n: within_role_sd(frames, n) for n in names}),
            "degenerate": sorted(n for n, v in sd.items() if not (v >= DEGENERATE_Z_SD))}


def holdout_forecast(frame: pd.DataFrame, coefs: Mapping[str, Mapping], n_boot: int = va.N_BOOT,
                     seed: int = va.SEED, ref: str = REFERENCE) -> dict:
    """Замороженные (train) коэффициенты на другой выборке.

    Наклон, переподогнанный на ней, — только диагностика согласованности шкал (не для выбора).
    """
    names = [n for n in coefs if n != ROLE_ONLY]
    pred = frame[["role", "target", "toi_s"]].copy()
    for n in _with_role_only(names):
        pred[f"p::{n}"] = predict(frame, n, coefs[n])
    rep = error_report(pred, names, n_boot, seed, ref)
    slopes = {n: fit_layer(frame, n)["b"] if len(frame) else float("nan") for n in names}
    return {**rep, "holdout_slope": slopes, "train_slope": {n: float(coefs[n]["b"]) for n in names}}


def cross_split_frame(pm_train: pd.DataFrame, pm_test: pd.DataFrame, rating_fns: Mapping[str, RatingFn],
                      min_games: int = MIN_GAMES_CROSS) -> tuple[pd.DataFrame, dict]:
    """Train → test: рейтинги по всему train, цель — on-ice xGD/60 при 5 на 5 в test.

    Участвуют игроки с ≥ `min_games` матчами с допущенной экспозицией и в train, и в test.
    """
    g_tr = pm_train[pm_train.toi_rate_s > 0].groupby("player_id").match_id.nunique()
    t_te = va.target_rates(target_table(pm_test))
    ids = g_tr.index[g_tr >= min_games].intersection(t_te.index[t_te.games >= min_games])
    ratings = {n: fn(pm_train) for n, fn in rating_fns.items()}
    frame = forecast_frame(ratings, t_te.loc[ids])
    return frame, {"n_overlap_players": int(len(g_tr.index.intersection(t_te.index))),
                   "n_min_games_both": int(len(ids)), "n_used": int(len(frame)), "min_games": min_games}
