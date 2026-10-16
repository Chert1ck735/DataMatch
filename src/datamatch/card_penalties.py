"""Карточка игрока: удаления — заработанные и полученные (METHODOLOGY 11.4). Вне O1.

δ L3 ценность удаления не несёт: V(s) — разница xG до ближайшей остановки, а большинство начинается
в новом отрезке после свистка (проверено на данных, отчёт `penalties_report.json`). Поэтому удаления —
отдельная строка: частоты на 60 минут, стягивание Пуассон–гамма по амплуа и перевод в xG через ценность
одного нарушения ν = Σ (xG большинства − xG меньшинства) / число нарушений.

Моменты Пуассон–гамма (x_i = n_i / T_i, T — часы): m = Σn / ΣT,
τ² = [Σ T_i (x_i − m)² − m (N − 1)] / (ΣT − ΣT² / ΣT) — взвешенная дисперсия ставок за вычетом
пуассоновского шума; α = m²/τ², β = m/τ²; апостериорно (n_i + α)/(T_i + β), w_i = T_i/(T_i + β).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

SHOOTOUT_PERIOD = 0
PP_STATE, SH_STATE = "PP", "SH"     # численность узла L0 с точки зрения команды исполнителя
NO_SIGNAL_REL = 1.0e-6              # τ² ≤ 1e-6·m — сигнала нет (METHODOLOGY 11.4, поправка 08.10)


def penalty_counts(inf: pd.DataFrame) -> pd.DataFrame:
    """Нарушения на игрока и матч: actor — получил удаление, other — заработал; буллиты не входят."""
    e = inf[inf.period.astype(int) != SHOOTOUT_PERIOD]
    taken = e.dropna(subset=["actor_id"]).groupby(["match_id", "actor_id"]).size().rename("taken")
    drawn = e.dropna(subset=["other_id"]).groupby(["match_id", "other_id"]).size().rename("drawn")
    taken.index.names = drawn.index.names = ["match_id", "player_id"]
    out = pd.concat([taken, drawn], axis=1).fillna(0).astype(int)
    return out.reset_index()


def gamma_moments(n: np.ndarray, T: np.ndarray) -> dict:
    """Метод моментов Пуассон–гамма по игрокам с T > 0 (см. формулы модуля)."""
    n, T = np.asarray(n, dtype=float), np.asarray(T, dtype=float)
    ok = T > 0
    n, T = n[ok], T[ok]
    m = float(n.sum() / T.sum())
    x = n / T
    denom = float(T.sum() - (T ** 2).sum() / T.sum())
    tau2 = float(((T * (x - m) ** 2).sum() - m * (len(T) - 1)) / denom) if denom > 0 else 0.0
    no_signal = not (tau2 > 0 and m > 0)
    return {"m": m, "tau2": tau2, "no_signal": no_signal, "n_players": int(len(T)),
            "alpha": None if no_signal else m * m / tau2, "beta": None if no_signal else m / tau2}


def _nb_negloglik(params: np.ndarray, n: np.ndarray, T: np.ndarray) -> float:
    """−log маргинального правдоподобия пуассон–гамма (отрицательно-биномиального) в (log m, log τ²)."""
    m, tau2 = np.exp(params)
    a, b = m * m / tau2, m / tau2
    ll = gammaln(n + a) - gammaln(a) + a * np.log(b) - (n + a) * np.log(T + b) + n * np.log(T)
    return -float(ll.sum())


def gamma_mle(n: np.ndarray, T: np.ndarray) -> dict:
    """Prior Gamma(α, β) по ML маргинального NB-правдоподобия (игроки с T > 0); вид выхода как у моментов."""
    n, T = np.asarray(n, dtype=float), np.asarray(T, dtype=float)
    ok = T > 0
    n, T = n[ok], T[ok]
    m0 = float(n.sum() / T.sum()) if T.sum() > 0 else 0.0
    base = {"method": "ml_negative_binomial", "n_players": int(len(T))}
    if m0 <= 0:
        return base | {"m": m0, "tau2": 0.0, "no_signal": True, "alpha": None, "beta": None}
    mom = gamma_moments(n, T)
    tau0 = mom["tau2"] if mom["tau2"] > 0 else 0.1 * m0 * m0
    res = minimize(_nb_negloglik, np.log([m0, tau0]), args=(n, T), method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 4000})
    m, tau2 = (float(v) for v in np.exp(res.x))
    no_signal = not (tau2 > NO_SIGNAL_REL * m and m > 0)
    return base | {"m": m, "tau2": tau2, "no_signal": no_signal, "converged": bool(res.success),
                   "alpha": None if no_signal else m * m / tau2, "beta": None if no_signal else m / tau2}


def rescale_prior(mo: dict, factor: float) -> dict:
    """Prior для выборки, где наблюдается доля `factor` событий: λ_набл = factor·λ (α то же, β/factor)."""
    out = dict(mo, m=mo["m"] * factor, tau2=mo["tau2"] * factor * factor)
    if mo.get("alpha") is not None and mo.get("beta") is not None:
        out["beta"] = mo["beta"] / factor
    return out


def half_r_ci(xa: np.ndarray, xb: np.ndarray, role: np.ndarray, B: int, seed: int) -> dict:
    """Корреляция половин после центрирования внутри амплуа и 95%-интервал (бутстрэп по игрокам)."""
    xa, xb, role = np.asarray(xa, dtype=float), np.asarray(xb, dtype=float), np.asarray(role, dtype=object)

    def corr(idx: np.ndarray) -> float:
        a, b, r = xa[idx], xb[idx], role[idx]
        ca, cb = a.copy(), b.copy()
        for g in np.unique(r):
            sel = r == g
            ca[sel] -= a[sel].mean()
            cb[sel] -= b[sel].mean()
        den = np.sqrt((ca ** 2).sum() * (cb ** 2).sum())
        return float((ca * cb).sum() / den) if den > 0 else float("nan")

    rng = np.random.default_rng(seed)
    point = corr(np.arange(len(xa)))
    draws = np.array([corr(rng.integers(0, len(xa), len(xa))) for _ in range(B)])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"r": point, "lo": float(lo), "hi": float(hi), "n_players": int(len(xa)), "B": int(B)}


def gamma_posterior(n: np.ndarray, T: np.ndarray, alpha: float | None, beta: float | None,
                    m: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Апостериорная ставка и вес w; без сигнала — общая ставка m и w = 0."""
    n, T = np.asarray(n, dtype=float), np.asarray(T, dtype=float)
    if alpha is None or beta is None:
        return np.full(len(n), float(m)), np.zeros(len(n))
    return (n + alpha) / (T + beta), T / (T + beta)


def pp_net_by_match(nodes: pd.DataFrame, inf_by_match: pd.Series) -> pd.DataFrame:
    """По матчу: чистые xG и голы команды в большинстве (узлы PP минус узлы SH) и число нарушений."""
    sign = np.where(nodes.mp == PP_STATE, 1.0, np.where(nodes.mp == SH_STATE, -1.0, 0.0))
    t = pd.DataFrame({"match_id": nodes.match_id.to_numpy(), "net_xg": sign * nodes.r.to_numpy(dtype=float),
                      "net_goals": sign * nodes.is_goal.to_numpy(dtype=float)})
    per = t.groupby("match_id")[["net_xg", "net_goals"]].sum()
    idx = per.index.union(inf_by_match.index)
    per = per.reindex(idx, fill_value=0.0)
    return per.assign(n_inf=inf_by_match.reindex(idx, fill_value=0).astype(float))


def ratio_of_sums(per: pd.DataFrame, col: str) -> float:
    """ν = Σ чистой величины / Σ нарушений."""
    return float(per[col].sum() / per.n_inf.sum())


def bootstrap_ratio(per: pd.DataFrame, col: str, B: int, seed: int) -> np.ndarray:
    """Бутстрэп ν по матчам (строки per — матчи)."""
    rng = np.random.default_rng(seed)
    v, k = per[col].to_numpy(dtype=float), per.n_inf.to_numpy(dtype=float)
    idx = rng.integers(0, len(per), (B, len(per)))
    return v[idx].sum(axis=1) / k[idx].sum(axis=1)
