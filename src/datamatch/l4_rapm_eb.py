"""L4 RAPM, этап 2: λ по эмпирическому Байесу и prior внутри регрессии (METHODOLOGY 4.3б).

Модель строки (отрезок 5 на 5 × атакующая сторона, вес h часов, цель y = xG/h):

    y_s = μ + γᵀc_s + (Σ_атака z_i)ᵀγ^O + (Σ_защита z_j)ᵀγ^D + Σ_атака u^O_i + Σ_защита u^D_j + ε_s,

Var(ε_s) = σ²/h_s; u^O ~ (0, τ²_O), u^D ~ (0, τ²_D). Командных столбцов нет (O2 «абсолютный»;
вариант с командами — `with_teams=True`, только диагностика). z_i — признаки игрока (компоненты O1
и фиктивные амплуа без первой — иначе μ и полный набор фиктивных коллинеарны); γ^O, γ^D, μ и
контроли не штрафуются. β^O = zᵀγ^O + u^O, β^D = zᵀγ^D + u^D, O2 = β^O − β^D; prior = zᵀ(γ^O − γ^D).

σ² — пуассоновская оценка Σ xG²_бросков / Σ h (колонка `xg2` строк), по остаткам — проверка.
τ² — ЭМ: τ² = (‖û‖² + tr Σ_post)/p, Σ_post = σ²(XᵀHX + Λ)⁻¹ (блок u), p — игроки с данными.
След: точный (обратная матрица) для малых задач, иначе Хатчинсон (`HUTCH_PROBES` фиксированных
векторов Радемахера, решения CG). ЭМ в две фазы: сначала дешёвое диагональное приближение
Σ_j 1/(H_jj/σ² + 1/τ²) до сходимости, затем уточнение точным/стохастическим следом. Диагональное
приближение само по себе занижает τ² (на синтетике вдвое: информация игрока делится с партнёрами и
признаками prior) — проверка 08.10.2026 до расчёта на реальных данных. Без следа τ² занижен ещё
сильнее (тест). λ = σ²/τ².
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import cg

from . import l4_rapm as rp

EM_TOL = 1e-4
# Фаза 1 — тёплый старт по диагональному приближению; фаза 2 (след) выполняется всегда (дефект 08.10.2026:
# раньше она шла только после сходимости фазы 1, а та на реальных данных не сходилась за 200 итераций)
DIAG_MAX_ITER = 100
TRACE_MAX_ITER = 100
# Хатчинсон: число векторов и допуск фазы уточнения (шум оценки следа ~ √(2/K) по относительной мере)
HUTCH_PROBES = 30
HUTCH_TOL = 2e-3
EXACT_TRACE_MAX_PARAMS = 3000
TRACE_SEED = 20261008
TAU2_START = 0.01
TAU2_FLOOR = 1e-8
SIDES = ("o", "d")


@dataclass(frozen=True)
class EbFit:
    beta_o: pd.Series
    beta_d: pd.Series
    o2: pd.Series
    prior_o: pd.Series
    prior_d: pd.Series
    prior: pd.Series
    gamma: dict
    tau2: dict
    lam: dict
    sigma2: float
    sigma2_resid: float
    sigma2_poisson: float | None
    converged: bool
    n_iter: int
    with_teams: bool
    toi_h: pd.Series
    controls: dict = field(default_factory=dict)
    tau2_team: float | None = None
    n_trace_iter: int = 0


def sigma2_poisson(rows: pd.DataFrame) -> float:
    """Шум ставки xG/60 за час: Σ xG² бросков / Σ часов (дисперсия суммы xG пуассоновского потока)."""
    if "xg2" not in rows.columns:
        raise KeyError("для пуассоновской σ² нужна колонка xg2 (сумма квадратов xG бросков строки)")
    return float(rows.xg2.sum() / (rows.dur_s.sum() / rp.SECONDS_PER_HOUR))


def feature_matrix(features: pd.DataFrame, players: pd.Index) -> pd.DataFrame:
    """z игроков: числовые признаки и фиктивные амплуа без первой по алфавиту; неизвестный — нули."""
    num = [c for c in features.columns if c != "role"]
    f = features.reindex(players)
    roles = sorted(features.role.dropna().astype(str).unique())
    dummies = pd.DataFrame({f"role_{r}": (f.role.astype(str) == r).astype(float) for r in roles[1:]},
                           index=players)
    return pd.concat([f[num].astype(float).fillna(0.0), dummies], axis=1)


@dataclass(frozen=True)
class _Blocks:
    X: sparse.csr_matrix
    xg: np.ndarray
    h: np.ndarray
    players: pd.Index
    zcols: list
    n_free: int          # μ, контроли, γ^O, γ^D — без штрафа
    n_p: int
    n_t: int


def _blocks(rows: pd.DataFrame, features: pd.DataFrame, with_teams: bool) -> _Blocks:
    d = rp.build_design(rows)
    P, T, nc = len(d.players), len(d.teams), d.n_ctrl
    Z = feature_matrix(features, d.players)
    Xo, Xd = d.X[:, d.o_slice], d.X[:, d.d_slice]
    zo = sparse.csr_matrix(Xo @ Z.to_numpy(float))
    zd = sparse.csr_matrix(Xd @ Z.to_numpy(float))
    parts = [d.X[:, : 1 + nc], zo, zd, Xo, Xd] + ([d.X[:, d.team_slice]] if with_teams else [])
    X = sparse.hstack(parts, format="csr")
    return _Blocks(X, d.xg, d.h, d.players, list(Z.columns), 1 + nc + 2 * Z.shape[1], P, T if with_teams else 0)


def _solve(A0: sparse.csr_matrix, rhs: np.ndarray, pen: np.ndarray, x0: np.ndarray | None) -> np.ndarray:
    A = A0 + sparse.diags(pen)
    diag = A.diagonal()
    M = sparse.diags(1.0 / np.where(diag > 0, diag, 1.0))
    b, info = cg(A, rhs, x0=x0, rtol=rp.CG_RTOL, maxiter=rp.CG_MAXITER, M=M)
    if info != 0:
        raise RuntimeError(f"RAPM ЭБ: CG не сошёлся (info={info})")
    return b


def _penalty(bl: _Blocks, lam_o: float, lam_d: float, lam_t: float | None) -> np.ndarray:
    pen = np.concatenate([np.full(bl.n_free, rp.UNPENALIZED), np.full(bl.n_p, lam_o), np.full(bl.n_p, lam_d)])
    if bl.n_t:
        pen = np.concatenate([pen, np.full(2 * bl.n_t, lam_t)])
    return pen


def _mackay(u2: float, p: int, tr_cov: float, tau2: float) -> float:
    """Обновление Маккея τ² = ‖û‖² / γ, γ = p − tr Σ_post / τ² (та же неподвижная точка, что у ЭМ
    τ² = (‖û‖² + tr Σ_post)/p — максимум маргинального правдоподобия, — но сходится быстрее)."""
    gamma = p - tr_cov / tau2
    if not gamma > 0:
        return max((u2 + tr_cov) / max(p, 1), TAU2_FLOOR)          # запасной шаг ЭМ
    return max(u2 / gamma, TAU2_FLOOR)


def _tau2_update(u: np.ndarray, H: np.ndarray, sigma2: float, tau2: float, use_trace: bool) -> float:
    """Фаза 1: диагональное приближение Σ_jj ≈ 1/(H_jj/σ² + 1/τ²); без следа — ‖û‖²/p (занижает τ²)."""
    m = H > 0
    p = max(int(m.sum()), 1)
    u2 = float((u[m] ** 2).sum())
    if not use_trace:
        return max(u2 / p, TAU2_FLOOR)
    return _mackay(u2, p, float(np.sum(1.0 / (H[m] / sigma2 + 1.0 / tau2))), tau2)


def _block_traces(A0, pen, idx_list: list, probes: np.ndarray | None, warm: dict) -> list[float]:
    """Σ diag((A0 + pen)⁻¹) по индексам игроков с данными: точно (плотная обратная) или Хатчинсоном."""
    if probes is None:
        inv = np.linalg.inv(A0.toarray() + np.diag(pen))
        return [float(np.trace(inv[np.ix_(ix, ix)])) for ix in idx_list]
    tot = np.zeros(len(idx_list))
    for k in range(probes.shape[1]):
        v = probes[:, k]
        x = _solve(A0, v, pen, warm.get(k))
        warm[k] = x
        for i, ix in enumerate(idx_list):
            tot[i] += float(v[ix] @ x[ix])
    return list(tot / probes.shape[1])


def _trace_probes(n_params: int, idx_list: list, trace: str) -> np.ndarray | None:
    """None — точный след; иначе фиксированные векторы Радемахера на игроках с данными."""
    if trace == "exact" or (trace == "auto" and n_params <= EXACT_TRACE_MAX_PARAMS):
        return None
    rng = np.random.default_rng(TRACE_SEED)
    probes = np.zeros((n_params, HUTCH_PROBES))
    for ix in idx_list:
        probes[ix] = rng.choice([-1.0, 1.0], size=(len(ix), HUTCH_PROBES))
    return probes


def _refine_tau2(A0, rhs, bl: _Blocks, sl: dict, sl_t: slice, s2: float, tau2: dict, tau2_t: float | None,
                 trace: str) -> tuple[bool, int, dict, float | None]:
    """Фаза 2 ЭМ: τ² = (‖û‖² + σ²·tr[(A0 + Λ)⁻¹]_u)/p по игрокам с данными (точный/стохастический след).

    Игроки блока без данных (H_jj = 0) не связаны с остальными: их апостериорная дисперсия ровно τ²,
    û = 0 — в оценку τ² они ничего не добавляют и исключаются вместе со следом.
    """
    Hdiag = A0.diagonal()
    blocks = [sl["o"], sl["d"]] + ([sl_t] if bl.n_t else [])
    idx = [np.arange(b.start, b.stop)[Hdiag[b] > 0] for b in blocks]
    probes = _trace_probes(A0.shape[0], idx, trace)
    tol = EM_TOL if probes is None else HUTCH_TOL
    warm, b = {}, None
    for it in range(1, TRACE_MAX_ITER + 1):
        pen = _penalty(bl, s2 / tau2["o"], s2 / tau2["d"], s2 / tau2_t if bl.n_t else None)
        b = _solve(A0, rhs, pen, b)
        tr = _block_traces(A0, pen, idx, probes, warm)
        cur = [tau2["o"], tau2["d"]] + ([tau2_t] if bl.n_t else [])
        new = [_mackay(float((b[ix] ** 2).sum()), len(ix), s2 * t, c) for ix, t, c in zip(idx, tr, cur)]
        change = max(abs(n - c) / c for n, c in zip(new, cur))
        tau2 = {"o": new[0], "d": new[1]}
        tau2_t = new[2] if bl.n_t else None
        if change < tol:
            return True, it, tau2, tau2_t
    return False, TRACE_MAX_ITER, tau2, tau2_t


def fit_eb(rows: pd.DataFrame, features: pd.DataFrame, sigma2: float | None = None, use_trace: bool = True,
           lam_scale: float = 1.0, with_teams: bool = False, trace: str = "auto",
           tau2_fixed: dict | None = None) -> EbFit:
    """RAPM к prior zᵀγ с λ = σ²/τ² (ЭМ по τ² отдельно для атаки и защиты); `lam_scale` — диагностика."""
    bl = _blocks(rows, features, with_teams)
    A0 = (bl.X.T.multiply(bl.h) @ bl.X).tocsr()
    rhs = np.asarray(bl.X.T @ bl.xg).ravel()
    s2_pois = sigma2_poisson(rows) if "xg2" in rows.columns else None
    s2 = float(sigma2) if sigma2 is not None else s2_pois
    if s2 is None or not s2 > 0:
        raise ValueError("RAPM ЭБ: нужна σ² (колонка xg2 или явное значение)")
    o0 = bl.n_free
    sl = {"o": slice(o0, o0 + bl.n_p), "d": slice(o0 + bl.n_p, o0 + 2 * bl.n_p)}
    sl_t = slice(o0 + 2 * bl.n_p, o0 + 2 * bl.n_p + 2 * bl.n_t)
    Hdiag = A0.diagonal()
    tau2 = {s: TAU2_START for s in SIDES}
    tau2_t = TAU2_START if bl.n_t else None
    b, converged, it = None, False, 0
    if tau2_fixed is not None:      # диагностика λ × k с τ² основной подгонки: ЭМ не повторяется
        tau2 = {s: float(tau2_fixed[s]) for s in SIDES}
        tau2_t = float(tau2_fixed["t"]) if bl.n_t else None
    for it in range(1, (0 if tau2_fixed is not None else DIAG_MAX_ITER) + 1):
        pen = _penalty(bl, s2 / tau2["o"], s2 / tau2["d"], s2 / tau2_t if bl.n_t else None)
        b = _solve(A0, rhs, pen, b)
        new = {s: _tau2_update(b[sl[s]], Hdiag[sl[s]], s2, tau2[s], use_trace) for s in SIDES}
        new_t = _tau2_update(b[sl_t], Hdiag[sl_t], s2, tau2_t, use_trace) if bl.n_t else None
        change = max(abs(new[s] - tau2[s]) / tau2[s] for s in SIDES)
        if bl.n_t:
            change = max(change, abs(new_t - tau2_t) / tau2_t)
        tau2, tau2_t = new, new_t
        if change < EM_TOL:
            converged = True
            break
    n_trace = 0
    if tau2_fixed is not None:
        it, converged = 0, True
    elif use_trace:      # фаза следа — всегда; фаза 1 лишь тёплый старт
        converged, n_trace, tau2, tau2_t = _refine_tau2(A0, rhs, bl, sl, sl_t, s2, tau2, tau2_t, trace)
        it += n_trace
    lam = {s: float(lam_scale) * s2 / tau2[s] for s in SIDES}
    lam_t = float(lam_scale) * s2 / tau2_t if bl.n_t else None
    b = _solve(A0, rhs, _penalty(bl, lam["o"], lam["d"], lam_t), b)
    q = len(bl.zcols)
    nc = len(rp.CONTROL_NAMES)
    g_o = pd.Series(b[1 + nc: 1 + nc + q], index=bl.zcols)
    g_d = pd.Series(b[1 + nc + q: 1 + nc + 2 * q], index=bl.zcols)
    Z = feature_matrix(features, bl.players)
    prior_o = pd.Series(Z.to_numpy(float) @ g_o.to_numpy(), index=bl.players)
    prior_d = pd.Series(Z.to_numpy(float) @ g_d.to_numpy(), index=bl.players)
    beta_o = prior_o + b[sl["o"]]
    beta_d = prior_d + b[sl["d"]]
    pred = bl.X @ b
    resid = float(np.mean((bl.xg - bl.h * pred) ** 2 / bl.h))
    toi = pd.Series((Hdiag[sl["o"]] + Hdiag[sl["d"]]) / 2.0, index=bl.players)
    return EbFit(beta_o.rename("beta_o"), beta_d.rename("beta_d"), (beta_o - beta_d).rename("O2"),
                 prior_o, prior_d, (prior_o - prior_d).rename("PRIOR"), {"o": g_o, "d": g_d},
                 {s: float(tau2[s]) for s in SIDES}, lam, s2, resid, s2_pois, converged, it, with_teams, toi,
                 dict(zip(["mu"] + rp.CONTROL_NAMES, b[: 1 + nc])), tau2_t, n_trace)
