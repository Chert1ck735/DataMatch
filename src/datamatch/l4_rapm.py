"""L4 (этап 1): RAPM при 5 на 5 — отделение игрока от партнёров, соперников и команды.

Модель (архитектура, L4). Каждый допущенный отрезок постоянного состава 5 на 5 даёт две строки —
по одной на атакующую сторону:

    y_s = xG_s / h_s = μ + Σ_{i ∈ атака} β^O_i + Σ_{j ∈ защита} β^D_j + u^O_{команда атаки}
          + u^D_{команда защиты} + γᵀc_s + ε_s,     вес h_s (часы),

c_s — контроли (атакующие — хозяева; счёт атакующих −2…+2; зона первого вбрасывания отрезка
с точки зрения атакующих или «со смены»). β^D > 0 — при игроке в защите соперник создаёт
больше. Итог O2 = β^O − β^D, xG/60 разницы при 5 на 5 (больше — лучше).

Ridge к prior:  min Σ h_s (y_s − x_sᵀb)² + λ Σ_{игроки, команды} (b_j − p_j)²,
где p — SPM-prior для игроков (0 для команд); μ и контроли не штрафуются (численный пол 1e-8).
Нормальные уравнения (XᵀHX + Λ) b = Xᵀxg + Λp решаются сопряжёнными градиентами с диагональным
предобусловливанием (X разреженная, ~2·10⁴ столбцов). λ — в часах игры: prior весит как λ часов
данных. λ выбирается кросс-валидацией по целым матчам: взвешенный MSE xG/60 на отложенных матчах.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import cg

ROW_COLS = ["match_id", "stint", "dur_s", "xg", "att", "dfn", "att_team", "def_team", "home", "score", "zone"]
SCORE_LEVELS = (-2, -1, 1, 2)            # 0 — база
ZONE_LEVELS = ("off", "neutral", "def")  # «со смены» (fly) — база
CONTROL_NAMES = ["home"] + [f"score{v:+d}" for v in SCORE_LEVELS] + [f"zone_{z}" for z in ZONE_LEVELS]
UNPENALIZED = 1e-8
CG_RTOL = 1e-10
CG_MAXITER = 5000
LAMBDA_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
CV_FOLDS = 5
SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class Design:
    """Разреженная матрица дизайна и индексы столбцов."""
    X: sparse.csr_matrix
    xg: np.ndarray
    h: np.ndarray
    match_id: np.ndarray
    players: pd.Index
    teams: pd.Index
    n_ctrl: int

    @property
    def o_slice(self) -> slice:
        s = 1 + self.n_ctrl
        return slice(s, s + len(self.players))

    @property
    def d_slice(self) -> slice:
        s = 1 + self.n_ctrl + len(self.players)
        return slice(s, s + len(self.players))

    @property
    def team_slice(self) -> slice:
        s = 1 + self.n_ctrl + 2 * len(self.players)
        return slice(s, s + 2 * len(self.teams))


@dataclass(frozen=True)
class RapmFit:
    beta_o: pd.Series
    beta_d: pd.Series
    o2: pd.Series
    controls: dict
    team_o: pd.Series
    team_d: pd.Series
    lam: float
    toi_h: pd.Series
    cg_info: dict = field(default_factory=dict)


def _validate(rows: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ROW_COLS if c not in rows.columns]
    if missing:
        raise KeyError(f"RAPM: нет колонок {missing}")
    bad = rows.att.map(len).ne(5) | rows.dfn.map(len).ne(5)
    if bad.any():
        raise ValueError(f"RAPM: {int(bad.sum())} строк не 5 на 5")
    return rows[rows.dur_s.to_numpy(dtype=float) > 0]


def build_design(rows: pd.DataFrame, players: pd.Index | None = None, teams: pd.Index | None = None) -> Design:
    """Матрица дизайна: [μ | контроли | β^O игроков | β^D игроков | u^O команд | u^D команд]."""
    r = _validate(rows)
    att = np.array(r.att.tolist(), dtype=object)
    dfn = np.array(r.dfn.tolist(), dtype=object)
    if players is None:
        players = pd.Index(sorted(set(att.ravel()) | set(dfn.ravel())))
    if teams is None:
        teams = pd.Index(sorted(set(r.att_team) | set(r.def_team)))
    n, P, T = len(r), len(players), len(teams)
    nc = len(CONTROL_NAMES)
    ctrl = np.column_stack([r.home.to_numpy(float)]
                           + [(r.score.to_numpy() == v).astype(float) for v in SCORE_LEVELS]
                           + [(r.zone.to_numpy() == z).astype(float) for z in ZONE_LEVELS])
    ai = players.get_indexer(att.ravel()).reshape(n, 5)
    di = players.get_indexer(dfn.ravel()).reshape(n, 5)
    if (ai < 0).any() or (di < 0).any():
        raise ValueError("RAPM: игрок строки отсутствует в индексе игроков")
    ti, tj = teams.get_indexer(r.att_team), teams.get_indexer(r.def_team)
    rowi = np.repeat(np.arange(n), 1 + nc + 10 + 2)
    cols = np.column_stack([np.zeros(n, int), np.tile(np.arange(1, 1 + nc), (n, 1)), 1 + nc + ai,
                            1 + nc + P + di, 1 + nc + 2 * P + ti, 1 + nc + 2 * P + T + tj]).ravel()
    vals = np.column_stack([np.ones(n), ctrl, np.ones((n, 10)), np.ones(n), np.ones(n)]).ravel()
    X = sparse.csr_matrix((vals, (rowi, cols)), shape=(n, 1 + nc + 2 * P + 2 * T))
    X.eliminate_zeros()
    return Design(X, r.xg.to_numpy(float), r.dur_s.to_numpy(float) / SECONDS_PER_HOUR,
                  r.match_id.to_numpy(), players, teams, nc)


def _penalty(d: Design, lam: float) -> np.ndarray:
    pen = np.full(d.X.shape[1], float(lam))
    pen[: 1 + d.n_ctrl] = UNPENALIZED
    return pen


def _prior_vec(d: Design, prior_o: pd.Series | None, prior_d: pd.Series | None) -> np.ndarray:
    p = np.zeros(d.X.shape[1])
    if prior_o is not None:
        p[d.o_slice] = prior_o.reindex(d.players).fillna(0.0).to_numpy(float)
    if prior_d is not None:
        p[d.d_slice] = prior_d.reindex(d.players).fillna(0.0).to_numpy(float)
    return p


def normal_parts(d: Design, rows_mask: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray]:
    """XᵀHX и Xᵀxg по строкам `rows_mask` (не зависят от λ — считаются один раз на фолд)."""
    Xm = d.X[rows_mask]
    return (Xm.T.multiply(d.h[rows_mask]) @ Xm).tocsr(), np.asarray(Xm.T @ d.xg[rows_mask]).ravel()


def solve(d: Design, rows_mask: np.ndarray, lam: float, prior: np.ndarray,
          parts: tuple[sparse.csr_matrix, np.ndarray] | None = None) -> tuple[np.ndarray, dict]:
    """(XᵀHX + Λ) b = Xᵀxg + Λp по строкам `rows_mask`; сопряжённые градиенты."""
    A0, rhs0 = parts if parts is not None else normal_parts(d, rows_mask)
    pen = _penalty(d, lam)
    A = A0 + sparse.diags(pen)
    rhs = rhs0 + pen * prior
    diag = A.diagonal()
    M = sparse.diags(1.0 / np.where(diag > 0, diag, 1.0))
    b, info = cg(A, rhs, x0=prior.copy(), rtol=CG_RTOL, maxiter=CG_MAXITER, M=M)
    if info != 0:
        raise RuntimeError(f"RAPM: CG не сошёлся (info={info}, λ={lam})")
    return b, {"cg_info": int(info), "n_rows": int(rows_mask.sum())}


def _fit_from_b(d: Design, b: np.ndarray, lam: float, info: dict) -> RapmFit:
    bo = pd.Series(b[d.o_slice], index=d.players, name="beta_o")
    bd = pd.Series(b[d.d_slice], index=d.players, name="beta_d")
    T = len(d.teams)
    tb = b[d.team_slice]
    Xp = d.X[:, d.o_slice] + d.X[:, d.d_slice]
    toi = pd.Series(np.asarray(Xp.T @ d.h).ravel() / 2.0, index=d.players, name="toi_h")
    return RapmFit(bo, bd, (bo - bd).rename("O2"), dict(zip(["mu"] + CONTROL_NAMES, b[: 1 + d.n_ctrl])),
                   pd.Series(tb[:T], index=d.teams), pd.Series(tb[T:], index=d.teams), float(lam), toi, info)


def fit_rapm(rows: pd.DataFrame, lam: float, prior_o: pd.Series | None = None,
             prior_d: pd.Series | None = None) -> RapmFit:
    """Ridge-RAPM на всех строках; без prior — ridge к нулю."""
    d = build_design(rows)
    b, info = solve(d, np.ones(len(d.xg), dtype=bool), lam, _prior_vec(d, prior_o, prior_d))
    return _fit_from_b(d, b, lam, info)


def match_folds(match_id, k: int = CV_FOLDS, seed: int = 0) -> np.ndarray:
    """Фолд строки = фолд её матча (случайная перестановка матчей, номер mod k)."""
    mids = pd.Series(np.asarray(match_id))
    uniq = np.array(sorted(mids.unique()), dtype=object)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold = pd.Series(np.empty(len(uniq), dtype=int), index=uniq)
    fold.iloc[perm] = np.arange(len(uniq)) % k
    return fold.reindex(mids).to_numpy(dtype=int)


def cv_lambda(rows: pd.DataFrame, grid=LAMBDA_GRID, k: int = CV_FOLDS, seed: int = 0,
              prior_o: pd.Series | None = None, prior_d: pd.Series | None = None) -> tuple[float, dict]:
    """λ с наименьшим взвешенным MSE xG/60 на отложенных целых матчах; игроки без строк — prior."""
    d = build_design(rows)
    folds = match_folds(d.match_id, k, seed)
    prior = _prior_vec(d, prior_o, prior_d)
    parts = {f: normal_parts(d, folds != f) for f in range(k) if (folds == f).any() and (folds != f).any()}
    loss = {}
    for lam in grid:
        sse = sh = 0.0
        for f, pf in parts.items():
            te = folds == f
            b, _ = solve(d, folds != f, lam, prior, pf)
            pred = d.X[te] @ b
            sse += float(((d.xg[te] - d.h[te] * pred) ** 2 / d.h[te]).sum())
            sh += float(d.h[te].sum())
        loss[float(lam)] = sse / sh if sh > 0 else float("nan")
    best = min(loss, key=lambda v: loss[v])
    return best, loss


def spm_prior(beta0: Mapping[str, pd.Series], features: pd.DataFrame, weights: pd.Series,
              eligible: pd.Index, k: int = CV_FOLDS, seed: int = 0) -> dict[str, pd.Series]:
    """SPM-prior: β^O, β^D ridge-к-нулю ≈ a_амплуа + признаки·w (WLS, вес — TOI 5×5).

    `features` — индекс player_id, колонка `role` и числовые признаки (компоненты L3 игрока).
    Веса учатся на игроках `eligible` (≥ 10 матчей); для них prior — перекрёстно по игрокам
    (без собственной цели), для остальных — по всем eligible.
    """
    num = [c for c in features.columns if c != "role"]
    roles = features.role.astype(str)
    design = pd.concat([pd.get_dummies(roles, dtype=float), features[num].astype(float)], axis=1)
    elig = design.index.intersection(eligible)
    rng = np.random.default_rng(seed + 23)
    fold = pd.Series(rng.permutation(len(elig)) % k, index=elig)
    out = {}
    for name, beta in beta0.items():
        # игрок из eligible без цели (нет допущенных отрезков 5 на 5) в веса prior не входит
        el = elig.intersection(beta.dropna().index)
        fold_n = fold.reindex(el)
        y = beta.reindex(el).to_numpy(float)
        w = np.sqrt(weights.reindex(el).fillna(0.0).to_numpy(float))

        def fit(mask: np.ndarray) -> np.ndarray:
            Xe = design.loc[el[mask]].to_numpy(float)
            return np.linalg.lstsq(Xe * w[mask, None], y[mask] * w[mask], rcond=None)[0]

        pred = pd.Series(design.to_numpy(float) @ fit(np.ones(len(el), dtype=bool)), index=design.index)
        for f in range(k):
            m = (fold_n == f).to_numpy()
            if m.any() and (~m).any():
                pred.loc[el[m]] = design.loc[el[m]].to_numpy(float) @ fit(~m)
        out[name] = pred
    return out
