"""Карточка игрока: вбрасывания — Bradley–Terry с поправкой на соперника, площадку и зону (METHODOLOGY 11.6).

logit P(выиграл игрок хозяев) = θ_h − θ_a + β_home + β_зона (зона хозяев: def/off, нейтральная — база),
θ ~ N(0, τ²). Оценка — максимум апостериорной плотности; неопределённость — обратный гессиан (Лаплас).
Отдельная строка карточки, в O1 не входит.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ZONES = ("def", "off")                    # нейтральная зона — база
FLIP = {"off": "def", "def": "off", "neutral": "neutral"}
MAX_DENSE = 6000                          # больше игроков — диагональное приближение обратного гессиана


@dataclass(frozen=True)
class BtFit:
    theta: pd.Series                      # навык игрока (логит), стянутый к 0
    sd: pd.Series                         # апостериорное SD
    beta: dict                            # home, def, off
    tau2: float


def home_perspective(df: pd.DataFrame) -> pd.DataFrame:
    """Вбрасывание глазами хозяев: y = 1, если выиграл игрок хозяев; зона — с точки зрения хозяев."""
    home = (df["actor_team_id"] == df["home_team_id"]).to_numpy()
    zone = df["zone"].fillna("neutral").to_numpy(dtype=object)
    zone_home = np.where(home, zone, pd.Series(zone).map(FLIP).fillna("neutral").to_numpy(dtype=object))
    out = pd.DataFrame({"y": home.astype(int),
                        "home_pid": np.where(home, df["actor_id"], df["other_id"]),
                        "away_pid": np.where(home, df["other_id"], df["actor_id"]),
                        "zone_home": zone_home})
    if "match_id" in df:
        out.insert(0, "match_id", df["match_id"].to_numpy())
    return out


def _design(df: pd.DataFrame, players: pd.Index):
    h = players.get_indexer(df["home_pid"])
    a = players.get_indexer(df["away_pid"])
    z = np.column_stack([np.ones(len(df))] + [(df["zone_home"] == zn).to_numpy(float) for zn in ZONES])
    return h, a, z, df["y"].to_numpy(dtype=float)


def _eta(theta, beta, h, a, z):
    return theta[h] - theta[a] + z @ beta


def fit_bt(df: pd.DataFrame, tau2: float, beta: dict | None = None) -> BtFit:
    """MAP-оценка θ (и β, если не заданы). df: y, home_pid, away_pid, zone_home."""
    players = pd.Index(sorted(set(df["home_pid"]) | set(df["away_pid"])))
    h, a, z, y = _design(df, players)
    P = len(players)
    fixed = None if beta is None else np.array([beta["home"]] + [beta[zn] for zn in ZONES])

    def unpack(x):
        return (x[:P], x[P:]) if fixed is None else (x, fixed)

    def obj(x):
        th, b = unpack(x)
        eta = _eta(th, b, h, a, z)
        p = 1.0 / (1.0 + np.exp(-eta))
        nll = float(np.sum(np.logaddexp(0.0, eta) - y * eta)) + float(th @ th) / (2.0 * tau2)
        r = p - y
        g_th = np.bincount(h, weights=r, minlength=P) - np.bincount(a, weights=r, minlength=P) + th / tau2
        grad = g_th if fixed is not None else np.concatenate([g_th, z.T @ r])
        return nll, grad

    x0 = np.zeros(P if fixed is not None else P + z.shape[1])
    res = minimize(obj, x0, jac=True, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-8})
    th, b = unpack(res.x)
    p = 1.0 / (1.0 + np.exp(-_eta(th, b, h, a, z)))
    wgt = p * (1.0 - p)
    if P <= MAX_DENSE:
        H = np.zeros((P, P))
        np.add.at(H, (h, h), wgt)
        np.add.at(H, (a, a), wgt)
        np.add.at(H, (h, a), -wgt)
        np.add.at(H, (a, h), -wgt)
        H[np.diag_indices(P)] += 1.0 / tau2
        var = np.diag(np.linalg.inv(H))
    else:
        var = 1.0 / (np.bincount(h, weights=wgt, minlength=P) + np.bincount(a, weights=wgt, minlength=P) + 1.0 / tau2)
    beta_out = {"home": float(b[0]), **{zn: float(v) for zn, v in zip(ZONES, b[1:])}} if fixed is None else dict(beta)
    return BtFit(pd.Series(th, index=players), pd.Series(np.sqrt(var), index=players), beta_out, float(tau2))


def predict(fit: BtFit, df: pd.DataFrame) -> np.ndarray:
    """P(выиграл игрок хозяев); незнакомые игроки — θ = 0 (средний)."""
    th_h = df["home_pid"].map(fit.theta).fillna(0.0).to_numpy()
    th_a = df["away_pid"].map(fit.theta).fillna(0.0).to_numpy()
    b = fit.beta
    ctx = b["home"] + sum(np.where(df["zone_home"] == zn, b[zn], 0.0) for zn in ZONES)
    return 1.0 / (1.0 + np.exp(-(th_h - th_a + ctx)))


def choose_tau2(df: pd.DataFrame, grid, n_folds: int, seed: int) -> tuple[float, dict]:
    """τ² по log-loss кросс-валидации с группировкой по матчам (только разработка)."""
    matches = np.array(sorted(df["match_id"].unique()), dtype=object)
    fold_of = pd.Series(np.random.default_rng(seed).permutation(len(matches)) % n_folds, index=matches)
    fold = df["match_id"].map(fold_of).to_numpy()
    scores = {}
    for t2 in grid:
        ll = []
        for k in range(n_folds):
            tr, va = df[fold != k], df[fold == k]
            p = np.clip(predict(fit_bt(tr, t2), va), 1e-12, 1 - 1e-12)
            y = va["y"].to_numpy()
            ll.append(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        scores[t2] = float(np.mean(ll))
    return min(scores, key=scores.get), scores
