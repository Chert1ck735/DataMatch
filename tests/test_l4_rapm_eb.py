"""RAPM этап 2 (METHODOLOGY 4.3б): λ по эмпирическому Байесу и prior внутри регрессии.

Синтетика с известными параметрами: β^O_i = z_iᵀγ^O + u^O_i, β^D_i = z_iᵀγ^D + u^D_i,
u ~ N(0, τ²), xG строки = h·(μ + Σ_атака β^O + Σ_защита β^D) + шум с дисперсией σ²·h.
Контракты: (1) σ² по остаткам и пуассоновская σ² (по xG² бросков) восстанавливаются; (2) ЭМ со
следом апостериорной ковариации восстанавливает τ², без следа — занижает; (3) γ восстанавливается;
(4) O2 = β^O − β^D коррелирует с истиной сильнее, чем prior zᵀγ; (5) без командных эффектов.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datamatch import l4_rapm as rp
from datamatch import l4_rapm_eb as eb

N_TEAMS = 10
PER_TEAM = 14            # полевых в команде; пятёрки набираются случайно — партнёры отделимы
TAU = 0.15               # sd индивидуальной части u, xG/60
GAMMA_O = np.array([0.20, -0.10])
GAMMA_D = np.array([-0.15, 0.05])
SIGMA2 = 0.35            # шум ставки xG/60 за час (как на реальных данных)


def synth(n_matches: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    players = [f"T{t}P{k:02d}" for t in range(N_TEAMS) for k in range(PER_TEAM)]
    z = pd.DataFrame(rng.normal(size=(len(players), 2)), index=players, columns=["f1", "f2"])
    roles = pd.Series(rng.choice(["C", "W", "D"], len(players)), index=players)
    uo = pd.Series(rng.normal(0, TAU, len(players)), index=players)
    ud = pd.Series(rng.normal(0, TAU, len(players)), index=players)
    bo = z.to_numpy() @ GAMMA_O + uo
    bd = z.to_numpy() @ GAMMA_D + ud
    team_players = {t: [p for p in players if p.startswith(f"T{t}P")] for t in range(N_TEAMS)}
    rows = []
    for m in range(n_matches):
        h_t, a_t = rng.choice(N_TEAMS, 2, replace=False)
        for s in range(50):
            dur = rng.uniform(15, 90)
            hh = dur / 3600
            hs = list(rng.choice(team_players[h_t], 5, replace=False))
            as_ = list(rng.choice(team_players[a_t], 5, replace=False))
            for att, dfn, home in ((hs, as_, 1), (as_, hs, 0)):
                rate = 2.3 + bo[att].sum() + bd[dfn].sum()
                xg = hh * rate + np.sqrt(SIGMA2 * hh) * rng.normal()
                rows.append((f"m{m:04d}", s, dur, xg, tuple(att), tuple(dfn), f"t{h_t if home else a_t}",
                             f"t{a_t if home else h_t}", home, 0, "fly", SIGMA2 * hh))
    df = pd.DataFrame(rows, columns=rp.ROW_COLS + ["xg2"])
    feats = z.assign(role=roles)
    return df, feats, bo, bd


@pytest.fixture(scope="module")
def data():
    return synth()


def test_sigma2_estimators(data):
    rows, feats, _, _ = data
    assert eb.sigma2_poisson(rows) == pytest.approx(SIGMA2, rel=0.02)     # xg2 здесь = E[шум²]
    fit = eb.fit_eb(rows, feats)
    assert fit.sigma2_resid == pytest.approx(SIGMA2, rel=0.1)


def test_tau2_with_trace_recovered_without_trace_underestimated(data):
    rows, feats, _, _ = data
    with_tr = eb.fit_eb(rows, feats, sigma2=SIGMA2, use_trace=True)
    no_tr = eb.fit_eb(rows, feats, sigma2=SIGMA2, use_trace=False)
    for side in ("o", "d"):
        assert with_tr.tau2[side] == pytest.approx(TAU ** 2, rel=0.35)
        assert no_tr.tau2[side] < 0.8 * with_tr.tau2[side]
    assert with_tr.converged


def test_gamma_recovered_and_o2_beats_prior(data):
    rows, feats, bo, bd = data
    fit = eb.fit_eb(rows, feats, sigma2=SIGMA2)
    np.testing.assert_allclose(fit.gamma["o"][["f1", "f2"]].to_numpy(), GAMMA_O, atol=0.05)
    np.testing.assert_allclose(fit.gamma["d"][["f1", "f2"]].to_numpy(), GAMMA_D, atol=0.05)
    truth = (bo - bd).reindex(fit.o2.index)
    r_o2 = np.corrcoef(fit.o2, truth)[0, 1]
    r_prior = np.corrcoef(fit.prior.reindex(fit.o2.index), truth)[0, 1]
    assert r_o2 > r_prior + 0.05


def test_lambda_is_sigma2_over_tau2_and_no_team_columns(data):
    rows, feats, _, _ = data
    fit = eb.fit_eb(rows, feats, sigma2=SIGMA2)
    for side in ("o", "d"):
        assert fit.lam[side] == pytest.approx(fit.sigma2 / fit.tau2[side])
    assert not fit.with_teams


def test_lambda_scale_diagnostic_moves_toward_prior(data):
    rows, feats, _, _ = data
    base = eb.fit_eb(rows, feats, sigma2=SIGMA2)
    big = eb.fit_eb(rows, feats, sigma2=SIGMA2, lam_scale=50.0)
    d_base = np.abs(base.o2 - base.prior.reindex(base.o2.index)).mean()
    d_big = np.abs(big.o2 - big.prior.reindex(big.o2.index)).mean()
    assert d_big < d_base


def test_hutchinson_trace_matches_exact(data):
    """Стохастический след (как на реальных данных) даёт ту же τ², что точный, в пределах шума оценки."""
    rows, feats, _, _ = data
    exact = eb.fit_eb(rows, feats, sigma2=SIGMA2, trace="exact")
    hutch = eb.fit_eb(rows, feats, sigma2=SIGMA2, trace="hutchinson")
    for side in ("o", "d"):
        assert hutch.tau2[side] == pytest.approx(exact.tau2[side], rel=0.1)
    assert hutch.converged


def test_fixed_tau2_skips_em_and_sets_lambda(data):
    rows, feats, _, _ = data
    fit = eb.fit_eb(rows, feats, sigma2=SIGMA2, tau2_fixed={"o": 0.02, "d": 0.04}, lam_scale=2.0)
    assert fit.lam["o"] == pytest.approx(2.0 * SIGMA2 / 0.02) and fit.lam["d"] == pytest.approx(2.0 * SIGMA2 / 0.04)
    assert fit.n_iter == 0


def test_verdict_fifth_condition_spread():
    from datamatch import l4_rapm_eval as ev
    ok = {"reliability": {"vs": {"onice_xgd_raw": {"diff_ci": [0.01, 0.2]}, "prior_only": {"diff_ci": [0.02, 0.3]}}},
          "forecast": {"delta_ss_ci": [0.001, 0.02]},
          "slope": {"A->B": {"b": 0.3, "b_ci": [0.1, 0.5]}, "B->A": {"b": 0.4, "b_ci": [0.2, 0.6]}},
          "robustness": {"spearman": 0.93}, "spread": {"ratio": 0.8}}
    assert ev.verdict(ok)["accepted"] and ev.verdict(ok)["conditions"]["5_spread"]
    flat = {**ok, "spread": {"ratio": 0.3}}
    assert not ev.verdict(flat)["accepted"] and not ev.verdict(flat)["conditions"]["5_spread"]


def test_trace_phase_always_runs_and_converges(data):
    """Дефект 08.10.2026: уточнение точным/стохастическим следом запускалось только после сходимости
    диагональной фазы, а та не сходилась за 200 итераций на реальных данных — τ² оставалась
    диагональной. Контракт: фаза следа выполняется всегда (n_trace_iter > 0) и сходится; при
    искусственно урезанной диагональной фазе результат тот же."""
    rows, feats, _, _ = data
    full = eb.fit_eb(rows, feats, sigma2=SIGMA2, trace="exact")
    assert full.converged and full.n_trace_iter > 0
    old = eb.DIAG_MAX_ITER
    eb.DIAG_MAX_ITER = 2
    try:
        short = eb.fit_eb(rows, feats, sigma2=SIGMA2, trace="exact")
    finally:
        eb.DIAG_MAX_ITER = old
    assert short.converged and short.n_trace_iter > 0
    for side in ("o", "d"):
        assert short.tau2[side] == pytest.approx(full.tau2[side], rel=1e-3)
