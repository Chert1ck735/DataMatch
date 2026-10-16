"""L5: оценка моментов (τ², σ², ω²) на синтетике с известной истиной и вырожденные случаи.

Генератор синтетики повторяет модель L5 в её собственных единицах. Ставка на 60 минут
считается как x = Σ / toi_s × 3600, поэтому при постоянном TOI на матч t_s и M матчах:

    credit_m = θ · t_s/3600 + (σ/60) · sqrt(t_s/60) · z_m + ω · t_s/3600 · g_m,

и тогда Var(x) = σ²/T_мин + ω²/M ровно по архитектуре. TOI на матч меняется между игроками:
при одинаковом TOI на матч столбцы плана (1/T_н + 1/T_ч) и (1/M_н + 1/M_ч) коллинеарны и ω²
из моментов не извлекается — это отдельно проверяется ниже.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datamatch.l5_shrink import ROLES, JointEB, UnivariateEB, psd_project

SHRINK = {"eps_abs": 1.0e-10, "eps_floor_rel": 1.0e-6, "jitter_start": 1.0e-6,
          "jitter_max": 1.0e-2, "min_toi_split_min": 5.0,
          "reliability_high": 0.7, "reliability_medium": 0.4}


def synth_panel(rng, n_players=600, k=2, T=None, Sig_s=None, Sig_o=None, cols=None,
                theta_fixed=None, role="C", toi_choices=(600.0, 1200.0, 2400.0),
                games_choices=(4, 10, 24)):
    """Панель «игрок × матч» с известными T_r, Σσ, Σω. Возвращает (кадр, амплуа, θ истинные)."""
    cols = cols or [f"c_{i}" for i in range(k)]
    k = len(cols)
    T = np.eye(k) * 0.04 if T is None else np.asarray(T, dtype=float)
    Sig_s = np.eye(k) * 9.0 if Sig_s is None else np.asarray(Sig_s, dtype=float)
    Sig_o = np.eye(k) * 0.05 if Sig_o is None else np.asarray(Sig_o, dtype=float)
    Lt, Ls, Lo = (np.linalg.cholesky(M + np.eye(k) * 1e-12) for M in (T, Sig_s, Sig_o))
    rows, thetas = [], {}
    mid = 0
    for i in range(n_players):
        pid = f"P{i:05d}"
        th = (np.asarray(theta_fixed, dtype=float) if theta_fixed is not None
              else Lt @ rng.standard_normal(k))
        thetas[pid] = th
        t_s = float(rng.choice(toi_choices))
        M = int(rng.choice(games_choices))
        t_h = t_s / 3600.0
        for _ in range(M):
            mid += 1
            z = Ls @ rng.standard_normal(k) / 60.0 * np.sqrt(t_s / 60.0)
            g = Lo @ rng.standard_normal(k) * t_h
            credit = th * t_h + z + g
            row = {"player_id": pid, "match_id": f"M{mid:06d}", "toi_s": t_s}
            row.update(dict(zip(cols, credit)))
            rows.append(row)
    frame = pd.DataFrame(rows)
    roles = pd.Series({p: role for p in thetas}, name="role")
    return frame, roles, thetas


# ---------------------------------------------------------------------------------------
# Сходимость оценок моментов

def test_joint_moments_recover_known_tau_sigma_omega():
    rng = np.random.default_rng(20261005)
    tau2, sig2, om2 = 0.05, 9.0, 0.06
    frame, roles, _ = synth_panel(rng, n_players=900, k=2, T=np.eye(2) * tau2,
                                  Sig_s=np.eye(2) * sig2, Sig_o=np.eye(2) * om2)
    eb = JointEB(["c_0", "c_1"], SHRINK).fit(frame, roles)
    assert eb.omega_identified, "при разном TOI на матч ω² обязан быть идентифицируем"
    assert not eb.no_signal["C"]
    tau_hat = np.diag(eb.T["C"])
    assert np.allclose(tau_hat, tau2, rtol=0.25), tau_hat
    assert np.allclose(np.diag(eb.Sig_s), sig2, rtol=0.20), np.diag(eb.Sig_s)
    # σ² и ω² разделяются только разбросом TOI на матч, поэтому ω² оценивается заметно грубее:
    # требуется верный порядок и знак, а точность проверяется по суммарному шуму s_i.
    assert np.allclose(np.diag(eb.Sig_o), om2, rtol=0.6), np.diag(eb.Sig_o)
    toi_min, games = 200.0, 10.0
    s_true = sig2 / toi_min + om2 / games
    s_hat = np.diag(eb.Sig_s) / toi_min + np.diag(eb.Sig_o) / games
    assert np.allclose(s_hat, s_true, rtol=0.15), (s_hat, s_true)
    # вне диагонали истинный ноль: оценка не должна создавать корреляцию из воздуха
    assert abs(eb.T["C"][0, 1]) < 0.5 * tau2


def test_joint_moments_recover_component_covariance():
    """Связанные компоненты: T_r должна воспроизводить знак и масштаб ковариации."""
    rng = np.random.default_rng(7)
    T = np.array([[0.05, 0.03], [0.03, 0.04]])
    frame, roles, _ = synth_panel(rng, n_players=900, T=T, Sig_s=np.eye(2) * 4.0,
                                  Sig_o=np.eye(2) * 0.02)
    eb = JointEB(["c_0", "c_1"], SHRINK).fit(frame, roles)
    assert eb.T["C"][0, 1] > 0.5 * T[0, 1]
    assert eb.T["C"][0, 1] < 2.0 * T[0, 1]
    assert np.allclose(eb.T["C"], eb.T["C"].T), "T_r обязана быть симметричной"
    assert np.linalg.eigvalsh(eb.T["C"]).min() > 0, "T_r обязана быть положительно определённой"


def test_univariate_moments_recover_known_values():
    rng = np.random.default_rng(99)
    tau2, sig2, om2 = 0.08, 6.0, 0.05
    frame, roles, _ = synth_panel(rng, n_players=900, k=1, T=np.array([[tau2]]),
                                  Sig_s=np.array([[sig2]]), Sig_o=np.array([[om2]]),
                                  cols=["x"])
    eb = UnivariateEB(1e-10).fit(frame, "x", roles, min_toi_s=300.0)
    assert eb.omega_identified
    assert eb.tau2["C"] == pytest.approx(tau2, rel=0.35)
    assert eb.sigma2 == pytest.approx(sig2, rel=0.25)
    assert eb.omega2 == pytest.approx(om2, rel=0.6)


def test_omega_not_identified_when_toi_per_match_constant():
    """Коллинеарный план моментов: ω² не выдаётся за оценку, а помечается неидентифицируемым."""
    rng = np.random.default_rng(3)
    frame, roles, _ = synth_panel(rng, n_players=300, k=1, cols=["x"],
                                  toi_choices=(1200.0,), games_choices=(10,))
    eb = UnivariateEB(1e-10).fit(frame, "x", roles, min_toi_s=300.0)
    assert not eb.omega_identified
    assert eb.omega2 == 0.0
    assert eb.sigma2 > 0


# ---------------------------------------------------------------------------------------
# Вырожденные случаи

def test_zero_between_player_signal_collapses_rating_to_prior():
    """Истинное τ² = 0: оценка моментов почти нулевая, w близко к нулю, рейтинг стянут к μ_r.

    Выборочная оценка τ² при истинном нуле положительна (шум оценивания), поэтому численная
    ветка tr(T_r) ≤ ε_abs здесь не срабатывает — проверяется именно статистический результат:
    разброс итогов должен быть на порядки меньше разброса сырых ставок на 60.
    """
    rng = np.random.default_rng(11)
    frame, roles, _ = synth_panel(rng, n_players=400, k=2, T=np.zeros((2, 2)),
                                  Sig_s=np.eye(2) * 4.0, Sig_o=np.eye(2) * 0.01,
                                  theta_fixed=[0.2, -0.1])
    eb = JointEB(["c_0", "c_1"], SHRINK).fit(frame, roles)
    g = frame.groupby("player_id").agg(c_0=("c_0", "sum"), c_1=("c_1", "sum"),
                                       toi_s=("toi_s", "sum"), games=("match_id", "nunique"))
    ids = list(g.index)
    res = eb.overall(ids, [roles[i] for i in ids], g[["c_0", "c_1"]].to_numpy(),
                     g.toi_s.to_numpy(dtype=float), g.games.to_numpy(dtype=float))
    raw = g[["c_0", "c_1"]].to_numpy().sum(1) / g.toi_s.to_numpy() * 3600.0
    assert float(np.trace(eb.T["C"])) < 0.01
    assert res["w"].max() < 0.15, res["w"].max()
    assert res["overall"].std() < 0.2 * raw.std()
    assert np.isfinite(res["overall"]).all()


def test_exactly_zero_covariance_triggers_no_signal_branch():
    """Нулевой сигнал в точном смысле (нет вклада вовсе): tr(T_r) = 0 -> запасная ветка, w = 0."""
    rng = np.random.default_rng(12)
    frame, roles, _ = synth_panel(rng, n_players=120, k=2)
    frame[["c_0", "c_1"]] = 0.0
    eb = JointEB(["c_0", "c_1"], SHRINK).fit(frame, roles)
    assert eb.no_signal["C"] is True
    assert eb.uni["C"]["tau2"] == pytest.approx(0.0, abs=1e-12)
    ids = list(frame.player_id.unique()[:20])
    g = frame[frame.player_id.isin(ids)].groupby("player_id").agg(
        c_0=("c_0", "sum"), c_1=("c_1", "sum"), toi_s=("toi_s", "sum"), games=("match_id", "nunique"))
    res = eb.overall(list(g.index), [roles[i] for i in g.index], g[["c_0", "c_1"]].to_numpy(),
                     g.toi_s.to_numpy(dtype=float), g.games.to_numpy(dtype=float))
    assert set(res["method"]) == {"univariate_fallback"}
    assert np.allclose(res["w"], 0.0)
    assert all(f == "сигнал не обнаружен" for f in res["flag"])
    assert np.isfinite(res["overall"]).all()


def test_single_match_per_player_does_not_crash_and_reports_no_halves():
    """Один матч на игрока: второй половины нет, моменты не оцениваются, расчёт не падает."""
    rng = np.random.default_rng(5)
    frame, roles, _ = synth_panel(rng, n_players=40, k=2, games_choices=(1,))
    eb = JointEB(["c_0", "c_1"], SHRINK).fit(frame, roles)
    assert eb.diag["n_split_players"] == 0
    assert eb.unstable is True
    assert all(eb.no_signal[r] for r in ROLES)
    ids = list(frame.player_id.unique()[:10])
    s = frame.set_index("player_id").loc[ids]
    res = eb.overall(ids, [roles[i] for i in ids], s[["c_0", "c_1"]].to_numpy(),
                     s.toi_s.to_numpy(dtype=float), np.ones(len(ids)))
    assert np.isfinite(res["overall"]).all()
    assert np.allclose(res["w"], 0.0)


def test_zero_toi_player_takes_prior_branch_without_division():
    """Нулевой TOI: апостериорное = prior (θ̂ = μ_r, V = T_r), без деления на TOI."""
    rng = np.random.default_rng(13)
    frame, roles, _ = synth_panel(rng, n_players=300, k=2)
    eb = JointEB(["c_0", "c_1"], SHRINK).fit(frame, roles)
    ids = ["GHOST"]
    res = eb.overall(ids, ["C"], np.zeros((1, 2)), np.zeros(1), np.zeros(1),
                     m_r={r: np.zeros(2) for r in ROLES})
    assert res["overall"][0] == pytest.approx(float(eb.mu["C"].sum()))
    assert res["var"][0] == pytest.approx(float(eb.T["C"].sum()))
    assert res["w"][0] == 0.0
    assert np.isfinite(res["overall"]).all()


def test_zero_games_with_positive_toi_does_not_divide_by_zero():
    rng = np.random.default_rng(17)
    frame, roles, _ = synth_panel(rng, n_players=300, k=2)
    eb = JointEB(["c_0", "c_1"], SHRINK).fit(frame, roles)
    res = eb.overall(["X"], ["C"], np.array([[0.1, 0.1]]), np.array([600.0]), np.array([0.0]))
    assert np.isfinite(res["overall"]).all() and np.isfinite(res["var"]).all()


# ---------------------------------------------------------------------------------------
# Согласованность и численная защита

def test_overall_equals_sum_of_centred_components():
    """Overall ровно равен сумме центрированных компонентов в совместной ветке."""
    rng = np.random.default_rng(21)
    frame, roles, _ = synth_panel(rng, n_players=400, k=3, T=np.eye(3) * 0.05,
                                  Sig_s=np.eye(3) * 4.0, Sig_o=np.eye(3) * 0.02)
    cols = ["c_0", "c_1", "c_2"]
    eb = JointEB(cols, SHRINK).fit(frame, roles)
    g = frame.groupby("player_id").agg(**{c: (c, "sum") for c in cols},
                                       toi_s=("toi_s", "sum"), games=("match_id", "nunique"))
    ids = list(g.index)
    res = eb.overall(ids, [roles[i] for i in ids], g[cols].to_numpy(),
                     g.toi_s.to_numpy(dtype=float), g.games.to_numpy(dtype=float))
    assert set(res["method"]) == {"joint"}
    assert np.allclose(res["profile"].sum(axis=1), res["overall"], atol=1e-12)


def test_posterior_variance_below_prior_and_weight_grows_with_toi():
    rng = np.random.default_rng(23)
    frame, roles, _ = synth_panel(rng, n_players=500, k=2)
    eb = JointEB(["c_0", "c_1"], SHRINK).fit(frame, roles)
    one_t1 = float(eb.T["C"].sum())
    res = eb.overall(["A", "B"], ["C", "C"], np.array([[0.05, 0.05], [0.5, 0.5]]),
                     np.array([600.0, 60000.0]), np.array([1.0, 60.0]),
                     m_r={r: np.zeros(2) for r in ROLES})
    assert (res["var"] <= one_t1 + 1e-12).all()
    assert res["w"][1] > res["w"][0]
    assert ((res["w"] >= 0) & (res["w"] <= 1)).all()


def test_psd_project_raises_floor_and_keeps_symmetry():
    A = np.array([[1.0, 2.0], [2.0, 1.0]])  # собственные значения 3 и -1
    P, info = psd_project(A, 1e-10, 1e-6)
    assert np.allclose(P, P.T)
    assert np.linalg.eigvalsh(P).min() > 0
    assert info["n_eig_raised"] == 1
    assert info["eig_min_raw"] == pytest.approx(-1.0)


def test_psd_project_zero_trace_uses_absolute_floor():
    P, info = psd_project(np.zeros((3, 3)), 1e-10, 1e-6)
    assert info["floor"] == pytest.approx(1e-10)
    assert np.linalg.eigvalsh(P).min() == pytest.approx(1e-10)


def test_cholesky_jitter_is_logged_per_player():
    """Неразложимая система уходит в ветку с нарастающим ε, и ε пишется в лог для игрока."""
    rng = np.random.default_rng(29)
    frame, roles, _ = synth_panel(rng, n_players=200, k=2)
    eb = JointEB(["c_0", "c_1"], SHRINK).fit(frame, roles)
    log: list = []
    nan_system = np.array([[np.nan, 0.0], [0.0, 1.0]])
    with pytest.raises(np.linalg.LinAlgError):
        eb._solve(nan_system, np.zeros((2, 2)), "BAD", log)
    bad = np.zeros((2, 2))
    sol, eps = eb._solve(bad, np.eye(2), "ZERO", log)
    assert eps >= SHRINK["jitter_start"]
    assert log and log[0]["player_id"] == "ZERO"
    assert np.isfinite(sol).all()


def test_prior_mu_uses_only_given_subsample():
    """μ_r — обучаемая величина: она берётся ровно из переданной части, а не из всей панели."""
    rng = np.random.default_rng(31)
    frame, roles, _ = synth_panel(rng, n_players=200, k=2)
    eb = JointEB(["c_0", "c_1"], SHRINK).fit(frame, roles)
    half = frame[frame.player_id.isin(sorted(frame.player_id.unique())[:50])]
    mu_sub = eb.prior_mu(half, roles)
    assert not np.allclose(mu_sub["C"], eb.mu["C"])
    eb2 = eb.with_mu(mu_sub)
    assert np.allclose(eb2.mu["C"], mu_sub["C"])
    assert np.allclose(eb.mu["C"], eb.diag["mu"]["C"]), "with_mu не должен менять исходную модель"
    assert np.allclose(eb2.T["C"], eb.T["C"]), "with_mu не меняет T_r"
