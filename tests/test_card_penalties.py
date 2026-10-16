"""Карточка, удаления (METHODOLOGY 11.4): ожидания — ручной расчёт и синтетика."""
import numpy as np
import pandas as pd
import pytest

from datamatch import card_penalties as cp


def test_penalty_counts_actor_taken_other_drawn_shootout_excluded():
    ev = pd.DataFrame({"match_id": ["M1", "M1", "M1", "M2"], "period": [1, 2, 0, 3],
                       "actor_id": ["a", "b", "a", None], "other_id": ["b", None, "b", "a"]})
    c = cp.penalty_counts(ev).set_index(["match_id", "player_id"])
    assert c.loc[("M1", "a"), "taken"] == 1 and c.loc[("M1", "a"), "drawn"] == 0
    assert c.loc[("M1", "b"), "taken"] == 1 and c.loc[("M1", "b"), "drawn"] == 1
    assert c.loc[("M2", "a"), "drawn"] == 1 and c.loc[("M2", "a"), "taken"] == 0
    assert len(c) == 3


def test_gamma_moments_hand_case():
    # T = [1, 1, 2] ч, n = [0, 2, 4]: m = 6/4 = 1,5; Σ T(x−m)² = 1·2,25 + 1·0,25 + 2·0,25 = 3,0
    # τ² = (3,0 − 1,5·2) / (4 − 6/4) = 0 → сигнала нет
    mo = cp.gamma_moments(np.array([0, 2, 4]), np.array([1.0, 1.0, 2.0]))
    assert mo["m"] == pytest.approx(1.5)
    assert mo["tau2"] == pytest.approx(0.0)
    assert mo["no_signal"]


def test_gamma_moments_with_signal_hand_case():
    # T = [1, 1, 1, 1], n = [0, 0, 4, 4]: m = 2; Σ T(x−m)² = 16; τ² = (16 − 2·3)/(4 − 1) = 10/3
    mo = cp.gamma_moments(np.array([0, 0, 4, 4]), np.ones(4))
    assert mo["tau2"] == pytest.approx(10 / 3)
    assert mo["alpha"] == pytest.approx(4 / (10 / 3))
    assert mo["beta"] == pytest.approx(2 / (10 / 3))
    assert not mo["no_signal"]


def test_gamma_posterior_mean_and_weight():
    mean, w = cp.gamma_posterior(np.array([3, 0]), np.array([2.0, 0.0]), alpha=1.0, beta=2.0)
    assert mean.tolist() == pytest.approx([(3 + 1) / (2 + 2), 0.5])     # без времени — prior m = α/β
    assert w.tolist() == pytest.approx([0.5, 0.0])


def test_gamma_posterior_no_signal_returns_pooled_rate():
    mean, w = cp.gamma_posterior(np.array([5, 0]), np.array([1.0, 3.0]), alpha=None, beta=None, m=1.2)
    assert mean.tolist() == pytest.approx([1.2, 1.2]) and w.tolist() == [0.0, 0.0]


def test_power_play_value_per_infraction_hand_case():
    nodes = pd.DataFrame({"match_id": ["M1"] * 4 + ["M2"] * 2,
                          "mp": ["PP", "SH", "EV5", "PP", "PP", "ENA"],
                          "r": [0.2, 0.05, 0.3, 0.1, 0.4, 0.5], "is_goal": [0, 0, 1, 1, 0, 1]})
    inf = pd.Series({"M1": 2, "M2": 1})
    per = cp.pp_net_by_match(nodes, inf)
    assert per.loc["M1", "net_xg"] == pytest.approx(0.25)                # 0,2 + 0,1 − 0,05
    assert per.loc["M1", "net_goals"] == pytest.approx(1.0)
    assert cp.ratio_of_sums(per, "net_xg") == pytest.approx((0.25 + 0.4) / 3)


def test_ratio_bootstrap_constant_ratio_has_no_spread():
    per = pd.DataFrame({"net_xg": [0.2, 0.4, 0.6], "n_inf": [1, 2, 3]}, index=["a", "b", "c"])
    draws = cp.bootstrap_ratio(per, "net_xg", B=50, seed=0)
    assert np.allclose(draws, 0.2)


def test_gamma_mle_recovers_known_prior_with_unequal_exposure():
    # Arrange: λ ~ Gamma(α=4, β=4) → m = 1, τ² = 0,25; экспозиция 0,2–6 ч; n ~ Poisson(λT).
    rng = np.random.default_rng(7)
    T = rng.uniform(0.2, 6.0, 6000)
    lam = rng.gamma(4.0, 1 / 4.0, len(T))
    n = rng.poisson(lam * T)

    # Act
    fit = cp.gamma_mle(n, T)

    # Assert
    assert not fit["no_signal"]
    assert fit["m"] == pytest.approx(1.0, abs=0.04)
    assert fit["tau2"] == pytest.approx(0.25, abs=0.05)
    assert fit["alpha"] == pytest.approx(fit["m"] ** 2 / fit["tau2"])
    assert fit["beta"] == pytest.approx(fit["m"] / fit["tau2"])


def test_gamma_mle_constant_rate_gives_near_zero_tau2():
    rng = np.random.default_rng(8)
    T = rng.uniform(0.2, 6.0, 6000)
    n = rng.poisson(1.0 * T)
    fit = cp.gamma_mle(n, T)
    assert fit["m"] == pytest.approx(1.0, abs=0.04)
    assert fit["no_signal"] or fit["tau2"] < 0.02


def test_gamma_mle_all_zero_counts_is_no_signal():
    fit = cp.gamma_mle(np.zeros(50), np.full(50, 2.0))
    assert fit["no_signal"] and fit["alpha"] is None and fit["beta"] is None


def test_half_r_ci_hand_case():
    # одно амплуа: a = (1,2,3,4), b = (1,3,2,4) → r = 4/5 = 0,8 (ручной расчёт)
    res = cp.half_r_ci(np.array([1., 2., 3., 4.]), np.array([1., 3., 2., 4.]), np.array(["C"] * 4), B=200, seed=1)
    assert res["r"] == pytest.approx(0.8)
    assert res["lo"] <= res["r"] <= res["hi"]


def test_half_r_ci_centres_within_role():
    # Разные уровни амплуа не должны создавать корреляцию: внутри амплуа половины независимы по знаку.
    a = np.array([10., 11., 0., 1.])
    b = np.array([11., 10., 1., 0.])
    res = cp.half_r_ci(a, b, np.array(["C", "C", "D", "D"]), B=100, seed=2)
    assert res["r"] == pytest.approx(-1.0)


def test_rescale_prior_for_missing_share():
    mo = {"m": 1.0, "tau2": 0.25, "alpha": 4.0, "beta": 4.0, "no_signal": False}
    f = 0.9 / 0.95
    out = cp.rescale_prior(mo, f)
    assert out["alpha"] == pytest.approx(4.0)
    assert out["beta"] == pytest.approx(4.0 / f)
    assert out["m"] == pytest.approx(f)
    assert out["tau2"] == pytest.approx(0.25 * f * f)
    assert mo["beta"] == 4.0                       # вход не меняется


def test_rescale_prior_no_signal_keeps_none():
    mo = {"m": 1.0, "tau2": 0.0, "alpha": None, "beta": None, "no_signal": True}
    out = cp.rescale_prior(mo, 0.5)
    assert out["alpha"] is None and out["m"] == pytest.approx(0.5)
