"""L7: общий линейный прогнозный слой ŷ = a_r + b·z и ошибка прогноза (RMSE, MAE, skill score).

Ручные расчёты: WLS-подгонка на точных данных восстанавливает a_r и b; «только амплуа» —
взвешенное среднее цели амплуа; SS = 1 − MSE/MSE(только амплуа). На синтетике с талантом
O1 обязан иметь SS > 0 (интервал выше нуля), на синтетике без таланта — SS ≈ 0 (интервал ∋ 0).
Перекрёстная подгонка: прогноз игрока не зависит от его собственной цели.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datamatch import l7_forecast as fc
from datamatch import rating_validation as rv

from .test_rating_validation import SHRINK, synth_pm

N_BOOT_TEST = 200
N_TEAMS = 6


def with_teams(pm: pd.DataFrame, seed: int = 3) -> pd.DataFrame:
    """Каждому игроку — постоянная команда (для базовой строки «только команда»)."""
    rng = np.random.default_rng(seed)
    ids = sorted(pm.player_id.unique())
    team = pd.Series(rng.integers(0, N_TEAMS, len(ids)), index=ids).map(lambda k: f"t{k}")
    return pm.assign(team_id=pm.player_id.map(team).to_numpy())


def toy_frame() -> pd.DataFrame:
    """y = a_r + b·z точно: a = {C: 1, W: 0, D: −1}, b = 0,5."""
    z = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 3.0, -1.0, 0.0, 2.0])
    role = np.array(["C"] * 3 + ["W"] * 3 + ["D"] * 3)
    a = pd.Series({"C": 1.0, "W": 0.0, "D": -1.0})
    y = a.reindex(role).to_numpy() + 0.5 * z
    return pd.DataFrame({"role": role, "target": y, "toi_s": np.arange(1, 10) * 600.0, "r::X": z},
                        index=[f"p{i}" for i in range(9)])


def test_fit_layer_recovers_exact_coefficients():
    f = toy_frame()
    coef = fc.fit_layer(f, "X")
    assert coef["b"] == pytest.approx(0.5)
    assert coef["a"] == pytest.approx({"C": 1.0, "W": 0.0, "D": -1.0})


def test_role_only_is_weighted_role_mean():
    f = toy_frame()
    coef = fc.fit_layer(f, fc.ROLE_ONLY)
    c = f[f.role == "C"]
    assert coef["b"] == 0.0
    assert coef["a"]["C"] == pytest.approx(np.average(c.target, weights=c.toi_s))


def test_error_metrics_hand_computed():
    pred = pd.DataFrame({"role": ["C", "C", "D", "D"], "target": [1.0, 3.0, 0.0, 2.0],
                         "toi_s": [1.0, 3.0, 1.0, 1.0], "p::role_only": [2.0, 2.0, 1.0, 1.0],
                         "p::X": [1.0, 2.5, 0.5, 1.5]})
    m = fc.error_metrics(pred, ["role_only", "X"])
    mse_base = (1 * 1 + 3 * 1 + 1 * 1 + 1 * 1) / 6.0
    mse_x = (0 + 3 * 0.25 + 0.25 + 0.25) / 6.0
    assert m["X"]["pooled"]["mse"] == pytest.approx(mse_x)
    assert m["X"]["pooled"]["rmse"] == pytest.approx(np.sqrt(mse_x))
    assert m["X"]["pooled"]["mae"] == pytest.approx((0 + 3 * 0.5 + 0.5 + 0.5) / 6.0)
    assert m["X"]["pooled"]["ss"] == pytest.approx(1 - mse_x / mse_base)
    assert m["role_only"]["pooled"]["ss"] == pytest.approx(0.0)
    assert m["X"]["by_role"]["C"]["mse"] == pytest.approx((0 + 3 * 0.25) / 4.0)


def test_crossfit_prediction_ignores_own_target():
    f = toy_frame()
    folds = pd.Series(np.arange(9) % 3, index=f.index)
    p1 = fc.crossfit_predict({"A->B": f}, ["X"], folds)["A->B"]
    f2 = f.copy()
    f2.loc["p0", "target"] += 100.0
    p2 = fc.crossfit_predict({"A->B": f2}, ["X"], folds)["A->B"]
    assert p1.loc["p0", "p::X"] == pytest.approx(p2.loc["p0", "p::X"])


def test_team_rating_is_team_onice_rate():
    pm = pd.DataFrame({"match_id": ["m1", "m1", "m2", "m2"], "player_id": ["a", "b", "a", "c"],
                       "team_id": ["t", "t", "t", "t"], "xgf_5v5": [1.0, 0.5, 2.0, 1.0],
                       "xga_5v5": [0.0, 0.5, 1.0, 0.0], "toi_rate_5v5_s": [1800.0, 1800.0, 3600.0, 3600.0]})
    z = fc.team_rating(pm)
    # m1: Σ(xgf − xga) = 1 за Σ toi = 3600 с; m2: 2 за 7200 с
    assert z["b"] == pytest.approx(1.0)
    assert z["c"] == pytest.approx(1.0)
    assert z["a"] == pytest.approx((1.0 + 2.0) / (3600.0 + 7200.0) * 3600.0)


@pytest.fixture(scope="module")
def talent_res():
    pm = with_teams(synth_pm(talent_sd=0.45))
    fns = rv.our_rating_fns(SHRINK) | rv.baseline_rating_fns(pm) | {fc.TEAM_ONLY: fc.team_rating}
    return fc.train_forecast(pm, fns, n_boot=N_BOOT_TEST)


@pytest.fixture(scope="module")
def noise_res():
    pm = with_teams(synth_pm(talent_sd=0.0, seed=8))
    fns = rv.our_rating_fns(SHRINK) | rv.baseline_rating_fns(pm)
    return fc.train_forecast(pm, fns, n_boot=N_BOOT_TEST)


def test_talent_o1_beats_role_only(talent_res):
    for d in ("A->B", "B->A"):
        ss = talent_res["directions"][d]["metrics"]["O1"]["pooled"]
        assert ss["ss"] > 0.0
        assert ss["ss_ci"][0] > 0.0
    assert talent_res["mean_ss"]["O1"] > talent_res["mean_ss"]["points_60_raw"]


def test_noise_ss_interval_covers_zero(noise_res):
    ss = noise_res["directions"]["A->B"]["metrics"]["O1"]["pooled"]
    assert ss["ss_ci"][0] <= 0.0 <= ss["ss_ci"][1] or abs(ss["ss"]) < 0.02


def test_paired_difference_reported(talent_res):
    res = talent_res["directions"]["A->B"]
    d = res["o1_minus"]["points_60_raw"]
    assert d["ss_diff"] == pytest.approx(res["metrics"]["O1"]["pooled"]["ss"]
                                         - res["metrics"]["points_60_raw"]["pooled"]["ss"])
    assert d["ss_diff_ci"][0] > 0.0


def test_holdout_uses_given_coefficients():
    f = toy_frame()
    coefs = {"X": {"a": {"C": 1.0, "W": 0.0, "D": -1.0}, "b": 0.5},
             fc.ROLE_ONLY: {"a": {"C": 0.0, "W": 0.0, "D": 0.0}, "b": 0.0}}
    res = fc.holdout_forecast(f, coefs, n_boot=50)
    assert res["metrics"]["X"]["pooled"]["mse"] == pytest.approx(0.0, abs=1e-12)
    assert res["metrics"]["X"]["pooled"]["ss"] == pytest.approx(1.0)
    assert res["holdout_slope"]["X"] == pytest.approx(0.5)


def test_constant_rating_marked_degenerate():
    f = toy_frame().assign(**{"r::K": 1.0})
    assert fc.within_role_sd({"A->B": f}, "K") == pytest.approx(0.0)
    assert fc.within_role_sd({"A->B": f}, "X") > fc.DEGENERATE_Z_SD


def test_constant_within_one_role_is_degenerate():
    f = toy_frame()
    z = f["r::X"].where(f.role != "W", 5.0)
    assert fc.within_role_sd({"A->B": f.assign(**{"r::X": z})}, "X") == pytest.approx(0.0)
