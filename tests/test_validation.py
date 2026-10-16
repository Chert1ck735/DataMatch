"""Проверочный контур: split-half надёжность, предсказательная сила, парное сравнение, хронология.

Синтетическая популяция с известным талантом: при наличии сигнала надёжность явно > 0,
при чистом шуме — около нуля (доверительный интервал накрывает 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datamatch import baselines as bl
from datamatch import validation as va

ROLES3 = ["C", "W", "D"]


def _population(signal_sd: float, n_players: int = 240, n_matches: int = 120, per_player: int = 40,
                seed: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    """Таблица «игрок × матч» в формате MF_COLS: num ~ Пуассон(θ_i · h), h — часы в матче."""
    rng = np.random.default_rng(seed)
    theta = np.exp(rng.normal(np.log(3.0), signal_sd, n_players)) if signal_sd > 0 \
        else np.full(n_players, 3.0)
    rows = []
    for i in range(n_players):
        for m in rng.choice(n_matches, per_player, replace=False):
            h = rng.uniform(0.25, 0.35)
            rows.append((f"p{i}", f"m{m:03d}", ROLES3[i % 3], float(rng.poisson(theta[i] * h)), h * 3600))
    mf = pd.DataFrame(rows, columns=bl.MF_COLS)
    return mf, pd.Series(theta, index=[f"p{i}" for i in range(n_players)])


def test_spearman_brown_and_weighted_pearson_hand_values():
    assert va.spearman_brown(0.5) == pytest.approx(2 / 3)
    assert va.spearman_brown(0.0) == 0.0
    x, y = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0])
    assert va.weighted_pearson(x, y, np.ones(3)) == pytest.approx(np.corrcoef(x, y)[0, 1])
    # вес только у двух точек — корреляция двух точек равна 1
    assert va.weighted_pearson(x, y, np.array([1.0, 1.0, 0.0])) == pytest.approx(1.0)


def test_assign_halves_is_deterministic_and_balanced():
    ids = [f"m{i}" for i in range(11)]
    a = va.assign_halves(ids, seed=5)
    b = va.assign_halves(list(reversed(ids)), seed=5)
    assert a.equals(b)
    assert set(a.unique()) == {0, 1}
    assert abs(int(a.sum()) - (11 - int(a.sum()))) <= 1


def test_split_half_detects_true_talent():
    mf, _ = _population(signal_sd=0.5)
    rep = va.split_half_reliability(mf, n_boot=200, min_toi_half_s=600.0)
    for r in ROLES3:
        g = rep["groups"][r]
        assert g["n"] == 80
        assert g["pearson_sb"] > 0.6
        assert g["pearson_sb_ci"][0] > 0.3
    assert rep["pooled"]["spearman_sb"] > 0.6


def test_split_half_match_bootstrap_ci_is_recentred_on_estimate():
    # повторная выборка матчей добавляет шум в ставки половин и занижает r* — интервал
    # сдвигается на (r̂ − медиана r*), поэтому точечная оценка лежит внутри интервала
    mf, _ = _population(signal_sd=0.3, n_players=120, n_matches=60, per_player=12)
    rep = va.split_half_reliability(mf, n_boot=200, min_toi_half_s=600.0)
    p = rep["pooled"]
    assert p["pearson_sb_ci"][0] <= p["pearson_sb"] <= p["pearson_sb_ci"][1]
    assert p["pearson_boot_shift"] > 0.0
    rp = va.split_half_reliability(mf, n_boot=200, min_toi_half_s=600.0, boot="players")
    assert rp["bootstrap"] == "players"
    assert rp["pooled"]["pearson_sb"] == pytest.approx(p["pearson_sb"])
    assert rp["pooled"]["pearson_sb_ci"][0] <= rp["pooled"]["pearson_sb"] <= rp["pooled"]["pearson_sb_ci"][1]
    with pytest.raises(ValueError, match="boot"):
        va.split_half_reliability(mf, boot="teams")


def test_split_half_pure_noise_is_near_zero():
    mf, _ = _population(signal_sd=0.0)
    rep = va.split_half_reliability(mf, n_boot=200, min_toi_half_s=600.0)
    p = rep["pooled"]
    assert abs(p["pearson"]) < 0.2
    # шкала надёжности [0, 1]: у шума нижняя граница отсекается в 0 (ревью 06.10.2026)
    assert p["pearson_sb_ci"][0] == 0.0 < p["pearson_sb_ci"][1] < 0.5


def test_split_half_does_not_mutate_and_validates():
    mf, _ = _population(signal_sd=0.5, n_players=30, n_matches=40, per_player=20)
    before = mf.copy()
    va.split_half_reliability(mf, n_boot=20, min_toi_half_s=0.0)
    pd.testing.assert_frame_equal(mf, before)
    with pytest.raises(KeyError):
        va.split_half_reliability(mf.drop(columns="toi_s"))
    with pytest.raises(ValueError, match="by"):
        va.split_half_reliability(mf, by="team")


def _ratings_and_target(seed: int = 11):
    mf, theta = _population(signal_sd=0.5, seed=seed)
    target = va.target_rates(mf)
    rng = np.random.default_rng(seed)
    good = pd.Series(np.log(theta.to_numpy()) + rng.normal(0, 0.1, len(theta)), index=theta.index)
    noise = pd.Series(rng.normal(0, 1, len(theta)), index=theta.index)
    return target, good, noise


def test_predictive_validity_sign_and_ci():
    target, good, noise = _ratings_and_target()
    rg = va.predictive_validity(good, target, n_boot=300, min_toi_s=600.0)
    assert rg["pooled"]["spearman"] > 0.5 and rg["pooled"]["spearman_ci"][0] > 0.3
    assert rg["pooled"]["wpearson"] > 0.4
    assert rg["by_role"]["C"]["n"] == 80
    neg = va.predictive_validity(-good, target, n_boot=300, min_toi_s=600.0)
    assert neg["pooled"]["spearman"] == pytest.approx(-rg["pooled"]["spearman"])
    rn = va.predictive_validity(noise, target, n_boot=300, min_toi_s=600.0)
    lo, hi = rn["pooled"]["spearman_ci"]
    assert lo < 0.0 < hi


def test_compare_ratings_paired_difference():
    target, good, noise = _ratings_and_target()
    cmp_ = va.compare_ratings({"good": good, "noise": noise}, target, baseline="noise",
                              n_boot=300, min_toi_s=600.0)
    d = cmp_["vs_baseline"]["good"]["pooled"]
    assert d["spearman_diff"] > 0.3 and d["spearman_diff_ci"][0] > 0.0
    assert cmp_["n_players"] == 240
    with pytest.raises(KeyError, match="baseline"):
        va.compare_ratings({"good": good}, target, baseline="missing")


def test_half_split_predictive_uses_disjoint_halves():
    mf, _ = _population(signal_sd=0.5, n_players=60, n_matches=60, per_player=30)
    seen: list[set] = []

    def rating(pm_half: pd.DataFrame) -> pd.Series:
        seen.append(set(pm_half.match_id))
        return bl.player_rates(pm_half).set_index("player_id").rate_60

    rep = va.half_split_predictive(mf, {"raw": rating}, mf, n_boot=50, min_toi_s=600.0)
    assert set(rep["directions"]) == {"A->B", "B->A"}
    assert not (seen[0] & seen[1])
    assert rep["mean"]["raw"]["pooled_spearman"] > 0.3


def test_temporal_check_rates_on_early_and_targets_late():
    mf, _ = _population(signal_sd=0.5, n_players=30, n_matches=40, per_player=40)
    games = pd.DataFrame({"match_id": sorted(mf.match_id.unique())})
    games["game_seq"] = np.arange(len(games))[::-1] + 1           # порядок не совпадает с id
    early = set(games.match_id[games.game_seq <= 20])
    seen: list[set] = []

    def rating(pm_half: pd.DataFrame) -> pd.Series:
        seen.append(set(pm_half.match_id))
        return bl.player_rates(pm_half).set_index("player_id").rate_60

    rep = va.temporal_check(mf, mf, games, {"raw": rating}, n_boot=50, min_toi_s=600.0)
    assert rep["diagnostic_only"] is True
    assert seen[0] == early
    assert rep["n_early_matches"] == 20 and rep["n_late_matches"] == 20
    with pytest.raises(KeyError, match="game_seq"):
        va.temporal_check(mf, mf, games.drop(columns="game_seq"), {"raw": rating})
