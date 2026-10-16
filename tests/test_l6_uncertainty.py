"""L6: формулы интервалов, надёжность и бутстрэп моделей ценности.

Проверяется на синтетике с известной дисперсией: покрытие интервала скрытого эффекта близко к
номиналу, Var_m по репликам восстанавливает заложенный разброс, пороги надёжности совпадают с
границами из конфигурации, нулевой сигнал не ломает расчёт.
"""
import numpy as np
import pandas as pd
import pytest

from datamatch import l6_uncertainty as U


# --------------------------------------------------------------------------------------
# Интервал скрытого эффекта
# --------------------------------------------------------------------------------------
def test_normal_interval_is_rating_plus_minus_196_sqrt_v():
    lo, hi = U.normal_interval([1.0, -2.0], [0.25, 0.0])
    assert lo == pytest.approx([1.0 - 1.96 * 0.5, -2.0])
    assert hi == pytest.approx([1.0 + 1.96 * 0.5, -2.0])


def test_total_variance_adds_model_spread_and_prefers_bootstrap_mean():
    v = U.total_variance([0.04, 0.09], var_model=[0.01, 0.0])
    assert v == pytest.approx([0.05, 0.09])
    # E_m[v] из реплик заменяет точечную апостериорную дисперсию там, где она есть
    v2 = U.total_variance([0.04, 0.09], var_model=[0.01, 0.01], var_post_boot=[0.06, np.nan])
    assert v2 == pytest.approx([0.07, 0.10])


def test_total_variance_without_bootstrap_equals_posterior_variance():
    """Невыполненный бутстрэп не добавляет и не вычитает ширину (Var_m неизвестна, не ноль)."""
    vp = [0.01, 0.02, 0.0]
    assert U.total_variance(vp) == pytest.approx(vp)
    # Неизвестная Var_m при выполненном бутстрэпе — ошибка, а не молчаливый ноль (аудит L6 №3)
    with pytest.raises(ValueError):
        U.total_variance(vp, var_model=[np.nan] * 3)


def test_negative_variance_is_clipped_not_propagated():
    lo, hi = U.normal_interval([0.5], [-1e-9])
    assert lo == pytest.approx([0.5]) and hi == pytest.approx([0.5])


@pytest.mark.parametrize("w,expected", [
    (0.0, "LOW"), (0.39999, "LOW"), (0.4, "MEDIUM"), (0.6999, "MEDIUM"), (0.7, "HIGH"), (1.0, "HIGH"),
])
def test_reliability_thresholds_match_config_boundaries(w, expected):
    assert U.reliability_labels([w], high=0.7, medium=0.4)[0] == expected


def test_reliability_nan_is_low_and_bad_thresholds_rejected():
    assert U.reliability_labels([np.nan])[0] == "LOW"
    with pytest.raises(ValueError):
        U.reliability_labels([0.5], high=0.3, medium=0.6)


def test_mc_bound_error_shrinks_with_B():
    e30 = U.mc_bound_error([0.01], [0.05], 30)[0]
    e200 = U.mc_bound_error([0.01], [0.05], 200)[0]
    assert e30 > e200 > 0
    assert np.isnan(U.mc_bound_error([0.01], [0.05], 1)[0])


# --------------------------------------------------------------------------------------
# Покрытие на синтетике с известной дисперсией
# --------------------------------------------------------------------------------------
def test_latent_interval_coverage_close_to_nominal():
    rng = np.random.default_rng(7)
    toi = rng.uniform(20, 400, 300)
    games = np.maximum(np.round(toi / 15), 1)
    res = U.simulate_latent_coverage(tau2=0.04, sigma2=1.5, omega2=0.02, toi_min=toi, games=games,
                                     n_sim=60, seed=11)
    assert res["nominal"] == pytest.approx(0.95, abs=0.002)
    assert res["coverage"] == pytest.approx(0.95, abs=0.01)


def test_posterior_coverage_from_published_w_and_var():
    """w и var_post однозначно задают τ² и s, поэтому калибровку можно проверить по таблице."""
    rng = np.random.default_rng(3)
    w = rng.uniform(0.05, 0.9, 400)
    tau2 = 0.04
    var_post = tau2 * (1 - w)
    res = U.simulate_posterior_coverage(w, var_post, n_sim=60, seed=5)
    assert res["coverage"] == pytest.approx(0.95, abs=0.01)
    assert res["n_players"] == 400


def test_zero_signal_does_not_break_coverage_or_intervals():
    """τ² ≈ 0: стягивание к среднему, нулевая ширина, конечные границы, LOW."""
    res = U.simulate_latent_coverage(tau2=0.0, sigma2=1.0, omega2=0.0, toi_min=[100.0, 200.0],
                                     games=[5, 10], n_sim=5, seed=1)
    assert res["mean_w"] == 0.0 and res["coverage"] == 1.0
    lo, hi = U.normal_interval([0.0, 0.0], [0.0, 0.0])
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    assert list(U.reliability_labels([0.0, 0.0])) == ["LOW", "LOW"]
    with pytest.raises(ValueError):
        U.simulate_posterior_coverage([0.0], [0.0])


def test_empirical_coverage_counts_only_finite_rows():
    c = U.empirical_coverage([0.0, 1.0, np.nan], [-1, 2, 0], [1, 3, 1])
    assert c == {"n": 2, "n_hit": 1, "coverage": 0.5}


# --------------------------------------------------------------------------------------
# Предсказательный интервал будущего (в файл не идёт)
# --------------------------------------------------------------------------------------
def test_predictive_variance_matches_formula_and_grows_with_drift():
    v = U.predictive_variance([0.01], sigma2=1.5, omega2=0.02, toi_future_min=300.0,
                              games_future=20.0, delta2=0.05)
    assert v == pytest.approx([0.01 + 1.5 / 300 + 0.02 / 20 + 0.05])
    grid = U.predictive_interval_grid([0.0], [0.01], 1.5, 0.02, 300.0, 20.0, (0.0, 0.1))
    assert grid[1]["mean_halfwidth"] > grid[0]["mean_halfwidth"]


def test_predictive_interval_rejects_nonpositive_exposure_and_negative_variance():
    with pytest.raises(ValueError):
        U.predictive_variance([0.01], 1.0, 0.0, 0.0, 10.0)
    with pytest.raises(ValueError):
        U.predictive_variance([0.01], -1.0, 0.0, 10.0, 10.0)


def test_predictive_interval_is_wider_than_latent_interval():
    lo_l, hi_l = U.normal_interval([0.0], [0.01])
    p = U.predictive_interval([0.0], [0.01], 1.5, 0.02, 300.0, 20.0, 0.0)
    assert p["hi"][0] - p["lo"][0] > hi_l[0] - lo_l[0]


# --------------------------------------------------------------------------------------
# Бутстрэп моделей ценности
# --------------------------------------------------------------------------------------
MATCHES = [f"M{i:03d}" for i in range(40)]


def _replicate_factory(scale, players=("P1", "P2", "P3")):
    """Реплика с заложенным разбросом Overall: Var_m известна по построению."""
    def replicate(matches, index, seed):
        rng = np.random.default_rng(seed)
        assert len(matches) == len(MATCHES)           # выборка с возвращением того же размера
        base = np.arange(len(players), dtype=float) / 10.0
        return pd.Series(base + rng.normal(0.0, scale, len(players)), index=list(players))
    return replicate


def test_draws_are_deterministic_resamples_of_train_matches():
    a = U.bootstrap_draws(MATCHES, 5, seed=42)
    b = U.bootstrap_draws(MATCHES, 5, seed=42)
    assert [d.seed for d in a] == [d.seed for d in b]
    assert all(np.array_equal(x.matches, y.matches) for x, y in zip(a, b))
    assert all(len(d.matches) == len(MATCHES) for d in a)
    assert any(len(set(d.matches)) < len(MATCHES) for d in a)      # действительно с возвращением
    assert [d.seed for d in U.bootstrap_draws(MATCHES, 5, seed=43)] != [d.seed for d in a]
    with pytest.raises(ValueError):
        U.bootstrap_draws([], 5, seed=1)


def test_model_bootstrap_recovers_known_variance():
    scale = 0.03
    res = U.run_model_bootstrap(_replicate_factory(scale), MATCHES, b=300, seed=1)
    assert res.status == "done" and res.b_done == 300 and res.available
    assert res.var_model.notna().all()
    assert res.var_model.to_numpy() == pytest.approx(scale ** 2, rel=0.25)
    assert res.to_meta()["percentile_intervals_used"] is False


def test_model_bootstrap_is_order_independent_under_parallelism():
    serial = U.run_model_bootstrap(_replicate_factory(0.02), MATCHES, b=16, seed=5, n_jobs=1)
    parallel = U.run_model_bootstrap(_replicate_factory(0.02), MATCHES, b=16, seed=5, n_jobs=4)
    assert serial.var_model.to_numpy() == pytest.approx(parallel.var_model.to_numpy())


def test_model_bootstrap_not_run_without_replicate_or_with_too_small_B():
    none = U.run_model_bootstrap(None, MATCHES, b=200, seed=1)
    assert none.status == "not_run" and not none.available and none.reason
    small = U.run_model_bootstrap(_replicate_factory(0.01), MATCHES, b=1, seed=1)
    assert small.status == "not_run" and "Var_m" in small.reason


def test_data_errors_above_threshold_fail_the_bootstrap():
    def flaky(matches, index, seed):
        if index % 2:
            raise ValueError("модель ценности не сошлась")
        return pd.Series({"P1": float(index)})
    res = U.run_model_bootstrap(flaky, MATCHES, b=10, seed=2)
    assert res.status == "failed" and not res.available and res.n_failed > 0
    assert res.failures and "не сошлась" in res.failures[0]
    assert res.to_meta()["max_failure_share"] == U.MAX_REPLICATE_FAILURE_SHARE


def test_all_replicates_failing_gives_failed():
    def broken(matches, index, seed):
        raise ValueError("нет xG-модели")
    res = U.run_model_bootstrap(broken, MATCHES, b=4, seed=2)
    assert res.status == "failed" and res.b_done == 0 and res.n_failed >= 1


def test_unexpected_replicate_error_propagates():
    def buggy(matches, index, seed):
        raise KeyError("c_shot")
    with pytest.raises(KeyError):
        U.run_model_bootstrap(buggy, MATCHES, b=4, seed=2)
    assert not issubclass(KeyError, U.EXPECTED_REPLICATE_ERRORS)
    assert not issubclass(TypeError, U.EXPECTED_REPLICATE_ERRORS)


def test_few_data_errors_are_replaced_by_extra_draws():
    def rare(matches, index, seed):
        if index == 1:
            raise ValueError("мало матчей в фолде")
        return pd.Series({"P1": float(index), "P2": 1.0})
    res = U.run_model_bootstrap(rare, MATCHES, b=40, seed=2)
    assert res.status == "done" and res.b_done == 40 and res.n_failed == 1 and res.n_attempted == 41


def test_nonfinite_overall_fails_the_bootstrap():
    def bad(matches, index, seed):
        return pd.Series({"P1": np.nan if index == 2 else 0.1 * index, "P2": 0.0})
    res = U.run_model_bootstrap(bad, MATCHES, b=6, seed=2)
    assert res.status == "failed" and "P1" in res.reason


def test_player_missing_in_a_replicate_has_unknown_variance():
    def partial_cover(matches, index, seed):
        ids = ["P1", "P2"] if index != 3 else ["P1"]
        return pd.Series({p: 0.1 * index for p in ids})
    res = U.run_model_bootstrap(partial_cover, MATCHES, b=6, seed=2)
    assert res.status == "done"
    assert np.isfinite(res.var_model["P1"]) and np.isnan(res.var_model["P2"])
    assert res.n_replicates["P1"] == 6 and res.n_replicates["P2"] == 5


def test_time_budget_stops_bootstrap_and_is_reported_as_partial():
    import time

    def slow(matches, index, seed):
        time.sleep(0.05)
        return pd.Series({"P1": float(index), "P2": -float(index)})
    res = U.run_model_bootstrap(slow, MATCHES, b=50, seed=3, time_budget_s=0.12)
    assert res.status == "partial" and 2 <= res.b_done < 50
    assert "бюджет" in res.reason


def test_replicate_may_return_frame_with_posterior_variance():
    def replicate(matches, index, seed):
        return pd.DataFrame({"player_id": ["P1", "P2"], "overall": [0.1 * index, 1.0],
                             "var_post": [0.02, 0.03]})
    res = U.run_model_bootstrap(replicate, MATCHES, b=6, seed=4)
    assert res.status == "done"
    assert res.var_post_mean.loc["P1"] == pytest.approx(0.02)
    assert res.var_model.loc["P2"] == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# Устойчивость пула (ревью F3) и время реплик (F5)
# --------------------------------------------------------------------------------------
class _FlakyPool:
    """Поддельный пул: в первом экземпляре задача index=1 «теряет процесс» (BrokenProcessPool)."""
    instances: list = []

    def __init__(self, width):
        self.width = width
        self.submitted = []
        _FlakyPool.instances.append(self)

    def submit(self, fn, draw):
        from concurrent.futures import Future
        from concurrent.futures.process import BrokenProcessPool
        self.submitted.append(draw.index)
        fut = Future()
        if len(_FlakyPool.instances) == 1 and draw.index == 1:
            fut.set_exception(BrokenProcessPool("процесс пула убит"))
        else:
            fut.set_result(fn(draw))
        return fut

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def test_broken_pool_is_recreated_and_only_unfinished_replicates_are_resubmitted(monkeypatch):
    # Arrange
    monkeypatch.setattr(_FlakyPool, "instances", [])

    # Act
    res = U.run_model_bootstrap(_replicate_factory(0.02), MATCHES, b=4, seed=2, n_jobs=2,
                                executor_factory=_FlakyPool)

    # Assert
    first, second = _FlakyPool.instances[:2]
    assert first.submitted[:2] == [0, 1]
    assert second.submitted[0] == 1 and 0 not in second.submitted        # готовая реплика 0 не пересчитана
    assert second.width < first.width                                    # после гибели процесса — уже
    assert res.status == "done" and res.b_done == 4 and res.pool_restarts == 1
    serial = U.run_model_bootstrap(_replicate_factory(0.02), MATCHES, b=4, seed=2)
    assert res.var_model.to_numpy() == pytest.approx(serial.var_model.to_numpy())


def test_bootstrap_logs_wall_time_per_replicate_and_reports_progress(caplog):
    import logging
    log = logging.getLogger("datamatch.test_l6_progress")
    caplog.set_level(logging.INFO, logger=log.name)

    res = U.run_model_bootstrap(_replicate_factory(0.02), MATCHES, b=4, seed=1, logger=log)

    assert len(res.replicate_seconds) == 4 and all(s >= 0 for s in res.replicate_seconds)
    prog = res.to_meta()["progress"]
    assert prog["done"] == 4 and prog["total"] == 4 and prog["eta_s"] == 0
    assert prog["mean_replicate_s"] == pytest.approx(float(np.mean(res.replicate_seconds)), abs=1e-3)
    text = caplog.text
    assert all(f"реплика {i}:" in text for i in range(4))
    assert "4/4" in text and "ETA" in text
