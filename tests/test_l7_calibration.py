"""L7: проверка меток надёжности HIGH/MEDIUM/LOW и покрытия интервалов.

Контракт (из модели L5, не из кода): при верных моментах для оценки θ̂ по половине A и ставки
x_B по независимой половине B выполняется E[(θ̂ − μ)(x_B − μ)] = E[(θ̂ − μ)²] = w·τ², поэтому
реализованная надёжность R = mean[(θ̂ − μ)(x_B − μ)] / τ² совпадает со средним w группы, а
наклон x_B на θ̂ равен 1. Предсказательный интервал θ̂ ± z·√(v + σ²/T_B + ω²/M_B) накрывает x_B
с частотой 95% (в train дрейфа нет). Двухэтапная симуляция (моменты по большой «train»-выборке,
интервалы для малой «test»-выборки) при верной модели даёт ≈ 95%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datamatch import l7_calibration as cal

from .test_rating_validation import COMP_MU, COMP_SHARE, COMP_SIGMA, ROLES, SHRINK

N_BOOT_TEST = 200


def pairs_toy() -> pd.DataFrame:
    return pd.DataFrame({
        "direction": ["A->B"] * 4, "player_id": ["a", "b", "c", "d"], "role": ["C", "C", "D", "D"],
        "theta": [1.0, -1.0, 0.5, 0.0], "mu": [0.0, 0.0, 0.0, 0.0], "tau2": [2.0, 2.0, 1.0, 1.0],
        "w": [0.8, 0.8, 0.5, 0.0], "var_post": [0.4, 0.4, 0.5, 1.0],
        "reliability": ["HIGH", "HIGH", "MEDIUM", "LOW"],
        "x_fut": [2.0, 0.0, 1.5, 3.0], "toi_fut_min": [60.0, 60.0, 30.0, 30.0], "games_fut": [2, 2, 1, 1],
        "sigma2": [1.0, 1.0, 1.0, 1.0], "omega2": [0.0, 0.0, 0.0, 0.0]})


def test_realized_reliability_hand_computed():
    out = cal.label_calibration(pairs_toy(), n_boot=20)
    hi = out["by_label"]["HIGH"]
    # (θ̂−μ)(x−μ)/τ²: a: 1·2/2 = 1; b: (−1)·0/2 = 0; равные веса → 0,5
    assert hi["realized_reliability"] == pytest.approx(0.5)
    assert hi["model_reliability"] == pytest.approx(0.5)      # (1 + 1)/2 / 2
    assert hi["mean_w"] == pytest.approx(0.8)
    assert hi["calibration_slope"] == pytest.approx(1.0)       # Σ(θ̂−μ)(x−μ) / Σ(θ̂−μ)² = 2/2
    assert hi["n"] == 2


def test_predictive_coverage_hand_computed():
    p = pairs_toy()
    out = cal.predictive_coverage(p, n_boot=20)
    # единицы L5/L6: T_f в минутах (`l6_uncertainty.predictive_variance`)
    sd = np.sqrt(p.var_post + p.sigma2 / p.toi_fut_min + p.omega2 / p.games_fut)
    hit = (np.abs(p.x_fut - p.theta) <= cal.Z95 * sd).to_numpy()
    assert out["pooled"]["coverage"] == pytest.approx(hit.mean())
    assert out["pooled"]["n"] == 4


def identifiable_pm(seed: int = 1, n_per_role: int = 150, n_matches: int = 60, omega: float = 0.6,
                    talent_sd: float = 0.45) -> pd.DataFrame:
    """Синтетика с идентифицируемыми σ² и ω²: у игроков разная доля сыгранных матчей и разное TOI.

    В `synth_pm` TOI на матч почти одинаково у всех, столбцы плана моментов 1/T и 1/M почти
    коллинеарны, и оценка шума L5 смещена (см. problems: P055) — для проверки контракта L7
    нужна выборка, где модель L5 оценима. Шум матча ω·h на компонент — ставка ±ω за матч.
    """
    rng = np.random.default_rng(seed)
    pid = [f"p{r}{i:03d}" for r in ROLES for i in range(n_per_role)]
    role = np.repeat(ROLES, n_per_role)
    n = len(pid)
    theta = rng.normal(0.0, talent_sd, n)
    plays = rng.random((n, n_matches)) < rng.uniform(0.25, 0.95, n)[:, None]
    pi, mi = np.nonzero(plays)
    toi = rng.uniform(300.0, 1300.0, n)[pi] * rng.uniform(0.6, 1.4, len(pi))
    h = toi / 3600.0
    cols = {c: (COMP_MU[c] + COMP_SHARE[c] * theta[pi]) * h + COMP_SIGMA * np.sqrt(h) * rng.normal(size=len(pi))
            + omega * h * rng.normal(size=len(pi)) for c in COMP_SHARE}
    return pd.DataFrame({"match_id": [f"m{m:03d}" for m in mi], "player_id": np.asarray(pid)[pi],
                         "role": role[pi], "toi_rate_s": toi, **cols})


@pytest.fixture(scope="module")
def talent_pairs():
    return cal.half_pairs(identifiable_pm(), SHRINK)


def test_half_pairs_use_disjoint_halves(talent_pairs):
    assert set(talent_pairs.direction) == {"A->B", "B->A"}
    assert talent_pairs.groupby("direction").player_id.nunique().min() > 100
    assert np.isfinite(talent_pairs[["theta", "x_fut", "var_post", "w", "tau2"]].to_numpy()).all()


def test_calibrated_model_realized_matches_model(talent_pairs):
    out = cal.label_calibration(talent_pairs, n_boot=N_BOOT_TEST)
    lo, hi = out["pooled"]["realized_reliability_ci"]
    assert lo <= out["pooled"]["model_reliability"] <= hi


# --- наклон калибровки: оракул строго, оценённые моменты — двусторонняя граница на большой выборке
# Истинные моменты `identifiable_pm` (единицы L5: T в минутах): T_r = 0,45²·s sᵀ (ранг 1),
# Σσ = 0,45²·60·I, Σω = 0,6²·I, μ_r = COMP_MU.
SLOPE_SEEDS = (1, 2, 3, 4, 5)
TRUE_T = 0.45 ** 2 * np.outer(list(COMP_SHARE.values()), list(COMP_SHARE.values()))
TRUE_SIG_S = np.eye(len(COMP_SHARE)) * COMP_SIGMA ** 2 * 60.0
TRUE_SIG_O = np.eye(len(COMP_SHARE)) * 0.6 ** 2
# Наклон при оценённых моментах — тест на большой выборке ниже (A3(в), аудит 08.10.2026).


def _slope(pm, shrink) -> float:
    return cal.label_calibration(cal.half_pairs(pm, shrink), n_boot=1)["pooled"]["calibration_slope"]


@pytest.fixture(scope="module")
def slope_runs():
    from datamatch import l5_shrink as ls
    fit = ls.JointEB.fit

    def oracle_fit(self, pm, roles, mu_pm=None):
        fit(self, pm, roles, mu_pm)
        for r in ls.ROLES:
            self.T[r], self.mu[r] = TRUE_T.copy(), np.array(list(COMP_MU.values()))
            self.no_signal[r] = self.overall_no_signal[r] = False
        self.Sig_s, self.Sig_o = TRUE_SIG_S.copy(), TRUE_SIG_O.copy()
        self.Sig_s_r = {r: TRUE_SIG_S.copy() for r in ls.ROLES}
        self.Sig_o_r = {r: TRUE_SIG_O.copy() for r in ls.ROLES}
        return self

    out = {"oracle": [], "oracle_ci": []}
    for seed in SLOPE_SEEDS:
        pm = identifiable_pm(seed=seed)
        ls.JointEB.fit = oracle_fit
        try:
            pairs = cal.half_pairs(pm, {**SHRINK, "noise_by_role": False})
        finally:
            ls.JointEB.fit = fit
        lab = cal.label_calibration(pairs, n_boot=N_BOOT_TEST)["pooled"]
        out["oracle"].append(lab["calibration_slope"])
        out["oracle_ci"].append(lab["calibration_slope_ci"])
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def test_oracle_moments_slope_is_one(slope_runs):
    """С истинными моментами стягивание верное: ДИ наклона накрывает 1 на каждом сиде, среднее ≈ 1."""
    for lo, hi in slope_runs["oracle_ci"]:
        assert lo <= 1.0 <= hi
    o = slope_runs["oracle"]
    assert abs(o.mean() - 1.0) <= 3 * o.std(ddof=1) / np.sqrt(len(o))


LARGE_SEEDS = (11, 12, 13)            # не использовались при замерах смещения
SLOPE_BOUND = 0.1                      # двусторонняя граница |наклон − 1| для среднего по сидам


@pytest.mark.parametrize("noise_by_role", [False, True])
def test_estimated_moments_slope_near_one_on_large_sample(noise_by_role):
    """A3(в), аудит 08.10.2026: на большой выборке (1000 игроков на амплуа) оценённые моменты
    состоятельны — средний наклон по трём сидам в [0,9; 1,1]; граница двусторонняя и не выводится из
    измеренного смещения (самоссылочный допуск BIAS заменён). Замер: 0,997 и 0,989."""
    slopes = [_slope(identifiable_pm(seed=s, n_per_role=1000), {**SHRINK, "noise_by_role": noise_by_role})
              for s in LARGE_SEEDS]
    assert abs(float(np.mean(slopes)) - 1.0) <= SLOPE_BOUND


def test_predictive_coverage_near_nominal_on_correct_model(talent_pairs):
    out = cal.predictive_coverage(talent_pairs, n_boot=50)
    assert out["pooled"]["coverage"] == pytest.approx(0.95, abs=0.03)


def test_two_stage_simulation_near_nominal():
    rng = np.random.default_rng(0)
    n_fit, n_eval = 3000, 300
    roles_fit = rng.choice(["C", "W", "D"], n_fit)
    roles_eval = rng.choice(["C", "W", "D"], n_eval)
    mom = {r: {"tau2": 0.05, "sigma2": 3.0, "omega2": 0.2} for r in ("C", "W", "D")}
    out = cal.simulate_two_stage_coverage(
        rng.uniform(30, 600, n_fit), rng.integers(2, 30, n_fit), roles_fit,
        rng.uniform(5, 600, n_eval), rng.integers(1, 30, n_eval), roles_eval, mom, n_sim=20, seed=1)
    assert out["refit_on_fit_structure"]["coverage"] == pytest.approx(0.95, abs=0.015)
    assert out["oracle_moments"]["coverage"] == pytest.approx(0.95, abs=0.015)
    assert set(out["refit_on_fit_structure"]["by_label"]) <= {"HIGH", "MEDIUM", "LOW"}


def test_split_half_no_leakage_from_corrupted_half():
    """A3(б), аудит 08.10.2026: испорченная половина B не меняет оценки по половине A бит в бит
    (θ̂, w, var_post, μ, τ² направления A→B: вся подгонка L5 — только по своей половине)."""
    from datamatch import validation as va
    pm = identifiable_pm(seed=4, n_per_role=60)
    halves = va.assign_halves(pm.match_id)
    in_b = pm.match_id.map(halves).to_numpy() == 1
    bad = pm.copy()
    for c in COMP_SHARE:
        bad.loc[in_b, c] = bad.loc[in_b, c] * 100.0
    a_ok = cal.half_pairs(pm, SHRINK).query("direction == 'A->B'").set_index("player_id")
    a_bad = cal.half_pairs(bad, SHRINK).query("direction == 'A->B'").set_index("player_id")
    for col in ("theta", "w", "var_post", "mu", "tau2"):
        np.testing.assert_array_equal(a_ok[col].to_numpy(), a_bad[col].reindex(a_ok.index).to_numpy())
