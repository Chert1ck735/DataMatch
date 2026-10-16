"""L5: шум σ², ω² по амплуа (M1, NNLS по амплуа) и P055 (переоценка после обрезки).

Контракты (из модели, не из кода):
* E[d²] = σ²(1/T_a + 1/T_b) + ω²(1/M_a + 1/M_b); при ω² = 0 и почти коллинеарном плане
  свободная регрессия даёт ω̂² < 0 и завышенную σ̂²; после обрезки ω² до 0 σ² переоценивается
  по одному столбцу и возвращается к истине (P055).
* С флагом `noise_by_role` моменты шума оцениваются по игрокам своего амплуа: на синтетике с
  разным шумом у C и D восстанавливается шум каждого амплуа; общая оценка — нет.
* Апостериорная часть использует шум своего амплуа: при равных T, M и τ² у амплуа с меньшим
  шумом w больше. Без флага поведение прежнее (шум общий).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datamatch import l5_shrink as ls

ROLES = ("C", "W", "D")
COLS = ["c_a", "c_b"]
NOISE = {"C": (40.0, 0.6), "W": (40.0, 0.6), "D": (8.0, 0.1)}   # (σ², ω²) суммы по амплуа
TAU_SD = 0.3
SHRINK = {"eps_abs": 1e-10, "eps_floor_rel": 1e-6, "jitter_start": 1e-6, "jitter_max": 1e-2,
          "min_toi_split_min": 5.0, "reliability_high": 0.7, "reliability_medium": 0.4}


def noise_pm(n_per_role: int = 400, n_matches: int = 40, seed: int = 0, noise=NOISE) -> pd.DataFrame:
    """«Игрок × матч»: ставка суммы x = θ + e, Var(e за матч) = σ²_r/t + ω²_r (t в минутах).

    Доля сыгранных матчей и TOI на матч различаются между игроками — моменты идентифицируемы.
    Сумма раскладывается на два компонента поровну (шум компонентов независим, по половине).
    """
    rng = np.random.default_rng(seed)
    pid = [f"{r}{i:04d}" for r in ROLES for i in range(n_per_role)]
    role = np.repeat(ROLES, n_per_role)
    n = len(pid)
    theta = rng.normal(0.0, TAU_SD, n)
    plays = rng.random((n, n_matches)) < rng.uniform(0.2, 0.95, n)[:, None]
    pi, mi = np.nonzero(plays)
    t_min = rng.uniform(5.0, 22.0, n)[pi] * rng.uniform(0.6, 1.4, len(pi))
    s2 = np.array([noise[r][0] for r in role])[pi]
    o2 = np.array([noise[r][1] for r in role])[pi]
    cols = {}
    for c in COLS:
        x = theta[pi] / 2 + rng.normal(size=len(pi)) * np.sqrt((s2 / t_min + o2) / 2)
        cols[c] = x * t_min / 60.0
    return pd.DataFrame({"match_id": [f"m{m:03d}" for m in mi], "player_id": np.asarray(pid)[pi],
                         "toi_s": t_min * 60.0, **cols})


def roles_of(pm: pd.DataFrame) -> pd.Series:
    ids = pm.player_id.drop_duplicates()
    return pd.Series(ids.str[0].to_numpy(), index=ids.to_numpy())


def collinear_design(n: int = 3000, seed: int = 1, sigma2: float = 60.0):
    """План, где минуты за матч почти одинаковы: 1/T ≈ (1/M)/15; истинное ω² = 0."""
    rng = np.random.default_rng(seed)
    Ma, Mb = rng.integers(5, 25, n).astype(float), rng.integers(5, 25, n).astype(float)
    Ta, Tb = Ma * 15.0 * rng.uniform(0.95, 1.05, n), Mb * 15.0 * rng.uniform(0.95, 1.05, n)
    A1, A2 = 1 / Ta + 1 / Tb, 1 / Ma + 1 / Mb
    d = rng.normal(size=n) * np.sqrt(sigma2 * A1)
    return d[:, None], A1, A2


# --- P055 ----------------------------------------------------------------------------------

def test_p055_clipped_omega_refits_sigma():
    for seed in range(1, 9):
        d, A1, A2 = collinear_design(seed=seed)
        raw = ls._wls(np.column_stack([A1, A2]), 1 / A1 ** 2, d[:, 0] ** 2)
        if raw[1] < 0:      # случай P055 найден
            break
    assert raw[1] < 0
    Ss, So, info = ls.fit_noise_moments(d, A1, A2, omega_identified=True, eps_abs=1e-10, floor_rel=1e-6,
                                        reweight=False)
    assert info["refit"]
    assert So[0, 0] == pytest.approx(0.0, abs=1e-9)
    # σ² по одному столбцу при ω² = 0 — оценка истины, а не завышенная σ² свободной регрессии
    alone = ls._wls(A1[:, None], 1 / A1 ** 2, d[:, 0] ** 2)[0]
    assert Ss[0, 0] == pytest.approx(alone, rel=1e-6)
    assert Ss[0, 0] < raw[0]


def test_p055_negative_sigma_refits_omega():
    rng = np.random.default_rng(2)
    n = 3000
    Ma, Mb = rng.integers(3, 25, n).astype(float), rng.integers(3, 25, n).astype(float)
    Ta, Tb = Ma * 15.0 * rng.uniform(0.9, 1.1, n), Mb * 15.0 * rng.uniform(0.9, 1.1, n)
    A1, A2 = 1 / Ta + 1 / Tb, 1 / Ma + 1 / Mb
    d = (rng.normal(size=n) * np.sqrt(0.5 * A2))[:, None]          # истинно σ² = 0, ω² = 0,5
    Ss, So, info = ls.fit_noise_moments(d, A1, A2, True, 1e-10, 1e-6, reweight=False)
    assert Ss[0, 0] >= 0.0 and So[0, 0] >= 0.0
    assert float(Ss[0, 0] * np.mean(A1) + So[0, 0] * np.mean(A2)) == pytest.approx(0.5 * np.mean(A2), rel=0.1)


def test_psd_inputs_unchanged_by_refit():
    """Если свободная оценка уже PSD, переоценки нет (O1 на прежних данных не меняется)."""
    rng = np.random.default_rng(3)
    n = 4000
    A1 = rng.uniform(0.005, 0.1, n)
    A2 = rng.uniform(0.05, 1.0, n)
    d = rng.normal(size=(n, 2)) * np.sqrt(30.0 * A1 + 0.4 * A2)[:, None]
    Ss, So, info = ls.fit_noise_moments(d, A1, A2, True, 1e-10, 1e-6, reweight=True)
    assert not info["refit"]
    assert np.linalg.eigvalsh(Ss).min() > 0 and np.linalg.eigvalsh(So).min() > 0


# --- шум по амплуа -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pm_noise():
    return noise_pm()


def _per_match(s2: float, o2: float, t: float = 15.0) -> float:
    """Шум ставки за матч при t минутах — то, что определяет w."""
    return s2 / t + o2


def test_joint_by_role_recovers_role_noise(pm_noise):
    eb = ls.JointEB(COLS, {**SHRINK, "noise_by_role": True}).fit(pm_noise, roles_of(pm_noise))
    assert eb.noise_by_role
    for r in ("C", "D"):
        truth = _per_match(*NOISE[r])
        assert _per_match(eb.Sig_s_r[r].sum(), eb.Sig_o_r[r].sum()) == pytest.approx(truth, rel=0.15)
        assert _per_match(eb.uni[r]["sigma2"], eb.uni[r]["omega2"]) == pytest.approx(truth, rel=0.15)


def test_pooled_noise_misses_role_difference(pm_noise):
    eb = ls.JointEB(COLS, dict(SHRINK)).fit(pm_noise, roles_of(pm_noise))
    assert not eb.noise_by_role
    assert _per_match(eb.Sig_s_r["D"].sum(), eb.Sig_o_r["D"].sum()) > 1.5 * _per_match(*NOISE["D"])
    np.testing.assert_allclose(eb.Sig_s_r["C"], eb.Sig_s_r["D"])
    np.testing.assert_allclose(eb.Sig_s_r["C"], eb.Sig_s)


def test_posterior_uses_role_noise(pm_noise):
    eb = ls.JointEB(COLS, {**SHRINK, "noise_by_role": True}).fit(pm_noise, roles_of(pm_noise))
    eb.T = {r: np.eye(2) * 0.02 for r in ROLES}
    res = eb.overall(["c", "d"], np.array(["C", "D"], dtype=object), np.array([[0.1, 0.1], [0.1, 0.1]]),
                     np.array([600.0 * 60, 600.0 * 60]), np.array([40.0, 40.0]))
    assert res["w"][1] > res["w"][0]


def test_state_roundtrip_keeps_role_noise(pm_noise):
    eb = ls.JointEB(COLS, {**SHRINK, "noise_by_role": True}).fit(pm_noise, roles_of(pm_noise))
    back = ls.JointEB.from_state(eb.to_state())
    assert back.noise_by_role
    for r in ROLES:
        np.testing.assert_allclose(back.Sig_s_r[r], eb.Sig_s_r[r])
        np.testing.assert_allclose(back.Sig_o_r[r], eb.Sig_o_r[r])


def test_univariate_by_role(pm_noise):
    pm = pm_noise.assign(y=pm_noise[COLS].sum(axis=1))
    eb = ls.UnivariateEB(1e-10, noise_by_role=True).fit(pm, "y", roles_of(pm), min_toi_s=300)
    for r in ("C", "D"):
        assert _per_match(eb.sigma2_r[r], eb.omega2_r[r]) == pytest.approx(_per_match(*NOISE[r]), rel=0.15)
    pooled = ls.UnivariateEB(1e-10).fit(pm, "y", roles_of(pm), min_toi_s=300)
    assert pooled.sigma2_r["C"] == pooled.sigma2_r["D"] == pooled.sigma2


def test_p055_matrix_refit_psd_and_closer_to_truth():
    """Матричный путь P055 (k = 3, Σω ранга 1 — свободная оценка не PSD): результат PSD и ближе к
    истине по шуму за матч при 15 минутах, чем прежняя раздельная обрезка (8 сидов).

    Сходимость здесь не утверждается: на этой синтетике блочный спуск иногда упирается в лимит
    200 итераций (отличие от полностью сошедшегося ≤ ~1%); на реальных данных — 2–3 итерации.
    Флаг сходимости отложен до следующего бутстрэпа (правка l5_shrink.py меняет ключ кэша реплик).
    """
    k = 3
    ss_true = np.array([[30.0, 5.0, 2.0], [5.0, 20.0, 3.0], [2.0, 3.0, 10.0]])
    v = np.array([1.0, 0.5, 0.2])
    so_true = 0.4 * np.outer(v, v)
    truth = ss_true / 15 + so_true
    for seed in range(8):
        rng = np.random.default_rng(seed)
        n = 1500
        Ma, Mb = rng.integers(3, 25, n).astype(float), rng.integers(3, 25, n).astype(float)
        Ta, Tb = Ma * 15 * rng.uniform(0.85, 1.15, n), Mb * 15 * rng.uniform(0.85, 1.15, n)
        A1, A2 = 1 / Ta + 1 / Tb, 1 / Ma + 1 / Mb
        chol = np.linalg.cholesky(ss_true[None] * A1[:, None, None] + so_true[None] * A2[:, None, None])
        d = np.einsum("nij,nj->ni", chol, rng.normal(size=(n, k)))
        Ss, So, info = ls.fit_noise_moments(d, A1, A2, True, 1e-10, 1e-6, reweight=True)
        old_s, _ = ls.psd_project(info["Sigma_sigma_raw"], 1e-10, 1e-6)
        old_o, _ = ls.psd_project(info["Sigma_omega_raw"], 1e-10, 1e-6)
        assert np.linalg.eigvalsh(Ss).min() >= 0 and np.linalg.eigvalsh(So).min() >= 0
        assert np.linalg.norm(Ss / 15 + So - truth) < np.linalg.norm(old_s / 15 + old_o - truth)


def _objective(d, A1, A2, w, Ss, So):
    """Σ w_n ‖d dᵀ − Σσ A1 − Σω A2‖²_F по полным матрицам (внедиагональный элемент — дважды)."""
    R = np.einsum("ni,nj->nij", d, d) - A1[:, None, None] * Ss - A2[:, None, None] * So
    return float((w[:, None, None] * R * R).sum())


def test_p055_matrix_refit_is_constrained_minimum_and_converges():
    """B6 (аудит 08.10.2026): при весе 2 внедиагональным элементам блочный спуск — точный условный
    минимум; проверка — ни одно PSD-возмущение результата не уменьшает целевую функцию; спуск сходится
    (флаг `converged`), в том числе на сидах, где прежний лимит 200 итераций не хватал."""
    k = 3
    ss_true = np.array([[30.0, 5.0, 2.0], [5.0, 20.0, 3.0], [2.0, 3.0, 10.0]])
    v = np.array([1.0, 0.5, 0.2])
    so_true = 0.4 * np.outer(v, v)
    for seed in (0, 3, 7):
        rng = np.random.default_rng(seed)
        n = 1500
        Ma, Mb = rng.integers(3, 25, n).astype(float), rng.integers(3, 25, n).astype(float)
        Ta, Tb = Ma * 15 * rng.uniform(0.85, 1.15, n), Mb * 15 * rng.uniform(0.85, 1.15, n)
        A1, A2 = 1 / Ta + 1 / Tb, 1 / Ma + 1 / Mb
        chol = np.linalg.cholesky(ss_true[None] * A1[:, None, None] + so_true[None] * A2[:, None, None])
        d = np.einsum("nij,nj->ni", chol, rng.normal(size=(n, k)))
        Ss, So, info = ls.fit_noise_moments(d, A1, A2, True, 1e-10, 1e-6, reweight=True)
        assert info["refit"] and info["converged"]
        w = info["weights"]
        best = _objective(d, A1, A2, w, Ss, So)
        for _ in range(30):
            E1, E2 = rng.normal(size=(k, k)), rng.normal(size=(k, k))
            S2, _ = ls.psd_project(Ss + 0.05 * (E1 + E1.T), 1e-10, 1e-6)
            O2, _ = ls.psd_project(So + 0.005 * (E2 + E2.T), 1e-10, 1e-6)
            assert _objective(d, A1, A2, w, S2, O2) >= best * (1 - 1e-9)


def test_noise_convergence_reported_by_l5(pm_noise):
    eb = ls.JointEB(COLS, {**SHRINK, "noise_by_role": True}).fit(pm_noise, roles_of(pm_noise))
    assert "noise_converged" in eb.diag
    assert all("converged" in eb.noise_info[r] for r in ROLES)


# --- A3(а), аудит 08.10.2026: шум своего амплуа — ручной расчёт и перестановка C↔D ---------------

def _hand_w_var(T, Ss, So, toi_min, games):
    """Независимо по формулам модели: S = Σσ/T + Σω/M, V = T − T(T+S)⁻¹T, w = 1 − 1ᵀV1/1ᵀT1."""
    S = Ss / toi_min + So / games
    V = T - T @ np.linalg.solve(T + S, T)
    one = np.ones(len(T))
    return 1 - one @ V @ one / (one @ T @ one), one @ V @ one


def test_role_noise_hand_computed_and_swap(pm_noise):
    eb = ls.JointEB(COLS, {**SHRINK, "noise_by_role": True}).fit(pm_noise, roles_of(pm_noise))
    T = np.array([[0.03, 0.01], [0.01, 0.02]])
    eb.T = {r: T.copy() for r in ROLES}
    eb.mu = {r: np.zeros(2) for r in ROLES}
    toi_s, games = 300.0 * 60, 20.0
    res = eb.overall(["c", "d"], np.array(["C", "D"], dtype=object), np.array([[0.1, 0.0], [0.1, 0.0]]),
                     np.array([toi_s, toi_s]), np.array([games, games]))
    for k, r in enumerate(("C", "D")):
        w, v = _hand_w_var(T, eb.Sig_s_r[r], eb.Sig_o_r[r], toi_s / 60, games)
        assert res["w"][k] == pytest.approx(w, rel=1e-10)
        assert res["var"][k] == pytest.approx(v, rel=1e-10)
    eb.Sig_s_r["C"], eb.Sig_s_r["D"] = eb.Sig_s_r["D"], eb.Sig_s_r["C"]
    eb.Sig_o_r["C"], eb.Sig_o_r["D"] = eb.Sig_o_r["D"], eb.Sig_o_r["C"]
    sw = eb.overall(["c", "d"], np.array(["C", "D"], dtype=object), np.array([[0.1, 0.0], [0.1, 0.0]]),
                    np.array([toi_s, toi_s]), np.array([games, games]))
    assert sw["w"][0] == pytest.approx(res["w"][1], rel=1e-12)
    assert sw["w"][1] == pytest.approx(res["w"][0], rel=1e-12)
