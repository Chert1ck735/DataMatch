"""L5: стягивание недиагональных элементов T_r к диагонали (Schäfer–Strimmer / Ledoit–Wolf), флаг конфигурации."""
import numpy as np
import pandas as pd
import pytest

from datamatch import l5_shrink as ls


def test_offdiag_shrink_hand_case():
    # Вклады игроков в C12: 0,5·(1·2 + 2·1) = 2 и 0,5·(1·0 + 0·1) = 0; веса 0,5 → C12 = 1,
    # Var(C12) = 0,25·(2−1)² + 0,25·(0−1)² = 0,5; ρ = 0,5 / 1² = 0,5 → C12* = 0,5; диагональ та же.
    ca = np.array([[1.0, 2.0], [1.0, 0.0]])
    cb = np.array([[1.0, 2.0], [1.0, 0.0]])
    w = np.array([0.5, 0.5])
    C, info = ls.offdiag_shrink(ca, cb, w)
    assert info["rho"] == pytest.approx(0.5)
    assert info["C_raw"] == pytest.approx(np.array([[1.0, 1.0], [1.0, 2.0]]))
    assert C == pytest.approx(np.array([[1.0, 0.5], [0.5, 2.0]]))


def test_offdiag_shrink_no_noise_keeps_matrix():
    # Одинаковые вклады игроков → дисперсия элементов 0 → ρ = 0.
    ca = np.array([[1.0, 1.0], [1.0, 1.0]])
    C, info = ls.offdiag_shrink(ca, ca.copy(), np.array([0.5, 0.5]))
    assert info["rho"] == 0.0
    assert C == pytest.approx(np.array([[1.0, 1.0], [1.0, 1.0]]))


def test_offdiag_shrink_rho_clipped_to_one():
    rng = np.random.default_rng(0)
    ca, cb = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))   # мало игроков: шум ≫ сигнала
    C, info = ls.offdiag_shrink(ca, cb, np.full(5, 0.2))
    assert 0.0 <= info["rho"] <= 1.0


def test_offdiag_shrink_does_not_lower_min_eigenvalue():
    # Почти вырожденная матрица (корреляция ≈ 1): стягивание к диагонали поднимает min собственное значение.
    rng = np.random.default_rng(1)
    f = rng.normal(size=400)
    ca = np.column_stack([f, f + 0.01 * rng.normal(size=400)]) + 0.3 * rng.normal(size=(400, 2))
    cb = np.column_stack([f, f + 0.01 * rng.normal(size=400)]) + 0.3 * rng.normal(size=(400, 2))
    w = np.full(400, 1 / 400)
    C, info = ls.offdiag_shrink(ca, cb, w)
    assert np.linalg.eigvalsh(C).min() >= np.linalg.eigvalsh(info["C_raw"]).min() - 1e-12


def _pm(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(150):
        theta = rng.normal(0, 0.3, 2)
        for g in range(12):
            toi = 900.0
            rows.append({"player_id": f"P{p}", "match_id": f"M{g}", "toi_s": toi, "games": 1,
                         "c_a": theta[0] * toi / 3600 + rng.normal(0, 0.05),
                         "c_b": theta[1] * toi / 3600 + rng.normal(0, 0.05)})
    return pd.DataFrame(rows)


def test_jointeb_flag_off_matches_default_and_on_changes_only_offdiagonal():
    pm = _pm(2)
    roles = pd.Series("C", index=pm.player_id.unique())
    base = {"eps_abs": 1e-10, "eps_floor_rel": 1e-6, "jitter_start": 1e-6, "jitter_max": 1e-2,
            "min_toi_split_min": 5.0, "reliability_high": 0.7, "reliability_medium": 0.4, "noise_by_role": False}
    off = ls.JointEB(["c_a", "c_b"], base).fit(pm, roles)
    off2 = ls.JointEB(["c_a", "c_b"], base | {"t_offdiag_shrink": False}).fit(pm, roles)
    on = ls.JointEB(["c_a", "c_b"], base | {"t_offdiag_shrink": True}).fit(pm, roles)
    assert np.array_equal(off.T["C"], off2.T["C"])
    assert np.allclose(np.diag(on.T_raw["C"]), np.diag(off.T_raw["C"]))
    assert "offdiag_rho" in on.T_info["C"] and "offdiag_rho" not in off.T_info["C"]
