"""Монотонная поправка условного среднего V (гейт калибровки по децилям L2).

Воспроизводится реальный режим отказа: после аффинной поправки у прогноза остаётся
систематический изгиб — на краях модель занижает фактическую знаковую сумму xG,
в середине завышает. Прямая такую форму исправить не может.
"""
import numpy as np
import pytest

from datamatch.l2_value import MonotoneCalibration


def s_shaped(n=20000, seed=0):
    """Прогноз v и цель y с изгибом: E[y|v] = v + a·v·(v−lo)·(v−hi) — три смены знака."""
    rng = np.random.default_rng(seed)
    v = rng.uniform(-0.03, 0.12, n)
    lo, hi = 0.0, 0.09
    mean = v + 90.0 * v * (v - lo) * (v - hi)
    y = mean + rng.normal(0, 0.05, n)
    w = np.full(n, 0.5)
    isfo = np.zeros(n, dtype=bool)
    return v, y, w, isfo, mean


def decile_residuals(v, y, w, cal, isfo):
    out = cal(v, isfo)
    edges = np.quantile(v, np.linspace(0, 1, 11))
    res = []
    for k in range(10):
        m = (v >= edges[k]) & ((v <= edges[k + 1]) if k == 9 else (v < edges[k + 1]))
        res.append(float(np.average(y[m] - out[m], weights=w[m])))
    return np.array(res)


def test_isotonic_fixes_bend_that_affine_cannot():
    v, y, w, isfo, _ = s_shaped()
    iso = MonotoneCalibration.fit(v, y, w, isfo)
    affine = MonotoneCalibration(iso.affine)
    assert iso.kind == "isotonic"
    assert affine.kind == "affine"

    r_aff = np.abs(decile_residuals(v, y, w, affine, isfo))
    r_iso = np.abs(decile_residuals(v, y, w, iso, isfo))
    # аффинная ветка оставляет изгиб в несколько тысячных — это и валило гейт
    assert r_aff.max() > 2e-3, r_aff
    # монотонная поправка уводит отклонения на порядок ниже
    assert r_iso.max() < r_aff.max() / 5, (r_iso.max(), r_aff.max())


def test_monotone_preserves_order_of_states():
    """Поправка не меняет ранжирование: иначе поехал бы вклад действий и игроков."""
    v, y, w, isfo, _ = s_shaped()
    cal = MonotoneCalibration.fit(v, y, w, isfo)
    grid = np.linspace(v.min(), v.max(), 500)
    out = cal(grid, np.zeros(len(grid), dtype=bool))
    assert np.all(np.diff(out) >= -1e-12)


def test_faceoff_calibration_is_antisymmetric():
    """V вбрасывания до исхода антисимметрична: g(−x) = −g(x)."""
    v, y, w, isfo, _ = s_shaped()
    cal = MonotoneCalibration.fit(v, y, w, isfo)
    x = np.linspace(-0.05, 0.05, 101)
    fo = np.ones(len(x), dtype=bool)
    assert np.allclose(cal(x, fo), -cal(-x, fo), atol=1e-12)
    # нулевое состояние вбрасывания обязано стоить ноль
    assert abs(float(cal(np.zeros(1), np.ones(1, dtype=bool))[0])) < 1e-12


def test_small_sample_falls_back_to_affine():
    """При малой выборке изотоническая оценка переобучается — остаётся прямая."""
    rng = np.random.default_rng(1)
    n = MonotoneCalibration.MIN_POINTS - 1
    v = rng.uniform(-0.02, 0.1, n)
    cal = MonotoneCalibration.fit(v, v + rng.normal(0, 0.05, n), np.ones(n), np.zeros(n, dtype=bool))
    assert cal.kind == "affine"


def test_few_distinct_predictions_falls_back_to_affine():
    """Таблица V даёт мало различных значений: ступенька вырождена."""
    rng = np.random.default_rng(2)
    n = 5000
    v = rng.choice(np.linspace(0, 0.05, MonotoneCalibration.MIN_DISTINCT - 1), n)
    cal = MonotoneCalibration.fit(v, v + rng.normal(0, 0.05, n), np.ones(n), np.zeros(n, dtype=bool))
    assert cal.kind == "affine"


def test_zero_weight_rows_do_not_enter_the_fit():
    v, y, w, isfo, _ = s_shaped(n=3000)
    w2, y2 = w.copy(), y.copy()
    w2[:1500] = 0.0
    y2[:1500] = 1e6          # отравленные строки с нулевым весом
    cal = MonotoneCalibration.fit(v, y2, w2, isfo)
    out = cal(v, isfo)
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) < 1.0


@pytest.mark.parametrize("shift", [0.0, 0.01])
def test_calibration_is_deterministic(shift):
    v, y, w, isfo, _ = s_shaped()
    a = MonotoneCalibration.fit(v + shift, y, w, isfo)
    b = MonotoneCalibration.fit(v + shift, y, w, isfo)
    assert np.allclose(a(v, isfo), b(v, isfo))
