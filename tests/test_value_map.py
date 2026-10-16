"""value_map: таблица оценки ситуации по клеткам площадки для заданного численного состава.

V марковской модели задана на состояниях «клетка × интервал времени до конца периода × состав». Для карты
значения клетки усредняются по интервалам времени с весом числа событий в состоянии; калибровка (нелинейная)
применяется к каждому состоянию до усреднения. Клетки без событий — NaN, с малым числом — помечаются.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from datamatch import l2_value

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "value_map.py"


def load_script():
    spec = importlib.util.spec_from_file_location("value_map", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeMarkov:
    """2 клетки по x × 1 по y, 2 интервала времени, 2 состава (как MarkovV: s = (c·nt + t)·nm + m)."""

    def __init__(self):
        self.g = l2_value.Grid([-10.0, 0.0, 10.0], [-5.0, 5.0])
        self.nt, self.nm, self.kappa = 2, 2, 20.0
        # состояния: c0t0m0, c0t0m1, c0t1m0, c0t1m1, c1t0m0, c1t0m1, c1t1m0, c1t1m1
        self.V = np.array([0.10, 0.50, 0.30, 0.70, -0.20, 0.0, 0.40, 0.0])
        self.n_state = np.array([30.0, 5.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0])


def test_cell_values_are_event_weighted_over_time_bins():
    vm = load_script()
    tab = vm.cell_table(FakeMarkov(), mp_index=0)

    c0 = tab[tab.cell == 0].iloc[0]
    assert c0.v_raw == pytest.approx((30 * 0.10 + 10 * 0.30) / 40)
    assert c0.n_events == 40
    assert not c0.low_data


def test_cell_without_events_is_nan_and_low_data():
    vm = load_script()
    tab = vm.cell_table(FakeMarkov(), mp_index=0)

    c1 = tab[tab.cell == 1].iloc[0]
    assert np.isnan(c1.v_raw) and c1.n_events == 0 and c1.low_data


def test_low_data_flag_uses_model_smoothing_strength():
    vm = load_script()
    tab = vm.cell_table(FakeMarkov(), mp_index=1)     # 5 + 5 = 10 событий < κ = 20

    assert tab[tab.cell == 0].iloc[0].low_data


def test_calibration_applied_per_state_before_averaging():
    vm = load_script()
    square = lambda raw, isfo: np.asarray(raw) ** 2  # нелинейная «калибровка» для проверки порядка операций
    tab = vm.cell_table(FakeMarkov(), mp_index=0, calibration=square)

    c0 = tab[tab.cell == 0].iloc[0]
    assert c0.v_calibrated == pytest.approx((30 * 0.10 ** 2 + 10 * 0.30 ** 2) / 40)


def test_cell_geometry_columns():
    vm = load_script()
    tab = vm.cell_table(FakeMarkov(), mp_index=0)

    assert tab[["x0", "x1", "y0", "y1"]].iloc[1].tolist() == [0.0, 10.0, -5.0, 5.0]
