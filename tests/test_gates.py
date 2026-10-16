"""Гейты блокируют результат; угол ворот; независимость L0 от размера порции."""
import json

import numpy as np
import pandas as pd
import pytest

from datamatch import l0_events, stage_l1
from datamatch.cache import load_checked_json
from datamatch.gates import GateError, enforce, require_ok


def test_visible_angle_is_zero_behind_goal_line():
    # перед воротами по оси: 2·atan(0.915/d)
    assert l0_events.visible_angle(10.0, 0.0, 0.915) == pytest.approx(2 * np.arctan(0.915 / 10.0))
    # из-за линии ворот створ не виден: видимый угол 0, а не угол между направлениями на штанги
    # (иначе точка в 0,3 м за воротами получает угол пятачка, и монотонный xG её завышает)
    for dx, dy in [(-0.3, 0.0), (-0.5, 0.3), (-2.0, 2.0), (0.0, 0.0)]:
        assert float(l0_events.visible_angle(dx, dy, 0.915)) == 0.0
    grid = np.array(np.meshgrid(np.linspace(-4, 30, 50), np.linspace(-15, 15, 50))).reshape(2, -1)
    a = l0_events.visible_angle(grid[0], grid[1], 0.915)
    assert np.all((a >= 0) & (a <= np.pi + 1e-12))
    assert np.all(a[grid[0] <= 0] == 0.0)


def _probe(axis=True, fixed=True, time_ok=True):
    return {"dist_monotone_nonincreasing": axis, "dist_monotone_fixed_angle": fixed, "period_time_no_jumps": time_ok, "angle_monotone_fixed_distance": True}


def test_probe_ok_requires_axis_monotonicity_and_all_folds():
    assert stage_l1.probe_ok(_probe())
    assert not stage_l1.probe_ok(_probe(axis=False))
    assert not stage_l1.probes_ok_all([_probe(), _probe(fixed=False), _probe()])


def test_failed_gate_blocks_result(tmp_path):
    f = tmp_path / "xg_selected.json"
    prov = {"code": "x"}
    with pytest.raises(GateError):
        stage_l1.finalize_selection(f, {"selected": "ebm"}, {"probes_final_model": False, "probes_all_folds": True}, prov)
    rep = json.loads((tmp_path / "xg_selected.failed.json").read_text())
    assert rep["status"] == "failed" and rep["failed_gates"] == ["probes_final_model"]
    # провал не создаёт принятого артефакта: следующему слою брать нечего
    assert not f.exists()
    # отчёт с провалом не принимается как актуальный артефакт и блокируется require_ok
    assert load_checked_json(f, prov) is None
    with pytest.raises(GateError):
        require_ok(rep, "L1")
    ok = stage_l1.finalize_selection(f, {"selected": "ebm"}, {"probes_final_model": True, "probes_all_folds": True}, prov)
    assert ok["status"] == "ok" and require_ok(load_checked_json(f, prov), "L1")["status"] == "ok"


def test_failed_gate_preserves_the_previously_accepted_artifact(tmp_path):
    """Провал гейта — диагностика, а не разрушение уже принятого результата слоя.

    Отчёт о провале писался поверх xg_selected.json / value_selected.json, то есть ровно
    туда, откуда следующий слой берёт принятый выбор. Одна неудачная попытка уничтожала
    результат предыдущей удачной, и восстановить его было неоткуда.
    """
    # Arrange: слой уже принят
    f = tmp_path / "xg_selected.json"
    prov = {"code": "x"}
    passing = {"probes_final_model": True, "probes_all_folds": True}
    assert stage_l1.finalize_selection(f, {"selected": "ebm"}, passing, prov)["status"] == "ok"

    # Act: следующая попытка проваливает гейт
    with pytest.raises(GateError):
        stage_l1.finalize_selection(f, {"selected": "lgbm"},
                                    {"probes_final_model": False, "probes_all_folds": True}, prov)

    # Assert: принятый артефакт цел и по-прежнему пригоден для следующего слоя
    assert require_ok(load_checked_json(f, prov), "L1")["selected"] == "ebm"
    # и диагностика провала при этом не потеряна, а лежит отдельно
    failed = json.loads((tmp_path / "xg_selected.failed.json").read_text())
    assert failed["status"] == "failed" and failed["failed_gates"] == ["probes_final_model"]


def test_enforce_writes_failed_report(tmp_path):
    f = tmp_path / "data_quality.json"
    with pytest.raises(GateError):
        enforce("L0", {"train:toi_95_100": False, "test:toi_95_100": True}, f, {"x": 1})
    rep = json.loads(f.read_text())
    assert rep["status"] == "failed" and rep["failed_gates"] == ["train:toi_95_100"] and rep["x"] == 1
