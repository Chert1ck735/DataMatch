"""L2: δ по исходной (некалиброванной) марковской V — флаг value.delta_from_raw (METHODOLOGY 3.x, 08.10).

Изотоническая калибровка V нелинейна и нарушает одношаговую согласованность Беллмана: на test средний δ
без вбрасываний −0,0013 с калибровкой и 0,0000 без неё (диагностика 08.10). Калиброванная V остаётся для
гейта калибровки и отчёта, δ считается по V, которая согласована с переходами.
"""
import numpy as np
import pandas as pd

from datamatch import l2_procedure


class _Model:
    def value(self, nodes):
        return nodes.x.to_numpy(dtype=float) / 10.0


def _proc(flag):
    p = l2_procedure.ValueProcedure.__new__(l2_procedure.ValueProcedure)
    p.cfg = {"value": {} if flag is None else {"delta_from_raw": flag}}
    p.model, p.strength = _Model(), {}
    p.calibration = (2.0, 0.5)          # аффинная поправка: v_cal = 2·raw + 0,5
    return p


def _nodes():
    return pd.DataFrame({"x": [1.0, -2.0, 3.0], "action": ["shot", "pass", "carry"],
                         "team_id": ["A", "B", "A"], "opp_team_id": ["B", "A", "B"]})


def test_value_applies_calibration():
    assert np.allclose(_proc(False).value(_nodes(), neutral=True), [0.7, 0.1, 1.1])


def test_delta_value_default_equals_calibrated_value():
    for flag in (None, False):
        p = _proc(flag)
        assert np.array_equal(p.delta_value(_nodes(), neutral=True), p.value(_nodes(), neutral=True))


def test_delta_value_flag_on_returns_uncalibrated():
    assert np.allclose(_proc(True).delta_value(_nodes(), neutral=True), [0.1, -0.2, 0.3])


def test_delta_value_does_not_mutate_calibration():
    p = _proc(True)
    p.delta_value(_nodes(), neutral=True)
    assert p.calibration == (2.0, 0.5)
    assert np.allclose(p.value(_nodes(), neutral=True), [0.7, 0.1, 1.1])
