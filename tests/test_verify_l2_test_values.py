"""verify_l2_artifacts: сверка сохранённых значений test с процедурой при обоих вариантах δ.

При value.delta_from_raw колонка `value` хранит V для δ (без калибровки), `value_calibrated` — калиброванную V
(stage_l2, поправка 08.10). Проверка обязана сравнивать каждую колонку со своим методом процедуры.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_l2_artifacts.py"


def load_script():
    spec = importlib.util.spec_from_file_location("verify_l2_artifacts", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeProcedure:
    """V для δ и калиброванная V различаются, как при delta_from_raw: true."""

    def delta_value(self, nodes, neutral=False):
        return nodes.x.to_numpy(float)

    def value(self, nodes, neutral=False):
        return 2.0 * nodes.x.to_numpy(float) + 0.01


def test_test_values_match_delta_value_and_calibrated_value():
    script = load_script()
    test = pd.DataFrame({"x": [0.0, 0.1, -0.2]})
    proc = FakeProcedure()
    tv = pd.DataFrame({"value": proc.delta_value(test), "value_calibrated": proc.value(test)})

    script.check_test_values(proc, test, tv)


def test_test_values_mismatch_is_reported():
    script = load_script()
    test = pd.DataFrame({"x": [0.0, 0.1, -0.2]})
    proc = FakeProcedure()
    tv = pd.DataFrame({"value": proc.value(test), "value_calibrated": proc.value(test)})

    with pytest.raises(AssertionError):
        script.check_test_values(proc, test, tv)


def test_older_cache_without_calibrated_column_checks_delta_value_only():
    script = load_script()
    test = pd.DataFrame({"x": [0.0, 0.1, -0.2]})
    proc = FakeProcedure()
    tv = pd.DataFrame({"value": proc.delta_value(test)})

    script.check_test_values(proc, test, tv)
    assert np.isfinite(tv.value).all()
