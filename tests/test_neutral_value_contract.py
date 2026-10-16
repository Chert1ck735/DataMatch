"""Доработки 06.10.2026: калибровка и гейты V по нейтральным значениям, потоки моделей, Спирмен–Браун."""
import json

import numpy as np
import pandas as pd
import pytest

from datamatch import config, l2_procedure, l2_value, validation
from tests.test_l2 import many_nodes


class _StrengthSensitiveModel:
    """Подмодель, чей прогноз зависит от силы команд: нейтральный 0.01, с силой 0.05."""

    def __init__(self, cfg):
        self.cfg = cfg

    def value(self, nodes, neutral=False):
        return np.full(len(nodes), 0.01 if neutral else 0.05)


def test_evaluation_view_can_score_neutral_exported_values():
    from datamatch.config import load_config
    cfg = load_config().with_overrides({"value": {"markov": {"x_edges": [-31, 31], "y_edges": [-16, 16]}}})
    nd = many_nodes(3)
    G = np.zeros(len(nd))
    model = _StrengthSensitiveModel(cfg)
    _, _, conditional = l2_procedure.evaluation_view(nd, G, model)
    _, _, neutral = l2_procedure.evaluation_view(nd, G, model, neutral=True)
    assert np.allclose(conditional, 0.05) and np.allclose(neutral, 0.01)


def test_value_calibration_is_learned_on_neutral_predictions(monkeypatch):
    # Экспорт δ идёт с нейтральной силой, поэтому и калибровка учится на нейтральных прогнозах.
    from datamatch.config import load_config
    cfg = load_config().with_overrides({"value": {"markov": {"x_edges": [-31, 31], "y_edges": [-16, 16]}}})
    seen = {}

    def capture(p, y, w, isfo):
        seen["p"] = np.asarray(p)
        return l2_value.MonotoneCalibration((1.0, 0.0))

    monkeypatch.setattr(l2_procedure.l2_value.MonotoneCalibration, "fit", staticmethod(capture))
    real_fit = l2_procedure.ValueProcedure.fit

    def fit_with_sensitive_submodels(self, nodes, r, calibrate=True, **kwargs):
        if not calibrate:
            self.value = _StrengthSensitiveModel(cfg).value
            self.cfg = cfg
            return self
        return real_fit(self, nodes, r, calibrate=True, **kwargs)

    monkeypatch.setattr(l2_procedure.ValueProcedure, "fit", fit_with_sensitive_submodels)
    nd = many_nodes(6)
    r = np.tile([0., 0., .2], 6)
    try:
        l2_procedure.ValueProcedure(cfg, {"kind": "markov"}, 0).fit(nd, r)
    except Exception:
        pass   # дальнейшие шаги полной подгонки здесь не важны; важно, на чём училась калибровка
    assert "p" in seen and np.allclose(seen["p"], 0.01)


def test_l2_gates_include_calibration_of_neutral_exported_values(tmp_path, monkeypatch):
    from datamatch import stage_l2
    from datamatch.cache import Cache
    from datamatch.config import load_config
    from datamatch.gates import GateError
    from datamatch.registry import Registry
    from tests.test_l2_final_fit import FixedXG, _splits
    cfg = load_config().with_overrides({"paths": {"artifacts": str(tmp_path / "artifacts"),
        "cache": str(tmp_path / "cache")}, "value": {"markov": {"x_edges": [-31, 31], "y_edges": [-16, 16]}}})
    nd = many_nodes(18)
    sp = _splits(nd.match_id.unique())
    monkeypatch.setattr(stage_l2, "fit_xg", lambda *args: FixedXG())
    monkeypatch.setattr(stage_l2.l2_procedure.ValueProcedure, "fit", lambda self, train, r, **kw: self)
    monkeypatch.setattr(stage_l2.l2_procedure.ValueProcedure, "value",
                        lambda self, validation, neutral=False: np.zeros(len(validation)))
    l0 = {"quality": {"status": "ok"}, "train": {"nodes": nd}, "test": {"nodes": nd}}
    with pytest.raises(GateError):
        stage_l2.run_l2(cfg, l0, sp, {"status": "ok", "spec": FixedXG.spec}, Registry(cfg), Cache(cfg), force=True)
    report = json.loads((cfg.paths.artifacts / "value_selected.failed.json").read_text())
    assert "calibrated_all_deciles_neutral" in report["gate_l2"]
    assert "deciles_ok" in report["metrics_neutral"]


def test_outer_audit_reports_neutral_value_calibration_without_blocking():
    # Решение 07.10.2026: калибровка V на внешнем блоке — диагностика со статусом warn, а не гейт
    # (блок просматривался прежними ревизиями; итоговая модель обучена и на нём).
    from scripts import audit_frozen_layers as audit
    gates = audit.value_gates(np.array([0.1, 0.2]))
    assert gates == {"finite_value": True}
    status = audit.value_calibration_status({"calibration_all_deciles_ok": False, "deciles_ok": 9,
                                             "deciles": [{"ok": True}] * 9 + [{"ok": False}]})
    assert status == {"status": "warn", "deciles_ok": 9, "failed_deciles": [9]}
    assert audit.value_calibration_status({"calibration_all_deciles_ok": True, "deciles_ok": 10,
                                           "deciles": [{"ok": True}] * 10})["status"] == "ok"


def test_outer_audit_gate_failure_exits_with_code_2(monkeypatch):
    from scripts import audit_frozen_layers as audit
    from datamatch.gates import GateError

    def failing():
        raise GateError("гейт")

    monkeypatch.setattr(audit, "main", failing)
    assert audit.run() == 2


def test_model_threads_follow_the_worker_limit(monkeypatch):
    monkeypatch.delenv("DATAMATCH_MODEL_THREADS", raising=False)
    assert config.model_threads() == config.DEFAULT_MODEL_THREADS
    monkeypatch.setenv("DATAMATCH_MODEL_THREADS", "2")
    assert config.model_threads() == 2
    monkeypatch.setenv("DATAMATCH_MODEL_THREADS", "0")
    assert config.model_threads() == 1


@pytest.mark.parametrize("r, expected", [(0.5, 2 * 0.5 / 1.5), (0.0, 0.0), (-0.5, 0.0), (-0.9, 0.0)])
def test_spearman_brown_never_leaves_the_reliability_scale(r, expected):
    # Ревью 06.10.2026: при r < 0 формула 2r/(1+r) давала −2 и −18 — вне шкалы надёжности [0, 1].
    from datamatch import phys_profile
    assert validation.spearman_brown(r) == pytest.approx(expected)
    assert phys_profile.spearman_brown(r) == pytest.approx(expected)
    assert np.isnan(validation.spearman_brown(float("nan")))


def test_spearman_brown_vectorised_inputs_are_clipped():
    out = validation.spearman_brown(pd.Series([0.5, -0.5]).to_numpy())
    np.testing.assert_allclose(out, [2 * 0.5 / 1.5, 0.0])
