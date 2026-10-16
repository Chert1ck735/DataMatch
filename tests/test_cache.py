"""Проверки кэша: новые признаки, фолды и --force дают настоящий пересчёт."""
import json

import numpy as np
import pandas as pd
import pytest

from datamatch import cli, splits, stage_l0, stage_l1
from datamatch.cache import frame_hash, load_checked_json, save_json_with_provenance
from datamatch.config import load_config
from datamatch.gates import GateError
from tests.test_l1 import _synthetic


def test_frame_hash_sensitive_to_values_order_and_folds():
    s = _synthetic(500)
    s["fold"] = np.arange(len(s)) % 5
    s["blocked"] = 0
    cols = stage_l1.SHOT_HASH_COLS
    h0 = frame_hash(s, cols)
    s2 = s.copy()
    s2.loc[3, "dist"] += 0.1
    s3 = s.copy()
    s3["fold"] = (s3.fold + 1) % 5
    s4 = s.iloc[::-1].reset_index(drop=True)
    assert len({h0, frame_hash(s2, cols), frame_hash(s3, cols), frame_hash(s4, cols)}) == 4


def test_oof_cache_recomputes_on_new_features_and_force(tmp_path):
    s = _synthetic(1500)
    s["fold"] = s.match_id.str[1:].astype(int) % 3
    s["blocked"] = 0
    spec = {"family": "logit_spline", "l2": 1.0}
    stage_l1._OOF_CACHE.update(dir=tmp_path, shots_hash=None, force=False)
    p1, _, _ = stage_l1.oof(spec, s, 0)
    assert len(list(tmp_path.glob("oof_*.pkl"))) == 1
    s2 = s.copy()
    s2["dist"] = s2.dist * 1.5            # другие признаки -> другой ключ, новое обучение
    p2, _, _ = stage_l1.oof(spec, s2, 0)
    assert len(list(tmp_path.glob("oof_*.pkl"))) == 2
    assert not np.allclose(p1, p2)
    # --force: файл кэша существует, но не читается
    f = sorted(tmp_path.glob("oof_*.pkl"), key=lambda x: x.stat().st_mtime)[0]
    m0 = f.stat().st_mtime_ns
    stage_l1._OOF_CACHE.update(force=True)
    stage_l1.oof(spec, s, 0)
    stage_l1._OOF_CACHE.update(force=False)
    assert any(x.stat().st_mtime_ns > m0 for x in tmp_path.glob("oof_*.pkl"))


def test_artifact_rejected_when_provenance_differs(tmp_path):
    f = tmp_path / "a.json"
    save_json_with_provenance(f, {"x": 1}, {"code": "a", "shots": "h1"})
    assert load_checked_json(f, {"code": "a", "shots": "h1"})["x"] == 1
    assert load_checked_json(f, {"code": "a", "shots": "h2"}) is None
    assert load_checked_json(f, {"code": "b", "shots": "h1"}) is None


def test_l0_cached_scheme_gate_uses_current_data_configuration(tmp_path):
    """P010: кэш v1 нельзя использовать после включения несовместимой v2."""
    cfg = load_config().with_overrides({"paths": {"outputs": str(tmp_path)},
                                        "data": {"train_schemes": ["v1", "v2"]}})
    ds = cfg["data"]["scheme_gate"]["distributions"]
    js = {d: {"v1": 0.001, "v2": 0.05} for d in ds}
    ns = {d: {"v1": 2000, "v2": 2000} for d in ds}
    res = {"quality": {"status": "ok", "js_next_node": js, "js_next_node_n": ns,
                       "gate_l0_all": {"train:scheme_compatibility": True}}}
    with pytest.raises(GateError, match="scheme_compatibility"):
        stage_l0._validate_cached_scheme_gate(cfg, res)
    assert res["quality"]["scheme_gate"]["status"] == "failed"
    assert not res["quality"]["gate_l0_all"]["train:scheme_compatibility"]


def test_l0_cached_scheme_gate_allows_explicit_stress_mode(tmp_path):
    cfg = load_config().with_overrides({"paths": {"outputs": str(tmp_path)}, "data": {"stress_mode": True,
                                        "train_schemes": ["v1", "v2"]}})
    ds = cfg["data"]["scheme_gate"]["distributions"]
    js = {d: {"v1": 0.001, "v2": 0.05} for d in ds}
    ns = {d: {"v1": 2000, "v2": 2000} for d in ds}
    res = {"quality": {"status": "stress", "js_next_node": js, "js_next_node_n": ns,
                       "gate_l0_all": {"train:scheme_compatibility": False}}}
    stage_l0._validate_cached_scheme_gate(cfg, res)
    assert res["quality"]["scheme_gate"]["status"] == "stress"
    assert res["quality"]["gate_l0_all"]["train:scheme_compatibility"]


def test_prepare_preserves_stress_status_in_quality_report(tmp_path, monkeypatch):
    """An explicitly admitted stress experiment must stay labelled in the CLI report."""
    cfg = load_config().with_overrides({"paths": {"outputs": str(tmp_path)}})
    l0 = {
        "quality": {"status": "stress"},
        "train": {"nodes": pd.DataFrame({"match_id": ["M1"], "scheme": ["v1"]}),
                  "games": pd.DataFrame({"match_id": ["M1"]})},
        "test": {"games": pd.DataFrame({"match_id": ["T1"]})},
    }
    monkeypatch.setattr(stage_l0, "run_l0", lambda *_args, **_kwargs: l0)
    monkeypatch.setattr(splits, "splits_provenance", lambda *_args, **_kwargs: {"test": True})
    monkeypatch.setattr(splits, "load_splits", lambda *_args, **_kwargs: {"matches": {}, "n_folds": 5})

    cli.cmd_prepare(cfg, cache=None, force=False)

    report = json.loads((tmp_path / "data_quality.json").read_text())
    assert report["status"] == "stress"


def test_run_identity_separates_machine_metadata_and_thread_settings(monkeypatch):
    import platform
    cfg = load_config()
    spec = {"family": "logit_spline", "features": ["dist", "angle_vis"]}
    monkeypatch.setattr(platform, "platform", lambda: "machine-A")
    a, ma = stage_l1.run_fingerprint(cfg, {"shots": "same"}, spec)
    monkeypatch.setattr(platform, "platform", lambda: "machine-B")
    b, mb = stage_l1.run_fingerprint(cfg, {"shots": "same"}, spec)
    assert a["id"] == b["id"] and ma != mb
    monkeypatch.setenv("OMP_NUM_THREADS", "17")
    changed, _ = stage_l1.run_fingerprint(cfg, {"shots": "same"}, spec)
    assert changed["id"] != a["id"]


def test_run_identity_changes_with_features_parameters_and_inputs():
    cfg = load_config()
    spec = {"family": "logit_spline", "features": ["dist", "angle_vis"], "l2": 1.}
    ids = [stage_l1.run_fingerprint(cfg, {"shots": "one"}, spec)[0]["id"],
           stage_l1.run_fingerprint(cfg, {"shots": "two"}, spec)[0]["id"],
           stage_l1.run_fingerprint(cfg, {"shots": "one"}, dict(spec, l2=2.))[0]["id"],
           stage_l1.run_fingerprint(cfg, {"shots": "one"}, dict(spec, features=spec["features"] + ["rebound"]))[0]["id"]]
    assert len(set(ids)) == 4


def test_l1_selection_without_fitted_model_recomputes(tmp_path, monkeypatch):
    from datamatch.cache import Cache
    from datamatch.registry import Registry
    from tests.test_l2 import many_nodes
    cfg = load_config().with_overrides({'paths': {'artifacts': str(tmp_path/'artifacts'),
        'cache': str(tmp_path/'cache')}})
    nd = many_nodes(15)
    mf = pd.DataFrame({'match_id': sorted(nd.match_id.unique()), 'fold': np.arange(15)%5,
        'outer': False, 'scheme': 'v1'})
    sp = {'matches': mf.to_dict(orient='list')}
    l0 = {'train': {'nodes': nd, 'events': nd, 'games': pd.DataFrame(), 'stints': pd.DataFrame()}}
    shots, _ = stage_l1.dev_shots(l0, sp, ('v1',))
    cache = Cache(cfg)
    prov = stage_l1.l1_provenance(cache, shots, sp)
    save_json_with_provenance(cfg.paths.artifacts/'xg_selected.json', {'status': 'ok',
        'spec': {'family': 'logit_spline', 'features': ['dist','angle_vis']}}, prov)
    def recompute(*args, **kwargs):
        raise RuntimeError('recomputation reached')
    monkeypatch.setattr(stage_l1.l1_audit, 'shot_time_perturbation', recompute)
    with pytest.raises(RuntimeError, match='recomputation reached'):
        stage_l1.run_l1(cfg, l0, sp, Registry(cfg), cache)
