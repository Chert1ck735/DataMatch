"""Проверки L1 (xG): отсутствие утечек в признаках, монотонность, метрики, парный бутстрэп."""
import numpy as np
import pandas as pd

from datamatch import l1_xg, stage_l1
from datamatch.config import load_config


def test_no_forbidden_or_outcome_features():
    feats = set(l1_xg.FEATURES)
    assert not feats & l1_xg.FORBIDDEN
    # время текущего события (сигнал исхода у голов) и идентификаторы не входят
    for f in ["dt_prev_c", "dt_last_own_shot_c", "t_remain_c", "actor_id", "net_x", "x2", "y2"]:
        assert f not in feats


def _synthetic(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.uniform(1.5, 40, n)
    ang = 2 * np.arctan2(0.915, d) * rng.uniform(0.3, 1.0, n)
    logit = 0.5 - 0.18 * d + 2.0 * ang
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    s = pd.DataFrame({
        "dist": d, "angle_vis": ang, "prev_pass_dy": 0.0, "score_diff_c": 0, "own_sk": 5, "opp_sk": 5,
        "opp_def": 0.0, "behind_net": 0, "rebound": 0, "prev_pass_cross": 0, "recent_entry": 0, "own_g": 1,
        "opp_g": 1, "mp": "EV5", "rink_type": "european_60x30", "period_c": "2", "goal": y,
        "match_id": [f"M{i % 60}" for i in range(n)], "opp_team_id": [f"T{i % 7}" for i in range(n)],
    })
    return s


def test_mono_logit_monotone_in_distance():
    s = _synthetic()
    m = l1_xg.MonoLogit(1.0).fit(s)
    g = s.iloc[:50].copy()
    g["dist"] = np.linspace(1.5, 50, 50)
    g["angle_vis"] = 0.1
    p = m.predict(g)
    assert np.all(np.diff(p) <= 1e-12)


def test_lgbm_monotone_in_distance():
    s = _synthetic()
    m = l1_xg.LGBMxG({"n_estimators": 60, "num_leaves": 7, "min_child_samples": 20}).fit(s)
    g = s.iloc[:50].copy()
    g["dist"] = np.linspace(1.5, 50, 50)
    g["angle_vis"] = 0.1
    assert np.all(np.diff(m.predict(g)) <= 1e-12)


def test_procedure_uses_neutral_strength_distribution():
    s = _synthetic()
    proc = l1_xg.XGProcedure({"family": "logit_spline"}).fit(s)
    a = s.copy()
    a["opp_team_id"] = "T0"
    b = s.copy()
    b["opp_team_id"] = "T3"
    # при расчёте xG подставляется средняя сила соперника -> от команды соперника не зависит
    assert np.allclose(proc.predict(a), proc.predict(b))


def test_neutral_prediction_integrates_probabilities_not_mean_covariate():
    from scipy.special import expit
    class Nonlinear:
        def predict(self, s):
            return expit(-3 + s.opp_def.to_numpy())
    proc = l1_xg.XGProcedure({"family": "lgbm_mono"})
    proc.model = Nonlinear()
    proc.strength_values, proc.strength_weights = np.array([-1., 1.]), np.array([.5, .5])
    s = _synthetic(4)
    assert np.allclose(proc.predict(s), .5 * expit(-4) + .5 * expit(-2))
    assert not np.allclose(proc.predict(s), expit(-3))


def test_strength_encoding_does_not_see_own_match_outcomes():
    s = _synthetic(1200)
    changed = s.copy()
    chosen = s.match_id == "M0"
    changed.loc[chosen, "goal"] = 1 - changed.loc[chosen, "goal"]
    a, _ = l1_xg.crossfit_strength(s, l1_xg.FEATURES, 42)
    b, _ = l1_xg.crossfit_strength(changed, l1_xg.FEATURES, 42)
    assert np.allclose(a[chosen], b[chosen])


def test_tuning_model_does_not_see_comparison_outcomes(monkeypatch):
    s = _synthetic(600)
    s["fold"] = s.match_id.str[1:].astype(int) % 5
    s["blocked"] = 0
    fitted = []
    class Model:
        def __init__(self, *_): pass
        def fit(self, train):
            fitted.append(set(train.fold))
            self.p = train.goal.mean()
            return self
        def predict(self, val): return np.full(len(val), self.p)
    monkeypatch.setattr(l1_xg, "XGProcedure", Model)
    monkeypatch.setattr(l1_xg, "probes", lambda *_: {})
    monkeypatch.setitem(stage_l1._OOF_CACHE, "dir", None)
    a, _, _ = stage_l1.oof({"family": "logit_spline"}, s, 0, [3, 4])
    changed = s.copy()
    changed.loc[s.fold.isin([3, 4]), "goal"] = 1 - changed.loc[s.fold.isin([3, 4]), "goal"]
    b, _, _ = stage_l1.oof({"family": "logit_spline"}, changed, 0, [3, 4])
    assert fitted == [{0, 1, 2}, {0, 1, 2}]
    assert np.allclose(a[s.fold.isin([3, 4])], b[s.fold.isin([3, 4])])


def test_time_probe_reports_not_applicable():
    proc = l1_xg.XGProcedure({"family": "logit_spline"}).fit(_synthetic(600))
    pr = l1_xg.probes(proc, _synthetic(600))
    assert pr["period_time_probe_status"] == "not_applicable"
    assert pr["period_time_no_jumps"] is None
    assert stage_l1.probe_ok(pr)


def test_borderline_ablation_keeps_audited_feature():
    s = _synthetic(600)
    p = np.full(len(s), .1)
    r = stage_l1.ablation_decision(s, p, p, load_config())
    assert r["keep"] and r["status"] == "borderline_retained"


def test_exact_strength_integration_shortcuts_match_direct_sum():
    s = _synthetic(1200)
    specs = [{"family": "logit_spline"},
             {"family": "lgbm_mono", "params": {"n_estimators": 30, "num_leaves": 7, "min_child_samples": 20}},
             {"family": "catboost_mono", "params": {"iterations": 30, "depth": 4}}]
    for spec in specs:
        proc = l1_xg.XGProcedure(spec).fit(s)
        values, counts = np.unique(proc.training_strength, return_counts=True)
        direct = np.zeros(20)
        sample = s.iloc[:20].copy()
        for value, weight in zip(values, counts / counts.sum()):
            sample["opp_def"] = value
            direct += weight * proc.model.predict(sample)
        np.testing.assert_allclose(proc.predict(sample), direct, rtol=1e-12, atol=1e-12)


def test_logit_is_monotone_in_angle_as_well_as_distance():
    model = l1_xg.MonoLogit().fit(_synthetic(1200))
    sample = _synthetic(80)
    sample["dist"] = 12.
    sample["angle_vis"] = np.linspace(.01, 1., len(sample))
    assert np.all(np.diff(model.predict(sample)) >= -1e-12)


def test_catboost_monotonicity_and_neutral_prediction():
    s = _synthetic(2000)
    proc = l1_xg.XGProcedure({"family": "catboost_mono", "params": {"iterations": 40, "depth": 4}}, 7).fit(s)
    assert stage_l1.probe_ok(l1_xg.probes(proc, s))
    first, second = s.iloc[:20].copy(), s.iloc[:20].copy()
    first["opp_team_id"], second["opp_team_id"] = "T0", "T6"
    np.testing.assert_array_equal(proc.predict(first), proc.predict(second))


def test_catboost_accepts_feature_set_without_opponent_strength():
    s = _synthetic(1200)
    spec = {"family": "catboost_mono", "features": ["dist", "angle_vis"],
            "params": {"iterations": 30, "depth": 4}}
    proc = l1_xg.XGProcedure(spec, 7).fit(s)
    np.testing.assert_allclose(proc.predict(s), proc.model.predict(s), rtol=1e-12, atol=1e-12)
    assert stage_l1.probe_ok(l1_xg.probes(proc, s))


def test_oof_rejects_match_split_across_folds(monkeypatch):
    import pytest
    s = _synthetic(600)
    s["fold"] = np.arange(len(s)) % 7
    s["blocked"] = 0
    monkeypatch.setitem(stage_l1._OOF_CACHE, "dir", None)
    with pytest.raises(ValueError, match="match crosses"):
        stage_l1.oof({"family": "logit_spline"}, s, 0)


def test_oof_returns_its_probes_instead_of_passing_them_through_a_module_global(monkeypatch):
    """Пробы монотонности входят в обязательный гейт L1, поэтому они обязаны приходить из
    того же вызова, что и предсказания.

    Передача через процессный словарь означает, что любой вызов oof между получением
    предсказаний и чтением проб припишет кандидату пробы чужой модели.
    """
    # Arrange
    s = _synthetic(1500)
    s["fold"] = s.match_id.str[1:].astype(int) % 3
    s["blocked"] = 0
    monkeypatch.setitem(stage_l1._OOF_CACHE, "dir", None)

    # Act
    returned = stage_l1.oof({"family": "logit_spline"}, s, 0)

    # Assert: пробы отданы вызовом, а не оставлены в глобале
    assert len(returned) == 3
    probes = returned[2]
    assert probes and all(isinstance(p, dict) for p in probes)
    assert "last_probe" not in stage_l1._OOF_CACHE


def test_metrics_and_paired_bootstrap():
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.1).astype(float)
    p_good = np.where(y == 1, 0.3, 0.08)
    p_const = np.full(4000, 0.1)
    m = l1_xg.metrics(y, p_good)
    assert m["auc_all"] > 0.9 and 0 <= m["ece"] < 1
    mids = np.array([f"M{i % 40}" for i in range(4000)])
    c = l1_xg.paired_bootstrap_loss(mids, l1_xg.logloss_vec(y, p_good), l1_xg.logloss_vec(y, p_const), 300)
    assert c["hi95"] < 0 and c["lo95"] <= c["diff"] <= c["hi95"]


def test_ebm_monotone_in_distance():
    s = _synthetic(4000)
    m = l1_xg.EBMxG(seed=0, interactions=3, max_rounds=300).fit(s)
    g = s.iloc[:50].copy()
    g["dist"] = np.linspace(1.5, 50, 50)
    g["angle_vis"] = 0.1
    p = m.predict(g)
    assert np.all(np.diff(p) <= 1e-9)
    # взаимодействия с расстоянием и углом запрещены
    assert stage_l1.probe_ok(l1_xg.probes(m, s))
    assert not [t for t in m.model.term_names_ if "&" in t and ("dist" in t or "angle_vis" in t)]


def test_every_feature_has_provenance():
    assert set(l1_xg.FEATURES) <= set(l1_xg.FEATURE_PROVENANCE)
    for cols in l1_xg.DISPUTED.values():
        assert set(cols) <= set(l1_xg.FEATURES)


def test_timing_audit_is_explicitly_unestimable_without_goals():
    from datamatch.l1_audit import pre_shot_timing_check
    from tests.test_l2 import nodes
    result = pre_shot_timing_check(nodes())
    assert result['pre_shot_lag_auc'] is None
    assert result['own_stamp_dt_auc'] is None
    assert result['auc_status'] == 'not_estimable_single_class'


def test_timing_audit_does_not_mix_other_past_shot_perturbations():
    from datamatch import l1_audit
    from datamatch.config import load_config
    from tests.test_revision_l0 import _full, WH
    from tests.test_l0 import HOME, GEO, lineup_start
    from tests.test_zone_starts import fo, touch
    rows = lineup_start(1) + [fo(HOME, 0.),
        dict(event_type='goal', actor_id='H1', actor_team_id=HOME, period=1, clock_s=10., x=20., y=0.),
        fo(HOME, 0., t=10.1),
        dict(event_type='pass_complete', actor_id='H1', actor_team_id=HOME, other_id='H2',
             period=1, clock_s=10.2, x=0., y=-3., x2=0., y2=3.),
        dict(touch(HOME, 'H2', 13.5), y=3.),
        dict(event_type='shot_on_target', actor_id='H2', actor_team_id=HOME,
             period=1, clock_s=13.8, x=20., y=3.), WH(20.)]
    tabs, _ = _full(rows)
    from tests.test_l0 import GAMES
    audit = l1_audit.shot_time_perturbation(tabs['events'], GAMES, tabs['stints'], GEO,
        load_config(), ['M1'], (4.,), features=['prev_pass_dy'])
    assert audit['time_dependent'] == []
