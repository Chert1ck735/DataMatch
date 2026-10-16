import numpy as np
import pandas as pd
import pytest

from datamatch import l2_gbm, l2_value
from tests.test_l0 import HOME, _nodes, lineup_start
from tests.test_revision_l0 import WH
from tests.test_zone_starts import fo, touch


def nodes():
    rows = lineup_start(1) + [fo(HOME, 0.), touch(HOME, 'H1', 2.),
        dict(event_type='shot_on_target', actor_id='H1', actor_team_id=HOME,
             period=1, clock_s=4., x=20., y=0.), WH(5.)]
    return _nodes(rows)[0]


def test_features_resolve_node_ids_after_match_filter():
    nd = nodes()
    before = l2_gbm.v_features(nd, False)
    offset = nd.copy()
    offset['node'] += 100
    for c in ['next_node', 'prev_node']:
        offset[c] = np.where(offset[c] >= 0, offset[c] + 100, -1)
    pd.testing.assert_frame_equal(before, l2_gbm.v_features(offset, False))


def test_faceoff_mirror_uses_supplied_geometry():
    X = l2_gbm.v_features(nodes(), False)
    X['goal_line_x'], X['post_half_width'] = 27.1, .915
    a, _, w = l2_gbm.mirror_faceoffs(X, np.ones(len(X)), np.ones(len(X)), X.is_faceoff.to_numpy().astype(bool))
    assert a.dist.iloc[-1] == pytest.approx(27.1)
    assert w.sum() == pytest.approx(len(X))  # mirror divides faceoff weight, never doubles it


def test_markov_uses_unit_event_weights_and_excludes_disputed_arrival():
    # Вес события — единица (решение 06.10.2026): два наблюдаемых перехода дают экспозицию 2.
    nd = nodes()
    m = l2_value.MarkovV(l2_value.Grid([-31, 31], [-16, 16])).fit(nd, np.array([0., 0., .2]))
    assert m.n_state.sum() == pytest.approx(2.0)
    altered = nd.assign(arrival='recovery')
    np.testing.assert_array_equal(m.state(nd), m.state(altered))


def test_discount_uses_monotone_time_for_return_and_td():
    nd = nodes().assign(clock=[0., 2., 1.], t_mono=[0., 2., 4.])
    r = np.array([0., 0., .2])
    G = l2_value.realized_return(nd, r, .9)
    assert G[0] == pytest.approx(.2 * .9**4)
    np.testing.assert_allclose(l2_value.td_delta(nd, r, G, .9), 0., atol=1e-12)


def test_first_post_gap_time_feature_does_not_use_own_outcome_timestamp():
    nd = nodes()
    nd.loc[2, 'prev_node'] = -1
    nd.loc[2, ['t_ref', 't_ref_source']] = [np.nan, 'none']
    a = l2_gbm.v_features(nd, False)
    nd.loc[2, ['clock', 't_mono']] = [8., 8.]
    pd.testing.assert_series_equal(a.iloc[2], l2_gbm.v_features(nd, False).iloc[2])


def test_calibration_clusters_matches_not_repeated_picks():
    nd = nodes()
    G = l2_value.realized_return(nd, np.array([0., 0., .2]))
    report = l2_value.eval_value_model(nd, G, G, 0., bins=1, B=50)
    assert report['mse'] == 0. and not report['deciles'][0]['ok']
    assert report['deciles'][0]['residual_interval']['status'] == 'inconclusive'
    ci = l2_value.paired_value_difference(nd, G, G+.1, G, B=50)
    assert ci['mean'] == pytest.approx(.01)
    assert ci['lo95'] is None and ci['hi95'] is None  # one match cannot prove a CI


def test_graph_selection_rejects_partial_matches_and_remaps_all_links():
    from datamatch.node_graph import subset_nodes
    nd = nodes()
    with pytest.raises(ValueError, match='complete matches'):
        subset_nodes(nd, [True, True, False])
    shifted = nd.assign(node=nd.node+100, last_pass_node=[-1, 100, 100])
    for col in ['prev_node', 'next_node']:
        shifted[col] = np.where(shifted[col] >= 0, shifted[col]+100, -1)
    result = subset_nodes(shifted, [True]*3)
    assert list(result.last_pass_node) == [-1, 0, 0]


def many_nodes(n=12):
    base = nodes()
    frames = []
    for k in range(n):
        sub = base.assign(match_id=f'M{k:03}', node=base.node+len(base)*k,
                          segment=base.segment+k*10, possession=base.possession+k*10)
        for c in ['next_node', 'prev_node', 'last_pass_node']:
            sub[c] = np.where(base[c] >= 0, base[c]+len(base)*k, -1)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def test_value_procedure_smoke_and_symmetric_faceoff_evaluation():
    from datamatch.config import load_config
    from datamatch.l2_procedure import ValueProcedure, evaluation_view
    nd = many_nodes()
    r = np.tile([0., 0., .2], 12)
    cfg = load_config().with_overrides({'value': {'markov': {'x_edges': [-31, 31], 'y_edges': [-16, 16]}}})
    model = ValueProcedure(cfg, {'kind': 'markov'}, 0).fit(nd, r)
    view, G, v = evaluation_view(nd, l2_value.realized_return(nd, r), model)
    assert np.isfinite(v).all()
    w = l2_value.evaluation_weights(view, G)
    # каждое физическое событие входит один раз: перспективы вбрасывания делят его 0,5/0,5
    assert w.sum() == pytest.approx(float(np.isfinite(G[:len(nd)]).sum()))
    f = nd.action.eq('faceoff').to_numpy()
    np.testing.assert_allclose(v[:len(nd)][f], -v[len(nd):], atol=1e-10)
    report = l2_value.eval_value_model(view, G, v, 0., bins=1, B=20)
    assert np.isfinite(report['mse'])


def test_l2_provenance_changes_for_context_and_xg_spec():
    from datamatch import io, stage_l2
    from datamatch.cache import Cache
    from datamatch.config import load_config
    cfg = load_config()
    cache = Cache(cfg, io.data_version(cfg))
    nd = nodes()
    selection = {'spec': {'family': 'logit_spline', 'features': ['dist', 'angle_vis']}}
    a = stage_l2.provenance(cfg, cache, nd, {'matches': {}}, selection)
    b = stage_l2.provenance(cfg, cache, nd.assign(score_diff=1), {'matches': {}}, selection)
    c = stage_l2.provenance(cfg, cache, nd, {'matches': {}}, {'spec': dict(selection['spec'], l2=3.)})
    d = stage_l2.provenance(cfg, cache, nd.assign(sigma_next=-1), {'matches': {}}, selection)
    assert a != b and a != c and a != d


def test_own_possession_return_matches_own_possession_td():
    nd = nodes()
    nd.loc[2, 'team_id'] = nd.loc[2, 'away_team_id']
    nd.loc[1, 'sigma_next'] = -1
    nd['possession'] = [0, 0, 1]
    r = np.array([0., .1, .2])
    G = l2_value.realized_return(nd, r, own_poss_only=True)
    np.testing.assert_allclose(G, [.1, .1, .2])
    np.testing.assert_allclose(l2_value.td_delta(nd, r, G, own_poss_only=True), 0., atol=1e-12)


def test_td_resolves_global_node_ids():
    nd = nodes()
    r = np.array([0., 0., .2])
    G = l2_value.realized_return(nd, r)
    nd['node'] += 100
    nd['next_node'] = np.where(nd.next_node >= 0, nd.next_node+100, -1)
    np.testing.assert_allclose(l2_value.td_delta(nd, r, G), 0., atol=1e-12)


def test_faceoff_value_is_antisymmetric_before_winner_is_known():
    from datamatch.config import load_config
    from datamatch.l2_procedure import ValueProcedure
    class OutcomeAsymmetric:
        def value(self, nd):
            return np.full(len(nd), .2)
    proc = ValueProcedure(load_config(), {'kind': 'gbm'})
    proc.model, proc.strength, proc.calibration = OutcomeAsymmetric(), {}, (1., .1)
    v = proc.value(nodes(), neutral=True)
    assert v[0] == pytest.approx(0.)  # identical pre-outcome state in opposite perspectives
    assert v[1] == pytest.approx(.3)


def test_value_features_ignore_outcome_identity_and_future_fields():
    nd = nodes()
    a = l2_gbm.v_features(nd, True)
    mutated = nd.assign(actor_id='another', is_goal=~nd.is_goal, primary='goal',
        x2=999., y2=-999., period_end_observed=5000., terminal=False)
    pd.testing.assert_frame_equal(a, l2_gbm.v_features(mutated, True))


def test_l2_cache_identity_ignores_observational_metadata():
    from datamatch import stage_l2
    from datamatch.cache import Cache
    from datamatch.config import load_config
    cfg = load_config()
    sel = {'spec': {'family': 'logit_spline', 'features': ['dist', 'angle_vis']}, 'provenance': {'code': 'a'}}
    a = stage_l2.provenance(cfg, Cache(cfg), nodes(), {'matches': {}}, sel)
    b = stage_l2.provenance(cfg, Cache(cfg), nodes(), {'matches': {}}, dict(sel, machine={'cpu_count': 999}, seconds=42))
    assert a == b


def test_cross_fit_refits_reward_and_value_without_validation_matches(monkeypatch):
    from datamatch import stage_l2
    from datamatch.config import load_config
    nd = many_nodes()
    folds = np.repeat(np.arange(12) % 3, 3)
    seen = []
    class FakeXG:
        def __init__(self, train):
            self.train = set(train.match_id)
        def predict(self, shots):
            return np.full(len(shots), .2)
    class FakeV:
        def __init__(self, *args):
            pass
        def fit(self, train, r, **kwargs):
            self.train = set(train.match_id)
            seen.append(self.train)
            return self
        def delta_value(self, validation, neutral=False):   # контракт процедуры: V для δ (без флага = value)
            return self.value(validation, neutral)
        def value(self, validation, neutral=False):
            assert not (self.train & set(validation.match_id))
            assert neutral
            return np.zeros(len(validation))
    monkeypatch.setattr(stage_l2, 'fit_xg', lambda cfg, spec, train: FakeXG(train))
    monkeypatch.setattr(stage_l2.l2_procedure, 'ValueProcedure', FakeV)
    result, proof = stage_l2.crossfit_values(load_config(), nd, folds, {}, {'kind': 'markov'})
    assert len(seen) == 3 and len(result) == len(nd)
    assert result.node.tolist() == nd.node.tolist()
    assert result.estimate_kind.eq('match_cross_fit').all()
    assert all(p['overlap'] == 0 for p in proof)
    np.testing.assert_allclose(result.r, np.tile([0., 0., .2], 12))


def test_unknown_prediction_cannot_corrupt_paired_observed_loss():
    nd = nodes()
    G = np.array([np.nan, .2, .2])
    a = np.array([np.nan, .3, .3])
    b = np.array([np.nan, .2, .2])
    ci = l2_value.paired_value_difference(nd, G, a, b, B=20)
    assert ci['mean'] == pytest.approx(.01)


@pytest.mark.parametrize("preexisting_done", [False, True])
def test_l2_stage_failure_blocks_done_and_reuse(tmp_path, monkeypatch, preexisting_done):
    import json
    from datamatch import stage_l2
    from datamatch.cache import Cache
    from datamatch.config import load_config
    from datamatch.gates import GateError
    from datamatch.registry import Registry
    cfg = load_config().with_overrides({'paths': {'artifacts': str(tmp_path/'artifacts'),
        'cache': str(tmp_path/'cache')}, 'value': {'markov': {'x_edges': [-31,31], 'y_edges': [-16,16]}}})
    nd = many_nodes(15)
    mf = pd.DataFrame({'match_id': sorted(nd.match_id.unique()), 'fold': np.arange(15)%5,
        'outer': False, 'scheme': 'v1'})
    sp = {'matches': mf.to_dict(orient='list')}
    class FixedXG:
        def predict(self, shots):
            return np.full(len(shots), .2)
    monkeypatch.setattr(stage_l2, 'fit_xg', lambda *args: FixedXG())
    # Keep actual mathematical procedure; failed statistical gate must block artifact completion.
    monkeypatch.setattr(stage_l2.l2_procedure.ValueProcedure, 'fit',
        lambda self, train, r, **kwargs: self)
    monkeypatch.setattr(stage_l2.l2_procedure.ValueProcedure, 'value',
        lambda self, validation, neutral=False: np.zeros(len(validation)))
    cache = Cache(cfg)
    sel = {'status': 'ok', 'spec': {'family': 'logit_spline', 'features': ['dist','angle_vis']}}
    l0 = {'quality': {'status': 'ok'}, 'train': {'nodes': nd}, 'test': {'nodes': nd}}
    if preexisting_done:
        from datamatch.cache import obj_hash
        dev = stage_l2.development_nodes(nd, sp, cfg['data']['train_schemes'])
        prov = stage_l2.provenance(cfg, cache, dev, sp, sel, nd)
        key = cache.key('l2_dev_model', ('value', 'data', 'eval', 'seed'),
                        extra=obj_hash(prov), code=stage_l2.L2_CODE)
        cache.mark_done(key)
    for k in range(2):
        with pytest.raises(GateError, match='calibrated_all_deciles'):
            stage_l2.run_l2(cfg, l0, sp, sel, Registry(cfg), cache, force=(k == 0))
    # Диагностика провала лежит отдельно; принятый артефакт провалом не уничтожается.
    report = json.loads((cfg.paths.artifacts/'value_selected.failed.json').read_text())
    assert report['status'] == 'failed'
    assert not (cfg.paths.artifacts/'value_selected.json').exists()
    # Повторное использование заблокировано тем, что провал не создал ни артефакта, ни таблиц,
    # а не снятием отметки у прежней генерации: оба прохода выше подняли GateError.
    assert not list(cfg.paths.cache.glob('*/dev_values.parquet'))


def test_calibration_and_nuisance_refit_xg_inside_training_folds(monkeypatch):
    from datamatch import stage_l2
    from datamatch.config import load_config
    from datamatch.l2_procedure import ValueProcedure
    nd = many_nodes()
    training_sets = []
    class XG:
        def predict(self, shots):
            return np.full(len(shots), .2)
    def fit(cfg, spec, train):
        training_sets.append(set(train.match_id))
        return XG()
    monkeypatch.setattr(stage_l2, 'fit_xg', fit)
    cfg = load_config().with_overrides({'value': {'gbm': {'n_estimators': 2},
        'markov': {'x_edges': [-31, 31], 'y_edges': [-16, 16]}}})
    proc = ValueProcedure(cfg, {'kind': 'gbm'}, 0).fit(nd, np.tile([0.,0.,.2],12), xg_spec={'family': 'fixed'})
    assert len(training_sets) == 12
    assert all(1 < len(s) < nd.match_id.nunique() for s in training_sets)
    assert np.isfinite(proc.value(nd, neutral=True)).all()


def test_calibration_quantiles_handle_ties_without_empty_false_failures():
    nd = many_nodes()
    y = np.full(len(nd), .2)
    report = l2_value.eval_value_model(nd, y, y, 0., bins=10, B=20)
    assert report['calibration_all_deciles_ok']
    np.testing.assert_allclose([b['weight'] for b in report['deciles']], len(nd) / 10)


def test_selected_l2_pipeline_cannot_fall_back_to_legacy_or_in_sample(monkeypatch):
    from datamatch import l2_procedure
    from datamatch.config import load_config
    from datamatch.stack import ValueStack
    from datamatch.node_graph import subset_nodes
    nd = many_nodes()
    class FixedXG:
        def predict(self, shots):
            return np.full(len(shots), .2)
    class Selected:
        def __init__(self, *args):
            pass
        def fit(self, train, r, **kwargs):
            return self
        def delta_value(self, validation, neutral=False):   # контракт процедуры: V для δ (без флага = value)
            return self.value(validation, neutral)
        def value(self, validation, neutral=False):
            assert neutral
            return np.full(len(validation), .1)
    monkeypatch.setattr(l2_procedure, 'ValueProcedure', Selected)
    tr = subset_nodes(nd, nd.match_id != 'M011')
    va = subset_nodes(nd, nd.match_id == 'M011')
    stack = ValueStack(load_config(), {}, {'kind': 'gbm', 'pipeline': 'fold_local_calibrated'})
    stack.fit(tr, xg_proc=FixedXG())
    assert np.all(stack.apply(va)['v'] == .1)
    with pytest.raises(ValueError, match='match-cross-fit'):
        stack.apply(tr)
    with pytest.raises(ValueError, match='bootstrap'):
        stack.fit(tr, match_w={}, xg_proc=FixedXG())


def test_crossfit_preserves_original_l0_ids_for_downstream_join(monkeypatch):
    from datamatch import stage_l2
    from datamatch.config import load_config
    nd = many_nodes().assign(source_node=np.arange(36)+1000)
    class XG:
        def predict(self, shots):
            return np.full(len(shots), .2)
    class V:
        def __init__(self, *args):
            pass
        def fit(self, *args, **kwargs):
            return self
        def delta_value(self, validation, neutral=False):   # контракт процедуры: V для δ (без флага = value)
            return self.value(validation, neutral)
        def value(self, validation, neutral=False):
            return np.zeros(len(validation))
    monkeypatch.setattr(stage_l2, 'fit_xg', lambda *args: XG())
    monkeypatch.setattr(stage_l2.l2_procedure, 'ValueProcedure', V)
    result, _ = stage_l2.crossfit_values(load_config(), nd, np.repeat(np.arange(12)%3,3), {}, {'kind':'markov'})
    assert result.source_node.tolist() == nd.source_node.tolist()


def test_own_labels_cannot_change_crossfit_prediction_with_fixed_recipe(monkeypatch):
    from datamatch import stage_l2
    from datamatch.config import load_config
    nd = many_nodes()
    folds = np.repeat(np.arange(12)%3,3)
    class XG:
        def __init__(self, train):
            self.mean = train.is_goal.mean()
        def predict(self, shots):
            return np.full(len(shots), .2+self.mean)
    class V:
        def __init__(self, *args):
            pass
        def fit(self, train, r, **kwargs):
            self.mean = train.is_goal.mean()
            return self
        def delta_value(self, validation, neutral=False):   # контракт процедуры: V для δ (без флага = value)
            return self.value(validation, neutral)
        def value(self, validation, neutral=False):
            return np.full(len(validation), self.mean)
    monkeypatch.setattr(stage_l2, 'fit_xg', lambda cfg, spec, train: XG(train))
    monkeypatch.setattr(stage_l2.l2_procedure, 'ValueProcedure', V)
    cfg = load_config()
    before, _ = stage_l2.crossfit_values(cfg, nd, folds, {}, {'kind':'markov'})
    changed = nd.copy()
    changed.loc[changed.match_id == 'M000', 'is_goal'] = True
    after, _ = stage_l2.crossfit_values(cfg, changed, folds, {}, {'kind':'markov'})
    own = before.match_id.eq('M000')
    pd.testing.assert_frame_equal(before.loc[own], after.loc[own])
    assert not np.array_equal(before.loc[~own, 'value'], after.loc[~own, 'value'])
