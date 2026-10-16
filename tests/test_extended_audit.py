"""Independent regressions from the extended audit, including flag corruption."""
import numpy as np
import pandas as pd
import pytest

from datamatch import l0_quality, l1_xg, l2_value, splits, stage_l1
from datamatch.config import load_config
from tests.test_l0 import HOME, lineup_start
from tests.test_l1 import _synthetic
from tests.test_revision_l0 import WH, _full
from tests.test_zone_starts import touch


def test_split_assignments_are_canonical_for_equal_and_small_components():
    cfg = load_config()
    games = pd.DataFrame([
        {'match_id': f'{c}{i:02}', 'home_team_id': f'{c}H', 'away_team_id': f'{c}A'}
        for c, n in [('X', 20), ('Y', 20), ('Z', 4), ('W', 4)] for i in range(n)])
    test = games.iloc[[0, 20]]
    scheme = pd.Series('v1', index=games.match_id)
    expected = splits.make_splits(cfg, games, test, scheme)
    for seed in range(6):
        actual = splits.make_splits(cfg, games.sample(frac=1, random_state=seed), test, scheme)
        assert actual == expected
    assert expected['assignment_hash']
    f = pd.DataFrame(expected['matches'])
    assert (f.loc[f.outer, 'fold'] == -1).all()
    assert f.match_id.is_unique
    assert (f.loc[~f.outer, 'fold'] >= 0).all()


class ReversedAngle:
    spec = {'features': l1_xg.FEATURES}

    def predict(self, frame):
        return 1 / (1 + np.exp(1 + frame.dist.to_numpy() + frame.angle_vis.to_numpy()))


def test_independent_angle_probe_rejects_wrong_predictor():
    pr = l1_xg.probes(ReversedAngle(), _synthetic(100))
    assert pr['dist_monotone_fixed_angle'] and pr['dist_monotone_nonincreasing']
    assert not stage_l1.probe_ok(pr)
    assert not pr['angle_monotone_fixed_distance']


def _overcapacity():
    rows = lineup_start(1) + [
        dict(event_type='shift_F1', actor_id='H8', actor_team_id=HOME, period=1, clock_s=1.),
        dict(event_type='shift_F2', actor_id='H9', actor_team_id=HOME, period=1, clock_s=1.),
        touch(HOME, 'H8', 2.), WH(5.)]
    rows[-2]['n_home'] = rows[-1]['n_home'] = 8.
    return _full(rows)


def test_overcapacity_is_not_eligible_even_when_observation_agrees():
    tabs, _ = _overcapacity()
    bad = tabs['stints'].n_home_sk > 6
    assert bad.any()
    assert not tabs['stints'].loc[bad, 'valid_for_rates'].any()
    assert not tabs['stints'].loc[bad, 'valid_for_action_attribution'].any()
    nd = tabs['nodes'][tabs['nodes'].stint.isin(tabs['stints'].loc[bad, 'stint'])]
    assert not nd.valid_for_action_attribution.any()
    assert not nd.valid_for_rate_metrics.any()


def test_coverage_independently_detects_corrupted_impossible_lineup_flags():
    tabs, _ = _overcapacity()
    st, nd = tabs['stints'].copy(), tabs['nodes'].copy()
    bad = st.n_home_sk > 6
    st.loc[bad, ['valid_for_rates', 'valid_for_action_attribution']] = True
    mask = nd.stint.isin(st.loc[bad, 'stint']) & nd.node_kind.eq('player')
    assert mask.any()
    nd.loc[mask, ['valid_for_action_attribution', 'valid_for_rate_metrics', 'actor_on_ice_confirmed']] = True
    report = l0_quality.coverage_checks(st, nd)
    assert report['invalid_player_nodes_marked_excluded'] > 0
    assert report['untrusted_lineup_marked_attributable'] > 0
    assert report['structurally_invalid_stints_marked_rate_eligible'] > 0


@pytest.mark.parametrize('order', [(0, 1, 2), (0, 2, 1), (2, 1, 0)])
def test_segment_errors_never_mix_segments(order):
    # Ошибки суммируются по событиям своего отрезка (вес события — единица), порядок строк не важен.
    nd = pd.DataFrame({'segment': [0, 0, 1], 'match_id': ['A', 'A', 'B']})
    G = np.array([0., 2., 10.])
    ix = np.array(order)
    errors, _ = l2_value.per_segment_pick_errors(nd.iloc[ix].reset_index(drop=True), G[ix], np.zeros(3))
    np.testing.assert_allclose(errors, [4., 100.])


@pytest.mark.parametrize('skaters,goalie,allowed', [(3, True, True), (5, True, True),
    (6, False, True), (6, True, False), (7, False, False)])
def test_physical_capacity_preserves_legal_manpower(skaters, goalie, allowed):
    from datamatch.l0_shifts import structurally_possible_lineups
    st = pd.DataFrame({'n_home_sk': [skaters], 'n_away_sk': [5],
        'home_sk': [[f'H{i}' for i in range(skaters)]], 'away_sk': [[f'A{i}' for i in range(5)]],
        'home_g_in': [goalie], 'away_g_in': [True], 'home_g': ['HG'], 'away_g': ['AG']})
    assert bool(structurally_possible_lineups(st)[0]) == allowed


@pytest.mark.parametrize('defect', ['duplicate', 'cross_side', 'goalie_as_skater'])
def test_physical_identity_is_independent_of_matching_counts(defect):
    from datamatch.l0_shifts import structurally_possible_lineups
    h, a = [f'H{i}' for i in range(5)], [f'A{i}' for i in range(5)]
    if defect == 'duplicate':
        h[-1] = h[0]
    elif defect == 'cross_side':
        a[-1] = h[0]
    else:
        h[-1] = 'HG'
    st = pd.DataFrame({'n_home_sk': [5], 'n_away_sk': [5], 'home_sk': [h], 'away_sk': [a],
        'home_g_in': [True], 'away_g_in': [True], 'home_g': ['HG'], 'away_g': ['AG']})
    assert not structurally_possible_lineups(st)[0]


@pytest.mark.parametrize('segments', [20, 500])
def test_single_match_cannot_prove_calibration_by_repeating_segments(segments):
    nd = pd.DataFrame({'segment': np.arange(segments), 'match_id': ['ONE'] * segments})
    G = np.zeros(segments)
    report = l2_value.eval_value_model(nd, G, G, .1, B=40)
    assert not report['calibration_all_deciles_ok']
    for decile in report['deciles']:
        interval = decile['residual_interval']
        assert interval['status'] == 'inconclusive'
        assert interval['n_matches'] == 1
        assert interval['lo95'] is None and interval['hi95'] is None


def test_cluster_interval_requires_two_matches_with_weight_in_this_bin():
    nd = pd.DataFrame({'match_id': ['A', 'B', 'C']})
    interval = l2_value.cluster_interval(nd, np.zeros(3), np.array([1., 0., 0.]), B=40)
    assert interval['lo95'] is None
    assert interval['n_matches'] == 1


def test_reused_segment_ids_across_matches_are_rejected():
    nd = pd.DataFrame({'segment': [0, 0], 'match_id': ['A', 'B']})
    with pytest.raises(ValueError, match='segment'):
        l2_value.evaluation_weights(nd, np.zeros(2))


def test_censoring_does_not_renormalize_suffix_to_a_whole_segment():
    # Цензурированные события выпадают, а наблюдаемый суффикс не получает их вес.
    nd = pd.DataFrame({'segment': [0] * 4 + [1], 'match_id': ['A'] * 4 + ['B']})
    G = np.array([np.nan, np.nan, np.nan, .2, .1])
    w = l2_value.evaluation_weights(nd, G)
    np.testing.assert_array_equal(w, [0., 0., 0., 1., 1.])


@pytest.mark.parametrize('label', ['assist_primary', 'goal_error', 'entry_success'])
def test_label_clock_cannot_enter_causal_time_indirectly_through_ffill(label):
    from datamatch import l0_events
    from tests.test_l0 import ev, GAMES
    def preprocess(label_clock):
        rows = [touch(HOME, 'H1', 1.), dict(event_type=label, actor_id='H1', actor_team_id=HOME,
            period=1, clock_s=label_clock), touch(HOME, 'H2', np.nan), touch(HOME, 'H3', 5.)]
        return l0_events.preprocess(ev(rows), GAMES, load_config(), {})[0]
    a, b = preprocess(30.), preprocess(40.)
    np.testing.assert_array_equal(a.t_mono, [1., 1., 1., 5.])
    np.testing.assert_array_equal(a.t_mono, b.t_mono)
    assert a.clock_s.iloc[1] == 30.  # retain the supplied label clock for diagnostics
    assert a.clock_s.iloc[2] == 1.   # impute from observed game time only
    assert a.clock_impute_kind.iloc[2] == 'ffill'


def test_service_label_counts_do_not_change_pre_shot_manpower():
    from tests.test_l0 import _nodes
    from tests.test_zone_starts import fo
    def build(label_count):
        rows = lineup_start(1) + [fo(HOME, 0.), touch(HOME, 'H1', 1.),
            dict(event_type='assist_primary', actor_id='H1', actor_team_id=HOME,
                 period=1, clock_s=2., n_home=label_count, n_away=6),
            dict(event_type='goal', actor_id='H1', actor_team_id=HOME, period=1,
                 clock_s=3., x=20., y=0., n_home=6, n_away=6), WH(5.)]
        return _nodes(rows)[0].query('is_shot').iloc[0]
    a, b = build(4), build(6)
    assert a.own_sk == b.own_sk == 5
    assert a.mp == b.mp == 'EV5'


def test_missing_prior_count_uses_seq_lineup_not_own_shot_record():
    from tests.test_l0 import _nodes
    def build(own_count):
        rows = lineup_start(1)
        rows[-1].update(n_home=np.nan, n_away=np.nan)
        rows += [dict(event_type='goal', actor_id='H1', actor_team_id=HOME, period=1,
                  clock_s=3., x=20., y=0., n_home=own_count, n_away=6), WH(5.)]
        return _nodes(rows)[0].query('is_shot').iloc[0]
    a, b = build(4), build(6)
    assert a.own_sk == b.own_sk == 5
    assert a.mp == b.mp == 'EV5'


def test_l1_cache_rejects_json_recipe_disagreeing_with_fitted_model(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from datamatch import stage_l1
    from datamatch.cache import Cache, obj_hash, save_json_with_provenance
    cfg = load_config().with_overrides({'paths': {'cache': str(tmp_path/'cache'),
        'artifacts': str(tmp_path/'artifacts')}})
    s = _synthetic(1000).assign(blocked=False)
    s['fold'] = s.match_id.map({m: k % 5 for k, m in enumerate(sorted(s.match_id.unique()))})
    monkeypatch.setattr(stage_l1, 'dev_shots', lambda *_: (s, {}))
    cache = Cache(cfg)
    prov = stage_l1.l1_provenance(cache, s, {})
    key = cache.key('l1_dev_model', ('xg', 'data', 'eval', 'seed'), extra=obj_hash(prov), code=stage_l1.L1_CODE)
    spec = {'family': 'logit_spline', 'features': l1_xg.FEATURES, 'l2': 1.}
    original = {'status': 'ok', 'spec': spec, 'provenance': prov}
    cache.save_obj(key, 'procedure', SimpleNamespace(spec=spec))
    cache.save_obj(key, 'selection', original)
    cache.mark_done(key)
    changed = dict(original, spec=dict(spec, l2=10.))
    save_json_with_provenance(cfg.paths.artifacts/'xg_selected.json', changed, prov)
    class RecomputeReached(Exception):
        pass
    class Registry:
        def clear_layer(self, layer):
            raise RecomputeReached
    with pytest.raises(RecomputeReached):
        stage_l1.run_l1(cfg, {'quality': {'status': 'ok'}}, {}, Registry(), cache)


def test_l2_fold_cache_is_bound_to_the_selected_recipe(tmp_path, monkeypatch):
    from datamatch import stage_l2
    from datamatch.cache import Cache
    from tests.test_l2 import many_nodes
    cfg = load_config().with_overrides({'paths': {'cache': str(tmp_path/'cache')}})
    nd = many_nodes(6)
    folds = nd.match_id.map({m: k % 3 for k, m in enumerate(sorted(nd.match_id.unique()))}).to_numpy()
    class FixedXG:
        def predict(self, shots):
            return np.full(len(shots), .2)
    class FixedValue:
        def __init__(self, cfg, spec, seed):
            self.spec = spec
        def fit(self, *args, **kwargs):
            return self
        def delta_value(self, validation, neutral=True):   # контракт процедуры: V для δ (без флага = value)
            return self.value(validation, neutral)
        def value(self, validation, neutral=True):
            return np.full(len(validation), .1 if self.spec['kind'] == 'markov' else .3)
    monkeypatch.setattr(stage_l2, 'fit_xg', lambda *_: FixedXG())
    monkeypatch.setattr(stage_l2.l2_procedure, 'ValueProcedure', FixedValue)
    cache = Cache(cfg)
    a, _ = stage_l2.crossfit_values(cfg, nd, folds, {'family': 'fixed'}, {'kind': 'markov'}, cache, 'same_inputs')
    b, _ = stage_l2.crossfit_values(cfg, nd, folds, {'family': 'fixed'}, {'kind': 'gbm'}, cache, 'same_inputs')
    np.testing.assert_allclose(a.value, .1)
    np.testing.assert_allclose(b.value, .3)


@pytest.mark.parametrize('column,value', [('mp', 'EVo'), ('rink_type', 'nhl_61x26')])
def test_angle_constraint_is_checked_in_each_context(column, value):
    class BadContext:
        def predict(self, frame):
            sign = np.where(frame[column].eq(value), -1., 1.)
            return 1 / (1 + np.exp(1 + frame.dist.to_numpy() - sign*frame.angle_vis.to_numpy()))
    pr = l1_xg.probes(BadContext(), _synthetic(100))
    assert pr['dist_monotone_fixed_angle'] and pr['dist_monotone_nonincreasing']
    assert not pr['angle_monotone_fixed_distance'] and not stage_l1.probe_ok(pr)


@pytest.mark.parametrize('skaters,goalie,cap,allowed', [(5, True, 5, True),
    (6, False, 5, False), (7, True, 8, False)])
def test_configured_skater_cap_does_not_replace_the_physical_player_cap(skaters, goalie, cap, allowed):
    from datamatch.l0_shifts import structurally_possible_lineups
    st = pd.DataFrame({'max_skaters': [cap], 'n_home_sk': [skaters], 'n_away_sk': [5],
        'home_sk': [[f'H{i}' for i in range(skaters)]], 'away_sk': [[f'A{i}' for i in range(5)]],
        'home_g_in': [goalie], 'away_g_in': [True], 'home_g': ['HG'], 'away_g': ['AG']})
    assert bool(structurally_possible_lineups(st)[0]) == allowed


@pytest.mark.parametrize('flags,periods,longest,long_matches', [
    ([False, True, False], [1, 1, 1], 20., 0),
    ([False, False, True], [1, 1, 1], 40., 1),
    ([False, False, True], [1, 2, 2], 20., 0)])
def test_unreliable_gaps_are_contiguous_within_a_period(flags, periods, longest, long_matches):
    from tests.test_zone_starts import fo
    tabs, _ = _full(lineup_start(1) + [fo(HOME, 0.), touch(HOME, 'H1', 1.), WH(10.)])
    base = tabs['stints'].query('duration > 0').iloc[[0]]
    st = pd.concat([base] * 3, ignore_index=True)
    st['stint'], st['period'] = np.arange(3), periods
    st['duration'] = [20., 100., 20.] if flags[1] else [20., 20., 100.]
    st['valid_for_rates'] = flags
    report = l0_quality.coverage_checks(st, tabs['nodes'].iloc[:0])
    assert report['matches_with_unreliable_gap_gt_30s'] == long_matches
    assert report['longest_unreliable_gap_s'] == longest
    assert report['matches_with_unreliable_time_gt_30s'] == 1  # total 40 s is a different estimand


@pytest.mark.parametrize('force,force_all', [(False, False), (True, False), (False, True), (True, True)])
def test_value_cli_force_all_recomputes_requested_and_preceding_layers(monkeypatch, force, force_all):
    from datamatch import cli, stage_l2
    calls = {}
    def prepare(cfg, cache, requested):
        calls['l0'] = requested
        return {'quality': {}, 'train': {'nodes': None}, 'test': {'nodes': None}}, {}
    def xg(cfg, l0, splits, registry, cache, force=False):
        calls['l1'] = force
        return {}
    def value(cfg, l0, splits, selection, registry, cache, force=False):
        calls['l2'] = force
        return 'complete'
    monkeypatch.setattr(cli, 'cmd_prepare', prepare)
    monkeypatch.setattr(cli, 'Registry', lambda cfg: None)
    monkeypatch.setattr(stage_l1, 'run_l1', xg)
    monkeypatch.setattr(stage_l2, 'run_l2', value)
    assert cli.cmd_value(None, None, force, force_all) == 'complete'
    assert calls == {'l0': force_all, 'l1': force_all, 'l2': force or force_all}


@pytest.mark.parametrize('missing,wrong_recipe', [('l1_dev_model', None), ('l2_dev_model', None),
    (None, 'l1_dev_model'), (None, 'l2_dev_model')])
def test_outer_audit_rejects_unfinished_or_inconsistent_models_before_opening_outer(tmp_path, monkeypatch, missing, wrong_recipe):
    import json
    from types import SimpleNamespace
    from scripts import audit_frozen_layers as audit
    cfg = load_config().with_overrides({'paths': {'artifacts': str(tmp_path/'artifacts'), 'outputs': str(tmp_path/'outputs')}})
    cfg.paths.artifacts.mkdir(exist_ok=True); cfg.paths.outputs.mkdir(exist_ok=True)
    l1 = {'status': 'ok', 'spec': {'family': 'fixed'}, 'provenance': {'code': 'current1'}}
    l2 = {'status': 'ok', 'spec': {'kind': 'markov'}, 'provenance': {'code': 'current2'}, 'cache_key': 'l2_dev_model'}
    for name, selected in [('xg_selected.json', l1), ('value_selected.json', l2)]:
        (cfg.paths.artifacts/name).write_text(json.dumps(selected))
    class FakeCache:
        def key(self, stage, *args, **kwargs):
            return stage
        def done(self, key):
            return key != missing
        def load_obj(self, key, name):
            spec = l1['spec'] if key == 'l1_dev_model' else l2['spec']
            return SimpleNamespace(spec={'incorrect': True} if key == wrong_recipe else spec)
    monkeypatch.setattr(audit, 'load_config', lambda: cfg)
    monkeypatch.setattr(audit, 'setup_logging', lambda *_: None)
    monkeypatch.setattr(audit, 'Cache', lambda *_: FakeCache())
    monkeypatch.setattr(audit.io, 'data_version', lambda *_: {})
    monkeypatch.setattr(audit.cli, 'cmd_prepare', lambda *_: ({
        'train': {'nodes': pd.DataFrame({'match_id': ['synthetic']})}, 'test': {'nodes': pd.DataFrame()}}, {}))
    monkeypatch.setattr(audit.stage_l1, 'dev_shots', lambda *_: (None, {}))
    monkeypatch.setattr(audit.stage_l1, 'l1_provenance', lambda *_: l1['provenance'])
    monkeypatch.setattr(audit.stage_l2, 'development_nodes', lambda *_: None)
    monkeypatch.setattr(audit.stage_l2, 'final_fit_nodes', lambda *_: None)
    monkeypatch.setattr(audit.stage_l2, 'provenance', lambda *_: l2['provenance'])
    def unopened(*args, **kwargs):
        raise AssertionError('Outer was opened before cache admission')
    monkeypatch.setattr(audit, 'matches_for', unopened)
    with pytest.raises(ValueError, match='Frozen'):
        audit.main()
