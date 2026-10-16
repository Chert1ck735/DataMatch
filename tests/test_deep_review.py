"""Independent boundary cases from the next code review, without full training."""
import json

import numpy as np
import pandas as pd
import pytest

from datamatch import l0_quality, l1_xg, l3_contrib, stage_l1, stage_l2
from datamatch.cache import Cache
from datamatch.config import load_config
from datamatch.gates import GateError
from datamatch.registry import Registry
from tests.test_l2 import many_nodes, nodes


def test_xg_single_match_cannot_prove_a_loss_improvement():
    ci = l1_xg.paired_bootstrap_loss(np.array(['M'] * 100), np.zeros(100), np.ones(100), B=40)
    assert ci['diff'] == -1.
    assert ci['lo95'] is None and ci['hi95'] is None
    assert ci['status'] == 'inconclusive'


def test_xg_ablation_retains_features_when_only_one_independent_match():
    s = pd.DataFrame({'match_id': ['M'] * 100, 'goal': np.zeros(100)})
    cfg = load_config().with_overrides({'eval': {'paired_bootstrap_B': 40}})
    decision = stage_l1.ablation_decision(s, np.full(100, .1), np.full(100, .5), cfg)
    assert decision['keep'] and decision['status'] == 'inconclusive_retained'


def test_l2_inconclusive_comparison_is_rejected_by_gate_not_type_error(tmp_path, monkeypatch):
    cfg = load_config().with_overrides({'paths': {'artifacts': str(tmp_path/'artifacts'),
        'cache': str(tmp_path/'cache')}, 'eval': {'paired_bootstrap_B': 40}})
    nd = many_nodes(4)
    mf = pd.DataFrame({'match_id': sorted(nd.match_id.unique()), 'fold': np.arange(4),
                       'outer': False, 'scheme': 'v1'})
    sp = {'matches': mf.to_dict(orient='list')}
    class XG:
        def predict(self, shots):
            return np.full(len(shots), .2)
    monkeypatch.setattr(stage_l2, 'fit_xg', lambda *args: XG())
    monkeypatch.setattr(stage_l2.l2_procedure.ValueProcedure, 'fit', lambda self, *a, **kw: self)
    monkeypatch.setattr(stage_l2.l2_procedure.ValueProcedure, 'value',
                        lambda self, validation, **kw: np.zeros(len(validation)))
    l0 = {'quality': {'status': 'ok'}, 'train': {'nodes': nd}, 'test': {'nodes': nd}}
    sel = {'status': 'ok', 'spec': {'features': ['dist', 'angle_vis']}}
    with pytest.raises(GateError):
        stage_l2.run_l2(cfg, l0, sp, sel, Registry(cfg), Cache(cfg), force=True)
    report = json.loads((cfg.paths.artifacts/'value_selected.failed.json').read_text())
    assert report['status'] == 'failed'
    assert report['all_candidates']['gbm']['difference_vs_current']['status'] == 'inconclusive'
    assert not list(cfg.paths.cache.glob('*/_DONE'))


def credit_nodes():
    nd = nodes().assign(valid_for_action_attribution=True, valid_for_rate_metrics=True,
                        valid_for_transition=True)
    nd.loc[1, 'team_id'] = nd.loc[1, 'away_team_id']
    nd.loc[1, 'actor_id'] = 'A1'
    nd['sigma_next'] = [-1, -1, 0]
    return nd


def test_credit_transfer_resolves_graph_ids_instead_of_row_positions():
    nd = credit_nodes()
    shifted = nd.copy()
    shifted['node'] += 100
    for col in ['next_node', 'prev_node', 'last_pass_node']:
        shifted[col] = np.where(shifted[col] >= 0, shifted[col]+100, -1)
    jr = l3_contrib.journal(shifted, np.array([1., 2., 3.]))
    # Source 0 splits +1: +.5 own, -.5 to opponent 1. Source 1 splits
    # +2: +1 own, -1 to opponent 2. Source 2 has its full shot delta +3.
    np.testing.assert_allclose(jr.credit, [.5, .5, 2.])


def test_credit_missing_successor_is_an_error_not_last_row_wraparound():
    nd = credit_nodes()
    nd.loc[0, 'next_node'] = -1
    with pytest.raises(ValueError, match='successor|Successor'):
        l3_contrib.journal(nd, np.array([1., 2., 3.]))


def test_missing_attribution_flag_never_authorizes_player_credit():
    nd = credit_nodes()
    nd['valid_for_action_attribution'] = [np.nan, True, True]
    jr = l3_contrib.journal(nd, np.array([1., 2., 3.]))
    assert not jr.valid_for_credit.iloc[0]
    assert np.isnan(jr.credit.iloc[0])


@pytest.mark.parametrize('exposure_flag', ['valid_for_rates', 'valid_for_rate_metrics'])
def test_toi_observed_comparison_sees_mismatching_excluded_stints(exposure_flag):
    st = pd.DataFrame({'match_id': ['M', 'M'], 'period': [1, 1], 'duration': [10., 90.],
        't_end': [10., 100.], 'valid_for_rates': [True, False],
        'valid_for_rate_metrics': [True, False], 'lineup_status': ['valid', 'mismatch'],
        'obs_events': [1, 1], 'n_home_sk': [5, 5], 'n_away_sk': [5, 5],
        'obs_n_home': [6, 7], 'obs_n_away': [6, 6], 'home_g_in': True, 'away_g_in': True})
    toi = pd.DataFrame({'match_id': ['M'], 'toi_s': [1000.], 'toi_valid_s': [100.], 'toi_rate_s': [100.]})
    q = l0_quality.toi_checks(st, toi)
    assert q['observed_all_reconstructed_to_observed_ratio'] == pytest.approx(1000./1090.)
    assert q['observed_mismatch_stints'] == 1
    assert q['observed_mismatch_stints_marked_rate_eligible'] == 0
    st.loc[1, exposure_flag] = True  # independently detect a corrupted exposure flag
    assert l0_quality.toi_checks(st, toi)['observed_mismatch_stints_marked_rate_eligible'] == 1


@pytest.mark.parametrize('student_error', [0., .01])
def test_distillation_zero_teacher_error_never_divides_by_zero(tmp_path, monkeypatch, student_error):
    cfg = load_config().with_overrides({'paths': {'artifacts': str(tmp_path/'artifacts'),
        'cache': str(tmp_path/'cache')}, 'eval': {'paired_bootstrap_B': 40}})
    nd = many_nodes(15)
    mf = pd.DataFrame({'match_id': sorted(nd.match_id.unique()), 'fold': np.arange(15)%5,
                       'outer': False, 'scheme': 'v1'})
    class XG:
        def predict(self, shots):
            return np.full(len(shots), .2)
    def perfect(validation):
        return np.where(validation.team_id == validation.home_team_id, .2, -.2)
    monkeypatch.setattr(stage_l2, 'fit_xg', lambda *args: XG())
    monkeypatch.setattr(stage_l2.l2_procedure.ValueProcedure, 'fit', lambda self, *a, **kw: self)
    monkeypatch.setattr(stage_l2.l2_procedure.ValueProcedure, 'value',
        lambda self, validation, **kw: np.zeros(len(validation)) if self.spec['kind']=='markov' else perfect(validation))
    monkeypatch.setattr(stage_l2.l2_procedure.TableDistillation, 'fit', lambda self, *a, **kw: self)
    monkeypatch.setattr(stage_l2.l2_procedure.TableDistillation, 'value',
                        lambda self, validation, **kw: perfect(validation)+student_error)
    # Stop before fitting/exporting the dummy procedures. The real acceptance
    # gate must be reached successfully even with a perfect teacher.
    def admitted(stage, gates, *args):
        assert all(gates.values())
        raise RuntimeError('admission reached')
    monkeypatch.setattr(stage_l2, 'enforce', admitted)
    reg = Registry(cfg)
    with pytest.raises(RuntimeError, match='admission reached'):
        stage_l2.run_l2(cfg, {'quality': {'status': 'ok'}, 'train': {'nodes': nd}, 'test': {'nodes': nd}},
            {'matches': mf.to_dict(orient='list')}, {'status': 'ok', 'spec': {'features': ['dist', 'angle_vis']}},
            reg, Cache(cfg), force=True)
    record = reg.get('value_distillation')
    assert record['status'] == ('accepted' if student_error == 0 else 'rejected')


def test_fqe_iterations_keep_the_supplied_rink_geometry(monkeypatch):
    from datamatch import io, l2_gbm
    from tests.test_l0 import GEO
    cfg = load_config().with_overrides({'custom_geometry_fixture': True})
    custom = {key: dict(value, goal_line_opp=20., post_half_width=.8) for key, value in GEO.items()}
    monkeypatch.setattr(io, 'rink_geometry',
                        lambda c: custom if c.get('custom_geometry_fixture') else GEO)
    class Model:
        def fit(self, X, y, **kwargs):
            assert (X.goal_line_x == 20.).all() and (X.post_half_width == .8).all()
            return self
        def predict(self, X):
            assert (X.goal_line_x == 20.).all() and (X.post_half_width == .8).all()
            return np.zeros(len(X))
    monkeypatch.setattr(l2_gbm.GBMV, '_model', lambda self: Model())
    l2_gbm.GBMV(cfg, 'fqe', opts={'fqe_iters': 1}).fit(nodes(), np.array([0., 0., .2]))


@pytest.mark.parametrize('change', ['state', 'reward_feature', 'identity'])
def test_l2_export_provenance_binds_test_inputs_even_when_development_is_unchanged(change):
    cfg = load_config()
    cache = Cache(cfg)
    dev, test = nodes(), nodes()
    selection = {'spec': {'features': ['dist', 'angle_vis']}}
    before = stage_l2.provenance(cfg, cache, dev, {'matches': {}}, selection, test)
    modified = test.copy()
    if change == 'state':
        modified['x'] += 1.
    elif change == 'reward_feature':
        modified['rebound'] = ~modified.rebound.astype(bool)
    else:
        modified['match_id'] = 'different_test_match'
    after = stage_l2.provenance(cfg, cache, dev, {'matches': {}}, selection, modified)
    assert before != after


@pytest.mark.parametrize('corruption', ['source_identity', 'nonfinite_value'])
def test_crossfit_cache_recomputes_invalid_fold_table(tmp_path, monkeypatch, corruption):
    from datamatch.cache import obj_hash
    cfg = load_config().with_overrides({'paths': {'cache': str(tmp_path/'cache')}})
    cache = Cache(cfg)
    nd = many_nodes(12)
    folds = np.repeat(np.arange(12)%3, 3)
    fits = []
    class XG:
        def predict(self, shots):
            return np.full(len(shots), .2)
    class V:
        def __init__(self, *args):
            pass
        def fit(self, train, *args, **kwargs):
            fits.append(set(train.match_id))
            return self
        def delta_value(self, validation, **kwargs):   # контракт процедуры: V для δ (без флага = value)
            return self.value(validation, **kwargs)
        def value(self, validation, **kwargs):
            return validation.source_node.to_numpy()*.001
    monkeypatch.setattr(stage_l2, 'fit_xg', lambda *args: XG())
    monkeypatch.setattr(stage_l2.l2_procedure, 'ValueProcedure', V)
    spec = {'kind': 'markov'}
    expected, _ = stage_l2.crossfit_values(cfg, nd, folds, {}, spec, cache, 'fixture')
    key = 'fixture_'+obj_hash({'xg': {}, 'value': spec})+'_fold0'
    saved = cache.load_df(key, 'values')
    if corruption == 'source_identity':
        saved.loc[0, 'source_node'] = 999999
    else:
        saved.loc[0, 'value'] = np.nan
    cache.save_df(key, 'values', saved)
    actual, _ = stage_l2.crossfit_values(cfg, nd, folds, {}, spec, cache, 'fixture')
    pd.testing.assert_frame_equal(expected, actual)
    assert len(fits) == 4  # only the damaged fold is recomputed
