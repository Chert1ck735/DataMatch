"""Regressions for the modified copy: censoring, pre-action time, folds and keys."""
import numpy as np
import pandas as pd
import pytest

from datamatch import l2_value
from datamatch.config import load_config
from datamatch.l2_procedure import ValueProcedure
from tests.test_l0 import HOME, lineup_start
from tests.test_l2 import many_nodes, nodes
from tests.test_revision_l0 import WH, _full
from tests.test_zone_starts import fo, touch


@pytest.mark.parametrize('gamma', [None, .9])
def test_all_prefixes_through_last_unknown_transition_are_censored(gamma):
    nd = nodes().assign(valid_for_transition=[False, False, True])
    # Both earlier horizons contain an unknown outgoing transition.
    np.testing.assert_allclose(l2_value.realized_return(nd, np.array([0., 0., .2]), gamma),
                               [np.nan, np.nan, .2], equal_nan=True)


def test_markov_time_bin_does_not_use_own_shot_timestamp():
    nd = nodes()
    model = l2_value.MarkovV(l2_value.Grid([-31, 31], [-16, 16]))
    nd.loc[2, 't_mono'] = 1100.
    before = model.time_index(nd)
    nd.loc[2, 't_mono'] = 1190.
    np.testing.assert_array_equal(before, model.time_index(nd))


def test_markov_post_gap_time_is_unknown():
    nd = nodes()
    nd.loc[2, ['prev_node', 't_ref', 't_ref_source']] = [-1, np.nan, 'none']
    model = l2_value.MarkovV(l2_value.Grid([-31, 31], [-16, 16]))
    assert model.time_index(nd)[2] == model.nt - 1


@pytest.mark.parametrize('kind', ['markov', 'gbm'])
@pytest.mark.parametrize('bad', ['partial_match', 'short', 'nan', 'one_fold', 'tiny_training_fold'])
def test_invalid_folds_are_rejected_instead_of_disabling_crossfit(kind, bad):
    nd = many_nodes(3)
    folds = np.repeat(np.arange(3), 3)
    if bad == 'partial_match':
        folds = np.tile(np.arange(3), 3)
    elif bad == 'short':
        folds = folds[:-1]
    elif bad == 'nan':
        folds = folds.astype(float)
        folds[0] = np.nan
    elif bad == 'one_fold':
        folds[:] = 0
    else:
        folds = np.repeat([0, 0, 1], 3)
    with pytest.raises(ValueError, match='[Ff]old'):
        ValueProcedure(load_config(), {'kind': kind}).fit(nd, np.tile([0., 0., .2], 3), folds=folds)


def test_local_segment_keys_use_physical_events_as_shared_reference():
    # The first physical segment has no usable node; independently factorizing
    # nodes would rename its successor from local segment 1 to local segment 0.
    missing = touch(HOME, 'H1', .5); missing['x'] = np.nan
    tabs, _ = _full(lineup_start(1) + [missing, WH(1.), fo(HOME, 2.),
                                     touch(HOME, 'H1', 3.), WH(4.)])
    reference = tabs['events'][['match_id', 'segment', 'segment_key']].drop_duplicates()
    for name in ['nodes', 'excluded']:
        table = tabs[name]
        joined = table.merge(reference, on=['match_id', 'segment'], suffixes=('', '_expected'),
                             validate='many_to_one')
        assert len(joined) == len(table)
        pd.testing.assert_series_equal(joined.segment_key, joined.segment_key_expected, check_names=False)
