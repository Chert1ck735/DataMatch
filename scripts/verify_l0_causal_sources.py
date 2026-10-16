"""Independent causal clock/count source checks on real saved L0 tables."""
import json

import numpy as np
import pandas as pd

from datamatch import io, stage_l0
from datamatch.cache import Cache
from datamatch.config import load_config

# The semantic annotation set is stated independently of the production helper.
ANNOTATIONS = {'entry_success', 'exit_success', 'exit_controlled', 'feed_break',
               'assist_primary', 'assist_secondary', 'goal_error', 'scoring_chance'}


def verify():
    cfg = load_config()
    cache = Cache(cfg, io.data_version(cfg))
    key = cache.key('l0', ('l0',), code=stage_l0.L0_CODE)
    if not cache.done(key):
        raise RuntimeError('Current L0 has not completed')
    report = {'status': 'ok', 'cache_key': key}
    old = cache.root/'l0__ed7270ddcc__ed7ee6d4__1a4bb141e6a6__x'
    for source in ('train', 'test'):
        e = pd.read_parquet(cache.root/key/(source+'_events.parquet'), columns=[
            'match_id','seq','period','event_type','clock_s','clock_imputed',
            'clock_impute_kind','t_mono','n_home','n_away'])
        clock = e.clock_s.to_numpy()
        annotation = e.event_type.isin(ANNOTATIONS).to_numpy()
        actual_clock = ~annotation & ~e.clock_imputed.to_numpy()
        expected = np.zeros(len(e))
        expected_prev = {'n_home': np.full(len(e), np.nan), 'n_away': np.full(len(e), np.nan)}
        for positions in e.groupby(['match_id','period'], sort=False).indices.values():
            values = np.where(actual_clock[positions], clock[positions], -np.inf)
            cumulative = np.maximum.accumulate(values)
            expected[positions] = np.where(np.isfinite(cumulative), cumulative, 0.)
            # Every previous counter comes from the last physical event; a missing
            # counter is not replaced by a label or the current shot's outcome.
            physical = np.where(~annotation[positions], positions, -1)
            previous = np.r_[-1, np.maximum.accumulate(physical)[:-1]]
            known = previous >= 0
            for name in expected_prev:
                expected_prev[name][positions] = np.where(known,
                    e[name].to_numpy()[np.maximum(previous, 0)], np.nan)
        np.testing.assert_allclose(e.t_mono, expected, rtol=0, atol=1e-10)
        # Intermediate previous counters are not persisted in the events contract;
        # check the actual pre-shot counters published in the retained node table.
        shots = pd.read_parquet(cache.root/key/(source+'_nodes.parquet'), columns=[
            'match_id','first_seq','is_shot','n_home','n_away']).query('is_shot')
        lookup = pd.MultiIndex.from_frame(e[['match_id','seq']])
        positions = lookup.get_indexer(pd.MultiIndex.from_frame(shots[['match_id','first_seq']]))
        assert (positions >= 0).all()
        for name, value in expected_prev.items():
            np.testing.assert_allclose(shots[name], value[positions], rtol=0, atol=0, equal_nan=True)
        row = {'events': len(e), 'imputed_clocks': int(e.clock_imputed.sum()),
               'causal_clock_mismatches': 0, 'pre_counter_source_mismatches': 0}
        if old.exists():
            before = pd.read_parquet(old/(source+'_events.parquet'), columns=['match_id','seq','clock_s','t_mono'])
            pd.testing.assert_frame_equal(e[['match_id','seq']], before[['match_id','seq']])
            row['changed_clock_s'] = int((~np.isclose(e.clock_s, before.clock_s, equal_nan=True)).sum())
            row['changed_t_mono'] = int((~np.isclose(e.t_mono, before.t_mono, equal_nan=True)).sum())
            cols = ['node','match_id','first_seq','own_sk','opp_sk','mp','rebound','recent_entry','prev_pass_dy']
            nd = pd.read_parquet(cache.root/key/(source+'_nodes.parquet'), columns=cols)
            prior = pd.read_parquet(old/(source+'_nodes.parquet'), columns=cols)
            pd.testing.assert_frame_equal(nd[['node','match_id','first_seq']], prior[['node','match_id','first_seq']])
            row['changed_node_features'] = {name: int((~np.isclose(nd[name], prior[name], equal_nan=True)).sum())
                for name in ['own_sk','opp_sk','rebound','recent_entry','prev_pass_dy']}
            row['changed_node_features']['mp'] = int((nd.mp != prior.mp).sum())
        report[source] = row
    (cfg.paths.outputs/'l0_causal_source_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    verify()
