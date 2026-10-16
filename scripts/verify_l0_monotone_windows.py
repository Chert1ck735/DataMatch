"""Read-only confirmation that pre-action elapsed-time windows use one causal clock."""
import json

import numpy as np

from datamatch import io, stage_l0
from datamatch.cache import Cache
from datamatch.config import load_config


def main():
    cfg = load_config()
    cache = Cache(cfg, io.data_version(cfg))
    key = cache.key('l0', ('l0',), code=stage_l0.L0_CODE)
    if not cache.done(key):
        raise ValueError('Fresh L0 must complete first')
    report = {'cache_key': key, 'status': 'ok'}
    old = 'l0__ed7270ddcc__ed7ee6d4__2bdd05ffa3d9__x'
    cols = ['node', 'prev_node', 't_mono', 't_ref', 't_ref_source', 'dt_prev',
        'dt_last_own_shot', 'rebound', 'recent_entry', 'prev_pass_dy', 'prev_pass_cross']
    import pandas as pd
    for source in ['train', 'test']:
        nd = pd.read_parquet(cache.root/key/(source+'_nodes.parquet'), columns=cols)
        pos = pd.Index(nd.node).get_indexer(nd.prev_node)
        has = pos >= 0
        past = nd.t_mono.to_numpy()[pos[has]]
        np.testing.assert_allclose(nd.loc[has, 'dt_prev'], nd.loc[has, 't_mono']-past)
        np.testing.assert_allclose(nd.loc[has, 't_ref'], past)
        delta = nd.dt_last_own_shot.dropna().to_numpy()
        assert (delta >= 0).all()
        row = {'nodes': len(nd), 'trusted_predecessors': int(has.sum()),
            'negative_rebound_elapsed_times': int((delta < 0).sum())}
        if cache.done(old):
            before = pd.read_parquet(cache.root/old/(source+'_nodes.parquet'), columns=cols)
            assert before.node.tolist() == nd.node.tolist()
            row['changed_from_previous_cache'] = {c: int((~np.isclose(before[c].to_numpy(dtype=float),
                nd[c].to_numpy(dtype=float), equal_nan=True)).sum())
                for c in ['rebound','recent_entry','prev_pass_dy','prev_pass_cross']}
        report[source] = row
    (cfg.paths.outputs/'l0_monotone_window_verification.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
