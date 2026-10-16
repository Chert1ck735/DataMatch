"""Independent physical-seq reference for match-local segment IDs across tables."""
import json

import numpy as np
import pandas as pd

from datamatch import io, stage_l0
from datamatch.cache import Cache
from datamatch.config import load_config


def main():
    cfg = load_config()
    cache = Cache(cfg, io.data_version(cfg))
    key = cache.key('l0', ('l0',), code=stage_l0.L0_CODE)
    assert cache.done(key)
    report = {'status': 'ok', 'cache_key': key}
    for source in ['train', 'test']:
        events = pd.read_parquet(cache.root/key/f'{source}_events.parquet',
                                 columns=['match_id', 'seq', 'segment', 'segment_local', 'segment_key'])
        first = events.groupby(['match_id', 'segment'], sort=False).seq.min().rename('first_seq').reset_index()
        first = first.sort_values(['match_id', 'first_seq'])
        first['expected_local'] = first.groupby('match_id', sort=False).cumcount()
        first['expected_key'] = first.match_id + ':' + first.expected_local.astype(str)
        counts = {}
        for name in ['events', 'nodes', 'excluded']:
            table = events if name == 'events' else pd.read_parquet(cache.root/key/f'{source}_{name}.parquet',
                columns=['match_id', 'segment', 'segment_local', 'segment_key'])
            joined = table.merge(first, on=['match_id', 'segment'], how='left', validate='many_to_one')
            assert joined.expected_local.notna().all()
            np.testing.assert_array_equal(joined.segment_local, joined.expected_local)
            np.testing.assert_array_equal(joined.segment_key, joined.expected_key)
            counts[name] = len(table)
        report[source] = dict(counts, segments=len(first), mismatches=0)
    (cfg.paths.outputs/'l0_local_keys_verification.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
