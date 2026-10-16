"""Whole-match graph selection with explicit ID-to-position remapping."""
import numpy as np
import pandas as pd


def subset_nodes(nodes, mask):
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != len(nodes):
        raise ValueError('Invalid node selection length')
    selected = pd.Series(mask).groupby(nodes.match_id.to_numpy()).agg(['any', 'all'])
    if (selected['any'] != selected['all']).any():
        raise ValueError('Node graphs must be selected as complete matches')
    sub = nodes.loc[mask].copy()
    ids = pd.Index(sub.node)
    for c in ['next_node', 'prev_node', 'last_pass_node']:
        if c not in sub:
            continue
        old = sub[c].to_numpy()
        mapped = ids.get_indexer(old)
        if ((old >= 0) & (mapped < 0)).any():
            raise ValueError('Unresolved graph reference')
        sub[c] = mapped
    sub['node'] = np.arange(len(sub))
    return sub.reset_index(drop=True)
