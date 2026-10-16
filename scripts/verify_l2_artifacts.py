"""Independent L0 → saved L2 join, sign, horizon and cross-fit contract checks."""
import json

import numpy as np
import pandas as pd

from datamatch import io, l1_xg, stage_l0, stage_l2
from datamatch.cache import Cache, code_hash, frame_hash, obj_hash
from datamatch.config import load_config
from datamatch.gates import require_ok
from datamatch.splits import splits_hash


def check_table(nodes, values):
    assert values.node.tolist() == nodes.node.tolist()
    assert values.match_id.tolist() == nodes.match_id.tolist()
    r, v, delta = (values[c].to_numpy() for c in ['r', 'value', 'delta'])
    assert np.isfinite(r).all() and np.isfinite(v).all()
    shot = nodes.action.eq('shot').to_numpy()
    assert (r[~shot] == 0).all() and ((r[shot] > 0) & (r[shot] < 1)).all()
    nxt = pd.Index(nodes.node).get_indexer(nodes.next_node)
    has = nxt >= 0
    assert not ((nodes.next_node >= 0).to_numpy() & ~has).any()
    sign = np.where(nodes.team_id.to_numpy()[np.maximum(nxt,0)] == nodes.team_id.to_numpy(), 1., -1.)
    expected = r-v
    expected[has] += sign[has]*v[nxt[has]]
    known = nodes.valid_for_transition.to_numpy(dtype=bool)
    expected[~known] = np.nan
    np.testing.assert_allclose(delta, expected, rtol=0, atol=1e-10, equal_nan=True)
    if 'valid_delta' in values:
        assert np.array_equal(values.valid_delta, np.isfinite(delta))
    # Independent full-segment telescoping, without using the production TD implementation.
    tau = np.where(nodes.team_id == nodes.home_team_id, 1., -1.)
    segment = nodes.segment.to_numpy()
    table = pd.DataFrame({'segment': segment, 'known': known, 'lhs': tau*delta, 'reward': tau*r,
                          'start_value': tau*v})
    grouped = table.groupby('segment', sort=False)
    valid = grouped.known.all()
    lhs = grouped.lhs.sum()
    rhs = grouped.reward.sum()-grouped.start_value.first()
    err = (lhs-rhs).abs()[valid]
    assert (err < 1e-9).all()
    return {'nodes': len(nodes), 'matches': int(nodes.match_id.nunique()),
        'unknown_delta': int((~known).sum()), 'fully_observed_segments': int(valid.sum()),
        'telescoping_max_error': float(err.max()) if len(err) else None}


def check_test_values(procedure, test, tv):
    """`value` — V для δ (при value.delta_from_raw без калибровки), `value_calibrated` — калиброванная V."""
    np.testing.assert_allclose(procedure.delta_value(test, neutral=True), tv.value, rtol=0, atol=1e-10)
    if 'value_calibrated' in tv:
        np.testing.assert_allclose(procedure.value(test, neutral=True), tv.value_calibrated, rtol=0, atol=1e-10)


def main():
    cfg = load_config(); cache = Cache(cfg, io.data_version(cfg))
    selection = json.loads((cfg.paths.artifacts/'value_selected.json').read_text())
    require_ok(selection, 'L2 selection')
    prov = selection['provenance']
    assert prov['code'] == code_hash(*stage_l2.L2_CODE)
    assert prov['data'] == cache.data_hash
    assert prov['config'] == cfg.section_hash('value', 'data', 'eval', 'seed')
    splits = json.loads((cfg.paths.artifacts/'splits.json').read_text())
    assert prov['splits'] == splits_hash(splits)
    key = selection['cache_key']; assert cache.done(key)
    l0key = cache.key('l0', ('l0',), code=stage_l0.L0_CODE); assert cache.done(l0key)
    columns = ['node','match_id','next_node','prev_node','last_pass_node','team_id','home_team_id',
        'action','segment','valid_for_transition']
    raw = pd.read_parquet(cache.root/l0key/'train_nodes.parquet', columns=columns)
    # Итоговые OOF-δ — по всему допустимому train (разработка + внешний блок, решение 06.10.2026).
    dev = stage_l2.final_fit_nodes(raw, splits, cfg['data']['train_schemes'])
    values = cache.load_df(key, 'train_values')
    assert values.source_node.tolist() == dev.source_node.tolist()
    assert values.estimate_kind.eq('match_cross_fit').all()
    folds = stage_l2.final_fit_folds(dev, splits)
    assert np.array_equal(values.fold, folds)
    xg_selection = json.loads((cfg.paths.artifacts/'xg_selected.json').read_text())
    recipe = obj_hash({'xg': xg_selection['spec'], 'value': selection['spec']})
    for proof in selection['train_cross_fit']:
        assert proof['recipe_hash'] == recipe
        k = proof['fold']
        tr, va = set(dev.match_id[folds != k]), set(dev.match_id[folds == k])
        assert not tr & va
        assert proof['training_matches_hash'] == obj_hash(sorted(tr))
        assert proof['held_out_matches_hash'] == obj_hash(sorted(va))
    report = {'status': 'ok', 'cache_key': key, 'source_join': 'exact original L0 IDs',
        'train': check_table(dev, values), 'final_fit': selection['final_fit']}
    test = pd.read_parquet(cache.root/l0key/'test_nodes.parquet')
    assert prov['test_nodes'] == frame_hash(test, stage_l2.HASH_COLS)
    assert prov['test_shots'] == frame_hash(l1_xg.shot_table(test),
        l1_xg.FEATURES + ['goal', 'match_id', 'opp_team_id'])
    tv = cache.load_df(key, 'test_values')
    report['test'] = check_table(test, tv)
    procedure = cache.load_obj(key, 'procedure_full')
    check_test_values(procedure, test, tv)
    fo = test[test.action.eq('faceoff')]
    mirror = stage_l2.l2_procedure.mirror_faceoff_nodes(fo, cfg)
    np.testing.assert_allclose(procedure.value(fo, neutral=True), -procedure.value(mirror, neutral=True), atol=1e-10)
    report['faceoff_antisymmetry_verified'] = len(fo)
    (cfg.paths.outputs/'l2_artifact_verification.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
