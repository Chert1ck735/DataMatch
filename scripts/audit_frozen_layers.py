"""One frozen outer-block check; never used to choose or tune the model recipe."""
import hashlib
import json
from pathlib import Path

import numpy as np

from datamatch import cli, io, l1_xg, l2_procedure, l2_value, stage_l1, stage_l2
from datamatch.cache import Cache, obj_hash, save_json_with_provenance
from datamatch.config import load_config, setup_logging
from datamatch.gates import GateError, enforce, require_ok
from datamatch.node_graph import subset_nodes
from datamatch.splits import matches_for



def value_gates(prediction):
    """Блокирующие гейты V на внешнем блоке: только конечность значений."""
    return {'finite_value': bool(np.isfinite(prediction).all())}


def value_calibration_status(neutral_metrics):
    """Калибровка экспортируемых (нейтральных) V на внешнем блоке — диагностика ok/warn, не гейт.

    Решение 07.10.2026: блок просматривался прежними ревизиями и входит в итоговую подгонку, а
    блокирующая калибровка V проверяется на разработке (гейт L2). Провал здесь пишется в отчёт.
    """
    failed = [i for i, d in enumerate(neutral_metrics['deciles']) if not d['ok']]
    return {'status': 'ok' if neutral_metrics['calibration_all_deciles_ok'] else 'warn',
            'deciles_ok': int(neutral_metrics['deciles_ok']), 'failed_deciles': failed}


def outer_xg_sum_gate(shots, p, B, seed):
    """Σ xG / Σ голов на внешнем блоке: провал, только если 95%-интервал по бутстрэпу матчей не накрывает 1.

    То же правило, что приёма кандидатов L1 (решение 06.10.2026, P049): жёсткий порог ±5% на ~175
    матчах был сопоставим с выборочным шумом и отклонял идеальную модель в 10–15% случаев.
    """
    interval = l1_xg.xg_sum_interval(shots.match_id.to_numpy(), shots.goal.to_numpy(), np.asarray(p), B, seed)
    return stage_l1.xg_sum_acceptable(interval), interval

def main():
    cfg = load_config()
    setup_logging(cfg, 'frozen_audit')
    cache = Cache(cfg, io.data_version(cfg))
    l0, splits = cli.cmd_prepare(cfg, cache, False)
    l1 = json.loads((cfg.paths.artifacts/'xg_selected.json').read_text())
    l2 = json.loads((cfg.paths.artifacts/'value_selected.json').read_text())
    require_ok(l1, 'frozen L1'); require_ok(l2, 'frozen L2')
    s, _ = stage_l1.dev_shots(l0, splits, cfg['data']['train_schemes'])
    p1 = stage_l1.l1_provenance(cache, s, splits)
    dev = stage_l2.development_nodes(l0['train']['nodes'], splits, cfg['data']['train_schemes'])
    full = stage_l2.final_fit_nodes(l0['train']['nodes'], splits, cfg['data']['train_schemes'])
    p2 = stage_l2.provenance(cfg, cache, dev, splits, l1, l0['test']['nodes'], full)
    if p1 != l1['provenance'] or p2 != l2['provenance']:
        raise ValueError('Frozen selected procedures are stale')
    # Admission must precede even reuse of a past outer report: a new force
    # generation may retain the old JSON/model files while DONE is absent.
    key1 = cache.key('l1_dev_model', ('xg', 'data', 'eval', 'seed'), extra=obj_hash(p1), code=stage_l1.L1_CODE)
    key2 = cache.key('l2_dev_model', ('value', 'data', 'eval', 'seed'), extra=obj_hash(p2), code=stage_l2.L2_CODE)
    if l2['cache_key'] != key2 or not cache.done(key1) or not cache.done(key2):
        raise ValueError('Frozen layer caches are incomplete or stale')
    xg = cache.load_obj(key1, 'procedure')
    value = cache.load_obj(key2, 'procedure')
    if xg is None or value is None:
        raise ValueError('Frozen fitted models missing')
    if obj_hash(xg.spec) != obj_hash(l1['spec']) or obj_hash(value.spec) != obj_hash(l2['spec']):
        raise ValueError('Frozen fitted recipes disagree with selected descriptors')
    # Хэш самого скрипта: смена правил аудита не должна переиспользовать прежний отчёт.
    audit_code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    audit_prov = {'l1': p1, 'l2': p2, 'audit_code': audit_code,
        'recipes': obj_hash({'xg': l1['spec'], 'value': l2['spec']})}
    path = cfg.paths.outputs/'frozen_outer_audit.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['provenance'] == audit_prov:
            require_ok(saved, 'saved frozen outer audit')
            print(json.dumps(saved, ensure_ascii=False, indent=1)); return
    outer = subset_nodes(l0['train']['nodes'], l0['train']['nodes'].match_id.isin(
        matches_for(splits, 'outer', schemes=cfg['data']['train_schemes'])))
    shots = l1_xg.shot_table(outer)
    p = xg.predict(shots)
    metrics = l1_xg.metrics(shots.goal.to_numpy(), p, shots.blocked.to_numpy())
    # Both V targets and predictions use the already-frozen development xG.
    r = stage_l2.reward(outer, xg)
    G = l2_value.realized_return(outer, r)
    view, target, prediction = l2_procedure.evaluation_view(outer, G, value)
    rd = stage_l2.reward(dev, xg)
    gd = l2_value.realized_return(dev, rd)
    wd = l2_procedure.event_weights(dev)
    observed = np.isfinite(gd)
    fo = dev.action.eq('faceoff').to_numpy()
    constant = float(np.sum(wd*np.where(observed & ~fo, gd, 0.))/np.sum(wd*observed))
    mv = l2_value.eval_value_model(view, target, prediction, constant,
        cfg.seed, B=cfg['eval']['paired_bootstrap_B'])
    mirrored = l2_procedure.mirror_faceoff_nodes(outer, cfg)
    neutral = np.r_[value.value(outer, neutral=True), value.value(mirrored, neutral=True)]
    mn = l2_value.eval_value_model(view, target, neutral, constant,
        cfg.seed, B=cfg['eval']['paired_bootstrap_B'])
    xg_sum_ok, xg_sum_interval = outer_xg_sum_gate(shots, p, cfg['eval']['paired_bootstrap_B'], cfg.seed)
    metrics['xg_sum_interval'] = xg_sum_interval
    gates = {'xg_sum_interval_covers_1': xg_sum_ok, **value_gates(prediction)}
    result = {'status': 'pending', 'matches': int(outer.match_id.nunique()), 'xg': metrics,
        'value_diagnostic': mv, 'neutral_value_diagnostic': mn,
        'value_calibration': value_calibration_status(mn),
        'constant': {'value': constant, 'fit': 'development targets; balanced faceoff perspectives'},
        'value_context': 'conditional quality and neutral export diagnostics reported separately', 'gates': gates,
        'policy': 'frozen one-shot assessment, no retuning on outer outcomes',
        'historical_caveat': 'earlier versions examined this outer block; this is not a new untouched holdout',
        'provenance': audit_prov}
    enforce('frozen L0→L1→L2', gates, path, result)
    result['status'] = 'ok'
    save_json_with_provenance(path, result, audit_prov)
    print(json.dumps(result, ensure_ascii=False, indent=1))


def run() -> int:
    """Код выхода как у CLI: 2 — провален гейт качества."""
    try:
        main()
    except GateError as ex:
        print(f'остановлено гейтом: {ex}')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(run())
