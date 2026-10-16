"""L2 tournament, frozen selection, provenance and reusable fitted value model."""
import time

import numpy as np

from . import io, l1_xg, l2_gbm, l2_procedure, l2_value, stage_l1
from .cache import frame_hash, load_checked_json, obj_hash, save_json_with_provenance
from .config import get_logger
from .gates import enforce, require_ok
from .node_graph import subset_nodes
from .splits import matches_for, split_frame, splits_hash

L2_CODE = tuple(sorted(set(stage_l1.L1_CODE + ('l2_value', 'l2_gbm', 'l2_procedure', 'stage_l2', 'node_graph'))))
HASH_COLS = ['node', 'match_id', 'next_node', 'prev_node', 'segment', 'possession', 'team_id', 'opp_team_id',
    'action', 'is_goal', 'valid_for_transition', 'sigma_next', 'home_team_id', 'is_home', 'x', 'y', 'dist', 'angle_vis', 'mp', 'own_sk', 'opp_sk',
    'own_g', 'opp_g', 'score_diff', 'period', 'period_nominal', 't_mono', 't_ref', 't_ref_source',
    't_seg', 't_poss', 'n_in_poss', 'dt_prev', 'rink_type']


def reward(nodes, xg):
    r = np.zeros(len(nodes))
    sh = nodes.action.eq('shot').to_numpy()
    if sh.any():
        r[sh] = xg.predict(l1_xg.shot_table(nodes.loc[sh]))
    return r


def fit_xg(cfg, spec, nodes):
    return l1_xg.XGProcedure(spec, cfg.seed).fit(l1_xg.shot_table(nodes))


def provenance(cfg, cache, nodes, splits, selection, test_nodes=None, final_nodes=None):
    runtime, _ = stage_l1.run_fingerprint(cfg, {}, selection['spec'])
    return cache.provenance('l2', ('value', 'data', 'eval', 'seed'), L2_CODE,
        nodes=frame_hash(nodes, HASH_COLS),
        shots=frame_hash(l1_xg.shot_table(nodes), l1_xg.FEATURES + ['goal', 'match_id', 'opp_team_id']),
        source_nodes=frame_hash(nodes[['match_id']].assign(source_node=nodes.get('source_node', nodes.node))),
        test_nodes=None if test_nodes is None else frame_hash(test_nodes, HASH_COLS),
        test_shots=None if test_nodes is None else frame_hash(l1_xg.shot_table(test_nodes),
            l1_xg.FEATURES + ['goal', 'match_id', 'opp_team_id']),
        final_nodes=None if final_nodes is None else frame_hash(final_nodes, HASH_COLS),
        splits=splits_hash(splits),
        xg=obj_hash({'spec': selection['spec'], 'provenance': selection.get('provenance')}), runtime=runtime['id'])


def development_nodes(raw_nodes, splits, schemes):
    mask = raw_nodes.match_id.isin(matches_for(splits, 'dev', schemes=schemes))
    source = raw_nodes.get('source_node', raw_nodes.node).to_numpy()[mask]
    return subset_nodes(raw_nodes, mask).assign(source_node=source)


def final_fit_match_ids(splits, schemes):
    """Матчи финальной подгонки: весь допустимый train — разработка и внешний блок.

    Правило holdout (L7): выбор делается на разработке, внешний блок проверяется замороженной
    моделью, затем процедура переобучается на всём train и применяется к test.
    """
    return sorted(matches_for(splits, 'train_all', schemes=schemes))


def final_fit_nodes(raw_nodes, splits, schemes):
    mask = raw_nodes.match_id.isin(final_fit_match_ids(splits, schemes))
    source = raw_nodes.get('source_node', raw_nodes.node).to_numpy()[mask]
    return subset_nodes(raw_nodes, mask).assign(source_node=source)


def final_fit_folds(nodes, splits):
    """Целоматчевые фолды всего train: фолды разработки сохраняются, внешний блок — свой фолд."""
    frame = split_frame(splits).set_index('match_id')
    outer = frame.outer.astype(bool)
    dev_folds = frame.loc[~outer, 'fold'].astype(int)
    if dev_folds.empty:
        raise ValueError('Final fit requires development matches with folds')
    fold = frame.fold.astype(int).where(~outer, int(dev_folds.max()) + 1)
    mapped = nodes.match_id.map(fold)
    if mapped.isna().any():
        raise ValueError('Every final-fit node must belong to a split match')
    return mapped.to_numpy(dtype=int)


def checked_fold_table(saved, validation, fold):
    """Admit only the exact held-out identities and finite, coherent TD values."""
    import pandas as pd
    required = ['node', 'match_id', 'source_node', 'r', 'value', 'delta', 'fold', 'estimate_kind']
    if saved is None or not set(required).issubset(saved.columns) or len(saved) != len(validation):
        return None
    try:
        if saved.node.duplicated().any():
            return None
        saved = saved.set_index('node').loc[validation.node].reset_index()
        if not all(np.array_equal(saved[c], validation[c]) for c in ['node', 'match_id', 'source_node']):
            return None
        if not saved.fold.eq(fold).all() or not saved.estimate_kind.eq('match_cross_fit').all():
            return None
        r, v, delta = (saved[c].to_numpy(dtype=float) for c in ['r', 'value', 'delta'])
        if not np.isfinite(r).all() or not np.isfinite(v).all():
            return None
        if not np.allclose(delta, l2_value.td_delta(validation, r, v), rtol=0, atol=1e-12, equal_nan=True):
            return None
        return saved[required].reset_index(drop=True)
    except (KeyError, ValueError, TypeError, pd.errors.InvalidIndexError):
        return None


def crossfit_values(cfg, nodes, folds, xg_spec, value_spec, cache=None, key=None, force=False):
    """Refit the entire xG → V pipeline without each held-out match fold."""
    import pandas as pd
    if "source_node" not in nodes:
        nodes = nodes.assign(source_node=nodes.node.to_numpy())
    tables, proof = [], []
    for k in sorted(set(folds)):
        tr = folds != k
        train, validation = subset_nodes(nodes, tr), subset_nodes(nodes, ~tr)
        if set(train.match_id) & set(validation.match_id):
            raise ValueError('Cross-fit match overlap')
        recipe = obj_hash({'xg': xg_spec, 'value': value_spec})
        fold_key = f'{key}_{recipe}_fold{k}'
        saved = None if force or cache is None or not cache.done(fold_key) else cache.load_df(fold_key, 'values')
        saved = checked_fold_table(saved, validation, k)
        if saved is None:
            if cache is not None:
                cache.invalidate_done(fold_key)
            get_logger().info('L2 cross-fit fold %s: %d held-out matches', k, validation.match_id.nunique())
            xg = fit_xg(cfg, xg_spec, train)
            rt, rv = reward(train, xg), reward(validation, xg)
            proc = l2_procedure.ValueProcedure(cfg, value_spec, cfg.seed).fit(train, rt, xg_spec=xg_spec, folds=folds[tr])
            # value — V, по которой считается δ (при value.delta_from_raw — без калибровки);
            # value_calibrated — калиброванная V для отчёта и гейта (METHODOLOGY 3, поправка 08.10).
            vd, vv = proc.delta_value(validation, neutral=True), proc.value(validation, neutral=True)
            saved = validation[['node', 'match_id', 'source_node']].assign(r=rv, value=vd, value_calibrated=vv,
                delta=l2_value.td_delta(validation, rv, vd), fold=int(k), estimate_kind='match_cross_fit')
            saved = checked_fold_table(saved, validation, k)
            if saved is None:
                raise ValueError('Invalid held-out fold values; completion is blocked')
            if cache is not None:
                cache.save_df(fold_key, 'values', saved)
                cache.mark_done(fold_key)
        saved = saved.copy()
        saved['node'] = nodes.node.to_numpy()[~tr]  # restore original graph IDs after local remapping
        tables.append(saved)
        proof.append({'fold': int(k), 'training_matches': int(train.match_id.nunique()),
            'held_out_matches': int(validation.match_id.nunique()), 'overlap': 0,
            'training_matches_hash': obj_hash(sorted(set(train.match_id))),
            'held_out_matches_hash': obj_hash(sorted(set(validation.match_id))),
            'recipe_hash': recipe})
    table = pd.concat(tables, ignore_index=True).sort_values('node').reset_index(drop=True)
    if set(table.node) != set(nodes.node) or table.node.duplicated().any():
        raise ValueError('Cross-fit does not cover each development node exactly once')
    table['valid_delta'] = np.isfinite(table.delta)
    return table, proof


def calibration_kind(procedure):
    """Вид калибровки выбранной процедуры для отчёта: isotonic, affine или unknown."""
    calibration = getattr(procedure, "calibration", None)
    if calibration is None:
        return "unknown"
    return "affine" if isinstance(calibration, tuple) else getattr(calibration, "kind", "unknown")


def run_l2(cfg, l0, splits, selection, reg, cache, force=False):
    if cfg['value'].get('gamma', 1.) != 1.:
        raise ValueError('The baseline tournament targets gamma=1; discount is a separate estimand')
    require_ok(selection, 'L1 selection')
    if l0['quality']['status'] not in ('ok', 'stress'):
        raise ValueError('L0 is not admitted')
    schemes = tuple(cfg['data']['train_schemes'])
    nodes = development_nodes(l0['train']['nodes'], splits, schemes)
    full_nodes = final_fit_nodes(l0['train']['nodes'], splits, schemes)
    prov = provenance(cfg, cache, nodes, splits, selection, l0['test']['nodes'], full_nodes)
    f = cfg.paths.artifacts / 'value_selected.json'
    key = cache.key('l2_dev_model', ('value', 'data', 'eval', 'seed'), extra=obj_hash(prov), code=L2_CODE)
    if not force:
        result = load_checked_json(f, prov)
        required = ['train_values', 'test_values']
        complete = cache.done(key) and all((cache.root/key/(name+'.parquet')).exists() for name in required)
        if result is not None and complete:
            proc, xg = cache.load_obj(key, 'procedure_full'), cache.load_obj(key, 'xg_full')
            complete = (proc is not None and xg is not None and cache.load_obj(key, 'procedure') is not None and
                obj_hash(proc.spec) == obj_hash(result['spec']) and
                obj_hash(xg.spec) == obj_hash(selection['spec']))
        if result is not None and complete:
            require_ok(result, 'L2 selection')
            get_logger().info('L2: current selection loaded from cache %s', key)
            return result
    log = get_logger()
    reg.clear_layer('L2')
    folds = nodes.match_id.map(split_frame(splits).set_index('match_id').fold).to_numpy()
    tuning = list(cfg['value'].get('tuning_folds', [0, 1, 2]))
    comparison = sorted(set(folds)-set(tuning))
    if not comparison or len(tuning) < 2:
        raise ValueError('L2 requires disjoint training and comparison folds')
    tr = np.isin(folds, tuning)
    train, validation = subset_nodes(nodes, tr), subset_nodes(nodes, ~tr)
    assert not (set(train.match_id) & set(validation.match_id))
    xg = fit_xg(cfg, selection['spec'], train)
    rt, rv = reward(train, xg), reward(validation, xg)
    Gt, Gv = l2_value.realized_return(train, rt), l2_value.realized_return(validation, rv)
    specs = [{'kind': k, 'pipeline': 'fold_local_calibrated'} for k in ('markov', 'gbm', 'gbm_hist')]
    models, reports, predictions = {}, {}, {}
    B = int(cfg['eval']['paired_bootstrap_B'])
    # The constant sees training targets only, with faceoff perspectives balanced.
    fo = train.action.eq('faceoff').to_numpy()
    weights = l2_procedure.event_weights(train)
    valid = np.isfinite(Gt)
    const = float(np.sum(weights*np.where(valid & ~fo, Gt, 0.))/np.sum(weights*valid))
    current = 'markov'
    train_folds = folds[tr]
    for spec in specs:
        cid = spec['kind']
        start = time.time()
        log.info('L2 fitting %s; %d training and %d comparison matches', cid, train.match_id.nunique(), validation.match_id.nunique())
        proc = l2_procedure.ValueProcedure(cfg, spec, cfg.seed).fit(train, rt, xg_spec=selection['spec'], folds=train_folds)
        models[cid] = proc
        view, G, v = l2_procedure.evaluation_view(validation, Gv, proc)
        m = l2_value.eval_value_model(view, G, v, const, cfg.seed, B=B)
        predictions[cid], reports[cid] = v, m
        if cid == 'markov':
            status, reason = 'baseline', 'Markov baseline (unit event weights) with a nominal-time state'
        else:
            ci = l2_value.paired_value_difference(view, G, v, predictions[current], cfg.seed, B)
            m['difference_vs_current'] = ci
            accepted = l1_xg.bootstrap_estimated(ci) and ci['hi95'] < 0 and m['calibration_all_deciles_ok']
            status = 'accepted' if accepted else 'rejected'
            reason = ('held-out MSE improves beyond match bootstrap and calibration passes' if accepted else
                      ('insufficient independent matches for paired comparison' if not l1_xg.bootstrap_estimated(ci)
                       else 'no calibrated held-out gain'))
        reg.add('value_'+cid, 'L2', 'A', None if cid == 'markov' else 'value_'+current,
            spec, m, {'seconds': time.time()-start, 'training_folds': tuning,
                      'comparison_folds': comparison,
                      'training_calibration_folds': int(len(np.unique(train_folds)))}, status, reason)
        if status == 'accepted':
            current = cid
        r2_text = 'not_estimable' if m['r2'] is None else f"{m['r2']:.4f}"
        log.info('L2 %s: R2 %s, calibration %d/10, status %s', cid, r2_text, m['deciles_ok'], status)
    distilled = False
    if current != 'markov':
        table = l2_procedure.TableDistillation(cfg).fit(train, models[current])
        view, G, vt = l2_procedure.evaluation_view(validation, Gv, table)
        teacher = models[current]
        vn = np.r_[teacher.value(validation, neutral=True),
                   teacher.value(l2_procedure.mirror_faceoff_nodes(validation, cfg), neutral=True)]
        mt = l2_value.eval_value_model(view, G, vt, const, cfg.seed, B=B)
        mn = l2_value.eval_value_model(view, G, vn, const, cfg.seed, B=B)
        ratio = mt['mse']/mn['mse'] - 1. if mn['mse'] > 0 else (0. if mt['mse'] == 0 else None)
        distilled = ratio is not None and ratio < .01 and mt['calibration_all_deciles_ok']
        reg.add('value_distillation', 'L2', 'A', 'value_'+current,
            {'kind': 'table'}, dict(mt, mse_increase_vs_neutral_teacher=ratio), {},
            'accepted' if distilled else 'rejected',
            'positive error cannot approximate a zero-error teacher' if ratio is None
            else 'requires <1% MSE increase and all calibration deciles')
    else:
        reg.add('value_distillation', 'L2', 'A', None, {}, {}, {}, 'not_run', 'selected baseline is already a table')
    # Optional experiments need their own target/reliability evidence. No false success.
    for name, reason in [('fqe', 'acceptance requires held-out player-sum reliability in L3'),
        ('catalytic', 'conditional on documented winning-GBM artifacts'),
        ('ensemble', 'requires a separate frozen candidate protocol'),
        ('discount', 'a different estimand; not compared to gamma=1 targets'),
        ('own_possession', 'a different estimand; not compared to stop-horizon targets')]:
        reg.add('value_'+name, 'L2', 'B', None, {}, {}, {}, 'not_run', reason)
    chosen = reports[current]
    # Экспортируемые значения (δ в L3) нейтральны по силе команд: их калибровка — тоже гейт.
    # Для Markov сила не используется, нейтральные и условные прогнозы совпадают.
    view_n, G_n, v_neutral = l2_procedure.evaluation_view(validation, Gv, models[current], neutral=True)
    chosen_neutral = l2_value.eval_value_model(view_n, G_n, v_neutral, const, cfg.seed, B=B)
    gates = {'r2_above_constant': chosen['r2'] is not None and chosen['r2'] > 0,
             'calibrated_all_deciles': chosen['calibration_all_deciles_ok'],
             'calibrated_all_deciles_neutral': chosen_neutral['calibration_all_deciles_ok'],
             'finite_predictions': bool(np.isfinite(predictions[current]).all()),
             'disjoint_matches': not bool(set(train.match_id) & set(validation.match_id))}
    result = {'status': 'pending', 'selected': current, 'spec': next(s for s in specs if s['kind']==current),
        'metrics': chosen, 'metrics_neutral': chosen_neutral, 'all_candidates': reports, 'gate_l2': gates,
        'feature_contract': l2_gbm.value_feature_contract(),
        'validation_design': {'method': 'disjoint_fit_and_comparison', 'training_folds': tuning,
            'comparison_folds': comparison, 'outer_block_used': False,
            'reward_model': 'selected xG recipe refitted without comparison matches',
            'calibration': {'kind': calibration_kind(models[current]),
                            'fit': 'learned only on training match cross-fits; symmetric faceoffs'},
            'event_weights': 'unit per physical event (no 1/N); dependence handled by whole-match bootstrap'},
        'evaluation_context': 'training-mapped team strength for predictive quality; neutral strength in exported values',
        'validation_scope': 'development comparison; frozen outer assessment is separate',
        'horizon': 'confirmed stop; censored paths excluded', 'units': 'expected signed goals',
        'team_strength': 'cross-fitted pair difference; neutral=0 on application',
        'node_identity': {'node': 'reindexed graph ID within the fitted subset', 'source_node': 'original L0 train graph ID'},
        'cache_key': key, 'data_mode': l0['quality']['status']}
    # Диагностика провала идёт в отдельный файл: запись поверх f уничтожала бы ранее принятый
    # выбор L2, из которого L3 берёт δ, и восстановить его неоткуда.
    enforce('L2', gates, f.with_suffix('.failed.json'), result)
    # Отметка завершённости снимается только здесь: до гейта в этот ключ ничего не пишется,
    # поэтому провал оставляет прежнюю генерацию целой, а не полуобновлённой. Снятие до гейта
    # уничтожало готовый результат предыдущего принятого прогона с теми же входами.
    # Модель разработки — только для замороженного внешнего аудита (scripts/audit_frozen_layers.py).
    dev_xg = fit_xg(cfg, selection['spec'], nodes)
    dev_final = l2_procedure.ValueProcedure(cfg, result['spec'], cfg.seed).fit(
        nodes, reward(nodes, dev_xg), xg_spec=selection['spec'], folds=folds)
    # Итоговая процедура: тот же замороженный рецепт на всём допустимом train (L7, правило holdout).
    full_folds = final_fit_folds(full_nodes, splits)
    final_xg = fit_xg(cfg, selection['spec'], full_nodes)
    final = l2_procedure.ValueProcedure(cfg, result['spec'], cfg.seed).fit(
        full_nodes, reward(full_nodes, final_xg), xg_spec=selection['spec'], folds=full_folds)
    oof, proof = crossfit_values(cfg, full_nodes, full_folds, selection['spec'], result['spec'], cache, key, force)
    result['train_cross_fit'] = proof
    result['oof_scope'] = 'match-exclusive fitting over all allowed train matches, conditional on the frozen development-selected recipe; not nested selection or an independent performance estimate'
    result['final_fit'] = {'matches': int(full_nodes.match_id.nunique()),
                           'development_matches': int(nodes.match_id.nunique()),
                           'outer_block_in_final_fit': True,
                           'folds': sorted(int(k) for k in set(full_folds)),
                           'rule': 'selection on development; frozen outer audit uses the development fit; '
                                   'final procedure refitted on all allowed train and applied to test'}
    test = l0['test']['nodes']
    rtest = reward(test, final_xg)
    vtest = final.delta_value(test, neutral=True)
    vtest_cal = final.value(test, neutral=True)
    # Отметка снимается только перед первой записью: долгие итоговые подгонки выше не трогают
    # прежнюю принятую генерацию (выключение машины в этом окне её не уничтожает).
    cache.invalidate_done(key)
    if distilled:
        cache.save_obj(key, 'distilled_table', l2_procedure.TableDistillation(cfg).fit(full_nodes, final))
    result['distillation_available'] = distilled
    cache.save_obj(key, 'procedure', dev_final)
    cache.save_obj(key, 'xg', dev_xg)
    cache.save_obj(key, 'procedure_full', final)
    cache.save_obj(key, 'xg_full', final_xg)
    cache.save_df(key, 'train_values', oof)
    cache.save_df(key, 'test_values', test[['node', 'match_id']].assign(r=rtest, value=vtest, value_calibrated=vtest_cal,
        delta=l2_value.td_delta(test, rtest, vtest), estimate_kind='full_train_fit'))
    cache.mark_done(key)
    result['status'] = 'ok'
    save_json_with_provenance(f, result, prov)
    result['provenance'] = prov
    return result
