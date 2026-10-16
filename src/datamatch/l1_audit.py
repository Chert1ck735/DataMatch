"""Аудит происхождения признаков xG: независимость от отметки времени самого броска.

Отметка времени события броска связана с исходом (у голов она систематически позже),
поэтому признак, меняющийся при сдвиге ТОЛЬКО собственной отметки броска, использует
информацию после броска. Проверка: в каждом отрезке между остановками сдвигается clock_s
ровно одного броска (вариант «last» — последний бросок отрезка, сюда входят все голы;
вариант «first» — первый), порядок seq не меняется; узлы L0 строятся заново, и признаки
сравниваются только у сдвинутых бросков. Отдельные волны меняют не больше одного
броска каждого периода: t_mono переносит прошлое через границы отрезков. Так зависимость от отметок предыдущих бросков
(законная для последующего броска) не смешивается с зависимостью от собственной отметки.
Признак, изменившийся хотя бы у одного броска, считается зависимым и исключается из xG.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import l0_events, l1_xg
from .l0_time import add_time_columns


def _segments(e: pd.DataFrame) -> np.ndarray:
    """Номера отрезков между остановками — тем же правилом, что в l0_events.build_nodes."""
    et = e.event_type.values
    mid = e.match_id.values
    per = e.period.values
    bb = np.isin(et, list(l0_events.BREAK_BEFORE))
    ba = np.isin(et, list(l0_events.BREAK_AFTER))
    start = bb | np.r_[False, ba[:-1]] | np.r_[True, mid[1:] != mid[:-1]] | np.r_[True, per[1:] != per[:-1]]
    return np.cumsum(start)


def shot_time_perturbation(events: pd.DataFrame, games: pd.DataFrame, stints: pd.DataFrame, goal_x: dict, cfg,
                           matches, deltas=(1.5, 4.0), features=None) -> dict:
    features = [f for f in (features or l1_xg.FEATURES) if f != "opp_def"]
    e = events[events.match_id.isin(set(matches))].reset_index(drop=True)
    st = stints[stints.match_id.isin(set(matches))]

    rebuilds = 0
    def shot_feats(ev):
        nonlocal rebuilds
        rebuilds += 1
        # расчётная шкала времени пересчитывается по (возможно, возмущённому) clock_s
        ev = add_time_columns(ev.drop(columns=[c for c in ("t_mono", "t_behind_scale") if c in ev]))
        nd, _, _ = l0_events.build_nodes(ev, games, st, goal_x, cfg, {})
        s = l1_xg.shot_table(nd).set_index(["match_id", "first_seq"])
        return s[features]

    base = shot_feats(e)
    e0 = add_time_columns(e.drop(columns=[c for c in ("t_mono", "t_behind_scale") if c in e]))
    nd0, _, _ = l0_events.build_nodes(e0, games, st, goal_x, cfg, {})
    sh0 = nd0[nd0.action == "shot"]
    out = {"n_shots": int(len(base)), "n_matches": int(e.match_id.nunique()), "deltas": list(deltas), "changed": {}}
    changed_total = pd.Series(0, index=features)
    for which in ("last", "first"):
        pick = sh0.groupby("segment").tail(1) if which == "last" else sh0.groupby("segment").head(1)
        # Monotone clocks carry prior observations across segment boundaries.
        # Separate waves retain every selected shot, but never move another past
        # shot in its period. Otherwise legitimate past changes look like own-time leakage.
        wave = pick.groupby(["match_id", "period"], sort=False).cumcount()
        totals = {d: {f: 0 for f in features} for d in deltas}
        exclusions = {d: 0 for d in deltas}
        compared = {d: 0 for d in deltas}
        for w in sorted(wave.unique()):
            chosen = pick[wave == w]
            active = e[e.match_id.isin(chosen.match_id)].copy()
            seg_e = _segments(e)
            # Keep global original segment IDs for matching the selected source events.
            original_seg = pd.Series(seg_e, index=e.index).loc[active.index].to_numpy()
            node_ev = set(zip(chosen.match_id, chosen.actor_id, np.round(chosen.clock.values, 1), chosen.segment))
            ck = np.round(active.clock_s.values, 1)
            is_sh = active.event_type.isin(list(l0_events.SHOT_TYPES)).to_numpy()
            mask = is_sh & np.array([(m, a, c, g) in node_ev for m, a, c, g in
                zip(active.match_id, active.actor_id, ck, original_seg)])
            for d in deltas:
                ep = active.copy()
                ep.loc[mask, "clock_s"] += d
                moved = set(zip(active.match_id.to_numpy()[mask], active.actor_id.to_numpy()[mask],
                    np.round(ep.clock_s.to_numpy()[mask], 1), original_seg[mask]))
                others = set(zip(active.match_id.to_numpy()[is_sh & ~mask], active.actor_id.to_numpy()[is_sh & ~mask],
                    ck[is_sh & ~mask], original_seg[is_sh & ~mask]))
                coll = moved & others
                keep = [(m, q) for (m, q), a_, c_, g_ in zip(zip(chosen.match_id, chosen.first_seq),
                    chosen.actor_id, np.round(chosen.clock.values, 1)+d, chosen.segment)
                    if (m, a_, round(c_, 1), g_) not in coll]
                exclusions[d] += len(chosen)-len(keep)
                compared[d] += len(keep)
                idx = pd.MultiIndex.from_tuples(sorted(keep), names=["match_id", "first_seq"])
                pert, b0 = shot_feats(ep).reindex(idx), base.reindex(idx)
                for f in features:
                    a, b = b0[f].to_numpy(), pert[f].to_numpy()
                    diff = ~np.isclose(a.astype(float), b.astype(float), equal_nan=True) if a.dtype.kind in "fc" else a.astype(str) != b.astype(str)
                    totals[d][f] += int(diff.sum())
        for d in deltas:
            out["changed"][f"{which}:+{d}"] = totals[d]
            out[f"n_excluded_coincidence_{which}:+{d}"] = exclusions[d]
            out[f"n_perturbed_{which}:+{d}"] = compared[d]
            changed_total += pd.Series(totals[d])
        out[f"n_perturbed_{which}"] = compared[deltas[-1]] if len(deltas) else 0
        out[f"independent_period_waves_{which}"] = int(wave.max()+1) if len(wave) else 0
    out["time_dependent"] = sorted(changed_total[changed_total > 0].index.tolist())
    out["node_rebuilds"] = rebuilds+1
    out["perturbation_design"] = "one moved shot per match-period per wave; all first/last segment shots retained"
    out["invariant"] = sorted(changed_total[changed_total == 0].index.tolist())
    return out


def pre_shot_timing_check(nodes: pd.DataFrame, matches=None) -> dict:
    """Проверка доверенности t_ref: интервал между двумя узлами ДО броска не должен
    различаться у голов и прочих бросков (иначе и время предыдущих узлов связано с исходом).
    nodes — полная таблица узлов (prev_node — глобальные индексы); matches — отбор матчей."""
    from sklearn.metrics import roc_auc_score
    sel = nodes.action == "shot"
    if matches is not None:
        sel &= nodes.match_id.isin(set(matches))
    sh = nodes[sel]
    ids = pd.Index(nodes.node)
    pv = ids.get_indexer(sh.prev_node.to_numpy())
    ok = pv >= 0
    parent_ids = np.where(ok, nodes.prev_node.to_numpy()[np.clip(pv, 0, None)], -1)
    pv2 = ids.get_indexer(parent_ids)
    ok2 = ok & (pv2 >= 0)
    lag = np.where(ok2, nodes.clock.values[np.clip(pv, 0, None)] - nodes.clock.values[np.clip(pv2, 0, None)], np.nan)
    y = sh.is_goal.values.astype(int)
    m = ~np.isnan(lag)
    own = sh.dt_prev.values  # интервал до САМОГО броска (использует его отметку) — для сравнения
    m2 = ~np.isnan(own)
    def auc(values, mask):
        return float(roc_auc_score(y[mask], values[mask])) if len(np.unique(y[mask])) == 2 else None
    def mean(values, mask):
        return float(np.mean(values[mask])) if mask.any() else None
    a, b = auc(lag, m), auc(own, m2)
    return {"pre_shot_lag_auc": a,
            "pre_shot_lag_mean_goal": mean(lag, m & (y == 1)),
            "pre_shot_lag_mean_other": mean(lag, m & (y == 0)),
            "own_stamp_dt_auc": b,
            "own_stamp_dt_mean_goal": mean(own, m2 & (y == 1)),
            "own_stamp_dt_mean_other": mean(own, m2 & (y == 0)),
            "auc_status": "ok" if a is not None and b is not None else "not_estimable_single_class"}
