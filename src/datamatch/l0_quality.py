"""L0: отчёт качества данных и гейт L0."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .l0_events import SHOT_TYPES

IGNORE_NEXT = {"scoring_chance", "assist_primary", "assist_secondary", "goal_error", "penalty_return",
               "entry_success", "exit_success", "exit_controlled", "feed_break", "turnover"}


def js_divergence(p: pd.Series, q: pd.Series) -> float:
    """Дженсен–Шеннон (основание 2) между двумя распределениями частот."""
    idx = p.index.union(q.index)
    p = p.reindex(idx, fill_value=0).astype(float)
    q = q.reindex(idx, fill_value=0).astype(float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def next_event_dist(e: pd.DataFrame, of_types: set, by_scheme: bool = True) -> dict:
    """Распределение типа следующего события (без служебных меток и смен) после событий of_types."""
    x = e[(e.period > 0) & ~e.event_type.str.startswith("shift_") & ~e.event_type.isin(IGNORE_NEXT)]
    nxt = x.groupby("match_id", sort=False).event_type.shift(-1)
    m = x.event_type.isin(of_types)
    out = {}
    keys = x.scheme if by_scheme else pd.Series("all", index=x.index)
    for k in keys[m].unique():
        out[k] = nxt[m & (keys == k)].value_counts()
    out["all"] = nxt[m].value_counts()
    return out


def transition_checks(nd: pd.DataFrame) -> dict:
    """Таблица допустимых переходов на уровне узлов; противоречия -> «неклассифицированные»."""
    nxt = nd.next_node.values
    has_next = nxt >= 0
    sig = nd.sigma_next.values
    prim = nd.primary.values
    nxp = np.where(has_next, prim[np.clip(nxt, 0, None)], "")
    rules = {
        # точный пас, сразу за которым действие соперника без касания получателя
        "pass_complete_to_opponent": (prim == "pass_complete") & (sig == -1) & ~nd.self_pass.values,
        # отбор (steal) без смены владения: предыдущий узел той же команды
        "steal_without_change": nd.has_steal.values & (nd.arrival.values != "recovery") & (nd.prev_node.values >= 0)
                                & (nd.arrival.values != "faceoff"),
        # гол, за которым в том же отрезке есть действие
        "goal_not_terminal": (prim == "goal") & has_next,
    }
    n = len(nd)
    res = {k: {"n": int(v.sum()), "share": float(v.mean())} for k, v in rules.items()}
    any_bad = np.zeros(n, dtype=bool)
    for v in rules.values():
        any_bad |= v
    res["unclassified_share"] = float(any_bad.mean())
    outcomes = np.where(~nd.valid_for_transition, "unknown",
                        np.where(nd.terminal, "terminal", np.where(nd.sigma_next == 1, "same", "opp")))
    res["unknown_transitions"] = int((~nd.valid_for_transition).sum())
    res["unknown_marked_terminal"] = int((~nd.valid_for_transition & nd.terminal).sum())
    tab = pd.crosstab(nd.action, outcomes,
                      normalize="index").round(4)
    res["action_outcome_table"] = tab.to_dict(orient="index")
    return res


def toi_checks(stints: pd.DataFrame, toi: pd.DataFrame) -> dict:
    """TOI только в пригодных для rates отрезках (P004).

    Сравнение с ``10 * duration`` оставляем только как диагностику: при снятом вратаре
    оно законно бывает выше единицы. Гейт сверяет восстановленные полевые человеко-секунды
    с независимой наблюдаемой численностью на отрезках, где она есть.
    """
    pos = stints[(stints.period > 0) & (stints.duration > 0)].copy()
    valid = pos[pos.valid_for_rates]
    # player_toi уже суммирует toi_valid_s только на valid_for_rates; эта величина —
    # контроль того, что табличный знаменатель и набор игроков в отрезках совпадают.
    valid_from_stints = (valid.duration * (valid.n_home_sk + valid.n_away_sk)).groupby(valid.match_id).sum()
    valid_from_toi = toi.groupby("match_id").toi_valid_s.sum()
    valid_ratio = (valid_from_toi / valid_from_stints).replace([np.inf, -np.inf], np.nan).dropna()
    rate = pos[pos.valid_for_rate_metrics]
    rate_exposure = (rate.duration * (rate.n_home_sk + rate.n_away_sk)).groupby(rate.match_id).sum()
    rate_toi = toi.groupby("match_id").toi_rate_s.sum()
    rate_difference = rate_toi.subtract(rate_exposure, fill_value=0.)
    # Inspect observations BEFORE any trust/status exclusion. The policy permits
    # mismatches only when their exposure is excluded; coverage controls the match.
    all_observed = pos[(pos.obs_events > 0) & pos.obs_n_home.notna() & pos.obs_n_away.notna()]
    all_rec = (all_observed.duration * (all_observed.n_home_sk + all_observed.n_away_sk)).groupby(all_observed.match_id).sum()
    all_raw = (all_observed.duration * (all_observed.obs_n_home + all_observed.obs_n_away
        - all_observed.home_g_in.astype(int) - all_observed.away_g_in.astype(int))).groupby(all_observed.match_id).sum()
    mismatch = ((all_observed.n_home_sk + all_observed.home_g_in.astype(int) != all_observed.obs_n_home)
                | (all_observed.n_away_sk + all_observed.away_g_in.astype(int) != all_observed.obs_n_away))
    observed = valid[(valid.obs_events > 0) & (valid.lineup_status == "valid")]
    rec_obs = (observed.duration * (observed.n_home_sk + observed.n_away_sk)).groupby(observed.match_id).sum()
    raw_obs = (observed.duration * ((observed.obs_n_home - observed.home_g_in.astype(int))
                                    + (observed.obs_n_away - observed.away_g_in.astype(int)))).groupby(observed.match_id).sum()
    observed_ratio = (rec_obs / raw_obs).replace([np.inf, -np.inf], np.nan).dropna()
    # Старая метрика сохраняется для обратной совместимости отчёта, но не используется гейтом.
    dur = pos.groupby(["match_id", "period"]).t_end.max().groupby("match_id").sum()
    tot = toi.groupby("match_id").toi_s.sum()
    raw_ratio = (tot / (10 * dur)).dropna()
    eps = 1e-8
    return {
        "observed_all_reconstructed_to_observed_ratio": float(all_rec.sum()/all_raw.sum()) if all_raw.sum() else None,
        "observed_all_stints": len(all_observed),
        "observed_mismatch_stints": int(mismatch.sum()),
        "observed_mismatch_duration_s": float(all_observed.loc[mismatch, 'duration'].sum()),
        "observed_mismatch_stints_marked_rate_eligible": int((mismatch &
            (all_observed.valid_for_rates | all_observed.valid_for_rate_metrics)).sum()),
        "rate_toi_mismatch_matches": int((rate_difference.abs() > 1e-6).sum()),
        "rate_toi_skater_seconds": float(rate_toi.sum()),
        "toi_ratio_overall": float(tot.sum() / (10 * dur.sum())),
        "toi_ratio_q": {str(k): float(v) for k, v in raw_ratio.quantile([0, 0.05, 0.5, 0.95, 1]).items()},
        "matches_outside_95_100": int(((raw_ratio < 0.95) | (raw_ratio > 1.0)).sum()),
        "matches_outside_list": raw_ratio[(raw_ratio < 0.95) | (raw_ratio > 1.0)].round(4).to_dict(),
        "raw_10_skater_ratio_diagnostic": float(tot.sum() / (10 * dur.sum())),
        "valid_toi_to_reconstructed_exposure_ratio": float(valid_from_toi.sum() / valid_from_stints.sum())
        if valid_from_stints.sum() else None,
        "valid_toi_to_reconstructed_exposure_per_match_q": {str(k): float(v) for k, v in
                                                               valid_ratio.quantile([0, .05, .5, .95, 1]).items()},
        "valid_toi_to_reconstructed_exposure_matches_outside_bounds": int(
            ((valid_ratio < .95 - eps) | (valid_ratio > 1.0 + eps)).sum()),
        "observed_stint_reconstructed_to_observed_ratio": float(rec_obs.sum() / raw_obs.sum()) if raw_obs.sum() else None,
        "observed_stint_reconstructed_to_observed_per_match_q": {str(k): float(v) for k, v in
                                                                    observed_ratio.quantile([0, .05, .5, .95, 1]).items()},
        "observed_stint_reconstructed_to_observed_matches_outside_bounds": int(
            ((observed_ratio < .95 - eps) | (observed_ratio > 1.0 + eps)).sum()),
        "matches_without_observed_valid_stint": int(pos.match_id.nunique() - observed.match_id.nunique()),
        "legal_extra_skater_raw_ratio_gt_100": int((raw_ratio > 1.0 + eps).sum()),
        "legal_extra_skater_raw_ratio_matches": {str(k): float(v) for k, v in raw_ratio[raw_ratio > 1.0 + eps].items()},
        "match_duration_s": {"median": float(dur.median()), "min": float(dur.min()), "max": float(dur.max())},
        "skater_counts_time_share": {str(k): float(v) for k, v in
                                     (valid.groupby(["n_home_sk", "n_away_sk"]).duration.sum()
                                      / valid.duration.sum()).round(4).items() if v > 0.0005},
    }


def identity_checks(e: pd.DataFrame, stints: pd.DataFrame, games: pd.DataFrame) -> dict:
    """Исполнитель события с шайбой находится на льду (личность, а не только численность)."""
    puck = {"touch", "pass_complete", "pass_failed", "steal", "dump_in_success", "dump_in_failed",
            "clear_out_success", "clear_out_failed", "deke_success", "deke_failed"} | SHOT_TYPES
    x = e[e.event_type.isin(puck) & (e.period > 0) & e.actor_id.notna() & (e.stint >= 0)]
    st = stints.set_index(["match_id", "stint"])
    j = st.reindex(pd.MultiIndex.from_arrays([x.match_id, x.stint]))
    ish = (x.actor_team_id.values == x.match_id.map(games.set_index("match_id").home_team_id).values)
    hs, as_, hg, ag = j.home_sk.values, j.away_sk.values, j.home_g.values, j.away_g.values
    on = np.fromiter(((a in (h_ if ih else a_)) or (a == (hg_ if ih else ag_))
                      for a, ih, h_, a_, hg_, ag_ in zip(x.actor_id.values, ish, hs, as_, hg, ag)), bool, len(x))
    # численность: n_home/n_away против реконструкции
    rh = j.n_home_sk.values + j.home_g_in.values.astype(int)
    ra = j.n_away_sk.values + j.away_g_in.values.astype(int)
    ok_n = (rh == x.n_home.values) & (ra == x.n_away.values)
    if len(x) == 0:
        return {"actor_on_ice_share": 1.0, "n_events_checked": 0, "n_counts_agree_share": 1.0,
                "actor_not_on_ice_by_type": {}}
    return {"actor_on_ice_share": float(on.mean()), "n_events_checked": int(len(x)),
            "n_counts_agree_share": float(np.nanmean(ok_n)),
            "actor_not_on_ice_by_type": x.event_type[~on].value_counts().head(10).to_dict()}


def possession_checks(ps: pd.DataFrame, stints: pd.DataFrame) -> dict:
    """Владения: не длиннее матча и ТОЧНО покрывают каждый отрезок от первого действия до
    остановки (свисток, гол, конец периода). Второе условие ловит обрезание последнего
    владения временем последнего действия, которое первое не замечает."""
    dur = stints[stints.period > 0].groupby(["match_id", "period"]).t_end.max().groupby("match_id").sum()
    pd_ = ps.groupby("match_id").duration.sum()
    ratio = (pd_ / dur).dropna()
    seg = ps.groupby("segment").agg(t0=("t_start", "min"), stop=("seg_stop", "first"), covered=("duration", "sum"))
    gap = (seg.stop - seg.t0) - seg.covered
    ext = ps[ps.extended_to_stop]
    ext_gap = (ext.seg_stop - ext.last_action)
    return {"possession_sum_over_match_max": float(ratio.max()),
            "possession_sum_over_match_median": float(ratio.median()),
            "matches_possession_exceeds": int((ratio > 1.0 + 1e-9).sum()),
            "segments_not_tiled": int((gap.abs() > 1e-6).sum()),
            "max_abs_tiling_gap_s": float(gap.abs().max()) if len(gap) else 0.0,
            "last_possessions_extended_to_stop": int(len(ext)),
            "extension_median_s": float(ext_gap.median()) if len(ext) else 0.0,
            "n_possessions": int(len(ps)), "end_reason": ps.end_reason.value_counts(normalize=True).round(4).to_dict()}


def gate_l0(q: dict, cfg) -> dict:
    g = cfg["l0"]["gates"]
    res = {
        "unknown_transitions_not_terminal": q["transitions"]["unknown_marked_terminal"] == 0,
        "rate_toi_matches_exposure": q["toi"]["rate_toi_mismatch_matches"] == 0,
        "actor_attribution_confirmed": q["coverage"]["all"]["unconfirmed_actor_marked_attributable"] == 0,
        "lineup_attribution_trusted": q["coverage"]["all"]["untrusted_lineup_marked_attributable"] == 0,
        "exposure_lineup_structurally_possible":
            q["coverage"]["all"]["structurally_invalid_stints_marked_rate_eligible"] == 0,
        "unclassified_lt_2pct": q["transitions"]["unclassified_share"] < g["unclassified_transitions_max"],
        "toi_valid_exposure_matches": q["toi"]["valid_toi_to_reconstructed_exposure_ratio"] is not None
        and g["toi_ratio_min"] - 1e-8 <= q["toi"]["valid_toi_to_reconstructed_exposure_ratio"] <= g["toi_ratio_max"] + 1e-8
        and q["toi"]["valid_toi_to_reconstructed_exposure_matches_outside_bounds"] == 0,
        "observed_stint_counts_match": q["toi"].get("observed_mismatch_stints_marked_rate_eligible") == 0
        and q["toi"]["observed_stint_reconstructed_to_observed_ratio"] is not None
        and g["toi_ratio_min"] - 1e-8 <= q["toi"]["observed_stint_reconstructed_to_observed_ratio"] <= g["toi_ratio_max"] + 1e-8
        and q["toi"]["observed_stint_reconstructed_to_observed_matches_outside_bounds"] == 0,
        "gap_nodes_excluded_from_rates": q["coverage"]["all"]["invalid_player_nodes_marked_excluded"] == 0,
        "low_coverage_matches_excluded_from_rate_metrics":
            q["coverage"]["all"]["low_coverage_player_nodes_marked_rate_eligible"] == 0,
        "actor_on_ice_ge_98": q["identity"]["actor_on_ice_share"] >= g["actor_on_ice_min"],
        "possessions_le_match": q["possessions"]["matches_possession_exceeds"] == 0,
        "possessions_tile_segments_to_stop": q["possessions"]["segments_not_tiled"] == 0,
        "player_shifts_toi_match": q["player_shifts"]["toi_mismatch_player_matches"] == 0,
        "player_shifts_no_overlap": q["player_shifts"]["overlaps"] == 0,
        "player_shifts_bounds": q["player_shifts"]["bound_violations"] == 0,
        "player_shifts_ids_unique": q["player_shifts"]["duplicate_ids"] == 0,
        "player_shifts_faceoff_on_ice": q["player_shifts"]["faceoff_not_on_ice"] == 0,
        "player_shifts_zone_only_from_faceoff": (q["player_shifts"]["on_the_fly_with_zone"] == 0
                                                 and q["player_shifts"]["zone_without_faceoff"] == 0),
        "skater_actions_linked_to_shift": q["player_shifts"]["skater_nodes_linked_share"] >= g["actor_on_ice_min"],
    }
    return res


def apply_rate_eligibility(stints: pd.DataFrame, nodes: pd.DataFrame, coverage_min: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Маркирует матчи, недостаточные для on-ice/rate/RAPM-показателей (P004).

    xG не использует этот флаг: неполный состав не делает попытку броска недостоверной.
    Знаменатель требует надёжную положительную экспозицию и допуск матча.
    Числитель требует подтверждённую атрибуцию действия и тот же допуск матча,
    независимо от длительности stint.
    """
    st, nd = stints.copy(), nodes.copy()
    pos = st[st.duration > 0]
    total = pos.groupby("match_id").duration.sum()
    trusted = pos[pos.valid_for_rates].groupby("match_id").duration.sum()
    coverage = (trusted / total).reindex(st.match_id).fillna(0.0)
    st["rate_match_coverage"] = coverage.to_numpy()
    st["rate_match_eligible"] = st.rate_match_coverage >= coverage_min
    st["valid_for_rate_metrics"] = st.valid_for_rates & st.rate_match_eligible
    key = st.set_index(["match_id", "stint"])
    j = key.reindex(pd.MultiIndex.from_arrays([nd.match_id, nd.stint]))
    nd["rate_match_coverage"] = j.rate_match_coverage.fillna(0.0).to_numpy()
    nd["rate_match_eligible"] = j.rate_match_eligible.fillna(False).astype(bool).to_numpy()
    # Action numerator is independent of a stint's positive-duration exposure.
    # Full L0 calls this after actor/shift confirmation; absent confirmation fails closed.
    action_ok = nd.get("valid_for_action_attribution", pd.Series(False, index=nd.index)).fillna(False).astype(bool)
    nd["valid_for_rate_metrics"] = action_ok & nd.rate_match_eligible
    return st, nd


def score_checks(events: pd.DataFrame, games: pd.DataFrame) -> dict:
    """Итоговый счёт по событиям goal периодов 1–5 (включая штрафные броски) против games.csv."""
    gl = events[(events.event_type == "goal") & (events.period > 0)]
    hg = (gl.actor_team_id == gl.home_team_id).groupby(gl.match_id).sum()
    ag = (gl.actor_team_id == gl.away_team_id).groupby(gl.match_id).sum()
    gg = games.set_index("match_id")[["home_goals", "away_goals"]].dropna()
    hh = hg.reindex(gg.index).fillna(0)
    aa = ag.reindex(gg.index).fillna(0)
    ok = (hh == gg.home_goals) & (aa == gg.away_goals)
    so = events[(events.event_type == "goal") & (events.period == 0)].match_id.unique()
    return {"matches_with_score": int(len(gg)), "score_matches_events": int(ok.sum()),
            "mismatch_with_shootout": int((~ok & gg.index.isin(so)).sum()),
            "mismatch_without_shootout": int((~ok & ~gg.index.isin(so)).sum()),
            "penalty_shot_goals_periods_1_5": int(gl.penalty_shot_seg.astype(bool).sum())}


def player_shift_checks(ps: pd.DataFrame, toi: pd.DataFrame, nodes: pd.DataFrame) -> dict:
    """Индивидуальные смены: структурные инварианты (провал -> гейт) и диагностика стартов."""
    out = {"n_shifts": int(len(ps))}
    out["start_type"] = ps.start_type.value_counts(dropna=False).to_dict()
    known = ps.start_zone.notna()
    out["start_zone_known"] = ps.loc[known, "start_zone"].value_counts().to_dict()
    fo = ps.start_type == "faceoff"
    out["faceoff_starts_zone_unknown"] = int((fo & ~known).sum())
    out["unknown_reason"] = ps.unknown_reason.value_counts(dropna=True).to_dict()
    out["faceoff_share_known_zone"] = float(known[fo].mean()) if fo.any() else 0.0
    # инварианты
    bad_bounds = (ps.seq_start >= ps.seq_end) | (ps.t_end < ps.t_start - 1e-9)
    out["bound_violations"] = int(bad_bounds.sum())
    o = ps.sort_values(["match_id", "player_id", "seq_start"], kind="mergesort")
    same = (o.match_id.values[1:] == o.match_id.values[:-1]) & (o.player_id.values[1:] == o.player_id.values[:-1])
    out["overlaps"] = int((same & (o.seq_start.values[1:] < o.seq_end.values[:-1])).sum())
    out["duplicate_ids"] = int(ps.player_shift_id.duplicated().sum())
    tsum = ps.groupby(["match_id", "player_id"]).duration_s.sum()
    tt = toi.set_index(["match_id", "player_id"]).toi_s
    j = pd.concat([tsum.rename("shifts"), tt.rename("toi")], axis=1).fillna(0.0)
    out["toi_mismatch_player_matches"] = int(((j.shifts - j.toi).abs() > 1e-6).sum())
    fos = ps[fo]
    out["faceoff_not_on_ice"] = int(((fos.faceoff_seq < fos.seq_start) | (fos.faceoff_seq >= fos.seq_end)).sum())
    out["on_the_fly_with_zone"] = int(((ps.start_type == "on_the_fly") & known).sum())
    out["zone_without_faceoff"] = int((known & ~fo).sum())
    lk = nodes.shift_link.value_counts().to_dict() if "shift_link" in nodes else {}
    out["node_links"] = lk
    sk = (~nodes.shift_link.isin(["goalie", "team_node"])) if "shift_link" in nodes else pd.Series(False, index=nodes.index)
    out["skater_nodes_unlinked"] = int((sk & (nodes.shift_link == "no_shift")).sum()) if "shift_link" in nodes else 0
    out["skater_nodes_linked_share"] = float((nodes.shift_link[sk] == "linked").mean()) if sk.any() else 1.0
    return out


def coverage_checks(stints: pd.DataFrame, nodes: pd.DataFrame, matches=None) -> dict:
    """Покрытие потока и составов (P004) в заданной области подсчёта (matches).

    Время по статусам состава, действия по статусам, покрытие по матчам и независимая сверка:
    человеко-секунды полевых по восстановленным составам против наблюдаемой численности
    n_home/n_away игровых событий (за вычетом восстановленного вратаря) на отрезках с наблюдениями.
    """
    st = stints if matches is None else stints[stints.match_id.isin(set(matches))]
    nd = nodes if matches is None else nodes[nodes.match_id.isin(set(matches))]
    pos = st[st.duration > 0]
    total = float(pos.duration.sum())
    by = pos.groupby("lineup_status").duration.sum()
    stj = st.set_index(["match_id", "stint"]).lineup_status
    ns = stj.reindex(pd.MultiIndex.from_arrays([nd.match_id, nd.stint])).values
    known = nd.node_kind.values == "player" if "node_kind" in nd else np.ones(len(nd), bool)
    action_flag = (nd.valid_for_action_attribution.to_numpy(dtype=bool)
                   if "valid_for_action_attribution" in nd else np.isin(ns, ["valid", "unverified"]))
    # Recompute trust from raw stint structure/observations and concrete actor membership.
    # Output flags may themselves be corrupted: they are not evidence for this check.
    from .l0_shifts import structurally_possible_lineups
    structural = structurally_possible_lineups(st)
    observed_agree = ((st.obs_events == 0) |
        ((st.obs_n_home == st.n_home_sk + st.home_g_in.astype(int)) &
         (st.obs_n_away == st.n_away_sk + st.away_g_in.astype(int))))
    trusted = structural & observed_agree.to_numpy()
    sj = st.set_index(["match_id", "stint"])
    keys = pd.MultiIndex.from_arrays([nd.match_id, nd.stint])
    trusted_node = pd.Series(trusted, index=sj.index).reindex(keys).fillna(False).to_numpy(dtype=bool)
    joined = sj[["home_team_id", "away_team_id", "home_sk", "away_sk",
        "home_g", "away_g", "home_g_in", "away_g_in"]].reindex(keys)
    actor_trust = np.zeros(len(nd), dtype=bool)
    links = nd.get("shift_link", pd.Series("no_shift", index=nd.index)).to_numpy()
    rows = zip(nd.team_id, nd.actor_id, joined.home_team_id, joined.away_team_id,
               joined.home_sk, joined.away_sk, joined.home_g, joined.away_g,
               joined.home_g_in, joined.away_g_in)
    for i, (team, actor, home, away, hs, ass, hg, ag, hgi, agi) in enumerate(rows):
        if not trusted_node[i] or not known[i] or links[i] not in ("linked", "goalie"):
            continue
        if team == home or team == away:
            sk, goalie, goalie_in = (hs, hg, hgi) if team == home else (ass, ag, agi)
            actor_trust[i] = actor in sk if links[i] == "linked" else bool(goalie_in) and actor == goalie
    invalid_node = known & (~action_flag | ~(trusted_node & actor_trust))
    flag = action_flag
    rate_flag = nd.valid_for_rate_metrics.to_numpy(dtype=bool) if "valid_for_rate_metrics" in nd else flag
    per_match = pos.groupby("match_id").apply(
        lambda d: float(d.duration[d.valid_for_rates].sum() / d.duration.sum()) if d.duration.sum() > 0 else 1.0,
        include_groups=False)
    bad = pos[~pos.valid_for_rates]
    bad_m = bad.groupby("match_id").duration.sum()
    # Missing exposure is contiguous only within the same match/period. A trusted
    # positive-duration stint separates gaps; zero-duration stints add no exposure.
    ordered = pos.sort_values(["match_id", "period", "stint"])
    unreliable = ~ordered.valid_for_rates
    boundary = (ordered.match_id.ne(ordered.match_id.shift()) |
                ordered.period.ne(ordered.period.shift()) |
                unreliable.ne(unreliable.shift()))
    runs = ordered.loc[unreliable, ["match_id", "duration"]].copy()
    runs["run"] = boundary.cumsum()[unreliable].to_numpy()
    gaps = runs.groupby(["match_id", "run"]).duration.sum()
    longest_by_match = gaps.groupby(level="match_id").max()
    obs = pos[pos.obs_events > 0]
    rec_ps = float((obs.duration * (obs.n_home_sk + obs.n_away_sk)).sum())
    obs_ps = float((obs.duration * ((obs.obs_n_home - obs.home_g_in.astype(int))
                                    + (obs.obs_n_away - obs.away_g_in.astype(int)))).sum())
    raw_ratio = ((obs.duration * (obs.n_home_sk + obs.n_away_sk)).groupby(obs.match_id).sum()
                 / (obs.duration * ((obs.obs_n_home - obs.home_g_in.astype(int))
                                    + (obs.obs_n_away - obs.away_g_in.astype(int)))).groupby(obs.match_id).sum())
    low_match = pos.groupby("match_id").apply(
        lambda d: float(d.duration[d.valid_for_rates].sum() / d.duration.sum()) if d.duration.sum() else 0.0,
        include_groups=False)
    if "rate_match_eligible" in pos:
        eligible_by_match = pos.groupby("match_id").rate_match_eligible.first()
        low = ~eligible_by_match
    else:
        low = low_match < .95
    low_node = known & ~nd.rate_match_eligible.to_numpy(dtype=bool) if "rate_match_eligible" in nd else np.zeros(len(nd), bool)
    return {
        "unconfirmed_actor_marked_attributable": int((action_flag & known &
            ~actor_trust).sum()),
        "untrusted_lineup_marked_attributable": int((known & action_flag & ~trusted_node).sum()),
        "structurally_invalid_stints_marked_rate_eligible": int(
            (~structural & st.valid_for_rates.to_numpy(dtype=bool)).sum()),
        "n_matches": int(st.match_id.nunique()),
        "game_time_min": total / 60,
        "time_share_by_status": {k: float(v / total) for k, v in by.items()} if total > 0 else {},
        "time_min_by_status": {k: float(v / 60) for k, v in by.items()},
        "player_actions_by_stint_status": pd.Series(ns[known]).value_counts().to_dict(),
        "invalid_player_nodes": int(invalid_node.sum()),
        "invalid_player_nodes_marked_excluded": int((invalid_node & rate_flag).sum()),
        "zero_duration_player_nodes_action_attributable": int(
            (known & (ns == "zero_duration") & action_flag).sum()),
        "rate_match_coverage_min": float(low_match.min()) if len(low_match) else 1.0,
        "rate_match_coverage_below_threshold": int(low.sum()),
        "rate_match_coverage_below_threshold_share": float(low.mean()) if len(low) else 0.0,
        "low_coverage_player_nodes": int(low_node.sum()),
        "low_coverage_player_nodes_marked_rate_eligible": int((low_node & rate_flag).sum()),
        "valid_coverage_share_overall": float(pos.duration[pos.valid_for_rates].sum() / total) if total > 0 else 1.0,
        "valid_coverage_per_match_q": {str(k): float(v) for k, v in per_match.quantile([0, 0.05, 0.5]).items()},
        "matches_with_unreliable_gap_gt_30s": int((longest_by_match > 30).sum()),
        "matches_with_unreliable_time_gt_30s": int((bad_m > 30).sum()),
        "longest_unreliable_gap_s": float(gaps.max()) if len(gaps) else 0.0,
        "longest_unreliable_interval_s": float(bad.duration.max()) if len(bad) else 0.0,
        "observed_skater_seconds_ratio_reconstructed_over_observed": rec_ps / obs_ps if obs_ps > 0 else None,
        "observed_skater_seconds_ratio_before_exclusion_q": {str(k): float(v) for k, v in
            raw_ratio.replace([np.inf, -np.inf], np.nan).dropna().quantile([0, .05, .5, .95, 1]).items()},
        "observed_count_disagreement_matches_before_exclusion": int(
            ((raw_ratio < .95) | (raw_ratio > 1.05)).sum()),
    }


def pre_shot_structure_audit(nodes: pd.DataFrame) -> dict:
    """Описательный аудит P005 до обучения V/Q.

    Это не делает признак разрешённым: L2 должен явно объявить результат аудита в своём
    отчёте. Здесь сохраняется сравнение структуры до броска для всех типов исхода.
    """
    shots = nodes[nodes.primary.isin(SHOT_TYPES)].copy()
    if shots.empty:
        return {"n_shots": 0, "by_outcome": {}, "status": "no_shots"}
    prev = shots.prev_node.to_numpy()
    has_prev = prev >= 0
    # prev_node — глобальный номер, поэтому обращаемся через node, а не позицию в текущем
    # срезе: аудит можно запускать и для одной схемы train.
    prior = nodes.set_index("node").reindex(prev)
    prev_actor = prior.actor_id.to_numpy()
    prev_primary = prior.primary.to_numpy()
    prev_other = prior.other_id.to_numpy()
    own_previous = has_prev & (prev_actor == shots.actor_id.to_numpy())
    # Не всякое действие типа pass передаёт шайбу текущему бросающему: нужен успешный пас
    # и совпадение получателя. Именно это определение использует P005.
    partner_pass = has_prev & (prev_primary == "pass_complete") & (prev_other == shots.actor_id.to_numpy())
    out = shots.primary.copy()
    tab = pd.DataFrame({"outcome": out, "own_previous": own_previous, "not_partner_pass": ~partner_pass})
    agg = tab.groupby("outcome").agg(n=("outcome", "size"), own_previous=("own_previous", "mean"),
                                      not_partner_pass=("not_partner_pass", "mean"))
    return {"n_shots": int(len(shots)), "by_outcome": {str(k): {a: (int(v) if a == "n" else float(v))
             for a, v in row.items()} for k, row in agg.to_dict(orient="index").items()},
            "status": "diagnostic_required_before_value_features"}


def period_checks(periods: pd.DataFrame, matches=None) -> dict:
    """Периоды (P003): статусы по отношению к номинальной длительности и значения наблюдаемых
    концов длинных/усечённых периодов (порог — допуск точности отметки конца)."""
    p = periods if matches is None else periods[periods.match_id.isin(set(matches))]
    reg = p[p.period <= 3]
    return {
        "n_periods": int(len(p)),
        "status": p.status.value_counts().to_dict(),
        "regular_status": reg.status.value_counts().to_dict(),
        "overlong_regular_end_values": reg[reg.status == "overlong"].observed_end.round(0).value_counts().head(10).to_dict(),
        "overlong_regular_matches": int(reg[reg.status == "overlong"].match_id.nunique()),
        "truncated_regular_end_values": reg[reg.status == "truncated"].observed_end.round(1).tolist()[:20],
        "end_event_vs_observed_max_abs_diff_s": float((p.end_event_clock - p.last_clock).abs().max())
        if p.end_event_clock.notna().any() else None,
    }


def scheme_gate(js_nodes: dict, js_n: dict, cfg) -> dict:
    """Гейт сопоставимости схем разметки train со схемой test (P010).

    Для каждой схемы из data.train_schemes по каждому заранее заданному распределению
    (исход перехода и тип следующего действия после паса, броска и отбора) требуется
    JS ≤ max_js_node и не меньше min_transitions переходов. Порог задан заранее в
    конфигурации; метки исходов test не используются (сравниваются распределения событий).
    Схема, не прошедшая проверку, в основном обучении даёт статус failed; режим
    data.stress_mode = true разрешает её только как явно помеченный стресс-тест.
    """
    sg = cfg["data"]["scheme_gate"]
    thr, nmin = float(sg["max_js_node"]), int(sg["min_transitions"])
    schemes = list(cfg["data"]["train_schemes"])
    res = {"max_js_node": thr, "min_transitions": nmin, "train_schemes": schemes,
           "stress_mode": bool(cfg["data"].get("stress_mode", False)), "schemes": {}}
    all_ok = True
    for sch in schemes:
        r = {}
        ok = True
        for dist in sg["distributions"]:
            js = js_nodes.get(dist, {}).get(sch)
            n = js_n.get(dist, {}).get(sch, 0)
            passed = js is not None and n >= nmin and js <= thr
            r[dist] = {"js": js, "n": int(n), "pass": bool(passed),
                       "reason": None if passed else ("мало переходов" if n < nmin else "JS выше порога")}
            ok &= passed
        r["all_pass"] = bool(ok)
        res["schemes"][sch] = r
        all_ok &= ok
    res["status"] = "ok" if all_ok else ("stress" if res["stress_mode"] else "failed")
    return res
