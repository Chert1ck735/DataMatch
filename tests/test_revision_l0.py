"""Регрессионные сценарии ревизии L0 (P003, P004, P007, P008, P009, P014, P015).

Каждый сценарий сначала воспроизводит дефект проверенной версии (287aab06), затем фиксирует
исправленное поведение.
"""
import numpy as np
import pandas as pd
import pytest

from datamatch import l0_events, l0_quality, l0_shifts, stage_l0
from datamatch.config import load_config
from tests.test_l0 import AWAY, GAMES, GEO, GOALIES, HOME, _nodes, ev, lineup_start
from tests.test_zone_starts import fo, run, sh, sub, touch

WH = lambda t, period=1: dict(event_type="whistle", period=period, clock_s=t)


def _full(rows):
    """L0 одного матча через штатный путь (порции): узлы, составы, смены, качество."""
    cfg = load_config()
    allev = ev(rows)
    roster = pd.DataFrame({"match_id": ["M1", "M1"], "team_id": [HOME, AWAY], "player_id": ["HG", "AG"],
                           "slot": "G", "line": 1})
    tabs, q, _ = stage_l0.process_source(cfg, "train", GAMES, roster, GOALIES, GEO,
                                         load_chunk=lambda ids: allev, match_ids=["M1"])
    return tabs, q


def _node_stint_bounds(tabs):
    nd, st = tabs["nodes"], tabs["stints"].set_index(["match_id", "stint"])
    j = st.reindex(pd.MultiIndex.from_arrays([nd.match_id, nd.stint]))
    return nd, j


# --- P014: единая шкала времени для узлов и границ составов ------------------------------
@pytest.mark.parametrize("case", ["rollback_before_sub", "equal_time", "late_mark", "multi_sub", "period_end"])
def test_p014_node_time_inside_stint_bounds(case):
    rows = lineup_start(1) + [fo(HOME, 0.0)]
    if case == "rollback_before_sub":
        # действие на 50 с, затем замена с откатом часов на 45 с (по seq — после действия)
        rows += [touch(HOME, "H1", 50.0), sub(HOME, "H9", "H2", 45.0), touch(HOME, "H9", 46.0), WH(60.0)]
    elif case == "equal_time":
        rows += [touch(HOME, "H2", 30.0), sub(HOME, "H9", "H2", 30.0), touch(HOME, "H9", 30.0), WH(40.0)]
    elif case == "late_mark":
        # поздняя отметка одного события (длинный скачок), затем обычное время
        rows += [touch(HOME, "H1", 10.0), touch(HOME, "H1", 400.0), sub(HOME, "H9", "H2", 20.0),
                 touch(HOME, "H9", 25.0), WH(30.0)]
    elif case == "multi_sub":
        rows += [touch(HOME, "H1", 50.0), sub(HOME, "H9", "H2", 49.0), sub(AWAY, "A9", "A2", 49.0),
                 sub(HOME, "H8", "H3", 49.0), touch(HOME, "H9", 49.0), WH(55.0)]
    else:
        rows += [touch(HOME, "H1", 1199.0), sub(HOME, "H9", "H2", 1198.0),
                 dict(event_type="intermission", period=1, clock_s=1200.0)]
    tabs, q = _full(rows)
    nd, j = _node_stint_bounds(tabs)
    assert ((nd.t_mono.values >= j.t_start.values - 1e-9) & (nd.t_mono.values <= j.t_end.values + 1e-9)).all()
    assert q["time_scale"]["nodes_outside_stint_time"] == 0
    # сырое время сохраняется отдельно
    assert "clock" in nd and (nd.t_mono >= nd.clock - 1e-9).all()
    # смены игроков согласованы с TOI
    ps = tabs["player_shifts"]
    tsum = ps.groupby("player_id").duration_s.sum()
    toi = tabs["toi"].groupby("player_id").toi_s.sum()
    pd.testing.assert_series_equal(tsum.reindex(toi.index), toi, check_names=False)


def test_p014_membership_by_seq_with_zero_duration_stint():
    rows = lineup_start(1) + [fo(HOME, 0.0), touch(HOME, "H2", 30.0), sub(HOME, "H9", "H2", 30.0),
                              sub(HOME, "H2", "H9", 30.0), touch(HOME, "H2", 30.0), WH(40.0)]
    tabs, _ = _full(rows)
    st = tabs["stints"]
    assert (st.duration == 0).any()                       # промежуточный состав нулевой длительности
    nd = tabs["nodes"]
    # оба действия H2 привязаны к составам, где H2 на льду
    stj = st.set_index("stint")
    for s in nd[nd.actor_id == "H2"].stint:
        assert "H2" in stj.loc[s].home_sk
    assert nd.loc[nd.actor_id == "H2", "valid_for_action_attribution"].all()


def test_p004_zero_duration_keeps_action_attribution_but_not_toi():
    st = pd.DataFrame({"match_id": ["M1"], "stint": [0], "duration": [0.0],
                       "n_home_sk": [5], "n_away_sk": [5], "home_g_in": [True], "away_g_in": [True],
                       "home_sk": [["H1", "H2", "H3", "H4", "H5"]],
                       "away_sk": [["A1", "A2", "A3", "A4", "A5"]],
                       "home_g": ["HG"], "away_g": ["AG"]})
    ev = pd.DataFrame({"match_id": ["M1"], "stint": [0], "n_home": [6], "n_away": [6],
                       "event_type": ["shot_on_target"]})
    out = l0_shifts.annotate_lineup_status(st, ev)
    assert out.lineup_status.iat[0] == "zero_duration"
    assert not out.valid_for_rates.iat[0]
    assert out.valid_for_action_attribution.iat[0]


# --- P003: наблюдаемый конец периода не входит в состояние -------------------------------
def test_p003_regular_period_keeps_nominal_end_and_observes_actual_end():
    rows = lineup_start(1) + [fo(HOME, 0.0), touch(HOME, "H1", 1250.0),
                              dict(event_type="intermission", period=1, clock_s=1260.0)]
    nd, _ = _nodes(rows)
    regular = nd[nd.period == 1].iloc[-1]
    assert regular.period_end_observed == pytest.approx(1260.0)
    assert regular.period_end == pytest.approx(1200.0)
    assert regular.t_remain == pytest.approx(0.0)
    assert regular.period_end_source == "nominal_regulation"

    games_ot = GAMES.copy()
    games_ot.loc[0, "stage"] = "regular"
    rows_ot = lineup_start(4) + [fo(HOME, 0.0, t=0.0), touch(HOME, "H1", 250.0),
                                  dict(event_type="goal", actor_id="H1", actor_team_id=HOME, other_id="AG",
                                       other_team_id=AWAY, period=4, clock_s=250.0, x=20.0, y=0.0)]
    for r in rows_ot:
        r["period"] = 4
    e = ev(rows_ot)
    cfg, q = load_config(), {}
    e2, _, _ = l0_events.preprocess(e, games_ot, cfg, q)
    sc, st, _, _ = l0_shifts.reconstruct_all(e2, games_ot, goalies_global=GOALIES)
    e2["stint"] = sc
    ot, _, _ = l0_events.build_nodes(e2, games_ot, st, GEO, cfg, q)
    assert ot.period_end_source.eq("nominal_regulation").all()
    assert ot.period_end.eq(300.0).all()


def test_p003_future_period_end_does_not_change_earlier_state_features():
    def build(end):
        rows = lineup_start(1) + [fo(HOME, 0.0), touch(HOME, "H1", 120.0),
                                  dict(event_type="intermission", period=1, clock_s=end)]
        return _nodes(rows)[0]
    a, b = build(1200.0), build(1260.0)
    aa, bb = a[a.actor_id == "H1"].iloc[0], b[b.actor_id == "H1"].iloc[0]
    assert aa.t_remain == bb.t_remain == pytest.approx(1080.0)
    assert aa.period_end == bb.period_end == pytest.approx(1200.0)
    assert aa.period_end_observed != bb.period_end_observed
    from datamatch.l2_gbm import v_features
    pd.testing.assert_series_equal(v_features(a, False).loc[aa.node], v_features(b, False).loc[bb.node],
                                   check_names=False)


# --- P004/P005/P008/P010/P011: контракты качества и признаков ---------------------------
def test_p004_gap_nodes_are_explicitly_excluded_from_rates():
    rows = lineup_start(1) + [fo(HOME, 0.0), touch(HOME, "H1", 2.0), WH(10.0)]
    tabs, _ = _full(rows)
    st, nd = tabs["stints"].copy(), tabs["nodes"].copy()
    st[["lineup_status", "valid_for_rates", "valid_for_rate_metrics"]] = ["empty", False, False]
    nd["lineup_status"] = "empty"
    nd["valid_for_rates"] = False
    nd["valid_for_action_attribution"] = False
    nd["valid_for_rate_metrics"] = False
    c = l0_quality.coverage_checks(st, nd)
    assert c["invalid_player_nodes"] > 0
    assert c["invalid_player_nodes_marked_excluded"] == 0


def test_p004_nearly_uncovered_match_is_excluded_from_rate_metrics():
    st = pd.DataFrame({
        "match_id": ["M1", "M1"], "stint": [0, 1], "duration": [10.0, 1190.0], "period": [1, 1],
        "lineup_status": ["valid", "mismatch"], "valid_for_rates": [True, False],
        "obs_events": [1, 1], "n_home_sk": [5, 5], "n_away_sk": [5, 5],
        "obs_n_home": [6, 6], "obs_n_away": [6, 6], "home_g_in": [True, True], "away_g_in": [True, True],
        "home_team_id": [HOME, HOME], "away_team_id": [AWAY, AWAY],
        "home_sk": [["H1", "H2", "H3", "H4", "H5"]] * 2,
        "away_sk": [["A1", "A2", "A3", "A4", "A5"]] * 2,
        "home_g": ["HG", "HG"], "away_g": ["AG", "AG"],
    })
    nd = pd.DataFrame({"match_id": ["M1", "M1"], "stint": [0, 1], "node_kind": ["player", "player"],
                       "team_id": [HOME, HOME], "actor_id": ["H1", "H1"]})
    st, nd = l0_quality.apply_rate_eligibility(st, nd, 0.95)
    c = l0_quality.coverage_checks(st, nd)
    assert st.rate_match_coverage.iat[0] == pytest.approx(10 / 1200)
    assert not st.rate_match_eligible.any() and not nd.valid_for_rate_metrics.any()
    assert c["rate_match_coverage_below_threshold"] == 1
    assert c["low_coverage_player_nodes_marked_rate_eligible"] == 0


def test_p005_structure_audit_uses_global_node_ids_in_scheme_slice():
    nd = pd.DataFrame({
        "node": [10, 11, 12, 13], "prev_node": [-1, 10, 11, 12],
        "primary": ["touch", "goal", "pass_complete", "shot_on_target"],
        "actor_id": ["a", "a", "a", "b"], "other_id": [None, None, "b", None],
        "scheme": ["v1", "v1", "v1", "v1"], "action": ["hold", "shot", "pass", "shot"],
    })
    a = l0_quality.pre_shot_structure_audit(nd[nd.scheme == "v1"])
    assert a["by_outcome"]["goal"]["own_previous"] == pytest.approx(1.0)
    assert a["by_outcome"]["shot_on_target"]["not_partner_pass"] == pytest.approx(0.0)


def test_p005_value_feature_contract_excludes_outcome_sensitive_structure():
    from datamatch.l2_gbm import value_feature_contract

    c = value_feature_contract()
    excluded = set(c["excluded_pending_structure_audit"])
    assert {"arrival", "prev_action_h", "prev2_action_h", "prev_self_pass", "recent_entry"} <= excluded
    assert not (excluded & set(c["included"]))


def test_p008_actorless_known_team_is_a_team_node():
    rows = lineup_start(1) + [fo(HOME, 0.0),
                              dict(event_type="touch", actor_id=None, actor_team_id=HOME, period=1, clock_s=2.0,
                                   x=5.0, y=0.0),
                              touch(HOME, "H1", 3.0), WH(5.0)]
    nd, q = _nodes(rows)
    assert (nd.node_kind == "team").any()
    assert q["actorless_team_nodes_by_type"]["touch"] == 1


@pytest.mark.parametrize("gap_kind", ["unknown_team", "missing_xy"])
def test_uncertain_event_clears_entire_following_shot_history(gap_kind):
    gap = dict(event_type="touch", actor_id=None, actor_team_id=None, period=1, clock_s=2., x=1., y=0.)
    if gap_kind == "missing_xy":
        gap.update(actor_id="A1", actor_team_id=AWAY, x=np.nan, y=np.nan)
    rows = lineup_start(1) + [fo(HOME, 0.),
        dict(event_type="entry_success", actor_id="H1", actor_team_id=HOME, period=1, clock_s=.5),
        touch(HOME, "H1", 1.), gap, touch(HOME, "H1", 3.),
        dict(event_type="shot_on_target", actor_id="H1", actor_team_id=HOME, period=1, clock_s=4., x=20., y=0.), WH(5.)]
    nd, _ = _nodes(rows)
    after = nd[(nd.actor_id == "H1") & (nd.clock == 3.)].iloc[0]
    shot = nd[nd.is_shot].iloc[0]
    assert after.chain_gap_before and after.prev_node == -1
    assert after.t_ref_source == "none" and after.arrival == "loose_team"
    assert not after.recent_entry and not shot.recent_entry


def test_zone_labels_with_all_spatial_nodes_excluded():
    rows = lineup_start(1) + [
        dict(event_type="entry_success", actor_id="H1", actor_team_id=HOME, period=1, clock_s=1.),
        dict(event_type="touch", actor_id="H1", actor_team_id=HOME, period=1, clock_s=2., x=np.nan, y=np.nan)]
    nd, q = _nodes(rows)
    assert nd.empty and q["zone_labels_attached"] == 0.


def test_p010_scheme_gate_fails_when_v2_is_enabled_without_compatibility():
    cfg = load_config()
    cfg["data"]["train_schemes"] = ["v1", "v2"]
    js = {d: {"v1": 0.001, "v2": 0.05} for d in cfg["data"]["scheme_gate"]["distributions"]}
    nn = {d: {"v1": 2000, "v2": 2000} for d in cfg["data"]["scheme_gate"]["distributions"]}
    out = l0_quality.scheme_gate(js, nn, cfg)
    assert out["status"] == "failed" and not out["schemes"]["v2"]["all_pass"]


def test_p011_node_zone_uses_supplied_rink_geometry():
    rows = lineup_start(1) + [fo(HOME, 0.0), touch(HOME, "H1", 2.0), WH(5.0)]
    rows[-2]["x"] = 9.0
    cfg, q = load_config(), {}
    e2, _, _ = l0_events.preprocess(ev(rows), GAMES, cfg, q)
    sc, st, _, _ = l0_shifts.reconstruct_all(e2, GAMES, goalies_global=GOALIES)
    e2["stint"] = sc
    geo = {k: dict(v) for k, v in GEO.items()}
    geo["european_60x30"]["blue_line_opp"] = 10.0
    nd, _, _ = l0_events.build_nodes(e2, GAMES, st, geo, cfg, q)
    assert nd.loc[nd.actor_id == "H1", "zone"].iat[0] == "neutral"


# --- P015: гол завершает ожидание открывающего вбрасывания --------------------------------
def test_p015_goal_resolves_pending_before_stoppage():
    base = lineup_start(1) + [fo(HOME, 0.0), touch(HOME, "H1", 2.0), WH(10.0),
                              sub(HOME, "H9", "H2", 10.0),             # старт в остановке
                              dict(event_type="goal", actor_id="H1", actor_team_id=HOME, other_id="AG",
                                   other_team_id=AWAY, period=1, clock_s=15.0, x=20.0, y=0.0),
                              sub(HOME, "H8", "H3", 15.0)]             # старт после гола, в остановке
    with_fo = base + [fo(HOME, 0.0, t=15.0), touch(HOME, "H8", 17.0), WH(20.0)]
    without_fo = base + [WH(20.0)]
    a, _, _ = run(with_fo)
    b, _, _ = run(without_fo)
    h9a, h9b = sh(a, "H9"), sh(b, "H9")
    assert h9a.start_type == "unknown" and h9a.unknown_reason == "play_resumed_without_faceoff"
    assert h9b.start_type == h9a.start_type and h9b.unknown_reason == h9a.unknown_reason
    assert pd.isna(h9a.start_zone)
    h8 = sh(a, "H8")
    assert h8.start_type == "faceoff" and h8.start_zone == "neutral"


def test_p015_penalty_shot_goal_is_not_play_resumption():
    rows = lineup_start(1) + [fo(HOME, 0.0), touch(HOME, "H1", 2.0), WH(10.0),
                              sub(HOME, "H9", "H2", 10.0),
                              dict(event_type="so_attempt", actor_id="H1", actor_team_id=HOME, period=1, clock_s=10.0),
                              dict(event_type="goal", actor_id="H1", actor_team_id=HOME, other_id="AG",
                                   other_team_id=AWAY, period=1, clock_s=14.0, x=20.0, y=0.0),
                              fo(HOME, 0.0, t=14.0), touch(HOME, "H9", 16.0), WH(20.0)]
    ps, _, _ = run(rows)
    assert sh(ps, "H9").start_type == "faceoff"


# --- P007: собственная отметка первого узла не задаёт t_ref -------------------------------
@pytest.mark.parametrize("with_opener", [True, False])
def test_p007_first_node_shot_timestamp_invariance(with_opener):
    from datamatch import l1_xg

    def build(t_shot):
        rows = lineup_start(1) if with_opener else [r for r in lineup_start(1) if r["event_type"] != "period_start"]
        rows += [dict(event_type="entry_success", actor_id="H1", actor_team_id=HOME, period=1, clock_s=3.0),
                 dict(event_type="shot_missed", actor_id="H1", actor_team_id=HOME, other_id="AG",
                      other_team_id=AWAY, period=1, clock_s=t_shot, x=18.0, y=2.0),
                 WH(30.0)]
        nd, q = _nodes(rows)
        s = l1_xg.shot_table(nd)
        return s.iloc[0], nd
    a, nda = build(10.0)
    b, ndb = build(4.0)
    feats = [f for f in l1_xg.FEATURES if f != "opp_def"]
    pd.testing.assert_series_equal(a[feats], b[feats], check_names=False)
    if with_opener:
        assert nda.t_ref_source.iat[0] == "segment_opener"
    else:
        assert nda.t_ref_source.iat[0] == "none" and not bool(a.recent_entry)


# --- P009: восстановленное время не служит основанием слияния -----------------------------
def test_p009_imputed_clock_blocks_merge():
    rows = lineup_start(1) + [fo(HOME, 0.0),
                              dict(event_type="touch", actor_id="H1", actor_team_id=HOME, period=1, clock_s=10.0,
                                   x=5.0, y=0.0),
                              dict(event_type="pass_complete", actor_id="H1", actor_team_id=HOME, other_id="H2",
                                   other_team_id=HOME, period=1, clock_s=np.nan, x=5.0, y=0.0, x2=12.0, y2=3.0),
                              dict(event_type="touch", actor_id="H2", actor_team_id=HOME, period=1, clock_s=11.0,
                                   x=12.0, y=3.0),
                              # корректный дубль с наблюдаемым временем по-прежнему сливается
                              dict(event_type="pass_complete", actor_id="H2", actor_team_id=HOME, other_id="H3",
                                   other_team_id=HOME, period=1, clock_s=11.0, x=12.0, y=3.0, x2=15.0, y2=0.0),
                              WH(20.0)]
    nd, q = _nodes(rows)
    h1 = nd[nd.actor_id == "H1"]
    assert len(h1) == 2 and h1.clock_imputed.any()
    assert q["merge_blocked_by_imputed_clock"] == 1
    assert len(nd[nd.actor_id == "H2"]) == 1


def test_p009_no_imputation_no_change():
    rows = lineup_start(1) + [fo(HOME, 0.0), touch(HOME, "H1", 10.0),
                              dict(event_type="pass_complete", actor_id="H1", actor_team_id=HOME, other_id="H2",
                                   other_team_id=HOME, period=1, clock_s=10.0, x=0.0, y=0.0), WH(20.0)]
    nd, q = _nodes(rows)
    assert len(nd[nd.actor_id == "H1"]) == 1 and q.get("merge_blocked_by_imputed_clock", 0) == 0


def test_past_clock_rollback_cannot_revive_expired_rebound_or_entry():
    # All times are in seq order: 1s shot/entry, 10s clock anchor, then a 2s rollback.
    # At the trusted pre-shot monotone time 10s, the 1s history is already expired.
    shot = lambda t: dict(event_type='shot_on_target', actor_id='H1', actor_team_id=HOME,
        period=1, clock_s=t, x=20., y=0.)
    rows = lineup_start(1) + [fo(HOME, 0.), shot(1.),
        dict(event_type='entry_success', actor_id='H1', actor_team_id=HOME, period=1, clock_s=1.),
        touch(HOME, 'H2', 10.), touch(HOME, 'H1', 2.), shot(3.), WH(11.)]
    nd = _nodes(rows)[0]
    last = nd[nd.action.eq('shot')].iloc[-1]
    assert last.t_mono == 10.
    assert not last.rebound
    assert not last.recent_entry


def test_future_clock_imputation_cannot_change_earlier_state():
    from datamatch.l2_gbm import v_features
    rows = lineup_start(1, t=np.nan) + [dict(event_type='shot_on_target', actor_id='H1', actor_team_id=HOME,
             period=1, clock_s=np.nan, x=20., y=0.), touch(HOME, 'H2', 10.), WH(20.)]
    a = _nodes(rows)[0]
    changed = [dict(r) for r in rows]
    changed[-2]['clock_s'] = 15.
    b = _nodes(changed)[0]
    pd.testing.assert_series_equal(v_features(a, False).iloc[0], v_features(b, False).iloc[0])
