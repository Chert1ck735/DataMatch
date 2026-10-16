"""Contracts from the second L0 audit: identity, rate exposure and censored paths."""
import json

import numpy as np
import pandas as pd
import pytest

from datamatch import cli, l0_events, l0_quality, splits, stage_l0
from datamatch.config import load_config
from datamatch.l2_value import Grid, MarkovV, realized_return, td_delta
from tests.test_l0 import AWAY, HOME, _nodes, lineup_start
from tests.test_revision_l0 import WH, _full
from tests.test_zone_starts import fo, sub, touch


def test_zero_duration_shot_keeps_rate_numerator():
    rows = lineup_start(1) + [fo(HOME, 0.), touch(HOME, "H1", 1.),
        sub(HOME, "H9", "H2", 30.),
        dict(event_type="goal", actor_id="H9", actor_team_id=HOME, period=1, clock_s=30., x=20., y=0.),
        sub(HOME, "H2", "H9", 30.), WH(40.)]
    tabs, _ = _full(rows)
    shot = tabs["nodes"][tabs["nodes"].is_shot].iloc[0]
    assert tabs["stints"].set_index("stint").loc[shot.stint, "duration"] == 0.
    assert shot.shift_link == "linked" and shot.valid_for_action_attribution
    assert shot.rate_match_eligible and shot.valid_for_rate_metrics
    assert not shot.valid_for_rates  # zero seconds of exposure
    assert tabs["toi"].toi_rate_s.sum() == pytest.approx(400.)


def test_rate_toi_excludes_low_coverage_match_but_preserves_diagnostics():
    tabs, _ = _full(lineup_start(1) + [fo(HOME, 0.), touch(HOME, "H1", 2.), WH(10.)])
    st = tabs["stints"].copy()
    st["duration"] = np.where(st.duration > 0, 10., 0.)
    bad = st[st.duration > 0].iloc[[0]].copy()
    bad["stint"] = st.stint.max() + 1
    bad["duration"], bad["valid_for_rates"] = 1190., False
    from datamatch.l0_shifts import player_toi
    st, nd = l0_quality.apply_rate_eligibility(pd.concat([st, bad]), tabs["nodes"], .95)
    toi = player_toi(st)
    assert toi.toi_valid_s.sum() == pytest.approx(100.)
    assert toi.toi_rate_s.sum() == 0.
    assert not nd.valid_for_rate_metrics.any()


@pytest.mark.parametrize("actor,kind,allowed", [("GHOST", "no_shift", False),
    ("H1", "linked", True), ("HG", "goalie", True), (None, "team_node", False)])
def test_action_attribution_requires_confirmed_actor(actor, kind, allowed):
    r = touch(HOME, actor, 2.)
    tabs, _ = _full(lineup_start(1) + [fo(HOME, 0.), r, WH(10.)])
    n = tabs["nodes"][tabs["nodes"].clock == 2.].iloc[0]
    assert n.shift_link == kind
    assert n.valid_for_lineup_attribution  # team state remains usable
    assert bool(n.valid_for_action_attribution) == allowed
    assert bool(n.valid_for_rate_metrics) == allowed


def _gap_rows(equal_time=False, tail=False, missing_xy=False):
    t = 1. if equal_time else 2.
    gap = dict(event_type="touch", actor_id=None, actor_team_id=None, period=1, clock_s=t, x=1., y=0.)
    if missing_xy:
        gap.update(actor_id="A1", actor_team_id=AWAY, x=np.nan, y=np.nan)
    rows = lineup_start(1) + [fo(HOME, 0.), touch(HOME, "H1", 1.), gap]
    if not tail:
        rows += [touch(HOME, "H1", 1. if equal_time else 3.),
                 dict(event_type="shot_on_target", actor_id="H1", actor_team_id=HOME,
                      period=1, clock_s=4., x=20., y=0.)]
    return rows + [WH(5.)]


def test_unknown_event_at_equal_time_prevents_merging_and_carry():
    nd, _ = _nodes(_gap_rows(equal_time=True))
    touches = nd[(nd.actor_id == "H1") & (nd.clock == 1.)]
    assert len(touches) == 2
    a, b = touches.iloc[0], touches.iloc[1]
    assert a.next_node == -1 and a.action == "hold"
    assert b.prev_node == -1 and b.chain_gap_before
    assert a.possession != b.possession


@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("missing_xy", [False, True])
def test_observation_gap_is_censored_not_terminal(tail, missing_xy):
    nd, _ = _nodes(_gap_rows(tail=tail, missing_xy=missing_xy))
    a = nd[(nd.actor_id == "H1") & (nd.clock == 1.)].iloc[0]
    assert a.next_node == -1
    assert not a.terminal and a.transition_status == "unknown"
    assert not a.valid_for_transition
    ps = l0_events.build_possessions(nd)
    assert ps.set_index("possession").loc[a.possession, "end_reason"] == "observation_gap"
    assert not ps.set_index("possession").loc[a.possession, "duration_observed"]
    if not tail:
        last = nd.iloc[-1]
        assert last.terminal and last.transition_status == "terminal" and last.valid_for_transition
        labels = stage_l0._node_next_label(nd)
        assert labels.iloc[int(a.node)] == "UNKNOWN" and labels.iloc[-1] == "STOP"


def _censored_nodes():
    # Observed H -> H, unknown continuation, then separately observed H terminal shot.
    return pd.DataFrame({"node": [0, 1, 2], "match_id": ["M"] * 3,
        "team_id": ["H"] * 3, "home_team_id": ["H"] * 3, "segment": [1] * 3,
        "next_node": [1, -1, -1], "sigma_next": [1, 0, 0], "clock": [1., 2., 3.],
        "terminal": [False, False, True], "transition_status": ["observed", "unknown", "terminal"],
        "valid_for_transition": [True, False, True], "action": ["hold", "hold", "shot"],
        "arrival": ["continuation"] * 3, "mp": ["EV5"] * 3, "x": [0.] * 3, "y": [0.] * 3})


@pytest.mark.parametrize("gamma", [None, .99])
def test_value_targets_and_td_do_not_treat_unknown_as_zero(gamma):
    nd, r = _censored_nodes(), np.array([0., 0., .2])
    G = realized_return(nd, r, gamma)
    assert np.isnan(G[:2]).all() and G[2] == pytest.approx(.2)
    delta = td_delta(nd, r, np.array([.1, .1, .2]), gamma)
    assert np.isfinite(delta[0]) and np.isnan(delta[1]) and delta[2] == pytest.approx(0.)
    # Boundary consistency on the observable suffix.
    assert td_delta(nd, r, G, gamma)[2] == pytest.approx(0.)


def test_markov_excludes_unknown_transition_from_state_exposure():
    nd = _censored_nodes().iloc[1:].reset_index(drop=True)
    nd["node"] = [0, 1]
    model = MarkovV(Grid([-1., 1.], [-1., 1.]), kappa=1.).fit(nd, np.array([0., .2]))
    state = model.state(nd)[0]
    assert model.n_state[state] == 1.  # one observed transition, unit event weight (no 1/N)
    assert model.R[state] == pytest.approx(.2)
    assert model.value(nd)[1] == pytest.approx(.2)


@pytest.mark.parametrize("initial,stress,schemes,expected", [
    ("ok", True, ["v1", "v2"], "stress"), ("stress", False, ["v1"], "ok")])
def test_cached_status_transition_reaches_final_cli_json(tmp_path, monkeypatch, initial, stress, schemes, expected):
    cfg = load_config().with_overrides({"paths": {"outputs": str(tmp_path)},
                                       "data": {"stress_mode": stress, "train_schemes": schemes}})
    ds = cfg["data"]["scheme_gate"]["distributions"]
    res = {"quality": {"status": initial, "js_next_node": {d: {"v1": .001, "v2": .05} for d in ds},
        "js_next_node_n": {d: {"v1": 2000, "v2": 2000} for d in ds},
        "gate_l0_all": {"train:scheme_compatibility": True}},
        "train": {"nodes": pd.DataFrame({"match_id": ["M"], "scheme": ["v1"]}),
                  "games": pd.DataFrame({"match_id": ["M"]})},
        "test": {"games": pd.DataFrame({"match_id": ["T"]})}}
    def cached(*_args, **_kw):
        stage_l0._validate_cached_scheme_gate(cfg, res)
        return res
    monkeypatch.setattr(stage_l0, "run_l0", cached)
    monkeypatch.setattr(splits, "splits_provenance", lambda *_a, **_kw: {})
    monkeypatch.setattr(splits, "load_splits", lambda *_a, **_kw: {"matches": {}, "n_folds": 5})
    cli.cmd_prepare(cfg, None, False)
    assert res["quality"]["status"] == expected
    report = json.loads((tmp_path / "data_quality.json").read_text())
    assert report["status"] == report["scheme_gate"]["status"] == expected


def test_l3_excludes_unknown_credit_and_uses_rate_denominator():
    from datamatch import l3_contrib, stack
    nd = _censored_nodes()
    nd["actor_id"] = ["P", "P", "P"]
    nd["period"], nd["is_goal"] = 1, False
    nd["valid_for_action_attribution"], nd["valid_for_rate_metrics"] = True, True
    d = td_delta(nd, np.array([0., 0., .2]), np.array([.1, .1, .2]))
    jr = l3_contrib.journal(nd, d)
    assert not jr.valid_for_credit.iloc[1] and np.isnan(jr.credit.iloc[1])
    report = l3_contrib.telescoping_check(nd, jr, np.array([0., 0., .2]), np.array([.1, .1, .2]))
    assert report["n_segments"] == 0 and report["excluded_segments"] == 1
    pm = pd.DataFrame({"player_id": ["P", "P"], "match_id": ["M1", "M2"], "team_id": ["H", "H"],
                       "toi_s": [120., 120.], "toi_rate_s": [100., 0.], "c_shot": [.2, 0.]})
    sums = stack.player_sums(pm, cols=["c_shot"])
    assert sums.loc["P", "toi_s"] == 100. and sums.loc["P", "games"] == 1


def test_onice_rate_numerator_keeps_zero_stint_and_excludes_ineligible_match():
    from datamatch.l3_contrib import onice_incidence
    tabs, _ = _full(lineup_start(1) + [fo(HOME, 0.), sub(HOME, "H9", "H2", 30.),
        dict(event_type="shot_on_target", actor_id="H9", actor_team_id=HOME,
             period=1, clock_s=30., x=20., y=0.), sub(HOME, "H2", "H9", 30.), WH(40.)])
    inc = onice_incidence(tabs["nodes"], tabs["stints"])
    assert len(inc) == 10 and "H9" in set(inc.player_id)
    # недопущенный к ставкам матч задаётся на stint (как и TOI), а не флагом бросающего
    tabs["stints"]["rate_match_eligible"] = False
    assert onice_incidence(tabs["nodes"], tabs["stints"]).empty


def test_gbm_training_excludes_censored_targets_and_keeps_team_nodes(monkeypatch):
    from datamatch import l2_gbm
    nd = _censored_nodes()
    nd["valid_for_action_attribution"] = False  # team-level target does not need personal identity
    nd["node_kind"] = "team"
    monkeypatch.setattr(l2_gbm, "v_features", lambda n, h, *args: pd.DataFrame({"x": np.arange(len(n))}))
    monkeypatch.setattr(l2_gbm, "mirror_faceoffs", lambda X, y, w, f: (X, y, w))
    class Recorder:
        def fit(self, X, y, sample_weight):
            self.X, self.y = X, y
    model = l2_gbm.GBMV(load_config())
    recorder = Recorder()
    monkeypatch.setattr(model, "_model", lambda: recorder)
    model._fit_one(nd, np.array([0., 0., .2]), None)
    assert list(recorder.X.x) == [2] and list(recorder.y) == [.2]


def test_value_evaluation_reports_censored_target_exclusions():
    from datamatch.l2_value import eval_value_model, per_segment_pick_errors
    nd = _censored_nodes()
    G = realized_return(nd, np.array([0., 0., .2]))
    report = eval_value_model(nd, G, np.array([.1, .1, .2]), 0.)
    assert report["excluded_nodes"] == 2 and report["n_segments"] == 1 and report["mse"] == 0.
    errors, mids = per_segment_pick_errors(nd, G, np.array([.1, .1, .2]))
    np.testing.assert_array_equal(errors, [0.])
    assert list(mids) == ["M"]


def test_truncated_stream_is_not_a_confirmed_stoppage():
    nd, _ = _nodes(lineup_start(1) + [fo(HOME, 0.), touch(HOME, "H1", 2.)])
    last = nd.iloc[-1]
    assert last.next_node == -1 and not last.terminal
    assert last.transition_status == "unknown" and not last.valid_for_transition


def test_invalid_team_at_equal_time_cannot_merge_away_a_barrier():
    gap = touch("NOT_IN_MATCH", "H1", 1.)
    nd, _ = _nodes(lineup_start(1) + [fo(HOME, 0.), touch(HOME, "H1", 1.), gap,
                                    touch(HOME, "H1", 1.), WH(5.)])
    touch_nodes = nd[(nd.actor_id == "H1") & (nd.clock == 1.)]
    assert len(touch_nodes) == 2
    assert touch_nodes.iloc[1].chain_gap_before
    assert touch_nodes.iloc[0].transition_status == "unknown"


def test_excluded_barrier_segment_ids_match_events_across_chunks():
    from tests.test_zone_starts import run
    rows = _gap_rows()
    _, tables, _ = run([], chunk=1, matches={"M1": rows, "M2": rows})
    excluded = tables["excluded"]
    events = tables["events"].set_index(["match_id", "seq"])
    expected = events.segment.reindex(pd.MultiIndex.from_arrays([excluded.match_id, excluded.seq]))
    np.testing.assert_array_equal(excluded.segment, expected)


def test_coverage_gate_detects_ineligible_action_in_rate_numerator():
    tabs, _ = _full(lineup_start(1) + [fo(HOME, 0.), touch(HOME, "H1", 2.), WH(10.)])
    nodes = tabs["nodes"].copy()
    nodes.loc[nodes.actor_id == "H1", "valid_for_action_attribution"] = False
    nodes.loc[nodes.actor_id == "H1", "valid_for_rate_metrics"] = True
    report = l0_quality.coverage_checks(tabs["stints"], nodes)
    assert report["invalid_player_nodes_marked_excluded"] == 1
