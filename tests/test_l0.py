"""Проверки L0 на маленьких синтетических примерах."""
import numpy as np
import pandas as pd
import pytest

from datamatch import io, l0_events, l0_shifts
from datamatch.config import load_config

GEO = io.rink_geometry(load_config())      # геометрия площадок — только из rink_geometry.csv

COLS = ["match_id", "seq", "event_type", "actor_id", "actor_team_id", "other_id", "other_team_id", "period",
        "clock_s", "zone", "x", "y", "x2", "y2", "n_home", "n_away", "net_x", "net_y"]
HOME, AWAY = "TH", "TA"
GAMES = pd.DataFrame({"match_id": ["M1"], "stage": ["regular"], "home_team_id": [HOME], "away_team_id": [AWAY],
                      "rink_type": ["european_60x30"], "n_events": [0]})


def ev(rows):
    out = []
    for i, r in enumerate(rows):
        d = dict.fromkeys(COLS, None)
        d.update(match_id="M1", seq=i, n_home=6.0, n_away=6.0)
        d.update(r)
        out.append(d)
    df = pd.DataFrame(out)[COLS]
    df["period"] = df.period.astype(np.int16)
    df["clock_s"] = df.clock_s.astype(float)
    for c in ["x", "y", "x2", "y2", "n_home", "n_away", "net_x", "net_y"]:
        df[c] = df[c].astype(float)
    return df


def lineup_start(period, t=0.0):
    rows = [dict(event_type="period_start", period=period, clock_s=t)]
    for team, pref in [(HOME, "H"), (AWAY, "A")]:
        rows.append(dict(event_type="shift_G", actor_id=f"{pref}G", actor_team_id=team, period=period, clock_s=t))
        for k, slot in enumerate(["shift_F1", "shift_F2", "shift_F3", "shift_D1", "shift_D2"]):
            rows.append(dict(event_type=slot, actor_id=f"{pref}{k}", actor_team_id=team, period=period, clock_s=t))
    return rows


GOALIES = {"HG", "AG"}


def test_substitution_toi_and_same_time_order():
    rows = lineup_start(1)
    rows += [
        dict(event_type="faceoff", actor_id="H0", actor_team_id=HOME, other_id="A0", other_team_id=AWAY,
             period=1, clock_s=0.0, x=0.0, y=0.0),
        # действие в t=30 ДО замены в seq -> старый состав
        dict(event_type="touch", actor_id="H1", actor_team_id=HOME, period=1, clock_s=30.0, x=1.0, y=1.0),
        dict(event_type="shift_F2", actor_id="H9", actor_team_id=HOME, other_id="H1", other_team_id=HOME,
             period=1, clock_s=30.0),
        # действие в t=30 ПОСЛЕ замены -> новый состав
        dict(event_type="touch", actor_id="H9", actor_team_id=HOME, period=1, clock_s=30.0, x=2.0, y=1.0),
        dict(event_type="whistle", period=1, clock_s=60.0),
    ]
    e = ev(rows)
    idx, st, an = l0_shifts.reconstruct_match(e, HOME, AWAY, goalies=GOALIES)
    stints = pd.DataFrame(st)
    t_old = e.index[(e.event_type == "touch") & (e.actor_id == "H1")][0]
    t_new = e.index[(e.event_type == "touch") & (e.actor_id == "H9")][0]
    s_old, s_new = stints.set_index("stint").loc[idx[t_old]], stints.set_index("stint").loc[idx[t_new]]
    assert "H1" in s_old.home_sk and "H9" not in s_old.home_sk
    assert "H9" in s_new.home_sk and "H1" not in s_new.home_sk
    stints["match_id"], stints["home_team_id"], stints["away_team_id"] = "M1", HOME, AWAY
    stints["n_home_sk"] = stints.home_sk.map(len)
    stints["n_away_sk"] = stints.away_sk.map(len)
    stints["home_g_in"], stints["away_g_in"] = stints.home_g.notna(), stints.away_g.notna()
    stints["is_5v5"] = (stints.n_home_sk == 5) & (stints.n_away_sk == 5)
    toi = l0_shifts.player_toi(stints).set_index("player_id").toi_s
    assert toi["H1"] == pytest.approx(30.0)
    assert toi["H9"] == pytest.approx(30.0)
    assert toi["H0"] == pytest.approx(60.0)
    assert toi.sum() == pytest.approx(10 * 60.0)
    assert not an


def test_period_boundary_clears_lineups():
    rows = lineup_start(1)
    rows += [dict(event_type="touch", actor_id="H0", actor_team_id=HOME, period=1, clock_s=1200.0, x=0.0, y=0.0)]
    rows += lineup_start(2)
    # во втором периоде H0 не выходит; H7 выходит на его место
    rows = [r for r in rows if not (r.get("period") == 2 and r.get("actor_id") == "H0")]
    rows.append(dict(event_type="shift_F1", actor_id="H7", actor_team_id=HOME, period=2, clock_s=0.0))
    rows.append(dict(event_type="touch", actor_id="H7", actor_team_id=HOME, period=2, clock_s=600.0, x=0.0, y=0.0))
    e = ev(rows)
    idx, st, an = l0_shifts.reconstruct_match(e, HOME, AWAY, goalies=GOALIES)
    s = pd.DataFrame(st)
    last_p2 = s[s.period == 2].iloc[-1]
    assert "H0" not in last_p2.home_sk and "H7" in last_p2.home_sk
    assert an["carryover_cleared_at_period_end"] == 12
    # отрезки не пересекают границу периода
    assert (s.groupby("period").t_end.max() <= pd.Series({1: 1200.0, 2: 600.0})).all()


def test_goalie_slot_extra_attacker():
    rows = lineup_start(3)
    rows += [
        dict(event_type="shift_G", actor_id="H8", actor_team_id=HOME, other_id="HG", other_team_id=HOME,
             period=3, clock_s=1100.0),
        dict(event_type="touch", actor_id="H8", actor_team_id=HOME, period=3, clock_s=1110.0, x=20.0, y=0.0),
        dict(event_type="shift_G", actor_id="HG", actor_team_id=HOME, other_id="H8", other_team_id=HOME,
             period=3, clock_s=1150.0),
        dict(event_type="whistle", period=3, clock_s=1200.0),
    ]
    e = ev(rows)
    idx, st, an = l0_shifts.reconstruct_match(e, HOME, AWAY, goalies=GOALIES)
    s = pd.DataFrame(st).set_index("stint")
    t = e.index[(e.event_type == "touch")][0]
    cur = s.loc[idx[t]]
    assert pd.isna(cur.home_g) and len(cur.home_sk) == 6 and "H8" in cur.home_sk
    assert s.iloc[-1].home_g == "HG" and len(s.iloc[-1].home_sk) == 5


def _nodes(rows):
    cfg = load_config()
    e = ev(rows)
    q = {}
    e2, scheme, _ = l0_events.preprocess(e, GAMES, cfg, q)
    sc, st, an, abm = l0_shifts.reconstruct_all(e2, GAMES, goalies_global=GOALIES)
    e2["stint"] = sc
    nd, excl, es = l0_events.build_nodes(e2, GAMES, st, GEO, cfg, q)
    return nd, q


def test_merge_selfpass_carry_and_segments():
    rows = lineup_start(1)
    rows += [
        dict(event_type="faceoff", actor_id="H0", actor_team_id=HOME, other_id="A0", other_team_id=AWAY,
             period=1, clock_s=0.0, x=0.0, y=0.0),
        dict(event_type="touch", actor_id="H1", actor_team_id=HOME, period=1, clock_s=1.0, x=-5.0, y=0.0),
        dict(event_type="pass_complete", actor_id="H1", actor_team_id=HOME, other_id="H2", other_team_id=HOME,
             period=1, clock_s=1.0, x=-5.0, y=0.0, x2=5.0, y2=3.0),
        dict(event_type="touch", actor_id="H2", actor_team_id=HOME, period=1, clock_s=2.0, x=5.0, y=3.0),
        # ведение: следующий узел того же игрока смещён > 3 м
        dict(event_type="touch", actor_id="H2", actor_team_id=HOME, period=1, clock_s=4.0, x=15.0, y=3.0),
        # пас самому себе -> удержание
        dict(event_type="pass_complete", actor_id="H2", actor_team_id=HOME, other_id="H2", other_team_id=HOME,
             period=1, clock_s=4.0, x=15.0, y=3.0),
        dict(event_type="shot_on_target", actor_id="H2", actor_team_id=HOME, other_id="AG", other_team_id=AWAY,
             period=1, clock_s=5.0, x=16.0, y=2.0, x2=26.0, y2=0.0),
        dict(event_type="touch", actor_id="A3", actor_team_id=AWAY, period=1, clock_s=6.0, x=-20.0, y=-2.0),
        dict(event_type="whistle", period=1, clock_s=8.0),
        dict(event_type="faceoff", actor_id="A0", actor_team_id=AWAY, other_id="H0", other_team_id=HOME,
             period=1, clock_s=8.0, x=-20.5, y=7.0),
        dict(event_type="touch", actor_id="A1", actor_team_id=AWAY, period=1, clock_s=9.0, x=-18.0, y=5.0),
        dict(event_type="goal", actor_id="A1", actor_team_id=AWAY, other_id="HG", other_team_id=HOME,
             period=1, clock_s=10.0, x=20.0, y=1.0),
        dict(event_type="assist_primary", actor_id="A2", actor_team_id=AWAY, period=1, clock_s=10.0),
    ]
    nd, q = _nodes(rows)
    # touch+pass одного игрока в ту же секунду -> один узел
    assert q["merged_events"] == 2
    acts = nd.action.tolist()
    assert acts[:5] == ["faceoff", "pass", "carry", "hold", "shot"]
    assert bool(nd.self_pass.iloc[3])
    # бросок -> соперник: смена владения по команде
    assert nd.sigma_next.iloc[4] == -1
    # свисток закрывает отрезок, вбрасывание открывает новый
    assert bool(nd.terminal.iloc[5]) and nd.segment.iloc[6] != nd.segment.iloc[5]
    # гол — терминальный узел, счёт после гола обновится
    g = nd[nd.is_goal].iloc[0]
    assert bool(g.terminal)
    assert nd.arrival.iloc[2] == "reception" and nd.arrival.iloc[5] == "recovery"


def test_missing_xy_opponent_is_chain_barrier_not_carry():
    rows = lineup_start(1) + [
        dict(event_type="faceoff", actor_id="H0", actor_team_id=HOME, other_id="A0", other_team_id=AWAY,
             period=1, clock_s=0.0, x=0.0, y=0.0),
        dict(event_type="touch", actor_id="H1", actor_team_id=HOME, period=1, clock_s=1.0, x=0.0, y=0.0),
        dict(event_type="touch", actor_id="A1", actor_team_id=AWAY, period=1, clock_s=2.0, x=np.nan, y=np.nan),
        dict(event_type="touch", actor_id="H1", actor_team_id=HOME, period=1, clock_s=3.0, x=10.0, y=0.0),
        dict(event_type="shot_on_target", actor_id="H1", actor_team_id=HOME, other_id="AG", other_team_id=AWAY,
             period=1, clock_s=4.0, x=15.0, y=0.0),
        dict(event_type="whistle", period=1, clock_s=5.0),
    ]
    nd, q = _nodes(rows)
    h1 = nd[(nd.actor_id == "H1") & (nd.clock == 3.0)].iloc[0]
    before = nd[(nd.actor_id == "H1") & (nd.clock == 1.0)].iloc[0]
    assert h1.chain_gap_before and h1.prev_node == -1
    assert before.next_node == -1 and before.action == "hold"
    assert q["chain_barrier_nodes_excluded"] == 1


def test_period_zero_excluded():
    rows = lineup_start(1)
    rows += [dict(event_type="touch", actor_id="H1", actor_team_id=HOME, period=1, clock_s=1.0, x=0.0, y=0.0),
             dict(event_type="so_attempt", actor_id="H2", actor_team_id=HOME, period=0, clock_s=0.0),
             dict(event_type="goal", actor_id="H2", actor_team_id=HOME, other_id="AG", other_team_id=AWAY,
                  period=0, clock_s=0.0, x=20.0, y=0.0)]
    nd, q = _nodes(rows)
    assert (nd.period > 0).all() and not nd.is_goal.any()


def test_zone_label_does_not_attach_across_periods():
    # метка входа в периоде 2 в момент 100 с; узел того же игрока в периоде 1 тоже в 100 с
    rows = lineup_start(1)
    rows += [dict(event_type="touch", actor_id="H1", actor_team_id=HOME, period=1, clock_s=100.0, x=8.0, y=0.0),
             dict(event_type="whistle", period=1, clock_s=1200.0)]
    rows += lineup_start(2)
    rows += [dict(event_type="faceoff", actor_id="H0", actor_team_id=HOME, other_id="A0", other_team_id=AWAY,
                  period=2, clock_s=0.0, x=0.0, y=0.0),
             dict(event_type="touch", actor_id="H2", actor_team_id=HOME, period=2, clock_s=99.0, x=5.0, y=0.0),
             dict(event_type="entry_success", actor_id="H1", actor_team_id=HOME, period=2, clock_s=100.0),
             dict(event_type="whistle", period=2, clock_s=120.0)]
    nd, q = _nodes(rows)
    n1 = nd[(nd.period == 1) & (nd.actor_id == "H1")]
    assert not n1.entry.any()
    assert q["zone_labels_rejected_boundary"] == 0


def test_pass_feature_two_nodes_back_and_time_windows():
    rows = lineup_start(1)
    rows += [
        dict(event_type="faceoff", actor_id="H0", actor_team_id=HOME, other_id="A0", other_team_id=AWAY,
             period=1, clock_s=0.0, x=0.0, y=0.0),
        dict(event_type="touch", actor_id="H1", actor_team_id=HOME, period=1, clock_s=10.0, x=18.0, y=-8.0),
        dict(event_type="pass_complete", actor_id="H1", actor_team_id=HOME, other_id="H2", other_team_id=HOME,
             period=1, clock_s=10.0, x=18.0, y=-8.0, x2=20.0, y2=6.0),
        dict(event_type="touch", actor_id="H2", actor_team_id=HOME, period=1, clock_s=11.0, x=20.0, y=6.0),
        # отметка времени гола намеренно поздняя: окна считаются от предыдущего узла (11 с)
        dict(event_type="goal", actor_id="H2", actor_team_id=HOME, other_id="AG", other_team_id=AWAY,
             period=1, clock_s=16.0, x=21.0, y=5.0),
    ]
    nd, q = _nodes(rows)
    g = nd[nd.is_goal].iloc[0]
    assert g.arrival == "continuation"
    assert g.prev_pass_dy == pytest.approx(14.0) and bool(g.prev_pass_cross)
    assert g.t_ref == pytest.approx(11.0)


def test_self_pass_never_carry_and_penalty_shot_goal_in_score():
    rows = lineup_start(1)
    rows += [
        dict(event_type="faceoff", actor_id="H0", actor_team_id=HOME, other_id="A0", other_team_id=AWAY,
             period=1, clock_s=0.0, x=0.0, y=0.0),
        dict(event_type="touch", actor_id="H2", actor_team_id=HOME, period=1, clock_s=4.0, x=5.0, y=3.0),
        dict(event_type="pass_complete", actor_id="H2", actor_team_id=HOME, other_id="H2", other_team_id=HOME,
             period=1, clock_s=4.0, x=5.0, y=3.0),
        dict(event_type="touch", actor_id="H2", actor_team_id=HOME, period=1, clock_s=6.0, x=15.0, y=3.0),
        dict(event_type="whistle", period=1, clock_s=8.0),
        # штрафной бросок в периоде 1: попытка исключается, гол входит в счёт
        dict(event_type="so_attempt", actor_id="H3", actor_team_id=HOME, period=1, clock_s=8.0),
        dict(event_type="goal", actor_id="H3", actor_team_id=HOME, other_id="AG", other_team_id=AWAY,
             period=1, clock_s=12.0, x=20.0, y=0.0),
        dict(event_type="faceoff", actor_id="A0", actor_team_id=AWAY, other_id="H0", other_team_id=HOME,
             period=1, clock_s=12.0, x=0.0, y=0.0),
        dict(event_type="touch", actor_id="A1", actor_team_id=AWAY, period=1, clock_s=13.0, x=-5.0, y=0.0),
    ]
    nd, q = _nodes(rows)
    sp = nd[nd.self_pass].iloc[0]
    assert sp.action == "hold"
    assert not nd.is_goal.any()                     # попытка штрафного броска не в цепи
    last = nd.iloc[-1]
    assert last.team_id == AWAY and last.score_diff == -1


@pytest.mark.parametrize("goal", [False, True])
def test_shot_features_invariant_to_shot_timestamp(goal):
    from datamatch import l1_xg

    def build(t_shot):
        rows = lineup_start(1)
        rows += [
            dict(event_type="faceoff", actor_id="H0", actor_team_id=HOME, other_id="A0", other_team_id=AWAY,
                 period=1, clock_s=0.0, x=0.0, y=0.0),
            dict(event_type="entry_success", actor_id="H1", actor_team_id=HOME, period=1, clock_s=0.5),
            dict(event_type="touch", actor_id="H1", actor_team_id=HOME, period=1, clock_s=0.5, x=15.0, y=-5.0),
            dict(event_type="pass_complete", actor_id="H1", actor_team_id=HOME, other_id="H2", other_team_id=HOME,
                 period=1, clock_s=0.5, x=15.0, y=-5.0, x2=18.0, y2=5.0),
            dict(event_type="touch", actor_id="H2", actor_team_id=HOME, period=1, clock_s=1.5, x=18.0, y=5.0),
            dict(event_type="goal" if goal else "shot_on_target", actor_id="H2", actor_team_id=HOME,
                 other_id="AG", other_team_id=AWAY, period=1, clock_s=t_shot, x=19.0, y=4.0),
            dict(event_type="whistle", period=1, clock_s=max(t_shot, 6.0)),
        ]
        nd, q = _nodes(rows)
        s = l1_xg.shot_table(nd)
        return s[[f for f in l1_xg.FEATURES if f != "opp_def"]].iloc[0]

    a, b = build(2.0), build(4.0)
    assert a.prev_pass_dy == pytest.approx(10.0)
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_shot_timestamp_audit_finds_no_dependence():
    from datamatch import l1_audit
    cfg = load_config()
    rows = lineup_start(1)
    t = 0.0
    rows.append(dict(event_type="faceoff", actor_id="H0", actor_team_id=HOME, other_id="A0", other_team_id=AWAY,
                     period=1, clock_s=0.0, x=0.0, y=0.0))
    for k in range(6):
        t += 3.0
        rows += [dict(event_type="entry_success", actor_id="H1", actor_team_id=HOME, period=1, clock_s=t),
                 dict(event_type="touch", actor_id="H1", actor_team_id=HOME, period=1, clock_s=t, x=12.0, y=-6.0),
                 dict(event_type="pass_complete", actor_id="H1", actor_team_id=HOME, other_id="H2",
                      other_team_id=HOME, period=1, clock_s=t, x=12.0, y=-6.0, x2=18.0, y2=4.0),
                 dict(event_type="touch", actor_id="H2", actor_team_id=HOME, period=1, clock_s=t + 0.8, x=18.0, y=4.0),
                 dict(event_type="shot_on_target", actor_id="H2", actor_team_id=HOME, other_id="AG",
                      other_team_id=AWAY, period=1, clock_s=t + 1.2, x=19.0, y=3.0),
                 dict(event_type="touch", actor_id="H3", actor_team_id=HOME, period=1, clock_s=t + 1.6, x=22.0, y=1.0),
                 dict(event_type="goal" if k % 2 else "shot_missed", actor_id="H3", actor_team_id=HOME,
                      other_id="AG", other_team_id=AWAY, period=1, clock_s=t + 2.0, x=23.0, y=1.0)]
        if k % 2 == 0:
            rows.append(dict(event_type="whistle", period=1, clock_s=t + 2.5))
        rows.append(dict(event_type="faceoff", actor_id="H0", actor_team_id=HOME, other_id="A0",
                         other_team_id=AWAY, period=1, clock_s=t + 2.5, x=0.0, y=0.0))
    e = ev(rows)
    q = {}
    e2, _, _ = l0_events.preprocess(e, GAMES, cfg, q)
    sc, st, _, _ = l0_shifts.reconstruct_all(e2, GAMES, goalies_global=GOALIES)
    e2["stint"] = sc
    for c in ["home_team_id", "away_team_id"]:
        e2[c] = GAMES[c].iat[0]
    r = l1_audit.shot_time_perturbation(e2, GAMES, st, GEO, cfg, ["M1"], (1.5, 4.0))
    assert r["time_dependent"] == []
    assert r["n_perturbed_last"] > 0


def test_l0_chunking_invariant():
    """Результат L0 не зависит от размера порции матчей (смещения номеров узлов/отрезков)."""
    from datamatch import stage_l0
    cfg = load_config()
    frames = []
    for k, mid in enumerate(["M1", "M2", "M3"]):
        rows = lineup_start(1)
        rows += [
            dict(event_type="faceoff", actor_id="H0", actor_team_id=HOME, other_id="A0", other_team_id=AWAY,
                 period=1, clock_s=0.0, x=0.0, y=0.0),
            dict(event_type="touch", actor_id="H1", actor_team_id=HOME, period=1, clock_s=2.0, x=10.0, y=1.0 + k),
            dict(event_type="pass_complete", actor_id="H1", actor_team_id=HOME, other_id="H2", other_team_id=HOME,
                 period=1, clock_s=2.0, x=10.0, y=1.0 + k, x2=15.0, y2=-4.0),
            dict(event_type="touch", actor_id="H2", actor_team_id=HOME, period=1, clock_s=3.0, x=15.0, y=-4.0),
            dict(event_type="shot_on_target", actor_id="H2", actor_team_id=HOME, other_id="AG", other_team_id=AWAY,
                 period=1, clock_s=3.5, x=16.0, y=-3.0),
            dict(event_type="touch", actor_id="A2", actor_team_id=AWAY, period=1, clock_s=5.0, x=-20.0, y=2.0),
            dict(event_type="whistle", period=1, clock_s=7.0),
        ]
        e = ev(rows)
        e["match_id"] = mid
        frames.append(e)
    allev = pd.concat(frames, ignore_index=True)
    games = pd.concat([GAMES.assign(match_id=m) for m in ["M1", "M2", "M3"]], ignore_index=True)
    roster = pd.DataFrame({"match_id": ["M1", "M2", "M3"] * 2, "team_id": [HOME] * 3 + [AWAY] * 3,
                           "player_id": ["HG"] * 3 + ["AG"] * 3, "slot": "G", "line": 1})
    load = lambda ids: allev[allev.match_id.isin(ids)].reset_index(drop=True)
    res = {}
    for C in (1, 3):
        c = cfg.with_overrides({"l0": {"chunk_matches": C}})
        tabs, q, js = stage_l0.process_source(c, "train", games, roster, GOALIES, GEO,
                                              load_chunk=load, match_ids=["M1", "M2", "M3"])
        res[C] = (tabs, q)
    n1, n3 = res[1][0]["nodes"], res[3][0]["nodes"]
    pd.testing.assert_frame_equal(n1.drop(columns=["segment", "possession"]), n3.drop(columns=["segment", "possession"]))
    # номера отрезков/владений могут отличаться сдвигом, но разбиение на группы совпадает
    assert (pd.factorize(n1.segment)[0] == pd.factorize(n3.segment)[0]).all()
    assert (pd.factorize(n1.possession)[0] == pd.factorize(n3.possession)[0]).all()
    assert n1.next_node.max() == len(n1) - 1 or (n1.next_node < len(n1)).all()
    assert res[1][1]["merged_events"] == res[3][1]["merged_events"]
    assert res[1][1]["identity"]["actor_on_ice_share"] == pytest.approx(res[3][1]["identity"]["actor_on_ice_share"])


def test_position_G_listed_as_skater_entering_goalie_slot_is_extra_attacker():
    """Игрок с position=G, заявленный в этом матче полевым (LD), выходит через shift_G вместо
    вратаря — это снятие вратаря (пустые ворота), а не замена вратаря."""
    rows = lineup_start(3)
    rows += [
        dict(event_type="shift_G", actor_id="HX", actor_team_id=HOME, other_id="HG", other_team_id=HOME,
             period=3, clock_s=1100.0),
        dict(event_type="touch", actor_id="H1", actor_team_id=HOME, period=3, clock_s=1110.0, x=20.0, y=0.0),
        dict(event_type="whistle", period=3, clock_s=1200.0),
    ]
    e = ev(rows)
    g = l0_shifts.match_goalie_set("M1", {"M1": {"HG", "AG"}}, GOALIES | {"HX"}, {"M1": {"HX"}})
    assert "HX" not in g and {"HG", "AG"} <= g
    idx, st, an = l0_shifts.reconstruct_match(e, HOME, AWAY, goalies=g)
    s = pd.DataFrame(st).set_index("stint")
    cur = s.loc[idx[e.index[e.event_type == "touch"][0]]]
    assert pd.isna(cur.home_g) and len(cur.home_sk) == 6 and "HX" in cur.home_sk
    # без учёта заявки матча (только справочник) — ошибочно «вратарь остался»
    idx2, st2, _ = l0_shifts.reconstruct_match(e, HOME, AWAY, goalies=GOALIES | {"HX"})
    cur2 = pd.DataFrame(st2).set_index("stint").loc[idx2[e.index[e.event_type == "touch"][0]]]
    assert cur2.home_g == "HX" and len(cur2.home_sk) == 5


def test_resolve_roles_by_usage_both_conflict_directions():
    from datamatch import io
    players = pd.DataFrame({"player_id": ["P1", "P2", "P3", "P4"], "position": ["G", "D", "G", "C"],
                            "lineup_slot_mode": [None, None, "G", None], "shoots": ["L"] * 4})
    su = pd.DataFrame({"player_id": ["P1", "P2", "P3", "P4"], "n_lineup_G": [0, 4, 6, 0],
                       "n_lineup_skater": [3, 1, 0, 5], "skater_slot_mode": ["LD", "RD", None, "C"],
                       "n_shift_G": [1, 0, 30, 0], "n_shift_skater": [50, 40, 0, 60]})
    r = io.resolve_roles(players, su).set_index("player_id")
    assert r.loc["P1", "role"] == "D" and bool(r.loc["P1", "conflict_position_G_plays_skater"])
    assert r.loc["P2", "role"] == "G" and bool(r.loc["P2", "conflict_position_skater_plays_G"])
    assert r.loc["P3", "role"] == "G" and not bool(r.loc["P3", "goalie_conflict"])
    assert r.loc["P4", "role"] == "C" and not bool(r.loc["P4", "goalie_conflict"])


def test_last_possession_runs_to_stoppage():
    rows = lineup_start(1)
    rows += [
        dict(event_type="faceoff", actor_id="H0", actor_team_id=HOME, other_id="A0", other_team_id=AWAY,
             period=1, clock_s=0.0, x=0.0, y=0.0),
        dict(event_type="touch", actor_id="H1", actor_team_id=HOME, period=1, clock_s=4.0, x=5.0, y=0.0),
        dict(event_type="touch", actor_id="A1", actor_team_id=AWAY, period=1, clock_s=10.0, x=-5.0, y=0.0),
        dict(event_type="whistle", period=1, clock_s=30.0),
        dict(event_type="faceoff", actor_id="H0", actor_team_id=HOME, other_id="A0", other_team_id=AWAY,
             period=1, clock_s=30.0, x=20.5, y=7.0),
        dict(event_type="touch", actor_id="H2", actor_team_id=HOME, period=1, clock_s=35.0, x=22.0, y=5.0),
        # в конце периода свистка нет: владение длится до конца периода
        dict(event_type="intermission", period=1, clock_s=1200.0),
    ]
    nd, q = _nodes(rows)
    ps = l0_events.build_possessions(nd)
    away = ps[ps.team_id == AWAY].iloc[0]
    assert away.t_start == 10.0 and away.t_end == 30.0 and away.duration == pytest.approx(20.0)
    last = ps.iloc[-1]
    assert last.t_end == 1200.0
    from datamatch import l0_quality
    st = pd.DataFrame({"match_id": ["M1"], "period": [1], "t_end": [1200.0]})
    chk = l0_quality.possession_checks(ps, st)
    assert chk["segments_not_tiled"] == 0 and chk["last_possessions_extended_to_stop"] == 2


def test_merge_does_not_cross_line_change():
    rows = lineup_start(1)
    rows += [
        dict(event_type="faceoff", actor_id="H0", actor_team_id=HOME, other_id="A0", other_team_id=AWAY,
             period=1, clock_s=0.0, x=0.0, y=0.0),
        dict(event_type="touch", actor_id="H1", actor_team_id=HOME, period=1, clock_s=20.0, x=-5.0, y=0.0),
        # замена соперника в ту же секунду, между касанием и пасом в seq
        dict(event_type="shift_F1", actor_id="A9", actor_team_id=AWAY, other_id="A1", other_team_id=AWAY,
             period=1, clock_s=20.0),
        dict(event_type="pass_complete", actor_id="H1", actor_team_id=HOME, other_id="H2", other_team_id=HOME,
             period=1, clock_s=20.0, x=-5.0, y=0.0, x2=5.0, y2=2.0),
        dict(event_type="touch", actor_id="H2", actor_team_id=HOME, period=1, clock_s=21.0, x=5.0, y=2.0),
        dict(event_type="whistle", period=1, clock_s=25.0),
    ]
    nd, q = _nodes(rows)
    h1 = nd[nd.actor_id == "H1"]
    assert len(h1) == 2 and h1.stint.nunique() == 2
    assert q["merge_blocked_by_stint_change"] == 1
    # пас относится к составу после замены
    assert h1[h1.primary == "pass_complete"].stint.iat[0] == h1.stint.max()


def test_future_clock_rollback_does_not_change_past_features():
    base = lineup_start(1) + [
        dict(event_type="faceoff", actor_id="H0", actor_team_id=HOME, other_id="A0", other_team_id=AWAY,
             period=1, clock_s=100.0, x=0.0, y=0.0),
        dict(event_type="touch", actor_id="H1", actor_team_id=HOME, period=1, clock_s=104.0, x=5.0, y=0.0),
        dict(event_type="touch", actor_id="H2", actor_team_id=HOME, period=1, clock_s=106.0, x=9.0, y=1.0),
    ]
    # позднее по seq событие с откатом часов (раньше начала отрезка)
    tail = [dict(event_type="touch", actor_id="H3", actor_team_id=HOME, period=1, clock_s=95.0, x=12.0, y=2.0),
            dict(event_type="whistle", period=1, clock_s=110.0)]
    nd_a, _ = _nodes(base + [dict(event_type="whistle", period=1, clock_s=110.0)])
    nd_b, _ = _nodes(base + tail)
    k = len(nd_a)
    for c in ["t_seg", "t_poss", "n_in_poss"]:
        np.testing.assert_allclose(nd_a[c].values, nd_b[c].values[:k])
    assert (nd_b.t_seg >= 0).all() and (nd_b.t_poss >= 0).all()


@pytest.mark.parametrize("labels", ["none", "exit_only"])
def test_match_without_entry_labels(labels):
    """Регрессия: матч без entry_success (G0029) обрабатывается отдельно, без TypeError."""
    from datamatch import stage_l0
    rows = lineup_start(1) + [
        dict(event_type="faceoff", actor_id="H0", actor_team_id=HOME, other_id="A0", other_team_id=AWAY,
             period=1, clock_s=0.0, x=0.0, y=0.0),
        dict(event_type="touch", actor_id="H1", actor_team_id=HOME, period=1, clock_s=2.0, x=-15.0, y=1.0),
        dict(event_type="shot_on_target", actor_id="H1", actor_team_id=HOME, other_id="AG", other_team_id=AWAY,
             period=1, clock_s=4.0, x=15.0, y=1.0),
        dict(event_type="whistle", period=1, clock_s=6.0),
    ]
    if labels == "exit_only":
        rows.insert(-2, dict(event_type="exit_success", actor_id="H1", actor_team_id=HOME, period=1, clock_s=2.5))
    allev = ev(rows)
    roster = pd.DataFrame({"match_id": ["M1", "M1"], "team_id": [HOME, AWAY], "player_id": ["HG", "AG"],
                           "slot": "G", "line": 1})
    tabs, q, _ = stage_l0.process_source(load_config(), "train", GAMES, roster, GOALIES, GEO,
                                         load_chunk=lambda ids: allev, match_ids=["M1"])
    nd = tabs["nodes"]
    assert len(nd) == 3 and not nd.recent_entry.any()
    assert nd.exit.any() == (labels == "exit_only")
