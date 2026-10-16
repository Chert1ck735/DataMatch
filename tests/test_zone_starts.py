"""Индивидуальные смены, тип и зона старта смены (L0)."""
import numpy as np
import pandas as pd
import pytest

from datamatch import l0_quality, stage_l0
from datamatch.cache import Cache
from datamatch.config import load_config
from tests.test_l0 import AWAY, GAMES, GEO, GOALIES, HOME, ev, lineup_start

ROSTER = pd.DataFrame({"match_id": ["M1", "M1"], "team_id": [HOME, AWAY], "player_id": ["HG", "AG"],
                       "slot": "G", "line": 1})


def run(rows, chunk=100, matches=None):
    cfg = load_config().with_overrides({"l0": {"chunk_matches": chunk}})
    if matches is None:
        allev = ev(rows)
        games, roster, mids = GAMES, ROSTER, ["M1"]
    else:
        allev = pd.concat([ev(r).assign(match_id=m) for m, r in matches.items()], ignore_index=True)
        mids = list(matches)
        games = pd.concat([GAMES.assign(match_id=m) for m in mids], ignore_index=True)
        roster = pd.concat([ROSTER.assign(match_id=m) for m in mids], ignore_index=True)
    tabs, q, _ = stage_l0.process_source(cfg, "train", games, roster, GOALIES, GEO,
                                         load_chunk=lambda ids: allev[allev.match_id.isin(ids)].reset_index(drop=True),
                                         match_ids=mids)
    ps = tabs["player_shifts"]
    chk = l0_quality.player_shift_checks(ps, tabs["toi"], tabs["nodes"])
    # обязательные инварианты в каждом сценарии
    for k in ["bound_violations", "overlaps", "duplicate_ids", "toi_mismatch_player_matches", "faceoff_not_on_ice",
              "on_the_fly_with_zone", "zone_without_faceoff"]:
        assert chk[k] == 0, (k, chk[k])
    return ps, tabs, chk


def sh(ps, pid, k=0):
    return ps[ps.player_id == pid].sort_values("seq_start").iloc[k]


def fo(team, x, y=0.0, t=0.0, actor=None, other=None, period=1):
    return dict(event_type="faceoff", actor_id=actor or ("H0" if team == HOME else "A0"), actor_team_id=team,
                other_id=other or ("A0" if team == HOME else "H0"), other_team_id=AWAY if team == HOME else HOME,
                period=period, clock_s=t, x=x, y=y)


def sub(team, enter, leave, t, period=1, slot="shift_F1"):
    return dict(event_type=slot, actor_id=enter, actor_team_id=team, other_id=leave, other_team_id=team,
                period=period, clock_s=t)


def touch(team, pid, t, x=0.0, period=1):
    return dict(event_type="touch", actor_id=pid, actor_team_id=team, period=period, clock_s=t, x=x, y=0.0)


WH = lambda t, period=1: dict(event_type="whistle", period=period, clock_s=t)


@pytest.mark.parametrize("x,home_zone,away_zone", [(20.5, "off", "def"), (-20.5, "def", "off"), (0.0, "neutral", "neutral")])
def test_faceoff_start_each_zone_and_both_perspectives(x, home_zone, away_zone):
    ps, _, chk = run(lineup_start(1) + [fo(HOME, x), touch(HOME, "H1", 2.0), WH(5.0)])
    assert sh(ps, "H1").start_type == "faceoff" and sh(ps, "H1").start_zone == home_zone
    assert sh(ps, "A1").start_type == "faceoff" and sh(ps, "A1").start_zone == away_zone
    assert not ps.player_id.isin(["HG", "AG"]).any()          # вратари не входят


def test_faceoff_winner_does_not_change_zone():
    # одна физическая точка (в зоне атаки хозяев): победил хозяин (x=20.5 в его системе)
    # или гость (x=−20.5 в системе гостя)
    a, _, _ = run(lineup_start(1) + [fo(HOME, 20.5), WH(3.0)])
    b, _, _ = run(lineup_start(1) + [fo(AWAY, -20.5), WH(3.0)])
    for pid in ["H1", "A1"]:
        assert sh(a, pid).start_zone == sh(b, pid).start_zone


def test_on_the_fly_and_teammate_or_opponent_changes_keep_shift():
    rows = lineup_start(1) + [fo(HOME, 0.0), touch(HOME, "H1", 2.0),
                              sub(HOME, "H9", "H2", 10.0),          # замена партнёра по ходу игры
                              sub(AWAY, "A9", "A2", 12.0),          # замена соперника
                              touch(HOME, "H1", 15.0), WH(20.0)]
    ps, tabs, _ = run(rows)
    assert (ps.player_id == "H1").sum() == 1                       # смена H1 продолжается
    h9 = sh(ps, "H9")
    assert h9.start_type == "on_the_fly" and pd.isna(h9.start_zone) and h9.faceoff_seq == -1
    assert tabs["nodes"][tabs["nodes"].actor_id == "H1"].player_shift_id.nunique() == 1


def test_second_faceoff_does_not_change_start_zone():
    rows = lineup_start(1) + [fo(HOME, 20.5), touch(HOME, "H1", 2.0), WH(5.0),
                              fo(HOME, -20.5, t=5.0), touch(HOME, "H1", 7.0), WH(9.0)]
    ps, tabs, _ = run(rows)
    s1 = sh(ps, "H1")
    assert (ps.player_id == "H1").sum() == 1 and s1.start_zone == "off"
    nd = tabs["nodes"][tabs["nodes"].actor_id == "H1"]
    assert (nd.shift_start_zone == "off").all()


def test_left_before_opening_faceoff_and_no_registered_faceoff():
    rows = lineup_start(1) + [fo(HOME, 0.0), touch(HOME, "H1", 2.0), WH(10.0),
                              sub(HOME, "H9", "H2", 10.0),          # вышел в остановке…
                              sub(HOME, "H2", "H9", 10.0),          # …и ушёл до вбрасывания
                              sub(HOME, "H8", "H3", 10.0),          # вышел в остановке
                              touch(HOME, "H1", 12.0),              # игра без зарегистрированного вбрасывания
                              WH(14.0)]
    ps, _, chk = run(rows)
    assert sh(ps, "H9").start_type == "unknown" and sh(ps, "H9").unknown_reason == "left_before_faceoff"
    h8 = sh(ps, "H8")
    assert h8.start_type == "unknown" and h8.unknown_reason == "play_resumed_without_faceoff"


def test_missing_coordinates_or_team_keep_faceoff_type():
    no_xy = fo(HOME, np.nan)
    a, _, _ = run(lineup_start(1) + [no_xy, WH(3.0)])
    assert sh(a, "H1").start_type == "faceoff" and pd.isna(sh(a, "H1").start_zone)
    assert sh(a, "H1").unknown_reason == "faceoff_xy_missing"
    no_team = fo(HOME, 20.5)
    no_team.update(actor_id=None, actor_team_id=None)
    b, _, chk = run(lineup_start(1) + [no_team, WH(3.0)])
    assert sh(b, "H1").start_type == "faceoff" and pd.isna(sh(b, "H1").start_zone)
    assert sh(b, "H1").unknown_reason == "faceoff_team_unknown"
    assert chk["faceoff_starts_zone_unknown"] > 0


def test_same_clock_different_seq_and_clock_rollback():
    rows = lineup_start(1) + [fo(HOME, 0.0), touch(HOME, "H1", 2.0), WH(50.0),
                              sub(HOME, "H9", "H2", 50.0),          # до вбрасывания по seq
                              fo(HOME, 20.5, t=50.0),
                              sub(HOME, "H8", "H3", 50.0),          # то же время, после вбрасывания
                              sub(HOME, "H7", "H4", 49.0),          # откат часов, по seq после вбрасывания
                              touch(HOME, "H1", 55.0), WH(60.0)]
    ps, _, _ = run(rows)
    assert sh(ps, "H9").start_type == "faceoff" and sh(ps, "H9").start_zone == "off"
    assert sh(ps, "H8").start_type == "on_the_fly"
    assert sh(ps, "H7").start_type == "on_the_fly"


def test_goalie_pull_extra_attacker_and_repeated_entry():
    rows = lineup_start(1) + [fo(HOME, 0.0), touch(HOME, "H1", 2.0),
                              dict(event_type="shift_G", actor_id="H8", actor_team_id=HOME, other_id="HG",
                                   other_team_id=HOME, period=1, clock_s=10.0),
                              dict(event_type="shift_F2", actor_id="H1", actor_team_id=HOME, period=1, clock_s=11.0),
                              touch(HOME, "H8", 12.0), WH(20.0)]
    ps, tabs, _ = run(rows)
    h8 = sh(ps, "H8")
    assert h8.start_type == "on_the_fly"                           # лишний полевой — в сменах
    assert (ps.player_id == "H1").sum() == 1                       # повторный выход не создаёт смену
    assert not ps.player_id.isin(["HG"]).any()


def test_period_boundaries_and_period_zero():
    rows = lineup_start(1) + [fo(HOME, 0.0), touch(HOME, "H1", 2.0),
                              dict(event_type="intermission", period=1, clock_s=1200.0)]
    rows += lineup_start(2) + [fo(AWAY, 20.5, period=2), touch(AWAY, "A1", 3.0, period=2), WH(5.0, period=2)]
    rows += [dict(event_type="so_attempt", actor_id="H1", actor_team_id=HOME, period=0, clock_s=0.0),
             dict(event_type="shift_F1", actor_id="H5", actor_team_id=HOME, period=0, clock_s=0.0)]
    ps, _, _ = run(rows)
    h1 = ps[ps.player_id == "H1"].sort_values("seq_start")
    assert list(h1.period) == [1, 2] and h1.iloc[0].t_end == 1200.0
    assert h1.iloc[1].start_zone == "def"                          # гости выиграли в своей зоне атаки
    assert (ps.period > 0).all() and not (ps.player_id == "H5").any()


def test_chunk_invariance_and_cache_roundtrip(tmp_path):
    m1 = lineup_start(1) + [fo(HOME, 20.5), touch(HOME, "H1", 2.0), sub(HOME, "H9", "H2", 4.0), WH(6.0)]
    m2 = lineup_start(1) + [fo(AWAY, 0.0), touch(AWAY, "A1", 2.0), WH(6.0), sub(AWAY, "A9", "A3", 6.0),
                            fo(AWAY, -20.5, t=6.0), WH(9.0)]
    a, _, _ = run(None, chunk=1, matches={"M1": m1, "M2": m2})
    b, _, _ = run(None, chunk=2, matches={"M1": m1, "M2": m2})
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    assert sh(b[b.match_id == "M2"], "A9").start_zone == "def"
    cfg = load_config().with_overrides({"paths": {"cache": str(tmp_path)}})
    c = Cache(cfg, {})
    c.save_df("k", "ps", a)
    r = c.load_df("k", "ps")
    pd.testing.assert_frame_equal(a.reset_index(drop=True), r, check_dtype=False)
