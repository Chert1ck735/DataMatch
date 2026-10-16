"""Generated L0 contracts; expected values follow seq, identity and exposure."""
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st

from datamatch import l0_events, l0_shifts
from datamatch.cache import Cache
from datamatch.config import load_config
from tests.test_l0 import AWAY, GAMES, GEO, GOALIES, HOME, ev, lineup_start
from tests.test_revision_l0 import _full
from tests.test_zone_starts import fo, sub, touch

CONTRACT_CASES = settings(max_examples=25, deadline=None, derandomize=True)


@CONTRACT_CASES
@given(times=st.lists(st.integers(1, 90), min_size=2, max_size=6))
def test_neighbors_stay_in_their_match_and_period(times):
    rows = []
    for period in (1, 2):
        rows += lineup_start(period)
        for i, t in enumerate(sorted(times)):
            rows.append(dict(event_type="touch", actor_id=f"H{i % 2}", actor_team_id=HOME,
                             period=period, clock_s=float(t), x=float(i), y=0.))
        rows.append(dict(event_type="intermission", period=period, clock_s=100.))
    games = pd.concat([GAMES, GAMES.assign(match_id="M2")], ignore_index=True)
    raw = pd.concat([ev(rows), ev(rows).assign(match_id="M2")], ignore_index=True)
    cfg, quality = load_config(), {}
    events, _, _ = l0_events.preprocess(raw, games, cfg, quality)
    indices, stints, _, _ = l0_shifts.reconstruct_all(events, games, goalies_global=GOALIES)
    events["stint"] = indices
    nodes, _, _ = l0_events.build_nodes(events, games, stints, GEO, cfg, quality)
    assert set(zip(nodes.match_id, nodes.period)) == {("M1", 1), ("M1", 2), ("M2", 1), ("M2", 2)}
    for column in ("next_node", "prev_node"):
        linked = nodes[column].to_numpy() >= 0
        destinations = nodes.iloc[nodes.loc[linked, column].to_numpy()]
        np.testing.assert_array_equal(nodes.loc[linked, "match_id"], destinations.match_id)
        np.testing.assert_array_equal(nodes.loc[linked, "period"], destinations.period)


@CONTRACT_CASES
@given(second=st.integers(1, 1000), tail=st.integers(1, 100))
def test_equal_timestamp_substitution_uses_seq_and_conserves_exposure(second, tail):
    rows = lineup_start(1) + [fo(HOME, 0.), touch(HOME, "H1", float(second)),
                             sub(HOME, "H9", "H1", float(second)), touch(HOME, "H9", float(second)),
                             dict(event_type="whistle", period=1, clock_s=float(second + tail))]
    tables, _ = _full(rows)
    nodes, stints = tables["nodes"], tables["stints"].set_index("stint")
    old = nodes[nodes.actor_id == "H1"].iloc[0]
    new = nodes[nodes.actor_id == "H9"].iloc[0]
    assert old.first_seq < new.first_seq and old.clock == new.clock
    assert "H1" in stints.loc[old.stint].home_sk and "H9" not in stints.loc[old.stint].home_sk
    assert "H9" in stints.loc[new.stint].home_sk and "H1" not in stints.loc[new.stint].home_sk
    toi = tables["toi"].groupby("player_id").toi_s.sum()
    assert toi["H1"] == second and toi["H9"] == tail
    assert toi.sum() == 10 * (second + tail)


@CONTRACT_CASES
@given(second=st.integers(2, 100), identity=st.sampled_from([(None, HOME), (None, None), ("GHOST", HOME)]))
def test_unknown_identity_does_not_fabricate_a_player_shift(second, identity):
    actor, team = identity
    gap = dict(event_type="touch", actor_id=actor, actor_team_id=team, period=1,
               clock_s=float(second), x=1., y=0.)
    tables, _ = _full(lineup_start(1) + [fo(HOME, 0.), touch(HOME, "H1", float(second - 1)),
                   gap, touch(HOME, "H1", float(second + 1)),
                   dict(event_type="whistle", period=1, clock_s=float(second + 2))])
    nodes = tables["nodes"]
    assert "GHOST" not in set(tables["player_shifts"].player_id)
    if actor is None and team == HOME:
        unknown = nodes[nodes.clock == second].iloc[0]
        assert unknown.node_kind == "team" and pd.isna(unknown.actor_id)
        assert unknown.shift_link == "team_node"
    elif actor is None:
        after = nodes[(nodes.actor_id == "H1") & (nodes.clock == second + 1)].iloc[0]
        assert after.chain_gap_before and after.prev_node == -1
    else:
        unknown = nodes[nodes.actor_id == "GHOST"].iloc[0]
        assert unknown.shift_link != "linked" and pd.isna(unknown.player_shift_id)


@CONTRACT_CASES
@given(seed=st.integers(0, 10000), values=st.lists(st.integers(0, 1200), min_size=1, max_size=10))
def test_same_inputs_and_configuration_roundtrip_through_cache(seed, values):
    with TemporaryDirectory(prefix="datamatch-contract-") as directory:
        cfg = load_config(overrides={"seed": seed, "paths": {"cache": directory}})
        first, second = Cache(cfg, {"fixture": "synthetic-v1"}), Cache(cfg, {"fixture": "synthetic-v1"})
        key = first.key("property_fixture", ("seed", "l0"), code=("l0_time",))
        assert key == second.key("property_fixture", ("seed", "l0"), code=("l0_time",))
        frame = pd.DataFrame({"clock_s": values})
        first.save_df(key, "events", frame)
        first.mark_done(key)
        assert second.done(key)
        pd.testing.assert_frame_equal(frame, second.load_df(key, "events"))
        changed = Cache(cfg.with_overrides({"seed": seed + 1}), {"fixture": "synthetic-v1"})
        assert changed.key("property_fixture", ("seed", "l0"), code=("l0_time",)) != key
