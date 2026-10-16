"""Independently check fresh L0 rate/identity/censoring contracts on saved real tables."""
import json

import numpy as np
import pandas as pd

from datamatch import io, stage_l0
from datamatch.cache import Cache
from datamatch.config import load_config


def verify():
    cfg = load_config()
    cache = Cache(cfg, io.data_version(cfg))
    key = cache.key("l0", ("l0",), code=stage_l0.L0_CODE)
    if not cache.done(key):
        raise RuntimeError("Current L0 calculation has not completed")
    quality = cache.load_obj(key, "quality")
    assert quality["status"] == "ok" and all(quality["gate_l0_all"].values())
    report = {"cache_key": key, "status": quality["status"], "gates_passed": len(quality["gate_l0_all"])}
    for source in ("train", "test"):
        path = cache.root / key
        nd = pd.read_parquet(path / f"{source}_nodes.parquet", columns=["node", "match_id", "stint", "scheme",
            "actor_id", "is_home", "is_shot", "is_goal", "shift_link", "valid_for_lineup_attribution",
            "valid_for_action_attribution", "valid_for_rate_metrics", "rate_match_eligible", "lineup_status",
            "next_node", "prev_node", "terminal", "transition_status", "valid_for_transition", "segment",
            "first_seq", "last_seq", "chain_gap_before", "possession"])
        st = pd.read_parquet(path / f"{source}_stints.parquet", columns=["match_id", "stint", "duration",
            "home_sk", "away_sk", "home_g", "away_g", "home_g_in", "away_g_in", "n_home_sk", "n_away_sk",
            "valid_for_rates", "valid_for_rate_metrics", "rate_match_eligible"])
        toi = pd.read_parquet(path / f"{source}_toi.parquet")
        j = st.set_index(["match_id", "stint"]).reindex(pd.MultiIndex.from_arrays([nd.match_id, nd.stint]))
        # Independent per-action identity check, directly against the seq-selected lineup.
        present = np.array([(actor in (home if is_home else away)) or
            (actor is not None and actor == (hg if is_home else ag) and (hgi if is_home else agi))
            for actor, is_home, home, away, hg, ag, hgi, agi in zip(nd.actor_id, nd.is_home,
                j.home_sk, j.away_sk, j.home_g, j.away_g, j.home_g_in, j.away_g_in)], dtype=bool)
        attributed = nd.valid_for_action_attribution.to_numpy()
        assert not (attributed & ~present).any()
        assert not ((nd.shift_link == "no_shift") & attributed).any()
        expected_rate = attributed & nd.rate_match_eligible.to_numpy()
        np.testing.assert_array_equal(nd.valid_for_rate_metrics, expected_rate)
        # Independent hockey capacity/identity check; do not use the production helper.
        impossible = np.array([len(h) < 3 or len(a) < 3 or
            len(h) + int(hgi) > 6 or len(a) + int(agi) > 6 or
            len(set(list(h) + ([hg] if hgi else []))) != len(h) + int(hgi) or
            len(set(list(a) + ([ag] if agi else []))) != len(a) + int(agi) or
            bool(set(list(h) + ([hg] if hgi else [])).intersection(list(a) + ([ag] if agi else [])))
            for h, a, hg, ag, hgi, agi in zip(st.home_sk, st.away_sk, st.home_g, st.away_g,
                st.home_g_in, st.away_g_in)], dtype=bool)
        assert not (impossible & st.valid_for_rates).any()
        impossible_node = pd.Series(impossible, index=pd.MultiIndex.from_frame(st[['match_id', 'stint']])).reindex(
            pd.MultiIndex.from_arrays([nd.match_id, nd.stint])).fillna(True).to_numpy()
        assert not (impossible_node & attributed).any()
        pos = st.duration > 0
        expected_exposure = pos & st.valid_for_rates & st.rate_match_eligible
        np.testing.assert_array_equal(st.valid_for_rate_metrics, st.valid_for_rates & st.rate_match_eligible)
        raw_rate = (st.duration[expected_exposure] *
                    (st.n_home_sk[expected_exposure] + st.n_away_sk[expected_exposure]))
        expected_toi = raw_rate.groupby(st.match_id[expected_exposure]).sum()
        actual_toi = toi.groupby("match_id").toi_rate_s.sum()
        assert actual_toi.subtract(expected_toi, fill_value=0.).abs().max() < 1e-6
        low_matches = st.loc[~st.rate_match_eligible, "match_id"].unique()
        assert toi.loc[toi.match_id.isin(low_matches), "toi_rate_s"].sum() == 0.
        unknown = nd.transition_status == "unknown"
        assert not nd.loc[unknown, "terminal"].any()
        assert (nd.loc[unknown, "next_node"] == -1).all()
        assert not nd.loc[unknown, "valid_for_transition"].any()
        assert (nd.loc[nd.terminal, "next_node"] == -1).all()
        linked = nd.next_node >= 0
        destinations = nd.iloc[nd.loc[linked, "next_node"].to_numpy()]
        np.testing.assert_array_equal(nd.loc[linked, "match_id"], destinations.match_id)
        np.testing.assert_array_equal(nd.loc[linked, "segment"], destinations.segment)
        np.testing.assert_array_equal(destinations.prev_node, nd.loc[linked, "node"])
        # For every post-gap retained node with a predecessor in its raw segment,
        # both directions of the spatial graph must have been cut before history.
        prev = nd.groupby(["match_id", "segment"], sort=False).node.shift(1)
        gap_after = nd.chain_gap_before & prev.notna()
        before = nd.iloc[prev[gap_after].astype(int).to_numpy()]
        assert (~before.valid_for_transition).all() and not before.terminal.any()
        assert (nd.loc[gap_after, "prev_node"] == -1).all()
        excluded = pd.read_parquet(path / f"{source}_excluded.parquet")
        events = pd.read_parquet(path / f"{source}_events.parquet", columns=["match_id", "seq", "segment"])
        expected_segments = events.set_index(["match_id", "seq"]).segment.reindex(
            pd.MultiIndex.from_arrays([excluded.match_id, excluded.seq]))
        np.testing.assert_array_equal(excluded.segment, expected_segments)
        ps = pd.read_parquet(path / f"{source}_possessions.parquet")
        reason = ps.set_index("possession").end_reason.reindex(nd.loc[unknown, "possession"])
        assert (reason == "observation_gap").all()
        zero_shots = nd.is_shot & (nd.lineup_status == "zero_duration") & (nd.scheme == "v1") & attributed
        report[source] = {"matches": int(nd.match_id.nunique()), "nodes": len(nd),
            "impossible_positive_stints": int((impossible & pos).sum()),
            "impossible_stints_marked_eligible": int((impossible & st.valid_for_rates).sum()),
            "zero_duration_v1_attributed_shots": int(zero_shots.sum()),
            "zero_duration_v1_attributed_goals": int((zero_shots & nd.is_goal).sum()),
            "zero_duration_v1_rate_shots": int((zero_shots & nd.valid_for_rate_metrics).sum()),
            "no_shift_attributed": int(((nd.shift_link == "no_shift") & attributed).sum()),
            "unknown_transitions": int(unknown.sum()), "unknown_terminal": int((unknown & nd.terminal).sum()),
            "excluded_rate_matches": len(low_matches),
            "excluded_match_diagnostic_toi_s": float(toi.loc[toi.match_id.isin(low_matches), "toi_valid_s"].sum()),
            "excluded_match_rate_toi_s": float(toi.loc[toi.match_id.isin(low_matches), "toi_rate_s"].sum())}
    destination = cfg.paths.outputs / "l0_followup_verification.json"
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    verify()
