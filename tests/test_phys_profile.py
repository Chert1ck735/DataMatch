"""Ветка Ф (Ф1/Ф2): ручные расчёты скорости, границы пар, фильтры и split-half надёжность.

Ожидания выведены независимо от реализации: 10 м за 2 с — 5 м/с; пара не пересекает игрока,
отрезок между остановками, период и матч; split-half на синтетике с известной дисперсией
таланта τ² и шумом наблюдения σ² сравнивается с теоретической надёжностью
τ² / (τ² + σ²/n_half), а на чистом шуме должна быть около нуля.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datamatch import phys_profile as pp


def _ev(rows: list[dict]) -> pd.DataFrame:
    """Минимальная таблица узлов: значения по умолчанию — один матч, период, отрезок."""
    base = {"match_id": "M1", "period": 1, "segment": 1, "team_id": "T1", "y": 0.0,
            "clock_imputed": False, "t_behind_scale": False, "chain_gap_before": False,
            "stint": 0, "home_team_id": "T1"}
    out = [{**base, "seq": i, **r} for i, r in enumerate(rows)]
    return pd.DataFrame(out)


def _pair_speed(rows: list[dict]) -> pd.DataFrame:
    kept, _ = pp.filter_speed_pairs(pp.candidate_pairs(_ev(rows)))
    return kept


# --- Ф1: скорость пары --------------------------------------------------------------------

def test_two_touches_10m_in_2s_give_5_mps():
    kept = _pair_speed([
        {"actor_id": "A", "primary": "touch", "t_mono": 10.0, "x": 0.0, "y": 0.0},
        {"actor_id": "A", "primary": "touch", "t_mono": 12.0, "x": 6.0, "y": 8.0},
    ])
    assert len(kept) == 1
    assert kept.speed.iat[0] == pytest.approx(5.0)
    assert kept.dist.iat[0] == pytest.approx(10.0)
    assert kept.dt.iat[0] == pytest.approx(2.0)


def test_input_frame_is_not_mutated():
    ev = _ev([
        {"actor_id": "A", "primary": "touch", "t_mono": 12.0, "x": 6.0, "y": 8.0},
        {"actor_id": "A", "primary": "touch", "t_mono": 10.0, "x": 0.0, "y": 0.0},
    ]).iloc[::-1]
    snapshot = ev.copy()
    pp.candidate_pairs(ev)
    pd.testing.assert_frame_equal(ev, snapshot)


@pytest.mark.parametrize("field,second", [
    ("segment", 2), ("period", 2), ("match_id", "M2"), ("actor_id", "B")])
def test_pairs_never_cross_boundaries(field, second):
    first = {"actor_id": "A", "primary": "touch", "t_mono": 10.0, "x": 0.0}
    nxt = {"actor_id": "A", "primary": "touch", "t_mono": 12.0, "x": 6.0, field: second}
    assert len(pp.candidate_pairs(_ev([first, nxt]))) == 0


def test_pair_uses_nearest_previous_event_of_same_player():
    kept = _pair_speed([
        {"actor_id": "A", "primary": "touch", "t_mono": 10.0, "x": 0.0},
        {"actor_id": "A", "primary": "touch", "t_mono": 11.0, "x": 2.0},
        {"actor_id": "A", "primary": "touch", "t_mono": 12.0, "x": 6.0},
    ])
    assert kept.speed.tolist() == pytest.approx([2.0, 4.0])


@pytest.mark.parametrize("dt,keep", [(0.4, False), (0.5, False), (0.6, True), (3.0, True), (3.5, False)])
def test_dt_bounds_are_open_below_closed_above(dt, keep):
    kept = _pair_speed([
        {"actor_id": "A", "primary": "touch", "t_mono": 10.0, "x": 0.0},
        {"actor_id": "A", "primary": "touch", "t_mono": 10.0 + dt, "x": 1.0},
    ])
    assert (len(kept) == 1) is keep


def test_implausible_speed_is_dropped_and_counted():
    pairs = pp.candidate_pairs(_ev([
        {"actor_id": "A", "primary": "touch", "t_mono": 10.0, "x": -20.0},
        {"actor_id": "A", "primary": "touch", "t_mono": 11.0, "x": -6.0},     # 14 м/с
        {"actor_id": "A", "primary": "touch", "t_mono": 12.0, "x": 6.0},      # 12 м/с
    ]))
    kept, counts = pp.filter_speed_pairs(pairs)
    assert kept.speed.tolist() == pytest.approx([12.0])
    assert counts["speed_implausible"] == 1
    assert counts["candidates"] == 2 and counts["kept"] == 1


@pytest.mark.parametrize("prev,nxt,keep", [
    ("touch", "shot_on_target", True), ("steal", "pass_complete", True), ("duel", "touch", True),
    ("pass_complete", "touch", False), ("touch", "dump_in_success", False),
    ("clear_out_success", "touch", False), ("touch", "deke_success", False)])
def test_event_type_whitelist(prev, nxt, keep):
    kept = _pair_speed([
        {"actor_id": "A", "primary": prev, "t_mono": 10.0, "x": 0.0},
        {"actor_id": "A", "primary": nxt, "t_mono": 12.0, "x": 6.0},
    ])
    assert (len(kept) == 1) is keep


def test_puck_event_of_other_player_between_interrupts_pair():
    kept = _pair_speed([
        {"actor_id": "A", "primary": "touch", "t_mono": 10.0, "x": 0.0},
        {"actor_id": "B", "primary": "touch", "t_mono": 11.0, "x": 3.0},
        {"actor_id": "A", "primary": "touch", "t_mono": 12.0, "x": 6.0},
    ])
    assert len(kept) == 0


def test_duel_of_other_player_between_does_not_interrupt():
    kept = _pair_speed([
        {"actor_id": "A", "primary": "touch", "t_mono": 10.0, "x": 0.0},
        {"actor_id": "B", "primary": "duel", "t_mono": 11.0, "x": 3.0, "team_id": "T2"},
        {"actor_id": "A", "primary": "touch", "t_mono": 12.0, "x": 6.0},
    ])
    assert kept.speed.tolist() == pytest.approx([3.0])


@pytest.mark.parametrize("flag", ["clock_imputed", "t_behind_scale", "chain_gap_before"])
def test_untrusted_time_or_chain_gap_drops_pair(flag):
    kept = _pair_speed([
        {"actor_id": "A", "primary": "touch", "t_mono": 10.0, "x": 0.0},
        {"actor_id": "A", "primary": "touch", "t_mono": 12.0, "x": 6.0, flag: True},
    ])
    assert len(kept) == 0


def test_missing_required_column_raises():
    with pytest.raises(ValueError, match="t_mono"):
        pp.candidate_pairs(_ev([{"actor_id": "A", "primary": "touch", "x": 0.0}]))


# --- Ф2: скорость шайбы в передаче и темп перехода -----------------------------------------

def test_pass_speed_uses_x2_and_receiver_next_event():
    ev = _ev([
        {"actor_id": "A", "other_id": "B", "primary": "pass_complete", "t_mono": 10.0, "x": 0.0,
         "x2": 20.0, "y2": 0.0},
        {"actor_id": "B", "other_id": None, "primary": "touch", "t_mono": 11.0, "x": 25.0,
         "x2": np.nan, "y2": np.nan},
    ])
    obs = pp.pass_speed_obs(ev)
    assert obs.player_id.tolist() == ["A"]
    assert obs.speed.iat[0] == pytest.approx(20.0)


def test_pass_speed_falls_back_to_receiver_position_and_requires_receiver():
    ev = _ev([
        {"actor_id": "A", "other_id": "B", "primary": "pass_complete", "t_mono": 10.0, "x": 0.0,
         "x2": np.nan, "y2": np.nan},
        {"actor_id": "B", "other_id": None, "primary": "touch", "t_mono": 11.0, "x": 15.0,
         "x2": np.nan, "y2": np.nan},
        {"actor_id": "B", "other_id": "C", "primary": "pass_complete", "t_mono": 12.0, "x": 15.0,
         "x2": 30.0, "y2": 0.0},
        {"actor_id": "D", "other_id": None, "primary": "touch", "t_mono": 13.0, "x": 30.0,
         "x2": np.nan, "y2": np.nan},
    ])
    obs = pp.pass_speed_obs(ev)
    assert obs.player_id.tolist() == ["A"]
    assert obs.speed.iat[0] == pytest.approx(15.0)


def _poss_nodes(rows: list[dict]) -> pd.DataFrame:
    base = {"possession": 7, "has_steal": False, "entry": False, "is_shot": False,
            "valid_for_transition": True, "mp": "EV5", "actor_id": "A", "x": 0.0}
    return _ev([{**base, **r} for r in rows])


def test_transition_time_from_def_zone_steal_to_offensive_zone():
    nodes = _poss_nodes([
        {"primary": "steal", "zone": "def", "t_mono": 10.0, "has_steal": True},
        {"primary": "touch", "zone": "neutral", "t_mono": 12.0},
        {"primary": "touch", "zone": "off", "t_mono": 14.0},
        {"primary": "shot_on_target", "zone": "off", "t_mono": 15.0, "is_shot": True},
    ])
    tr = pp.transition_table(nodes)
    assert len(tr) == 1
    assert tr.t_transition.iat[0] == pytest.approx(4.0)
    assert tr.t_first_shot.iat[0] == pytest.approx(5.0)
    assert bool(tr.reached.iat[0])


def test_transition_requires_def_zone_recovery_start():
    nodes = _poss_nodes([
        {"primary": "touch", "zone": "def", "t_mono": 10.0},
        {"primary": "touch", "zone": "off", "t_mono": 14.0},
    ])
    assert len(pp.transition_table(nodes)) == 0


def test_censored_transition_has_unknown_outcome():
    nodes = _poss_nodes([
        {"primary": "steal", "zone": "def", "t_mono": 10.0, "has_steal": True},
        {"primary": "touch", "zone": "neutral", "t_mono": 12.0, "valid_for_transition": False},
    ])
    tr = pp.transition_table(nodes)
    assert len(tr) == 1
    assert bool(tr.censored.iat[0]) and not bool(tr.reached.iat[0])
    assert np.isnan(tr.t_transition.iat[0])
    assert tr.t_observed.iat[0] == pytest.approx(2.0)
    assert np.isnan(pp.fast_attack_values(tr, 6.0).iat[0])
    assert pp.fast_attack_values(tr, 2.0).iat[0] == 0.0


def test_onice_expansion_gives_each_skater_one_observation_with_mates():
    poss = pd.DataFrame({"match_id": ["M1"], "value": [4.0], "skaters": [["A", "B", "C"]]})
    obs = pp.expand_onice(poss, "value")
    assert sorted(obs.player_id) == ["A", "B", "C"]
    mates = dict(zip(obs.player_id, obs.mates))
    assert sorted(mates["A"]) == ["B", "C"]


# --- надёжность -------------------------------------------------------------------------

def test_spearman_brown():
    assert pp.spearman_brown(0.5) == pytest.approx(2 / 3)
    assert pp.spearman_brown(0.0) == pytest.approx(0.0)


def test_halves_are_deterministic_balanced_and_not_match_order():
    ids = [f"M{i:03d}" for i in range(100)]
    h1 = pp.assign_halves(ids, seed=1)
    h2 = pp.assign_halves(list(reversed(ids)), seed=1)
    pd.testing.assert_series_equal(h1.sort_index(), h2.sort_index())
    assert h1.sum() == 50
    alternating = pd.Series(np.arange(100) % 2, index=ids)
    assert (h1.sort_index() != alternating).any()


def _synthetic_obs(rng, tau: float, sigma: float, n_players=400, n_matches=40, per_match=4):
    """Игрок играет все матчи; наблюдение = θ_i + шум. Амплуа — по кругу C/W/D."""
    rows = []
    roles = {}
    for p in range(n_players):
        pid = f"P{p:04d}"
        roles[pid] = pp.ROLES[p % 3]
        theta = 5.0 + tau * rng.standard_normal()
        for m in range(n_matches):
            vals = theta + sigma * rng.standard_normal(per_match)
            rows.extend({"match_id": f"M{m:03d}", "player_id": pid, "value": v} for v in vals)
    return pd.DataFrame(rows), pd.Series(roles)


def test_split_half_recovers_known_reliability():
    rng = np.random.default_rng(0)
    tau, sigma = 0.3, 1.0
    obs, roles = _synthetic_obs(rng, tau, sigma)
    spec = pp.MetricSpec("mean_value", agg="mean", min_obs_half=5)
    res = pp.reliability(obs, roles, spec, seed=3, n_boot=0)
    n_half = 40 / 2 * 4
    r_theory = tau ** 2 / (tau ** 2 + sigma ** 2 / n_half)
    assert res["r_half"] == pytest.approx(r_theory, abs=0.06)
    assert res["split_half_sb"] > pp.RELIABILITY_GATE
    assert res["gate_passed"]


def test_split_half_on_pure_noise_is_near_zero_and_fails_gate():
    rng = np.random.default_rng(1)
    obs, roles = _synthetic_obs(rng, 0.0, 1.0)
    spec = pp.MetricSpec("p80", agg="quantile", q=0.8, min_obs_half=5)
    res = pp.reliability(obs, roles, spec, seed=3, n_boot=50)
    assert abs(res["r_half"]) < 0.12
    assert not res["gate_passed"]
    lo, hi = res["boot_sb_ci"]
    assert lo == 0.0 and hi < pp.RELIABILITY_GATE   # надёжность не уходит ниже 0 (ревью 06.10.2026)


def test_bootstrap_ci_brackets_point_estimate_for_real_talent():
    rng = np.random.default_rng(2)
    obs, roles = _synthetic_obs(rng, 0.3, 1.0, n_players=200)
    spec = pp.MetricSpec("p80", agg="quantile", q=0.8, min_obs_half=5)
    res = pp.reliability(obs, roles, spec, seed=3, n_boot=60)
    lo, hi = res["boot_sb_ci"]
    assert lo > 0.0
    assert lo <= res["split_half_sb"] <= hi


def test_role_offsets_do_not_create_reliability():
    """Надёжность внутри амплуа: разные средние амплуа без личного таланта дают ≈ 0."""
    rng = np.random.default_rng(4)
    obs, roles = _synthetic_obs(rng, 0.0, 1.0)
    shift = obs.player_id.map(roles).map({"C": 0.0, "W": 2.0, "D": -2.0})
    obs = obs.assign(value=obs.value + shift)
    res = pp.reliability(obs, roles, pp.MetricSpec("m", agg="mean", min_obs_half=5), seed=3, n_boot=0)
    assert abs(res["r_half"]) < 0.12


def test_line_baseline_matches_individual_when_value_is_pure_line_effect():
    """Звенья по 5 игроков; значение — эффект звена + шум: звено предсказывает так же, как сам игрок."""
    rng = np.random.default_rng(5)
    rows, roles = [], {}
    for line in range(80):
        effect = 0.5 * rng.standard_normal()
        members = [f"L{line:02d}_{k}" for k in range(5)]
        for k, pid in enumerate(members):
            roles[pid] = pp.ROLES[k % 3]
        for m in range(30):
            for _ in range(3):
                val = 5.0 + effect + 0.3 * rng.standard_normal()
                rows.extend({"match_id": f"M{m:03d}", "player_id": pid, "value": val,
                             "mates": [q for q in members if q != pid]} for pid in members)
    obs = pd.DataFrame(rows)
    res = pp.reliability(obs, pd.Series(roles), pp.MetricSpec("m", agg="mean", min_obs_half=5),
                         seed=3, n_boot=0)
    assert res["r_line"] == pytest.approx(res["r_half"], abs=0.05)


def test_noise_model_and_shrinkage_weights():
    rng = np.random.default_rng(6)
    obs, roles = _synthetic_obs(rng, 0.3, 1.0)
    spec = pp.MetricSpec("m", agg="mean", min_obs_half=5)
    params = pp.fit_shrinkage(obs, roles, spec, seed=3)
    assert params["sigma2"] == pytest.approx(1.0, rel=0.2)
    for role in pp.ROLES:
        assert params["tau2"][role] == pytest.approx(0.09, abs=0.04)
    summary = pp.player_summary(obs, spec)
    shrunk = pp.apply_shrinkage(summary, roles, params)
    expected_w = params["tau2"]["C"] / (params["tau2"]["C"] + params["sigma2"] / 160
                                        + params["omega2"] / 40)
    c_rows = shrunk[shrunk.role == "C"]
    assert c_rows.weight.to_numpy() == pytest.approx(expected_w, rel=1e-6)
    assert set(shrunk.label) <= {"HIGH", "MEDIUM", "LOW"}
    dev_raw = (c_rows.value - params["mu"]["C"]).abs()
    dev_shr = (c_rows.shrunk - params["mu"]["C"]).abs()
    assert (dev_shr <= dev_raw + 1e-12).all()


def _confound_pairs(rng, speed_from_density: bool) -> tuple[pd.DataFrame, pd.Series]:
    rows, roles = [], {}
    for p in range(150):
        pid = f"P{p:03d}"
        roles[pid] = pp.ROLES[p % 3]
        per_match = int(rng.integers(3, 15))
        for m in range(4):
            for _ in range(per_match):
                base = 4.0 + 0.3 * per_match if speed_from_density else 6.0 + rng.standard_normal()
                rows.append({"match_id": f"M{m}", "player_id": pid, "speed": base, "dt": 2.0})
    return pd.DataFrame(rows), pd.Series(roles)


def test_confound_check_flags_speed_explained_by_event_density():
    rng = np.random.default_rng(7)
    res = pp.speed_confound_check(*_confound_pairs(rng, speed_from_density=True))
    assert res["refuted"] is True
    assert res["r_density"] == pytest.approx(1.0, abs=1e-6)


def test_confound_check_passes_independent_speed():
    rng = np.random.default_rng(8)
    res = pp.speed_confound_check(*_confound_pairs(rng, speed_from_density=False))
    assert res["refuted"] is False


def test_fast_attack_values_unknown_when_censored_without_shot():
    tr = pd.DataFrame({"t_first_shot": [3.0, np.nan, np.nan, 20.0, np.nan],
                       "t_observed": [5.0, 4.0, 4.0, 20.0, 12.0],
                       "censored": [False, False, True, True, True]})
    vals = pp.fast_attack_values(tr, 10.0)
    assert vals.tolist()[:2] == [1.0, 0.0]
    assert np.isnan(vals.iloc[2])
    assert vals.iloc[3] == 0.0 and vals.iloc[4] == 0.0


def test_onice_skaters_picks_actor_team_side():
    ev = pd.DataFrame({"match_id": ["M1", "M1", "M1"], "stint": [0, 0, 9],
                       "team_id": ["H", "A", "H"], "home_team_id": ["H", "H", "H"]})
    stints = pd.DataFrame({"match_id": ["M1"], "stint": [0], "home_sk": [["h1", "h2"]], "away_sk": [["a1"]]})
    out = pp.onice_skaters(ev, stints)
    assert out.tolist() == [["h1", "h2"], ["a1"], []]
