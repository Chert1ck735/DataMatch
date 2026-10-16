"""Карточка, пункт 5: физический профиль — решение по разработке train, объединённая выборка описательно."""
import numpy as np
import pandas as pd
import pytest

from datamatch import phys_profile as pp
from datamatch.card_phys import (
    LEVEL_PLAYER,
    apply_level,
    card_level,
    decision_table,
    pooled_descriptive,
    restrict_to_matches,
)

N_PER_ROLE = 30
OBS_PER_MATCH = 8


def _roles() -> pd.Series:
    ids = [f"P{i:03d}" for i in range(3 * N_PER_ROLE)]
    return pd.Series(np.repeat(pp.ROLES, N_PER_ROLE), index=ids, name="role")


def _obs(match_ids: list[str], signal_sd: float, seed: int) -> pd.DataFrame:
    """Каждый игрок в каждом матче: value = эффект игрока · signal_sd + N(0, 1)."""
    rng = np.random.default_rng(seed)
    roles = _roles()
    effect = pd.Series(rng.normal(size=len(roles)), index=roles.index)
    rows = []
    for m in match_ids:
        for p in roles.index:
            vals = effect[p] * signal_sd + rng.normal(size=OBS_PER_MATCH)
            rows.append(pd.DataFrame({"match_id": m, "player_id": p, "value": vals}))
    return pd.concat(rows, ignore_index=True)


DEV = [f"D{i:03d}" for i in range(40)]
OUTER = [f"O{i:03d}" for i in range(40)]
SPEC = pp.MetricSpec("synthetic", agg="mean", min_obs_half=5)


def test_restrict_keeps_only_listed_matches():
    obs = pd.concat([_obs(DEV[:2], 0.0, 1), _obs(OUTER[:2], 0.0, 2)], ignore_index=True)
    out = restrict_to_matches(obs, set(DEV))
    assert set(out.match_id) == set(DEV[:2])
    assert len(out) == 2 * 3 * N_PER_ROLE * OBS_PER_MATCH


def test_restrict_does_not_mutate_input():
    obs = _obs(DEV[:1] + OUTER[:1], 0.0, 3)
    before = obs.copy()
    restrict_to_matches(obs, set(DEV))
    pd.testing.assert_frame_equal(obs, before)


@pytest.mark.parametrize("passed, fallback, expected", [
    (True, "ROLE", LEVEL_PLAYER),
    (True, "LINE", LEVEL_PLAYER),
    (False, "ROLE", "уровень амплуа"),
    (False, "LINE", "характеристика звена"),
])
def test_card_level(passed, fallback, expected):
    assert card_level(passed, fallback) == expected


def test_card_level_rejects_unknown_fallback():
    with pytest.raises(ValueError):
        card_level(False, "TEAM")


def test_decision_comes_from_dev_only_even_if_pooled_passes():
    # Arrange: на разработке сигнала нет, на остальных матчах — сильный (SB ≫ 0,3).
    roles = _roles()
    dev = _obs(DEV, 0.0, 11)
    other = _obs(OUTER, 1.0, 12)
    rel_dev = pp.reliability(dev, roles, SPEC, n_boot=0)

    # Act
    table = decision_table({"synthetic": rel_dev}, {"synthetic": "LINE"})
    pooled = pooled_descriptive(dev, other, roles, SPEC)

    # Assert
    row = table.set_index("metric").loc["synthetic"]
    assert row.split_half_sb_dev < pp.RELIABILITY_GATE
    assert not row.passed
    assert row.level == "характеристика звена"
    assert pooled["split_half_sb"] > pp.RELIABILITY_GATE
    assert pooled["descriptive_only"] is True
    assert "gate_passed" not in pooled


def test_decision_passes_with_strong_signal_on_dev():
    roles = _roles()
    rel_dev = pp.reliability(_obs(DEV, 1.0, 21), roles, SPEC, n_boot=0)
    row = decision_table({"synthetic": rel_dev}, {"synthetic": "ROLE"}).iloc[0]
    assert row.passed and row.level == LEVEL_PLAYER
    assert row.gate == pp.RELIABILITY_GATE


def test_decision_nan_reliability_fails_gate():
    rel = {"split_half_sb": float("nan"), "n_players": 0, "gate_passed": False}
    row = decision_table({"m": rel}, {"m": "ROLE"}).iloc[0]
    assert not row.passed and row.level == "уровень амплуа"


def test_pooled_rejects_overlapping_matches():
    roles = _roles()
    a = _obs(DEV[:20], 0.5, 31)
    with pytest.raises(ValueError):
        pooled_descriptive(a, a, roles, SPEC)


def test_decision_agrees_with_phys_profile_gate_flag():
    rel = {"split_half_sb": 0.31, "n_players": 50, "gate_passed": False}   # расхождение с порогом 0,3
    with pytest.raises(ValueError):
        decision_table({"m": rel}, {"m": "ROLE"})


def test_decision_gate_boundary_exactly_threshold_passes():
    rel = {"split_half_sb": pp.RELIABILITY_GATE, "n_players": 50, "gate_passed": True}
    assert decision_table({"m": rel}, {"m": "ROLE"}).iloc[0].passed


def _csv():
    return pd.DataFrame({"player_id": ["A", "B", "C"], "role": ["C", "C", "D"],
                         "speed": [7.4, 7.0, 6.1], "speed_raw": [7.9, 6.5, 6.0],
                         "speed_weight": [0.06, 0.05, 0.04], "speed_label": ["ROLE", "ROLE", "ROLE"]})


def test_apply_level_role_shows_role_prior_not_player_value():
    out = apply_level(_csv(), "speed", "уровень амплуа", {"C": 7.2, "D": 6.3, "W": 7.1})
    assert out.speed.tolist() == [7.2, 7.2, 6.3]
    assert out.speed_raw.isna().all() and out.speed_weight.isna().all()
    assert (out.speed_level == "уровень амплуа").all()


def test_apply_level_line_keeps_values_labelled_as_line():
    src = _csv()
    out = apply_level(src, "speed", "характеристика звена", {"C": 7.2, "D": 6.3, "W": 7.1})
    assert out.speed.tolist() == src.speed.tolist()
    assert (out.speed_level == "характеристика звена").all()
    assert "speed_level" not in src.columns            # вход не меняется


def test_apply_level_player_keeps_values():
    out = apply_level(_csv(), "speed", LEVEL_PLAYER, {"C": 7.2, "D": 6.3, "W": 7.1})
    assert out.speed.tolist() == [7.4, 7.0, 6.1] and out.speed_raw.notna().all()
