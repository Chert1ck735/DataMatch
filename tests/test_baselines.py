"""Базовые показатели игроков: ставки на 60, on-ice атрибуция, стягивание по амплуа.

Ожидаемые значения выведены ручным расчётом на крошечных синтетических таблицах.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datamatch import baselines as bl

H, A = "TH", "TA"
HOME5 = ["h1", "h2", "h3", "h4", "h5"]
AWAY5 = ["a1", "a2", "a3", "a4", "a5"]


def _stints() -> pd.DataFrame:
    """stint 1 — 5×5, stint 2 — большинство хозяев (5×4), stint 3 — недопущенный к rate-метрикам."""
    return pd.DataFrame({
        "match_id": ["M1"] * 3, "stint": [1, 2, 3],
        "home_sk": [HOME5, HOME5, HOME5], "away_sk": [AWAY5, AWAY5[:4], AWAY5],
        "home_team_id": [H] * 3, "is_5v5": [True, False, True],
        "valid_for_rate_metrics": [True, True, False],
    })


def _nodes() -> pd.DataFrame:
    """Гол хозяев в 5×5 (h1), бросок гостей в 5×4 (a1), гол хозяев в недопущенном stint, пас."""
    return pd.DataFrame({
        "match_id": ["M1"] * 4, "seq": [10, 20, 30, 40], "stint": [1, 2, 3, 1],
        "period": [1, 1, 2, 2], "action": ["shot", "shot", "shot", "pass"],
        "actor_id": ["h1", "a1", "h2", "h3"], "team_id": [H, A, H, H],
        "is_goal": [True, False, True, False],
    })


def _events() -> pd.DataFrame:
    """Передачи к голу h1 (stint 1), к голу h2 (stint 3, недопущен) и в серии буллитов (period 0)."""
    return pd.DataFrame({
        "match_id": ["M1"] * 5, "stint": [1, 1, 3, 3, 1], "period": [1, 1, 2, 2, 0],
        "event_type": ["assist_primary", "assist_secondary", "assist_primary", "goal", "assist_primary"],
        "actor_id": ["h2", "h3", "h4", "h2", "h5"],
    })


def _toi() -> pd.DataFrame:
    players = HOME5 + AWAY5 + ["g1"]
    teams = [H] * 5 + [A] * 5 + [H]
    return pd.DataFrame({"match_id": "M1", "player_id": players, "team_id": teams,
                         "toi_rate_s": 1800.0, "toi_rate_5v5_s": 1200.0})


def _roles() -> pd.Series:
    r = {p: "W" for p in HOME5 + AWAY5}
    r.update({"h1": "C", "a1": "D", "g1": "G"})
    return pd.Series(r, name="role")


def _pm(xg=None) -> pd.DataFrame:
    return bl.player_match_frame(_nodes(), _stints(), _toi(), _events(), _roles(), xg=xg)


def test_shot_table_keeps_shots_and_aligns_xg_by_match_seq():
    xg = pd.Series([0.3, 0.1, 0.5], index=pd.MultiIndex.from_tuples(
        [("M1", 20), ("M1", 10), ("M1", 30)], names=["match_id", "seq"]))
    s = bl.shot_table(_nodes(), _stints(), xg=xg)
    assert list(s.seq) == [10, 20, 30]
    assert np.allclose(s.xg, [0.1, 0.3, 0.5])
    assert list(s.valid) == [True, True, False]
    assert list(s.is_home) == [True, False, True]


def test_shot_table_rejects_xg_with_missing_shots():
    xg = pd.Series([0.3], index=pd.MultiIndex.from_tuples([("M1", 10)], names=["match_id", "seq"]))
    with pytest.raises(ValueError, match="xG"):
        bl.shot_table(_nodes(), _stints(), xg=xg)


def test_onice_goal_credits_exactly_on_ice_skaters_with_sign():
    pm = _pm().set_index("player_id")
    assert pm.loc[HOME5, "gf"].tolist() == [1.0] * 5
    assert pm.loc[AWAY5, "ga"].tolist() == [1.0] * 5
    assert pm.gf.sum() == 5.0 and pm.ga.sum() == 5.0          # гол в stint 3 не засчитан
    assert pm.loc[HOME5, "gf_5v5"].tolist() == [1.0] * 5
    # бросок гостей в 5×4: a5 не на льду, хозяева — против
    assert pm.loc[AWAY5[:4], "cf"].tolist() == [1.0] * 4 and pm.loc["a5", "cf"] == 0.0
    assert pm.loc[HOME5, "ca"].tolist() == [1.0] * 5
    assert pm.loc[HOME5, "ca_5v5"].tolist() == [0.0] * 5        # 5×4 — не 5×5
    assert pm.loc[HOME5, "cf"].tolist() == [1.0] * 5


def test_individual_counts_use_rate_eligible_stints_only():
    pm = _pm().set_index("player_id")
    assert pm.loc["h1", "igoals"] == 1.0 and pm.loc["h1", "ishots"] == 1.0
    assert pm.loc["h2", "igoals"] == 0.0                         # гол в недопущенном stint
    assert pm.loc["h2", "a1"] == 1.0 and pm.loc["h3", "a2"] == 1.0
    assert pm.loc["h4", "a1"] == 0.0 and pm.loc["h5", "a1"] == 0.0  # недопущен / буллиты
    assert pm.loc["h2", "points"] == 1.0 and pm.loc["h1", "points"] == 1.0


def test_goalies_and_unknown_roles_are_excluded():
    pm = _pm()
    assert "g1" not in set(pm.player_id)
    assert set(pm.role) <= set(bl.FIELD_ROLES)
    assert len(pm) == 10


def test_xg_columns_only_when_xg_given():
    assert "xgf" not in _pm().columns
    xg = pd.Series([0.4, 0.2, 0.9], index=pd.MultiIndex.from_tuples(
        [("M1", 10), ("M1", 20), ("M1", 30)], names=["match_id", "seq"]))
    pm = _pm(xg).set_index("player_id")
    assert pm.loc["h1", "ixg"] == pytest.approx(0.4)
    assert pm.loc["h3", "xgf_5v5"] == pytest.approx(0.4)
    assert pm.loc["h3", "xga"] == pytest.approx(0.2) and pm.loc["h3", "xga_5v5"] == 0.0


def test_metric_frame_and_per60_math():
    pm = _pm()
    mf = bl.metric_frame(pm, bl.METRICS["gd_60"])
    assert list(mf.columns) == bl.MF_COLS
    rates = bl.player_rates(mf).set_index("player_id")
    # +1 гол за 1800 с = 2,0 за 60 минут; −1 у гостей
    assert rates.loc["h1", "rate_60"] == pytest.approx(2.0)
    assert rates.loc["a2", "rate_60"] == pytest.approx(-2.0)
    r5 = bl.player_rates(bl.metric_frame(pm, bl.METRICS["gd_5v5_60"])).set_index("player_id")
    assert r5.loc["h1", "rate_60"] == pytest.approx(3.0)        # 1 / 1200 с × 3600


def test_metric_frame_drops_zero_toi_rows_with_numerator():
    pm = _pm()
    pm = pm.assign(toi_rate_s=np.where(pm.player_id == "h1", 0.0, pm.toi_rate_s))
    mf = bl.metric_frame(pm, bl.METRICS["points_60"])
    assert "h1" not in set(mf.player_id)
    assert (mf.toi_s > 0).all()


def test_xg_metric_requires_xg_columns():
    with pytest.raises(KeyError, match="xgf_5v5"):
        bl.metric_frame(_pm(), bl.METRICS["xgd_5v5_60"])
    assert "xgd_5v5_60" not in bl.available_metrics(_pm())
    assert "cd_5v5_60" in bl.available_metrics(_pm())


# --- стягивание по амплуа ---------------------------------------------------------------

def _mf_from(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=bl.MF_COLS)


def test_shrinkage_hand_formula():
    params = bl.ShrinkParams(mu={"C": 2.0}, tau2={"C": 1.0}, sigma2={"C": 1.0}, n={"C": 50},
                             pooled={"C": False})
    rates = pd.DataFrame({"player_id": ["p"], "role": ["C"], "num": [4.0], "toi_s": [3600.0],
                          "games": [1], "rate_60": [4.0]})
    out = bl.apply_role_shrinkage(rates, params)
    # B = τ² / (τ² + σ²/h) = 1 / (1 + 1/1) = 0,5;  θ = 2 + 0,5 (4 − 2) = 3
    assert out.b.iloc[0] == pytest.approx(0.5)
    assert out.rate_shrunk.iloc[0] == pytest.approx(3.0)
    assert rates.columns.tolist() == ["player_id", "role", "num", "toi_s", "games", "rate_60"]


def test_shrinkage_huge_toi_returns_raw_and_zero_toi_returns_mean():
    params = bl.ShrinkParams(mu={"W": 0.0}, tau2={"W": 0.5}, sigma2={"W": 10.0}, n={"W": 50},
                             pooled={"W": False})
    rates = pd.DataFrame({"player_id": ["big", "none"], "role": ["W", "W"], "num": [1e8, 0.0],
                          "toi_s": [1e11, 0.0], "games": [1, 0], "rate_60": [3.6, np.nan]})
    out = bl.apply_role_shrinkage(rates, params).set_index("player_id")
    assert out.loc["big", "rate_shrunk"] == pytest.approx(3.6, rel=1e-6)
    assert out.loc["none", "rate_shrunk"] == 0.0


def test_fit_shrinkage_zero_between_variance_gives_role_mean():
    rows = []
    for i in range(30):
        rows.append((f"p{i}", "m1", "D", 1.0, 3600.0))   # ставка 1
        rows.append((f"p{i}", "m2", "D", 3.0, 3600.0))   # ставка 3 → у всех среднее 2
    mf = _mf_from(rows)
    params = bl.fit_role_shrinkage(mf, min_toi_s=0.0)
    assert params.tau2["D"] == 0.0
    assert params.sigma2["D"] == pytest.approx(2.0)          # Σ h (x − x̄)² / Σ (M − 1) = 2/1
    out = bl.apply_role_shrinkage(bl.player_rates(mf), params)
    assert np.allclose(out.rate_shrunk, 2.0)


def test_fit_shrinkage_recovers_known_moments():
    rng = np.random.default_rng(7)
    tau2_true, n_players, n_matches = 1.0, 400, 30
    theta = rng.normal(2.0, np.sqrt(tau2_true), n_players)
    rows = []
    for i in range(n_players):
        h = rng.uniform(0.2, 0.4, n_matches)                  # часы в матче
        x = theta[i] + rng.normal(0.0, np.sqrt(4.0 / h))      # σ² = 4 на час
        rows += [(f"p{i}", f"m{m}", "C", x[m] * h[m], h[m] * 3600.0) for m in range(n_matches)]
    params = bl.fit_role_shrinkage(_mf_from(rows), min_toi_s=0.0)
    assert params.sigma2["C"] == pytest.approx(4.0, rel=0.1)
    assert params.tau2["C"] == pytest.approx(tau2_true, rel=0.3)
    assert params.mu["C"] == pytest.approx(2.0, abs=0.15)


def test_fit_shrinkage_rejects_bad_input():
    with pytest.raises(KeyError):
        bl.fit_role_shrinkage(pd.DataFrame({"player_id": ["a"]}))
    with pytest.raises(ValueError, match="role"):
        bl.fit_role_shrinkage(_mf_from([("a", "m", "X", 1.0, 60.0)]), min_toi_s=0.0)
