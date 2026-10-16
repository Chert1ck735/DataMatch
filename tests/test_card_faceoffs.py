"""Карточка, вбрасывания: Bradley–Terry с поправкой на соперника, зону и площадку (METHODOLOGY 11.6)."""
import numpy as np
import pandas as pd
import pytest

from datamatch import card_faceoffs as cf


def test_home_perspective_hand_case():
    df = pd.DataFrame({"match_id": ["M1", "M1"], "actor_id": ["A", "B"], "other_id": ["B", "A"],
                       "actor_team_id": ["H", "V"], "home_team_id": ["H", "H"], "zone": ["off", "off"]})
    out = cf.home_perspective(df)
    # 1) выиграл игрок хозяев A в своей атакующей зоне; 2) выиграл гость B в своей атакующей = защитная хозяев
    assert out.y.tolist() == [1, 0]
    assert out.home_pid.tolist() == ["A", "A"] and out.away_pid.tolist() == ["B", "B"]
    assert out.zone_home.tolist() == ["off", "def"]


def _synthetic(n_players=60, n=20000, seed=0, home=0.1, zdef=0.3):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.4, n_players)
    h, a = rng.integers(0, n_players, n), rng.integers(0, n_players, n)
    keep = h != a
    h, a = h[keep], a[keep]
    zone = rng.choice(["off", "def", "neutral"], len(h))
    eta = theta[h] - theta[a] + home + np.where(zone == "def", zdef, 0.0) + np.where(zone == "off", -zdef, 0.0)
    y = (rng.random(len(h)) < 1 / (1 + np.exp(-eta))).astype(int)
    ids = np.array([f"P{i}" for i in range(n_players)])
    return pd.DataFrame({"y": y, "home_pid": ids[h], "away_pid": ids[a], "zone_home": zone}), theta, ids


def test_fit_recovers_skills_and_context():
    df, theta, ids = _synthetic()
    fit = cf.fit_bt(df, tau2=0.2)
    est = fit.theta.reindex(ids).to_numpy()
    assert np.corrcoef(est, theta)[0, 1] > 0.9
    assert fit.beta["home"] == pytest.approx(0.1, abs=0.06)
    assert fit.beta["def"] == pytest.approx(0.3, abs=0.08)
    assert fit.beta["off"] == pytest.approx(-0.3, abs=0.08)
    assert (fit.sd > 0).all()


def test_tiny_prior_shrinks_skills_to_zero():
    df, _, _ = _synthetic(n=3000)
    fit = cf.fit_bt(df, tau2=1e-6)
    assert np.abs(fit.theta).max() < 0.01


def test_frozen_context_applied_to_new_data():
    df, _, _ = _synthetic(n=5000, seed=1)
    frozen = {"home": 0.5, "def": 0.0, "off": 0.0}
    fit = cf.fit_bt(df, tau2=0.2, beta=frozen)
    assert fit.beta == frozen


def test_choose_tau2_returns_grid_value_and_scores():
    df, _, _ = _synthetic(n=6000, seed=2)
    df = df.assign(match_id=[f"M{i // 30}" for i in range(len(df))])
    best, scores = cf.choose_tau2(df, grid=(0.01, 0.2), n_folds=3, seed=0)
    assert best in (0.01, 0.2) and set(scores) == {0.01, 0.2}
    assert best == 0.2                      # истинный разброс навыков 0,4² = 0,16 — ближе к 0,2
