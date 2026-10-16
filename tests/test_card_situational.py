"""Карточка, ситуационные рейтинги (METHODOLOGY 11.3): ожидания — ручной расчёт и синтетика."""
import numpy as np
import pandas as pd
import pytest

from datamatch import card_situational as cs


def _pm():
    return pd.DataFrame({
        "match_id": ["M1", "M2"], "player_id": ["p", "p"], "role": ["C", "C"],
        "toi_rate_s": [900.0, 600.0], "toi_rate_5v5_s": [700.0, 0.0], "toi_rate_pp_s": [100.0, 120.0],
        "toi_rate_sh_s": [50.0, 0.0],
        **{f"c_{c}": [9.0, 9.0] for c in cs.BASE},
        **{f"c_{c}_5v5": [1.0, 0.0] for c in cs.BASE},
        **{f"c_{c}_pp": [2.0, 3.0] for c in cs.BASE},
        **{f"c_{c}_pk": [4.0, 0.0] for c in cs.BASE}})


def test_situation_frame_takes_situation_columns_and_exposure():
    f = cs.situation_frame(_pm(), "pp")
    assert f.toi_rate_s.tolist() == [100.0, 120.0]
    assert f.c_shot.tolist() == [2.0, 3.0]
    assert set(cs.COMP_COLS) <= set(f.columns)


def test_situation_frame_pk_uses_shorthanded_toi():
    f = cs.situation_frame(_pm(), "pk")
    assert f.toi_rate_s.tolist() == [50.0, 0.0]
    assert f.c_loss.tolist() == [4.0, 0.0]


def test_situation_frame_rejects_unknown_situation():
    with pytest.raises(KeyError):
        cs.situation_frame(_pm(), "4v4")


def test_split_half_reliability_identical_halves_is_one():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        lvl = rng.standard_normal()
        for m in range(4):                       # одинаковый уровень в каждом матче → половины равны
            rows.append({"player_id": f"p{i}", "match_id": f"M{m}", "role": "W", "toi_s": 600.0,
                         "value": lvl * 10.0})
    rel = cs.split_half_reliability(pd.DataFrame(rows), min_toi_s=300.0)
    assert rel["r"] == pytest.approx(1.0)
    assert rel["r_sb"] == pytest.approx(1.0)
    assert rel["n_players"] == 60


def test_split_half_reliability_hand_case_and_spearman_brown():
    # две роли, по 3 игрока; внутри роли половины A = [1,2,3], B = [1,3,2] (на 60, TOI 3600 с)
    rows = []
    for role, shift in (("C", 0.0), ("D", 100.0)):
        for i, (a, b) in enumerate([(1, 1), (2, 3), (3, 2)]):
            for m, v in ((0, a), (1, b)):
                rows.append({"player_id": f"{role}{i}", "match_id": f"M{m}", "role": role,
                             "toi_s": 3600.0, "value": v + shift})
    rel = cs.split_half_reliability(pd.DataFrame(rows), min_toi_s=60.0)
    # центрирование по роли убирает сдвиг 100: r = corr([−1,0,1]×2, [−1,1,0]×2) = 0,5
    assert rel["r"] == pytest.approx(0.5)
    assert rel["r_sb"] == pytest.approx(2 * 0.5 / 1.5)


def test_split_half_reliability_drops_players_below_min_toi_in_a_half():
    rows = [{"player_id": "p", "match_id": "M0", "role": "C", "toi_s": 10.0, "value": 1.0},
            {"player_id": "p", "match_id": "M1", "role": "C", "toi_s": 600.0, "value": 1.0}]
    assert cs.split_half_reliability(pd.DataFrame(rows), min_toi_s=60.0)["n_players"] == 0


def test_display_label_low_minutes_overrides_model_label():
    lab = cs.display_label(np.array([0.9, 0.9, 0.5, 0.1]), np.array([30.0, 120.0, 120.0, 500.0]),
                           high=0.7, medium=0.4, min_toi_min=60.0)
    assert lab["label"].tolist() == ["LOW", "HIGH", "MEDIUM", "LOW"]
    assert lab["flag"].tolist() == ["мало минут", "", "", ""]


def test_consistency_weighted_sum_hand_case():
    o1 = pd.DataFrame({"player_id": ["a", "b"], "rating": [0.5, -0.2]})
    sit = {"5v5": pd.DataFrame({"player_id": ["a", "b"], "rating": [0.4, -0.2], "toi_min": [80.0, 50.0]}),
           "pp": pd.DataFrame({"player_id": ["a", "b"], "rating": [1.0, 0.0], "toi_min": [20.0, 0.0]})}
    total = pd.Series({"a": 100.0, "b": 50.0})
    res = cs.consistency(o1, sit, total)
    # a: 0,8·0,4 + 0,2·1,0 = 0,52; b: 1,0·(−0,2) = −0,2 → расхождения 0,02 и 0
    assert res["weighted"].loc["a"] == pytest.approx(0.52)
    assert res["mae"] == pytest.approx(0.01)


def test_merge_flags_keeps_model_flag_and_minutes_flag():
    # Флаг модели L5 («сигнал не обнаружен») не должен теряться при правиле «мало минут».
    model = np.array(["сигнал не обнаружен", "", None, "сигнал не обнаружен"], dtype=object)
    minutes = np.array(["", cs.LOW_MINUTES, "", cs.LOW_MINUTES], dtype=object)
    out = cs.merge_flags(model, minutes)
    assert list(out) == ["сигнал не обнаружен", cs.LOW_MINUTES, "",
                         f"сигнал не обнаружен; {cs.LOW_MINUTES}"]


def test_merge_flags_rejects_length_mismatch():
    with pytest.raises(ValueError):
        cs.merge_flags(np.array([""], dtype=object), np.array(["", ""], dtype=object))


def test_posterior_sd_clips_tiny_negative_variance():
    sd = cs.posterior_sd(np.array([0.04, -1e-12, 0.0]))
    assert sd.tolist() == pytest.approx([0.2, 0.0, 0.0])
    assert np.isfinite(sd).all()
