"""Карточка: сводная таблица «Авангарда» из готовых блоков (сборка без пересчёта)."""
import pandas as pd
import pytest

from datamatch import card_report as cr


def _blocks():
    ids = ["P1", "P2", "P3"]
    war = pd.DataFrame({"player_id": ids, "is_avangard": [True, False, True], "role": ["C", "W", "D"],
                        "rating": [0.1, 0.5, 0.3], "reliability": ["HIGH", "LOW", "MEDIUM"], "toi_min": [900, 50, 700],
                        "war": [1.0, 0.1, 2.0], "war_lo95": [0.5, -0.1, 1.0], "war_hi95": [1.5, 0.3, 3.0]})
    roles = pd.DataFrame({"player_id": ids, "role_name": ["a", "b", "c"], "own_profile": ["x", "y", "z"],
                          "flag": ["", "профиль неустойчив", ""]})
    sit = pd.DataFrame({"player_id": ids, "rating_5v5": [0.1, 0.2, 0.3], "reliability_5v5": ["HIGH", "LOW", "LOW"],
                        "rating_pp": [0.0, 0.0, 0.1], "reliability_pp": ["LOW"] * 3,
                        "rating_pk": [0.0, 0.1, 0.0], "reliability_pk": ["LOW"] * 3})
    pen = pd.DataFrame({"player_id": ids, "drawn_per60": [1.0, 0.5, 0.2], "taken_per60": [0.3, 0.4, 0.9]})
    phys = pd.DataFrame({"player_id": ids, "speed_p80": [7.2, 7.1, 6.2], "speed_p80_level": ["уровень амплуа"] * 3})
    return war, roles, sit, pen, phys


def test_avangard_table_selects_merges_and_sorts():
    t = cr.avangard_table(*_blocks())
    assert t.player_id.tolist() == ["P3", "P1"]                 # только «Авангард», по рейтингу
    assert t.player_id.is_unique
    assert {"war", "role_name", "own_profile", "rating_5v5", "drawn_per60", "speed_p80_level"} <= set(t.columns)
    assert t.set_index("player_id").loc["P1", "war"] == pytest.approx(1.0)


def test_avangard_table_rejects_duplicate_players():
    war, roles, sit, pen, phys = _blocks()
    with pytest.raises(ValueError):
        cr.avangard_table(war, pd.concat([roles, roles.iloc[:1]]), sit, pen, phys)


def test_markdown_table_hand_case():
    df = pd.DataFrame({"a": [1, None], "b": ["x|y", "z"]})
    assert cr.markdown_table(df) == "| a | b |\n|---|---|\n| 1.0 | x/y |\n|  | z |"
