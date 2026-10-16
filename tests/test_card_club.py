"""Карточка для клуба: понятная шкала и текст по шаблону без LLM (METHODOLOGY 11.7)."""
import pytest

from datamatch import card_club as cc


def test_percentile_hand_values():
    assert cc.percentile(0.0, 0.2) == pytest.approx(50.0)
    assert cc.percentile(0.2, 0.2) == pytest.approx(84.13, abs=0.01)     # Φ(1)
    assert cc.percentile(-0.392, 0.2) == pytest.approx(2.5, abs=0.01)    # Φ(−1,96)


@pytest.mark.parametrize("lo, hi, expected", [(0.1, 0.5, "выше среднего амплуа"),
                                              (-0.5, -0.01, "ниже среднего амплуа"),
                                              (-0.1, 0.3, "отличие от среднего амплуа не подтверждено")])
def test_verdict(lo, hi, expected):
    assert cc.verdict(lo, hi) == expected


@pytest.mark.parametrize("label, flag, expected", [("HIGH", "", "оценка надёжная"),
                                                   ("MEDIUM", "", "оценка предварительная"),
                                                   ("LOW", "", "данных мало"),
                                                   ("LOW", "мало минут", "данных мало (меньше 60 минут на льду)")])
def test_reliability_phrase_for_text(label, flag, expected):
    assert cc.reliability_phrase(label, flag) == expected


@pytest.mark.parametrize("label, expected", [("HIGH", True), ("MEDIUM", True), ("LOW", False)])
def test_profile_shown_only_with_enough_data(label, expected):
    assert cc.show_profile(label) is expected


@pytest.mark.parametrize("label, flag, expected", [("HIGH", "", "надёжная"), ("MEDIUM", "", "предварительная"),
                                                   ("LOW", "", "мало данных"),
                                                   ("LOW", "мало минут", "мало данных (меньше 60 минут на льду)")])
def test_reliability_words(label, flag, expected):
    assert cc.reliability_words(label, flag) == expected


def test_strengths_and_weaknesses_from_profile():
    z = {"shot": 1.2, "loss": -0.9, "recovery": 0.1, "creation": 0.6, "faceoff": None}
    s, w = cc.strengths(z)
    assert s == ["бросок", "созидание"] and w == ["много потерь"]


def test_strengths_none_when_near_average():
    assert cc.strengths({"shot": 0.2, "loss": -0.3}) == ([], [])


def test_war_per_season_hand_case():
    # 0,03 победы на 60 мин × 15 мин/матч × 82 матча = 0,03 × 20,5 = 0,615
    assert cc.war_per_season(0.03, 15.0) == pytest.approx(0.615)


def test_player_text_confirmed_difference():
    t = cc.player_text(role="C", pct=80.4, pct_lo=61.2, pct_hi=91.7, verdict="выше среднего амплуа", label="HIGH",
                       flag="", strengths=["созидание"], weaknesses=["много потерь"])
    assert t == ("Центральный нападающий: лучше 80% игроков амплуа (диапазон 61–92%), выше среднего амплуа; "
                 "оценка надёжная. Сильные стороны: созидание. Слабые стороны: много потерь.")


def test_player_text_unconfirmed_difference_shows_range_not_point():
    t = cc.player_text(role="D", pct=24.1, pct_lo=0.6, pct_hi=86.5,
                       verdict="отличие от среднего амплуа не подтверждено", label="MEDIUM", flag="",
                       strengths=[], weaknesses=[])
    assert t.startswith("Защитник: место среди игроков амплуа — от 1% до 87%, отличие от среднего амплуа не "
                        "подтверждено; оценка предварительная.")
    assert "лучше 24%" not in t
    assert "Выраженных сильных и слабых сторон профиль не показывает." in t


def test_player_text_low_data_names_no_profile():
    t = cc.player_text(role="W", pct=40.0, pct_lo=3.0, pct_hi=95.0,
                       verdict="отличие от среднего амплуа не подтверждено", label="LOW", flag="мало минут",
                       strengths=["бросок"], weaknesses=["много потерь"])
    assert "данных мало (меньше 60 минут на льду)" in t
    assert "бросок" not in t and "потерь" not in t
    assert "Сильные и слабые стороны по такому объёму данных не называем." in t


def test_player_text_has_no_wins_claim():
    t = cc.player_text(role="C", pct=99.0, pct_lo=97.0, pct_hi=99.9, verdict="выше среднего амплуа", label="HIGH",
                       flag="", strengths=["бросок"], weaknesses=[])
    assert "побед" not in t
