"""Карточка: L5 берётся только если совпадает со сдаваемым файлом (test_ratings_detail.csv)."""
import pandas as pd
import pytest

from datamatch import card_io


def _o1():
    return pd.DataFrame({"player_id": ["a", "b"], "rating": [0.1, -0.2]})


def test_delivered_check_passes_on_identical_ratings():
    detail = pd.DataFrame({"player_id": ["b", "a", "c"], "rating": [-0.2, 0.1, 0.0]})
    card_io.assert_matches_delivered(_o1(), detail)          # c — игрок только из заявки, в O1 нет


def test_delivered_check_fails_on_rating_mismatch():
    detail = pd.DataFrame({"player_id": ["a", "b"], "rating": [0.1, -0.25]})
    with pytest.raises(card_io.CardInputMismatch, match="b"):
        card_io.assert_matches_delivered(_o1(), detail)


def test_delivered_check_fails_when_player_missing_from_file():
    detail = pd.DataFrame({"player_id": ["a"], "rating": [0.1]})
    with pytest.raises(card_io.CardInputMismatch, match="нет в"):
        card_io.assert_matches_delivered(_o1(), detail)


def _pm(v):
    cols = {c: [v] for c in card_io.PM_HASH_COLS}
    return pd.DataFrame(cols)


def test_player_match_bound_to_l5_provenance_by_frame_hash():
    from datamatch.cache import frame_hash
    tr, te = _pm(1.0), _pm(2.0)
    prov = {"train_pm": frame_hash(tr[list(card_io.PM_HASH_COLS)]),
            "test_pm": frame_hash(te[list(card_io.PM_HASH_COLS)])}
    card_io.assert_pm_matches_l5(tr, te, prov)
    with pytest.raises(card_io.CardInputMismatch, match="player_match"):
        card_io.assert_pm_matches_l5(te, tr, prov)
