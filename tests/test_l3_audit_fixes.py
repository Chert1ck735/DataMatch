"""L3: находки аудита 06.10.2026 — гейт телескопирования, переносы, выборка ставок, on-ice."""
import numpy as np
import pandas as pd

from datamatch import l3_contrib
from tests.test_l3_contrib import chain


def test_telescoping_gate_fails_closed_when_no_segment_is_checked():
    jr = l3_contrib.journal(chain(), np.array([1., 2., 4., 8., 16., 32.]))
    pm = pd.DataFrame({"iav_sum": [0.0], **{f"c_{c}": [0.0] for c in l3_contrib.CATEGORIES}})
    tele = {"max_abs_diff": None, "n_segments": 0, "excluded_segments": 1}
    gates = l3_contrib.journal_gates(jr, pm, tele)
    assert gates["telescoping_split"] is False


def test_dropped_transfer_is_flagged_on_loser_row():
    # Шайбу вернул командный узел без допуска к атрибуции: доля −(1−α)·δ никому не начислена.
    nd = chain().assign(valid_for_action_attribution=[True, True, False, True, True, True])
    jr = l3_contrib.journal(nd, np.array([1., 2., 4., 8., 16., 32.]))
    assert bool(jr.transfer_dropped[1])
    assert not jr.transfer_dropped[[0, 3, 4, 5]].any()


def test_recoverer_with_unknown_transfer_is_not_rate_eligible():
    # Неизвестная доля не вменяется нулём: строка вернувшего не входит в ставки.
    jr = l3_contrib.journal(chain(), np.array([1., np.nan, 4., 8., 16., 32.]))
    assert bool(jr.recv_from_unknown[2])
    assert not jr.valid_for_rate_metrics[2]
    assert jr.valid_for_rate_metrics[3]


def test_onice_shots_follow_stint_lineup_attribution_not_shooter_attribution():
    stints = pd.DataFrame({"match_id": "M", "stint": [0, 1], "is_5v5": True,
                           "valid_for_action_attribution": [True, False], "rate_match_eligible": True,
                           "home_sk": [[f"H{i}" for i in range(5)]] * 2,
                           "away_sk": [[f"A{i}" for i in range(5)]] * 2})
    nodes = pd.DataFrame({"match_id": "M", "stint": [0, 1], "action": "shot", "team_id": "H",
                          "home_team_id": "H", "valid_for_rate_metrics": [False, True]})
    inc = l3_contrib.onice_incidence(nodes, stints)
    assert set(inc.node_idx) == {0}
    assert len(inc) == 10


def test_onice_shots_skip_matches_excluded_from_rates():
    stints = pd.DataFrame({"match_id": "M", "stint": [0], "is_5v5": True, "valid_for_action_attribution": True,
                           "rate_match_eligible": False, "home_sk": [[f"H{i}" for i in range(5)]],
                           "away_sk": [[f"A{i}" for i in range(5)]]})
    nodes = pd.DataFrame({"match_id": "M", "stint": [0], "action": "shot", "team_id": "H",
                          "home_team_id": "H", "valid_for_rate_metrics": [True]})
    assert l3_contrib.onice_incidence(nodes, stints).empty
