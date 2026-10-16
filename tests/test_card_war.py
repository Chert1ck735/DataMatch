"""Карточка, WAR: G_win по предельной ценности гола, восстановление счёта, интервал WAR.

Ожидания — ручной расчёт на синтетике (METHODOLOGY 11.1), а не вывод текущего кода.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from datamatch import card_war as cw


def _games():
    return pd.DataFrame({"match_id": ["M1", "M2", "M3", "M4"],
                         "home_team_id": ["T1", "T1", "T3", "T1"],
                         "away_team_id": ["T2", "T3", "T2", "T2"]})


def _goals():
    rows = [("M1", 1, "T1"), ("M1", 2, "T1"), ("M1", 3, "T2"),      # 2:1 в основное время
            ("M2", 1, "T1"), ("M2", 3, "T3"), ("M2", 4, "T3"),      # 1:1, победа гостей в ОТ
            ("M3", 0, "T3"), ("M3", 0, "T3"), ("M3", 0, "T2"),      # 0:0, буллиты 2:1 хозяевам
            ("M4", 1, "T2")]                                        # обрезанный матч
    return pd.DataFrame(rows, columns=["match_id", "period", "actor_team_id"])


def test_match_scores_split_regulation_overtime_shootout():
    s = cw.match_scores(_goals(), _games(), bad_matches={"M4"}).set_index("match_id")
    assert list(s.index) == ["M1", "M2", "M3"]
    assert (s.loc["M1", "home_reg"], s.loc["M1", "away_reg"]) == (2, 1)
    assert (s.loc["M2", "home_reg"], s.loc["M2", "away_reg"], s.loc["M2", "away_ot"]) == (1, 1, 1)
    assert (s.loc["M3", "home_so"], s.loc["M3", "away_so"]) == (2, 1)
    assert s.loc["M1", "home_win"] == 1 and s.loc["M2", "home_win"] == 0 and s.loc["M3", "home_win"] == 1


def test_match_scores_rejects_goal_of_team_not_in_match():
    goals = pd.concat([_goals(), pd.DataFrame({"match_id": ["M1"], "period": [1], "actor_team_id": ["T9"]})])
    with pytest.raises(ValueError, match="T9"):
        cw.match_scores(goals, _games(), bad_matches=set())


def test_team_game_rows_two_per_match_with_opposite_margins():
    s = cw.match_scores(_goals(), _games(), bad_matches={"M4"})
    tg = cw.team_game_rows(s)
    assert len(tg) == 6
    m1 = tg[tg.match_id == "M1"].set_index("team_id")
    assert m1.loc["T1", "d_reg"] == 1 and m1.loc["T2", "d_reg"] == -1
    assert m1.loc["T1", "win"] == 1 and m1.loc["T2", "win"] == 0
    m3 = tg[tg.match_id == "M3"].set_index("team_id")
    assert m3.loc["T3", "d_reg"] == 0 and m3.loc["T3", "win"] == 1


def test_marginal_win_per_goal_hand_calculation():
    # P(D=0) = 2/10, P(D=-1) = 2/10 → ΔW = 0,5·0,2 + 0,5·0,2 = 0,2 → G_win = 5
    d = np.array([0, 0, 1, -1, 2, -2, 1, -1, 3, -3])
    assert cw.marginal_win_per_goal(d) == pytest.approx(0.2)
    assert cw.g_win_marginal(d) == pytest.approx(5.0)


def test_marginal_win_per_goal_rejects_empty_and_degenerate():
    with pytest.raises(ValueError):
        cw.marginal_win_per_goal(np.array([], dtype=int))
    with pytest.raises(ValueError):
        cw.g_win_marginal(np.array([5, -5]))   # ни ничьих, ни проигрышей в один гол: ΔW = 0


def test_skellam_equals_numeric_derivative_of_win_probability():
    # W(λf) = P(X > Y) + ½ P(X = Y); ΔW = dW/dλf — проверка численной производной по прямой свёртке
    lf, la, h = 2.9, 2.7, 1e-4
    k = np.arange(0, 60)

    def win(l1):
        px, py = poisson.pmf(k, l1), poisson.pmf(k, la)
        joint = np.outer(px, py)
        return np.tril(joint, -1).sum() + 0.5 * np.trace(joint)

    deriv = (win(lf + h) - win(lf - h)) / (2 * h)
    assert cw.skellam_marginal(lf, la) == pytest.approx(deriv, rel=1e-5)
    assert cw.g_win_skellam(lf, la) == pytest.approx(1.0 / deriv, rel=1e-5)


def test_pythagorean_and_ols_hand_values():
    assert cw.g_win_pythagorean(3.0, 2.0) == pytest.approx(6.0)            # 4·3/2
    # d = [1,-1,2,-2], win = [1,0,1,0]: cov = 0,75, var = 2,5 → наклон 0,3 → G = 10/3
    assert cw.g_win_ols(np.array([1, -1, 2, -2]), np.array([1, 0, 1, 0])) == pytest.approx(10 / 3)


def test_cluster_bootstrap_identical_clusters_have_no_spread_and_seed_is_reproducible():
    rows = pd.DataFrame({"team_id": ["A", "A", "B", "B"], "d_reg": [0, -1, 0, -1]})
    draws = cw.cluster_bootstrap(rows, "team_id", lambda r: cw.g_win_marginal(r.d_reg.to_numpy()), 50, seed=1)
    assert np.allclose(draws, 2.0)                       # P(0) = P(−1) = 0,5 → ΔW = 0,5 в каждой реплике
    rows2 = pd.DataFrame({"team_id": list("ABCDEF"), "d_reg": [0, -1, 1, 2, -2, 0]})
    f = lambda r: cw.marginal_win_per_goal(r.d_reg.to_numpy())             # noqa: E731
    assert np.array_equal(cw.cluster_bootstrap(rows2, "team_id", f, 30, seed=7),
                          cw.cluster_bootstrap(rows2, "team_id", f, 30, seed=7))


def test_war_point_and_degenerate_interval():
    df = pd.DataFrame({"player_id": ["a", "b"], "role": ["C", "D"], "xgar": [1.0, -0.5],
                       "toi_min": [600.0, 300.0], "v_tot": [0.0, 0.0]})
    R_draws = {"C": np.array([0.6, 0.6]), "D": np.array([0.4, 0.4])}
    out = cw.war_table(df, g_hat=5.0, g_draws=np.array([5.0, 5.0]), R_hat={"C": 0.6, "D": 0.4},
                       R_draws=R_draws, n_draws=200, seed=3).set_index("player_id")
    assert out.loc["a", "war"] == pytest.approx(0.2)
    assert out.loc["b", "war"] == pytest.approx(-0.1)
    assert out.loc["a", "war_lo95"] == pytest.approx(0.2) and out.loc["a", "war_hi95"] == pytest.approx(0.2)


def test_war_interval_reflects_replacement_level_shift_hand_case():
    # TOI = 60 мин, θ точно известен, R_draw = R ± 0,1 поровну, G = 2:
    # xGAR_draw = xGAR ∓ 0,1 → WAR_draw ∈ {0,45; 0,55} при WAR = 0,5
    df = pd.DataFrame({"player_id": ["a"], "role": ["W"], "xgar": [1.0], "toi_min": [60.0], "v_tot": [0.0]})
    out = cw.war_table(df, g_hat=2.0, g_draws=np.array([2.0]), R_hat={"W": 0.5},
                       R_draws={"W": np.array([0.4, 0.6])}, n_draws=4000, seed=5).iloc[0]
    assert out.war == pytest.approx(0.5)
    assert out.war_lo95 == pytest.approx(0.45) and out.war_hi95 == pytest.approx(0.55)


def test_war_interval_widens_with_rating_uncertainty_and_contains_point():
    df = pd.DataFrame({"player_id": ["a", "b"], "role": ["C", "C"], "xgar": [2.0, 2.0],
                       "toi_min": [600.0, 600.0], "v_tot": [0.001, 0.04]})
    out = cw.war_table(df, g_hat=5.0, g_draws=np.full(10, 5.0), R_hat={"C": 0.6},
                       R_draws={"C": np.full(10, 0.6)}, n_draws=20000, seed=11).set_index("player_id")
    width = out.war_hi95 - out.war_lo95
    assert width["b"] > width["a"]
    # sd WAR = √v_tot·TOI/60/G: у b — 0,2·10/5 = 0,4 → ширина ≈ 2·1,96·0,4 = 1,568
    assert width["b"] == pytest.approx(1.568, rel=0.05)
    assert ((out.war_lo95 <= out.war) & (out.war <= out.war_hi95)).all()


def test_war_table_rejects_missing_role_level():
    df = pd.DataFrame({"player_id": ["a"], "role": ["X"], "xgar": [1.0], "toi_min": [60.0], "v_tot": [0.0]})
    with pytest.raises(KeyError):
        cw.war_table(df, g_hat=5.0, g_draws=np.array([5.0]), R_hat={"C": 0.6},
                     R_draws={"C": np.array([0.6])}, n_draws=10, seed=0)


def test_replacement_bootstrap_constant_theta_gives_constant_level():
    o1 = pd.DataFrame({"player_id": [f"p{i}" for i in range(40)],
                       "role": ["C"] * 20 + ["D"] * 20, "theta": [0.5] * 20 + [0.3] * 20,
                       "toi_min": np.linspace(10, 400, 40), "games": [3] * 40})
    draws = cw.replacement_bootstrap(o1, B=20, seed=2)
    assert np.allclose(draws["C"], 0.5) and np.allclose(draws["D"], 0.3)
    assert len(draws["W"]) == 0


def _scores_frame():
    # M1: 2:1 в основное время; M2: 2:2 + буллиты 1:0 → 3:2 по протоколу; M3: 1:1, ОТ 0:1 → 1:2.
    return pd.DataFrame({"match_id": ["M1", "M2", "M3"],
                         "home_reg": [2, 2, 1], "away_reg": [1, 2, 1],
                         "home_ot": [0, 0, 0], "away_ot": [0, 0, 1],
                         "home_so": [0, 1, 0], "away_so": [0, 0, 0]})


def test_known_score_check_hand_case_passes():
    known = pd.DataFrame({"match_id": ["M1", "M2", "M3", "M9"],
                          "home_goals": [2, 3, 1, 5], "away_goals": [1, 2, 2, 0]})
    info = cw.known_score_check(_scores_frame(), known)
    assert info["known_scores_checked"] == 3          # M9 нет среди восстановленных
    assert info["known_scores_match"] == 3
    assert info["known_scores_mismatch"] == []


def test_known_score_check_raises_on_mismatch():
    known = pd.DataFrame({"match_id": ["M2"], "home_goals": [2], "away_goals": [2]})  # без победы по буллитам
    with pytest.raises(ValueError, match="M2"):
        cw.known_score_check(_scores_frame(), known)


def test_known_score_check_raises_when_nothing_to_check():
    known = pd.DataFrame({"match_id": ["M9"], "home_goals": [1], "away_goals": [0]})
    with pytest.raises(ValueError):
        cw.known_score_check(_scores_frame(), known)


def test_attach_xgar_zero_only_for_players_outside_o1():
    detail = pd.DataFrame({"player_id": ["A", "B"], "rating": [0.1, 0.0]})
    o1 = pd.DataFrame({"player_id": ["A"], "xgar": [1.5], "theta": [0.2]})
    out = cw.attach_xgar(detail, o1)
    assert out.set_index("player_id").xgar.to_dict() == {"A": 1.5, "B": 0.0}


def test_attach_xgar_rejects_nan_for_o1_player():
    detail = pd.DataFrame({"player_id": ["A"], "rating": [0.1]})
    o1 = pd.DataFrame({"player_id": ["A"], "xgar": [np.nan], "theta": [0.2]})
    with pytest.raises(ValueError, match="A"):
        cw.attach_xgar(detail, o1)
