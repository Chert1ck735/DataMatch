"""L4 RAPM: синтетика с известными вкладами и сильной корреляцией партнёров.

Контракт (из модели, не из кода): xG атакующей пятёрки на отрезке —
    xG_s = h_s · (μ + Σ_атака β^O_i + Σ_защита β^D_j + γ·дом) + шум,
β^D > 0 — защитник пропускает больше. Итог O2 = β^O − β^D (больше — лучше).
* Сырой on-ice xGD/60 смешивает игрока с его звеном; RAPM разделяет их: корреляция O2 с истинным
  вкладом выше, чем у on-ice xGD/60.
* Знак: игроку с большим β^O и меньшим β^D — больший O2.
* Ridge к точному prior даёт ошибку меньше, чем ridge к нулю; λ выбирается кросс-валидацией по
  целым матчам из сетки.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datamatch import l4_rapm as rp

N_TEAMS = 8
FWD_LINES, D_PAIRS = 4, 3
SD_BETA = 0.25          # разброс вкладов, xG/60
BASE = 2.5              # xG/60 команды при 5 на 5
HOME = 0.15
# Шум xG на √час. При 1,3 ошибка индивидуальной оценки (~0,28 при ~21 ч на игрока) больше разброса
# самих вкладов (0,25) — разделять партнёров нечего, контракт не проверяется. При 0,3 различие
# определяется корреляцией партнёров, а не шумом: RAPM 0,74 против 0,52 у сырого on-ice (07.10.2026).
NOISE = 0.3
MIX = 0.1               # доля смен с чужим звеном (партнёры сильно коррелированы)


def league(n_matches: int = 400, seed: int = 0):
    """Длинная таблица отрезков (одна строка на атакующую сторону) и истинные вклады."""
    rng = np.random.default_rng(seed)
    teams = {}
    for t in range(N_TEAMS):
        fw = [[f"T{t}F{line}{k}" for k in range(3)] for line in range(FWD_LINES)]
        dp = [[f"T{t}D{p}{k}" for k in range(2)] for p in range(D_PAIRS)]
        teams[t] = (fw, dp)
    players = [p for fw, dp in teams.values() for grp in fw + dp for p in grp]
    true_o = pd.Series(rng.normal(0, SD_BETA, len(players)), index=players)
    true_d = pd.Series(rng.normal(0, SD_BETA, len(players)), index=players)

    def five(t):
        fw, dp = teams[t]
        line = rng.integers(FWD_LINES)
        pair = line % D_PAIRS if rng.random() > MIX else rng.integers(D_PAIRS)
        f = list(fw[line])
        if rng.random() < MIX:
            f[rng.integers(3)] = fw[(line + 1) % FWD_LINES][rng.integers(3)]
        return f + list(dp[pair])

    rows = []
    for m in range(n_matches):
        h, a = rng.choice(N_TEAMS, 2, replace=False)
        for s in range(60):
            dur = rng.uniform(10, 90)
            hs, as_ = five(h), five(a)
            hh = dur / 3600
            for att, dfn, at, dt, home in ((hs, as_, h, a, 1), (as_, hs, a, h, 0)):
                rate = BASE + true_o[att].sum() + true_d[dfn].sum() + HOME * home
                xg = hh * rate + np.sqrt(hh) * NOISE * rng.normal()
                rows.append((f"m{m:04d}", s, dur, xg, tuple(att), tuple(dfn), f"t{at}", f"t{dt}", home, 0, "fly"))
    df = pd.DataFrame(rows, columns=rp.ROW_COLS)
    return df, true_o, true_d


@pytest.fixture(scope="module")
def data():
    return league()


def naive_onice(rows: pd.DataFrame) -> pd.Series:
    """Сырой on-ice xGD/60 игрока: xG своей пятёрки минус xG соперника на его отрезках."""
    recs = []
    for r in rows.itertuples():
        for p in r.att:
            recs.append((p, r.xg, r.dur_s))
        for p in r.dfn:
            recs.append((p, -r.xg, 0.0))
    d = pd.DataFrame(recs, columns=["p", "xg", "dur"]).groupby("p").sum()
    return d.xg / d.dur * 3600


def test_rapm_separates_partners(data):
    rows, true_o, true_d = data
    fit = rp.fit_rapm(rows, lam=1.0)
    truth = (true_o - true_d)
    o2 = fit.o2.reindex(truth.index)
    naive = naive_onice(rows).reindex(truth.index)
    r_rapm = np.corrcoef(o2, truth)[0, 1]
    r_naive = np.corrcoef(naive, truth)[0, 1]
    assert r_rapm > r_naive + 0.1
    assert r_rapm > 0.6


def test_sign_convention(data):
    rows, true_o, true_d = data
    fit = rp.fit_rapm(rows, lam=4.0)
    assert np.corrcoef(fit.beta_o.reindex(true_o.index), true_o)[0, 1] > 0.5
    assert np.corrcoef(fit.beta_d.reindex(true_d.index), true_d)[0, 1] > 0.5


def test_exact_prior_beats_zero_prior(data):
    rows, true_o, true_d = data
    zero = rp.fit_rapm(rows, lam=16.0)
    exact = rp.fit_rapm(rows, lam=16.0, prior_o=true_o, prior_d=true_d)
    truth = true_o - true_d
    err0 = float(((zero.o2.reindex(truth.index) - truth) ** 2).mean())
    err1 = float(((exact.o2.reindex(truth.index) - truth) ** 2).mean())
    assert err1 < err0


def test_home_control_recovered(data):
    rows, _, _ = data
    fit = rp.fit_rapm(rows, lam=4.0)
    assert fit.controls["home"] == pytest.approx(HOME, abs=0.1)


def test_lambda_cv_by_whole_matches(data):
    rows, _, _ = data
    best, table = rp.cv_lambda(rows, grid=(0.5, 4.0, 64.0), k=4, seed=1)
    assert best in (0.5, 4.0, 64.0)
    assert set(table) == {0.5, 4.0, 64.0}
    assert all(np.isfinite(v) for v in table.values())


def test_fold_assignment_keeps_matches_whole(data):
    rows, _, _ = data
    folds = rp.match_folds(rows.match_id, k=4, seed=1)
    assert rows.assign(f=folds).groupby("match_id").f.nunique().max() == 1


# --- сборка строк из таблиц L0 / L2 ------------------------------------------------------------

def toy_l0():
    """Два отрезка 5 на 5 одного матча: бросок хозяев (xG 0,1) и гостей (0,3); вбрасывание в зоне гостей."""
    from datamatch import l4_rapm_data as rd  # noqa: F401  (модуль под тестом)
    home = tuple(f"h{i}" for i in range(5))
    away = tuple(f"a{i}" for i in range(5))
    stints = pd.DataFrame({
        "match_id": ["m1", "m1", "m1"], "stint": [1, 2, 3], "duration": [40.0, 20.0, 30.0],
        "home_sk": [home, home, home[:4]], "away_sk": [away, away, away], "n_home_sk": [5, 5, 4],
        "n_away_sk": [5, 5, 5], "home_g_in": [True, True, True], "away_g_in": [True, True, True],
        "is_5v5": [True, True, False], "valid_for_rate_metrics": [True, True, True],
        "home_team_id": ["H", "H", "H"], "away_team_id": ["A", "A", "A"]})
    nodes = pd.DataFrame({
        "match_id": ["m1"] * 4, "stint": [1, 1, 2, 3], "seq": [1, 2, 3, 4],
        "action": ["faceoff", "shot", "shot", "shot"], "team_id": ["H", "H", "A", "H"],
        "zone": ["off", "off", "off", "off"], "score_home_before": [1, 1, 1, 2],
        "score_away_before": [0, 0, 0, 0]})
    r = np.array([0.0, 0.1, 0.3, 0.5])
    return nodes, stints, r


def test_stint_rows_from_l0():
    from datamatch import l4_rapm_data as rd
    nodes, stints, r = toy_l0()
    rows = rd.stint_rows(nodes, stints, r)
    assert len(rows) == 4                                    # 2 отрезка 5 на 5 × 2 стороны; 5 на 4 исключён
    s1h = rows[(rows.stint == 1) & (rows.home == 1)].iloc[0]
    s1a = rows[(rows.stint == 1) & (rows.home == 0)].iloc[0]
    assert s1h.xg == pytest.approx(0.1) and s1a.xg == pytest.approx(0.0)
    assert s1h.zone == "off" and s1a.zone == "def"           # вбрасывание хозяев в чужой зоне
    assert s1h.score == 1 and s1a.score == -1
    assert s1h.att == tuple(f"h{i}" for i in range(5)) and s1h.dfn == tuple(f"a{i}" for i in range(5))
    s2a = rows[(rows.stint == 2) & (rows.home == 0)].iloc[0]
    assert s2a.xg == pytest.approx(0.3) and s2a.zone == "fly" and s2a.dur_s == 20.0


# --- правило приёма (METHODOLOGY 4.3а) ---------------------------------------------------------

def test_pooled_delta_ss_hand_computed():
    from datamatch import l4_rapm_eval as ev
    p = pd.DataFrame({"player_id": ["a", "b", "a", "b"], "role": ["C", "C", "C", "C"],
                      "target": [1.0, -1.0, 2.0, 0.0], "toi_s": [1.0, 1.0, 1.0, 1.0],
                      "p::role_only": [0.5, 0.5, 0.5, 0.5], "p::O1": [0.5, 0.0, 0.5, 0.0],
                      "p::O2": [1.0, -1.0, 2.0, 0.0]})
    out = ev.pooled_delta_ss(p, "O2", "O1", n_boot=50)
    mse_base = (0.25 + 2.25 + 2.25 + 0.25) / 4
    mse_o1 = (0.25 + 1.0 + 2.25 + 0.0) / 4
    assert out["ss"]["O2"] == pytest.approx(1.0)
    assert out["ss"]["O1"] == pytest.approx(1 - mse_o1 / mse_base)
    assert out["delta_ss"] == pytest.approx(1.0 - (1 - mse_o1 / mse_base))


def test_verdict_requires_all_conditions():
    from datamatch import l4_rapm_eval as ev
    ok = {"reliability": {"vs": {"onice_xgd_raw": {"diff_ci": [0.01, 0.2]}, "prior_only": {"diff_ci": [0.02, 0.3]}}},
          "forecast": {"delta_ss_ci": [0.001, 0.02]},
          "slope": {"A->B": {"b": 0.3, "b_ci": [0.1, 0.5]}, "B->A": {"b": 0.4, "b_ci": [0.2, 0.6]}},
          "robustness": {"spearman": 0.93}}
    assert ev.verdict(ok)["accepted"]
    bad = {**ok, "forecast": {"delta_ss_ci": [-0.001, 0.02]}}
    v = ev.verdict(bad)
    assert not v["accepted"] and not v["conditions"]["2_forecast"]
    bad_slope = {**ok, "slope": {"A->B": {"b": 0.1, "b_ci": [0.05, 0.2]}, "B->A": {"b": 0.3, "b_ci": [0.1, 0.5]}}}
    assert not ev.verdict(bad_slope)["conditions"]["3_slope"]          # наклоны отличаются втрое


def test_prior_min_games_scales_with_part():
    from datamatch import l4_rapm_eval as ev
    assert ev.prior_min_games(887, 887) == 10
    assert ev.prior_min_games(443, 887) == 5
    assert ev.prior_min_games(10, 887) == ev.MIN_GAMES_PRIOR_FLOOR


def test_defender_lowering_opponent_xg_gets_higher_o2(data):
    """Знак β^D на синтетике: игроку, при котором соперник создаёт на 0,8 xG/60 меньше, RAPM даёт
    меньший β^D и больший O2. Часть эффекта уходит в ridge и постоянному партнёру по паре, поэтому
    проверяются направление, величина не меньше четверти внесённого эффекта и то, что эффект
    достаётся ему, а не партнёру; β^O при этом почти не меняется."""
    rows, _, _ = data
    # нападающий: звенья иногда перемешиваются (MIX), поэтому он отделим от партнёра; пара защитников
    # всегда на льду вместе — её вклад RAPM делит поровну (неразделимая пара, архитектура L4)
    star, partner = "T0F00", "T0F01"
    injected = 0.8
    rows2 = rows.copy()
    hit = rows2.dfn.map(lambda t: star in t).to_numpy()
    h = rows2.dur_s.to_numpy() / 3600.0
    rows2.loc[hit, "xg"] = rows2.xg.to_numpy()[hit] - injected * h[hit]
    fit = rp.fit_rapm(rows2, lam=1.0)
    base = rp.fit_rapm(rows, lam=1.0)
    d_star = fit.beta_d[star] - base.beta_d[star]
    d_partner = fit.beta_d[partner] - base.beta_d[partner]
    assert d_star < -injected / 4
    assert fit.o2[star] - base.o2[star] > injected / 4
    assert d_star < d_partner
    assert abs(fit.beta_o[star] - base.beta_o[star]) < injected / 8


# --- замечания ревью 07.10.2026 --------------------------------------------------------------

def test_score_for_stint_without_nodes_uses_next_known_score():
    """Отрезок без узлов не содержит событий: его счёт — счёт перед ближайшим следующим узлом матча.
    Гол в недопущенном отрезке (5 на 4) между допущенными не должен теряться (раньше — перенос счёта
    с предыдущего допущенного отрезка)."""
    from datamatch import l4_rapm_data as rd
    home = tuple(f"h{i}" for i in range(5))
    away = tuple(f"a{i}" for i in range(5))
    stints = pd.DataFrame({
        "match_id": ["m1"] * 4, "stint": [1, 2, 3, 4], "duration": [30.0, 30.0, 30.0, 30.0],
        "home_sk": [home, home[:4], home, home], "away_sk": [away] * 4, "n_home_sk": [5, 4, 5, 5],
        "n_away_sk": [5, 5, 5, 5], "home_g_in": [True] * 4, "away_g_in": [True] * 4,
        "is_5v5": [True, False, True, True], "valid_for_rate_metrics": [True] * 4,
        "home_team_id": ["H"] * 4, "away_team_id": ["A"] * 4})
    nodes = pd.DataFrame({
        "match_id": ["m1"] * 4, "stint": [1, 2, 2, 4], "seq": [1, 2, 3, 4],
        "action": ["pass", "shot", "pass", "pass"], "team_id": ["H"] * 4, "zone": ["off"] * 4,
        "score_home_before": [0, 0, 1, 1], "score_away_before": [0, 0, 0, 0]})
    rows = rd.stint_rows(nodes, stints, np.zeros(4))
    s3 = rows[(rows.stint == 3) & (rows.home == 1)].iloc[0]
    assert s3.score == 1                       # гол хозяев в отрезке 2 учтён
    s1 = rows[(rows.stint == 1) & (rows.home == 1)].iloc[0]
    assert s1.score == 0


def test_spm_prior_ignores_eligible_without_target_and_is_crossfitted():
    """Игрок из eligible без β (нет допущенных отрезков) не роняет подгонку; prior игрока из eligible
    не зависит от его собственной цели (кросс-подгонка по игрокам)."""
    rng = np.random.default_rng(0)
    ids = [f"p{i}" for i in range(60)]
    feats = pd.DataFrame({"role": rng.choice(["C", "W", "D"], 60), "x": rng.normal(size=60)}, index=ids)
    beta = pd.Series(0.2 * feats.x.to_numpy() + rng.normal(0, 0.05, 60), index=ids)
    w = pd.Series(1.0, index=ids)
    elig = pd.Index(ids + ["ghost"])                                   # без цели и без признаков
    feats_g = pd.concat([feats, pd.DataFrame({"role": ["C"], "x": [0.0]}, index=["ghost"])])
    out = rp.spm_prior({"o": beta}, feats_g, w, elig, k=5, seed=1)["o"]
    assert np.isfinite(out.to_numpy()).all()
    beta2 = beta.copy()
    beta2["p0"] += 10.0
    out2 = rp.spm_prior({"o": beta2}, feats_g, w, elig, k=5, seed=1)["o"]
    assert out2["p0"] == pytest.approx(out["p0"])


def test_verdict_fails_closed_on_nan():
    from datamatch import l4_rapm_eval as ev
    nan = [float("nan"), float("nan")]
    res = {"reliability": {"vs": {"onice_xgd_raw": {"diff_ci": nan}, "prior_only": {"diff_ci": [0.1, 0.2]}}},
           "forecast": {"delta_ss_ci": nan},
           "slope": {"A->B": {"b": 0.3, "b_ci": nan}, "B->A": {"b": 0.3, "b_ci": [0.1, 0.5]}},
           "robustness": {"spearman": float("nan")}}
    v = ev.verdict(res)
    assert not any(v["conditions"].values()) and not v["accepted"]


def test_stint_rows_carry_sum_of_squared_shot_xg():
    """xg2 — сумма квадратов xG бросков стороны (для пуассоновской σ² этапа 2)."""
    from datamatch import l4_rapm_data as rd
    nodes, stints, r = toy_l0()
    rows = rd.stint_rows(nodes, stints, r)
    s1h = rows[(rows.stint == 1) & (rows.home == 1)].iloc[0]
    s2a = rows[(rows.stint == 2) & (rows.home == 0)].iloc[0]
    assert s1h.xg2 == pytest.approx(0.1 ** 2) and s2a.xg2 == pytest.approx(0.3 ** 2)
    assert rows[(rows.stint == 1) & (rows.home == 0)].iloc[0].xg2 == 0.0


def test_cv_lambda_picks_grid_max_with_exact_prior(data):
    """A3(д), аудит 08.10.2026: при точном prior (истинные β) чем сильнее ridge к нему, тем лучше прогноз
    отложенных матчей — CV обязана выбрать максимум сетки."""
    rows, true_o, true_d = data
    grid = (0.5, 4.0, 64.0)
    best, table = rp.cv_lambda(rows, grid=grid, k=4, seed=2, prior_o=true_o, prior_d=true_d)
    assert best == max(grid)
    assert table[64.0] < table[0.5]
