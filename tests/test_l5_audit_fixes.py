"""L5: регрессионные тесты по подтверждённым находкам аудита (стягивание O1/O0).

Каждый тест воспроизводит дефект, найденный аудитом, и выводит ожидание из контракта
(docs/METHODOLOGY.md, раздел 5: численная защита и правило надёжности w;
контракт L5 → L6), а не из текущего поведения кода.
"""
from __future__ import annotations

import copy
import json
import logging

import numpy as np
import pandas as pd
import pytest

from datamatch import stage_l5
from datamatch.cache import Cache
from datamatch.l5_shrink import (
    NO_SIGNAL_FLAG,
    ROLES,
    JointEB,
    UnivariateEB,
    overall_no_signal,
)
from tests.test_l5_shrink import SHRINK, synth_panel
from tests.test_l5_stage import cfg_for, synth_pm

COLS2 = ["c_0", "c_1"]
# Истинная T_r с нулевым сигналом суммы: 1ᵀT1 = 0 при ненулевых компонентах.
T_SUM_ZERO = np.array([[0.04, -0.04], [-0.04, 0.04]])
# Сиды синтетики, при которых выборочная 1ᵀT1 (до проекции) отрицательна (проверено вручную).
SEED_RAW_NEGATIVE = 1
SEED_UNI_TAU2_NEGATIVE = 0
FROZEN_COLUMNS = ["player_id", "overall", "var_post", "w", "method", "flag"]


def _player_sums(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return frame.groupby("player_id").agg(**{c: (c, "sum") for c in cols},
                                          toi_s=("toi_s", "sum"), games=("match_id", "nunique"))


def _overall(eb: JointEB, frame: pd.DataFrame, roles: pd.Series, cols=COLS2) -> dict:
    g = _player_sums(frame, cols)
    ids = list(g.index)
    return eb.overall(ids, np.asarray([roles[i] for i in ids], dtype=object), g[cols].to_numpy(),
                      g.toi_s.to_numpy(dtype=float), g.games.to_numpy(dtype=float))


def _run(cfg, l3, force=False):
    cache = Cache(cfg, {"synthetic": 1})
    return stage_l5.run_l5(cfg, cache, l3, force=force), cache


def _l3(pm_tr, pm_te):
    return {"train": {"journal": pd.DataFrame(), "player_match": pm_tr},
            "test": {"journal": pd.DataFrame(), "player_match": pm_te}, "report": {}}


@pytest.fixture(scope="module")
def l3_small():
    rng = np.random.default_rng(20261006)
    train = synth_pm(rng, "train", n_players=150)
    test = synth_pm(rng, "test", n_players=60, match_offset=600_000,
                    zero_rate_players=("P00000",))
    return _l3(train, test)


# ---------------------------------------------------------------------------------------
# Находка 1: «нет сигнала в сумме» решается по сырой 1ᵀT_r1 с относительным порогом

def test_overall_no_signal_threshold_is_relative_to_raw_trace():
    """1ᵀT1 = 2e-7 > ε_abs, но < 1e-3·tr(T): в сумме компонентов сигнала нет."""
    eps_abs = SHRINK["eps_abs"]
    tiny = np.array([[0.04, -0.0399999], [-0.0399999, 0.04]])
    assert float(tiny.sum()) > eps_abs
    assert overall_no_signal(tiny, eps_abs) is True
    assert overall_no_signal(np.eye(2) * 0.04, eps_abs) is False
    assert overall_no_signal(np.zeros((2, 2)), eps_abs) is True
    assert overall_no_signal(T_SUM_ZERO - 0.001, eps_abs) is True


def test_raw_negative_oneT1_routes_all_players_to_no_signal_not_joint():
    """Сырая 1ᵀT1 < 0, а после проекции на PD она > ε: раньше все игроки шли в joint с w до 0,87."""
    rng = np.random.default_rng(SEED_RAW_NEGATIVE)
    frame, roles, _ = synth_panel(rng, n_players=300, k=2, T=T_SUM_ZERO)
    eb = JointEB(COLS2, SHRINK).fit(frame, roles)
    assert eb.T_info["C"]["oneT1_raw"] < 0
    assert float(eb.T["C"].sum()) > SHRINK["eps_abs"], "сценарий: проекция подняла 1ᵀT1 над ε"
    res = _overall(eb, frame, roles)
    assert set(res["method"]) == {"univariate_fallback"}
    assert np.allclose(res["w"], 0.0)
    assert all(f == NO_SIGNAL_FLAG for f in res["flag"])
    assert (res["var"] > 0).all()
    dummy = pd.DataFrame({"method": res["method"], "fallback": True})
    gates = stage_l5.soft_gates(eb, dummy, {"shrink": SHRINK})
    assert gates["overall_signal_C"] is False


# ---------------------------------------------------------------------------------------
# Находка 2: строки «сигнал не обнаружен» не получают нулевую дисперсию

def test_no_signal_joint_rows_get_positive_univariate_prior_variance():
    rng = np.random.default_rng(SEED_RAW_NEGATIVE)
    frame, roles, _ = synth_panel(rng, n_players=300, k=2, T=T_SUM_ZERO)
    eb = JointEB(COLS2, SHRINK).fit(frame, roles)
    u = eb.uni["C"]
    assert u["tau2"] <= 0
    assert u["tau2_upper"] > max(u["tau2"], 0.0)
    res = _overall(eb, frame, roles)
    assert np.allclose(res["var"], u["tau2_upper"])
    ghost = eb.overall(["GHOST"], np.array(["C"], dtype=object), np.zeros((1, 2)), np.zeros(1),
                       np.zeros(1))
    assert ghost["var"][0] == pytest.approx(u["tau2_upper"])
    assert ghost["flag"][0] == NO_SIGNAL_FLAG
    assert eb.prior_var_overall("C") == pytest.approx(u["tau2_upper"])


def test_o0_no_signal_posterior_variance_is_positive():
    rng = np.random.default_rng(SEED_UNI_TAU2_NEGATIVE)
    frame, roles, _ = synth_panel(rng, n_players=200, k=1, T=np.zeros((1, 1)), cols=["x"],
                                  theta_fixed=[0.1])
    u = UnivariateEB(SHRINK["eps_abs"]).fit(frame, "x", roles, min_toi_s=300.0)
    assert u.tau2["C"] < 0
    th, v, w = u.posterior(np.array(["C", "C"]), np.array([1.0, 0.0]), np.array([3600.0, 0.0]),
                           np.array([3.0, 0.0]))
    assert np.allclose(w, 0.0)
    assert (v > 0).all()
    assert np.allclose(v, u.tau2_upper["C"])
    assert u.prior_var("C") == pytest.approx(u.tau2_upper["C"])


def test_hard_gate_rejects_zero_variance_where_w_is_zero():
    df = pd.DataFrame({"player_id": ["a", "b"], "rating": [0.0, 0.0], "var_post": [0.0, 0.1],
                       "w": [0.0, 0.0], "fallback": [True, True],
                       "method": ["univariate_fallback"] * 2})
    g = stage_l5.hard_gates({"O0_test": df}, {})
    assert g["var_positive_where_w_zero_O0_test"] is False


def test_stage_no_signal_rows_carry_flag_and_positive_variance(tmp_path):
    rng = np.random.default_rng(14)
    pm_tr = synth_pm(rng, "train", n_players=120)
    pm_te = synth_pm(rng, "test", n_players=45, match_offset=700_000,
                     zero_rate_players=("P00000",))
    for pm in (pm_tr, pm_te):
        pm[stage_l5.COMP_COLS] = 0.0
    out, _ = _run(cfg_for(tmp_path), _l3(pm_tr, pm_te))
    o1 = out["O1"]["test"]
    no_sig = o1[o1.w == 0]
    assert len(no_sig) == len(o1)
    assert (no_sig.var_post > 0).all()
    assert (no_sig.flag == NO_SIGNAL_FLAG).all()
    assert (no_sig.reliability == "LOW").all()
    for r in ROLES:
        assert out["prior"][r]["O1"]["var"] > 0
        assert out["prior"][r]["O0"]["var"] > 0


# ---------------------------------------------------------------------------------------
# Находка 3: _solve не маскирует ошибку формы под провал Холецкого

def test_solve_shape_mismatch_raises_value_error_and_logs_nothing():
    rng = np.random.default_rng(29)
    frame, roles, _ = synth_panel(rng, n_players=200, k=2)
    eb = JointEB(COLS2, SHRINK).fit(frame, roles)
    log: list = []
    with pytest.raises(ValueError):
        eb._solve(np.eye(2), np.ones((3, 3)), "PX", log)
    assert log == []
    with pytest.raises(ValueError):
        eb._solve(np.ones((2, 3)), np.ones((2, 1)), "PY", log)
    assert log == []


def test_solve_logs_only_the_eps_that_succeeded():
    """diag(1, −1e-3): ε = 1e-6…1e-3 не помогают, проходит только 1e-2 — в логе ровно он."""
    rng = np.random.default_rng(29)
    frame, roles, _ = synth_panel(rng, n_players=200, k=2)
    eb = JointEB(COLS2, SHRINK).fit(frame, roles)
    log: list = []
    _, eps = eb._solve(np.diag([1.0, -1e-3]), np.eye(2), "J", log)
    assert eps == pytest.approx(1e-2)
    assert log == [{"player_id": "J", "eps": eps}]


# ---------------------------------------------------------------------------------------
# Находка 4: строка центрируется центром того оценщика, которым она посчитана

def test_univariate_fallback_rows_centred_on_univariate_estimates():
    """m_r — среднее по амплуа с весом TOI, поэтому взвешенное среднее Overall обязано быть 0."""
    rng = np.random.default_rng(2)
    frame, roles, _ = synth_panel(rng, n_players=300, k=2)
    eb = JointEB(COLS2, SHRINK).fit(frame, roles)
    forced = copy.deepcopy(eb.with_mu({r: np.array([0.5, 0.5]) for r in ROLES}))
    forced.overall_no_signal = {r: True for r in ROLES}
    forced.uni["C"]["tau2"] = 0.02
    res = _overall(forced, frame, roles)
    assert set(res["method"]) == {"univariate_fallback"}
    assert (res["w"] > 0).all(), "сценарий: одномерная модель с сигналом"
    toi = _player_sums(frame, COLS2).toi_s.to_numpy(dtype=float)
    assert abs(float((res["overall"] * toi).sum() / toi.sum())) < 1e-10
    assert np.allclose(res["overall"] + res["m_r_overall"]["C"], res["theta_overall"])


# ---------------------------------------------------------------------------------------
# Находка 5: жёсткие гейты не проходят вхолостую

def test_hard_gates_fail_on_empty_frame():
    empty = pd.DataFrame({"player_id": [], "rating": [], "var_post": [], "w": [],
                          "fallback": [], "method": [], "comp_x": []})
    g = stage_l5.hard_gates({"O1_test": empty}, {})
    assert g["nonempty_O1_test"] is False


def test_components_gate_not_applicable_without_joint_rows():
    df = pd.DataFrame({"player_id": ["a", "b"], "rating": [1.0, 2.0], "var_post": [0.1, 0.1],
                       "w": [0.5, 0.5], "fallback": [True, True],
                       "method": ["univariate_fallback"] * 2, "comp_x": [99.0, -99.0]})
    g = stage_l5.hard_gates({"O1_test": df}, {})
    assert g["components_sum_to_overall_O1_test"] == stage_l5.NOT_APPLICABLE
    assert "components_sum_to_overall_O1_test" not in stage_l5.enforceable(g)


# ---------------------------------------------------------------------------------------
# Находка 6: «нет допущенной экспозиции» не стирается заполнением нулём

def test_no_rate_exposure_flag_is_explicit(tmp_path, l3_small):
    out, _ = _run(cfg_for(tmp_path), l3_small)
    for name in ("O1", "O0"):
        df = out[name]["test"].set_index("player_id")
        assert df.no_rate_exposure.dtype == bool
        assert bool(df.loc["P00000", "no_rate_exposure"]) is True
        assert not df.drop(index="P00000").no_rate_exposure.any()
        assert df.loc["P00000", "toi_min"] == 0


# ---------------------------------------------------------------------------------------
# Находка 7: источник μ_r не меняется молча, его колонка входит в ключ кэша

def test_mu_source_fallback_warns_and_prior_basis_follows(tmp_path, l3_small, caplog):
    pm_tr = l3_small["train"]["player_match"].drop(columns=["is_khl_component"])
    with caplog.at_level(logging.WARNING, logger="datamatch"):
        out, _ = _run(cfg_for(tmp_path), _l3(pm_tr, l3_small["test"]["player_match"]))
    rep = out["report"]
    assert rep["prior_mu"]["test_source"] == "train_all"
    assert any("μ_r" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
    assert "всему train" in rep["prior_basis"]
    assert "КХЛ-компоненте" not in rep["prior_basis"]


def test_khl_component_column_is_part_of_cache_key(tmp_path, l3_small):
    cfg = cfg_for(tmp_path)
    first, _ = _run(cfg, l3_small)
    assert first["report"]["prior_mu"]["test_source"] == "khl_component"
    pm_tr = l3_small["train"]["player_match"].copy()
    pm_tr["is_khl_component"] = False
    second, _ = _run(cfg, _l3(pm_tr, l3_small["test"]["player_match"]))
    assert second["report"]["prior_mu"]["test_source"] == "train_all"
    assert second["report"]["cache_key"] != first["report"]["cache_key"]


# ---------------------------------------------------------------------------------------
# Новое требование: замороженное состояние EB для бутстрэпа L6

def _positions(pm: pd.DataFrame) -> pd.DataFrame:
    return stage_l5.player_roles(pm).rename_axis("player_id").reset_index()


def test_apply_frozen_overall_reproduces_point_l5_test_output(tmp_path, l3_small):
    cfg = cfg_for(tmp_path)
    out, cache = _run(cfg, l3_small)
    state = stage_l5.load_frozen_overall(cfg, cache)
    pm_te = l3_small["test"]["player_match"]
    before = pm_te.copy()
    got = stage_l5.apply_frozen_overall(state, pm_te, _positions(pm_te), cfg)
    pd.testing.assert_frame_equal(pm_te, before)
    assert list(got.columns) == FROZEN_COLUMNS
    ref = out["O1"]["test"].rename(columns={"rating": "overall"})[FROZEN_COLUMNS]
    pd.testing.assert_frame_equal(got.reset_index(drop=True), ref.reset_index(drop=True),
                                  check_exact=True)


def test_frozen_state_is_json_serialisable_and_roundtrips(tmp_path, l3_small):
    cfg = cfg_for(tmp_path)
    _, cache = _run(cfg, l3_small)
    state = stage_l5.load_frozen_overall(cfg, cache)
    restored = json.loads(json.dumps(state))
    pm_te = l3_small["test"]["player_match"]
    a = stage_l5.apply_frozen_overall(state, pm_te, _positions(pm_te), cfg)
    b = stage_l5.apply_frozen_overall(restored, pm_te, _positions(pm_te), cfg)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert state["m_r_policy"]
    assert set(state["roles_train"]["role"]) <= set(ROLES)


def test_apply_frozen_overall_on_replicate_recentres_and_keeps_priors(tmp_path, l3_small):
    """Реплика матчей: моменты и μ_r заморожены, m_r пересчитан на реплике (как в точке L5)."""
    cfg = cfg_for(tmp_path)
    _, cache = _run(cfg, l3_small)
    state = stage_l5.load_frozen_overall(cfg, cache)
    frozen_copy = copy.deepcopy(state)
    pm_te = l3_small["test"]["player_match"]
    rng = np.random.default_rng(0)
    matches = pm_te.match_id.unique()
    pick = rng.choice(matches, size=len(matches), replace=True)
    rep = pd.concat([pm_te[pm_te.match_id == m].assign(match_id=f"{m}#{j}")
                     for j, m in enumerate(pick)], ignore_index=True)
    got = stage_l5.apply_frozen_overall(state, rep, _positions(pm_te), cfg)
    assert state == frozen_copy, "применение не меняет замороженное состояние"
    assert set(got.player_id) == set(pm_te.player_id)
    assert np.isfinite(got.overall).all() and (got.var_post > 0).all()
    toi = (rep[rep.toi_rate_s > 0].groupby("player_id").toi_rate_s.sum()
           .reindex(got.player_id).fillna(0.0).to_numpy())
    roles = stage_l5.player_roles(pm_te).reindex(got.player_id).to_numpy()
    joint = got.method.to_numpy() == "joint"
    for r in ROLES:
        m = (roles == r) & joint
        if toi[m].sum() > 0:
            assert abs(float((got.overall.to_numpy()[m] * toi[m]).sum() / toi[m].sum())) < 1e-9


def test_load_frozen_overall_raises_clear_error_when_missing(tmp_path):
    cfg = cfg_for(tmp_path)
    with pytest.raises(stage_l5.FrozenStateMissing, match="L5"):
        stage_l5.load_frozen_overall(cfg, Cache(cfg, {"synthetic": 1}))
