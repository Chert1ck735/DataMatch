"""L3 (уровень A): категории журнала, раздел δ при потере, ситуации, player_match и стадия.

Ожидания выведены вручную из контракта журнала (архитектура, L3 «Журнал вклада»):
одна категория на action_id с приоритетом бросок → потеря → отбор → вбрасывание → создание;
вариант split: потерявшему α·δ, вернувшему −(1−α)·δ; после броска передачи нет.
"""
import numpy as np
import pandas as pd
import pytest

from datamatch import l3_contrib, stage_l3
from datamatch.cache import Cache
from datamatch.config import load_config
from datamatch.gates import GateError
from datamatch.l2_value import td_delta
from tests.test_l0 import AWAY, HOME, lineup_start
from tests.test_revision_l0 import WH, _full
from tests.test_zone_starts import fo, sub, touch


def chain():
    """H: faceoff -> pass, потеря -> A: возврат и сразу неточный пас (потеря) -> H: возврат, бросок
    с потерей -> A: подбор после броска (терминальный)."""
    return pd.DataFrame({
        "node": np.arange(6), "match_id": "M", "segment": 1, "period": 1, "clock": np.arange(6.),
        "home_team_id": "H",
        "team_id": ["H", "H", "A", "H", "H", "A"],
        "actor_id": ["h0", "h1", "a1", "h2", "h2", "a2"],
        "action": ["faceoff", "pass", "pass", "hold", "shot", "hold"],
        "arrival": ["faceoff", "after_faceoff", "recovery", "recovery", "continuation", "recovery"],
        "sigma_next": [1, -1, -1, 1, -1, 0],
        "next_node": [1, 2, 3, 4, 5, -1],
        "valid_for_transition": True, "valid_for_action_attribution": True,
        "valid_for_rate_metrics": True, "mp": "EV5"})


def test_journal_one_category_per_action_with_priority():
    jr = l3_contrib.journal(chain(), np.array([1., 2., 4., 8., 16., 32.]))
    assert jr.action_id.is_unique
    # a1: возврат (recovery) и потеря одновременно -> потеря; h2 hold после возврата -> отбор;
    # бросок со сменой владения -> бросок; вбрасывание с сохранением шайбы -> вбрасывание
    assert list(jr.category) == ["faceoff", "loss", "loss", "recovery", "shot", "recovery"]


def test_split_credit_matches_manual_ledger_and_telescopes():
    nd = chain()
    r = np.array([0., 0., 0., 0., .1, 0.])
    v = np.array([.02, .03, -.04, .05, .06, .01])
    d = td_delta(nd, r, v)
    jr = l3_contrib.journal(nd, d, "split", .5)
    exp = np.array([d[0], .5 * d[1], .5 * d[2] - .5 * d[1], d[3] - .5 * d[2], d[4], d[5]])
    np.testing.assert_allclose(jr.credit, exp)
    rep = l3_contrib.telescoping_check(nd, jr, r, v)
    assert rep["n_segments"] == 1 and rep["max_abs_diff"] < 1e-12
    # вариант full считает переход дважды: сумма уже не телескопируется
    full = l3_contrib.journal(nd, d, "full", .5)
    np.testing.assert_allclose(full.credit[[1, 2]], [d[1], d[2] - d[1]])
    assert l3_contrib.telescoping_check(nd, full, r, v)["max_abs_diff"] > 1e-6


def test_unknown_turnover_delta_transfers_nothing_and_keeps_recoverer_delta():
    d = np.array([1., np.nan, 4., 8., 16., 32.])
    jr = l3_contrib.journal(chain(), d)
    assert np.isnan(jr.credit[1]) and not jr.valid_for_credit[1]
    assert jr.credit[2] == pytest.approx(2.)  # своя половина δ2, без NaN от предыдущей потери
    assert bool(jr.recv_from_unknown[2]) and not jr.recv_from_unknown[3]


def test_situation_uses_stints_from_actor_perspective():
    st = pd.DataFrame({"match_id": "M", "stint": [0, 1, 2, 3],
                       "n_home_sk": [5, 5, 4, 6], "n_away_sk": [5, 4, 4, 5],
                       "home_g_in": [True, True, True, False], "away_g_in": True,
                       "is_5v5": [True, False, False, False]})
    nd = pd.DataFrame({"match_id": "M", "stint": [0, 1, 1, 2, 3, 9], "home_team_id": "H",
                       "team_id": ["H", "H", "A", "A", "H", "H"]})
    assert list(l3_contrib.situation(nd, st)) == ["5v5", "pp", "sh", "other", "other", "unknown"]


def scenario():
    rows = lineup_start(1) + [fo(HOME, 0.), touch(HOME, "H1", 2.),
        dict(event_type="pass_failed", actor_id="H1", actor_team_id=HOME, period=1, clock_s=3., x=5., y=0.),
        touch(AWAY, "A1", 4., x=-5.), sub(AWAY, "A9", "A2", 10.),
        dict(event_type="shot_on_target", actor_id="A9", actor_team_id=AWAY, period=1, clock_s=10., x=20., y=0.),
        sub(AWAY, "A2", "A9", 10.), touch(HOME, "H2", 12.), WH(20.)]
    tabs, _ = _full(rows)
    nd = tabs["nodes"]
    r = np.where(nd.action.eq("shot"), .1, 0.)
    v = np.linspace(.01, .06, len(nd))
    values = nd[["node", "match_id"]].assign(r=r, value=v, delta=td_delta(nd, r, v))
    ids = [f"{p}{k}" for p in "HA" for k in range(5)] + ["A9", "HG", "AG"]
    players = pd.DataFrame({"player_id": ids,
        "position": ["C", "LW", None, "D", "D", "C", "RW", "LW", "D", "D", "G", "G", "G"],
        "role": ["C", "W", "W", "D", "D", "C", "W", "W", "D", "D", "C", "G", "G"]})
    roster = pd.DataFrame({"match_id": "M1", "player_id": ids,
        "team_id": [HOME] * 5 + [AWAY] * 6 + [HOME, AWAY],
        "slot": ["C", "LW", "RW", "LD", "RD", "C", "RW", "LW", "LD", "RD", "C", "G", "G"]})
    return tabs, values, players, roster


CONTRACT_COLUMNS = (["match_id", "player_id", "team_id", "split", "position", "role",
    "toi_s", "toi_5v5_s", "toi_pp_s", "toi_sh_s", "toi_rate_s", "toi_rate_5v5_s", "toi_rate_pp_s",
    "toi_rate_sh_s", "goal", "ast", "points", "pm"] + [f"c_{c}" for c in l3_contrib.CATEGORIES] + ["iav_sum"]
    + [f"c_{c}_{s}" for c in l3_contrib.CATEGORIES for s in ("5v5", "pp", "pk")]
    + ["ixg", "igoals", "ishots", "xgf_5v5", "xga_5v5", "component", "is_khl_component"])


def test_player_positions_slot_fallback_and_goalies_excluded():
    _, _, players, roster = scenario()
    pos = stage_l3.player_positions(players, roster)
    assert pos.loc["H2", "position"] == "RW" and pos.loc["H2", "role"] == "W"  # из слота заявки
    assert pos.loc["A1", "role"] == "W" and pos.loc["H3", "role"] == "D"
    assert pos.loc["A9", "position"] == "C"  # позиция G в справочнике, полевой слот в заявке
    assert "HG" not in pos.index and "AG" not in pos.index


def test_contribute_player_match_contract_and_zero_toi_attribution():
    tabs, values, players, roster = scenario()
    pos = stage_l3.player_positions(players, roster)
    comp = {HOME: 0, AWAY: 3}
    out = stage_l3.contribute(tabs["nodes"], values, tabs["toi"], tabs["stints"], tabs["events"],
                              pos, comp, 0, "train", "split", .5)
    pm, jr = out["player_match"], out["journal"]
    assert set(CONTRACT_COLUMNS) <= set(pm.columns)
    assert not pm.duplicated(["match_id", "player_id"]).any()
    assert set(pm.split) == {"train"}
    assert "situation" in jr and set(jr.situation) <= set(l3_contrib.SITUATIONS)
    # A9 бросил в stint нулевой длительности: строки TOI нет, атрибуция сохранена
    a9 = pm.set_index("player_id").loc["A9"]
    assert a9.toi_s == 0 and a9.n_actions == 1 and a9.c_shot != 0 and a9.ishots == 1
    assert a9.c_shot_5v5 == pytest.approx(a9.c_shot)  # 5 на 5 по stints, а не по TOI
    jl = jr.set_index("player_id")
    assert jl.loc["A9", "category"] == "shot" and jl.loc["A1", "category"] == "recovery"
    # компоненты = IAV; разрезы по ситуациям — подмножества; всё полевое = весь журнал
    comps = pm[[f"c_{c}" for c in l3_contrib.CATEGORIES]].sum(axis=1)
    np.testing.assert_allclose(pm.iav_sum, comps)
    for c in l3_contrib.CATEGORIES:
        sits = pm[[f"c_{c}_{s}" for s in ("5v5", "pp", "pk")]].sum(axis=1)
        np.testing.assert_allclose(sits, pm[f"c_{c}"])  # здесь все действия при 5 на 5
    assert pm.iav_sum.sum() == pytest.approx(np.nansum(jr.credit[jr.valid_for_rate_metrics]))
    assert all(out["report"]["gates"].values())
    assert out["report"]["telescoping"]["max_abs_diff"] < 1e-12
    # амплуа и компонента; вратари строк не получают
    assert set(pm.role) <= {"C", "W", "D"} and not pm.player_id.isin(["HG", "AG"]).any()
    p = pm.set_index("player_id")
    assert p.loc["H1", "component"] == 0 and bool(p.loc["H1", "is_khl_component"])
    assert p.loc["A1", "component"] == 3 and not bool(p.loc["A1", "is_khl_component"])
    # on-ice 5 на 5: xG броска A9 — xGF у пятёрки A, xGA у пятёрки H
    assert p.loc["A1", "xgf_5v5"] == pytest.approx(.1) and p.loc["H1", "xga_5v5"] == pytest.approx(.1)


def test_align_values_rejects_partial_or_foreign_coverage():
    nd = chain()
    vals = nd[["node", "match_id"]].assign(r=0., value=0., delta=0.)
    assert stage_l3.align_values(nd, vals.iloc[::-1]).node.tolist() == list(range(6))
    with pytest.raises(ValueError, match="exactly once"):
        stage_l3.align_values(nd, vals.iloc[1:])
    with pytest.raises(ValueError, match="match"):
        stage_l3.align_values(nd, vals.assign(match_id="X"))


def test_run_l3_never_computes_missing_upstream(tmp_path):
    cfg = load_config().with_overrides({"paths": {"cache": str(tmp_path / "cache"),
                                                  "artifacts": str(tmp_path / "artifacts")}})
    with pytest.raises(stage_l3.UpstreamNotReady, match="prepare"):
        stage_l3.run_l3(cfg, Cache(cfg))
    assert not list((tmp_path / "cache").iterdir())


def test_run_l3_caches_result_and_reloads(tmp_path, monkeypatch):
    cfg = load_config().with_overrides({"paths": {"cache": str(tmp_path / "cache"),
                                                  "artifacts": str(tmp_path / "artifacts")}})
    tabs, values, players, roster = scenario()
    part = dict(tabs, roster=roster, games=pd.DataFrame({"match_id": ["M1"], "home_team_id": [HOME],
                                                         "away_team_id": [AWAY]}))
    l0 = {"key": "l0_test", "players": players, "train": part, "test": part}
    monkeypatch.setattr(stage_l3, "_load_l0", lambda *a: l0)
    monkeypatch.setattr(stage_l3, "_load_splits", lambda *a: {"khl_component": 0, "matches": {}})
    monkeypatch.setattr(stage_l3, "_load_l1_selection", lambda *a: {"status": "ok"})
    monkeypatch.setattr(stage_l3, "_load_l2", lambda *a: {
        "key": "l2_test", "selection": {"selected": "markov"}, "train_nodes": tabs["nodes"],
        "train_values": values, "test_values": values})
    first = stage_l3.run_l3(cfg, Cache(cfg))
    assert first["report"]["status"] == "ok"
    assert set(first["test"]["player_match"].split) == {"test"}
    calls = []
    monkeypatch.setattr(stage_l3, "contribute", lambda *a, **k: calls.append(1))
    again = stage_l3.run_l3(cfg, Cache(cfg))
    assert not calls
    pd.testing.assert_frame_equal(again["train"]["player_match"], first["train"]["player_match"])
    assert (tmp_path / "artifacts" / "contrib_report.json").exists()


def test_run_l3_refuses_value_layer_that_failed_its_gate(tmp_path, monkeypatch):
    """L2 со статусом failed не становится входом L3: GateError, а не тихий расчёт на нём."""
    cfg = load_config().with_overrides({"paths": {"cache": str(tmp_path / "cache"),
                                                  "artifacts": str(tmp_path / "artifacts")}})
    tabs, values, players, roster = scenario()
    part = dict(tabs, roster=roster, games=pd.DataFrame({"match_id": ["M1"], "home_team_id": [HOME],
                                                         "away_team_id": [AWAY]}))
    monkeypatch.setattr(stage_l3, "_load_l0", lambda *a: {"key": "k", "players": players,
                                                          "train": part, "test": part})
    monkeypatch.setattr(stage_l3, "_load_splits", lambda *a: {"khl_component": 0, "matches": {}})
    monkeypatch.setattr(stage_l3, "_load_l1_selection", lambda *a: {"status": "ok"})

    def failed_l2(cfg_, cache_, l0_, sp_, sel_):
        from datamatch.gates import require_ok
        return require_ok({"status": "failed", "failed_gates": ["calibrated_all_deciles"]},
                          "L2 value_selected.json")

    monkeypatch.setattr(stage_l3, "_load_l2", failed_l2)
    with pytest.raises(GateError, match="failed"):
        stage_l3.run_l3(cfg, Cache(cfg))
    assert not list((tmp_path / "cache").glob("l3*/_DONE"))
