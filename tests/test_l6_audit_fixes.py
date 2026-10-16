"""L6: регрессии подтверждённых находок аудита (06.10.2026).

Каждый тест воспроизводит дефект на синтетике схемы контракта L5 → L6 и фиксирует ожидаемое
поведение из независимого контракта: архитектура L6 (полная дисперсия по B ≥ 200 репликам,
интервал O0 — только апостериорная дисперсия, «сигнал не обнаружен»), правила проекта (никаких
молчаливых подстановок и устаревших результатов) и README задания (test_ratings.csv).
"""
import json

import numpy as np
import pandas as pd
import pytest

from datamatch import l6_replicate as R
from datamatch import l6_uncertainty as U
from datamatch import stage_l6 as S
from datamatch.cache import Cache
from datamatch.config import load_config
from datamatch.gates import GateError
from tests.test_l6_ratings import MATCHES, ROLES, l5_dict, latent_table, members_frame, replicate_factory


@pytest.fixture
def env(tmp_path):
    cfg = load_config().with_overrides({"paths": {
        "outputs": str(tmp_path / "outputs"), "artifacts": str(tmp_path / "artifacts"),
        "cache": str(tmp_path / "cache"), "logs": str(tmp_path / "logs")}})
    return cfg, Cache(cfg, {})


def _report(cfg) -> dict:
    return json.loads((cfg.paths.artifacts / S.L6_REPORT).read_text())


def _covering_replicate(ids, scale=0.05):
    def replicate(matches, index, seed):
        rng = np.random.default_rng(seed)
        return pd.Series(rng.normal(0.0, scale, len(ids)), index=list(ids))
    return replicate


# --------------------------------------------------------------------------------------
# 1. CRITICAL: full без выполненного бутстрэпа не пишет сдаваемый файл; fast — отдельный путь
# --------------------------------------------------------------------------------------
def test_full_mode_without_replicate_hook_raises_and_writes_no_deliverable(env):
    # Arrange
    cfg, cache = env
    t = latent_table()

    # Act / Assert
    with pytest.raises(GateError):
        S.run_l6(cfg, cache, l5_dict(t), mode="full", members=members_frame(t))
    full = S.output_paths(cfg, "full")
    assert not full.csv.exists() and not full.meta.exists()
    rep = _report(cfg)
    assert rep["status"] == "failed" and rep["mode"] == "full"


def test_full_mode_with_partial_bootstrap_raises(env):
    # Arrange: бюджет времени исчерпан после части реплик -> partial
    cfg, cache = env
    t = latent_table(n=3)
    m = members_frame(t, lineup_only=0)
    exhausted = cfg.with_overrides({"bootstrap": {"time_budget_s": 0}})   # бюджет полного режима — из конфига

    # Act / Assert
    with pytest.raises(GateError):
        S.run_l6(exhausted, cache, l5_dict(t), mode="full", members=m,
                 replicate=replicate_factory(0.01), bootstrap_matches=MATCHES)
    assert not S.output_paths(cfg, "full").csv.exists()


def test_fast_mode_writes_its_own_file_and_never_the_deliverable(env):
    # Arrange
    cfg, cache = env
    t = latent_table()

    # Act
    res = S.run_l6(cfg, cache, l5_dict(t), mode="fast", members=members_frame(t))

    # Assert
    fast, full = S.output_paths(cfg, "fast"), S.output_paths(cfg, "full")
    assert fast.csv.name == "test_ratings_fast.csv" and full.csv.name == "test_ratings.csv"
    assert fast.csv.exists() and fast.meta.exists() and fast.detail.exists()
    assert not full.csv.exists() and not full.meta.exists()
    assert res["meta"]["model_bootstrap"]["B_requested"] == cfg["bootstrap"]["debug_B"]
    assert res["meta"]["model_bootstrap"]["B_done"] == 0


def test_full_mode_with_done_bootstrap_records_B_in_meta_and_detail(env):
    # Arrange
    cfg, cache = env
    t = latent_table(n=3)
    m = members_frame(t, lineup_only=0)

    # Act
    res = S.run_l6(cfg, cache, l5_dict(t), mode="full", members=m,
                   replicate=replicate_factory(0.02), bootstrap_matches=MATCHES)

    # Assert
    full = S.output_paths(cfg, "full")
    b = cfg["bootstrap"]["final_B"]
    assert full.csv.exists() and full.detail.exists()
    meta = json.loads(full.meta.read_text())
    assert meta["model_bootstrap"]["B_requested"] == b and meta["model_bootstrap"]["B_done"] == b
    detail = pd.read_csv(full.detail, dtype={"player_id": str})
    assert (detail.n_replicates == b).all()
    assert res["report"]["status"] == "ok"
    assert _report(cfg)["status"] == "ok"


# --------------------------------------------------------------------------------------
# 2. HIGH: бутстрэп ресэмплирует ровно матчи финальной подгонки L2
# --------------------------------------------------------------------------------------
def test_bootstrap_resamples_exactly_the_final_fit_matches(env, monkeypatch):
    # Arrange
    cfg, cache = env
    t = latent_table(n=3)
    m = members_frame(t, lineup_only=0)
    allowed = [f"F{i:03d}" for i in range(25)]
    seen: set[str] = set()
    monkeypatch.setattr(R, "bootstrap_match_ids", lambda cfg_, cache_: list(allowed))

    def recording(matches, index, seed):
        seen.update(map(str, matches))
        assert len(matches) == len(allowed)
        return pd.Series(np.random.default_rng(seed).normal(0, 0.01, 3), index=list(t.player_id))

    # Act
    S.run_l6(cfg, cache, l5_dict(t), members=m, replicate=recording)

    # Assert
    assert seen and seen <= set(allowed)
    assert not hasattr(S, "_train_match_ids")


def test_final_fit_matches_come_from_stage_l2_contract(monkeypatch):
    # Arrange
    from datamatch import stage_l2
    expected = ["G0002", "G0001", "G0003"]
    calls = []

    def fake(splits, schemes):
        calls.append((splits, schemes))
        return list(expected)
    monkeypatch.setattr(stage_l2, "final_fit_match_ids", fake)

    # Act
    got = R.final_fit_matches({"matches": {}}, ("v1",))

    # Assert
    assert got == sorted(expected) and calls == [({"matches": {}}, ("v1",))]


def test_final_fit_matches_reject_empty_or_duplicated_lists(monkeypatch):
    from datamatch import stage_l2
    monkeypatch.setattr(stage_l2, "final_fit_match_ids", lambda s, sc: [])
    with pytest.raises(R.ReplicateUnavailable):
        R.final_fit_matches({}, ("v1",))
    monkeypatch.setattr(stage_l2, "final_fit_match_ids", lambda s, sc: ["G1", "G1"])
    with pytest.raises(R.ReplicateUnavailable):
        R.final_fit_matches({}, ("v1",))


# --------------------------------------------------------------------------------------
# 3. HIGH: пропущенная или нефинитная Var_m не превращается в ноль
# --------------------------------------------------------------------------------------
def test_replicate_missing_players_fails_var_model_gate(env):
    # Arrange: реплика возвращает Overall только части игроков
    cfg, cache = env
    t = latent_table()
    m = members_frame(t)
    subset = _covering_replicate([f"P{i:05d}" for i in range(5)])

    # Act / Assert
    with pytest.raises(GateError):
        S.run_l6(cfg, cache, l5_dict(t), members=m, replicate=subset, bootstrap_matches=MATCHES)
    assert _report(cfg)["status"] == "failed"
    assert not S.output_paths(cfg, "fast").csv.exists()


def test_nonfinite_overall_in_any_replicate_fails_bootstrap(env):
    # Arrange
    cfg, cache = env
    t = latent_table()
    m = members_frame(t)
    ids = list(m.player_id)

    def with_inf(matches, index, seed):
        v = np.random.default_rng(seed).normal(0, 0.1, len(ids))
        if index == 3:
            v[0] = np.inf
        return pd.Series(v, index=ids)

    # Act / Assert
    with pytest.raises(GateError):
        S.run_l6(cfg, cache, l5_dict(t), members=m, replicate=with_inf, bootstrap_matches=MATCHES)
    assert "failed" in json.dumps(_report(cfg)["model_bootstrap"])


def test_total_variance_rejects_nonfinite_model_variance():
    with pytest.raises(ValueError):
        U.total_variance([0.01, 0.02], var_model=[0.01, np.nan])
    with pytest.raises(ValueError):
        U.total_variance([0.01, 0.02], var_model=[0.01, np.inf])
    with pytest.raises(ValueError):
        U.total_variance([0.01, 0.02], var_model=[0.01, -0.5])


# --------------------------------------------------------------------------------------
# 4. HIGH: ошибки реплик не глотаются целиком
# --------------------------------------------------------------------------------------
def test_programming_error_in_replicate_propagates(env):
    # Arrange
    cfg, cache = env
    t = latent_table(n=3)

    def buggy(matches, index, seed):
        raise KeyError("c_shot")

    # Act / Assert
    with pytest.raises(KeyError):
        S.run_l6(cfg, cache, l5_dict(t), members=members_frame(t, lineup_only=0),
                 replicate=buggy, bootstrap_matches=MATCHES)
    assert _report(cfg)["status"] == "failed"


def test_data_errors_above_threshold_fail_the_bootstrap(env):
    # Arrange: 40% реплик падают на ожидаемой ошибке данных
    cfg, cache = env
    t = latent_table(n=3)
    ids = list(t.player_id)

    def flaky(matches, index, seed):
        if index % 5 < 2:
            raise ValueError("Folds leave insufficient training matches")
        return pd.Series(np.random.default_rng(seed).normal(0, 0.1, 3), index=ids)

    # Act / Assert
    with pytest.raises(GateError):
        S.run_l6(cfg, cache, l5_dict(t), members=members_frame(t, lineup_only=0),
                 replicate=flaky, bootstrap_matches=MATCHES)
    mb = _report(cfg)["model_bootstrap"]
    assert mb["status"] == "failed" and mb["n_failed"] > 0
    assert mb["max_failure_share"] == U.MAX_REPLICATE_FAILURE_SHARE


# --------------------------------------------------------------------------------------
# 5. HIGH: O0 не получает Var_m
# --------------------------------------------------------------------------------------
def test_o0_interval_is_posterior_only_even_with_a_replicate(env):
    # Arrange
    cfg, cache = env
    t = latent_table()
    m = members_frame(t)
    called = []

    def replicate(matches, index, seed):
        called.append(index)
        return replicate_factory(0.3)(matches, index, seed)

    # Act
    res = S.run_l6(cfg, cache, l5_dict(t, report={"final_definition": "O0"}), mode="full",
                   members=m, replicate=replicate, bootstrap_matches=MATCHES)

    # Assert
    d = res["detail"].set_index("player_id")
    assert called == []
    assert res["meta"]["rating_definition"] == "O0"
    assert res["meta"]["model_bootstrap"]["status"] == "not_applicable"
    assert res["meta"]["interval"]["var_model_included"] is False
    assert np.allclose(d.v_tot, d.var_post)
    assert S.output_paths(cfg, "full").csv.exists()


# --------------------------------------------------------------------------------------
# 6. HIGH: провенанс L6 видит цепочку L1–L5, доступность реплики и список матчей
# --------------------------------------------------------------------------------------
def test_provenance_covers_upstream_code_hook_and_match_list(env):
    # Arrange
    cfg, cache = env
    t = latent_table()
    m = members_frame(t)

    # Act
    res = S.run_l6(cfg, cache, l5_dict(t), members=m)

    # Assert
    prov = res["meta"]["provenance"]
    for k in ("chain_code", "replicate_hook", "bootstrap_matches"):
        assert k in prov, k


def test_file_is_not_reused_after_replicate_hook_becomes_available(env, monkeypatch):
    # Arrange
    cfg, cache = env
    t = latent_table()
    m = members_frame(t)
    S.run_l6(cfg, cache, l5_dict(t), members=m)
    ids = list(m.player_id)
    monkeypatch.setattr(R, "bootstrap_match_ids", lambda cfg_, cache_: list(MATCHES))
    monkeypatch.setattr(R, "make_replicate", lambda cfg_, cache_, force=False: _covering_replicate(ids))
    monkeypatch.setattr(R, "executor_factory", lambda cfg_, cache_=None: None)

    # Act
    again = S.run_l6(cfg, cache, l5_dict(t), members=m)

    # Assert
    assert again["reused"] is False
    assert again["meta"]["model_bootstrap"]["status"] == "done"


def test_full_mode_never_reuses_a_partial_file(env):
    # Arrange: файл full с прежним статусом partial (например, от старого кода)
    cfg, cache = env
    t = latent_table(n=3)
    m = members_frame(t, lineup_only=0)
    S.run_l6(cfg, cache, l5_dict(t), mode="full", members=m,
             replicate=replicate_factory(0.01), bootstrap_matches=MATCHES)
    full = S.output_paths(cfg, "full")
    meta = json.loads(full.meta.read_text())
    meta["status"] = "partial"
    full.meta.write_text(json.dumps(meta, ensure_ascii=False))

    # Act
    again = S.run_l6(cfg, cache, l5_dict(t), mode="full", members=m,
                     replicate=replicate_factory(0.01), bootstrap_matches=MATCHES)

    # Assert
    assert again["reused"] is False


# --------------------------------------------------------------------------------------
# 7. MEDIUM: нет prior амплуа -> ошибка, а не рейтинг 0 с нулевой шириной
# --------------------------------------------------------------------------------------
def test_missing_role_prior_raises(env):
    cfg, cache = env
    t = latent_table()
    l5 = l5_dict(t)
    l5["prior"] = {}
    with pytest.raises(GateError):
        S.run_l6(cfg, cache, l5, members=members_frame(t, lineup_only=2))


# --------------------------------------------------------------------------------------
# 8. MEDIUM: флаг L5 переносится, вырожденные интервалы считаются и блокируются
# --------------------------------------------------------------------------------------
def test_l5_flag_is_carried_to_detail_and_counted_in_meta(env):
    # Arrange
    cfg, cache = env
    t = latent_table()
    t["flag"] = ""
    t.loc[0, "flag"] = "сигнал не обнаружен"
    t.loc[0, "w"] = 0.0

    # Act
    res = S.run_l6(cfg, cache, l5_dict(t), members=members_frame(t))

    # Assert
    d = res["detail"].set_index("player_id")
    assert d.loc[t.player_id[0], "flag"] == "сигнал не обнаружен"
    assert res["meta"]["rows"]["by_flag"]["сигнал не обнаружен"] == 1
    assert res["meta"]["rows"]["n_degenerate_intervals"] == 0
    detail = pd.read_csv(S.output_paths(cfg, "fast").detail, dtype={"player_id": str}, keep_default_na=False)
    assert "сигнал не обнаружен" in set(detail.flag)


def test_zero_signal_without_l5_flag_column_fails_loudly(env):
    cfg, cache = env
    t = latent_table().drop(columns=["flag"])
    t.loc[0, "w"] = 0.0
    with pytest.raises(GateError):
        S.run_l6(cfg, cache, l5_dict(t), members=members_frame(t))


def test_zero_width_interval_fails_the_hard_gate(env):
    cfg, cache = env
    t = latent_table()
    t.loc[0, ["var_post", "w"]] = [0.0, 0.0]
    t.loc[0, "flag"] = "сигнал не обнаружен"
    with pytest.raises(GateError):
        S.run_l6(cfg, cache, l5_dict(t), members=members_frame(t))
    rep = _report(cfg)
    assert "interval_nondegenerate" in rep["failed_gates"]


# --------------------------------------------------------------------------------------
# 9. MEDIUM: проверка покрытия по опубликованным w/var_post — не тест покрытия
# --------------------------------------------------------------------------------------
def test_self_consistency_check_is_labelled_not_a_coverage_test(env):
    cfg, cache = env
    t = latent_table(n=50, seed=3)
    res = S.run_l6(cfg, cache, l5_dict(t), members=members_frame(t, lineup_only=0))
    cov = res["report"]["coverage_simulation"]
    assert cov["self_consistency"]["kind"] == "not_a_coverage_test"
    assert cov["refit_simulation"]["status"] == "not_run"         # моментов в отчёте L5 нет


def test_refit_simulation_runs_when_l5_publishes_moments(env):
    cfg, cache = env
    t = latent_table(n=120, seed=5)
    report = {"gates_O1": {"ok": True},
              "moments": {"univariate": {r: {"tau2": 0.04, "sigma2": 1.5, "omega2": 0.02} for r in ROLES}}}
    res = S.run_l6(cfg, cache, l5_dict(t, report=report), members=members_frame(t, lineup_only=0))
    sim = res["report"]["coverage_simulation"]["refit_simulation"]
    assert sim["status"] == "ok" and sim["kind"] == "refit_simulation"
    assert 0.80 <= sim["coverage"] <= 1.0


def test_refit_simulation_is_near_nominal_when_moments_are_identified():
    """Моменты оцениваются заново процедурой L5; при различимых σ² и ω² покрытие ≈ 95%.

    Матчи и минуты на матч варьируются независимо, иначе 1/T и 1/M коллинеарны и σ², ω²
    не разделяются (это и есть то, что симуляция должна уметь показать на реальной структуре).
    """
    rng = np.random.default_rng(0)
    n = 300
    games = rng.integers(4, 60, n).astype(float)
    toi = games * rng.uniform(6, 24, n)
    roles = np.array([ROLES[i % 3] for i in range(n)], dtype=object)
    mom = {r: {"tau2": 0.04, "sigma2": 1.5, "omega2": 0.02} for r in ROLES}
    res = U.simulate_refit_coverage(toi, games, roles, mom, n_sim=20, seed=1, min_toi_s=300.0)
    assert res["coverage"] == pytest.approx(0.95, abs=0.03)
    assert res["kind"] == "refit_simulation" and res["moments_reestimated"] is True


# --------------------------------------------------------------------------------------
# 10. MEDIUM: Монте-Карло-ошибка границ по каждому игроку — в файле детализации
# --------------------------------------------------------------------------------------
def test_detail_file_has_per_player_mc_error_and_replicate_counts(env):
    # Arrange
    cfg, cache = env
    t = latent_table()
    m = members_frame(t)

    # Act
    S.run_l6(cfg, cache, l5_dict(t), members=m, replicate=replicate_factory(0.05),
             bootstrap_matches=MATCHES)

    # Assert
    detail = pd.read_csv(S.output_paths(cfg, "fast").detail, dtype={"player_id": str}, keep_default_na=False)
    for c in S.DETAIL_COLUMNS:
        assert c in detail.columns, c
    for c in ("player_id", "rating", "var_post", "var_model", "mc_se_bound", "w", "flag", "method",
              "n_replicates"):
        assert c in S.DETAIL_COLUMNS, c
    assert sorted(detail.player_id) == sorted(m.player_id)
    assert np.isfinite(detail.mc_se_bound.astype(float)).all()
    assert (detail.n_replicates == cfg["bootstrap"]["debug_B"]).all()


# --------------------------------------------------------------------------------------
# 11. MEDIUM: полевой игрок с неопределённым амплуа не выпадает молча
# --------------------------------------------------------------------------------------
def _role_env(tmp_path, players, toi, roster, seasons_ids):
    td = tmp_path / "test"
    td.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"player_id": seasons_ids, "position": None, "lineup_slot_mode": None,
                  "shoots": None}).to_csv(td / "players_new.csv", index=False)
    pd.DataFrame({"player_id": seasons_ids, "league": "KHL", "season": "2025-2026", "team_id": "T113",
                  "games_in_sample": 1}).to_csv(td / "player_seasons_test.csv", index=False)
    cfg = load_config().with_overrides({"paths": {"test_dir": str(td), **{
        k: str(tmp_path / k) for k in ("outputs", "artifacts", "cache", "logs")}}})
    tabs = {"players": players, "toi": toi, "roster": roster, "key": "k", "quality_status": "ok"}
    return cfg, tabs


def test_field_player_with_shifts_and_unresolved_role_raises(tmp_path, monkeypatch):
    # Arrange
    players = pd.DataFrame({"player_id": ["P00000", "P00001"], "role": ["C", None]})
    toi = pd.DataFrame({"player_id": ["P00000", "P00001"], "match_id": ["M1", "M1"], "toi_s": [600.0, 500.0]})
    roster = pd.DataFrame({"player_id": ["P00000", "P00001"], "team_id": "T113"})
    cfg, tabs = _role_env(tmp_path, players, toi, roster, ["P00000", "P00001"])
    monkeypatch.setattr(S, "_l0_test_tables", lambda cfg_, cache_: tabs)

    # Act / Assert
    with pytest.raises(GateError, match="P00001"):
        S.test_field_players(cfg, Cache(cfg, {}))


def test_unresolved_role_without_shifts_breaks_coverage_against_untrimmed_set(tmp_path, monkeypatch):
    # Arrange
    players = pd.DataFrame({"player_id": ["P00000", "P99999"], "role": ["C", None]})
    toi = pd.DataFrame({"player_id": ["P00000"], "match_id": ["M1"], "toi_s": [600.0]})
    roster = pd.DataFrame({"player_id": ["P00000", "P99999"], "team_id": "T113"})
    cfg, tabs = _role_env(tmp_path, players, toi, roster, ["P00000", "P99999"])
    monkeypatch.setattr(S, "_l0_test_tables", lambda cfg_, cache_: tabs)
    t = latent_table(n=1)

    # Act / Assert
    members, rep = S.test_field_players(cfg, Cache(cfg, {}))
    assert "P99999" in rep["expected_field_ids"] and "P99999" not in set(members.player_id)
    with pytest.raises(GateError):
        S.run_l6(cfg, Cache(cfg, {}), l5_dict(t), members=None)
    assert "covers_all_field_players" in _report(cfg)["failed_gates"]


# --------------------------------------------------------------------------------------
# 12. LOW: строгие гейты L5, пустой состав, avangard_team_id, устаревший сдаваемый файл
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("gates", [
    {"gates_l5": {"signal_present_C": None}},
    {"gates_l5": {"cholesky_ok": float("nan")}},
    {"gates_l5": {"x": 1}},
    {"gates_l5": {}},
    {},
])
def test_l5_gates_pass_only_on_explicit_true(gates):
    ok, _ = S._l5_gates_ok(gates, "O1")
    assert ok is False


def test_l5_gates_accept_numpy_true_and_choose_o1():
    assert S._l5_gates_ok({"gates_l5": {"x": np.True_, "y": True}}, "O1")[0] is True
    t = latent_table(n=3)
    assert S.choose_definition(l5_dict(t, report={}))[0] == "O0"


def test_explicit_o1_with_failed_gates_is_rejected():
    t = latent_table(n=3)
    with pytest.raises(S.L6InputsNotReady):
        S.choose_definition(l5_dict(t, report={"rating_definition": "O1", "gates_O1": {"x": False}}))


def test_empty_member_list_fails(env):
    cfg, cache = env
    t = latent_table()
    with pytest.raises(GateError):
        S.run_l6(cfg, cache, l5_dict(t), members=members_frame(t).iloc[0:0])


def test_missing_avangard_team_id_is_an_explicit_error(tmp_path, monkeypatch):
    players = pd.DataFrame({"player_id": ["P00000"], "role": ["C"]})
    toi = pd.DataFrame({"player_id": ["P00000"], "match_id": ["M1"], "toi_s": [600.0]})
    roster = pd.DataFrame({"player_id": ["P00000"], "team_id": "T113"})
    cfg, tabs = _role_env(tmp_path, players, toi, roster, ["P00000"])
    cfg = cfg.with_overrides({"avangard_team_id": None})
    monkeypatch.setattr(S, "_l0_test_tables", lambda cfg_, cache_: tabs)
    with pytest.raises(ValueError, match="avangard_team_id"):
        S.test_field_players(cfg, Cache(cfg, {}))


def test_failed_full_run_marks_previous_deliverable_stale(env):
    # Arrange: прежний успешный full
    cfg, cache = env
    t = latent_table(n=3)
    m = members_frame(t, lineup_only=0)
    S.run_l6(cfg, cache, l5_dict(t), mode="full", members=m,
             replicate=replicate_factory(0.01), bootstrap_matches=MATCHES)
    full = S.output_paths(cfg, "full")
    old = full.csv.read_text()

    # Act: следующий full падает (нет реплики)
    with pytest.raises(GateError):
        S.run_l6(cfg, cache, l5_dict(t), mode="full", members=m, force=True)

    # Assert
    assert not full.csv.exists() and not full.meta.exists()
    stale = full.csv.with_name(full.csv.name + S.STALE_SUFFIX)
    assert stale.exists() and stale.read_text() == old
    assert _report(cfg)["status"] == "failed"


def test_failed_fast_run_does_not_touch_the_full_deliverable(env):
    cfg, cache = env
    t = latent_table(n=3)
    m = members_frame(t, lineup_only=0)
    S.run_l6(cfg, cache, l5_dict(t), mode="full", members=m,
             replicate=replicate_factory(0.01), bootstrap_matches=MATCHES)
    bad = m.copy()
    bad.loc[0, "role"] = "G"
    with pytest.raises(GateError):
        S.run_l6(cfg, cache, l5_dict(t), mode="fast", members=bad)
    assert S.output_paths(cfg, "full").csv.exists()


# --------------------------------------------------------------------------------------
# Интеграционное ревью №1 и №4, ревью L6 F5: --force до кэша реплик, атомарная запись, время
# --------------------------------------------------------------------------------------
def test_run_l6_force_reaches_the_replicate_cache(env, monkeypatch):
    # Arrange
    cfg, cache = env
    t = latent_table()
    m = members_frame(t)
    ids = list(m.player_id)
    seen = []
    monkeypatch.setattr(R, "bootstrap_match_ids", lambda cfg_, cache_: list(MATCHES))
    monkeypatch.setattr(R, "make_replicate",
                        lambda cfg_, cache_, force=False: seen.append(force) or _covering_replicate(ids))
    monkeypatch.setattr(R, "executor_factory", lambda cfg_, cache_=None: None)

    # Act
    S.run_l6(cfg, cache, l5_dict(t), members=m, force=True)
    S.run_l6(cfg, cache, l5_dict(t), members=m)

    # Assert
    assert seen == [True, False]


def test_outputs_are_written_atomically_and_meta_records_csv_sha256(env, monkeypatch):
    # Arrange
    import hashlib
    cfg, cache = env
    t = latent_table()
    written = []
    real = S.atomic_write
    monkeypatch.setattr(S, "atomic_write", lambda path, write: written.append(path.name) or real(path, write))

    # Act
    res = S.run_l6(cfg, cache, l5_dict(t), members=members_frame(t))

    # Assert
    paths = S.output_paths(cfg, "fast")
    assert {paths.csv.name, paths.detail.name} <= set(written)
    meta = json.loads(paths.meta.read_text())
    assert meta["csv_sha256"] == hashlib.sha256(paths.csv.read_bytes()).hexdigest()
    assert meta["detail_sha256"] == hashlib.sha256(paths.detail.read_bytes()).hexdigest()
    assert res["reused"] is False


def test_truncated_csv_with_matching_provenance_is_not_reused(env):
    # Arrange: обрыв записи (SIGKILL/питание) оставил обрезанный CSV при прежней meta
    cfg, cache = env
    t = latent_table()
    m = members_frame(t)
    first = S.run_l6(cfg, cache, l5_dict(t), members=m)
    paths = S.output_paths(cfg, "fast")
    text = paths.csv.read_text()
    paths.csv.write_text(text[: len(text) // 2])

    # Act
    again = S.run_l6(cfg, cache, l5_dict(t), members=m)

    # Assert
    assert again["reused"] is False
    assert paths.csv.read_text() == text
    assert len(pd.read_csv(paths.csv)) == len(first["ratings"])


def test_full_bootstrap_time_budget_is_read_from_config(env):
    cfg, cache = env
    assert S.bootstrap_time_budget(cfg, "full") == float(cfg["bootstrap"].get("time_budget_s", 21600.0))
    assert S.bootstrap_time_budget(cfg, "fast") == S.BOOTSTRAP_TIME_BUDGET_S["fast"]
    assert S.bootstrap_time_budget(cfg.with_overrides({"bootstrap": {"time_budget_s": 43200}}), "full") == 43200.0
    with pytest.raises(ValueError):
        S.bootstrap_time_budget(cfg.with_overrides({"bootstrap": {"time_budget_s": -1}}), "full")

    # исчерпанный бюджет из конфигурации -> partial -> full не пишет сдаваемый файл
    t = latent_table(n=3)
    with pytest.raises(GateError):
        S.run_l6(cfg.with_overrides({"bootstrap": {"time_budget_s": 0}}), cache, l5_dict(t), mode="full",
                 members=members_frame(t, lineup_only=0), replicate=replicate_factory(0.01),
                 bootstrap_matches=MATCHES)
    assert not S.output_paths(cfg, "full").csv.exists()


def test_l6_report_carries_bootstrap_progress_summary(env):
    cfg, cache = env
    t = latent_table()
    S.run_l6(cfg, cache, l5_dict(t), members=members_frame(t), replicate=replicate_factory(0.02),
             bootstrap_matches=MATCHES)
    prog = _report(cfg)["model_bootstrap"]["progress"]
    b = cfg["bootstrap"]["debug_B"]
    assert prog["done"] == b and prog["total"] == b and prog["eta_s"] == 0
    assert prog["mean_replicate_s"] >= 0
