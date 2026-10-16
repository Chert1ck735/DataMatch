"""L6: состав и формат `outputs/test_ratings.csv`, meta и провенанс.

Латентные таблицы L5 здесь синтетические, ровно той схемы, что в docs/TIER_A_CONTRACT.md:
слой L6 не должен зависеть ни от чего, кроме опубликованного контракта. Отдельная проверка
на реальных данных считает фактическое число полевых игроков теста (пропускается, если
кэша L0 нет).
"""
import json

import numpy as np
import pandas as pd
import pytest

from datamatch import l6_uncertainty as U
from datamatch import stage_l6 as S
from datamatch.cache import Cache
from datamatch.config import load_config
from datamatch.gates import GateError

ROLES = ["C", "W", "D"]
MATCHES = [f"M{i:03d}" for i in range(30)]


# --------------------------------------------------------------------------------------
# Синтетические входы схемы контракта
# --------------------------------------------------------------------------------------
def latent_table(n=9, seed=0, zero_signal=False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = [f"P{i:05d}" for i in range(n)]
    w = np.zeros(n) if zero_signal else np.clip(rng.uniform(0.05, 0.95, n), 0, 1)
    tau2 = 0.04
    var_post = np.zeros(n) if zero_signal else tau2 * (1 - w)
    return pd.DataFrame({
        "player_id": ids, "role": [ROLES[i % 3] for i in range(n)],
        "games": rng.integers(1, 9, n), "toi_min": rng.uniform(5, 300, n).round(2),
        "rating": np.zeros(n) if zero_signal else rng.normal(0, 0.1, n),
        "var_post": var_post, "w": w, "definition": "O1", "fallback": zero_signal,
        "flag": "сигнал не обнаружен" if zero_signal else "",
    })


# Отчёт L5 с явно пройденными гейтами O1: без гейтов L6 выбирает O0 (строгая проверка).
PASSING_REPORT = {"gates_O1": {"signal_present": True}}


def l5_dict(table: pd.DataFrame, report=None, with_o0=True) -> dict:
    prior = {r: {"O1": {"mean": 0.01 * k, "var": 0.04}, "O0": {"mean": 0.0, "var": 0.02}}
             for k, r in enumerate(ROLES)}
    o0 = table.assign(definition="O0", var_post=table.var_post * 2)
    out = {"O1": {"train": table, "test": table}, "prior": prior,
           "report": dict(PASSING_REPORT) if report is None else report}
    if with_o0:
        out["O0"] = {"train": o0, "test": o0}
    return out


def members_frame(table: pd.DataFrame, lineup_only=2) -> pd.DataFrame:
    """Все полевые игроки теста: из таблицы L5 плюс заявленные без смен."""
    rows = pd.DataFrame({
        "player_id": list(table.player_id) + [f"X{i:05d}" for i in range(lineup_only)],
        "role": list(table.role) + [ROLES[i % 3] for i in range(lineup_only)],
        "toi_min": list(table.toi_min) + [0.0] * lineup_only,
        "games": list(table.games) + [0] * lineup_only,
    })
    rows["team_id"] = "T113"
    return rows.sort_values("player_id").reset_index(drop=True)


@pytest.fixture
def env(tmp_path):
    cfg = load_config().with_overrides({"paths": {
        "outputs": str(tmp_path / "outputs"), "artifacts": str(tmp_path / "artifacts"),
        "cache": str(tmp_path / "cache"), "logs": str(tmp_path / "logs")}})
    return cfg, Cache(cfg, {})


def replicate_factory(scale=0.05):
    def replicate(matches, index, seed):
        rng = np.random.default_rng(seed)
        ids = [f"P{i:05d}" for i in range(9)] + [f"X{i:05d}" for i in range(2)]
        return pd.Series(rng.normal(0.0, scale, len(ids)), index=ids)
    return replicate


# --------------------------------------------------------------------------------------
# Формат и состав файла
# --------------------------------------------------------------------------------------
def test_file_has_exactly_contract_columns_and_all_field_players(env):
    cfg, cache = env
    t = latent_table()
    m = members_frame(t)
    res = S.run_l6(cfg, cache, l5_dict(t), mode="fast", members=m)
    path = S.output_paths(cfg, "fast").csv
    raw = path.read_text().splitlines()
    assert raw[0] == "player_id,rating,lo95,hi95,reliability,toi_min,games,role"
    assert len(raw) == len(m) + 1
    got = pd.read_csv(path, dtype={"player_id": str})
    assert list(got.columns) == S.OUTPUT_COLUMNS
    assert sorted(got.player_id) == sorted(m.player_id)
    assert got.player_id.is_monotonic_increasing
    assert set(got.role) <= set(ROLES)
    assert res["meta"]["rows"]["n_rows"] == len(m)


def test_lineup_only_players_come_from_prior_with_zero_exposure_and_low(env):
    cfg, cache = env
    t = latent_table()
    m = members_frame(t, lineup_only=3)
    res = S.run_l6(cfg, cache, l5_dict(t), members=m)
    d = res["detail"].set_index("player_id")
    only = d.loc[[f"X{i:05d}" for i in range(3)]]
    assert (only.toi_min == 0).all() and (only.games == 0).all()
    assert (only.reliability == "LOW").all() and (only.w == 0).all()
    # rating = 1ᵀ(μ_r − m_r) из prior, без принудительного нуля; интервал из 1ᵀT_r1
    prior = l5_dict(t)["prior"]
    for pid, row in only.iterrows():
        exp = prior[row.role]["O1"]
        assert row.rating == pytest.approx(exp["mean"])
        assert row.hi95 - row.rating == pytest.approx(U.Z95 * np.sqrt(exp["var"]))
    assert res["meta"]["rows"]["n_prior_lineup_only"] == 3
    assert res["meta"]["rows"]["n_from_l5"] == len(t)


def test_player_with_shifts_missing_from_l5_keeps_exposure_and_is_flagged(env):
    cfg, cache = env
    t = latent_table()
    m = members_frame(t, lineup_only=0)
    m = pd.concat([m, pd.DataFrame([{"player_id": "Z00001", "role": "D", "toi_min": 42.0,
                                     "games": 3, "team_id": "T001"}])], ignore_index=True)
    res = S.run_l6(cfg, cache, l5_dict(t), members=m)
    row = res["detail"].set_index("player_id").loc["Z00001"]
    assert row.source == "prior_missing_in_l5" and row.toi_min == 42.0 and row.games == 3
    assert res["meta"]["rows"]["n_prior_missing_in_l5"] == 1
    assert any("prior" in s for s in res["report"]["limitations"])


def test_l5_rows_outside_the_test_list_are_not_written(env):
    cfg, cache = env
    t = latent_table()
    m = members_frame(t, lineup_only=0).iloc[:-2].reset_index(drop=True)
    res = S.run_l6(cfg, cache, l5_dict(t), members=m)
    assert len(res["ratings"]) == len(m)
    assert res["meta"]["rows"]["n_l5_rows_not_in_test_list"] == 2


def test_reliability_follows_config_thresholds(env):
    cfg, cache = env
    t = latent_table(n=6)
    t["w"] = [0.95, 0.7, 0.69, 0.4, 0.39, 0.0]
    res = S.run_l6(cfg, cache, l5_dict(t), members=members_frame(t, lineup_only=0))
    d = res["detail"].set_index("player_id").reliability
    assert list(d.loc[t.player_id]) == ["HIGH", "HIGH", "MEDIUM", "MEDIUM", "LOW", "LOW"]
    assert res["meta"]["reliability"]["thresholds"] == {"HIGH": 0.7, "MEDIUM": 0.4}


def test_interval_is_posterior_only_when_bootstrap_not_run(env):
    cfg, cache = env
    t = latent_table()
    res = S.run_l6(cfg, cache, l5_dict(t), members=members_frame(t, lineup_only=0))
    d = res["detail"].set_index("player_id")
    assert res["meta"]["model_bootstrap"]["status"] == "not_run"
    assert res["meta"]["model_bootstrap"]["reason"]
    assert res["meta"]["interval"]["var_model_included"] is False
    for pid, row in t.set_index("player_id").iterrows():
        assert d.loc[pid, "v_tot"] == pytest.approx(row.var_post)
        assert d.loc[pid, "hi95"] - d.loc[pid, "rating"] == pytest.approx(U.Z95 * np.sqrt(row.var_post))
    assert any("Var_m" in s for s in res["report"]["limitations"])
    assert res["report"]["status"] == "partial"


def test_bootstrap_adds_model_spread_and_widens_interval(env):
    cfg, cache = env
    t = latent_table()
    m = members_frame(t)
    narrow = S.run_l6(cfg, cache, l5_dict(t), members=m)
    wide = S.run_l6(cfg, cache, l5_dict(t), members=m, force=True,
                    replicate=replicate_factory(0.05), bootstrap_matches=MATCHES)
    assert wide["meta"]["model_bootstrap"]["status"] == "done"
    assert wide["meta"]["B"] == cfg["bootstrap"]["debug_B"]
    assert wide["meta"]["B_done"] == cfg["bootstrap"]["debug_B"]
    assert wide["meta"]["interval"]["var_model_included"] is True
    a = narrow["detail"].set_index("player_id")
    b = wide["detail"].set_index("player_id")
    assert (b.v_tot >= a.v_tot - 1e-12).all() and (b.v_tot > a.v_tot).any()
    assert np.isfinite(b.mc_se_bound).all()
    assert wide["report"]["status"] == "ok"


def test_full_mode_uses_final_B(env):
    cfg, cache = env
    t = latent_table(n=3)
    res = S.run_l6(cfg, cache, l5_dict(t), mode="full", members=members_frame(t, lineup_only=0),
                   replicate=replicate_factory(0.01), bootstrap_matches=MATCHES)
    assert res["meta"]["mode"] == "full"
    assert res["meta"]["B"] == cfg["bootstrap"]["final_B"] >= 200
    assert res["meta"]["model_bootstrap"]["B_done"] == cfg["bootstrap"]["final_B"]
    assert res["report"]["limitations"] == [] or all("B =" not in s for s in res["report"]["limitations"])


def test_unknown_mode_rejected(env):
    cfg, cache = env
    t = latent_table(n=3)
    with pytest.raises(ValueError):
        S.run_l6(cfg, cache, l5_dict(t), mode="turbo", members=members_frame(t, lineup_only=0))


def test_zero_signal_table_gives_degenerate_intervals_that_fail_the_hard_gate(env):
    """Нулевая ширина интервала — заявление о полной уверенности: гейт lo95 < hi95 (аудит №8)."""
    cfg, cache = env
    t = latent_table(n=6, zero_signal=True)
    with pytest.raises(GateError):
        S.run_l6(cfg, cache, l5_dict(t), members=members_frame(t, lineup_only=1))
    rep = json.loads((cfg.paths.artifacts / S.L6_REPORT).read_text())
    assert "interval_nondegenerate" in rep["failed_gates"]
    assert not S.output_paths(cfg, "fast").csv.exists()


# --------------------------------------------------------------------------------------
# Выбор определения итога
# --------------------------------------------------------------------------------------
def test_definition_prefers_o1_only_when_gates_explicitly_pass():
    t = latent_table(n=3)
    assert S.choose_definition(l5_dict(t))[0] == "O1"
    assert S.choose_definition(l5_dict(t, report={"gates": {"cholesky": True}}))[0] == "O1"
    assert S.choose_definition(l5_dict(t, report={}))[0] == "O0"          # гейтов нет — не пройдены


def test_definition_falls_back_to_o0_on_failed_l5_gate_or_missing_o1():
    t = latent_table(n=3)
    d, why = S.choose_definition(l5_dict(t, report={"gates_O1": {"oneT1_positive": False}}))
    assert d == "O0" and "гейты" in why
    l5 = l5_dict(t)
    l5["O1"] = {"train": t, "test": pd.DataFrame()}
    assert S.choose_definition(l5)[0] == "O0"


def test_definition_respects_explicit_l5_choice_and_errors_without_tables():
    t = latent_table(n=3)
    assert S.choose_definition(l5_dict(t, report={"final_definition": "O0"}))[0] == "O0"
    with pytest.raises(S.L6InputsNotReady):
        S.choose_definition({"report": {}})


def test_o0_definition_is_recorded_in_meta_with_its_limitation(env):
    cfg, cache = env
    t = latent_table(n=4)
    l5 = l5_dict(t, report={"final_definition": "O0"})
    res = S.run_l6(cfg, cache, l5, members=members_frame(t, lineup_only=1))
    assert res["meta"]["rating_definition"] == "O0"
    assert any("O0" in s for s in res["report"]["limitations"])
    d = res["detail"].set_index("player_id")
    assert d.loc[t.player_id[0], "var_post"] == pytest.approx(t.var_post.iloc[0] * 2)
    # prior для O0 берётся из своей ветки prior[role]["O0"]
    assert d.loc["X00000", "rating"] == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# Контракт входа и гейты
# --------------------------------------------------------------------------------------
def test_missing_required_l5_column_is_rejected(env):
    cfg, cache = env
    full = latent_table(n=3)
    t = full.drop(columns=["var_post"])
    l5 = {"O1": {"train": t, "test": t}, "prior": l5_dict(full)["prior"], "report": {}}
    with pytest.raises(S.L6InputsNotReady):
        S.run_l6(cfg, cache, l5, members=members_frame(t, lineup_only=0))


def test_duplicate_player_in_l5_table_is_rejected(env):
    cfg, cache = env
    t = latent_table(n=3)
    t = pd.concat([t, t.iloc[[0]]], ignore_index=True)
    with pytest.raises(S.L6InputsNotReady):
        S.run_l6(cfg, cache, l5_dict(t), members=members_frame(t, lineup_only=0))


def test_goalie_in_member_list_fails_the_structural_gate(env):
    cfg, cache = env
    t = latent_table(n=3)
    m = members_frame(t, lineup_only=0)
    m.loc[0, "role"] = "G"
    with pytest.raises(GateError):
        S.run_l6(cfg, cache, l5_dict(t), members=m)
    assert not S.output_paths(cfg, "fast").csv.exists()
    assert json.loads((cfg.paths.artifacts / "l6_report.json").read_text())["status"] == "failed"


def test_structural_gates_catch_broken_coverage_and_order():
    t = latent_table(n=3)
    m = members_frame(t, lineup_only=0)
    boot = U.BootstrapResult("not_run", "", 30, 0)
    detail, _ = S.build_rows(m, t, l5_dict(t)["prior"], "O1", boot, 0.7, 0.4)
    expected = set(m.player_id)
    good = S.structural_gates(detail, m, S.file_frame(detail), expected, boot)
    assert all(good.values())
    broken = detail.copy()
    broken.loc[0, "lo95"] = broken.loc[0, "hi95"] + 1.0
    assert S.structural_gates(broken, m, S.file_frame(broken), expected, boot)["interval_ordered"] is False
    assert S.structural_gates(detail.iloc[1:], m, S.file_frame(detail.iloc[1:]), expected, boot)[
        "covers_all_field_players"] is False


# --------------------------------------------------------------------------------------
# meta, провенанс и повторный запуск
# --------------------------------------------------------------------------------------
def test_meta_records_definition_mode_bootstrap_rows_and_provenance(env):
    cfg, cache = env
    t = latent_table()
    res = S.run_l6(cfg, cache, l5_dict(t), mode="fast", members=members_frame(t))
    meta = json.loads(S.output_paths(cfg, "fast").meta.read_text())
    for k in ("rating_definition", "mode", "B", "model_bootstrap", "rows", "columns",
              "reliability", "interval", "provenance", "field_players"):
        assert k in meta, k
    assert meta["rating_definition"] == "O1" and meta["mode"] == "fast"
    assert meta["interval"]["predictive_interval_in_file"] is False
    assert set(meta["provenance"]) >= {"stage", "data", "config", "code", "l5_table", "members", "B"}
    assert meta["rows"]["n_rows"] == len(res["ratings"])


def test_rerun_reuses_file_until_inputs_or_force_change(env):
    cfg, cache = env
    t = latent_table()
    m = members_frame(t)
    S.run_l6(cfg, cache, l5_dict(t), members=m)
    again = S.run_l6(cfg, cache, l5_dict(t), members=m)
    assert again["reused"] is True
    assert S.run_l6(cfg, cache, l5_dict(t), members=m, force=True)["reused"] is False
    changed = t.copy()
    changed.loc[0, "rating"] += 0.5
    assert S.run_l6(cfg, cache, l5_dict(changed), members=m)["reused"] is False
    with pytest.raises(GateError):                      # full без реплики не пишет файл
        S.run_l6(cfg, cache, l5_dict(changed), members=m, mode="full")


def test_predictive_interval_reported_only_with_published_moments(env):
    cfg, cache = env
    t = latent_table()
    m = members_frame(t, lineup_only=0)
    off = S.run_l6(cfg, cache, l5_dict(t), members=m)
    assert off["report"]["predictive_interval"]["status"] == "not_run"
    on = S.run_l6(cfg, cache, l5_dict(t), members=m, force=True,
                  noise_moments={"sigma2": 1.5, "omega2": 0.02, "source": "test"})
    pr = on["report"]["predictive_interval"]
    assert pr["status"] == "ok" and pr["in_file"] is False
    widths = [pr["mean_halfwidth_by_delta2"][k] for k in pr["mean_halfwidth_by_delta2"]]
    assert widths == sorted(widths) and len(widths) == len(pr["drift_grid"])


def test_l5_report_shape_of_stage_l5_is_understood(env):
    """Отчёт L5 в том виде, в котором его публикует stage_l5: итог, гейты и моменты."""
    cfg, cache = env
    t = latent_table(n=5)
    report = {"stage": "l5", "status": "ok", "rating_definition": "O1", "o1_usable": True,
              "gates_l5": {"split_half": True}, "gates_hard": {"roles_known_test": True},
              "moments": {"univariate": {"C": {"tau2": 0.04, "sigma2": 1.5, "omega2": 0.02},
                                         "D": {"tau2": 0.03, "sigma2": 1.4, "omega2": 0.01}},
                          "sigma2": [0.1, 0.2], "omega2": [0.01, 0.02],
                          "Sigma_sigma": [[0.1, 0.0], [0.0, 0.2]],
                          "Sigma_omega": [[0.01, 0.0], [0.0, 0.02]]}}
    res = S.run_l6(cfg, cache, l5_dict(t, report=report), members=members_frame(t, lineup_only=1))
    assert res["meta"]["rating_definition"] == "O1"
    # B5 (08.10.2026): моменты C и D различаются, шум по амплуа не помечен — общей пары нет, запасная
    # роль «C» не берётся; интервал не строится с понятной причиной
    pr = res["report"]["predictive_interval"]
    assert pr["status"] == "not_run" and "различаются по амплуа" in pr["reason"]
    same = dict(report, moments=dict(report["moments"], univariate={
        r: {"tau2": 0.04, "sigma2": 1.5, "omega2": 0.02} for r in ("C", "D")}))
    res2 = S.run_l6(cfg, cache, l5_dict(t, report=same), members=members_frame(t, lineup_only=1), force=True)
    mom = res2["report"]["predictive_interval"]["moments"]
    assert (mom["sigma2"], mom["omega2"]) == (1.5, 0.02)        # одномерная модель суммы, не диагонали Σσ
    degraded = dict(report, status="degraded", rating_definition="O0", o1_usable=False,
                    gates_l5={"split_half": False})
    out = S.run_l6(cfg, cache, l5_dict(t, report=degraded), members=members_frame(t, lineup_only=1),
                   force=True)
    assert out["meta"]["rating_definition"] == "O0"


def test_noise_moments_fall_back_to_sum_of_moment_matrices():
    mom = S.overall_noise_moments({"moments": {"Sigma_sigma": [[0.1, 0.05], [0.05, 0.2]],
                                               "Sigma_omega": [[0.01, 0.0], [0.0, 0.02]]}})
    assert mom["sigma2"] == pytest.approx(0.4) and mom["omega2"] == pytest.approx(0.03)
    assert S.overall_noise_moments({"moments": {"sigma2": [0.1, 0.2]}}) is None
    assert S.overall_noise_moments({}) is None


def test_failed_l5_gate_without_explicit_definition_falls_back_to_o0():
    t = latent_table(n=3)
    for key in ("gates_l5", "gates_hard", "gates", "gates_O1"):
        d, _ = S.choose_definition(l5_dict(t, report={key: {"x": False}}))
        assert d == "O0", key


def test_noise_moments_are_read_from_l5_report_when_published(env):
    cfg, cache = env
    t = latent_table(n=4)
    l5 = l5_dict(t, report={"overall_moments": {"sigma2": 1.0, "omega2": 0.01}})
    res = S.run_l6(cfg, cache, l5, members=members_frame(t, lineup_only=0))
    assert res["report"]["predictive_interval"]["moments"]["source"] == "report.overall_moments"


def test_self_consistency_check_is_reported_but_labelled(env):
    cfg, cache = env
    t = latent_table(n=200, seed=4)
    res = S.run_l6(cfg, cache, l5_dict(t), members=members_frame(t, lineup_only=0))
    cov = res["report"]["coverage_simulation"]["self_consistency"]
    assert cov["status"] == "ok" and cov["kind"] == "not_a_coverage_test"
    assert cov["coverage"] == pytest.approx(0.95, abs=0.02)


def test_value_model_plan_reports_unavailable_replicate(env):
    cfg, cache = env
    plan = S.bootstrap_plan(cfg, cache, "O1", None, None)
    assert plan.replicate is None and plan.reason
    assert plan.hook.startswith("unavailable")


# --------------------------------------------------------------------------------------
# Фактическое число полевых игроков теста (реальные данные)
# --------------------------------------------------------------------------------------
def test_actual_test_field_player_count_from_data():
    cfg = load_config()
    cache = Cache(cfg, __import__("datamatch.io", fromlist=["io"]).data_version(cfg))
    try:
        members, rep = S.test_field_players(cfg, cache)
    except S.L6InputsNotReady as ex:
        pytest.skip(f"кэш L0 не готов: {ex}")
    assert set(members.role) <= set(ROLES)                       # вратарей нет
    assert members.player_id.is_unique
    assert rep["n_field"] == len(members) == 568
    assert rep["n_with_shifts"] + rep["n_lineup_only"] == rep["n_field"]
    assert rep["n_goalies_excluded"] > 0 and rep["n_role_unresolved"] == 0
    assert len(rep["expected_field_ids"]) == rep["n_field"]
    assert (members.toi_min[members.games == 0] == 0).all()
