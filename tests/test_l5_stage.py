"""L5: контракт `stage_l5.run_l5` на синтетической таблице «игрок × матч» схемы L3.

Таблица строится ровно по `docs/TIER_A_CONTRACT.md` (раздел L3 → L5), поэтому проверки не
зависят от готовности L3 на реальных данных. Проверяется: форма выхода, обучение только на
train, знаменатель ставок на 60, согласованность Overall с суммой компонентов, prior для
игроков без смен, источник μ_r для test, независимость O0 от компонентов L2/L3, гейты и кэш.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from datamatch import l5_o0, stage_l5
from datamatch.cache import Cache
from datamatch.config import load_config
from datamatch.gates import GateError
from datamatch.l5_shrink import ROLES

ROLE_CYCLE = ["C", "W", "D"]
POSITION = {"C": "C", "W": "LW", "D": "D"}
TOI_CHOICES = (600.0, 1200.0, 2400.0)
GAMES_CHOICES = (4, 10, 24)


def cfg_for(tmp_path):
    return load_config().with_overrides({"paths": {
        "cache": str(tmp_path / "cache"), "artifacts": str(tmp_path / "artifacts"),
        "outputs": str(tmp_path / "outputs"), "logs": str(tmp_path / "logs")}})


def synth_pm(rng, split: str, n_players: int = 150, match_offset: int = 0,
             tau: float = 0.22, sigma: float = 2.0, omega: float = 0.1,
             khl_share: float = 0.6, zero_rate_players: tuple = (), zero_rate_rows: float = 0.0):
    """Синтетическая `player_match` схемы контракта L3 → L5.

    Компоненты и on-ice xG порождены моделью L5 в её единицах: ставка на 60 равна
    Σ / toi_rate_s × 3600, поэтому вклад за матч пропорционален TOI в часах.
    """
    cols = stage_l5.COMP_COLS
    k = len(cols)
    rows = []
    mid = match_offset
    for i in range(n_players):
        pid = f"P{i:05d}"
        role = ROLE_CYCLE[i % 3]
        th = rng.normal(0.0, tau, size=k)
        th_o0 = float(rng.normal(0.0, 0.5))
        t_s = float(rng.choice(TOI_CHOICES))
        games = int(rng.choice(GAMES_CHOICES))
        khl = i < int(n_players * khl_share)
        for _ in range(games):
            mid += 1
            t_h = t_s / 3600.0
            noise = rng.normal(0.0, sigma, size=k) / 60.0 * np.sqrt(t_s / 60.0)
            per_match = rng.normal(0.0, omega, size=k) * t_h
            credit = th * t_h + noise + per_match
            rate = 0.0 if (pid in zero_rate_players or rng.random() < zero_rate_rows) else t_s
            rate5 = rate * 0.7
            xgd = th_o0 * rate5 / 3600.0 + rng.normal(0.0, 0.35) * np.sqrt(rate5 / 3600.0)
            base = 0.6 * rate5 / 3600.0
            row = {
                "match_id": f"{split[:2].upper()}M{mid:06d}", "player_id": pid,
                "team_id": f"T{i % 12:03d}", "split": split,
                "position": POSITION[role], "role": role,
                "toi_s": t_s * 2.0, "toi_5v5_s": t_s * 0.7, "toi_pp_s": t_s * 0.2,
                "toi_sh_s": t_s * 0.1, "toi_rate_s": rate, "toi_rate_5v5_s": rate5,
                "toi_rate_pp_s": rate * 0.2, "toi_rate_sh_s": rate * 0.1,
                "goal": 0.0, "ast": 0.0, "points": 0.0, "pm": 0.0,
                "ixg": 0.1, "igoals": 0.0, "ishots": 2.0,
                "xgf_5v5": max(base + xgd / 2.0, 0.0), "xga_5v5": max(base - xgd / 2.0, 0.0),
                "component": 1 if khl else 2, "is_khl_component": bool(khl),
                "n_actions": 5, "rated": rate > 0,
            }
            row.update(dict(zip(cols, credit)))
            row["iav_sum"] = float(np.sum(credit))
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def l3_synth():
    rng = np.random.default_rng(20261005)
    train = synth_pm(rng, "train", n_players=180, match_offset=0)
    test = synth_pm(rng, "test", n_players=90, match_offset=500_000,
                    zero_rate_players=("P00000",))
    return {"train": {"journal": pd.DataFrame(), "player_match": train},
            "test": {"journal": pd.DataFrame(), "player_match": test},
            "report": {"synthetic": True}}


def run(cfg, l3, force=False):
    return stage_l5.run_l5(cfg, Cache(cfg, {"synthetic": 1}), l3, force=force)


# ---------------------------------------------------------------------------------------
# Форма выхода по контракту

def test_run_l5_output_matches_contract(tmp_path, l3_synth):
    out = run(cfg_for(tmp_path), l3_synth)
    assert set(out) == {"O1", "O0", "prior", "report"}
    assert set(out["O1"]) == {"train", "test"} and set(out["O0"]) == {"train", "test"}
    required = {"player_id", "role", "games", "toi_min", "rating", "var_post", "w",
                "definition", "fallback"}
    for name in ("O1", "O0"):
        for split in ("train", "test"):
            df = out[name][split]
            assert required <= set(df.columns), (name, split, required - set(df.columns))
            assert df.player_id.is_unique
            assert (df.definition == name).all()
            assert df.role.isin(ROLES).all()
    o1 = out["O1"]["test"]
    assert [c for c in o1.columns if c.startswith("comp_")] == [
        f"comp_{c}" for c in stage_l5.BASE_CATEGORIES]
    assert not [c for c in out["O0"]["test"].columns if c.startswith("comp_")]
    assert len(o1) == l3_synth["test"]["player_match"].player_id.nunique()
    for r in ROLES:
        assert set(out["prior"][r]) == {"O1", "O0"}
        for d in ("O1", "O0"):
            assert set(out["prior"][r][d]) == {"mean", "var"}
            assert np.isfinite(out["prior"][r][d]["mean"])
            assert out["prior"][r][d]["var"] >= 0


def test_report_carries_moments_fallbacks_eps_log_and_gates(tmp_path, l3_synth):
    rep = run(cfg_for(tmp_path), l3_synth)["report"]
    m = rep["moments"]
    for r in ROLES:
        assert len(m["T_r"][r]) == len(stage_l5.COMP_COLS)
        assert np.isfinite(m["oneT1"][r])
        assert set(m["univariate"][r]) >= {"tau2", "sigma2", "omega2"}
    assert len(m["sigma2"]) == len(stage_l5.COMP_COLS)
    assert len(m["omega2"]) == len(stage_l5.COMP_COLS)
    assert m["psd_projection"]["T_r"]["C"]["floor"] > 0
    assert "eps_counts" in rep["eps_log"]["O1_test"]
    assert "univariate_rule" in rep["fallback"]
    assert rep["gates_l5"] and rep["gates_hard"]
    # L6 выбирает итог по rating_definition, а при его отсутствии — по gates_O1
    assert rep["gates_O1"] == rep["gates_l5"]
    assert rep["rating_definition"] in ("O1", "O0")
    assert rep["replacement"]["O1"]["R"] and rep["replacement"]["applied_from"].startswith("train")
    assert set(rep["not_run"]) >= {"model_bootstrap", "o2_rapm", "war_g_win"}
    assert all(str(v).startswith("not_run") for v in rep["not_run"].values())
    assert rep["exposure_filter"]["train"]["rows_dropped"] >= 0
    art = tmp_path / "artifacts" / "shrink_report.json"
    assert art.exists(), "отчёт L5 должен сохраняться с provenance"
    saved = json.loads(art.read_text())
    assert saved["provenance"]["stage"] == "l5"
    assert saved["rating_definition"] in ("O0", "O1")


# ---------------------------------------------------------------------------------------
# Обучение только на train

def test_fit_ignores_test_entirely(tmp_path, l3_synth):
    """Изменение тестовых компонентов не должно менять ни один обучаемый параметр."""
    cfg = cfg_for(tmp_path)
    base = run(cfg, l3_synth)
    assert base["report"]["gates_hard"]["no_test_matches_in_fit"] is True
    pm_te = l3_synth["test"]["player_match"].copy()
    pm_te[stage_l5.COMP_COLS] = pm_te[stage_l5.COMP_COLS] * 7.0 + 3.0
    shifted = run(cfg, {"train": l3_synth["train"],
                        "test": {"journal": pd.DataFrame(), "player_match": pm_te},
                        "report": {}})
    a, b = base["report"]["moments"], shifted["report"]["moments"]
    for r in ROLES:
        assert np.allclose(a["T_r"][r], b["T_r"][r])
        assert a["univariate"][r] == b["univariate"][r]
    assert np.allclose(a["Sigma_sigma"], b["Sigma_sigma"])
    assert np.allclose(a["Sigma_omega"], b["Sigma_omega"])
    assert base["report"]["prior_mu"]["test"] == shifted["report"]["prior_mu"]["test"]
    assert np.allclose(base["O1"]["train"].rating, shifted["O1"]["train"].rating)


def test_mu_for_test_comes_from_khl_component(tmp_path, l3_synth):
    cfg = cfg_for(tmp_path)
    rep = run(cfg, l3_synth)["report"]
    assert rep["prior_mu"]["test_source"] == "khl_component"
    assert rep["prior_mu"]["test_source_reason"] == ""
    assert rep["prior_mu"]["test"] != rep["prior_mu"]["train"]


def test_missing_khl_component_falls_back_to_full_train_with_reason(tmp_path, l3_synth):
    pm_tr = l3_synth["train"]["player_match"].drop(columns=["is_khl_component"])
    rep = run(cfg_for(tmp_path), {"train": {"journal": pd.DataFrame(), "player_match": pm_tr},
                                  "test": l3_synth["test"], "report": {}})["report"]
    assert rep["prior_mu"]["test_source"] == "train_all"
    assert "is_khl_component" in rep["prior_mu"]["test_source_reason"]
    assert rep["prior_mu"]["test"] == rep["prior_mu"]["train"]


def test_empty_khl_component_falls_back_with_reason(tmp_path, l3_synth):
    pm_tr = l3_synth["train"]["player_match"].copy()
    pm_tr["is_khl_component"] = False
    rep = run(cfg_for(tmp_path), {"train": {"journal": pd.DataFrame(), "player_match": pm_tr},
                                  "test": l3_synth["test"], "report": {}})["report"]
    assert rep["prior_mu"]["test_source"] == "train_all"
    assert "КХЛ" in rep["prior_mu"]["test_source_reason"]


# ---------------------------------------------------------------------------------------
# Знаменатель ставок на 60

def test_per60_denominator_is_rate_toi_not_total_toi(tmp_path, l3_synth):
    """Удвоение `toi_s` при неизменном `toi_rate_s` не должно менять ни одной оценки."""
    cfg = cfg_for(tmp_path)
    base = run(cfg, l3_synth)
    changed = {}
    for split in ("train", "test"):
        pm = l3_synth[split]["player_match"].copy()
        pm["toi_s"] = pm.toi_s * 3.0 + 100.0
        pm["toi_5v5_s"] = pm.toi_5v5_s * 3.0
        changed[split] = {"journal": pd.DataFrame(), "player_match": pm}
    other = run(cfg, dict(changed, report={}))
    for split in ("train", "test"):
        for name in ("O1", "O0"):
            assert np.allclose(base[name][split].rating, other[name][split].rating)
            assert np.allclose(base[name][split].toi_min, other[name][split].toi_min)


def test_zero_rate_exposure_rows_are_dropped_from_both_sides(tmp_path):
    rng = np.random.default_rng(4)
    pm_tr = synth_pm(rng, "train", n_players=150, zero_rate_rows=0.25)
    pm_te = synth_pm(rng, "test", n_players=60, match_offset=900_000)
    out = run(cfg_for(tmp_path), {
        "train": {"journal": pd.DataFrame(), "player_match": pm_tr},
        "test": {"journal": pd.DataFrame(), "player_match": pm_te}, "report": {}})
    exp = out["report"]["exposure_filter"]["train"]
    assert exp["rows_dropped"] > 0
    assert exp["dropped_abs_credit"] > 0
    assert 0.0 < exp["dropped_abs_credit_share"] < 1.0
    # M_i считается по той же выборке: матчи без допущенной экспозиции в games не входят
    games = out["O1"]["train"].set_index("player_id").games
    kept = pm_tr[pm_tr.toi_rate_s > 0].groupby("player_id").match_id.nunique()
    assert games.reindex(kept.index).equals(kept.astype(int).rename("games"))


# ---------------------------------------------------------------------------------------
# Согласованность профиля и итога, prior

def test_components_sum_to_overall_in_output(tmp_path, l3_synth):
    out = run(cfg_for(tmp_path), l3_synth)
    for split in ("train", "test"):
        df = out["O1"][split]
        joint = df[~df.fallback & (df.method == "joint")]
        assert len(joint) > 0
        comp = joint[[f"comp_{c}" for c in stage_l5.BASE_CATEGORIES]].to_numpy()
        assert np.allclose(comp.sum(axis=1), joint.rating.to_numpy(), atol=1e-10)


def test_player_without_shifts_equals_prior(tmp_path, l3_synth):
    out = run(cfg_for(tmp_path), l3_synth)
    o1 = out["O1"]["test"].set_index("player_id")
    ghost = o1.loc["P00000"]
    assert ghost.toi_min == 0 and ghost.games == 0
    pr = out["prior"][ghost.role]["O1"]
    assert ghost.rating == pytest.approx(pr["mean"])
    assert ghost.var_post == pytest.approx(pr["var"])
    assert ghost.w == 0.0 and ghost.reliability == "LOW"
    o0 = out["O0"]["test"].set_index("player_id").loc["P00000"]
    assert o0.rating == pytest.approx(out["prior"][ghost.role]["O0"]["mean"])


def test_joint_shrinkage_contracts_dispersion_towards_prior(tmp_path, l3_synth):
    """Совместное стягивание сжимает разброс итогов относительно сырых ставок на 60.

    Скалярного свойства «θ̂ между μ_r и x» у совместной модели нет: вес T_r(T_r + S_i)^{-1} —
    матрица, она перемешивает компоненты, и сумма 1ᵀθ̂ может выйти за пределы [Σμ, Σx].
    Проверяемое свойство — сжатие разброса и ограниченность апостериорной дисперсии.
    """
    out = run(cfg_for(tmp_path), l3_synth)
    rep, df = out["report"], out["O1"]["train"]
    pm = l3_synth["train"]["player_match"]
    kept = pm[pm.toi_rate_s > 0]
    raw = (kept.groupby("player_id")[stage_l5.COMP_COLS].sum().sum(axis=1)
           / kept.groupby("player_id").toi_rate_s.sum() * 3600.0)
    d = df[(df.method == "joint") & df.player_id.isin(raw.index)].set_index("player_id")
    mu = np.array([float(np.sum(rep["prior_mu"]["train"][r])) for r in d.role])
    theta = d.theta.to_numpy()
    raw_v = raw.reindex(d.index).to_numpy()
    assert np.abs(theta - mu).std() < np.abs(raw_v - mu).std()
    one_t1 = np.array([rep["moments"]["oneT1"][r] for r in d.role])
    assert (d.var_post.to_numpy() <= one_t1 + 1e-12).all()


def test_univariate_o0_estimate_lies_between_prior_and_raw_rate(tmp_path, l3_synth):
    """Для одномерного O0 свойство θ̂ = μ + w(x − μ), w ∈ [0, 1], проверяется напрямую."""
    out = run(cfg_for(tmp_path), l3_synth)
    rep = out["report"]["o0"]["train"]
    df = out["O0"]["train"]
    pm = l3_synth["train"]["player_match"]
    kept = pm[pm.toi_rate_5v5_s > 0]
    raw = ((kept.groupby("player_id").xgf_5v5.sum() - kept.groupby("player_id").xga_5v5.sum())
           / kept.groupby("player_id").toi_rate_5v5_s.sum() * 3600.0)
    d = df[(df.method == "univariate") & df.player_id.isin(raw.index)].set_index("player_id")
    mu = np.array([rep["mu"][r] for r in d.role])
    raw_v = raw.reindex(d.index).to_numpy()
    theta = d.theta.to_numpy()
    assert ((theta >= np.minimum(mu, raw_v) - 1e-9)
            & (theta <= np.maximum(mu, raw_v) + 1e-9)).all()


# ---------------------------------------------------------------------------------------
# O0 как страховка

def test_o0_does_not_use_l2_l3_components(tmp_path, l3_synth):
    """O0 считается из L0/L1-величин: обнуление всех компонентов вклада его не меняет."""
    cfg = cfg_for(tmp_path)
    base = run(cfg, l3_synth)
    zeroed = {}
    for split in ("train", "test"):
        pm = l3_synth[split]["player_match"].copy()
        pm[stage_l5.COMP_COLS] = 0.0
        pm["iav_sum"] = 0.0
        zeroed[split] = {"journal": pd.DataFrame(), "player_match": pm}
    other = run(cfg, dict(zeroed, report={}))
    for split in ("train", "test"):
        assert np.allclose(base["O0"][split].rating, other["O0"][split].rating)
        assert np.allclose(base["O0"][split].var_post, other["O0"][split].var_post)


def test_o0_frame_uses_5v5_rate_exposure():
    rng = np.random.default_rng(8)
    pm = synth_pm(rng, "train", n_players=12, zero_rate_rows=0.3)
    f, info = l5_o0.o0_frame(pm)
    assert (f.toi_s > 0).all()
    assert info["rows_dropped_zero_5v5_toi"] == int((pm.toi_rate_5v5_s <= 0).sum())
    merged = pm[pm.toi_rate_5v5_s > 0].reset_index(drop=True)
    assert np.allclose(f[l5_o0.O0_COL].to_numpy(),
                       (merged.xgf_5v5 - merged.xga_5v5).to_numpy())
    assert np.allclose(f.toi_s.to_numpy(), merged.toi_rate_5v5_s.to_numpy())


def test_o0_requires_only_l0_l1_columns():
    rng = np.random.default_rng(9)
    pm = synth_pm(rng, "train", n_players=6)[l5_o0.O0_REQUIRED]
    f, _ = l5_o0.o0_frame(pm)
    assert len(f) > 0
    with pytest.raises(KeyError):
        l5_o0.o0_frame(pm.drop(columns=["xga_5v5"]))


# ---------------------------------------------------------------------------------------
# Гейты и уровень замены

def test_no_signal_degrades_definition_to_o0_without_raising(tmp_path):
    """Нет сигнала в компонентах -> O1 не допущен, итог по контракту O0, расчёт не падает."""
    rng = np.random.default_rng(14)
    pm_tr = synth_pm(rng, "train", n_players=120)
    pm_te = synth_pm(rng, "test", n_players=60, match_offset=700_000)
    for pm in (pm_tr, pm_te):
        pm[stage_l5.COMP_COLS] = 0.0
    out = run(cfg_for(tmp_path), {"train": {"journal": pd.DataFrame(), "player_match": pm_tr},
                                  "test": {"journal": pd.DataFrame(), "player_match": pm_te},
                                  "report": {}})
    rep = out["report"]
    assert rep["o1_usable"] is False
    assert rep["rating_definition"] == "O0"
    assert rep["status"] == "degraded"
    assert not all(rep["gates_l5"][f"signal_present_{r}"] for r in ROLES)
    assert np.isfinite(out["O1"]["test"].rating).all()
    assert np.isfinite(out["O0"]["test"].rating).all()


def test_missing_schema_column_raises_gate_error(tmp_path, l3_synth):
    pm = l3_synth["test"]["player_match"].drop(columns=["toi_rate_5v5_s"])
    with pytest.raises(GateError):
        run(cfg_for(tmp_path), {"train": l3_synth["train"],
                                "test": {"journal": pd.DataFrame(), "player_match": pm},
                                "report": {}})


def test_replacement_level_and_xgar(tmp_path, l3_synth):
    out = run(cfg_for(tmp_path), l3_synth)
    repl = out["report"]["replacement"]["O1"]
    assert all(repl["R"][r] is not None for r in ROLES)
    assert all(v is not None for v in repl["max_spread"].values())
    df = out["O1"]["test"]
    played = df[df.toi_min > 0]
    assert np.isfinite(played.xgar).all()
    rv = np.array([repl["R"][r] for r in played.role])
    above = played.theta.to_numpy() > rv
    assert (played.xgar.to_numpy()[above] > 0).all()
    assert (played.xgar.to_numpy()[~above] <= 0).all()
    # xGAR масштабируется личной экспозицией
    assert played.xgar.abs().idxmax() in played.index


# ---------------------------------------------------------------------------------------
# Кэш

def test_cache_roundtrip_and_force(tmp_path, l3_synth, monkeypatch):
    cfg = cfg_for(tmp_path)
    first = run(cfg, l3_synth)
    calls = []
    real_fit = stage_l5.JointEB.fit

    def spy(self, *a, **kw):
        calls.append(1)
        return real_fit(self, *a, **kw)

    monkeypatch.setattr(stage_l5.JointEB, "fit", spy)
    again = run(cfg, l3_synth)
    assert not calls, "повторный вызов при готовом кэше не должен заново оценивать моменты"
    pd.testing.assert_frame_equal(first["O1"]["test"], again["O1"]["test"])
    assert first["prior"] == again["prior"]
    run(cfg, l3_synth, force=True)
    assert calls, "--force обязан пересчитать моменты"
