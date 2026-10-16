"""L7: «наш рейтинг против базовых» (`rating_validation`) на синтетике с известным талантом.

Популяция «талант»: латентный θ_i (xG/60) порождает компоненты кредита `c_*` и on-ice xGD при
5 на 5, а очки — чистый шум. Тогда O1, подогнанный на половине A, обязан предсказывать xGD на
половине B лучше сырых очков/60 (интервал разности выше нуля). Популяция «шум»: θ_i = 0 у всех —
интервалы корреляций обязаны накрывать ноль. Отдельно: половины не пересекаются по матчам,
рейтинги видят только матчи своей половины, изменение будущей половины не меняет рейтинг.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datamatch import rating_validation as rv
from datamatch import stage_l5
from datamatch import validation as va
from datamatch.cache import Cache
from datamatch.config import load_config

ROLES = ("C", "W", "D")
COMP_SHARE = {"c_shot": 0.35, "c_loss": 0.15, "c_recovery": 0.2, "c_faceoff": 0.05, "c_creation": 0.25}
COMP_MU = {"c_shot": 0.30, "c_loss": -0.40, "c_recovery": 0.25, "c_faceoff": 0.02, "c_creation": 0.15}
COMP_SIGMA = 0.45          # шум компонента, xG на √час
XGF_MU = 2.6               # on-ice xGF/60 при 5 на 5
XG_SIGMA = 1.2             # шум on-ice xG, на √час
POINTS_RATE = 2.0          # очки/60 — одинаковы у всех (шум)
GF_SHARE = 0.1             # голы ≈ доля xG
N_BOOT_TEST = 200
MIN_TOI_TEST_S = 60.0 * 60.0
SEED_DATA = 7
SHRINK = dict(load_config()["shrink"])
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validation_report.py"


def synth_pm(talent_sd: float, n_per_role: int = 60, n_matches: int = 60, p_play: float = 0.75,
             seed: int = SEED_DATA) -> pd.DataFrame:
    """Таблица «игрок × матч» схемы L3; ставки на 60 = Σ / toi_rate × 3600 (как в L5)."""
    rng = np.random.default_rng(seed)
    pid = [f"p{r}{i:03d}" for r in ROLES for i in range(n_per_role)]
    role = np.repeat(ROLES, n_per_role)
    theta = rng.normal(0.0, talent_sd, len(pid))
    plays = rng.random((len(pid), n_matches)) < p_play
    pi, mi = np.nonzero(plays)
    n = len(pi)
    toi = rng.uniform(600.0, 1300.0, n)
    toi5 = 0.8 * toi
    h, h5 = toi / 3600.0, toi5 / 3600.0
    cols = {c: (COMP_MU[c] + COMP_SHARE[c] * theta[pi]) * h + COMP_SIGMA * np.sqrt(h) * rng.normal(size=n)
            for c in COMP_SHARE}
    xgf = (XGF_MU + theta[pi] / 2) * h5 + XG_SIGMA / np.sqrt(2) * np.sqrt(h5) * rng.normal(size=n)
    xga = (XGF_MU - theta[pi] / 2) * h5 + XG_SIGMA / np.sqrt(2) * np.sqrt(h5) * rng.normal(size=n)
    df = pd.DataFrame({"match_id": [f"m{m:03d}" for m in mi], "player_id": np.asarray(pid)[pi],
                       "role": role[pi], "toi_rate_s": toi, "toi_rate_5v5_s": toi5, **cols,
                       "xgf_5v5": xgf, "xga_5v5": xga,
                       "points": rng.poisson(POINTS_RATE * h).astype(float),
                       "ishots": rng.poisson(8.0 * h).astype(float),
                       "ixg": rng.gamma(2.0, 0.4, n) * h,
                       "gf_5v5": rng.poisson(GF_SHARE * np.maximum(xgf, 0) * 10).astype(float) / 10,
                       "ga_5v5": rng.poisson(GF_SHARE * np.maximum(xga, 0) * 10).astype(float) / 10,
                       "is_khl_component": True, "true_theta": theta[pi]})
    df["iav_sum"] = df[list(COMP_SHARE)].sum(axis=1)
    return df


def all_fns() -> dict:
    return {**rv.our_rating_fns(SHRINK), **rv.baseline_rating_fns(synth_pm(0.0, n_per_role=5, n_matches=4))}


@pytest.fixture(scope="module")
def talent_pm() -> pd.DataFrame:
    return synth_pm(talent_sd=0.45)


@pytest.fixture(scope="module")
def noise_pm() -> pd.DataFrame:
    return synth_pm(talent_sd=0.0, seed=SEED_DATA + 1)


@pytest.fixture(scope="module")
def talent_pred(talent_pm) -> dict:
    return rv.train_predictive(talent_pm, all_fns(), min_toi_s=MIN_TOI_TEST_S, n_boot=N_BOOT_TEST)


@pytest.fixture(scope="module")
def noise_pred(noise_pm) -> dict:
    return rv.train_predictive(noise_pm, all_fns(), min_toi_s=MIN_TOI_TEST_S, n_boot=N_BOOT_TEST)


# --- разбиение и утечки ---------------------------------------------------------------------

def test_halves_are_disjoint_by_match_and_cover_all(talent_pm):
    pm_a, pm_b, halves = rv.split_halves(talent_pm)
    assert not set(pm_a.match_id) & set(pm_b.match_id)
    assert set(pm_a.match_id) | set(pm_b.match_id) == set(talent_pm.match_id)
    assert len(pm_a) + len(pm_b) == len(talent_pm)
    assert abs(int(halves.sum()) - len(halves) / 2) <= 1


def test_split_does_not_depend_on_row_order(talent_pm):
    shuffled = talent_pm.sample(frac=1.0, random_state=0)
    a1, _, _ = rv.split_halves(talent_pm)
    a2, _, _ = rv.split_halves(shuffled)
    assert set(a1.match_id) == set(a2.match_id)


def test_rating_functions_see_only_past_matches(talent_pm):
    seen: list[set] = []

    def spy(pm_half: pd.DataFrame) -> pd.Series:
        seen.append(set(pm_half.match_id))
        return rv.o1_rating(pm_half, SHRINK)

    halves = va.assign_halves(talent_pm.match_id)
    a, b = set(halves.index[halves == 0]), set(halves.index[halves == 1])
    targets = rv.target_frames(talent_pm)
    rv.predict_future(talent_pm, {"O1": spy}, targets, a, b, MIN_TOI_TEST_S, n_boot=10)
    assert seen and all(s <= a for s in seen)


def test_future_half_does_not_change_past_rating(talent_pm):
    pm_a, pm_b, _ = rv.split_halves(talent_pm)
    noisy_b = pm_b.assign(**{c: pm_b[c] * 50.0 + 3.0 for c in COMP_SHARE})
    before = rv.o1_rating(rv.split_halves(talent_pm)[0], SHRINK)
    after = rv.o1_rating(rv.split_halves(pd.concat([pm_a, noisy_b]))[0], SHRINK)
    pd.testing.assert_series_equal(before.sort_index(), after.sort_index())


def test_overlapping_past_and_future_is_rejected(talent_pm):
    with pytest.raises(ValueError, match="пересекаются"):
        rv.predict_future(talent_pm, rv.our_rating_fns(SHRINK), rv.target_frames(talent_pm),
                          {"m000", "m001"}, {"m001"}, n_boot=5)


def test_input_is_not_mutated(talent_pm):
    before = talent_pm.copy()
    rv.o1_rating(talent_pm, SHRINK)
    rv.o0_rating(talent_pm, SHRINK)
    pd.testing.assert_frame_equal(talent_pm, before)


# --- реальный код L5 -------------------------------------------------------------------------

def test_o1_uses_l5_code_and_tracks_talent(talent_pm):
    rating = rv.o1_rating(talent_pm, SHRINK)
    theta = talent_pm.groupby("player_id").true_theta.first()
    assert rating.index.is_unique and np.isfinite(rating).all()
    assert rating.corr(theta.reindex(rating.index), method="spearman") > 0.6


def test_frozen_models_reproduce_run_l5_test_formula(talent_pm):
    """Замороженные модели = ветка test в run_l5: μ_r по КХЛ-компоненте, m_r по оцениваемой выборке."""
    pm_a, pm_b, _ = rv.split_halves(talent_pm)
    models = rv.fit_frozen_models(pm_a, SHRINK)
    fns = rv.frozen_rating_fns(models)
    o1 = fns["O1"](pm_b)
    rf, _ = stage_l5.rate_frame(pm_b, stage_l5.COMP_COLS)
    roles = stage_l5.player_roles(pm_b).combine_first(models.roles)
    ids = sorted(pm_b.player_id.unique())
    ref, _ = stage_l5.apply_o1(models.o1, stage_l5.player_sums(rf, stage_l5.COMP_COLS), roles, ids,
                               {"shrink": SHRINK})
    np.testing.assert_array_equal(o1.reindex(ids).to_numpy(), ref.rating.to_numpy())
    assert models.mu_source == "khl_component"
    assert np.isfinite(fns["O0"](pm_b)).all()



# --- прогноз: талант против шума -------------------------------------------------------------

def test_o1_predicts_future_xgd_better_than_points(talent_pred):
    res = talent_pred["A->B"]["targets"][rv.MAIN_TARGET]
    o1 = res["ratings"]["O1"][va.POOLED]
    diff = res["ref_minus"]["points_60_raw"][va.POOLED]
    assert o1["spearman_ci"][0] > 0
    assert diff["spearman_diff_ci"][0] > 0
    assert res["n_players"] > 100


def test_reference_minus_is_negated_vs_baseline(talent_pred):
    res = talent_pred["A->B"]["targets"][rv.MAIN_TARGET]
    for name, groups in res["ref_minus"].items():
        p = res["ratings"]["O1"][va.POOLED]["spearman"] - res["ratings"][name][va.POOLED]["spearman"]
        assert groups[va.POOLED]["spearman_diff"] == pytest.approx(p)
        lo, hi = groups[va.POOLED]["spearman_diff_ci"]
        assert lo <= hi


def test_noise_population_intervals_cover_zero(noise_pred):
    checked = 0
    for direction in ("A->B", "B->A"):
        res = noise_pred[direction]["targets"][rv.MAIN_TARGET]
        for name, r in res["ratings"].items():
            lo, hi = r[va.POOLED]["spearman_ci"]
            if np.isfinite(lo):
                assert lo <= 0.0 <= hi, (direction, name, lo, hi)
                checked += 1
        lo, hi = res["ref_minus"]["points_60_raw"][va.POOLED]["spearman_diff_ci"]
        if np.isfinite(lo):
            assert lo <= 0.0 <= hi
    assert checked >= 8


def test_all_targets_present(talent_pred):
    assert set(talent_pred["A->B"]["targets"]) == {"xgd_5v5_60", "gd_5v5_60", "iav_60"}


def test_missing_main_target_columns_raise(talent_pm):
    with pytest.raises(KeyError, match="основной цели"):
        rv.target_frames(talent_pm.drop(columns=["xgf_5v5"]))


# --- надёжность и правило вывода ------------------------------------------------------------

def test_reliability_o1_above_points_on_talent(talent_pm):
    rel = rv.rating_reliability(talent_pm, all_fns(), min_toi_s=MIN_TOI_TEST_S, n_boot=N_BOOT_TEST)
    o1 = rel["ratings"]["O1"][va.POOLED]
    pts = rel["ratings"]["points_60_raw"][va.POOLED]
    assert o1["spearman_sb"] > pts["spearman_sb"]
    assert rel["ref_minus"]["points_60_raw"]["spearman_sb_diff_ci"][0] > 0
    assert o1["spearman_sb"] == pytest.approx(va.spearman_brown(o1["spearman"]))


def test_reliability_noise_covers_zero(noise_pm):
    rel = rv.rating_reliability(noise_pm, rv.baseline_rating_fns(noise_pm) | rv.our_rating_fns(SHRINK),
                                min_toi_s=MIN_TOI_TEST_S, n_boot=N_BOOT_TEST)
    lo, hi = rel["ratings"]["points_60_raw"][va.POOLED]["spearman_ci"]
    assert lo <= 0.0 <= hi


def test_acceptance_rule_on_talent(talent_pm, talent_pred):
    rel = rv.rating_reliability(talent_pm, all_fns(), min_toi_s=MIN_TOI_TEST_S, n_boot=N_BOOT_TEST)
    verdict = rv.acceptance(rel, talent_pred)
    assert verdict["reliability_above_m0"] is True
    assert isinstance(verdict["prediction_not_worse_than_o0"], bool)


# --- хронология test ------------------------------------------------------------------------

def test_temporal_is_diagnostic_and_uses_frozen_models(talent_pm):
    pm_tr, pm_te = talent_pm[talent_pm.match_id < "m040"], talent_pm[talent_pm.match_id >= "m040"]
    games = pd.DataFrame({"match_id": sorted(pm_te.match_id.unique())})
    games = games.assign(game_seq=np.arange(len(games))[::-1])
    models = rv.fit_frozen_models(pm_tr, SHRINK)
    fns = rv.frozen_rating_fns(models) | rv.baseline_rating_fns(pm_tr, rv.baseline_shrink_params(pm_tr))
    res = rv.temporal_predictive(pm_te, games, fns, min_toi_s=20 * 60.0, n_boot=50)
    assert res["diagnostic_only"] is True
    early, late = rv.temporal_halves(games)
    assert not early & late and len(early) == len(late) == 10
    assert max(games.set_index("match_id").game_seq[list(early)]) < min(games.set_index("match_id").game_seq[list(late)])
    assert res["targets"][rv.MAIN_TARGET]["n_players"] > 50


# --- скрипт: неготовый кэш — понятная ошибка без пересчёта ----------------------------------

def load_script():
    spec = importlib.util.spec_from_file_location("validation_report", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_raises_clear_error_when_caches_missing(tmp_path):
    script = load_script()
    cfg = load_config().with_overrides({"paths": {
        "cache": str(tmp_path / "cache"), "artifacts": str(tmp_path / "artifacts"),
        "outputs": str(tmp_path / "outputs"), "logs": str(tmp_path / "logs")}})
    cache = script.ReadOnlyCache(cfg, {})
    with pytest.raises(script.CachesNotReady, match="shrink"):
        script.load_pipeline_tables(cfg, cache)
    assert not any((tmp_path / "cache").glob("l*__*/_DONE"))


def test_read_only_cache_blocks_recompute(tmp_path):
    script = load_script()
    cfg = load_config().with_overrides({"paths": {"cache": str(tmp_path / "c"), "artifacts": str(tmp_path / "a"),
                                                  "outputs": str(tmp_path / "o"), "logs": str(tmp_path / "l")}})
    with pytest.raises(script.CachesNotReady):
        script.ReadOnlyCache(cfg, {}).invalidate_done("l3__x")


def test_attach_onice_goals_overrides_points_without_mutation(talent_pm):
    l3 = talent_pm.drop(columns=["gf_5v5", "ga_5v5"])
    l0 = talent_pm[["match_id", "player_id", "toi_rate_s", "toi_rate_5v5_s"]].assign(
        gf=1.0, ga=0.0, gf_5v5=0.5, ga_5v5=0.25, points=9.0).iloc[:-3]
    before = l3.copy()
    out, check = rv.attach_onice_goals(l3, l0)
    pd.testing.assert_frame_equal(l3, before)
    assert len(out) == len(l3) and check["rows_matched_l0"] == len(l3) - 3
    assert check["max_abs_toi_rate_diff_s"] == 0.0
    assert out.points.iloc[0] == 9.0 and out.points.iloc[-1] == 0.0
    assert "gd_5v5_60" in rv.target_frames(out)
