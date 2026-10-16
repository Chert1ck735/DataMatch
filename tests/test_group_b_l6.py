"""Группа B (аудит 08.10.2026): провенанс L6, гейт упавших реплик, явный отказ от O0 в полном
режиме, исходное исключение инициализатора воркера, моменты шума без запасной роли «C».

Контракты: (B1) смена seed или eval.drift_grid делает файл L6 устаревшим; (B2) в полном режиме любая
упавшая реплика — непройденный гейт, а не тихая замена; (B3) полный режим без явного назначения O0
не пишет сдаваемый файл с O0; (B4) падение загрузки контекста в воркере сообщается родителю исходным
текстом, без повторных перезапусков «как при OOM»; (B5) при разных моментах по амплуа общей пары
σ², ω² нет — предсказательный интервал не берёт моменты первого амплуа.
"""
from __future__ import annotations

import inspect

import pytest

from datamatch import l6_replicate as R
from datamatch import l6_uncertainty as U
from datamatch import stage_l6 as S
from datamatch.config import load_config

from .test_l6_ratings import l5_dict, latent_table


# --- B1 -------------------------------------------------------------------------------------

def test_l6_provenance_covers_seed_and_eval():
    cfg = load_config()
    assert {"seed", "eval"} <= set(S.L6_PROV_SECTIONS)
    base = cfg.section_hash(*S.L6_PROV_SECTIONS)
    assert cfg.with_overrides({"seed": cfg.seed + 1}).section_hash(*S.L6_PROV_SECTIONS) != base
    grid = list(cfg["eval"]["drift_grid"]) + [0.3]
    assert cfg.with_overrides({"eval": {**cfg["eval"], "drift_grid": grid}}).section_hash(*S.L6_PROV_SECTIONS) != base
    assert "L6_PROV_SECTIONS" in inspect.getsource(S._provenance)


# --- B2 -------------------------------------------------------------------------------------

def _boot(n_failed: int) -> U.BootstrapResult:
    return U.BootstrapResult(U.STATUS_DONE, "", 200, 200, n_failed=n_failed, n_attempted=200 + n_failed)


def test_full_mode_gate_on_failed_replicates():
    assert S.bootstrap_gates(_boot(0), "full")["full_mode_no_failed_replicates"]
    assert not S.bootstrap_gates(_boot(1), "full")["full_mode_no_failed_replicates"]
    assert S.bootstrap_gates(_boot(1), "fast")["full_mode_no_failed_replicates"]     # fast — допуск 5%


# --- B3 -------------------------------------------------------------------------------------

def test_full_mode_refuses_silent_o0():
    t = latent_table(n=3)
    definition, why = S.choose_definition(l5_dict(t, report={}))   # гейтов O1 нет — молча был бы O0
    assert definition == "O0"
    with pytest.raises(S.L6InputsNotReady):
        S.require_final_definition(definition, why, "full")
    S.require_final_definition(definition, why, "fast")              # отладочный режим — допустимо
    forced = S.choose_definition(l5_dict(t, report={"final_definition": "O0"}))
    S.require_final_definition(*forced, "full")                        # явное назначение L5 — допустимо


# --- B4 -------------------------------------------------------------------------------------

def _broken_loader(cfg, cache):
    raise ValueError("контекст не загружен: нет кэша L2")


def test_worker_init_error_keeps_original_exception(tmp_path):
    class FakeCache:
        root = tmp_path

    R.clear_init_error(FakeCache)
    with pytest.raises(ValueError):
        R._init_worker(1, None, FakeCache, _broken_loader)
    msg = R.read_init_error(FakeCache)
    assert "ValueError" in msg and "нет кэша L2" in msg

    class Broken:
        def shutdown(self, *a, **k):
            pass

    runner = U._PoolRunner(lambda w: Broken(), 2, init_error=lambda: R.read_init_error(FakeCache))
    reason = runner._restart(RuntimeError("broken"), 1)
    assert "нет кэша L2" in reason and runner.restarts == 0       # не перезапуск «как при OOM»


# --- B5 -------------------------------------------------------------------------------------

def _report(sig):
    return {"moments": {"univariate": {r: {"tau2": 0.05, "sigma2": s, "omega2": 0.2} for r, s in sig.items()}}}


def test_overall_noise_moments_without_role_fallback():
    assert U.overall_noise_moments(_report({"C": 1.0, "W": 2.0, "D": 3.0})) is None
    same = U.overall_noise_moments(_report({"C": 1.0, "W": 1.0, "D": 1.0}))
    assert same is not None and same["sigma2"] == 1.0
