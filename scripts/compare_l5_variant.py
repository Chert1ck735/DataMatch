"""Сравнение «было / стало» для проверяемого улучшения L5 (METHODOLOGY 5.2, флаг shrink.t_offdiag_shrink).

Запуск из Solve (кэши L3/L5 только читаются):

    PYTHONPATH=src .venv/bin/python scripts/compare_l5_variant.py [--n-boot 1000]

O1 «было» (флаг выключен) и O1 «стало» (флаг включён) переподгоняются на одних и тех же половинах
матчей train; парный бутстрэп игроков даёт разности надёжности половин и прогноза. Покрытие 95%-интервала
«стало» — на тех же половинах. Решение — по правилу приёма METHODOLOGY 5.2. Выход: outputs/l5_variant_compare.json
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import validation_report as vr

from datamatch import card_io, io
from datamatch import l7_calibration as cal
from datamatch import l7_forecast as fc
from datamatch import rating_validation as rv
from datamatch import validation as va
from datamatch.config import load_config, setup_logging

OUT = Path(__file__).resolve().parents[1] / "outputs" / "l5_variant_compare.json"
FLAG = "t_offdiag_shrink"
BASE, VARIANT = "O1", "O1_offdiag"
# CONFIG: правило приёма METHODOLOGY 5.2 (записано до расчёта)
REL_NONINF = -0.02
SS_NONINF = -0.01
COVERAGE_BAND = (0.93, 0.97)


def rating_fns(shrink: dict) -> dict:
    return {BASE: partial(rv.o1_rating, shrink=shrink | {FLAG: False}),
            VARIANT: partial(rv.o1_rating, shrink=shrink | {FLAG: True})}


def _flip(point: float, ci) -> tuple[float, list[float]]:
    """Отчёты дают «было − стало»; правило — в «стало − было»."""
    return -float(point), [-float(ci[1]), -float(ci[0])]


def decision(rel: dict, pred: dict, cov: dict) -> dict:
    """Правило приёма METHODOLOGY 5.2 (гейты L1–L6 — при пересчёте L5/L6 с флагом)."""
    dr, dr_ci = _flip(rel["ref_minus"][VARIANT]["spearman_sb_diff"], rel["ref_minus"][VARIANT]["spearman_sb_diff_ci"])
    o1m = pred["directions"]["A->B"]["o1_minus"][VARIANT]
    dss, dss_ci = _flip(o1m["ss_diff"], o1m["ss_diff_ci"])
    c = float(cov["pooled"]["coverage"])
    checks = {"reliability_not_worse": dr_ci[0] >= REL_NONINF, "forecast_not_worse": dss_ci[0] >= SS_NONINF,
              "coverage_in_band": COVERAGE_BAND[0] <= c <= COVERAGE_BAND[1]}
    return {"delta_r_sb": dr, "delta_r_sb_ci": dr_ci, "delta_ss_a_to_b": dss, "delta_ss_ci": dss_ci,
            "coverage_variant": c, "checks": checks, "better_reliability": dr_ci[0] > 0,
            "accepted_pending_gates": all(checks.values())}


def run(args) -> dict:
    cfg = load_config()
    setup_logging(cfg, "compare_l5_variant")
    cache = vr.ReadOnlyCache(cfg, io.data_version(cfg))
    pm_tr = card_io.load_card_tables(cfg, cache)["pm_train"]
    shrink = dict(cfg["shrink"])
    fns = rating_fns(shrink)
    min_toi = args.min_toi_min * 60.0
    rel = rv.rating_reliability(pm_tr, fns, args.seed, min_toi, args.n_boot)
    pred = fc.train_forecast(pm_tr, fns, args.seed, min_toi, args.n_boot)
    pairs = cal.half_pairs(pm_tr, shrink | {FLAG: True}, args.seed)
    cov = cal.predictive_coverage(pairs, 0.0, n_boot=args.n_boot, seed=args.seed)
    return {"seed": args.seed, "n_boot": args.n_boot, "min_toi_min": args.min_toi_min, "flag": FLAG,
            "reliability": rel, "predictive": pred, "coverage_variant": cov, "decision": decision(rel, pred, cov)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=va.N_BOOT)
    ap.add_argument("--seed", type=int, default=va.SEED)
    ap.add_argument("--min-toi-min", type=float, default=vr.DEFAULT_MIN_TOI_MIN)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    rep = run(args)
    args.out.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(rep["decision"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
