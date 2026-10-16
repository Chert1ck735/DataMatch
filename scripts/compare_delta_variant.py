"""Сравнение «было / стало» для δ по исходной марковской V (METHODOLOGY 3, поправка 08.10; правило 5.2).

Запуск из Solve после расчёта обоих вариантов L3 (кэши только читаются):

    PYTHONPATH=src .venv/bin/python scripts/compare_delta_variant.py --base cache/l3__A --variant cache/l3__B

player_match train обоих вариантов склеивается по (player_id, match_id): время на льду и состав строк общие
(L0 один), различаются только компоненты c_* (из δ). O1 «было» и «стало» переподгоняются на одних и тех же
половинах матчей; парный бутстрэп игроков даёт разности надёжности и прогноза; покрытие — для «стало».
Выход: outputs/delta_variant_compare.json.
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import compare_l5_variant as cmp
import pandas as pd

from datamatch import l7_calibration as cal
from datamatch import l7_forecast as fc
from datamatch import rating_validation as rv
from datamatch import validation as va
from datamatch.config import load_config, setup_logging

OUT = Path(__file__).resolve().parents[1] / "outputs" / "delta_variant_compare.json"
SUFFIX = "__variant"
KEYS = ["player_id", "match_id"]


def merged_player_match(base: pd.DataFrame, variant: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Строки «было» + компоненты «стало» с суффиксом; строки и время на льду обязаны совпадать."""
    comp = [c for c in base.columns if c.startswith("c_")]
    if len(base) != len(variant) or not base[KEYS].sort_values(KEYS).reset_index(drop=True).equals(
            variant[KEYS].sort_values(KEYS).reset_index(drop=True)):
        raise ValueError("варианты L3 различаются составом строк player_match")
    m = base.merge(variant[KEYS + ["toi_s"] + comp].add_suffix(SUFFIX).rename(
        columns={f"{k}{SUFFIX}": k for k in KEYS}), on=KEYS, how="inner", validate="one_to_one")
    if not (m.toi_s == m[f"toi_s{SUFFIX}"]).all():
        raise ValueError("варианты L3 различаются временем на льду")
    return m.drop(columns=[f"toi_s{SUFFIX}"]), comp


def _variant_view(pm_half: pd.DataFrame, comp: list[str]) -> pd.DataFrame:
    return pm_half.drop(columns=comp).rename(columns={f"{c}{SUFFIX}": c for c in comp})


def _o1_variant(pm_half: pd.DataFrame, shrink: dict, comp: list[str]) -> pd.Series:
    return rv.o1_rating(_variant_view(pm_half, comp), shrink)


def run(args) -> dict:
    cfg = load_config()
    setup_logging(cfg, "compare_delta_variant")
    base = pd.read_parquet(Path(args.base) / "train_player_match.parquet")
    var = pd.read_parquet(Path(args.variant) / "train_player_match.parquet")
    pm, comp = merged_player_match(base, var)
    shrink = dict(cfg["shrink"])
    fns = {cmp.BASE: partial(rv.o1_rating, shrink=shrink),
           cmp.VARIANT: partial(_o1_variant, shrink=shrink, comp=comp)}
    min_toi = args.min_toi_min * 60.0
    rel = rv.rating_reliability(pm, fns, args.seed, min_toi, args.n_boot)
    pred = fc.train_forecast(pm, fns, args.seed, min_toi, args.n_boot)
    pairs = cal.half_pairs(_variant_view(pm, comp), shrink, args.seed)
    cov = cal.predictive_coverage(pairs, 0.0, n_boot=args.n_boot, seed=args.seed)
    return {"base_l3": Path(args.base).name, "variant_l3": Path(args.variant).name, "seed": args.seed,
            "n_boot": args.n_boot, "reliability": rel, "predictive": pred, "coverage_variant": cov,
            "decision": cmp.decision(rel, pred, cov)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="каталог кэша L3 «было»")
    ap.add_argument("--variant", required=True, help="каталог кэша L3 «стало» (value.delta_from_raw: true)")
    ap.add_argument("--n-boot", type=int, default=va.N_BOOT)
    ap.add_argument("--seed", type=int, default=va.SEED)
    ap.add_argument("--min-toi-min", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    rep = run(args)
    args.out.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(rep["decision"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
