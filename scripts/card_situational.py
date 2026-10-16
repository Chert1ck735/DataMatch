"""Карточка игрока, пункт 3: ситуационные рейтинги 5 на 5, большинство, меньшинство (METHODOLOGY 11.3).

Запуск из Solve (кэши L3/L5 только читаются):

    PYTHONPATH=src .venv/bin/python scripts/card_situational.py

Моменты каждой ситуации (T_r, Σσ, Σω) — класс `JointEB` L5 на матчах разработки train; μ_r для test —
КХЛ-компонента разработки (как в L5). Интервал — апостериорный, без бутстрэпа моделей ценности.
Выход: outputs/card/situational.csv и situational_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import validation_report as vr

from datamatch import card_io, io, stage_l5
from datamatch import card_situational as cs
from datamatch.cache import code_hash
from datamatch.config import load_config, setup_logging
from datamatch.l5_shrink import ROLES, JointEB
from datamatch.splits import load_splits, matches_for

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "card"
# CONFIG: зафиксировано в METHODOLOGY 11.3 до расчёта.
MIN_TOI_MIN = 60.0          # «мало минут» в ситуации
SPLIT_MIN_TOI_S = 5 * 60.0  # минимум TOI ситуации в каждой половине для split-half
Z95 = 1.96
# Диагностика (не критерий): тот же минимум, что у split-half O1 в разделе 7.1, — для сравнения.
COMPARE_MIN_TOI_S = 60 * 60.0


def fit_situation(sit: str, pm_dev: pd.DataFrame, pm_te: pd.DataFrame, roles_tr: pd.Series,
                  roles_all: pd.Series, cfg, log) -> tuple[pd.DataFrame, dict]:
    dev = cs.situation_frame(pm_dev, sit)
    rf_dev, exp_dev = stage_l5.rate_frame(dev, cs.COMP_COLS)
    eb = JointEB(cs.COMP_COLS, cfg["shrink"], logger=log).fit(rf_dev, roles_tr)
    khl = dev[dev.is_khl_component.fillna(False).astype(bool)] if "is_khl_component" in dev else dev.iloc[:0]
    rf_khl = stage_l5.rate_frame(khl, cs.COMP_COLS)[0]
    # Как в L5 (_mu_source): нет КХЛ-экспозиции в ситуации → μ_r по всей разработке, с записью в отчёт.
    mu_source = "khl_component_dev" if len(rf_khl) and float(rf_khl.toi_s.sum()) > 0 else "all_dev"
    eb_test = eb.with_mu(eb.prior_mu(rf_khl if mu_source == "khl_component_dev" else rf_dev, roles_tr))
    te = cs.situation_frame(pm_te, sit)
    rf_te, exp_te = stage_l5.rate_frame(te, cs.COMP_COLS)
    ids = sorted(pm_te.player_id.unique())
    df, rep = stage_l5.apply_o1(eb_test, stage_l5.player_sums(rf_te, cs.COMP_COLS), roles_all, ids, cfg)
    sh = cfg["shrink"]
    lab = cs.display_label(df.w.to_numpy(), df.toi_min.to_numpy(), float(sh["reliability_high"]),
                           float(sh["reliability_medium"]), MIN_TOI_MIN)
    sd = cs.posterior_sd(df.var_post.to_numpy())
    out = pd.DataFrame({"player_id": df.player_id, "rating": df.rating, "lo95": df.rating - Z95 * sd,
                        "hi95": df.rating + Z95 * sd, "w": df.w, "reliability": lab.label.to_numpy(),
                        "flag": cs.merge_flags(df.flag.to_numpy(), lab.flag.to_numpy()), "toi_min": df.toi_min, "method": df.method})
    rel_in = rf_dev.assign(value=rf_dev[cs.COMP_COLS].sum(axis=1), role=rf_dev.player_id.map(roles_tr))
    rel_cols = rel_in[["player_id", "match_id", "role", "toi_s", "value"]]
    report = {"mu_source": mu_source, "split_half_dev": cs.split_half_reliability(rel_cols, SPLIT_MIN_TOI_S),
              "split_half_dev_60min_diag": cs.split_half_reliability(rel_cols, COMPARE_MIN_TOI_S),
              "gates_l5_style": stage_l5.soft_gates(eb, df, cfg),
              "oneT1": {r: float(eb.T[r].sum()) for r in ROLES},
              "method_counts": rep["method_counts"], "exposure_dev": exp_dev, "exposure_test": exp_te,
              "test_labels": out.reliability.value_counts().to_dict(),
              "test_low_minutes": int(out.flag.str.contains(cs.LOW_MINUTES, regex=False).sum()),
              "test_toi_min_median": float(out.toi_min.median())}
    return out, report


def _finite(o):
    """NaN/inf → None: строгий JSON."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def run(args) -> dict:
    cfg = load_config()
    log = setup_logging(cfg, "card_situational")
    cache = vr.ReadOnlyCache(cfg, io.data_version(cfg))
    tables = card_io.load_card_tables(cfg, cache)
    dev_ids = set(matches_for(load_splits(cfg), "dev"))
    pm_tr, pm_te = tables["pm_train"], tables["pm_test"]
    pm_dev = pm_tr[pm_tr.match_id.isin(dev_ids)]
    # Роли для подгонки — по матчам разработки (внешний блок не открывается); для test — как в L5.
    roles_tr = stage_l5.player_roles(pm_dev)
    roles_all = stage_l5.player_roles(pm_te).combine_first(stage_l5.player_roles(pm_tr))
    tabs, reps = {}, {}
    for sit in cs.SITUATIONS:
        tabs[sit], reps[sit] = fit_situation(sit, pm_dev, pm_te, roles_tr, roles_all, cfg, log)
    o1 = tables["l5"]["O1"]["test"]
    cons = cs.consistency(o1[["player_id", "rating"]], tabs, o1.set_index("player_id").toi_min)
    team = pm_te.groupby("player_id").team_id.agg(lambda s: s.mode().iat[0])
    wide = o1[["player_id", "role", "rating", "toi_min"]].rename(columns={"rating": "o1", "toi_min": "toi_min_all"})
    for sit, t in tabs.items():
        wide = wide.merge(t.add_suffix(f"_{sit}").rename(columns={f"player_id_{sit}": "player_id"}), on="player_id")
    wide.insert(1, "team_id", wide.player_id.map(team))
    wide.insert(2, "is_avangard", wide.team_id.eq(str(cfg["avangard_team_id"])))
    wide["o1_from_situations"] = wide.player_id.map(cons["weighted"])
    args.out.mkdir(parents=True, exist_ok=True)
    wide.to_csv(args.out / "situational.csv", index=False)
    report = {"l5_cache": tables["l5"]["key"], "l3_key": tables["l3_key"], "l5_shrink_code": code_hash("l5_shrink"),
              "dev_matches": len(dev_ids), "interval": "апостериорный ±1,96·√var_post, без бутстрэпа моделей ценности",
              "situations": reps,
              "consistency_vs_o1": {k: v for k, v in cons.items() if k != "weighted"},
              "avangard": {sit: tabs[sit][tabs[sit].player_id.isin(wide.player_id[wide.is_avangard])]
                           .reliability.value_counts().to_dict() for sit in tabs}}
    (args.out / "situational_report.json").write_text(json.dumps(_finite(report), ensure_ascii=False, indent=2, default=str, allow_nan=False))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    try:
        rep = run(args)
    except (vr.CachesNotReady, card_io.CardInputMismatch) as ex:
        print(f"кэши не готовы: {ex}", file=sys.stderr)
        return 3
    for sit, r in rep["situations"].items():
        sh = r["split_half_dev"]
        print(f"{sit}: r_SB={sh['r_sb']:.3f} (n={sh['n_players']}), метки test {r['test_labels']}, "
              f"методы {r['method_counts']}")
    c = rep["consistency_vs_o1"]
    print(f"Σ ситуаций vs O1: corr {c['corr']:.3f}, MAE {c['mae']:.4f}, смещение {c['mean_diff']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
