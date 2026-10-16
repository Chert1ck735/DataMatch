"""Карточка игрока, пункт 5: физический профиль Ф1–Ф2 на текущем L0 (METHODOLOGY 11.5).

Запуск из Solve (кэш L0 только читается):

    PYTHONPATH=src .venv/bin/python scripts/card_phys.py [--n-boot 200]

Метрики и их расчёт — функции `scripts/phys_report.py` без правок; train сужается до матчей разработки
до построения наблюдений (внешний блок не читается). На разработке: k доли быстрых атак, надёжность
split-half и решение по порогу 0,3, стягивание; test только применяется. Скорость с шайбой на
объединении «разработка + test» — описательно. Выход: outputs/card/phys.csv и phys_card_report.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import phys_report as pr

from datamatch import card_phys as cp
from datamatch.config import load_config, setup_logging
from datamatch.splits import load_splits, matches_for, team_components

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "card"
NAMES = {"speed_p80": "speed_p80", "pass_speed_median": "pass_speed_median",
         "tempo_median_s": "tempo_median_s", "fast_share": "fast_share"}


def khl_dev_matches(l0: Path, dev: set) -> tuple[set, dict]:
    """Матчи разработки в компоненте связности (граф только по матчам разработки) большинства команд test."""
    tr = pd.read_parquet(l0 / "train_games.parquet", columns=["match_id", "home_team_id", "away_team_id"])
    tr = tr[tr.match_id.isin(dev)]
    te = pd.read_parquet(l0 / "test_games.parquet", columns=["home_team_id", "away_team_id"])
    comp = team_components(tr)
    test_teams = pd.Series(pd.concat([te.home_team_id, te.away_team_id]).unique())
    known = test_teams.map(comp).dropna()
    if known.empty:
        raise ValueError("khl_dev_matches: команды test не встречаются в матчах разработки")
    cid = int(known.mode().iat[0])
    ids = set(tr.match_id[tr.home_team_id.map(comp) == cid])
    return ids, {"component": cid, "n_dev_matches": len(ids),
                 "test_teams_in_component": int((known == cid).sum()), "test_teams": int(len(test_teams))}


def build_dev_part(l0: Path, dev: set) -> dict:
    ev, stints = pr.load_part(l0, "train")
    ev = cp.restrict_to_matches(ev, dev)
    stints = stints[stints.match_id.isin(dev)].reset_index(drop=True)
    return pr.build_part(ev, stints)


def _finite(o):
    """NaN/inf → None (строгий JSON); numpy-скаляры — в Python."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def run(args) -> dict:
    cfg = load_config()
    log = setup_logging(cfg, "card_phys")
    l0 = pr.current_l0_cache_dir()
    dev = set(matches_for(load_splits(cfg), "dev"))
    roles = pr.load_roles(l0)
    parts = {"dev": build_dev_part(l0, dev), "test": pr.build_part(*pr.load_part(l0, "test"))}
    log.info("card_phys: наблюдения разработки и test построены")
    khl, khl_info = khl_dev_matches(l0, dev)
    k_best, k_grid = pr.choose_fast_k(parts["dev"]["transitions"], roles)
    specs = {"speed_p80": pr.SPEED, "pass_speed_median": pr.PASS, "tempo_median_s": pr.TEMPO,
             "fast_share": pr.fast_spec(k_best)}
    obs = {p: {**parts[p]["obs"], "fast_share": pr.fast_obs(parts[p]["transitions"], k_best)} for p in parts}
    reports, shrunk = {}, {}
    for key, spec in specs.items():
        reports[key], shrunk[key] = pr.evaluate_metric(key, spec, obs["dev"][key], obs["test"][key],
                                                       roles, khl, args.n_boot)
    decision = cp.decision_table({k: r["train"] for k, r in reports.items()}, pr.FALLBACK)
    level = decision.set_index("metric").level
    csv = pr.assemble_csv(pr.test_players(l0, roles), shrunk, NAMES)
    for k in specs:
        csv = cp.apply_level(csv, NAMES[k], level[k], reports[k]["mu_test_prior_khl"])
    csv = csv.assign(is_avangard=csv.team_id.eq(str(cfg["avangard_team_id"])))
    pooled = cp.pooled_descriptive(obs["dev"]["speed_p80"], obs["test"]["speed_p80"], roles, pr.SPEED)
    report = {
        "l0_cache": l0.name, "dev_matches": len(dev), "n_boot": args.n_boot, "khl_prior_dev": khl_info,
        "fast_attack_k_s": k_best, "fast_attack_k_grid_dev_sb": k_grid,
        "decision_dev": decision.to_dict(orient="records"),
        "speed_p80_pooled_dev_test_descriptive": pooled,
        "metrics": {k: {"dev": r["train"], "test_diagnostic": r["test_diagnostic"], "shrinkage_dev": r["shrinkage"],
                        "mu_test_prior_khl_dev": r["mu_test_prior_khl"], "level": level[k]}
                    for k, r in reports.items()},
        "counts": {p: {"speed_pairs": parts[p]["pair_counts"], "passes": parts[p]["pass_counts"],
                       "n_obs": {k: int(len(v)) for k, v in obs[p].items()}} for p in parts},
        "test_rows": int(len(csv)),
        "test_label_counts": {k: csv[f"{NAMES[k]}_label"].value_counts().to_dict() for k in specs},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    csv.to_csv(args.out / "phys.csv", index=False)
    (args.out / "phys_card_report.json").write_text(
        json.dumps(_finite(report), ensure_ascii=False, indent=2, default=pr._json_default, allow_nan=False))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=pr.DEFAULT_N_BOOT)
    args = ap.parse_args(argv)
    rep = run(args)
    for row in rep["decision_dev"]:
        print(f"{row['metric']}: SB разработки {row['split_half_sb_dev']:.3f} (n={row['n_players_dev']}) "
              f"→ {'прошёл' if row['passed'] else 'не прошёл'}, уровень: {row['level']}")
    p = rep["speed_p80_pooled_dev_test_descriptive"]
    print(f"скорость с шайбой, разработка + test (описательно): SB {p['split_half_sb']:.3f} (n={p['n_players']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
