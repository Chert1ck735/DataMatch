"""Отчёт по базовым строкам: устойчивость (split-half) и прогноз будущего, сравнение с M0.

Запуск из Solve:

    PYTHONPATH=src .venv/bin/python scripts/baseline_report.py [--cache-dir DIR] [--xg FILE.parquet]

`--cache-dir` — каталог кэша L0 (train/test таблицы и players.parquet); по умолчанию —
каталог разработки. `--xg` — необязательный parquet с колонками match_id, seq, xg по узлам-
броскам (train и test в одном файле); без него xG-показатели заменяются попытками бросков с
явной пометкой «fallback». Результат: outputs/baseline_report.json и .md.

Train: случайные половины матчей (порядка во времени нет). Test: хронология «Авангарда» по
game_seq, моменты стягивания — по всему train. Test — только диагностика.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from datamatch import baselines as bl
from datamatch import validation as va

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs"
RAW_METRICS = ["points_60", "ishots_60", "igoals_60", "gd_60", "gd_5v5_60", "cd_5v5_60"]
SHRUNK_METRICS = ["points_60", "gd_5v5_60", "cd_5v5_60"]
XG_RAW_METRICS = ["ixg_60", "xgd_5v5_60"]
XG_SHRUNK_METRICS = ["xgd_5v5_60"]
BASELINE_RATING = "points_60_raw"      # M0 турнира L7
GOAL_TARGET = "gd_5v5_60"
XG_TARGET, XG_FALLBACK_TARGET = "xgd_5v5_60", "cd_5v5_60"



def current_l0_cache_dir() -> Path:
    """Каталог готового кэша L0 текущей ревизии: тот же ключ, что у конвейера (данные, конфигурация, код)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from datamatch import io, stage_l0
    from datamatch.cache import Cache
    from datamatch.config import load_config
    cfg = load_config()
    cache = Cache(cfg, io.data_version(cfg))
    key = cache.key("l0", ("l0",), code=stage_l0.L0_CODE)
    if not cache.done(key):
        raise SystemExit(f"нет готового кэша L0 {key}: сначала `python -m datamatch.cli prepare`")
    return cache.root / key

def load_xg(path: Path | None) -> pd.Series | None:
    """xG по броскам: Series с MultiIndex (match_id, seq)."""
    if path is None:
        return None
    df = pd.read_parquet(path, columns=["match_id", "seq", "xg"])
    if df.duplicated(["match_id", "seq"]).any():
        raise ValueError(f"{path}: повторяющиеся ключи (match_id, seq)")
    return df.set_index(["match_id", "seq"]).xg.astype(float)


def raw_rating(pm_half: pd.DataFrame, metric: str) -> pd.Series:
    return bl.player_rates(bl.metric_frame(pm_half, bl.METRICS[metric])).set_index("player_id").rate_60


def shrunk_rating(pm_half: pd.DataFrame, metric: str, params: bl.ShrinkParams | None = None) -> pd.Series:
    out = bl.shrunk_rates(bl.metric_frame(pm_half, bl.METRICS[metric]), params)
    return out.set_index("player_id").rate_shrunk


def rating_fns(has_xg: bool, params: dict | None = None) -> dict:
    """Функции рейтинга; `params` — фиксированные моменты стягивания (для test — из train)."""
    raw = RAW_METRICS + (XG_RAW_METRICS if has_xg else [])
    shr = SHRUNK_METRICS + (XG_SHRUNK_METRICS if has_xg else [])
    fns = {f"{m}_raw": partial(raw_rating, metric=m) for m in raw}
    for m in shr:
        fns[f"{m}_shrunk"] = partial(shrunk_rating, metric=m, params=None if params is None else params[m])
    return fns


def targets(has_xg: bool) -> dict:
    t = {GOAL_TARGET: bl.METRICS[GOAL_TARGET].label}
    if has_xg:
        t[XG_TARGET] = bl.METRICS[XG_TARGET].label
    else:
        t[XG_FALLBACK_TARGET] = bl.METRICS[XG_FALLBACK_TARGET].label + " [xG нет: fallback]"
    return t


def reliability_block(pm: pd.DataFrame, metrics: list[str], n_boot: int, seed: int) -> dict:
    out = {}
    for m in metrics:
        mf = bl.metric_frame(pm, bl.METRICS[m])
        by_players = va.split_half_reliability(mf, n_boot=n_boot, seed=seed, boot="players")
        out[m] = {"label": bl.METRICS[m].label, **va.split_half_reliability(mf, n_boot=n_boot, seed=seed),
                  "players_bootstrap": {k: by_players[k] for k in ("groups", "pooled")}}
    return out


def predictive_block(pm: pd.DataFrame, fns: dict, tg: dict, n_boot: int, seed: int) -> dict:
    return {t: {"label": label, **va.half_split_predictive(pm, fns, bl.metric_frame(pm, bl.METRICS[t]),
                                                          baseline=BASELINE_RATING, n_boot=n_boot, seed=seed)}
            for t, label in tg.items()}


def temporal_block(pm_tr: pd.DataFrame, pm_te: pd.DataFrame, games: pd.DataFrame, has_xg: bool,
                   tg: dict, n_boot: int, seed: int) -> dict:
    shr = SHRUNK_METRICS + (XG_SHRUNK_METRICS if has_xg else [])
    params = {m: bl.fit_role_shrinkage(bl.metric_frame(pm_tr, bl.METRICS[m])) for m in shr}
    fns = rating_fns(has_xg, params)
    return {t: {"label": label, **va.temporal_check(pm_te, bl.metric_frame(pm_te, bl.METRICS[t]), games, fns,
                                                    baseline=BASELINE_RATING, n_boot=n_boot, seed=seed)}
            for t, label in tg.items()}


def clean(o):
    """JSON без NaN/inf и numpy-типов."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def provenance(args, has_xg: bool, rows: dict) -> dict:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             check=False).stdout.strip()
    except OSError:
        rev = ""
    return {"cache_dir": str(args.cache_dir), "xg_file": str(args.xg) if args.xg else None,
            "xg_mode": "per-shot xG" if has_xg else "fallback: попытки бросков вместо xG",
            "seed": args.seed, "n_boot": args.n_boot, "git_head": rev, "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__, "rows": rows,
            "constants": {"MIN_TOI_HALF_S": va.MIN_TOI_HALF_S, "MIN_TOI_PRED_S": va.MIN_TOI_PRED_S,
                          "MIN_TOI_TEMPORAL_S": va.MIN_TOI_TEMPORAL_S, "MIN_TOI_SHRINK_S": bl.MIN_TOI_SHRINK_S,
                          "CI_LEVEL": va.CI_LEVEL}}


# --- markdown -------------------------------------------------------------------------------

def _f(x, nd: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def _ci(c) -> str:
    return f"[{_f(c[0])}; {_f(c[1])}]"


def md_reliability(rel: dict) -> list[str]:
    lines = ["## Устойчивость: split-half (train, случайные половины матчей, Spearman–Brown)", "",
             "| показатель | n | Pearson r_SB [CI матчи] | Spearman r_SB [CI матчи] | Spearman r_SB CI игроки "
             "| C | W | D |",
             "|:--|--:|:--|:--|:--|--:|--:|--:|"]
    for r in rel.values():
        p, g = r["pooled"], r["groups"]
        pp = r["players_bootstrap"]["pooled"]
        roles = " | ".join(_f(g.get(x, {}).get("spearman_sb")) for x in bl.FIELD_ROLES)
        lines.append(f"| {r['label']} | {p['n']} | {_f(p['pearson_sb'])} {_ci(p['pearson_sb_ci'])} | "
                     f"{_f(p['spearman_sb'])} {_ci(p['spearman_sb_ci'])} | {_ci(pp['spearman_sb_ci'])} | {roles} |")
    return lines + ["", "Объединённо — после центрирования внутри амплуа; C/W/D — Spearman r_SB. CI матчей "
                    "сдвинут на смещение бутстрэпа (r̂ − медиана r*); CI игроков — стратифицирован по амплуа.", ""]


def md_predictive(block: dict, title: str, key: str) -> list[str]:
    lines = [f"## {title}", ""]
    for b in block.values():
        res = b["directions"]["A->B"] if key == "directions" else b["result"]
        lines += [f"### Цель: {b['label']} (n = {res['n_players']})", "",
                  "| рейтинг | Spearman [95% CI] | wPearson [95% CI] | Δ Spearman vs очки/60 [95% CI] |",
                  "|:--|:--|:--|:--|"]
        for n, r in res["ratings"].items():
            p = r["pooled"]
            d = res["vs_baseline"].get(n, {}).get("pooled")
            dd = f"{_f(d['spearman_diff'])} {_ci(d['spearman_diff_ci'])}" if d else "база"
            lines.append(f"| {n} | {_f(p['spearman'])} {_ci(p['spearman_ci'])} | "
                         f"{_f(p['wpearson'])} {_ci(p['wpearson_ci'])} | {dd} |")
        lines.append("")
    return lines


def write_outputs(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = clean(report)
    (out_dir / "baseline_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["# Базовые строки: устойчивость и прогноз будущего", "",
          f"Режим xG: **{rep['provenance']['xg_mode']}**. Seed {rep['provenance']['seed']}, "
          f"бутстрэп {rep['provenance']['n_boot']}.", ""]
    md += md_reliability(rep["reliability_train"])
    md += md_predictive(rep["predictive_train"], "Прогноз: train, половина A → половина B "
                        "(бутстрэп игроков, объединённо по амплуа)", "directions")
    md += md_predictive(rep["temporal_test"], "Хронология test (диагностика): первая половина сезона → "
                        "вторая", "result")
    (out_dir / "baseline_report.md").write_text("\n".join(md), encoding="utf-8")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--xg", type=Path, default=None, help="parquet: match_id, seq, xg")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-boot", type=int, default=va.N_BOOT)
    ap.add_argument("--seed", type=int, default=va.SEED)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.cache_dir is None:
        args.cache_dir = current_l0_cache_dir()
    t0 = time.time()
    xg = load_xg(args.xg)
    has_xg = xg is not None
    roles = bl.player_roles(pd.read_parquet(args.cache_dir / "players.parquet", columns=["player_id", "role"]))
    pm_tr = bl.load_player_match(args.cache_dir, "train", roles, xg)
    pm_te = bl.load_player_match(args.cache_dir, "test", roles, xg)
    games = pd.read_parquet(args.cache_dir / "test_games.parquet", columns=["match_id", "game_seq"])
    tg = targets(has_xg)
    raw = RAW_METRICS + (XG_RAW_METRICS if has_xg else [])
    report = {
        "reliability_train": reliability_block(pm_tr, raw, args.n_boot, args.seed),
        "predictive_train": predictive_block(pm_tr, rating_fns(has_xg), tg, args.n_boot, args.seed),
        "temporal_test": temporal_block(pm_tr, pm_te, games, has_xg, tg, args.n_boot, args.seed),
    }
    rows = {"train_player_match": len(pm_tr), "test_player_match": len(pm_te),
            "train_players": int(pm_tr.player_id.nunique()), "test_players": int(pm_te.player_id.nunique())}
    report["provenance"] = provenance(args, has_xg, rows)
    report["provenance"]["runtime_s"] = round(time.time() - t0, 1)
    report["provenance"]["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    write_outputs(report, args.out_dir)
    print(f"готово за {report['provenance']['runtime_s']} с; пик RSS {report['provenance']['max_rss_mb']} МБ; "
          f"{args.out_dir / 'baseline_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
