"""Отчёт «наш рейтинг (O1/O0) против базовых»: устойчивость и прогноз будущей игры (L7).

Запуск из Solve после завершения `datamatch.cli shrink`:

    PYTHONPATH=src .venv/bin/python scripts/validation_report.py [--n-boot 1000] [--seed S]

Читает ТОЛЬКО готовые кэши текущей конфигурации: L3 через `stage_l3.run_l3`, L5 через
`stage_l5.run_l5` (при актуальном кэше они только загружают таблицы), on-ice голы и `game_seq` —
из кэша L0 (`baselines.load_player_match`). Кэш обёрнут в `ReadOnlyCache`: любая попытка
пересчёта слоя (каждый пересчёт начинается с `invalidate_done`) прерывается `CachesNotReady`
до начала вычислений. Пока идёт другой прогон пайплайна (занят `cache/_run.lock`), отчёт не
строится. Результат: outputs/validation_report.json и .md.

Train: случайные половины матчей (seed), рейтинги — по половине A, цель — по B (и обратно).
Test: первая половина сезона «Авангарда» → вторая, модели обучены на train; только диагностика.
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
from pathlib import Path

import numpy as np
import pandas as pd

from datamatch import baselines as bl
from datamatch import io, stage_l0, stage_l3, stage_l5
from datamatch import l7_calibration as cal
from datamatch import l7_forecast as fc
from datamatch import rating_validation as rv
from datamatch import validation as va
from datamatch.cache import Cache, ConcurrentRunError
from datamatch.config import load_config, setup_logging
from datamatch.l6_uncertainty import role_moments

ROOT = Path(__file__).resolve().parents[1]
SECONDS_PER_MINUTE = 60.0
DEFAULT_MIN_TOI_MIN = va.MIN_TOI_PRED_S / SECONDS_PER_MINUTE
# Число повторов симуляции покрытия (как у refit-симуляции L6).
DEFAULT_N_SIM = 50
NOT_READY_HINT = "отчёт строится по готовым кэшам после `PYTHONPATH=src .venv/bin/python -m datamatch.cli shrink`"


class CachesNotReady(RuntimeError):
    """Кэш L0/L3/L5 текущей конфигурации не готов; пересчёт из отчёта запрещён."""


class ReadOnlyCache(Cache):
    """Кэш, в котором загрузчики слоёв могут читать, но не начинать пересчёт."""

    def invalidate_done(self, key: str):
        raise CachesNotReady(f"кэш {key} отсутствует или устарел для текущей конфигурации; {NOT_READY_HINT}")


# --- загрузка ------------------------------------------------------------------------------

def load_pipeline_tables(cfg, cache: Cache) -> dict:
    """player_match L3 (train/test) и результат L5 из готового кэша; иначе `CachesNotReady`."""
    try:
        l3 = stage_l3.run_l3(cfg, cache, force=False)
    except stage_l3.UpstreamNotReady as ex:
        raise CachesNotReady(f"{ex}; {NOT_READY_HINT}") from ex
    slim = {s: {"player_match": l3[s]["player_match"]} for s in ("train", "test")}
    slim["report"] = l3["report"]
    del l3  # журналы L3 отчёту не нужны
    l5 = stage_l5.run_l5(cfg, cache, slim, force=False)
    return {"pm_train": slim["train"]["player_match"], "pm_test": slim["test"]["player_match"],
            "l3_key": slim["report"].get("cache_key"), "l5": l5}


def l0_dir(cache: Cache) -> Path:
    """Каталог текущего кэша L0 (тот же ключ, что у `stage_l0.run_l0`)."""
    d = cache.root / cache.key("l0", ("l0",), code=stage_l0.L0_CODE)
    if not cache.done(d.name):
        raise CachesNotReady(f"L0: нет готового кэша {d.name}; {NOT_READY_HINT}")
    return d


def with_onice_goals(pm_l3: pd.DataFrame, d: Path, split: str) -> tuple[pd.DataFrame, dict]:
    """L3 player_match + on-ice голы и rate-согласованные очки из L0 (`baselines`)."""
    roles = stage_l5.player_roles(pm_l3)
    pm_l0 = bl.load_player_match(d, split, roles)
    return rv.attach_onice_goals(pm_l3, pm_l0)


# --- блоки отчёта ---------------------------------------------------------------------------

def with_team(fns: dict, pm: pd.DataFrame) -> dict:
    """Базовая строка «только команда» (если в таблице есть team_id)."""
    return fns | ({fc.TEAM_ONLY: fc.team_rating} if "team_id" in pm.columns else {})


def train_blocks(pm_tr: pd.DataFrame, shrink: dict, args) -> dict:
    fns = rv.our_rating_fns(shrink) | rv.baseline_rating_fns(pm_tr)
    min_toi = args.min_toi_min * SECONDS_PER_MINUTE
    rel = rv.rating_reliability(pm_tr, fns, args.seed, min_toi, args.n_boot)
    pred = rv.train_predictive(pm_tr, fns, args.seed, min_toi, args.n_boot)
    forecast = fc.train_forecast(pm_tr, with_team(fns, pm_tr), args.seed, min_toi, args.n_boot)
    return {"reliability_train": rel, "predictive_train": pred, "acceptance": rv.acceptance(rel, pred),
            "forecast_train": forecast}


def cross_split_block(pm_tr: pd.DataFrame, pm_te: pd.DataFrame, shrink: dict, coefs: dict, args) -> dict:
    """Train → test (≥ 5 матчей в обоих): рейтинги по всему train, коэффициенты слоя — из train."""
    fns = with_team(rv.our_rating_fns(shrink) | rv.baseline_rating_fns(pm_tr), pm_tr)
    frame, info = fc.cross_split_frame(pm_tr, pm_te, fns)
    return {"diagnostic_only": True, **info, **fc.holdout_forecast(frame, coefs, args.n_boot, args.seed)}


def calibration_block(pm_tr: pd.DataFrame, pm_te: pd.DataFrame, shrink: dict, l5: dict, args) -> dict:
    """Метки надёжности и покрытие: train-половины, train → test, двухэтапная симуляция."""
    hi, med = float(shrink["reliability_high"]), float(shrink["reliability_medium"])
    pairs = cal.half_pairs(pm_tr, shrink, args.seed)
    cross = cal.cross_split_pairs(pm_tr, pm_te, shrink)
    out = {"labels_train": cal.label_calibration(pairs, args.n_boot, args.seed, hi, med),
           "predictive_coverage_train": cal.predictive_coverage(pairs, 0.0, n_boot=args.n_boot, seed=args.seed),
           "predictive_coverage_train_to_test": {
               "diagnostic_only": True, "n_players": int(len(cross)),
               "grid": cal.coverage_grid(cross, n_boot=args.n_boot, seed=args.seed)}}
    mom = role_moments(l5.get("report") or {})
    if mom is None:
        out["latent_coverage_simulation"] = {"status": "not_run", "reason": "отчёт L5 не публикует моменты"}
        return out
    st_tr, st_te = cal.player_structure(pm_tr), cal.player_structure(pm_te)
    out["latent_coverage_simulation"] = cal.simulate_two_stage_coverage(
        st_tr.toi_min, st_tr.games, st_tr.role, st_te.toi_min, st_te.games, st_te.role, mom,
        n_sim=args.n_sim, seed=args.seed, min_toi_s=float(shrink["min_toi_split_min"]) * SECONDS_PER_MINUTE,
        eps_abs=float(shrink["eps_abs"]), high=hi, medium=med, noise_by_role=bool(shrink.get("noise_by_role", False)))
    return out


def temporal_block(pm_tr: pd.DataFrame, pm_te: pd.DataFrame, games: pd.DataFrame, shrink: dict,
                   l5: dict, args, coefs: dict | None = None) -> dict:
    """Хронология test с замороженными на train моделями; сверка O1 с точечным L5 test."""
    models = rv.fit_frozen_models(pm_tr, shrink)
    fns = rv.frozen_rating_fns(models) | rv.baseline_rating_fns(pm_tr, rv.baseline_shrink_params(pm_tr))
    o1_full = fns["O1"](pm_te)
    l5_o1 = l5["O1"]["test"].set_index("player_id").rating.reindex(o1_full.index)
    res = rv.temporal_predictive(pm_te, games, fns, min_toi_s=args.min_toi_min * SECONDS_PER_MINUTE,
                                 n_boot=args.n_boot, seed=args.seed)
    res["frozen_o1_vs_l5_test_max_abs_diff"] = float(np.nanmax(np.abs(o1_full - l5_o1)))
    res["mu_source"] = models.mu_source
    if coefs is not None:
        early, late = rv.temporal_halves(games)
        frame = fc.half_frame(pm_te, with_team(fns, pm_te), early, late, args.min_toi_min * SECONDS_PER_MINUTE)
        res["forecast"] = fc.holdout_forecast(frame, coefs, args.n_boot, args.seed)
    return res


def provenance(args, tables: dict, checks: dict, cache: Cache) -> dict:
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                         check=False).stdout.strip()
    rep5 = tables["l5"]["report"]
    return {"seed": args.seed, "n_boot": args.n_boot, "min_toi_min": args.min_toi_min, "git_head": rev,
            "l3_cache_key": tables["l3_key"], "l5_cache_key": rep5.get("cache_key"),
            "l5_rating_definition": rep5.get("rating_definition"), "l5_o1_usable": rep5.get("o1_usable"),
            "data_hash": cache.data_hash, "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "onice_goals_check": checks,
            "rows": {"train_player_match": len(tables["pm_train"]), "test_player_match": len(tables["pm_test"])},
            "scope": "все допустимые матчи train из L3 (разработка и внешний блок): только отчёт, без подбора"}


# --- вывод ----------------------------------------------------------------------------------

def clean(o):
    """JSON без NaN/inf и numpy-типов."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    return o


def _f(x, nd: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def _ci(c) -> str:
    return f"[{_f(c[0])}; {_f(c[1])}]"


def md_reliability(rel: dict) -> list[str]:
    lines = [f"## Устойчивость: split-half рейтингов (train, n = {rel['n_players']})", "",
             "| рейтинг | сырой r половин | Spearman r_SB [95% CI] | C | W | D | O1 − X, r_SB [95% CI] |",
             "|:--|--:|:--|--:|--:|--:|:--|"]
    for n, r in rel["ratings"].items():
        p, g = r["pooled"], r["by_role"]
        roles = " | ".join(_f(g.get(x, {}).get("spearman_sb")) for x in bl.FIELD_ROLES)
        d = rel["ref_minus"].get(n)
        dd = f"{_f(d['spearman_sb_diff'])} {_ci(d['spearman_sb_diff_ci'])}" if d else "ориентир"
        lines.append(f"| {n} | {_f(p.get('spearman'))} | {_f(p.get('spearman_sb'))} "
                     f"{_ci(p.get('spearman_sb_ci', [None, None]))} | {roles} | {dd} |")
    return lines + ["", "Каждый рейтинг подогнан отдельно на половинах A и B; CI — бутстрэп игроков "
                    "со стратификацией по амплуа (парный). r_SB = 2r/(1 + r) при r > 0, отрицательный r "
                    "обрезается до 0 — поэтому рядом показан сырой r (Спирмен половин, объединённо). "
                    "Правило вывода 7.3 использует только разность O1 − очки/60; остальные сравнения — "
                    "исследовательские.", ""]


def _pred_cell(res: dict, n: str) -> str:
    p = res["ratings"][n]["pooled"]
    d = res["ref_minus"].get(n, {}).get("pooled")
    dd = f"{_f(d['spearman_diff'])} {_ci(d['spearman_diff_ci'])}" if d else "ориентир"
    return f"{_f(p['spearman'])} {_ci(p['spearman_ci'])} | {_f(p['wpearson'])} {_ci(p['wpearson_ci'])} | {dd}"


def md_predictive(blocks: dict, title: str) -> list[str]:
    lines = [f"## {title}", "", "Исследовательское сравнение: правило вывода 7.3 использует отсюда только "
             "разность O1 − O0 (A → B); остальные строки и направления ничего не выбирают.", ""]
    for direction, block in blocks.items():
        for t, res in block["targets"].items():
            lines += [f"### {direction}: цель {res['label']} (n = {res['n_players']})", "",
                      "| рейтинг | Spearman [95% CI] | wPearson [95% CI] | O1 − X, Spearman [95% CI] |",
                      "|:--|:--|:--|:--|"]
            lines += [f"| {n} | {_pred_cell(res, n)} |" for n in res["ratings"]] + [""]
    return lines


def md_verdict(acc: dict) -> list[str]:
    yes = {True: "да", False: "нет"}
    return ["## Правило вывода (METHODOLOGY 7.3)", "", f"- надёжность O1 выше очков/60: "
            f"**{yes[acc['reliability_above_m0']]}**", f"- прогноз xGD/60 не хуже O0: "
            f"**{yes[acc['prediction_not_worse_than_o0']]}**", f"- O1 полезнее простых показателей: "
            f"**{yes[acc['o1_more_useful']]}**", "", acc["rule"], ""]


def _ss_cell(m: dict) -> str:
    return f"{_f(m.get('ss'), 3)} {_ci(m.get('ss_ci', [None, None]))}"


def md_forecast(train: dict, holdouts: dict) -> list[str]:
    dirs = list(train["directions"])
    lines = ["## Ошибка прогноза: общий линейный слой ŷ = a_амплуа + b·рейтинг (цель — on-ice xGD/60 5×5)", "",
             f"Train, случайные половины; коэффициенты — перекрёстно по игрокам ({train['k_folds']} фолдов), "
             "вес — будущий TOI. SS = 1 − MSE / MSE(только амплуа); CI — бутстрэп игроков.", "",
             "| рейтинг | " + " | ".join(f"SS {d} [95% CI]" for d in dirs) + f" | RMSE {dirs[0]} | "
             f"O1 − X, ΔSS {dirs[0]} [95% CI] |", "|:--|" + ":--|" * len(dirs) + "--:|:--|"]
    first = train["directions"][dirs[0]]
    for n in first["metrics"]:
        cells = " | ".join(_ss_cell(train["directions"][d]["metrics"][n]["pooled"]) for d in dirs)
        dd = first["o1_minus"].get(n)
        diff = f"{_f(dd['ss_diff'], 3)} {_ci(dd['ss_diff_ci'])}" if dd else "ориентир"
        mark = " (вырожден)" if n in train.get("degenerate", []) else ""
        lines.append(f"| {n}{mark} | {cells} | {_f(first['metrics'][n]['pooled']['rmse'], 3)} | {diff} |")
    if train.get("degenerate"):
        lines += ["", f"Вырождены (в каком-то амплуа одной из половин разброс < {fc.DEGENERATE_Z_SD}: рейтинг — "
                  f"константа, наклон слоя неустойчив): {', '.join(train['degenerate'])}. Их SS не интерпретируется."]
    lines.append("")
    for title, res in holdouts.items():
        if not res:
            continue
        lines += [f"### {title} (только диагностика, n = {res['n_players']})", "",
                  "| рейтинг | SS [95% CI] | RMSE | O1 − X, ΔSS [95% CI] | наклон train / здесь |",
                  "|:--|:--|--:|:--|:--|"]
        for n, m in res["metrics"].items():
            dd = res["o1_minus"].get(n)
            diff = f"{_f(dd['ss_diff'], 3)} {_ci(dd['ss_diff_ci'])}" if dd else "ориентир"
            slope = (f"{_f(res['train_slope'].get(n), 3)} / {_f(res['holdout_slope'].get(n), 3)}"
                     if n in res.get("train_slope", {}) else "—")
            lines.append(f"| {n} | {_ss_cell(m['pooled'])} | {_f(m['pooled']['rmse'], 3)} | {diff} | {slope} |")
        lines.append("")
    return lines


def md_labels(lab: dict) -> list[str]:
    yes = {True: "да", False: "нет"}
    lines = ["## Метки надёжности против реальной устойчивости (train, половина A → B и B → A)", "",
             "R = Σ t·(θ̂_A − μ)(x_B − μ)/τ² / Σ t — реализованная надёжность; при верной модели равна "
             "среднему w группы. Наклон x_B на θ̂_A: 1 — стягивание верное, < 1 — недостаточное, "
             "> 1 — избыточное. Вес — будущий TOI; CI — бутстрэп игроков.", "",
             "| группа | n | среднее w | R модели | R реализованная [95% CI] | наклон [95% CI] | диапазон метки | согласуется |",
             "|:--|--:|--:|--:|:--|:--|:--|:--|"]
    rows = [(g, r) for g, r in lab["by_label"].items()] + [("все", lab["pooled"])]
    for g, r in rows:
        rng_ = r.get("label_range")
        lines.append(f"| {g} | {r['n']} | {_f(r['mean_w'])} | {_f(r['model_reliability'])} | "
                     f"{_f(r['realized_reliability'])} {_ci(r['realized_reliability_ci'])} | "
                     f"{_f(r['calibration_slope'])} {_ci(r['calibration_slope_ci'])} | "
                     f"{_ci(rng_) if rng_ else '—'} | {yes.get(r.get('consistent'), '—')} |")
    lines += ["", "| интервал w | n | среднее w | R реализованная [95% CI] | наклон [95% CI] |",
              "|:--|--:|--:|:--|:--|"]
    for g, r in lab["by_w_bin"].items():
        lines.append(f"| {g} | {r['n']} | {_f(r['mean_w'])} | {_f(r['realized_reliability'])} "
                     f"{_ci(r['realized_reliability_ci'])} | {_f(r['calibration_slope'])} {_ci(r['calibration_slope_ci'])} |")
    return lines + [""]


def _cov_rows(groups: dict) -> list[str]:
    return [f"| {g} | {r['n']} | {_f(r['coverage'], 3)} | {_f(r['mean_halfwidth'])} |" for g, r in groups.items()]


def md_coverage(cov: dict) -> list[str]:
    tr = cov["predictive_coverage_train"]
    lines = ["## Покрытие интервалов", "",
             "### Предсказательный интервал на реальных данных train (половина A → B, δ² = 0)", "",
             f"θ̂_A ± 1,96·√(v_A + σ²/T_B + ω²/M_B) против ставки Overall x_B, номинал 0,95. "
             f"**Объединённо {_f(tr['pooled']['coverage'], 3)}** {_ci(tr['pooled']['coverage_ci'])} "
             f"(n = {tr['pooled']['n']}); по меткам — "
             + ", ".join(f"{g} {_f(tr['by_label'].get(g, {}).get('coverage'), 3)}" for g in ("HIGH", "MEDIUM", "LOW"))
             + "; по амплуа — "
             + ", ".join(f"{g} {_f(tr['by_role'].get(g, {}).get('coverage'), 3)}" for g in bl.FIELD_ROLES)
             + ".", "", "| группа | n | покрытие | средняя полуширина |", "|:--|--:|--:|--:|"]
    for key in ("by_label", "by_role", "by_future_toi_min"):
        lines += _cov_rows(tr[key])
    cross = cov["predictive_coverage_train_to_test"]
    lines += ["", f"### Train → test (только диагностика, n = {cross['n_players']}, ≥ 5 матчей в обоих)", "",
              "| δ² | покрытие [95% CI] | средняя полуширина |", "|:--|:--|--:|"]
    for d, r in cross["grid"].items():
        lines.append(f"| {d} | {_f(r['pooled']['coverage'], 3)} {_ci(r['pooled']['coverage_ci'])} | "
                     f"{_f(r['pooled']['mean_halfwidth'])} |")
    sim = cov["latent_coverage_simulation"]
    if sim.get("status") == "not_run":
        return lines + ["", f"Симуляция не выполнена: {sim['reason']}", ""]
    names = {"oracle_moments": "истинные моменты (оракул)",
             "refit_on_fit_structure": "моменты по выборке структуры train (как в конвейере)",
             "refit_on_eval_structure": "моменты по выборке структуры test (вариант отчёта L6)"}
    lines += ["", f"### Симуляция интервала скрытого θ (структура test, {sim['n_eval_players']} игроков; "
              f"train — {sim['n_fit_players']}; {sim['n_sim']} повторов)", "",
              "| оценка моментов | покрытие | HIGH | MEDIUM | LOW |", "|:--|--:|--:|--:|--:|"]
    for k, title in names.items():
        r = sim[k]
        labs = " | ".join(_f(r["by_label"].get(lab, {}).get("coverage"), 3) for lab in ("HIGH", "MEDIUM", "LOW"))
        lines.append(f"| {title} | {_f(r['coverage'], 3)} | {labs} |")
    return lines + ["", "Интервалы по этим результатам не расширяются (правило L6); test ничего не настраивает.", ""]


def write_outputs(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = clean(report)
    (out_dir / "validation_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    pv = rep["provenance"]
    md = ["# Наш рейтинг против базовых: устойчивость и прогноз будущего", "",
          f"Итог L5: **{pv['l5_rating_definition']}** (O1 допущен: {pv['l5_o1_usable']}). Seed {pv['seed']}, "
          f"бутстрэп {pv['n_boot']}, порог TOI {pv['min_toi_min']} мин в каждой части. Разности — «O1 − X».", ""]
    md += md_verdict(rep["acceptance"]) + md_reliability(rep["reliability_train"])
    pred = {k: v for k, v in rep["predictive_train"].items() if k in ("A->B", "B->A")}
    md += md_predictive(pred, "Прогноз: train, случайные половины матчей (рейтинг по одной, цель по другой)")
    md += md_predictive({"test": rep["temporal_test"]}, "Хронология test (только диагностика): первая "
                        "половина сезона → вторая, модели обучены на train")
    md += md_forecast(rep["forecast_train"], {"Train → test, ≥ 5 матчей в обоих": rep.get("forecast_train_to_test"),
                                              "Хронология «Авангарда»": rep["temporal_test"].get("forecast")})
    md += md_labels(rep["calibration"]["labels_train"]) + md_coverage(rep["calibration"])
    path = out_dir / "validation_report.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", type=Path, default=None, help="по умолчанию paths.outputs")
    ap.add_argument("--n-boot", type=int, default=va.N_BOOT)
    ap.add_argument("--seed", type=int, default=va.SEED)
    ap.add_argument("--min-toi-min", type=float, default=DEFAULT_MIN_TOI_MIN)
    ap.add_argument("--n-sim", type=int, default=DEFAULT_N_SIM, help="повторы симуляции покрытия")
    args = ap.parse_args(argv)
    if args.n_boot < 1 or args.min_toi_min < 0 or args.n_sim < 1:
        ap.error("--n-boot ≥ 1, --n-sim ≥ 1 и --min-toi-min ≥ 0")
    return args


def build_report(cfg, cache: Cache, args) -> dict:
    tables = load_pipeline_tables(cfg, cache)
    d = l0_dir(cache)
    pm_tr, chk_tr = with_onice_goals(tables["pm_train"], d, "train")
    pm_te, chk_te = with_onice_goals(tables["pm_test"], d, "test")
    games = pd.read_parquet(d / "test_games.parquet", columns=["match_id", "game_seq"])
    shrink = dict(cfg["shrink"])
    report = train_blocks(pm_tr, shrink, args)
    coefs = report["forecast_train"]["coefficients"]
    report["temporal_test"] = temporal_block(pm_tr, pm_te, games, shrink, tables["l5"], args, coefs)
    report["forecast_train_to_test"] = cross_split_block(pm_tr, pm_te, shrink, coefs, args)
    report["calibration"] = calibration_block(pm_tr, pm_te, shrink, tables["l5"], args)
    report["provenance"] = provenance(args, tables, {"train": chk_tr, "test": chk_te}, cache)
    return report


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    log = setup_logging(cfg, "validation_report")
    cache = ReadOnlyCache(cfg, io.data_version(cfg))
    t0 = time.time()
    try:
        with cache.run_lock():
            report = build_report(cfg, cache, args)
    except (CachesNotReady, ConcurrentRunError) as ex:
        log.error("%s", ex)
        return 2
    report["provenance"]["runtime_s"] = round(time.time() - t0, 1)
    report["provenance"]["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    path = write_outputs(report, args.out_dir or cfg.paths.outputs)
    log.info("готово за %.1f с; пик RSS %.0f МБ; %s", report["provenance"]["runtime_s"],
             report["provenance"]["max_rss_mb"], path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
