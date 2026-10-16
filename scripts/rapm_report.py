"""RAPM (L4), этап 1: проверка правила приёма O2 вместо O1 на train (METHODOLOGY 4.3а).

Запуск из Solve после `datamatch.cli shrink` (кэши L0–L5 готовы, пересчёт не запускается):

    PYTHONPATH=src .venv/bin/python scripts/rapm_report.py [--n-boot 1000] [--seed S]

Результат: outputs/rapm_report.json и .md. Test — только диагностика наклона (ничего не выбирает).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import validation_report as vr

from datamatch import io
from datamatch import l4_rapm_data as rd
from datamatch import l4_rapm_eval as ev
from datamatch import l7_forecast as fc
from datamatch import rating_validation as rv
from datamatch import validation as va
from datamatch.cache import ConcurrentRunError
from datamatch.config import load_config, setup_logging


def build(cfg, cache, args) -> dict:
    tables = vr.load_pipeline_tables(cfg, cache)
    pm_tr, pm_te = tables["pm_train"], tables["pm_test"]
    rows = rd.load_rows(cfg, cache)
    shrink = dict(cfg["shrink"])
    rr = ev.RapmRatings(rows["train"], shrink, seed=args.seed)
    o1 = rv.our_rating_fns(shrink)["O1"]
    raw = partial(rv.raw_rating, metric="xgd_5v5_60")
    rel = ev.reliability(pm_tr, {"O2": rr.fn("O2"), "O1": o1, "onice_xgd_raw": raw,
                                 "prior_only": rr.fn("PRIOR"), "RAPM0": rr.fn("RAPM0")}, args.seed, args.n_boot)
    fns = {"O1": o1, "O2": rr.fn("O2"), "RAPM0": rr.fn("RAPM0"), "onice_xgd_raw": raw}
    fcast, frames = ev.forecast(pm_tr, fns, args.seed, n_boot=args.n_boot)
    slope = ev.slopes(frames, "O2", args.n_boot, args.seed)
    rob = ev.robustness(pm_tr, rr, args.seed)
    frame_t, info_t = fc.cross_split_frame(pm_tr, pm_te, {"O1": o1, "O2": rr.fn("O2")})
    coefs = fc.fit_coefs(frames, ["O1", "O2"])
    cross = {"diagnostic_only": True, **info_t, **fc.holdout_forecast(frame_t, coefs, args.n_boot, args.seed)}
    res = {"reliability": rel, "forecast": fcast, "slope": slope, "robustness": rob,
           "train_to_test_diagnostic": cross,
           "fits": {f"{len(k)} матчей": v["info"] for k, v in rr.memo.items()},
           "rows": {s: int(len(r)) for s, r in rows.items()}}
    res["verdict"] = ev.verdict(res)
    return res


def build_stage2(cfg, cache, args) -> dict:
    """Этап 2 (METHODOLOGY 4.3б): λ по ЭБ, prior внутри регрессии, без команд; 5 условий приёма."""
    tables = vr.load_pipeline_tables(cfg, cache)
    pm_tr, pm_te = tables["pm_train"], tables["pm_test"]
    rows = rd.load_rows(cfg, cache)
    shrink = dict(cfg["shrink"])
    er = ev.EbRatings(rows["train"], shrink, seed=args.seed)
    o1 = rv.our_rating_fns(shrink)["O1"]
    raw = partial(rv.raw_rating, metric="xgd_5v5_60")
    rel = ev.reliability(pm_tr, {"O2": er.fn("O2"), "O1": o1, "onice_xgd_raw": raw, "prior_only": er.fn("PRIOR"),
                                 "O2_lam_x0.5": er.fn("O2", 0.5), "O2_lam_x2": er.fn("O2", 2.0),
                                 "O2_teams": er.fn("O2", 1.0, True)}, args.seed, args.n_boot)
    fns = {"O1": o1, "O2": er.fn("O2"), "onice_xgd_raw": raw}
    fcast, frames = ev.forecast(pm_tr, fns, args.seed, n_boot=args.n_boot)
    slope = ev.slopes(frames, "O2", args.n_boot, args.seed)
    rob = ev.robustness(pm_tr, er, args.seed)
    spr = ev.spread(pm_tr, er)
    frame_t, info_t = fc.cross_split_frame(pm_tr, pm_te, {"O1": o1, "O2": er.fn("O2")})
    coefs = fc.fit_coefs(frames, ["O1", "O2"])
    cross = {"diagnostic_only": True, **info_t, **fc.holdout_forecast(frame_t, coefs, args.n_boot, args.seed)}
    res = {"stage": 2, "reliability": rel, "forecast": fcast, "slope": slope, "robustness": rob, "spread": spr,
           "train_to_test_diagnostic": cross,
           "fits": {f"{len(k[0])} матчей, λ×{k[1]:g}{', команды' if k[2] else ''}": v["info"]
                    for k, v in er.memo.items()},
           "rows": {s: int(len(r)) for s, r in rows.items()}}
    res["verdict"] = ev.verdict(res)
    return res


def md(rep: dict) -> str:
    f = vr._f
    v = rep["verdict"]
    yes = {True: "да", False: "нет"}
    title = "этап 2 (λ по ЭБ, METHODOLOGY 4.3б)" if rep.get("stage") == 2 else "этап 1"
    lines = [f"# RAPM (L4), {title}: правило приёма O2 вместо O1", "",
             f"**Итог: {'O2 принят' if v['accepted'] else 'O2 не принят — итог остаётся O1, O2 — диагностика'}.**", ""]
    lines += [f"- {k}: **{yes[x]}**" for k, x in v["conditions"].items()] + ["", v["rule"], ""]
    rel = rep["reliability"]
    lines += [f"## 1. Split-half надёжность (игроки с ≥ {rel['min_games_half']} матчами в каждой половине, n = {rel['n_players']})", "",
              "| рейтинг | r_SB [95% ДИ] | O2 − X [95% ДИ] |", "|:--|:--|:--|"]
    for n, r in rel["ratings"].items():
        d = rel["vs"].get(n)
        lines.append(f"| {n} | {f(r['r_sb'])} {vr._ci(r['r_sb_ci'])} | "
                     + (f"{f(d['diff'])} {vr._ci(d['diff_ci'])}" if d else "ориентир") + " |")
    fcst = rep["forecast"]
    lines += ["", f"## 2. Прогноз (общий слой L7, направления объединены, n = {fcst['n_players']} игроков)", "",
              f"ΔSS = SS(O2) − SS(O1) = **{f(fcst['delta_ss'], 4)}** {vr._ci(fcst['delta_ss_ci'])}", "",
              "| рейтинг | SS |", "|:--|--:|"] + [f"| {n} | {f(s, 4)} |" for n, s in fcst["ss_all"].items()]
    lines += ["", "## 3. Наклон слоя для O2", "", "| направление | b [95% ДИ] | n |", "|:--|:--|--:|"]
    lines += [f"| {d} | {f(s['b'], 3)} {vr._ci(s['b_ci'])} | {s['n']} |" for d, s in rep["slope"].items()]
    rb = rep["robustness"]
    lines += ["", f"## 4. Устойчивость: Спирмен O2 (весь train) и O2 (без {rb['dropped_matches']} матчей) = "
              f"**{f(rb['spearman'], 3)}** (n = {rb['n_players']}, порог 0,9)", ""]
    ct = rep["train_to_test_diagnostic"]
    lines += [f"## Диагностика train → test (n = {ct['n_used']}): наклон O2 train {f(ct['train_slope'].get('O2'), 3)} / "
              f"test {f(ct['holdout_slope'].get('O2'), 3)}; SS O2 {f(ct['metrics']['O2']['pooled']['ss'], 3)}, "
              f"O1 {f(ct['metrics']['O1']['pooled']['ss'], 3)}", "", "## Подгонки", ""]
    if rep.get("stage") == 2:
        sp = rep["spread"]
        lines += [f"## 5. Разброс: sd(O2) / sd(prior) = **{f(sp['ratio'], 3)}** (n = {sp['n_players']}, "
                  f"порог {sp['min_ratio']})", "", "## Подгонки (σ², τ², λ)", "",
                  "| часть | σ² пуассон / остатки | τ²_O / τ²_D | λ_O / λ_D, ч | сошлось | итераций (из них след) |",
                  "|:--|:--|:--|:--|:--|--:|"]
        for k, i in rep["fits"].items():
            lines.append(f"| {k} | {f(i['sigma2_poisson'], 3)} / {f(i['sigma2_resid'], 3)} | "
                         f"{f(i['tau2']['o'], 4)} / {f(i['tau2']['d'], 4)} | {f(i['lambda']['o'], 1)} / "
                         f"{f(i['lambda']['d'], 1)} | {i['converged']} | {i['n_iter']} ({i.get('n_trace_iter', 0)}) |")
        return "\n".join(lines) + "\n"
    lams = sorted({float(x) for i in rep["fits"].values() for x in i["cv_zero"]})
    lines += ["Кривые кросс-валидации по целым матчам (взвешенный MSE xG/60 на отложенных матчах):", "",
              "| часть | кривая | " + " | ".join(f"λ={x:g}" for x in lams) + " |",
              "|:--|:--|" + "--:|" * len(lams)]
    for k, i in rep["fits"].items():
        for name in ("cv_zero", "cv_prior"):
            vals = {float(x): v for x, v in i[name].items()}
            lines.append(f"| {k} | {name} | " + " | ".join(f(vals.get(x), 3) for x in lams) + " |")
    lines += ["", "Кривые почти плоские: CV на уровне отрезков λ практически не определяет.", ""]
    for k, i in rep["fits"].items():
        lines.append(f"- {k}: λ к нулю {i['lambda_zero']}, λ к prior {i['lambda_prior']}, "
                     f"порог матчей для весов prior {i.get('prior_min_games')}, игроков {i['n_prior_eligible']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-boot", type=int, default=va.N_BOOT)
    ap.add_argument("--seed", type=int, default=va.SEED)
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    log = setup_logging(cfg, "rapm_report")
    cache = vr.ReadOnlyCache(cfg, io.data_version(cfg))
    t0 = time.time()
    try:
        with cache.run_lock():
            rep = build_stage2(cfg, cache, args) if args.stage == 2 else build(cfg, cache, args)
    except (vr.CachesNotReady, ConcurrentRunError) as ex:
        log.error("%s", ex)
        return 2
    rep["runtime_s"] = round(time.time() - t0, 1)
    out = Path(cfg.paths.outputs)
    stem = "rapm_report_stage2" if args.stage == 2 else "rapm_report"
    (out / f"{stem}.json").write_text(json.dumps(vr.clean(rep), ensure_ascii=False, indent=1), encoding="utf-8")
    (out / f"{stem}.md").write_text(md(vr.clean(rep)), encoding="utf-8")
    log.info("готово за %.0f с; O2 принят: %s", rep["runtime_s"], rep["verdict"]["accepted"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
