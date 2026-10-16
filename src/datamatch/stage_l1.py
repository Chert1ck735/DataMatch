"""Этап L1: выбор процедуры xG во внутренней валидации (5-fold по матчам на 80% train).

Порядок сравнения по PDF:
  logit_spline (базовая) -> LightGBM с монотонностью (принят, если logloss лучше за пределами
  бутстрэпа и калибровка не хуже) -> подбор гиперпараметров (Optuna против grid: то же качество
  за меньший бюджет) -> EBM против LightGBM (не хуже в пределах шума -> EBM как интерпретируемая)
  -> отдельные модели по манпауэру против единой (logloss лучше).
Внешний блок здесь не используется.
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd

from . import io, l1_audit, l1_xg
from .cache import (TRUNCATED_OBJECT, Cache, _json_default, atomic_write, code_hash, frame_hash, load_checked_json,
                    obj_hash, save_json_with_provenance)
from .config import Config, get_logger
from .gates import enforce, require_ok
from .registry import Registry
from .splits import matches_for, split_frame, splits_hash

L1_CODE = ("l1_xg", "stage_l1", "l1_audit", "l0_events", "l0_time", "l0_shifts", "l0_quality", "l0_zone_starts", "stage_l0", "io", "splits", "config", "cache", "gates")
SHOT_HASH_COLS = l1_xg.FEATURES + ["goal", "blocked", "match_id", "fold", "opp_team_id"]

ECE_TOL = 0.002   # «калибровка не хуже»: ECE кандидата не больше ECE базовой + 0,2 п.п.
# Гейт L1 требует Σ xG в пределах ±5% от голов на внешнем блоке. ECE по децилям смещение
# суммы не ловит, поэтому оно проверяется отдельным критерием приёма. Жёсткий порог ±2%
# был уже выборочного шума числа голов (~2,5%) и отклонял идеальную модель примерно в 40%
# случаев (аудит 06.10.2026): смещение признаётся, только если 95%-интервал Σ xG / Σ голов
# по бутстрэпу матчей не накрывает 1.
XG_SUM_RULE = "95% match-bootstrap interval of sum(xG)/sum(goals) covers 1"


def xg_sum_acceptable(interval: dict) -> bool:
    """Сумма xG согласована с голами: интервал оценён и накрывает 1."""
    return bool(l1_xg.bootstrap_estimated(interval) and interval["lo95"] <= 1.0 <= interval["hi95"])


def same_selection(saved: dict, published: dict) -> bool:
    """Выбор из кэша (pickle) совпадает с опубликованным JSON после той же сериализации numpy → JSON.

    Сравнение сырых obj_hash давало вечный промах кэша: np.int64 фолдов в pickle против int в JSON.
    """
    normalized = json.loads(json.dumps(saved, default=_json_default))
    return obj_hash(normalized) == obj_hash(published)


def prefer_optuna(tuning_comparison: dict, n_trials: int, n_grid: int) -> bool:
    """Optuna вместо сетки: не хуже сетки на поисковых фолдах и дешевле по бюджету.

    Решение принимается только по фолдам поиска: блок сравнения (фолды 3–4) остаётся
    независимым для проверки выбранной модели против эталона.
    """
    same_quality = l1_xg.bootstrap_estimated(tuning_comparison) and tuning_comparison["lo95"] <= 0
    return bool(same_quality and n_trials < n_grid)


def run_fingerprint(cfg, provenance, spec):
    """Numerical identity is separate from observational machine metadata."""
    import hashlib
    import importlib.metadata
    import os
    import platform
    from .config import PROJECT_ROOT
    packages = {p: importlib.metadata.version(p) for p in
                ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "catboost", "optuna", "interpret-core", "pyarrow")}
    identity = {"provenance": provenance, "features": spec["features"], "spec": spec, "seed": cfg.seed,
                "packages": packages, "python": platform.python_version(),
                "lock_sha256": hashlib.sha256((PROJECT_ROOT / "requirements.lock.txt").read_bytes()).hexdigest(),
                "model_threads": 4,
                "thread_environment": {v: os.environ.get(v) for v in
                                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    identity["id"] = obj_hash(identity)
    machine = {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count()}
    return identity, machine


def numerical_environment():
    from .config import load_config
    identity, _ = run_fingerprint(load_config(), {}, {"features": l1_xg.FEATURES})
    return identity["id"]


def ablation_decision(shots, without, full, cfg):
    """Retain audited PDF features unless removal shows stable material benefit."""
    y = shots.goal.to_numpy(dtype=float)
    intervals = [l1_xg.paired_bootstrap_loss(shots.match_id.to_numpy(),
                 l1_xg.logloss_vec(y, without), l1_xg.logloss_vec(y, full),
                 int(cfg["eval"]["paired_bootstrap_B"]), cfg.seed + offset)
                 for offset in (0, 1, 2)]
    margin = float(cfg["xg"].get("ablation_harm_margin", 0.0001))
    estimated = all(l1_xg.bootstrap_estimated(c) for c in intervals)
    harmful = estimated and all(c["hi95"] < -margin for c in intervals)
    return {"keep": not harmful, "intervals": intervals, "harm_margin": margin,
            "status": ('inconclusive_retained' if not estimated else
                       ("harmful" if harmful else ("supported" if all(c["lo95"] > 0 for c in intervals) else "borderline_retained")))}


def dev_shots(l0: dict, splits: dict, schemes) -> tuple[pd.DataFrame, dict]:
    f = split_frame(splits)
    f = f[(~f.outer) & f.scheme.isin(schemes)]
    fold = dict(zip(f.match_id, f.fold))
    nd = l0["train"]["nodes"]
    s = l1_xg.shot_table(nd[nd.match_id.isin(fold)])
    s["fold"] = s.match_id.map(fold)
    return s.reset_index(drop=True), fold


_OOF_CACHE = {"dir": None, "shots_hash": None, "force": False}


def oof(spec: dict, s: pd.DataFrame, seed: int,
        evaluation_folds=None) -> tuple[np.ndarray, float, list[dict]]:
    """Cross-validation for search, or one independent two-fold comparison block.

    Ключ кэша: спецификация, seed, хэш таблицы бросков (признаки, метки, порядок строк, фолды,
    команда соперника) и хэш кода L1. При --force кэш не читается, а перезаписывается.
    """
    import pickle
    sh = frame_hash(s, SHOT_HASH_COLS)
    prov = {"spec": json.dumps(spec, sort_keys=True, default=str), "seed": seed, "shots": sh,
            "code": code_hash(*L1_CODE), "evaluation_folds": evaluation_folds,
            "runtime": numerical_environment()}
    key = obj_hash(prov)
    d = _OOF_CACHE["dir"]
    f = d / f"oof_{key}.pkl" if d is not None else None
    if f is not None and f.exists() and not _OOF_CACHE["force"]:
        try:
            with open(f, "rb") as fh:
                saved_prov, p, sec, prs = pickle.load(fh)
        except TRUNCATED_OBJECT as ex:
            get_logger().warning("L1 oof: повреждённый кэш %s (%s) — пересчёт", f.name, type(ex).__name__)
        else:
            if saved_prov == prov and len(p) == len(s):
                return p, sec, prs
    p = np.full(len(s), np.nan)
    t0 = time.time()
    prs = []
    if evaluation_folds is None:
        partitions = [(s.fold != k, s.fold == k) for k in sorted(s.fold.unique())]
    else:
        # Both held-out folds stay invisible to every model fitted for comparison.
        va = s.fold.isin(evaluation_folds)
        partitions = [(~va, va)]
    for tr, va in partitions:
        if set(s.match_id[tr]) & set(s.match_id[va]):
            raise ValueError("a match crosses xG train/validation folds")
        proc = l1_xg.XGProcedure(spec, seed).fit(s[tr])
        p[va.values] = proc.predict(s[va])
        prs.append(l1_xg.probes(proc, s[tr]))
    sec = time.time() - t0
    if f is not None:
        def write(tmp):
            with open(tmp, "wb") as fh:
                pickle.dump((prov, p, sec, prs), fh)
        atomic_write(f, write)
    return p, sec, prs


def probe_ok(pr: dict) -> bool:
    """Require distance AND independent angle monotonicity, plus applicable time probes."""
    time_ok = pr.get("period_time_probe_status") == "not_applicable" or pr["period_time_no_jumps"] is True
    return bool(pr["dist_monotone_fixed_angle"] and pr["dist_monotone_nonincreasing"]
                and pr.get("angle_monotone_fixed_distance") is True and time_ok)


def probes_ok_all(prs) -> bool:
    prs = prs if isinstance(prs, list) else [prs]
    return bool(prs) and all(probe_ok(p) for p in prs)


def l1_provenance(cache: Cache, s: pd.DataFrame, splits: dict) -> dict:
    runtime, _ = run_fingerprint(cache.cfg, {}, {"features": l1_xg.FEATURES})
    return cache.provenance("l1", ("xg", "data", "eval", "seed"), L1_CODE,
                            shots=frame_hash(s, SHOT_HASH_COLS), splits=splits_hash(splits), runtime=runtime["id"])


def run_l1(cfg: Config, l0: dict, splits: dict, reg: Registry, cache: Cache, force: bool = False) -> dict:
    log = get_logger()
    out_f = cfg.paths.artifacts / "xg_selected.json"
    data_mode = l0.get("quality", {}).get("status", "ok")
    if data_mode not in ("ok", "stress"):
        from .gates import GateError
        raise GateError("L1 requires admitted L0 data")
    schemes = tuple(cfg["data"]["train_schemes"])
    s, _ = dev_shots(l0, splits, schemes)
    prov = l1_provenance(cache, s, splits)
    model_key = cache.key("l1_dev_model", ("xg", "data", "eval", "seed"), extra=obj_hash(prov), code=L1_CODE)
    if not force:
        sel = load_checked_json(out_f, prov)
        complete = cache.done(model_key) and (cache.root/model_key/"procedure.pkl").exists()
        if sel is not None and complete:
            model = cache.load_obj(model_key, "procedure")
            saved = cache.load_obj(model_key, "selection")
            complete = (model is not None and saved is not None and same_selection(saved, sel)
                        and obj_hash(model.spec) == obj_hash(sel["spec"]))
        if sel is not None and complete:
            require_ok(sel, "L1 xg_selected.json")
            log.info("L1: выбор xG актуален, загружен из %s", out_f.name)
            return sel
    cache.invalidate_done(model_key)
    _OOF_CACHE["dir"] = cfg.paths.cache / "l1_oof"
    _OOF_CACHE["dir"].mkdir(parents=True, exist_ok=True)
    _OOF_CACHE["shots_hash"] = prov["shots"]
    _OOF_CACHE["force"] = force
    reg.clear_layer("L1")
    y = s.goal.values.astype(float)
    blk = s.blocked.values
    mids = s.match_id.values
    tuning_folds = list(cfg["xg"].get("tuning_folds", [0, 1, 2]))
    evaluation_folds = sorted(int(k) for k in set(s.fold.unique()) - set(tuning_folds))
    if len(tuning_folds) < 2 or not evaluation_folds or not set(tuning_folds) <= set(s.fold.unique()):
        raise ValueError("xG needs at least two tuning folds and separate comparison folds")
    tuning = s[s.fold.isin(tuning_folds)].reset_index(drop=True)
    evaluated = s.fold.isin(evaluation_folds).to_numpy()
    # Hyperparameters AND feature decisions see only tuning folds. Comparison
    # predictions are obtained from models that see neither comparison fold.
    yt = tuning.goal.to_numpy(dtype=float)
    B = int(cfg["eval"]["paired_bootstrap_B"])
    seed = cfg.seed
    log.info("L1: бросков в разработке %d (голов %d, заблокированных %d); поиск %d, сравнение %d",
             len(s), int(y.sum()), int(blk.sum()), len(tuning), int(evaluated.sum()))
    res = {}
    preds = {}

    def evaluate(cid, spec):
        p, sec, prs = oof(spec, s, seed, evaluation_folds)
        preds[cid] = p
        m = l1_xg.metrics(y[evaluated], p[evaluated], blk[evaluated], cfg["xg"]["ece_bins"])
        m["fit_seconds"] = sec
        m["probes_folds"] = prs
        m["probe_ok"] = probes_ok_all(m["probes_folds"])
        m["xg_sum_interval"] = l1_xg.xg_sum_interval(mids[evaluated], y[evaluated], p[evaluated], B, seed)
        res[cid] = m
        log.info("L1 %s: logloss %.5f brier %.5f ece %.4f auc %.4f auc_unbl %.4f (%.0f c)", cid, m["logloss"],
                 m["brier"], m["ece"], m["auc_all"], m["auc_unblocked"], sec)
        return m

    def compare(a, b):
        return l1_xg.paired_bootstrap_loss(mids[evaluated], l1_xg.logloss_vec(y[evaluated], preds[a][evaluated]),
                                         l1_xg.logloss_vec(y[evaluated], preds[b][evaluated]), B, seed)

    # --- шаг 0: аудит происхождения признаков (независимость от отметки самого броска) -----
    dev = sorted(tuning.match_id.unique())
    rng = np.random.default_rng(seed)
    n_aud = min(int(cfg["xg"]["audit_matches"]), len(dev))
    aud_m = list(rng.choice(dev, n_aud, replace=False))
    tr = l0["train"]
    audit = l1_audit.shot_time_perturbation(tr["events"], tr["games"], tr["stints"], io.rink_geometry(cfg), cfg,
                                            aud_m, tuple(cfg["xg"]["audit_deltas"]))
    timing = l1_audit.pre_shot_timing_check(tr["nodes"], dev)
    feats = [f for f in l1_xg.FEATURES if f not in audit["time_dependent"]]
    reg.add("xg_feature_audit", "L1", "A", None, {"matches": n_aud, "deltas": cfg["xg"]["audit_deltas"]},
            {"audit": audit, "pre_shot_timing": timing}, {"node_rebuilds": audit.get("node_rebuilds")},
            "info", ("признаки, зависящие от отметки броска, исключены: " + ", ".join(audit["time_dependent"]))
            if audit["time_dependent"] else "все признаки независимы от собственной отметки броска")
    log.info("L1 аудит отметки броска: зависимые %s; t_ref AUC %s (собственная отметка %s)",
             audit["time_dependent"], timing["pre_shot_lag_auc"], timing["own_stamp_dt_auc"])

    # --- шаг 1: абляция спорных временных групп признаков ---------------------------------
    ref = {"family": "lgbm_mono", "params": dict(cfg["xg"]["lgbm"]), "features": feats}
    full_abl, _, _ = oof(ref, tuning, seed)
    reg.add("xg_ablation_full", "L1", "A", None, ref,
            {"oof_logloss": float(np.mean(l1_xg.logloss_vec(yt, full_abl)))}, {"folds": len(tuning_folds)}, "baseline",
            "эталон абляции: LightGBM по умолчанию со всеми признаками, прошедшими аудит")
    ablation = {}
    for gname, cols in l1_xg.DISPUTED.items():
        if not any(c in feats for c in cols):
            continue
        spec_a = dict(ref, features=[f for f in feats if f not in cols])
        without, _, _ = oof(spec_a, tuning, seed)
        decisions = ablation_decision(tuning, without, full_abl, cfg)
        keep = decisions["keep"]
        c = decisions["intervals"][0]
        ablation[gname] = {"columns": cols, "dlogloss_without_vs_with": c, **decisions}
        reg.add(f"xg_ablation_{gname}", "L1", "A", "xg_ablation_full", {"removed": cols, "model": "lgbm_mono default"},
                {"dlogloss_without_vs_with": c}, {"folds": len(tuning_folds)}, "accepted" if keep else "rejected",
                "допустимый признак оставлен: нет устойчивого вреда на нескольких сидах" if keep
                else "устойчивый вред группы выше заранее заданного порога — группа исключена")
    for gname, a in ablation.items():
        if not a["keep"]:
            feats = [f for f in feats if f not in a["columns"]]
    log.info("L1 абляция: %s; итоговые признаки %s", {k: v["keep"] for k, v in ablation.items()}, feats)

    specs = {"logit_spline": {"family": "logit_spline", "l2": 1.0, "features": feats}}
    evaluate("logit_spline", specs["logit_spline"])
    reg.add("xg_logit_spline", "L1", "A", None, specs["logit_spline"], _short(res["logit_spline"]),
            {"comparison_folds": evaluation_folds, "model_fits": 1, "fit_seconds": res["logit_spline"]["fit_seconds"]}, "baseline", "базовая модель PDF")
    current = "logit_spline"
    # `current` — предпочитаемая модель (её и сдаём), `benchmark` — сильнейшая по logloss
    # из принятых. Они расходятся, когда модель принята не по выигрышу, а по равенству
    # (EBM как интерпретируемая). Планка для следующих кандидатов — всегда `benchmark`:
    # сравнение с ослабленной моделью непереходно и пропускает кандидата, который
    # сильнейшую модель не обыгрывает.
    benchmark = "logit_spline"

    def xg_sum_ok(cid):
        return xg_sum_acceptable(res[cid]["xg_sum_interval"])

    # LightGBM с монотонностью (параметры по умолчанию из конфигурации)
    specs["lgbm_mono"] = {"family": "lgbm_mono", "params": dict(cfg["xg"]["lgbm"]), "features": feats}
    evaluate("lgbm_mono", specs["lgbm_mono"])
    c = compare("lgbm_mono", "logit_spline")
    ok = (l1_xg.bootstrap_estimated(c) and (c["hi95"] < 0)
          and (res["lgbm_mono"]["ece"] <= res["logit_spline"]["ece"] + ECE_TOL)
          and res["lgbm_mono"]["probe_ok"] and xg_sum_ok("lgbm_mono"))
    reg.add("xg_lgbm_mono", "L1", "A", "xg_logit_spline", specs["lgbm_mono"],
            {**_short(res["lgbm_mono"]), "dlogloss_vs_logit": c},
            {"comparison_folds": evaluation_folds, "model_fits": 1, "fit_seconds": res["lgbm_mono"]["fit_seconds"]}, "accepted" if ok else "rejected",
            "logloss лучше за пределами бутстрэпа, калибровка не хуже" if ok else
            "нет выигрыша logloss за пределами бутстрэпа или калибровка хуже")
    if ok:
        current = benchmark = "lgbm_mono"

    # Гиперпараметры: grid против Optuna (одна и та же цель — OOF logloss)
    grid = cfg["xg"]["grid"]
    grid_combos = list(itertools.product(grid["num_leaves"], grid["learning_rate"], grid["min_child_samples"]))
    t0 = time.time()
    grid_res = []
    for nl, lr, mcs in grid_combos:
        prm = dict(cfg["xg"]["lgbm"], num_leaves=nl, learning_rate=lr, min_child_samples=mcs)
        p, _, _ = oof({"family": "lgbm_mono", "params": prm, "features": feats}, tuning, seed)
        grid_res.append((float(np.mean(l1_xg.logloss_vec(yt, p))), prm, p))
        log.info("L1 grid: %d/%d, tuning logloss %.5f", len(grid_res), len(grid_combos), grid_res[-1][0])
    grid_sec = time.time() - t0
    gbest = min(grid_res, key=lambda r: r[0])
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        cache_p = {}

        def objective(trial):
            prm = dict(cfg["xg"]["lgbm"])
            prm.update(num_leaves=trial.suggest_int("num_leaves", 5, 63, log=True),
                       learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                       min_child_samples=trial.suggest_int("min_child_samples", 30, 400, log=True),
                       reg_lambda=trial.suggest_float("reg_lambda", 0.1, 30, log=True))
            p, _, _ = oof({"family": "lgbm_mono", "params": prm, "features": feats}, tuning, seed)
            v = float(np.mean(l1_xg.logloss_vec(yt, p)))
            cache_p[trial.number] = (v, prm, p)
            log.info("L1 Optuna: trial %d, tuning logloss %.5f", trial.number + 1, v)
            return v

        t0 = time.time()
        n_trials = int(cfg["xg"]["optuna_trials"])
        n_startup = int(cfg["xg"]["optuna_startup"])
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=n_startup))
        study.optimize(objective, n_trials=n_trials)
        opt_sec = time.time() - t0
        obest = cache_p[study.best_trial.number]
        evaluate("lgbm_grid", {"family": "lgbm_mono", "params": gbest[1], "features": feats})
        evaluate("lgbm_optuna", {"family": "lgbm_mono", "params": obest[1], "features": feats})
        c_go = compare("lgbm_optuna", "lgbm_grid")   # только для отчёта: блок сравнения в выборе не участвует
        c_go_tuning = l1_xg.paired_bootstrap_loss(tuning.match_id.to_numpy(), l1_xg.logloss_vec(yt, obest[2]),
                                                  l1_xg.logloss_vec(yt, gbest[2]), B, seed)
        optuna_ok = prefer_optuna(c_go_tuning, n_trials, len(grid_combos))
        reg.add("xg_hparam_grid", "L1", "A", None, {"grid": grid, "n_evals": len(grid_combos)},
                {"best_oof_logloss": gbest[0], "best_params": gbest[1]}, {"evals": len(grid_combos), "seconds": grid_sec},
                "baseline", "поиск по сетке — ориентир для Optuna")
        reg.add("xg_hparam_optuna", "L1", "A", "xg_hparam_grid",
                {"trials": n_trials, "startup_random_trials": n_startup, "adaptive_tpe_trials": max(n_trials - n_startup, 0),
                 "space": "num_leaves, lr, min_child, lambda"},
                {"best_oof_logloss": obest[0], "best_params": obest[1], "dlogloss_vs_grid_tuning": c_go_tuning,
                 "dlogloss_vs_grid_comparison_report_only": c_go},
                {"evals": n_trials, "seconds": opt_sec}, "accepted" if optuna_ok else "rejected",
                "то же качество (разница в пределах бутстрэпа или лучше) за меньший бюджет" if optuna_ok
                else "хуже сетки за пределами шума или не дешевле")
        tuned = obest if optuna_ok else gbest
    except Exception as ex:  # pragma: no cover
        log.warning("Optuna: ошибка %s", ex)
        reg.add("xg_hparam_optuna", "L1", "A", "xg_hparam_grid", {}, {}, {}, "error", str(ex))
        tuned = gbest
    specs["lgbm_tuned"] = {"family": "lgbm_mono", "params": tuned[1], "features": feats}
    evaluate("lgbm_tuned", specs["lgbm_tuned"])
    c_t = compare("lgbm_tuned", benchmark)
    tuned_ok = (l1_xg.bootstrap_estimated(c_t) and (c_t["hi95"] < 0)
                and (res["lgbm_tuned"]["ece"] <= res[benchmark]["ece"] + ECE_TOL)
                and res["lgbm_tuned"]["probe_ok"] and xg_sum_ok("lgbm_tuned"))
    reg.add("xg_lgbm_tuned", "L1", "A", f"xg_{benchmark}", specs["lgbm_tuned"],
            {**_short(res["lgbm_tuned"]), "dlogloss_vs_current": c_t}, {"comparison_folds": evaluation_folds, "model_fits": 1},
            "accepted" if tuned_ok else "rejected",
            "подобранные гиперпараметры лучше текущей модели за пределами бутстрэпа" if tuned_ok
            else "выигрыш подбора в пределах шума")
    if tuned_ok:
        current = benchmark = "lgbm_tuned"

    # EBM против лучшей LightGBM: не хуже в пределах шума -> EBM (интерпретируемая)
    specs["ebm"] = {"family": "ebm", "features": feats}
    try:
        evaluate("ebm", specs["ebm"])
        best_lgbm = benchmark if benchmark.startswith("lgbm") else "lgbm_mono"
        c_e = compare("ebm", best_lgbm)
        ebm_ok = (l1_xg.bootstrap_estimated(c_e) and c_e["lo95"] <= 0 and (res["ebm"]["ece"] <= res[best_lgbm]["ece"] + ECE_TOL)
                  and benchmark != "logit_spline" and res["ebm"]["probe_ok"] and xg_sum_ok("ebm"))
        reg.add("xg_ebm", "L1", "A", f"xg_{best_lgbm}", specs["ebm"], {**_short(res["ebm"]), "dlogloss_vs_lgbm": c_e},
                {"comparison_folds": evaluation_folds, "model_fits": 1, "fit_seconds": res["ebm"]["fit_seconds"]}, "accepted" if ebm_ok else "rejected",
                "не хуже LightGBM в пределах шума, пробы монотонности пройдены — берём интерпретируемую" if ebm_ok else
                ("LightGBM не принят против логистической — EBM не рассматривается как замена" if current == "logit_spline"
                 else ("пробы монотонности/времени периода не пройдены" if not res["ebm"]["probe_ok"]
                       else "хуже LightGBM за пределами бутстрэпа или хуже калибровка")))
        if ebm_ok:
            # EBM принята по равенству, а не по выигрышу: планка `benchmark` не смягчается,
            # иначе следующий кандидат сравнивался бы с заведомо более слабой моделью.
            current = "ebm"
    except Exception as ex:  # pragma: no cover
        log.warning("EBM: ошибка %s", ex)
        reg.add("xg_ebm", "L1", "A", "xg_lgbm_mono", specs["ebm"], {}, {}, "error", repr(ex))

    # CatBoost uses exactly the same tuning/comparison partition and acceptance
    # rule. Its grid is searched only on the three tuning folds.
    cat_grid = cfg["xg"]["catboost_grid"]
    cat_results = []
    t_cat = time.time()
    for depth, lr, l2 in itertools.product(cat_grid["depth"], cat_grid["learning_rate"], cat_grid["l2_leaf_reg"]):
        params = dict(cfg["xg"]["catboost"], depth=depth, learning_rate=lr, l2_leaf_reg=l2)
        spec = {"family": "catboost_mono", "params": params, "features": feats}
        p, _, _ = oof(spec, tuning, seed)
        cat_results.append((float(np.mean(l1_xg.logloss_vec(yt, p))), spec))
        log.info("L1 CatBoost search: %d, tuning logloss %.5f", len(cat_results), cat_results[-1][0])
    cat_loss, specs["catboost_mono"] = min(cat_results, key=lambda a: a[0])
    evaluate("catboost_mono", specs["catboost_mono"])
    c_cat = compare("catboost_mono", benchmark)
    cat_ok = (l1_xg.bootstrap_estimated(c_cat) and c_cat["hi95"] < 0
              and res["catboost_mono"]["ece"] <= res[benchmark]["ece"] + ECE_TOL
              and res["catboost_mono"]["probe_ok"] and xg_sum_ok("catboost_mono"))
    reg.add("xg_catboost_mono", "L1", "A", f"xg_{benchmark}", specs["catboost_mono"],
            {**_short(res["catboost_mono"]), "tuning_logloss": cat_loss, "dlogloss_vs_current": c_cat},
            {"search_evals": len(cat_results), "search_folds": len(tuning_folds), "seconds": time.time() - t_cat},
            "accepted" if cat_ok else "rejected",
            "независимый logloss лучше, калибровка не хуже, пробы пройдены" if cat_ok
            else "нет независимого выигрыша за пределами шума или калибровка/пробы хуже")
    if cat_ok:
        current = benchmark = "catboost_mono"

    # Отдельные модели по манпауэру для выбранного семейства
    spec_mp = dict(specs[current], by_manpower=True)
    evaluate("by_manpower", spec_mp)
    c_m = compare("by_manpower", current)
    mp_ok = (l1_xg.bootstrap_estimated(c_m) and c_m["hi95"] < 0
             and res["by_manpower"]["probe_ok"] and xg_sum_ok("by_manpower"))
    reg.add("xg_by_manpower", "L1", "A", f"xg_{current}", spec_mp, {**_short(res["by_manpower"]), "dlogloss_vs_unified": c_m},
            {"comparison_folds": evaluation_folds, "model_fits": 1, "fit_seconds": res["by_manpower"]["fit_seconds"]}, "accepted" if mp_ok else "rejected",
            "logloss лучше единой модели" if mp_ok else "выигрыш над единой моделью в пределах шума")
    final_spec = spec_mp if mp_ok else specs[current]
    final_name = "by_manpower" if mp_ok else current

    # A final-family ablation is descriptive. It must not change the feature set
    # after seeing comparison outcomes. It is still run on tuning matches only.
    final_abl, _, _ = oof(final_spec, tuning, seed)
    for gname, a in ablation.items():
        removed = dict(final_spec, features=[f for f in final_spec["features"] if f not in a["columns"]])
        if removed["features"] == final_spec["features"]:
            continue
        without, _, _ = oof(removed, tuning, seed)
        a["final_family_diagnostic"] = ablation_decision(tuning, without, final_abl, cfg)
        a["final_family_diagnostic"]["changes_selection"] = False

    # обязательный гейт L1: пробы итоговой модели (обученной на всей разработке) и всех фолдов
    proc = l1_xg.XGProcedure(final_spec, seed).fit(s)
    pr = l1_xg.probes(proc, s)
    gates = {"probes_final_model": probe_ok(pr), "probes_all_folds": bool(res[final_name]["probe_ok"]),
             "xg_sum_within_tolerance": xg_sum_ok(final_name)}
    sel = {"status": "pending", "data_mode": data_mode,
           "selection_chain": {"benchmark": benchmark, "preferred": current, "final": final_name,
                               "xg_sum_rule": XG_SUM_RULE,
                               "rule": "кандидат сравнивается с сильнейшей по logloss принятой моделью; "
                                       "модель, принятая по равенству (EBM), планку не смягчает"},
           "validation_scope": "development comparison; frozen outer assessment is separate", "selected": final_name, "spec": final_spec, "features": feats,
           "feature_provenance": {f: l1_xg.FEATURE_PROVENANCE[f] for f in feats},
           "feature_audit": audit, "pre_shot_timing": timing, "ablation": ablation,
           "inner_cv_metrics": _short(res[final_name]),
           "probes_dev_model": pr, "n_dev_shots": int(len(s)), "shot_sample": "все попытки, включая заблокированные; без штрафных бросков и периода 0",
           "all_candidates": {k: _short(v) for k, v in res.items()}}
    sel["validation_design"] = {"method": "disjoint_search_and_comparison", "tuning_folds": tuning_folds,
                                "comparison_folds": evaluation_folds, "n_tuning_shots": len(tuning),
                                "n_comparison_shots": int(evaluated.sum()),
                                "n_tuning_matches": int(tuning.match_id.nunique()),
                                "n_comparison_matches": int(s.loc[evaluated, "match_id"].nunique()),
                                "final_fit": "all development matches, frozen selected parameters",
                                "outer_block_used": False}
    sel["run_identity"], sel["machine"] = run_fingerprint(cfg, prov, final_spec)
    sel["strength_policy"] = "match cross-fitting; exact empirical integration of probabilities"
    sel = finalize_selection(out_f, sel, gates, prov)
    model_key = cache.key("l1_dev_model", ("xg", "data", "eval", "seed"), extra=obj_hash(prov), code=L1_CODE)
    cache.save_obj(model_key, "procedure", proc)
    cache.save_obj(model_key, "selection", sel)
    cache.mark_done(model_key)
    log.info("L1 выбрано: %s; пробы %s", final_name, pr)
    return sel


def finalize_selection(out_f, sel: dict, gates: dict, prov: dict) -> dict:
    """Гейт L1: при провале пишет диагностический отчёт со статусом failed (без provenance —
    повторно не загружается) и поднимает GateError; при успехе сохраняет выбор со статусом ok."""
    sel = dict(sel)
    sel["gate_l1"] = dict(gates, sum_xg_outer_block="проверяется однократно на внешнем блоке")
    # Диагностика провала идёт в отдельный файл: запись поверх out_f уничтожала бы ранее
    # принятый выбор, из которого L2 и L3 берут рецепт xG, и восстановить его неоткуда.
    enforce("L1", gates, out_f.with_suffix(".failed.json"), sel)
    sel["status"] = "ok"
    save_json_with_provenance(out_f, sel, prov)
    sel["provenance"] = prov
    return sel


def _short(m: dict) -> dict:
    return {k: v for k, v in m.items() if k not in ("calibration",)}
