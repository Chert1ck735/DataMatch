"""Отчёт ветки Ф (Ф1 скорость с шайбой, Ф2 темп перехода и скорость шайбы) по кэшу L0.

Запуск из Solve:
    PYTHONPATH=src .venv/bin/python scripts/phys_report.py [--cache DIR] [--out DIR] [--n-boot N]

Пишет <out>/phys_profile_test.csv (строка на полевого игрока test) и <out>/phys_report.json
(надёжность, гейты, базовые строки, параметры стягивания, счётчики фильтров). Кэш только
читается (явные колонки, duel — фильтром pyarrow); данные никуда не передаются.

Протокол: надёжность и параметры стягивания (σ², ω², τ²_r) оцениваются на train (обучающая
часть для применения к test); μ_r для test — по матчам КХЛ-компоненты train (компонента связности
команд, в которую попадают команды test), как в правиле prior-средних L7; test-надёжность — только
диагностика. Личные показатели — профиль карточки, в Overall не входят.
"""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from datamatch import phys_profile as pp
from datamatch.splits import team_components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "outputs"
DEFAULT_N_BOOT = 200

NODE_COLS = ["match_id", "seq", "primary", "actor_id", "team_id", "other_id", "home_team_id", "period",
             "segment", "t_mono", "x", "y", "x2", "y2", "clock_imputed", "t_behind_scale", "chain_gap_before",
             "stint", "possession", "zone", "entry", "is_shot", "has_steal", "mp", "valid_for_transition"]
DUEL_COLS = ["match_id", "seq", "event_type", "actor_id", "actor_team_id", "home_team_id", "period", "segment",
             "t_mono", "x", "y", "clock_imputed", "t_behind_scale", "stint", "penalty_shot_seg"]
STINT_COLS = ["match_id", "stint", "home_sk", "away_sk"]

# Метрики: (спецификация, уровень при провале гейта). Ф1 при низкой надёжности остаётся на уровне
# амплуа; показатели Ф2 — на уровне звена (спецификация Ф2).
SPEED = pp.MetricSpec("speed_p80", agg="quantile", q=pp.SPEED_QUANTILE, min_obs_half=5)
PASS = pp.MetricSpec("pass_speed_median", agg="quantile", q=0.5, min_obs_half=5)
TEMPO = pp.MetricSpec("tempo_median_s", agg="quantile", q=0.5, min_obs_half=3)
SPEED_MIN_OBS_GRID = (5, 10, 20)
FALLBACK = {"speed_p80": "ROLE", "pass_speed_median": "LINE", "tempo_median_s": "LINE", "fast_share": "LINE"}



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

def fast_spec(k_s: float) -> pp.MetricSpec:
    return pp.MetricSpec(f"fast_share_k{k_s:g}", agg="mean", min_obs_half=5)


def load_part(cache: Path, part: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Узлы + события duel (единая схема узла) и составы stint одной части."""
    nodes = pq.read_table(cache / f"{part}_nodes.parquet", columns=NODE_COLS).to_pandas(
        split_blocks=True, self_destruct=True)
    duel = pq.read_table(cache / f"{part}_events.parquet", columns=DUEL_COLS,
                         filters=[("event_type", "==", "duel")]).to_pandas()
    duel = duel[(duel.period > 0) & ~duel.penalty_shot_seg.fillna(False).astype(bool)
                & duel.actor_id.notna() & duel.x.notna() & duel.y.notna()]
    duel = duel.drop(columns=["penalty_shot_seg"]).rename(columns={"event_type": "primary", "actor_team_id": "team_id"})
    ev = pd.concat([nodes, duel], ignore_index=True, sort=False)
    stints = pd.read_parquet(cache / f"{part}_stints.parquet", columns=STINT_COLS)
    return ev, stints


def load_roles(cache: Path) -> pd.Series:
    pl = pd.read_parquet(cache / "players.parquet", columns=["player_id", "role", "is_goalie"])
    pl = pl[pl.role.isin(pp.ROLES) & ~pl.is_goalie.fillna(False).astype(bool)]
    return pl.set_index("player_id")["role"]


def attach_skaters(df: pd.DataFrame, ev: pd.DataFrame, stints: pd.DataFrame, seq_col: str = "seq") -> pd.DataFrame:
    """Полевые своей команды на льду в момент события (match_id, seq_col) — для строки «звено»."""
    keys = ev[["match_id", "seq", "stint", "home_team_id"]].drop_duplicates(["match_id", "seq"])
    m = df[["match_id", seq_col, "team_id"]].merge(keys, left_on=["match_id", seq_col],
                                                   right_on=["match_id", "seq"], how="left")
    return df.assign(skaters=pp.onice_skaters(m, stints).to_numpy())


def build_part(ev: pd.DataFrame, stints: pd.DataFrame) -> dict:
    """Наблюдения всех метрик одной части и счётчики фильтров (пары хранятся без составов)."""
    kept, pair_counts = pp.filter_speed_pairs(pp.candidate_pairs(ev))
    speed_obs = pp.to_obs(attach_skaters(kept, ev, stints, "seq1"), "speed")
    kept = kept[["match_id", "player_id", "speed", "dt", "type1", "type2"]]
    passes, pass_counts = pp.pass_speed_obs(ev, return_counts=True)
    pass_obs = pp.to_obs(attach_skaters(passes, ev, stints), "speed")
    tr = attach_skaters(pp.transition_table(ev), ev, stints)
    tempo = tr.assign(value=tr.t_transition.where(tr.reached))
    return {"pairs": kept, "pair_counts": pair_counts, "pass_counts": pass_counts, "transitions": tr,
            "pass_speed_median_mps": float(passes.speed.median()),
            "obs": {"speed_p80": speed_obs, "pass_speed_median": pass_obs,
                    "tempo_median_s": pp.expand_onice(tempo, "value")}}


def speed_min_obs_sensitivity(obs: pd.DataFrame, roles: pd.Series) -> dict:
    """Надёжность p80 при разном минимуме пар в половине (диагностика; гейт — при 5, задан заранее)."""
    out = {}
    for m in SPEED_MIN_OBS_GRID:
        spec = pp.MetricSpec(SPEED.name, agg="quantile", q=pp.SPEED_QUANTILE, min_obs_half=m)
        rel = pp.reliability(obs, roles, spec, n_boot=0)
        out[str(m)] = {"split_half_sb": rel["split_half_sb"], "n_players": rel["n_players"]}
    return out


def fast_obs(tr: pd.DataFrame, k_s: float) -> pd.DataFrame:
    return pp.expand_onice(tr.assign(value=pp.fast_attack_values(tr, k_s)), "value")


def khl_train_matches(cache: Path) -> tuple[set, dict]:
    """Матчи train в компоненте связности команд, куда входит большинство команд test."""
    tr = pd.read_parquet(cache / "train_games.parquet", columns=["match_id", "home_team_id", "away_team_id"])
    te = pd.read_parquet(cache / "test_games.parquet", columns=["home_team_id", "away_team_id"])
    comp = team_components(tr)
    test_teams = pd.Series(pd.concat([te.home_team_id, te.away_team_id]).unique())
    known = test_teams.map(comp).dropna()
    if known.empty:
        raise ValueError("khl_train_matches: команды test не встречаются в train")
    cid = int(known.mode().iat[0])
    ids = set(tr.match_id[tr.home_team_id.map(comp) == cid])
    return ids, {"component": cid, "n_train_matches": len(ids),
                 "test_teams_in_component": int((known == cid).sum()), "test_teams": int(len(test_teams))}


def choose_fast_k(tr: pd.DataFrame, roles: pd.Series) -> tuple[float, dict]:
    """k доли быстрых атак — по максимуму split-half на train (внутренняя валидация)."""
    grid = {}
    for k in pp.FAST_ATTACK_K_GRID_S:
        grid[f"{k:g}"] = pp.reliability(fast_obs(tr, k), roles, fast_spec(k), n_boot=0)["split_half_sb"]
    best = max(pp.FAST_ATTACK_K_GRID_S, key=lambda k: np.nan_to_num(grid[f"{k:g}"], nan=-np.inf))
    return float(best), grid


def evaluate_metric(key: str, spec: pp.MetricSpec, obs_tr: pd.DataFrame, obs_te: pd.DataFrame,
                    roles: pd.Series, khl: set, n_boot: int) -> tuple[dict, pd.DataFrame]:
    """Надёжность (train — гейт, test — диагностика), стягивание test к μ_r КХЛ-компоненты."""
    rel_tr = pp.reliability(obs_tr, roles, spec, n_boot=n_boot)
    rel_te = pp.reliability(obs_te, roles, spec, n_boot=n_boot)
    params = pp.fit_shrinkage(obs_tr, roles, spec)
    khl_sum = pp.player_summary(obs_tr[obs_tr.match_id.isin(khl)], spec)
    khl_sum = khl_sum[khl_sum.n_obs >= spec.min_obs_half].assign(role=lambda d: d.player_id.map(roles))
    mu_khl = {r: float(khl_sum.loc[khl_sum.role == r, "value"].mean()) for r in pp.ROLES}
    params_test = {**params, "mu": {r: (mu_khl[r] if np.isfinite(mu_khl[r]) else params["mu"][r]) for r in pp.ROLES}}
    shrunk = pp.apply_shrinkage(pp.player_summary(obs_te, spec), roles, params_test)
    gate = rel_tr["gate_passed"]
    shrunk = shrunk.assign(label=shrunk.label if gate else FALLBACK[key])
    test_raw = shrunk[shrunk.n_obs >= spec.min_obs_half].groupby("role")["value"].mean().to_dict()
    report = {"spec": spec.name, "train": rel_tr, "test_diagnostic": rel_te, "shrinkage": params,
              "mu_test_prior_khl": params_test["mu"], "test_raw_role_mean": test_raw,
              "gate_passed_train": gate, "level": "player" if gate else FALLBACK[key].lower()}
    return report, shrunk


def role_speed_table(pairs: pd.DataFrame, roles: pd.Series) -> dict:
    """Типичные скорости пар по амплуа: медиана, p80, p95 (м/с)."""
    r = pairs.player_id.map(roles)
    out = {}
    for role in pp.ROLES:
        v = pairs.speed[r == role]
        out[role] = {"n_pairs": int(len(v)), "median": float(v.median()), "p80": float(v.quantile(0.8)),
                     "p95": float(v.quantile(0.95))}
    return out


def transition_detail(tr: pd.DataFrame) -> dict:
    """Счётчики переходов: всего, дошли до чужой зоны, оборваны наблюдением; медианы времени."""
    return {"recoveries_def_ev5": int(len(tr)), "reached": int(tr.reached.sum()),
            "censored": int(tr.censored.sum()), "median_t_transition_s": float(tr.t_transition.median()),
            "share_shot_within_10s": float(tr.t_first_shot.le(10.0).mean())}


def test_players(cache: Path, roles: pd.Series) -> pd.DataFrame:
    toi = pd.read_parquet(cache / "test_toi.parquet", columns=["match_id", "team_id", "player_id", "toi_s"])
    toi = toi[(toi.toi_s > 0) & toi.player_id.isin(roles.index)]
    g = toi.groupby("player_id")
    base = pd.DataFrame({"team_id": g.team_id.agg(lambda s: s.mode().iat[0]), "n_matches_test": g.match_id.nunique(),
                         "toi_min_test": g.toi_s.sum() / 60.0}).reset_index()
    return base.assign(role=base.player_id.map(roles))


def assemble_csv(players: pd.DataFrame, shrunk: dict[str, pd.DataFrame], names: dict[str, str]) -> pd.DataFrame:
    out = players.copy()
    for key, df in shrunk.items():
        col = names[key]
        part = df.set_index("player_id")[["n_obs", "value", "shrunk", "weight", "label"]]
        part.columns = [f"n_{col}", f"{col}_raw", col, f"{col}_weight", f"{col}_label"]
        out = out.join(part, on="player_id")
    for key in shrunk:
        col = names[key]
        out[f"n_{col}"] = out[f"n_{col}"].fillna(0).astype(int)
        out[f"{col}_label"] = out[f"{col}_label"].fillna("NO_DATA")
    out["reliability_label"] = out[[f"{names[k]}_label" for k in shrunk]].astype(str).agg(";".join, axis=1)
    return out.rename(columns={"n_speed_p80": "n_pairs"}).sort_values(["role", "player_id"]).reset_index(drop=True)


def constants() -> dict:
    names = ["PAIR_DT_MIN_S", "PAIR_DT_MAX_S", "SPEED_MAX_MPS", "SPEED_QUANTILE", "PASS_DT_MIN_S",
             "PASS_DT_MAX_S", "PASS_SPEED_MAX_MPS", "TRANSITION_STRENGTH", "FAST_ATTACK_K_GRID_S",
             "RELIABILITY_GATE", "MIN_SPLIT_PLAYERS", "MIN_ROLE_PLAYERS", "HALF_SEED", "CI_LEVEL"]
    return {n: getattr(pp, n) for n in names} | {"PAIR_PREV_TYPES": sorted(pp.PAIR_PREV_TYPES),
                                                  "PAIR_NEXT_TYPES": sorted(pp.PAIR_NEXT_TYPES)}


def run(cache: Path, out: Path, n_boot: int) -> dict:
    roles = load_roles(cache)
    parts = {}
    for part in ("train", "test"):
        ev, stints = load_part(cache, part)
        parts[part] = build_part(ev, stints)
        del ev, stints
    khl, khl_info = khl_train_matches(cache)
    k_best, k_grid = choose_fast_k(parts["train"]["transitions"], roles)
    metrics = {"speed_p80": SPEED, "pass_speed_median": PASS, "tempo_median_s": TEMPO, "fast_share": fast_spec(k_best)}
    obs = {p: {**parts[p]["obs"], "fast_share": fast_obs(parts[p]["transitions"], k_best)} for p in parts}
    reports, shrunk = {}, {}
    for key, spec in metrics.items():
        reports[key], shrunk[key] = evaluate_metric(key, spec, obs["train"][key], obs["test"][key], roles, khl, n_boot)
    names = {"speed_p80": "speed_p80", "pass_speed_median": "pass_speed_median",
             "tempo_median_s": "tempo_median_s", "fast_share": "fast_share"}
    csv = assemble_csv(test_players(cache, roles), shrunk, names)
    report = {
        "cache": cache.name, "n_boot": n_boot, "constants": constants(), "khl_prior": khl_info,
        "fast_attack_k_s": k_best, "fast_attack_k_grid_train_sb": k_grid,
        "counts": {p: {"speed_pairs": parts[p]["pair_counts"], "passes": parts[p]["pass_counts"],
                       "n_obs": {k: int(len(v)) for k, v in obs[p].items()}} for p in parts},
        "pair_speed_by_role": {p: role_speed_table(parts[p]["pairs"], roles) for p in parts},
        "pass_speed_median_mps": {p: parts[p]["pass_speed_median_mps"] for p in parts},
        "transition_detail": {p: transition_detail(parts[p]["transitions"]) for p in parts},
        "speed_confound_check": {p: pp.speed_confound_check(parts[p]["pairs"], roles) for p in parts},
        "speed_min_obs_sensitivity_train": speed_min_obs_sensitivity(obs["train"]["speed_p80"], roles),
        "metrics": reports, "test_csv_rows": int(len(csv)),
        "test_label_counts": {k: csv[f"{names[k]}_label"].value_counts().to_dict() for k in names},
    }
    out.mkdir(parents=True, exist_ok=True)
    csv.to_csv(out / "phys_profile_test.csv", index=False)
    (out / "phys_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return report


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (tuple, set, frozenset)):
        return list(o)
    raise TypeError(f"не сериализуется в JSON: {type(o)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=None, help="каталог кэша L0")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="каталог вывода")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT, help="повторов парного бутстрэпа по матчам")
    args = ap.parse_args(argv)
    if args.cache is None:
        args.cache = current_l0_cache_dir()
    if not (args.cache / "_DONE").exists():
        raise SystemExit(f"кэш L0 не завершён или не найден: {args.cache}")
    if args.n_boot < 0:
        raise SystemExit("--n-boot должен быть неотрицательным")
    rep = run(args.cache, args.out, args.n_boot)
    for key, m in rep["metrics"].items():
        tr = m["train"]
        print(f"{key}: train SB={tr['split_half_sb']:.3f} (n={tr['n_players']}), "
              f"gate={'OK' if m['gate_passed_train'] else 'FAIL'}, r_line={tr['r_line']:.3f}")
    print(f"written: {args.out / 'phys_profile_test.csv'} ({rep['test_csv_rows']} rows), {args.out / 'phys_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
