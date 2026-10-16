"""Карточка игрока, пункт 4: удаления — заработанные и полученные (METHODOLOGY 11.4). Вне O1.

Запуск из Solve (кэши L0/L2/L3/L5 только читаются):

    PYTHONPATH=src .venv/bin/python scripts/card_penalties.py [--n-boot 2000] [--n-draws 20000]

Всё обучаемое (моменты Пуассон–гамма по амплуа, ν) — по матчам разработки train. Выход:
outputs/card/penalties.csv и penalties_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import validation_report as vr

from datamatch import card_io, io
from datamatch import card_penalties as cp
from datamatch import card_situational as cs
from datamatch.config import load_config, setup_logging
from datamatch.l5_shrink import ROLES
from datamatch.splits import load_splits, matches_for

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "card"
DEFAULT_N_BOOT = 2000
DEFAULT_N_DRAWS = 20000
MIN_TOI_MIN = 60.0                 # «мало минут» (METHODOLOGY 11)
SPLIT_MIN_TOI_S = 60 * 60.0        # минимум минут в половине — как у проверок раздела 7.1 / 10.2
INF_COLS = ["match_id", "seq", "period", "segment", "n_home", "n_away", "event_type"]
KINDS = ("drawn", "taken")


def load_infractions(l0: Path, part: str) -> pd.DataFrame:
    return pq.read_table(l0 / f"{part}_events.parquet", columns=["match_id", "period", "actor_id", "other_id"],
                         filters=[("event_type", "==", "infraction")]).to_pandas()


def manpower_timing(l0: Path, matches: set) -> dict:
    """Где меняется численность после нарушения: в следующем отрезке (после свистка) или в том же."""
    ev = pq.read_table(l0 / "train_events.parquet", columns=INF_COLS).to_pandas()
    ev = ev[ev.match_id.isin(matches)].sort_values(["match_id", "seq"]).reset_index(drop=True)
    mp = ev.n_home.astype(float).fillna(-1).to_numpy() * 10 + ev.n_away.astype(float).fillna(-1).to_numpy()
    mid, seg, per = ev.match_id.to_numpy(), ev.segment.to_numpy(), ev.period.to_numpy()
    out = {"later_segment": 0, "same_segment": 0, "no_change_in_period": 0}
    for i in np.flatnonzero((ev.event_type == "infraction").to_numpy() & (per >= 1)):
        j = i + 1
        while j < len(ev) and mid[j] == mid[i] and per[j] == per[i] and mp[j] == mp[i]:
            j += 1
        if j >= len(ev) or mid[j] != mid[i] or per[j] != per[i]:
            out["no_change_in_period"] += 1
        else:
            out["later_segment" if seg[j] != seg[i] else "same_segment"] += 1
    return out


def nu_block(l0: Path, cache, dev: set, inf_dev: pd.DataFrame, n_boot: int, seed: int,
             key: str) -> tuple[dict, np.ndarray]:
    """ν — чистые xG (и голы) большинства на одно нарушение по матчам разработки; r — OOF L2 (`key` — модель L2
    сдаваемого L3)."""
    nodes = pq.read_table(l0 / "train_nodes.parquet", columns=["match_id", "node", "mp", "is_goal"]).to_pandas()
    nodes = nodes[nodes.match_id.isin(dev)]
    if not cache.done(key):
        raise card_io.CardInputMismatch(f"нет готового кэша L2 {key}, на котором построен сдаваемый L3")
    # `node` в train_values — номер узла выборки L2; узел L0 — `source_node` (как в stage_l3.align_values)
    vals = cache.load_df(key, "train_values")[["match_id", "source_node", "r"]].rename(columns={"source_node": "node"})
    vals = vals[vals.match_id.isin(dev)]
    nodes = nodes.merge(vals, on=["match_id", "node"], how="inner")
    if len(nodes) != len(vals):
        raise card_io.CardInputMismatch(f"узлы L0 и значения L2 не совпали: {len(nodes)} из {len(vals)}")
    n_inf = inf_dev[inf_dev.period.astype(int) >= 1].groupby("match_id").size()
    per = cp.pp_net_by_match(nodes, n_inf.reindex(sorted(dev), fill_value=0))
    draws = cp.bootstrap_ratio(per, "net_xg", n_boot, seed)
    gdraws = cp.bootstrap_ratio(per, "net_goals", n_boot, seed + 1)
    rep = {"nu_xg": cp.ratio_of_sums(per, "net_xg"), "nu_xg_ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
           "nu_goals": cp.ratio_of_sums(per, "net_goals"), "nu_goals_ci95": np.percentile(gdraws, [2.5, 97.5]).tolist(),
           "n_matches": int(len(per)), "n_infractions": int(per.n_inf.sum()), "l2_values_cache": key,
           "definition": "Σ r узлов PP − Σ r узлов SH (r — OOF xG L2) / число нарушений; ENA/ENF не входят"}
    return rep, draws


def player_rows(pm: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Игрок × матч с TOI (все ситуации) и числом нарушений; нарушения без строки TOI не входят."""
    f = pm[["match_id", "player_id", "role", "toi_s"]].merge(counts, on=["match_id", "player_id"], how="left")
    return f.fillna({"taken": 0, "drawn": 0})


def per_player(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.groupby("player_id").agg(role=("role", "first"), toi_s=("toi_s", "sum"),
                                         taken=("taken", "sum"), drawn=("drawn", "sum")).reset_index()


def fit_moments(rows: pd.DataFrame) -> dict:
    """Prior по амплуа — ML маргинального NB-правдоподобия на разработке (METHODOLOGY 11.4, поправка 08.10)."""
    g = per_player(rows)
    return {kind: {r: cp.gamma_mle(g.loc[g.role == r, kind].to_numpy(), g.loc[g.role == r, "toi_s"].to_numpy() / 3600)
                   for r in ROLES} for kind in KINDS}


def observed_share(inf: pd.DataFrame) -> dict:
    """Доля нарушений (без буллитов) с известным участником: drawn — other_id, taken — actor_id."""
    e = inf[inf.period.astype(int) != cp.SHOOTOUT_PERIOD]
    return {"drawn": float(e.other_id.notna().mean()), "taken": float(e.actor_id.notna().mean())}


def half_rates(rows_dev: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Частоты вида на час по половинам (матчи игрока через один), игроки с ≥ SPLIT_MIN_TOI_S в каждой."""
    h = rows_dev.sort_values(["player_id", "match_id"]).assign(half=lambda d: d.groupby("player_id").cumcount() % 2)
    t = h.groupby(["player_id", "half"]).agg(n=(kind, "sum"), toi_s=("toi_s", "sum"), role=("role", "first")).unstack("half")
    t = t[(t[("toi_s", 0)] >= SPLIT_MIN_TOI_S) & (t[("toi_s", 1)] >= SPLIT_MIN_TOI_S)]
    return pd.DataFrame({"role": t[("role", 0)], "xa": t[("n", 0)] / (t[("toi_s", 0)] / 3600),
                         "xb": t[("n", 1)] / (t[("toi_s", 1)] / 3600)})


def acceptance(rows_dev: pd.DataFrame, mom: dict, n_boot: int, seed: int) -> dict:
    """Критерий приёма стягивания (задан до расчёта): обещанная моделью r половин ∈ 95%-ДИ фактической."""
    out = {}
    for i, kind in enumerate(KINDS):
        hr = half_rates(rows_dev, kind)
        obs = cp.half_r_ci(hr.xa.to_numpy(), hr.xb.to_numpy(), hr.role.to_numpy(), n_boot, seed + 10 + i)
        no_signal = any(mom[kind][r]["no_signal"] for r in ROLES)
        implied = float("nan") if no_signal else implied_half_r(rows_dev, mom, kind)
        out[kind] = {"observed": obs, "model_implied_half_r": implied, "no_signal_any_role": no_signal,
                     "accepted": bool(np.isfinite(implied) and obs["lo"] <= implied <= obs["hi"])}
    return out


def _posterior(g: pd.DataFrame, kind: str, mom: dict, T: np.ndarray) -> tuple:
    mean, w = np.zeros(len(g)), np.zeros(len(g))
    shape, rate = np.full(len(g), np.nan), np.full(len(g), np.nan)
    for r in ROLES:
        m = (g.role == r).to_numpy()
        mo = mom[kind][r]
        n = g[kind].to_numpy(dtype=float)[m]
        mean[m], w[m] = cp.gamma_posterior(n, T[m], mo["alpha"], mo["beta"], mo["m"])
        if not mo["no_signal"]:
            shape[m], rate[m] = n + mo["alpha"], T[m] + mo["beta"]
    return mean, w, shape, rate


def apply_test(rows: pd.DataFrame, mom: dict, accepted: dict, share: dict, nu_hat: float, nu_draws: np.ndarray,
               cfg, n_draws: int, seed: int) -> pd.DataFrame:
    """Частоты на 60 с поправкой на пропуски участника; стягивание и xG/60 — только для принятых видов.

    mom — prior разработки, уже перенесённый на долю наблюдаемых участников test; share — эта доля в test.
    """
    g = per_player(rows)
    T = g.toi_s.to_numpy() / 3600
    out = g.assign(toi_min=g.toi_s / 60)
    rng = np.random.default_rng(seed)
    sim = {}
    for kind in KINDS:
        q = share[kind]
        raw = np.divide(g[kind].to_numpy(dtype=float), T, out=np.full(len(g), np.nan), where=T > 0)
        out[f"{kind}_per60_raw"] = raw / q
        if not accepted[kind]:
            out[f"{kind}_per60"], out[f"w_{kind}"] = raw / q, np.nan
            continue
        mean, w, shape, rate = _posterior(g, kind, mom, T)
        out[f"{kind}_per60"], out[f"w_{kind}"] = mean / q, w
        fixed = np.isnan(shape)
        draw = rng.gamma(np.where(fixed, 1.0, shape), 1.0 / np.where(fixed, 1.0, rate), (n_draws, len(g)))
        sim[kind] = np.where(fixed[None, :], mean[None, :], draw) / q
    out["net_per60"] = out.drawn_per60 - out.taken_per60
    both = all(accepted[k] for k in KINDS)
    if both:
        nu = nu_draws[rng.integers(0, len(nu_draws), n_draws)][:, None]
        value = (sim["drawn"] - sim["taken"]) * nu
        out["penalty_xg60"] = out.net_per60 * nu_hat
        out["penalty_xg60_lo95"], out["penalty_xg60_hi95"] = np.percentile(value, [2.5, 97.5], axis=0)
        sh = cfg["shrink"]
        lab = cs.display_label(np.minimum(out.w_drawn, out.w_taken).to_numpy(), out.toi_min.to_numpy(),
                               float(sh["reliability_high"]), float(sh["reliability_medium"]), MIN_TOI_MIN)
        return out.assign(reliability=lab.label.to_numpy(), flag=lab.flag.to_numpy())
    rejected = ", ".join(k for k in KINDS if not accepted[k])
    few = out.toi_min.to_numpy() < MIN_TOI_MIN
    flag = np.where(few, f"стягивание не принято ({rejected}); {cs.LOW_MINUTES}", f"стягивание не принято ({rejected})")
    return out.assign(penalty_xg60=np.nan, penalty_xg60_lo95=np.nan, penalty_xg60_hi95=np.nan,
                      reliability="описательно", flag=flag)


def implied_half_r(rows_dev: pd.DataFrame, mom: dict, kind: str) -> float:
    """Корреляция половин, которую обещает модель: среднее √(w_a·w_b) по игрокам split-half."""
    h = rows_dev.sort_values(["player_id", "match_id"]).assign(
        half=lambda d: d.groupby("player_id").cumcount() % 2)
    t = h.groupby(["player_id", "half"]).agg(toi_s=("toi_s", "sum"), role=("role", "first")).unstack("half")
    t = t[(t[("toi_s", 0)] >= SPLIT_MIN_TOI_S) & (t[("toi_s", 1)] >= SPLIT_MIN_TOI_S)]
    beta = t[("role", 0)].map(lambda r: mom[kind][r]["beta"]).astype(float)
    wa = t[("toi_s", 0)] / 3600 / (t[("toi_s", 0)] / 3600 + beta)
    wb = t[("toi_s", 1)] / 3600 / (t[("toi_s", 1)] / 3600 + beta)
    return float(np.sqrt(wa * wb).mean())


def reliability_block(rows_dev: pd.DataFrame, mom: dict) -> dict:
    cols = ["player_id", "match_id", "role", "toi_s", "value"]
    rel = {kind: cs.split_half_reliability(rows_dev.rename(columns={kind: "value"})[cols], SPLIT_MIN_TOI_S)
           | {"model_implied_half_r": implied_half_r(rows_dev, mom, kind)} for kind in KINDS}
    net = rows_dev.assign(value=rows_dev.drawn - rows_dev.taken)
    return rel | {"net": cs.split_half_reliability(net[cols], SPLIT_MIN_TOI_S)}


def _finite(o):
    """NaN/inf → None (строгий JSON)."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def run(args) -> dict:
    cfg = load_config()
    setup_logging(cfg, "card_penalties")
    cache = vr.ReadOnlyCache(cfg, io.data_version(cfg))
    l0 = vr.l0_dir(cache)
    tables = card_io.load_card_tables(cfg, cache)
    dev = set(matches_for(load_splits(cfg), "dev"))
    seed = int(cfg["seed"])
    inf_tr, inf_te = load_infractions(l0, "train"), load_infractions(l0, "test")
    inf_dev = inf_tr[inf_tr.match_id.isin(dev)]
    timing = manpower_timing(l0, dev)
    l2_key = cache.load_obj(tables["l3_key"], "report")["inputs"]["l2"]   # модель L2 сдаваемого L3
    nu, nu_draws = nu_block(l0, cache, dev, inf_dev, args.n_boot, seed, l2_key)
    pm_tr, pm_te = tables["pm_train"], tables["pm_test"]
    rows_dev = player_rows(pm_tr[pm_tr.match_id.isin(dev)], cp.penalty_counts(inf_dev))
    mom = fit_moments(rows_dev)
    acc = acceptance(rows_dev, mom, args.n_boot, seed)
    share = {"train_dev": observed_share(inf_dev), "test": observed_share(inf_te)}
    mom_test = {k: {r: cp.rescale_prior(mom[k][r], share["test"][k] / share["train_dev"][k]) for r in ROLES}
                for k in KINDS}
    out = apply_test(player_rows(pm_te, cp.penalty_counts(inf_te)), mom_test,
                     {k: acc[k]["accepted"] for k in KINDS}, share["test"], nu["nu_xg"], nu_draws, cfg,
                     args.n_draws, seed + 2)
    team = pm_te.groupby("player_id").team_id.agg(lambda s: s.mode().iat[0])
    out.insert(1, "team_id", out.player_id.map(team))
    out.insert(2, "is_avangard", out.team_id.eq(str(cfg["avangard_team_id"])))
    args.out.mkdir(parents=True, exist_ok=True)
    out.drop(columns=["toi_s"]).to_csv(args.out / "penalties.csv", index=False)
    report = {"l5_cache": tables["l5"]["key"], "dev_matches": len(dev), "manpower_timing_dev": timing,
              "double_counting": "нет: численность меняется в новом отрезке, горизонт V обрывается на остановке",
              "other_id_missing_share": {"train_dev": float(inf_dev.other_id.isna().mean()),
                                         "test": float(inf_te.other_id.isna().mean())},
              "actor_id_missing_share": {"train_dev": float(inf_dev.actor_id.isna().mean()),
                                         "test": float(inf_te.actor_id.isna().mean())},
              "observed_participant_share": share, "missing_policy": "пропуски случайны: λ_набл = λ·доля",
              "nu": nu, "prior_dev_ml": mom, "acceptance": acc, "split_half_dev": reliability_block(rows_dev, mom),
              "test_labels": out.reliability.value_counts().to_dict()}
    (args.out / "penalties_report.json").write_text(json.dumps(_finite(report), ensure_ascii=False, indent=2, default=str, allow_nan=False))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--n-draws", type=int, default=DEFAULT_N_DRAWS)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    try:
        rep = run(args)
    except (vr.CachesNotReady, card_io.CardInputMismatch) as ex:
        print(f"кэши не готовы: {ex}", file=sys.stderr)
        return 3
    nu = rep["nu"]
    print(f"ν = {nu['nu_xg']:.4f} xG [{nu['nu_xg_ci95'][0]:.4f}; {nu['nu_xg_ci95'][1]:.4f}], голы {nu['nu_goals']:.4f}; "
          f"split-half {({k: round(v['r_sb'], 3) for k, v in rep['split_half_dev'].items()})}; метки {rep['test_labels']}")
    for k, a in rep["acceptance"].items():
        o = a["observed"]
        print(f"{k}: r половин {o['r']:.3f} [{o['lo']:.3f}; {o['hi']:.3f}], модель {a['model_implied_half_r']}, "
              f"принято={a['accepted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
