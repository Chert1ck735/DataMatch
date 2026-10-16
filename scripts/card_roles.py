"""Карточка игрока, пункт 2: роли (GMM по компонентам L5) и похожие игроки (METHODOLOGY 11.2).

Запуск из Solve (кэши L3/L5 только читаются):

    PYTHONPATH=src .venv/bin/python scripts/card_roles.py

Обучение (масштаб, число кластеров, GMM, ковариация для расстояний) — на матчах разработки train:
совместная модель L5 (`JointEB`) переподгоняется на них без правок класса. Test — только применение.
Выход: outputs/card/roles.csv, similar.csv, roles_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import validation_report as vr

from datamatch import card_io, card_roles as cr
from datamatch import io, stage_l5
from datamatch.cache import code_hash
from datamatch.config import load_config, setup_logging
from datamatch.l5_shrink import ROLES, JointEB
from datamatch.splits import load_splits, matches_for

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "card"
# CONFIG: параметры, зафиксированные в METHODOLOGY 11.2 до расчёта.
MIN_W = 0.4                 # обучающий набор, пул похожих, «профиль неустойчив»
K_MAX = 6
N_INIT = 10
ARI_THRESHOLD = 0.3
MIN_HALF_TOI_MIN = 60.0
TOP_K = 5
UNSTABLE_PROFILE = "профиль неустойчив"
UNSTABLE_ROLE = "роль неустойчива"


def posterior_components(eb: JointEB, pm: pd.DataFrame, roles: pd.Series, cfg) -> pd.DataFrame:
    """Апостериорные компоненты (θ̂ − m_r) и w; m_r — центр этой же выборки (профиль относительно своих)."""
    rf, _ = stage_l5.rate_frame(pm, stage_l5.COMP_COLS)
    ids = sorted(pm.player_id.unique())
    df, _ = stage_l5.apply_o1(eb, stage_l5.player_sums(rf, stage_l5.COMP_COLS), roles, ids, cfg)
    return df


def fit_role(r: str, dev: pd.DataFrame, half_a: pd.DataFrame, half_b: pd.DataFrame, seed: int) -> dict:
    feats = list(cr.role_features(r))
    fit = dev[(dev.role == r) & (dev.w >= MIN_W)]
    sc = cr.fit_scaler(fit, feats)
    Zfit = cr.apply_scaler(sc, fit)
    sub = cr.fit_subspace(Zfit)
    X = cr.project(sub, Zfit)
    sel = cr.select_gmm(X, K_MAX, seed, N_INIT)
    ok = fit.player_id[fit.player_id.isin(half_a.player_id) & fit.player_id.isin(half_b.player_id)]
    a = half_a.set_index("player_id").loc[ok]
    b = half_b.set_index("player_id").loc[ok]
    both = (a.toi_min >= MIN_HALF_TOI_MIN).to_numpy() & (b.toi_min >= MIN_HALF_TOI_MIN).to_numpy()
    ari = cr.split_half_ari(sel.model, cr.project(sub, cr.apply_scaler(sc, a[both])),
                            cr.project(sub, cr.apply_scaler(sc, b[both])))
    centers = np.array([cr.to_z(sub, c) for c in sel.model.means_])
    names = cr.unique_names([cr.cluster_name(c, feats) for c in centers])
    labels = sel.model.predict(X)
    summary = cr.cluster_summary(Zfit, labels, centers, fit.player_id.to_numpy(), feats)
    eig_all = np.linalg.eigvalsh(np.cov(Zfit, rowvar=False))[::-1]
    return {"features": feats, "scaler": sc, "subspace": sub, "sel": sel, "cov": np.eye(len(sub.eig)),
            "names": names, "ari": ari, "stable": bool(np.isfinite(ari) and ari >= ARI_THRESHOLD),
            "report": {"n_fit": int(len(fit)), "eig_standardized": eig_all.tolist(),
                       "subspace_dim": int(len(sub.eig)), "k": sel.k, "bic": sel.bic,
                       "silhouette_kmeans_diag": cr.kmeans_silhouette(X, range(2, K_MAX + 1), seed),
                       "ari_split_half": ari, "n_ari_players": int(both.sum()), "ari_threshold": ARI_THRESHOLD,
                       "roles_accepted": bool(np.isfinite(ari) and ari >= ARI_THRESHOLD),
                       "covariance_type": cr.COVARIANCE_TYPE,
                       "clusters": [{"name": n, "center_z": dict(zip(feats, map(float, c))),
                                     "share_fit": float(np.mean(labels == i)), **summary[i]}
                                    for i, (n, c) in enumerate(zip(names, centers))]}}


def apply_roles(test: pd.DataFrame, fits: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """Роль и z-профиль игроков test (масштаб и GMM — из разработки)."""
    rows, Z = [], np.zeros((len(test), len(cr.FEATURES["C"])))
    for r, f in fits.items():
        m = (test.role == r).to_numpy()
        if not m.any():
            continue
        z = cr.apply_scaler(f["scaler"], test[m])
        u = cr.project(f["subspace"], z)
        Z[np.flatnonzero(m), : u.shape[1]] = u
        prob = f["sel"].model.predict_proba(u)
        lab = prob.argmax(axis=1)
        for i, (pid, w) in enumerate(zip(test.player_id[m], test.w[m])):
            flag = [UNSTABLE_PROFILE] if w < MIN_W else []
            flag += [] if f["stable"] or f["sel"].k == 1 else [UNSTABLE_ROLE]
            rows.append({"player_id": pid, "cluster": int(lab[i]), "role_name": f["names"][lab[i]],
                         "own_profile": cr.cluster_name(z[i], f["features"]),
                         "cluster_prob": float(prob[i].max()), "flag": "; ".join(flag),
                         **{f"z_{c[5:]}": float(v) for c, v in zip(f["features"], z[i])}})
    return pd.DataFrame(rows), Z


def run(args) -> dict:
    cfg = load_config()
    log = setup_logging(cfg, "card_roles")
    cache = vr.ReadOnlyCache(cfg, io.data_version(cfg))
    tables = card_io.load_card_tables(cfg, cache)
    dev_ids = set(matches_for(load_splits(cfg), "dev"))
    pm_tr = tables["pm_train"]
    pm_dev = pm_tr[pm_tr.match_id.isin(dev_ids)]
    # Роли — по матчам разработки (внешний блок не открывается); для test — как в L5.
    roles = stage_l5.player_roles(pm_dev)
    roles_all = stage_l5.player_roles(tables["pm_test"]).combine_first(stage_l5.player_roles(pm_tr))
    eb = JointEB(stage_l5.COMP_COLS, cfg["shrink"], logger=log).fit(stage_l5.rate_frame(pm_dev, stage_l5.COMP_COLS)[0],
                                                                    roles)
    # Поправка после первого расчёта (METHODOLOGY 11.2): одна модель разработки для разработки, её половин
    # и test (μ_r test — КХЛ-компонента разработки); центр m_r — у каждой выборки свой, как в L5.
    dev = posterior_components(eb, pm_dev, roles, cfg)
    ha, hb = cr.alternate_halves(pm_dev)
    half_a, half_b = posterior_components(eb, ha, roles, cfg), posterior_components(eb, hb, roles, cfg)
    seed = int(cfg["seed"])
    fits = {r: fit_role(r, dev, half_a, half_b, seed) for r in ROLES}

    pm_dev_khl = pm_dev[pm_dev.is_khl_component.fillna(False).astype(bool)]
    rf_khl = stage_l5.rate_frame(pm_dev_khl, stage_l5.COMP_COLS)[0]
    if rf_khl.empty:
        raise ValueError("card_roles: в матчах разработки нет КХЛ-компоненты для μ_r test")
    eb_test = eb.with_mu(eb.prior_mu(rf_khl, roles))
    # Метка надёжности — показываемая (правило «мало минут», METHODOLOGY 7.7), как в test_ratings.csv.
    o1 = tables["detail"][["player_id", "rating", "reliability_display"]].rename(
        columns={"reliability_display": "reliability"})
    test = posterior_components(eb_test, tables["pm_test"], roles_all, cfg).drop(columns=["rating", "reliability"])
    test = test.merge(o1, on="player_id", how="left")
    team = tables["pm_test"].groupby("player_id").team_id.agg(lambda s: s.mode().iat[0])
    test = test.assign(team_id=test.player_id.map(team).fillna("unknown"))
    roles_df, Z = apply_roles(test, fits)
    avangard = str(cfg["avangard_team_id"])
    out = test[["player_id", "team_id", "role", "toi_min", "rating", "w", "reliability"]].rename(
        columns={"w": "w_profile"}).merge(roles_df, on="player_id")
    out["is_avangard"] = out.team_id.eq(avangard)
    sim = cr.similar_table(test[["player_id", "role", "team_id", "w"]], Z,
                           {r: f["cov"] for r, f in fits.items()}, TOP_K, MIN_W)
    sim = sim.assign(is_avangard=sim.player_id.map(test.set_index("player_id").team_id).eq(avangard))
    sim = sim[(sim.pool == "all") | sim.is_avangard]
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out / "roles.csv", index=False)
    sim.to_csv(args.out / "similar.csv", index=False)
    report = {"l5_cache": tables["l5"]["key"], "l5_shrink_code": code_hash("l5_shrink"),
              "dev_matches": len(dev_ids), "dev_players": int(len(dev)),
              "test_profile_model": "JointEB разработки, μ_r — КХЛ-компонента разработки, центр m_r — по test; "
                                    "rating/reliability — O1 (сдаваемый результат)",
              "params": {"min_w": MIN_W, "k_max": K_MAX, "n_init": N_INIT, "ari_threshold": ARI_THRESHOLD,
                         "min_half_toi_min": MIN_HALF_TOI_MIN, "top_k": TOP_K},
              "roles": {r: f["report"] for r, f in fits.items()},
              "test_counts": {r: out[out.role == r].role_name.value_counts().to_dict() for r in ROLES},
              "test_unstable_profile": int(out.flag.str.contains(UNSTABLE_PROFILE).sum()),
              "similar_pool_sizes": {r: int(((test.role == r) & (test.w >= MIN_W)).sum()) for r in ROLES}}
    (args.out / "roles_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
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
    for r, x in rep["roles"].items():
        print(f"{r}: k={x['k']}, ARI={x['ari_split_half']:.3f} (n={x['n_ari_players']}), "
              f"приняты={x['roles_accepted']}, набор {x['n_fit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
