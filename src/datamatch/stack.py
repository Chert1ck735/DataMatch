"""Стек моделей ценности: xG -> V -> δ -> журнал -> таблица «игрок × матч» -> рейтинг.

Один и тот же код используется во внутренней валидации (обучение на обучающих фолдах),
на внешнем блоке (обучение на 80%), для финального применения к test (обучение на всём
train) и в бутстрэпе моделей ценности (веса матчей).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import l1_xg, l2_value, l3_contrib
from .l5_shrink import JointEB, ROLES, reliability

COMP = [f"c_{c}" for c in l3_contrib.CATEGORIES]


from .node_graph import subset_nodes


def subset_matches(nodes: pd.DataFrame, matches) -> pd.DataFrame:
    return subset_nodes(nodes, nodes.match_id.isin(set(matches)).values)


class ValueStack:
    def __init__(self, cfg, xg_spec: dict, v_spec: dict | None = None, seed: int = 0):
        self.cfg = cfg
        self.xg_spec = xg_spec
        self.v_spec = v_spec or {"kind": "markov"}
        self.seed = seed

    def fit(self, nodes: pd.DataFrame, match_w: dict | None = None, xg_proc=None):
        """nodes — узлы обучающих матчей; match_w — веса матчей (бутстрэп)."""
        w_node = None if match_w is None else nodes.match_id.map(match_w).fillna(0.0).values
        shots = l1_xg.shot_table(nodes)
        if xg_proc is None:
            if match_w is not None:
                # бутстрэп: повтор матчей = повтор его бросков
                rep = shots.match_id.map(match_w).fillna(0).astype(int).values
                shots_b = shots.loc[shots.index.repeat(rep)].reset_index(drop=True)
                xg_proc = l1_xg.XGProcedure(self.xg_spec, self.seed).fit(shots_b)
            else:
                xg_proc = l1_xg.XGProcedure(self.xg_spec, self.seed).fit(shots)
        self.xg = xg_proc
        r = self.reward(nodes)
        vs = self.v_spec
        kind = vs.get("kind", "markov")
        if vs.get("pipeline") == "fold_local_calibrated":
            if match_w is not None:
                raise ValueError("Calibrated L2 bootstrap is not implemented; do not silently substitute the legacy model")
            from .l2_procedure import ValueProcedure
            self.v = ValueProcedure(self.cfg, vs, self.seed).fit(nodes, r, xg_spec=self.xg_spec)
            self.training_matches_ = set(nodes.match_id)
        elif kind == "markov":
            c = self.cfg["value"]["markov"]
            grid = l2_value.Grid(vs.get("x_edges", c["x_edges"]), vs.get("y_edges", c["y_edges"]))
            self.v = l2_value.MarkovV(grid, kappa=vs.get("kappa", c["prior_strength"]),
                                      gamma_per_s=vs.get("gamma_per_s"), own_poss_only=vs.get("own_poss_only", False))
            self.v.fit(nodes, r, w_node)
        elif kind in ("gbm", "gbm_hist", "fqe"):
            from . import l2_gbm
            self.v = l2_gbm.GBMV(self.cfg, kind=kind, seed=self.seed, opts=vs).fit(nodes, r, w_node)
        else:
            raise ValueError(kind)
        return self

    def reward(self, nodes: pd.DataFrame) -> np.ndarray:
        r = np.zeros(len(nodes))
        sh = nodes.action.values == "shot"
        if sh.any():
            r[sh] = self.xg.predict(l1_xg.shot_table(nodes[sh]))
        return r

    def apply(self, nodes: pd.DataFrame) -> dict:
        r = self.reward(nodes)
        vs = self.v_spec
        if vs.get("pipeline") == "fold_local_calibrated":
            if self.training_matches_ & set(nodes.match_id):
                raise ValueError("Training credits require the layer's match-cross-fit dev_values, not in-sample application")
            v = self.v.value(nodes, neutral=True)
        else:
            v = self.v.value(nodes)
        d = l2_value.td_delta(nodes, r, v, vs.get("gamma_per_s"), vs.get("own_poss_only", False))
        return {"r": r, "v": v, "delta": d}


def build_pm(nodes, out, static_pm, inc, cfg, variant=None, alpha=None, extra=None):
    c = cfg["contrib"]
    jr = l3_contrib.journal(nodes, out["delta"], variant or c["turnover_variant"],
                            float(c["alpha"] if alpha is None else alpha), extra=extra)
    extra_cols = None
    if extra:
        extra_cols = {f"x_{k}": k for k in extra}
    pm = l3_contrib.dynamic_player_match(static_pm, nodes, jr, inc, out["r"], extra_cols)
    return pm, jr


def player_sums(pm: pd.DataFrame, matches=None, cols=COMP) -> pd.DataFrame:
    p = pm if matches is None else pm[pm.match_id.isin(matches)]
    # Contributions are already rate-filtered; use the same approved exposure.
    p = p[p.toi_rate_s > 0]
    g = p.groupby("player_id").agg(**{c: (c, "sum") for c in cols}, toi_s=("toi_rate_s", "sum"),
                                   games=("match_id", "nunique"), team_id=("team_id", lambda s: s.value_counts().index[0]))
    return g


def rate_players(eb: JointEB, sums: pd.DataFrame, roles: pd.Series, ids=None, m_r=None, cfg=None):
    """O1 по совместной модели. ids — оцениваемые игроки (включая без TOI: ветка prior)."""
    ids = list(sums.index) if ids is None else list(ids)
    s = sums.reindex(ids)
    toi = s.toi_s.fillna(0.0).values
    games = s.games.fillna(0).values.astype(float)
    vals = s[eb.cols].fillna(0.0).values
    rr = roles.reindex(ids).values
    res = eb.overall(ids, rr, vals, toi, games, m_r=m_r)
    sh = cfg["shrink"] if cfg is not None else {"reliability_high": 0.7, "reliability_medium": 0.4}
    df = pd.DataFrame({"player_id": ids, "role": rr, "rating": res["overall"], "post_var": res["var"],
                       "w": res["w"], "toi_min": toi / 60, "games": games.astype(int), "method": res["method"],
                       "flag": res["flag"]})
    df["reliability"] = reliability(df.w.values, float(sh["reliability_high"]), float(sh["reliability_medium"]))
    for j, c in enumerate(eb.cols):
        df[f"prof_{c}"] = res["profile"][:, j]
    return df, res
