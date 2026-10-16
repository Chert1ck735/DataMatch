"""L2 regression estimators: signed xG return to a confirmed stop, unit event weights.

Features describe the pre-action state, without identity or disputed structural
labels. Previous coordinates may enter the history candidate. Temporal features
use trusted past monotone times and nominal period ends. Faceoff perspectives
share one physical event's weight; ValueProcedure enforces antisymmetry and learns
calibration/nuisance strength by match cross-fitting. stage_l2 cross-fits the full
xG→V pipeline for development credits. FQE is an optional different training scheme.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import model_threads
from .l0_events import ARRIVALS, MP_CLASSES, MP_FLIP
from .l2_value import realized_return, td_delta

CAT_V = ["mp", "rink_type"]
CATS_V = {"mp": MP_CLASSES, "rink_type": ["european_60x30", "finnish_60x28", "nhl_61x26"]}
# P005: arrival, типы предыдущих действий и self-pass могут зависеть от того, как оператор
# записал эпизод у гола. Они исключены до отдельного аудита и абляции; это является контрактом
# признаков V/Q, а не неявным выбором модели.
VALUE_FEATURE_EXCLUSIONS = ["arrival", "prev_action_h", "prev2_action_h", "prev_self_pass", "recent_entry"]
BASE_NUM = ["x", "y", "dist", "angle_vis", "own_sk", "opp_sk", "own_g", "opp_g", "score_diff", "period",
            "t_remain_pre", "t_seg_pre", "t_poss_pre", "n_in_poss", "dt_prev_lag", "is_faceoff",
            "goal_line_x", "post_half_width", "team_strength"]
HIST_NUM = [f"prev{k}_{axis}" for k in range(1, 4) for axis in ("x", "y")]


def value_feature_contract() -> dict:
    """Явный контракт P005 для отчёта L2 и воспроизводимых запусков."""
    return {"included": BASE_NUM + CAT_V + HIST_NUM,
            "excluded_pending_structure_audit": VALUE_FEATURE_EXCLUSIONS}


def v_features(nodes: pd.DataFrame, hist: bool, geo=None) -> pd.DataFrame:
    from . import io
    from .config import load_config
    n = nodes
    ids = pd.Index(n.node)
    prv = ids.get_indexer(n.prev_node.to_numpy())
    has = prv >= 0
    if ((n.prev_node.to_numpy() >= 0) & ~has).any():
        raise ValueError("Value features require complete match chains; unresolved predecessor")
    pv = np.maximum(prv, 0)
    nominal = n.period_nominal.to_numpy(dtype=float)
    tm = n.t_mono.to_numpy(dtype=float)
    ref = np.where(has, tm[pv], np.nan)
    # Opening faceoff clocks are known before their outcome; other first actions
    # use the recorded opener, never their own outcome-sensitive timestamp.
    opener = n.t_ref_source.eq("segment_opener").to_numpy()
    ref = np.where(opener, n.t_ref.to_numpy(dtype=float), ref)
    ref = np.where(n.action.eq("faceoff"), tm, ref)
    same = has & (n.possession.to_numpy()[pv] == n.possession.to_numpy())
    geo = io.rink_geometry(load_config()) if geo is None else geo
    gx = n.rink_type.map({k: v["goal_line_opp"] for k, v in geo.items()}).to_numpy()
    ph = n.rink_type.map({k: v["post_half_width"] for k, v in geo.items()}).to_numpy()
    X = pd.DataFrame({
        "x": n.x.to_numpy(), "y": n.y.to_numpy(), "dist": n.dist.to_numpy(), "angle_vis": n.angle_vis.to_numpy(),
        "goal_line_x": gx, "post_half_width": ph,
        "own_sk": n.own_sk.to_numpy(), "opp_sk": n.opp_sk.to_numpy(),
        "own_g": n.own_g.to_numpy(dtype=int), "opp_g": n.opp_g.to_numpy(dtype=int),
        "score_diff": np.clip(n.score_diff.to_numpy(), -3, 3), "period": np.minimum(n.period.to_numpy(), 4),
        "t_remain_pre": np.clip(nominal-ref, 0, None),
        "t_seg_pre": np.where(has, n.t_seg.to_numpy()[pv], 0.),
        "t_poss_pre": np.where(same, n.t_poss.to_numpy()[pv], 0.),
        "n_in_poss": n.n_in_poss.to_numpy(),
        "dt_prev_lag": np.where(has, np.nan_to_num(n.dt_prev.to_numpy()[pv], nan=-1.), -1.),
        "is_faceoff": n.action.eq("faceoff").to_numpy(dtype=int),
        "mp": n.mp.to_numpy(), "rink_type": n.rink_type.to_numpy(),
        "team_strength": n.get("team_strength", pd.Series(0., index=n.index)).to_numpy()})
    if hist:
        for k in range(1, 4):
            X[f"prev{k}_x"] = np.where(has, n.x.to_numpy()[pv], np.nan)
            X[f"prev{k}_y"] = np.where(has, n.y.to_numpy()[pv], np.nan)
            # Coordinates remain in the current owner's perspective.
            sign = np.where(n.team_id.to_numpy()[pv] == n.team_id.to_numpy(), 1., -1.)
            X[f"prev{k}_x"] *= sign
            X[f"prev{k}_y"] *= sign
            parent = ids.get_indexer(n.prev_node.to_numpy()[pv])
            has = has & (parent >= 0)
            pv = np.maximum(parent, 0)
    for c in CAT_V:
        X[c] = pd.Categorical(X[c], categories=CATS_V[c])
    return X


def mirror_faceoffs(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, is_fo: np.ndarray):
    """Перспектива проигравшего вбрасывание: зеркальная клетка, перевёрнутые манпауэр и счёт, цель −G.

    Вес faceoff: каждая перспектива получает 0.5 * w, суммарно w — вес физического события.
    Не-faceoff: вес w. Баланс соблюдён — физическое событие не удваивается.
    """
    f = X[is_fo].copy()
    f["x"], f["y"] = -f.x, -f.y
    # расстояние и видимый угол до чужих ворот у проигравшего — от зеркальной точки
    dx = X.goal_line_x.to_numpy()[is_fo] + X.x.values[is_fo]
    dy = -X.y.values[is_fo]
    f["dist"] = np.sqrt(dx ** 2 + dy ** 2)
    from .l0_events import visible_angle
    f["angle_vis"] = visible_angle(dx, dy, X.post_half_width.to_numpy()[is_fo])
    f["own_sk"], f["opp_sk"] = X.opp_sk.values[is_fo], X.own_sk.values[is_fo]
    f["own_g"], f["opp_g"] = X.opp_g.values[is_fo], X.own_g.values[is_fo]
    f["score_diff"] = -X.score_diff.values[is_fo]
    f["team_strength"] = -X.team_strength.values[is_fo]
    for c in HIST_NUM:
        if c in f:
            f[c] = -f[c]
    f["mp"] = pd.Categorical(pd.Series(X.mp.values[is_fo]).astype(str).map(MP_FLIP).values, categories=CATS_V["mp"])
    weights = np.asarray(w, float).copy()
    weights[is_fo] *= .5
    return (pd.concat([X, f], ignore_index=True), np.r_[y, -y[is_fo]], np.r_[weights, weights[is_fo]])


class GBMV:
    def __init__(self, cfg, kind: str = "gbm", seed: int = 0, opts: dict | None = None):
        self.cfg = cfg
        self.kind = kind
        self.seed = seed
        self.opts = opts or {}
        from . import io
        self.geo = io.rink_geometry(cfg)
        self.hist = kind in ("gbm_hist",) or self.opts.get("hist", False)
        self.name = {"gbm": "M-GBM", "gbm_hist": "M-GBM-hist", "fqe": "M-FQE"}[kind]

    def _model(self):
        import lightgbm as lgb
        p = dict(self.cfg["value"]["gbm"])
        p.update(self.opts.get("params", {}))
        return lgb.LGBMRegressor(objective="regression", random_state=self.seed, verbose=-1, n_jobs=model_threads(), deterministic=True, force_col_wise=True, **p)

    def _fit_one(self, nodes, r, w_node, target=None):
        X = v_features(nodes, self.hist, self.geo)
        y = realized_return(nodes, r, self.opts.get("gamma_per_s"), self.opts.get("own_poss_only", False)) if target is None else target
        # Вес события — единица (не 1/N): цель — беллмановская V, корреляцию событий одного
        # отрезка учитывает бутстрэп матчей при оценке, а не перевзвешивание цели.
        w = np.ones(len(nodes))
        # State value is a team target; personal identity/TOI eligibility does not
        # define it. Only observed finite full-horizon (or FQE) targets can train.
        if w_node is not None:
            w = w * w_node
        keep = (w > 0) & np.isfinite(w) & np.isfinite(y)
        if not keep.any():
            raise ValueError("No observed finite value targets for training")
        is_fo = nodes.action.values == "faceoff"
        Xa, ya, wa = mirror_faceoffs(X[keep].reset_index(drop=True), y[keep], w[keep], is_fo[keep])
        m = self._model()
        m.fit(Xa, ya, sample_weight=wa)
        return m

    def fit(self, nodes, r, w_node=None):
        self.model = self._fit_one(nodes, r, w_node)
        if self.kind == "fqe":
            n_it = int(self.opts.get("fqe_iters", 3))
            for _ in range(n_it):
                v = self.model.predict(v_features(nodes, self.hist, self.geo))
                tgt = td_delta(nodes, r, v, self.opts.get("gamma_per_s"), self.opts.get("own_poss_only", False)) + v      # r + σ V̂(s')
                self.model = self._fit_one(nodes, r, w_node, target=tgt)
        return self

    def value(self, nodes):
        return self.model.predict(v_features(nodes, self.hist, self.geo))
