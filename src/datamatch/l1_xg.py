"""L1: модель ожидаемых голов (xG) — вероятность гола с попытки броска до её исхода.

Выборка бросков: все попытки (shot_on_target, shot_missed, shot_blocked, goal) периодов 1–5,
кроме штрафных бросков; заблокированные входят с меткой 0 — xG оценивает попытку до исхода,
включая шанс блока. AUC дополнительно отчитывается без заблокированных.

Признаки — только известные до броска. Запрещены (утечки): идентификаторы игроков,
исход броска (тип shot_*), net_x/net_y, x2/y2 бросков (вратарь/блокирующий), assist_*,
goal_error, scoring_chance.

Сила соперника (opp_def) оценивается кросс-подгонкой по матчам обучающей части;
нейтральный xG усредняет вероятности по её эмпирическому распределению.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import model_threads
from scipy.optimize import minimize
from scipy.special import expit

FORBIDDEN = {"net_x", "net_y", "x2", "y2", "actor_id", "other_id", "primary", "blocked", "is_goal",
             "assist_primary", "assist_secondary", "goal_error", "scoring_chance",
             "dt_prev", "t_remain_c", "dt_last_own_shot_c"}
# Признаки по списку PDF (L1): расстояние и видимый угол ворот, манпауэр, счёт, период,
# добивание (Δt ≤ 3 с после броска своей команды), поперечное смещение предыдущего паса,
# вход в зону за последние k секунд, тип площадки, оборонительная сила соперника.
# Не используются: dt_prev и непрерывное время от предыдущего броска (у голов отметка
# времени систематически позже — сигнал исхода, AUC одного dt_prev 0,74), время периода
# (вне списка PDF, давало скачки по времени периода), x/|y| (дублируют расстояние/угол).
NUM = ["dist", "angle_vis", "prev_pass_dy", "score_diff_c", "own_sk", "opp_sk", "opp_def"]
BIN = ["behind_net", "rebound", "prev_pass_cross", "recent_entry", "own_g", "opp_g"]
CAT = ["mp", "rink_type", "period_c"]
CATS = {
    "mp": ["EV5", "EVo", "PP", "SH", "ENF", "ENA"],
    "rink_type": ["european_60x30", "finnish_60x28", "nhl_61x26"],
    "period_c": ["1", "2", "3", "OT"],
}
FEATURES = NUM + BIN + CAT
CORE = ["dist", "angle_vis"]          # геометрия всегда в модели (монотонность)

# Происхождение каждого признака: откуда берётся и от какого момента считается.
# «отметка броска» — clock_s самого события броска (у голов систематически позже, сигнал
# исхода) — не используется ни одним признаком; это проверяется аудитом возмущения
# (l1_audit.shot_time_perturbation), а не только списком запрещённых имён.
FEATURE_PROVENANCE = {
    "dist": "x, y броска в системе бросающей команды; линия ворот по типу площадки",
    "angle_vis": "x, y броска; штанги y = ±0,915",
    "behind_net": "x броска относительно линии ворот",
    "prev_pass_dy": "пас партнёра, доставивший шайбу в текущую серию узлов игрока (раньше по seq); "
                    "|y_приёма − y_паса|; окно: t_ref − время паса ≤ 3 с; t_ref — время предыдущего узла",
    "prev_pass_cross": "тот же пас: смена знака y (через ось ворот); то же окно от t_ref",
    "rebound": "бросок своей команды раньше по seq в том же отрезке; t_ref − его время ≤ 3 с",
    "recent_entry": "метка entry_success своей команды раньше по seq в том же матче, периоде и отрезке; "
                    "t_ref − её время ≤ k с",
    "score_diff_c": "голы раньше по seq (без серии буллитов), с точки зрения бросающей команды, ±3",
    "own_sk": "n_home/n_away предыдущего события по seq минус вратарь (состав на позиции броска)",
    "opp_sk": "то же для соперника",
    "own_g": "вратарь своей команды в слоте G по реконструкции смен на позиции броска в seq",
    "opp_g": "то же для соперника (пустые ворота)",
    "mp": "класс манпауэра из own_sk, opp_sk, own_g, opp_g",
    "rink_type": "games.csv",
    "period_c": "период события",
    "opp_def": "оборонительная сила соперника вне фолда матча; нейтральный прогноз — среднее вероятностей",
}
DISPUTED = {"rebound": ["rebound"], "recent_entry": ["recent_entry"], "prev_pass": ["prev_pass_dy", "prev_pass_cross"]}


def split_features(features):
    f = list(features)
    return [c for c in NUM if c in f], [c for c in BIN if c in f], [c for c in CAT if c in f]


def shot_table(nodes: pd.DataFrame) -> pd.DataFrame:
    s = nodes[nodes.action == "shot"].copy()
    s["y_abs"] = s.y.abs()
    s["dt_prev_c"] = s.dt_prev.fillna(30.0).clip(0, 30)
    s["dt_last_own_shot_c"] = s.dt_last_own_shot.fillna(60.0).clip(0, 60)
    s["score_diff_c"] = s.score_diff.clip(-3, 3)
    s["t_remain_c"] = s.t_remain.clip(0, 1200)
    s["period_c"] = np.where(s.period <= 3, s.period.astype(str), "OT")
    s["goal"] = s.is_goal.astype(int)
    for b in BIN:
        s[b] = s[b].astype(int)
    if "opp_def" not in s:
        s["opp_def"] = 0.0
    return s


def team_def_strength(shots: pd.DataFrame, p0: np.ndarray, k: float = 10.0) -> dict:
    """Оборонительная сила команды: log((голы против + k)/(xG0 против + k)) на обучающей части."""
    d = pd.DataFrame({"team": shots.opp_team_id.values, "g": shots.goal.values, "xg": p0})
    a = d.groupby("team").agg(g=("g", "sum"), xg=("xg", "sum"))
    return np.log((a.g + k) / (a.xg + k)).to_dict()


# ---------------------------------------------------------------------------------------
# Дизайн-матрица для логистической модели: монотонный шарнирный базис по расстоянию
DIST_KNOTS = [2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0, 25.0, 35.0, 60.0]
ANG_KNOTS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7]


def _design(s: pd.DataFrame, stats: dict | None = None, features=None):
    num_f, bin_f, cat_f = split_features(features or FEATURES)
    cols, names, mono = [], [], []
    d = s.dist.values
    for k in DIST_KNOTS:            # max(0, k - d): невозрастающая по d; коэффициент >= 0
        cols.append(np.maximum(0.0, k - d) / k)
        names.append(f"hinge_dist_{k}")
        mono.append(True)
    a = s.angle_vis.values
    for k in ANG_KNOTS:             # max(0, a - k): неубывающая по углу; коэффициент >= 0
        cols.append(np.maximum(0.0, a - k))
        names.append(f"hinge_ang_{k}")
        mono.append(True)
    num = [c for c in num_f if c not in ("dist", "angle_vis")]
    X = s[num].astype(float).values
    if stats is None:
        stats = {"mu": X.mean(0), "sd": X.std(0) + 1e-9}
    X = (X - stats["mu"]) / stats["sd"]
    for j, n in enumerate(num):
        cols.append(X[:, j])
        names.append(n)
        mono.append(False)
    for b in bin_f:
        cols.append(s[b].astype(float).values)
        names.append(b)
        mono.append(False)
    for c in cat_f:
        for lev in CATS[c][1:]:
            cols.append((s[c].astype(str).values == lev).astype(float))
            names.append(f"{c}={lev}")
            mono.append(False)
    M = np.column_stack(cols)
    return M, names, np.array(mono), stats


class MonoLogit:
    """Логистическая регрессия со шарнирным базисом (монотонно убывает по расстоянию), L2."""

    name = "logit_spline"

    def __init__(self, l2: float = 1.0, features=None):
        self.l2 = l2
        self.features = list(features or FEATURES)

    def fit(self, s: pd.DataFrame, w: np.ndarray | None = None):
        X, self.names, mono, self.stats = _design(s, None, self.features)
        y = s.goal.values.astype(float)
        w = np.ones(len(y)) if w is None else w
        n, p = X.shape
        Xb = np.column_stack([np.ones(n), X])

        def f(beta):
            z = Xb @ beta
            ll = np.sum(w * (np.logaddexp(0, z) - y * z))
            pen = 0.5 * self.l2 * np.sum(beta[1:] ** 2)
            g = Xb.T @ (w * (expit(z) - y))
            g[1:] += self.l2 * beta[1:]
            return ll + pen, g

        bounds = [(None, None)] + [(0.0, None) if m else (None, None) for m in mono]
        b0 = np.zeros(p + 1)
        rate = np.clip(np.average(y, weights=w), 1e-6, 1 - 1e-6)
        b0[0] = np.log(rate / (1 - rate))
        r = minimize(f, b0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000})
        if not r.success or not np.isfinite(r.x).all():
            raise RuntimeError(f"MonoLogit did not converge: {r.message}")
        self.beta = r.x
        self.converged = bool(r.success)
        return self

    def predict(self, s: pd.DataFrame) -> np.ndarray:
        X, _, _, _ = _design(s, self.stats, self.features)
        return expit(self.beta[0] + X @ self.beta[1:])


class LGBMxG:
    name = "lgbm_mono"

    def __init__(self, params: dict, seed: int = 0, features=None):
        self.params = dict(params)
        self.seed = seed
        self.features = list(features or FEATURES)

    def _X(self, s):
        X = s[self.features].copy()
        for c in split_features(self.features)[2]:
            X[c] = pd.Categorical(X[c].astype(str), categories=CATS[c])
        return X

    def fit(self, s, w=None):
        import lightgbm as lgb
        mono = [(-1 if f == "dist" else (1 if f == "angle_vis" else 0)) for f in self.features]
        self.model = lgb.LGBMClassifier(objective="binary", monotone_constraints=mono,
                                        monotone_constraints_method="advanced", random_state=self.seed,
                                        verbose=-1, n_jobs=model_threads(), deterministic=True, force_col_wise=True, **self.params)
        self.model.fit(self._X(s), s.goal.values, sample_weight=w)
        return self

    def predict(self, s):
        return self.model.predict_proba(self._X(s))[:, 1]

    # Нейтральный xG усредняет вероятность по ~100 уровням силы соперника. Деревья без разбиений
    # по opp_def дают одинаковый вклад на любом уровне: их сумма считается один раз, а на каждом
    # уровне пересчитываются только зависящие от opp_def (ускорение реплик L6, 06.10.2026).
    MIXTURE_SELF_CHECK_TOL = 1e-9

    def _dynamic_trees(self) -> list[int]:
        feature = self.features.index("opp_def")

        def splits_on(node) -> bool:
            if "split_index" not in node:
                return False
            return (node["split_feature"] == feature or splits_on(node["left_child"])
                    or splits_on(node["right_child"]))

        info = self.model.booster_.dump_model()["tree_info"]
        return [t["tree_index"] for t in info if splits_on(t["tree_structure"])]

    def dynamic_tree_count(self) -> int:
        return len(self._dynamic_trees())

    def _dynamic_booster(self):
        """Модель только из деревьев с разбиениями по opp_def (тот же формат LightGBM); None, если их нет."""
        import re

        import lightgbm as lgb
        if getattr(self, "_dynamic_cache", None) is not None:
            return self._dynamic_cache[0]
        dynamic = self._dynamic_trees()
        if not dynamic:
            self._dynamic_cache = (None,)
            return None
        text = self.model.booster_.model_to_string()
        start, end = text.index("Tree=0\n"), text.index("end of trees")
        blocks = re.split(r"(?m)^Tree=\d+\n", text[start:end])[1:]
        header = re.sub(r"(?m)^tree_sizes=.*\n", "", text[:start])
        trees = "".join(f"Tree={j}\n{blocks[i]}" for j, i in enumerate(dynamic))
        booster = lgb.Booster(model_str=header + trees + text[end:])
        self._dynamic_cache = (booster,)
        return booster

    def neutral_mixture(self, s, values, weights) -> np.ndarray:
        """Σ_k w_k·P(гол | opp_def = v_k) — та же точная смесь, что перебор уровней, но быстрее."""
        from scipy.special import expit
        X = self._X(s.assign(opp_def=float(values[0])))
        dynamic = self._dynamic_booster()
        if dynamic is None:
            return float(np.sum(weights)) * self.model.predict_proba(X)[:, 1]
        static = self.model.booster_.predict(X, raw_score=True) - dynamic.predict(X, raw_score=True)
        reference = self.model.predict_proba(X)[:, 1]
        first = expit(static + dynamic.predict(X, raw_score=True))
        if np.max(np.abs(first - reference)) > self.MIXTURE_SELF_CHECK_TOL:
            import logging
            logging.getLogger("datamatch").warning("нейтральный xG: самопроверка ускорения не прошла — полный перебор")
            return sum(w * self.predict(s.assign(opp_def=float(v))) for v, w in zip(values, weights))
        p = np.zeros(len(s))
        for v, w in zip(values, weights):
            X["opp_def"] = float(v)
            p += w * expit(static + dynamic.predict(X, raw_score=True))
        return p


class CatBoostxG:
    """CPU CatBoost with the same feature and monotonicity contract as LGBM."""

    name = "catboost_mono"

    def __init__(self, params, seed=0, features=None):
        self.params, self.seed = dict(params), seed
        self.features = list(features or FEATURES)

    def _X(self, s):
        X = s[self.features].copy()
        for c in split_features(self.features)[2]:
            X[c] = X[c].astype(str)
        return X

    def fit(self, s, w=None):
        from catboost import CatBoostClassifier
        self.model = CatBoostClassifier(loss_function="Logloss", task_type="CPU", thread_count=4,
            random_seed=self.seed, monotone_constraints={"dist": -1, "angle_vis": 1},
            cat_features=split_features(self.features)[2], one_hot_max_size=16,
            allow_writing_files=False, verbose=False, **self.params)
        # No validation pool or stopping decision may see the comparison folds.
        self.model.fit(self._X(s), s.goal.to_numpy(), sample_weight=w)
        return self

    def predict(self, s):
        return self.model.predict_proba(self._X(s), thread_count=4)[:, 1]


class EBMxG:
    name = "ebm"

    def __init__(self, seed: int = 0, interactions: int = 10, max_rounds: int = 2000, features=None):
        self.seed = seed
        self.interactions = interactions
        self.max_rounds = max_rounds
        self.features = list(features or FEATURES)

    def _X(self, s):
        num_f, bin_f, cat_f = split_features(self.features)
        X = s[num_f + bin_f + cat_f].copy()
        for c in cat_f:
            X[c] = X[c].astype(str)
        return X

    def fit(self, s, w=None):
        from interpret.glassbox import ExplainableBoostingClassifier
        # монотонность: по расстоянию — убывание, по видимому углу — неубывание; пары
        # взаимодействий с расстоянием или углом запрещены, иначе взаимодействия ломают монотонность
        num_f, bin_f, cat_f = split_features(self.features)
        order = num_f + bin_f + cat_f
        mono = [(-1 if f == "dist" else (1 if f == "angle_vis" else 0)) for f in order]
        excl = [(a, b) for a in ("dist", "angle_vis") for b in order if b != a]
        self.model = ExplainableBoostingClassifier(random_state=self.seed, interactions=self.interactions,
                                                   max_rounds=self.max_rounds, n_jobs=model_threads(), outer_bags=4,
                                                   monotone_constraints=mono, exclude=excl,
                                                   feature_types=["continuous"] * len(num_f) + ["nominal"] * (len(bin_f) + len(cat_f)))
        self.model.fit(self._X(s), s.goal.values, sample_weight=w)
        try:
            self.model.monotonize("dist", increasing=False)
            self.model.monotonize("angle_vis", increasing=True)
            self.monotonized = True
        except Exception:
            self.monotonized = False
        return self

    def predict(self, s):
        return self.model.predict_proba(self._X(s))[:, 1]


class ByManpower:
    """Отдельные модели для равных составов / большинства / меньшинства."""

    def __init__(self, make_model, min_shots: int = 1500):
        self.make_model = make_model
        self.min_shots = min_shots
        self.name = "by_manpower"

    @staticmethod
    def group(s):
        return np.where(s.own_sk > s.opp_sk, "PP", np.where(s.own_sk < s.opp_sk, "SH", "EV"))

    def fit(self, s, w=None):
        g = self.group(s)
        self.models = {}
        self.fallback = self.make_model().fit(s, w)
        for k in ["EV", "PP", "SH"]:
            m = g == k
            if m.sum() >= self.min_shots and s.goal.values[m].sum() >= 50:
                self.models[k] = self.make_model().fit(s[m], None if w is None else w[m])
        return self

    def predict(self, s):
        g = self.group(s)
        p = self.fallback.predict(s)
        for k, mdl in self.models.items():
            m = g == k
            if m.any():
                p[m] = mdl.predict(s[m])
        return p


_STRENGTH_CACHE = {}


# Копии матча в бутстрэпе L6 называются <match_id>~k (l6_replicate.COPY_SEPARATOR): для
# целоматчевых фолдов это один и тот же матч.
BOOTSTRAP_COPY_SEPARATOR = "~"


def original_match(match_ids: pd.Series) -> pd.Series:
    """Исходный матч для копии из бутстрэпа: G0001~2 -> G0001; обычные ID не меняются."""
    return match_ids.astype(str).str.split(BOOTSTRAP_COPY_SEPARATOR, n=1).str[0]


def crossfit_strength(shots, features, seed, k=3):
    """A match's outcomes never enter its training-time opponent encoding.

    Stage zero and the opponent map for each held-out match group see only the
    remaining matches. The deployment map uses stage-zero out-of-match predictions.
    """
    from .cache import frame_hash
    cols = list(dict.fromkeys(list(features) + ["goal", "match_id", "opp_team_id"]))
    key = (frame_hash(shots, cols), tuple(features), seed, k)
    if key in _STRENGTH_CACHE:
        return _STRENGTH_CACHE[key]
    # Фолды по исходному матчу: копии одного матча в бутстрэпе не учатся на бросках «близнеца».
    groups = original_match(shots.match_id)
    mids = np.sort(groups.unique())
    if len(mids) < 2:
        raise ValueError("opp_def cross-fitting requires at least two matches")
    folds = dict(zip(mids, np.random.default_rng(seed).permutation(len(mids)) % min(k, len(mids))))
    f = groups.map(folds).to_numpy()
    enc, p0 = np.zeros(len(shots)), np.zeros(len(shots))
    for j in np.unique(f):
        train, val = f != j, f == j
        tr = shots.iloc[np.flatnonzero(train)].copy()
        va = shots.iloc[np.flatnonzero(val)].copy()
        tr["opp_def"], va["opp_def"] = 0.0, 0.0
        stage0 = MonoLogit(1.0, features).fit(tr)
        strength = team_def_strength(tr, stage0.predict(tr))
        enc[val] = va.opp_team_id.map(strength).fillna(0.0).to_numpy()
        p0[val] = stage0.predict(va)
    result = enc, team_def_strength(shots, p0)
    if len(_STRENGTH_CACHE) >= 32:
        _STRENGTH_CACHE.clear()
    _STRENGTH_CACHE[key] = result
    return result


class XGProcedure:
    """Cross-fitted opponent encoding and an empirical neutral-opponent scorer."""

    def __init__(self, spec: dict, seed: int = 0):
        self.spec = spec
        self.seed = seed

    def _make(self):
        sp = self.spec
        fam = sp["family"]
        feats = list(sp.get("features", FEATURES))
        if fam == "logit_spline":
            base = lambda: MonoLogit(sp.get("l2", 1.0), feats)
        elif fam == "lgbm_mono":
            base = lambda: LGBMxG(sp["params"], self.seed, feats)
        elif fam == "ebm":
            base = lambda: EBMxG(self.seed, features=feats)
        elif fam == "catboost_mono":
            base = lambda: CatBoostxG(sp["params"], self.seed, feats)
        else:
            raise ValueError(fam)
        if sp.get("by_manpower"):
            return ByManpower(base)
        return base()

    def fit(self, shots: pd.DataFrame):
        s = shots.copy()
        s["opp_def"] = 0.0
        feats = self.spec.get("features", FEATURES)
        if set(feats) & FORBIDDEN or not set(feats) <= set(FEATURES) or not set(CORE) <= set(feats):
            raise ValueError("xG feature contract violated")
        encoded, self.opp_def = crossfit_strength(s, feats, self.seed)
        s["opp_def"] = encoded
        self.opp_def_mean = float(s.opp_def.mean())
        self.strength_values, counts = np.unique(encoded, return_counts=True)
        self.strength_weights = counts / counts.sum()
        self.training_strength = encoded.copy()
        self.model = self._make().fit(s)
        # LightGBM is constant between strength split thresholds. Combining
        # empirical masses in those intervals is an exact integration shortcut.
        def split_points(model):
            if isinstance(model, ByManpower):
                return [t for m in [model.fallback, *model.models.values()] for t in split_points(m)]
            if isinstance(model, CatBoostxG):
                if "opp_def" not in model.features:
                    return []
                return model.model.get_borders().get(model.features.index("opp_def"), [])
            if not isinstance(model, LGBMxG) or "opp_def" not in model.features:
                return []
            feature = model.features.index("opp_def")
            points = []
            def visit(tree):
                if "split_index" in tree:
                    if tree["split_feature"] == feature:
                        points.append(float(tree["threshold"]))
                    visit(tree["left_child"])
                    visit(tree["right_child"])
            for tree in model.model.booster_.dump_model()["tree_info"]:
                visit(tree["tree_structure"])
            return points
        if isinstance(self.model, (LGBMxG, CatBoostxG, ByManpower)) and self.spec["family"] in ("lgbm_mono", "catboost_mono"):
            bins = np.searchsorted(np.unique(split_points(self.model)), self.strength_values, side="left")
            _, first, inverse = np.unique(bins, return_index=True, return_inverse=True)
            self.strength_values = self.strength_values[first]
            self.strength_weights = np.bincount(inverse, weights=self.strength_weights)
        return self

    def predict(self, shots: pd.DataFrame, use_mean_strength: bool = True) -> np.ndarray:
        s = shots.copy()
        if use_mean_strength:
            if "opp_def" not in self.spec.get("features", FEATURES):
                s["opp_def"] = 0.0
                return np.clip(self.model.predict(s), 1e-6, 1 - 1e-6)
            # Integrate probabilities, not a nonlinear model at the mean covariate.
            # Exact finite empirical sum; no test outcomes or team frequencies enter it.
            if isinstance(self.model, MonoLogit):
                s["opp_def"] = 0.0
                X, _, _, _ = _design(s, self.model.stats, self.model.features)
                z = self.model.beta[0] + X @ self.model.beta[1:]
                j = self.model.names.index("opp_def")
                num = [c for c in split_features(self.model.features)[0] if c not in ("dist", "angle_vis")]
                slope = self.model.beta[j + 1] / self.model.stats["sd"][num.index("opp_def")]
                p = np.zeros(len(s))
                for value, weight in zip(self.strength_values, self.strength_weights):
                    p += weight * expit(z + slope * value)
                return np.clip(p, 1e-6, 1 - 1e-6)
            if isinstance(self.model, LGBMxG):
                p = self.model.neutral_mixture(s, self.strength_values, self.strength_weights)
                return np.clip(p, 1e-6, 1 - 1e-6)
            p = np.zeros(len(s))
            for value, weight in zip(self.strength_values, self.strength_weights):
                s["opp_def"] = value
                p += weight * self.model.predict(s)
            return np.clip(p, 1e-6, 1 - 1e-6)
        else:
            s["opp_def"] = s.opp_team_id.map(self.opp_def).fillna(self.opp_def_mean)
        return np.clip(self.model.predict(s), 1e-6, 1 - 1e-6)


# ---------------------------------------------------------------------------------------
def metrics(y: np.ndarray, p: np.ndarray, blocked: np.ndarray | None = None, bins: int = 10) -> dict:
    from sklearn.metrics import roc_auc_score
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    br = float(np.mean((p - y) ** 2))
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    ece = 0.0
    calib = []
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
            calib.append({"bin": b, "n": int(m.sum()), "pred": float(p[m].mean()), "obs": float(y[m].mean())})
    out = {"n": int(len(y)), "goals": int(y.sum()), "xg_sum": float(p.sum()), "xg_over_goals": float(p.sum() / max(y.sum(), 1)),
           "logloss": ll, "brier": br, "ece": float(ece), "auc_all": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None, "calibration": calib}
    if blocked is not None:
        nb = ~blocked.astype(bool)
        out["auc_unblocked"] = float(roc_auc_score(y[nb], p[nb])) if len(np.unique(y[nb])) == 2 else None
        out["n_unblocked"] = int(nb.sum())
    return out


def paired_bootstrap_loss(match_ids: np.ndarray, loss_a: np.ndarray, loss_b: np.ndarray, B: int = 1000,
                          seed: int = 0) -> dict:
    """Разница средних потерь (a − b) с бутстрэпом по матчам (кластерный)."""
    match_ids, loss_a, loss_b = map(np.asarray, (match_ids, loss_a, loss_b))
    if (match_ids.ndim != 1 or loss_a.shape != match_ids.shape or loss_b.shape != match_ids.shape
            or not len(match_ids) or pd.isna(match_ids).any()
            or not np.isfinite(loss_a).all() or not np.isfinite(loss_b).all()):
        raise ValueError('Paired losses require aligned finite observations with match IDs')
    if B < 2 or int(B) != B:
        raise ValueError('At least two integer bootstrap replicates required')
    mid, inv = np.unique(match_ids, return_inverse=True)
    sa = np.bincount(inv, loss_a)
    sb = np.bincount(inv, loss_b)
    cnt = np.bincount(inv).astype(float)
    point = sa.sum() / cnt.sum() - sb.sum() / cnt.sum()
    if len(mid) < 2:
        return {'diff': float(point), 'lo95': None, 'hi95': None, 'p_a_better': None,
                'n_matches': len(mid), 'status': 'inconclusive'}
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(mid), np.full(len(mid), 1.0 / len(mid)), size=B).astype(float)
    da = (W @ sa) / (W @ cnt)
    db = (W @ sb) / (W @ cnt)
    d = da - db
    return {"diff": float(point), "lo95": float(np.quantile(d, 0.025)), "hi95": float(np.quantile(d, 0.975)),
            "p_a_better": float(np.mean(d < 0)), 'n_matches': len(mid), 'status': 'estimated'}


def xg_sum_interval(match_ids: np.ndarray, y: np.ndarray, p: np.ndarray, B: int = 1000, seed: int = 0) -> dict:
    """Отношение Σ xG / Σ голов с бутстрэпом по матчам (кластерный, как в сравнении потерь).

    Жёсткий порог на отношение игнорирует выборочный шум числа голов (~2,5% на блоке
    сравнения), поэтому смещение суммы признаётся только тогда, когда интервал не накрывает 1.
    """
    match_ids, y, p = map(np.asarray, (match_ids, y, p))
    if (match_ids.ndim != 1 or y.shape != match_ids.shape or p.shape != match_ids.shape
            or not len(match_ids) or pd.isna(match_ids).any() or not np.isfinite(p).all()):
        raise ValueError("xG sum interval requires aligned finite predictions with match IDs")
    if B < 2 or int(B) != B:
        raise ValueError("At least two integer bootstrap replicates required")
    mid, inv = np.unique(match_ids, return_inverse=True)
    sp = np.bincount(inv, p.astype(float))
    sy = np.bincount(inv, y.astype(float))
    ratio = float(sp.sum() / sy.sum()) if sy.sum() > 0 else float("nan")
    if len(mid) < 2 or sy.sum() <= 0:
        return {"ratio": ratio, "lo95": None, "hi95": None, "n_matches": len(mid), "status": "inconclusive"}
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(mid), np.full(len(mid), 1.0 / len(mid)), size=B).astype(float)
    goals = W @ sy
    r = (W @ sp)[goals > 0] / goals[goals > 0]
    return {"ratio": ratio, "lo95": float(np.quantile(r, 0.025)), "hi95": float(np.quantile(r, 0.975)),
            "n_matches": len(mid), "status": "estimated"}


def bootstrap_estimated(interval):
    """An unavailable interval cannot support acceptance or feature deletion."""
    lo, hi = interval.get('lo95'), interval.get('hi95')
    return bool(interval.get('status') == 'estimated' and lo is not None and hi is not None
                and np.isfinite(lo) and np.isfinite(hi) and lo <= hi)


def logloss_vec(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def probes(proc: XGProcedure, template: pd.DataFrame) -> dict:
    """Independent distance/angle probes across manpower/rinks; applicable time diagnostics."""
    base = template.iloc[[0]].copy()
    base = pd.concat([base] * 1, ignore_index=True)
    for c, v in {"prev_pass_dy": 0.0, "score_diff_c": 0, "own_sk": 5, "opp_sk": 5, "behind_net": 0, "rebound": 0,
                 "prev_pass_cross": 0, "recent_entry": 0, "own_g": 1, "opp_g": 1, "mp": "EV5",
                 "rink_type": "european_60x30", "period_c": "2", "t_remain_c": 600.0}.items():
        base[c] = v
    res = {}
    # 1) расстояние по оси ворот (y = 0)
    ds = np.linspace(1.5, 50, 60)
    g = pd.concat([base] * len(ds), ignore_index=True)
    g["dist"] = ds
    g["angle_vis"] = 2 * np.arctan2(0.915, ds)
    p = proc.predict(g)
    res["dist_monotone_nonincreasing"] = bool(np.all(np.diff(p) <= 1e-9))
    # 2) при фиксированном угле (монотонность по расстоянию при прочих равных)
    g2 = g.copy()
    g2["angle_vis"] = 0.1
    p2 = proc.predict(g2)
    res["dist_monotone_fixed_angle"] = bool(np.all(np.diff(p2) <= 1e-9))
    res["xg_at_dist"] = {f"{d:.0f}m": float(v) for d, v in zip(ds[::10], p[::10])}
    # Angle varies independently of distance; distance probes cannot establish this constraint.
    angle_checks = []
    angles = np.linspace(.001, .7, 40)
    for distance in (3., 10., 30., 45.):
        for mp, own, opp, own_g, opp_g in (
            ("EV5", 5, 5, 1, 1), ("EVo", 3, 3, 1, 1), ("PP", 5, 4, 1, 1), ("SH", 4, 5, 1, 1),
            ("ENF", 5, 6, 1, 0), ("ENA", 6, 5, 0, 1)):
            for rink in ("european_60x30", "nhl_61x26", "finnish_60x28"):
                a = pd.concat([base] * len(angles), ignore_index=True)
                a["dist"], a["angle_vis"] = distance, angles
                a["mp"], a["own_sk"], a["opp_sk"] = mp, own, opp
                a["own_g"], a["opp_g"], a["rink_type"] = own_g, opp_g, rink
                angle_checks.append(a)
    ag = pd.concat(angle_checks, ignore_index=True)
    ap = proc.predict(ag).reshape(len(angle_checks), len(angles))
    finite = np.isfinite(ap).all()
    res["angle_monotone_fixed_distance"] = bool(finite and np.all(np.diff(ap, axis=1) >= -1e-9))
    res["angle_probe_contexts"] = len(angle_checks)
    res["angle_probe_min_step"] = float(np.min(np.diff(ap, axis=1))) if finite else None
    # 3) время периода: нет скачков — максимальный относительный шаг по сетке 10 с
    #    (время периода не входит в признаки; проба подтверждает это на уровне модели)
    if "t_remain_c" not in getattr(proc, "spec", {}).get("features", FEATURES):
        res.update(period_time_probe_status="not_applicable", period_time_no_jumps=None,
                   period_time_reason="period clock is not an xG feature")
        return res
    ts = np.linspace(0, 1200, 121)
    g3 = pd.concat([base] * len(ts), ignore_index=True)
    g3["dist"] = 10.0
    g3["angle_vis"] = 2 * np.arctan2(0.915, 10.0)
    g3["t_remain_c"] = ts
    p3 = proc.predict(g3)
    rel = np.abs(np.diff(p3)) / np.maximum(p3[:-1], 1e-6)
    res["period_time_max_rel_step"] = float(rel.max())
    res["period_time_range_rel"] = float((p3.max() - p3.min()) / p3.mean())
    res["period_time_no_jumps"] = bool(rel.max() < 0.10)
    return res
