"""L7: соответствуют ли метки HIGH/MEDIUM/LOW реальной устойчивости и накрывают ли интервалы.

1. Метки надёжности. Модель L5 для Overall: x_i = θ_i + e_i, θ ~ (μ_r, τ²_r), θ̂ = E[θ | x];
   w_i = 1 − v_i/τ²_r — доля собственного сигнала (метка: HIGH при w ≥ 0,7, MEDIUM при ≥ 0,4).
   Для оценки θ̂_A по половине матчей A и ставки x_B по независимой половине B при верных
   моментах E[(θ̂_A − μ)(x_B − μ)] = Cov(θ̂_A, θ) = Var(θ̂_A) = w·τ². Отсюда «реализованная
   надёжность» группы R = Σ t (θ̂_A − μ)(x_B − μ)/τ² / Σ t (t — будущий TOI) — несмещённая оценка
   среднего w группы, если модель верна; наклон x_B на θ̂_A должен быть равен 1 (< 1 — оценки
   растянуты, стягивания мало; > 1 — стягивание избыточно). Проверка не использует test.

2. Предсказательное покрытие на реальных данных. Интервал будущей ставки
   θ̂_A ± 1,96·√(v_A + σ²/T_B + ω²/M_B + δ²) должен накрывать x_B в 95% случаев. В train
   половины случайны, дрейфа нет: δ² = 0. Train → test — диагностика с сеткой δ².

3. Симуляция покрытия интервала скрытого эффекта в двух вариантах оценки моментов:
   по выборке структуры train (как в конвейере: L5 оценивает моменты на всём train и применяет
   их к test) и по самой выборке структуры test (вариант L6 — пессимистичный, 562 игрока
   с ~4 матчами). «Оракул» — с истинными моментами. Интервалы по итогам не расширяются
   (правило методики L6); результат — описание свойств, не настройка.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from . import stage_l5
from . import validation as va
from .config import get_logger
from .l5_shrink import METHOD_UNIVARIATE, ROLES, JointEB, UnivariateEB
from .l6_uncertainty import _simulated_player_match
from .validation import _ci

Z95 = 1.96
LABELS = ("HIGH", "MEDIUM", "LOW")
HIGH_DEFAULT, MEDIUM_DEFAULT = 0.7, 0.4
# Интервалы w для детальной таблицы (левая граница включительно).
W_BINS = (0.0, 0.2, 0.4, 0.55, 0.7, 0.85, 1.0000001)
# Интервалы будущего TOI, мин, для покрытия.
TOI_BINS_MIN = (0.0, 30.0, 60.0, 150.0, 300.0, np.inf)
DRIFT_GRID = (0.0, 0.05, 0.1, 0.2)
PAIR_COLS = ["direction", "player_id", "role", "theta", "mu", "tau2", "w", "var_post", "reliability",
             "x_fut", "toi_fut_min", "games_fut", "sigma2", "omega2"]


# --- пары «оценка по прошлому → ставка в будущем» -------------------------------------------

def o1_detail(pm_past: pd.DataFrame, shrink: Mapping) -> tuple[pd.DataFrame, JointEB]:
    """O1 игроков `pm_past` моделью, подогнанной на `pm_past` (как `rating_validation.o1_rating`)."""
    roles = stage_l5.player_roles(pm_past)
    rf, _ = stage_l5.rate_frame(pm_past, stage_l5.COMP_COLS)
    eb = JointEB(stage_l5.COMP_COLS, dict(shrink), logger=get_logger()).fit(rf, roles)
    ids = sorted(pm_past.player_id.unique())
    df, _ = stage_l5.apply_o1(eb, stage_l5.player_sums(rf, stage_l5.COMP_COLS), roles, ids,
                              {"shrink": dict(shrink)})
    return df, eb


def overall_rates(pm: pd.DataFrame) -> pd.DataFrame:
    """Ставка Overall (Σ компонентов на 60), TOI в минутах и матчи по допущенной экспозиции."""
    rf, _ = stage_l5.rate_frame(pm, stage_l5.COMP_COLS)
    s = stage_l5.player_sums(rf, stage_l5.COMP_COLS)
    return pd.DataFrame({"x_fut": s[stage_l5.COMP_COLS].sum(axis=1) / s.toi_s * 3600.0,
                         "toi_fut_min": s.toi_s / 60.0, "games_fut": s.games}, index=s.index)


def make_pairs(pm_past: pd.DataFrame, pm_future: pd.DataFrame, shrink: Mapping, direction: str,
               ids=None) -> pd.DataFrame:
    """θ̂, w, v, метка по прошлому + ставка Overall по будущему; только игроки с экспозицией в обоих."""
    det, eb = o1_detail(pm_past, shrink)
    det = det[(det.toi_min > 0) & (det.games > 0)].set_index("player_id")
    fut = overall_rates(pm_future)
    keep = det.index.intersection(fut.index)
    if ids is not None:
        keep = keep.intersection(pd.Index(ids))
    det, fut = det.loc[keep], fut.loc[keep]
    roles = det.role.to_numpy().astype(str)
    uni = det.method.to_numpy() == METHOD_UNIVARIATE
    tau_joint = np.array([eb.T[r].sum() for r in roles], dtype=float)
    tau_uni = np.array([max(eb.uni[r]["tau2"], 0.0) for r in roles], dtype=float)
    out = pd.DataFrame({
        "direction": direction, "player_id": keep, "role": roles, "theta": det.theta.to_numpy(float),
        "mu": [float(eb.mu[r].sum()) for r in roles], "tau2": np.where(uni, tau_uni, tau_joint),
        "w": det.w.to_numpy(float), "var_post": det.var_post.to_numpy(float),
        "reliability": det.reliability.to_numpy(), "x_fut": fut.x_fut.to_numpy(float),
        "toi_fut_min": fut.toi_fut_min.to_numpy(float), "games_fut": fut.games_fut.to_numpy(float),
        "sigma2": [float(eb.uni[r]["sigma2"]) for r in roles], "omega2": [float(eb.uni[r]["omega2"]) for r in roles],
        "toi_past_min": det.toi_min.to_numpy(float)})
    return out.reset_index(drop=True)


def half_pairs(pm: pd.DataFrame, shrink: Mapping, seed: int = va.SEED) -> pd.DataFrame:
    """Train: модель по половине A → ставка по B, и наоборот (случайные половины матчей)."""
    halves = va.assign_halves(pm.match_id, seed)
    h = pm.match_id.map(halves).to_numpy(dtype=int)
    pm_a, pm_b = pm[h == 0], pm[h == 1]
    return pd.concat([make_pairs(pm_a, pm_b, shrink, "A->B"), make_pairs(pm_b, pm_a, shrink, "B->A")],
                     ignore_index=True)


def cross_split_pairs(pm_train: pd.DataFrame, pm_test: pd.DataFrame, shrink: Mapping,
                      min_games: int = 5) -> pd.DataFrame:
    """Train → test: модель и θ̂ по всему train, ставка — в test; ≥ `min_games` матчей в обоих."""
    g_tr = pm_train[pm_train.toi_rate_s > 0].groupby("player_id").match_id.nunique()
    g_te = pm_test[pm_test.toi_rate_s > 0].groupby("player_id").match_id.nunique()
    ids = g_tr.index[g_tr >= min_games].intersection(g_te.index[g_te >= min_games])
    return make_pairs(pm_train, pm_test, shrink, "train->test", ids)


# --- бутстрэп игроков -----------------------------------------------------------------------

def _player_boot(pairs: pd.DataFrame, n_boot: int, seed: int):
    """Индексы строк бутстрэпа игроков (стратификация по амплуа; все строки игрока вместе)."""
    codes, uniq = pd.factorize(pairs.player_id.to_numpy())
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    rows = [order[starts[k]:starts[k + 1]] for k in range(len(uniq))]
    prole = pairs.groupby(codes).role.first().to_numpy().astype(str)
    by_role = [np.flatnonzero(prole == r) for r in np.unique(prole)]
    rng = np.random.default_rng(seed + 17)
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in by_role]) if by_role else []
        yield np.concatenate([rows[k] for k in pick]) if len(pick) else np.zeros(0, dtype=int)


# --- 1. метки надёжности --------------------------------------------------------------------

def _rel_stats(p: pd.DataFrame) -> dict:
    t = p.toi_fut_min.to_numpy(float)
    a = p.theta.to_numpy(float) - p.mu.to_numpy(float)
    b = p.x_fut.to_numpy(float) - p.mu.to_numpy(float)
    tau2 = p.tau2.to_numpy(float)
    st = t.sum()
    if not len(p) or st <= 0:
        return {"n": int(len(p)), "mean_w": np.nan, "model_reliability": np.nan,
                "realized_reliability": np.nan, "calibration_slope": np.nan}
    saa = float((t * a * a).sum())
    return {"n": int(len(p)), "mean_w": float((t * p.w.to_numpy(float)).sum() / st),
            "model_reliability": float((t * a * a / tau2).sum() / st),
            "realized_reliability": float((t * a * b / tau2).sum() / st),
            "calibration_slope": float((t * a * b).sum() / saa) if saa > 0 else np.nan}


def _label_range(label: str, high: float, medium: float) -> list[float]:
    return {"HIGH": [high, 1.0], "MEDIUM": [medium, high], "LOW": [0.0, medium]}[label]


def _w_bin(w: np.ndarray) -> np.ndarray:
    k = np.clip(np.digitize(w, W_BINS) - 1, 0, len(W_BINS) - 2)
    return np.array([f"[{W_BINS[i]:.2f}; {min(W_BINS[i + 1], 1.0):.2f})" for i in k], dtype=object)


def _group_masks(p: pd.DataFrame) -> dict:
    return {"pooled": {"all": np.ones(len(p), dtype=bool)},
            "by_label": {lab: (p.reliability == lab).to_numpy() for lab in LABELS},
            "by_w_bin": {b: (p.w_bin == b).to_numpy() for b in sorted(p.w_bin.unique())}}


def label_calibration(pairs: pd.DataFrame, n_boot: int = va.N_BOOT, seed: int = va.SEED,
                      high: float = HIGH_DEFAULT, medium: float = MEDIUM_DEFAULT) -> dict:
    """Реализованная надёжность и наклон калибровки по меткам, по интервалам w и объединённо."""
    va._require(pairs, PAIR_COLS, "label_calibration")
    p = pairs[(pairs.tau2 > 0) & np.isfinite(pairs.tau2)].reset_index(drop=True)
    p = p.assign(w_bin=_w_bin(p.w.to_numpy(float)))
    groups = _group_masks(p)
    point = {k: {g: _rel_stats(p[m]) for g, m in gs.items()} for k, gs in groups.items()}
    boots = []
    for ix in _player_boot(p, n_boot, seed):
        q = p.iloc[ix].reset_index(drop=True)
        boots.append({k: {g: _rel_stats(q[m]) for g, m in gs.items()} for k, gs in _group_masks(q).items()})
    out = {}
    for k, gs in groups.items():
        out[k] = {}
        for g in gs:
            row = dict(point[k][g])
            for s in ("realized_reliability", "calibration_slope"):
                row[f"{s}_ci"] = _ci([b[k].get(g, {}).get(s, np.nan) for b in boots])
            if k == "by_label":
                rng_ = _label_range(g, high, medium)
                lo, hi = row["realized_reliability_ci"]
                row["label_range"] = rng_
                row["consistent"] = bool(np.isfinite(lo) and lo <= rng_[1] and hi >= rng_[0])
            out[k][g] = row
    out["pooled"] = out["pooled"]["all"]
    return {**out, "n_rows": int(len(p)), "n_players": int(p.player_id.nunique()), "n_boot": n_boot,
            "weights": "будущий TOI", "thresholds": {"HIGH": high, "MEDIUM": medium},
            "bootstrap": "players, stratified by role (both directions of a player together)"}


# --- 2. предсказательное покрытие на реальных данных ----------------------------------------

def _pred_sd(p: pd.DataFrame, drift2: float) -> np.ndarray:
    """√(v + σ²/T_f + ω²/M_f + δ²); T_f в минутах — те же единицы, что у моментов L5 и L6."""
    return np.sqrt(p.var_post.to_numpy(float) + p.sigma2.to_numpy(float) / p.toi_fut_min.to_numpy(float)
                   + p.omega2.to_numpy(float) / np.maximum(p.games_fut.to_numpy(float), 1.0) + drift2)


def _cov(hit: np.ndarray, half: np.ndarray) -> dict:
    return {"n": int(len(hit)), "coverage": float(hit.mean()) if len(hit) else np.nan,
            "mean_halfwidth": float(half.mean()) if len(hit) else np.nan}


def predictive_coverage(pairs: pd.DataFrame, drift2: float = 0.0, z: float = Z95, n_boot: int = va.N_BOOT,
                        seed: int = va.SEED) -> dict:
    """Доля x_B внутри θ̂_A ± z·√(v_A + σ²/T_B + ω²/M_B + δ²): объединённо, по меткам, амплуа, TOI."""
    va._require(pairs, PAIR_COLS, "predictive_coverage")
    p = pairs.reset_index(drop=True)
    half = z * _pred_sd(p, drift2)
    hit = np.abs(p.x_fut.to_numpy(float) - p.theta.to_numpy(float)) <= half
    tbin = pd.cut(p.toi_fut_min, TOI_BINS_MIN, right=False).astype(str).to_numpy()
    out = {"pooled": _cov(hit, half), "delta2": drift2, "nominal": 0.95}
    out["pooled"]["coverage_ci"] = _ci([hit[ix].mean() for ix in _player_boot(p, n_boot, seed)])
    for name, key in (("by_label", p.reliability.to_numpy()), ("by_role", p.role.to_numpy()),
                      ("by_future_toi_min", tbin)):
        out[name] = {str(g): _cov(hit[key == g], half[key == g]) for g in pd.unique(key)}
    return out


def coverage_grid(pairs: pd.DataFrame, grid=DRIFT_GRID, n_boot: int = va.N_BOOT, seed: int = va.SEED) -> dict:
    return {str(d): predictive_coverage(pairs, d, n_boot=n_boot, seed=seed) for d in grid}


# --- 3. симуляция покрытия с двумя вариантами оценки моментов -------------------------------

def _labels(w: np.ndarray, high: float, medium: float) -> np.ndarray:
    return np.where(w >= high, "HIGH", np.where(w >= medium, "MEDIUM", "LOW")).astype(object)


def _sim_sums(pm: pd.DataFrame, n: int) -> pd.DataFrame:
    ids = pd.Index([f"S{i:05d}" for i in range(n)])
    return pm.groupby("player_id").agg(y=("y", "sum"), toi_s=("toi_s", "sum"),
                                       games=("match_id", "nunique")).reindex(ids)


class _Tally:
    """Счётчик попаданий по ключам «all», «role:X», «label:Y»."""

    def __init__(self):
        self.hit, self.n = {}, {}

    def add(self, key: str, hit: np.ndarray):
        self.hit[key] = self.hit.get(key, 0) + int(hit.sum())
        self.n[key] = self.n.get(key, 0) + int(len(hit))

    def rate(self, key: str) -> dict:
        n = self.n.get(key, 0)
        return {"n": n, "coverage": self.hit[key] / n if n else np.nan}

    def summary(self) -> dict:
        out = self.rate("all")
        for prefix, name in (("role:", "by_role"), ("label:", "by_label")):
            out[name] = {k[len(prefix):]: self.rate(k) for k in sorted(self.n) if k.startswith(prefix)}
        return out


def _record(t: _Tally, theta, th, v, roles, labels, z):
    hit = np.abs(theta - th) <= z * np.sqrt(np.maximum(v, 0.0))
    t.add("all", hit)
    for r in np.unique(roles):
        t.add(f"role:{r}", hit[roles == r])
    for lab in np.unique(labels):
        t.add(f"label:{lab}", hit[labels == lab])


def _structure(toi_min, games, roles, moments: Mapping):
    T = np.asarray(toi_min, float)
    M = np.maximum(np.round(np.asarray(games, float)), 1.0)
    R = np.asarray(roles, dtype=object)
    ok = (T > 0) & np.isin(R, list(moments))
    if not ok.any():
        raise ValueError("simulate_two_stage_coverage: пустая выборка fit или eval")
    return T[ok], M[ok], R[ok], pd.Series(R[ok], index=[f"S{i:05d}" for i in range(int(ok.sum()))])


def simulate_two_stage_coverage(fit_toi_min, fit_games, fit_roles, eval_toi_min, eval_games, eval_roles,
                                moments: Mapping, n_sim: int = 50, seed: int = 0, min_toi_s: float = 300.0,
                                eps_abs: float = 1e-10, high: float = HIGH_DEFAULT,
                                medium: float = MEDIUM_DEFAULT, z: float = Z95, noise_by_role: bool = False) -> dict:
    """Покрытие интервала скрытого θ: моменты — по выборке структуры «fit», интервалы — для «eval».

    На каждой итерации: θ ~ N(0, τ²_r) и наблюдения по матчам для обеих выборок (та же модель
    шума, что у `l6_uncertainty.simulate_refit_coverage`); моменты `UnivariateEB` оцениваются
    (а) на выборке fit (как в конвейере: моменты train → test), (б) на самой выборке eval
    (вариант L6), (в) берутся истинными («оракул»). Доля θ внутри θ̂ ± z√v по каждому варианту,
    по амплуа и по метке, построенной из w того же варианта.
    """
    T_f, M_f, R_f, roles_f = _structure(fit_toi_min, fit_games, fit_roles, moments)
    T_e, M_e, R_e, roles_e = _structure(eval_toi_min, eval_games, eval_roles, moments)
    tau2_f = np.array([float(moments[r]["tau2"]) for r in R_f])
    t2_e = np.array([float(moments[r]["tau2"]) for r in R_e])
    s_true = (np.array([moments[r]["sigma2"] for r in R_e], float) / T_e
              + np.array([moments[r]["omega2"] for r in R_e], float) / M_e)
    w_o = t2_e / (t2_e + s_true)
    rng = np.random.default_rng(int(seed))
    tallies = {k: _Tally() for k in ("oracle_moments", "refit_on_fit_structure", "refit_on_eval_structure")}
    for _ in range(int(n_sim)):
        pm_f = _simulated_player_match(rng.normal(0.0, 1.0, len(T_f)) * np.sqrt(tau2_f), T_f, M_f, R_f, moments, rng)
        eb_fit = UnivariateEB(eps_abs, noise_by_role=noise_by_role).fit(pm_f, "y", roles_f, min_toi_s=min_toi_s)
        th_e = rng.normal(0.0, 1.0, len(T_e)) * np.sqrt(t2_e)
        pm_e = _simulated_player_match(th_e, T_e, M_e, R_e, moments, rng)
        sums = _sim_sums(pm_e, len(T_e))
        x = sums.y.to_numpy() / sums.toi_s.to_numpy() * 3600.0
        _record(tallies["oracle_moments"], th_e, w_o * x, t2_e * s_true / (t2_e + s_true), R_e,
                _labels(w_o, high, medium), z)
        eb_eval = UnivariateEB(eps_abs, noise_by_role=noise_by_role).fit(pm_e, "y", roles_e, min_toi_s=min_toi_s)
        for key, eb in (("refit_on_fit_structure", eb_fit), ("refit_on_eval_structure", eb_eval)):
            th, v, w = eb.posterior(R_e, sums.y.to_numpy(), sums.toi_s.to_numpy(), sums.games.to_numpy())
            _record(tallies[key], th_e, th, v, R_e, _labels(w, high, medium), z)
    return {**{k: t.summary() for k, t in tallies.items()}, "n_sim": int(n_sim),
            "n_fit_players": int(len(T_f)), "n_eval_players": int(len(T_e)), "nominal": 0.95,
            "estimator": "l5_shrink.UnivariateEB (шкала суммы)",
            "moments_true": {r: dict(moments[r]) for r in moments}}


def player_structure(pm: pd.DataFrame) -> pd.DataFrame:
    """T_i (мин), M_i и амплуа игроков с допущенной экспозицией — структура для симуляции."""
    rf, _ = stage_l5.rate_frame(pm, stage_l5.COMP_COLS)
    s = stage_l5.player_sums(rf, stage_l5.COMP_COLS)
    roles = stage_l5.player_roles(pm).reindex(s.index)
    out = pd.DataFrame({"toi_min": s.toi_s / 60.0, "games": s.games, "role": roles})
    return out[out.role.isin(ROLES)]
