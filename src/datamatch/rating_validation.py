"""L7: «наш рейтинг против базовых» — устойчивость и прогноз будущей игры на одних разбиениях.

Смысл. Рейтинг полезен, если (1) он повторяется на непересекающихся матчах и (2) рейтинг,
построенный по одним матчам, предсказывает игру в других. Здесь O1 (совместное эмпирическое
стягивание компонентов L5) и O0 (стянутый on-ice xGD/60 при 5 на 5) сравниваются с простыми
показателями из `baselines.py` на ОДНИХ И ТЕХ ЖЕ половинах матчей и одних и тех же игроках.

Вход — `player_match` L3 (одна строка «игрок × матч»: `c_*`, `iav_sum`, `toi_rate_s`,
`toi_rate_5v5_s`, `xgf_5v5`, `xga_5v5`, `points`, `ixg`, `ishots`; необязательно `gf_5v5`/`ga_5v5`).

Разбиение train — `validation.assign_halves` (фиксированный seed, порядок match_id — не время).
Каждая функция рейтинга получает только строки «прошлой» половины и сама подгоняет на ней всё
обучаемое: O1/O0 — моменты T_r, Σσ, Σω, τ², μ_r реальным кодом L5 (`JointEB`, `apply_o1`,
`l5_o0.fit_o0`/`rate_o0`), стянутые базовые — моменты `baselines.fit_role_shrinkage`. Цель —
ставка на 60 той же величины на другой половине:

* `xgd_5v5_60` — on-ice xGD/60 при 5 на 5 (общая цель турнира, замороженный xG L1);
* `gd_5v5_60` — on-ice GD/60 при 5 на 5 (если есть `gf_5v5`/`ga_5v5`; очень шумная);
* `iav_60` — собственный кредит IAV/60: повторяемость собственной величины O1, не честное сравнение.

Метрики — `validation.compare_ratings` (Spearman и TOI-взвешенный Pearson, бутстрэп игроков со
стратификацией по амплуа, парные разности на одних бутстрэп-выборках) и split-half рейтингов с
поправкой Спирмена–Брауна. Test (хронология по game_seq) — только диагностика: модели обучены
на всём train и заморожены, по test ничего не выбирается.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from . import baselines as bl
from . import l5_o0, stage_l5
from . import validation as va
from .config import get_logger
from .l5_shrink import JointEB, UnivariateEB
from .validation import _boot_indices, _ci, _group_stats

RatingFn = Callable[[pd.DataFrame], pd.Series]

# Рейтинг-ориентир: разности считаются как «O1 − другой рейтинг».
REFERENCE = "O1"
# Базовая строка M0 турнира L7.
M0_RATING = "points_60_raw"
# Основная цель прогноза (общая цель турнира).
MAIN_TARGET = "xgd_5v5_60"
IAV_METRIC = bl.RateMetric("iav_60", ("iav_sum",), (), "toi_rate_s",
                           "IAV/60 (собственный кредит O1; только повторяемость, не честное сравнение)")
TARGET_ORDER = ("xgd_5v5_60", "gd_5v5_60", "iav_60")
# Сырые и стянутые базовые показатели (берутся те, для которых есть колонки).
RAW_BASELINES = ("points_60", "gd_5v5_60", "ixg_60", "ishots_60", "xgd_5v5_60")
SHRUNK_BASELINES = ("points_60", "gd_5v5_60", "ixg_60")
PM_REQUIRED = ["match_id", "player_id", "role", "toi_rate_s"] + stage_l5.COMP_COLS + l5_o0.O0_REQUIRED[2:]
# Колонки on-ice голов из L0-таблицы baselines, которых нет в L3 player_match.
ONICE_GOAL_COLS = ["gf", "ga", "gf_5v5", "ga_5v5"]
STATS = ("spearman", "wpearson")


def require_columns(df: pd.DataFrame, cols: Iterable[str], what: str) -> None:
    """Проверка схемы на границе модуля: пропуск колонки — ошибка, а не подстановка."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{what}: нет колонок {missing}")


def _cfg(shrink: Mapping) -> dict:
    """Минимальная конфигурация для функций L5 (они читают только `cfg["shrink"]`)."""
    return {"shrink": dict(shrink)}


def _series(df: pd.DataFrame, col: str, name: str) -> pd.Series:
    return pd.Series(df[col].to_numpy(dtype=float), index=pd.Index(df.player_id, name="player_id"), name=name)


# --- разбиение -----------------------------------------------------------------------------

def split_halves(pm: pd.DataFrame, seed: int = va.SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Половины A (0) и B (1) матчей train: (pm_a, pm_b, halves match_id → 0/1); вход не меняется."""
    require_columns(pm, ["match_id"], "split_halves")
    halves = va.assign_halves(pm.match_id, seed)
    h = pm.match_id.map(halves).to_numpy(dtype=int)
    return pm[h == 0], pm[h == 1], halves


def temporal_halves(games: pd.DataFrame, seq_col: str = "game_seq") -> tuple[set, set]:
    """Первая и вторая половина сезона по `seq_col` (как `validation.temporal_check`)."""
    require_columns(games, ["match_id", seq_col], "games")
    g = games.dropna(subset=[seq_col]).sort_values(seq_col)
    n_early = len(g) // 2
    return set(g.match_id.iloc[:n_early]), set(g.match_id.iloc[n_early:])


# --- наш рейтинг: O1 и O0 реальным кодом L5 ------------------------------------------------

@dataclass(frozen=True)
class FrozenModels:
    """Модели L5, обученные на одной выборке и применяемые к другой (μ_r — по КХЛ-компоненте)."""
    o1: JointEB
    o0: UnivariateEB
    roles: pd.Series
    shrink: dict
    mu_source: str


def fit_frozen_models(pm_train: pd.DataFrame, shrink: Mapping) -> FrozenModels:
    """Как `stage_l5.run_l5` для test: моменты по train, μ_r — по КХЛ-компоненте train (если есть)."""
    require_columns(pm_train, PM_REQUIRED, "fit_frozen_models")
    cfg = _cfg(shrink)
    roles = stage_l5.player_roles(pm_train)
    rf, _ = stage_l5.rate_frame(pm_train, stage_l5.COMP_COLS)
    eb = JointEB(stage_l5.COMP_COLS, cfg["shrink"], logger=get_logger()).fit(rf, roles)
    mu_src, mu_source, _ = stage_l5._mu_source(pm_train, stage_l5.COMP_COLS)
    eb_test = eb.with_mu(eb.prior_mu(stage_l5.rate_frame(mu_src, stage_l5.COMP_COLS)[0], roles))
    eb0, _ = l5_o0.fit_o0(cfg, pm_train, roles)
    eb0_test = eb0.with_mu(eb0.prior_mu(l5_o0.o0_frame(mu_src)[0], roles))
    return FrozenModels(eb_test, eb0_test, roles, dict(shrink), mu_source)


def _eval_roles(pm: pd.DataFrame, fallback: pd.Series | None) -> pd.Series:
    roles = stage_l5.player_roles(pm)
    return roles if fallback is None else roles.combine_first(fallback)


def apply_o1_rating(pm: pd.DataFrame, eb: JointEB, shrink: Mapping,
                    roles_fallback: pd.Series | None = None) -> pd.Series:
    """Overall O1 игроков `pm` подогнанной моделью (`stage_l5.apply_o1`, m_r — по этой выборке)."""
    roles = _eval_roles(pm, roles_fallback)
    ids = sorted(pm.player_id.unique())
    rf, _ = stage_l5.rate_frame(pm, eb.cols)
    df, _ = stage_l5.apply_o1(eb, stage_l5.player_sums(rf, eb.cols), roles, ids, _cfg(shrink))
    return _series(df, "rating", "O1")


def apply_o0_rating(pm: pd.DataFrame, eb0: UnivariateEB, shrink: Mapping,
                    roles_fallback: pd.Series | None = None) -> pd.Series:
    """O0 игроков `pm` подогнанной одномерной моделью (`l5_o0.rate_o0`)."""
    roles = _eval_roles(pm, roles_fallback)
    df, _ = l5_o0.rate_o0(eb0, pm, roles, sorted(pm.player_id.unique()), _cfg(shrink))
    return _series(df, "rating", "O0")


def o1_rating(pm_half: pd.DataFrame, shrink: Mapping) -> pd.Series:
    """O1, полностью подогнанный на `pm_half` (моменты и μ_r только по этой половине)."""
    require_columns(pm_half, PM_REQUIRED, "o1_rating")
    roles = stage_l5.player_roles(pm_half)
    rf, _ = stage_l5.rate_frame(pm_half, stage_l5.COMP_COLS)
    eb = JointEB(stage_l5.COMP_COLS, dict(shrink), logger=get_logger()).fit(rf, roles)
    return apply_o1_rating(pm_half, eb, shrink)


def o0_rating(pm_half: pd.DataFrame, shrink: Mapping) -> pd.Series:
    """O0, полностью подогнанный на `pm_half`."""
    require_columns(pm_half, PM_REQUIRED, "o0_rating")
    eb0, _ = l5_o0.fit_o0(_cfg(shrink), pm_half, stage_l5.player_roles(pm_half))
    return apply_o0_rating(pm_half, eb0, shrink)


def our_rating_fns(shrink: Mapping) -> dict[str, RatingFn]:
    """O1 и O0, подгоняемые на переданной половине."""
    return {"O1": partial(o1_rating, shrink=dict(shrink)), "O0": partial(o0_rating, shrink=dict(shrink))}


def frozen_rating_fns(models: FrozenModels) -> dict[str, RatingFn]:
    """O1 и O0 с замороженными (обученными на train) моментами — для хронологии test."""
    return {"O1": partial(apply_o1_rating, eb=models.o1, shrink=models.shrink, roles_fallback=models.roles),
            "O0": partial(apply_o0_rating, eb0=models.o0, shrink=models.shrink, roles_fallback=models.roles)}


# --- базовые рейтинги ----------------------------------------------------------------------

def raw_rating(pm_half: pd.DataFrame, metric: str) -> pd.Series:
    """Сырая ставка на 60 (NaN у игрока без экспозиции показателя)."""
    r = bl.player_rates(bl.metric_frame(pm_half, bl.METRICS[metric]))
    return _series(r, "rate_60", f"{metric}_raw")


def shrunk_rating(pm_half: pd.DataFrame, metric: str, params: bl.ShrinkParams | None = None) -> pd.Series:
    """Ставка, стянутая к среднему амплуа; моменты — по `pm_half`, если не заданы."""
    r = bl.shrunk_rates(bl.metric_frame(pm_half, bl.METRICS[metric]), params)
    return _series(r, "rate_shrunk", f"{metric}_shrunk")


def _available(pm: pd.DataFrame, names: Iterable[str]) -> list[str]:
    avail = set(bl.available_metrics(pm))
    return [m for m in names if m in avail]


def baseline_rating_fns(pm: pd.DataFrame, shrink_params: Mapping[str, bl.ShrinkParams] | None = None
                        ) -> dict[str, RatingFn]:
    """Базовые рейтинги по колонкам `pm`; `shrink_params` — замороженные моменты (для test)."""
    fns: dict[str, RatingFn] = {f"{m}_raw": partial(raw_rating, metric=m) for m in _available(pm, RAW_BASELINES)}
    for m in _available(pm, SHRUNK_BASELINES):
        params = None if shrink_params is None else shrink_params[m]
        fns[f"{m}_shrunk"] = partial(shrunk_rating, metric=m, params=params)
    return fns


def baseline_shrink_params(pm_train: pd.DataFrame) -> dict[str, bl.ShrinkParams]:
    """Моменты стягивания базовых показателей по train (для применения к test)."""
    return {m: bl.fit_role_shrinkage(bl.metric_frame(pm_train, bl.METRICS[m]))
            for m in _available(pm_train, SHRUNK_BASELINES)}


def attach_onice_goals(pm_l3: pd.DataFrame, pm_l0: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Копия L3 player_match с on-ice голами и rate-согласованными очками из L0-таблицы baselines.

    `points` L3 считает все голы/передачи матча, а знаменатель — `toi_rate_s`; очки baselines —
    только в допущенных stints (та же выборка, что TOI), поэтому M0 берёт их. Отсутствующие в L0
    строки получают 0 (нет допущенной экспозиции). Возвращает (таблица, сверка TOI).
    """
    require_columns(pm_l0, bl.KEY + ONICE_GOAL_COLS + ["points", "toi_rate_s", "toi_rate_5v5_s"], "pm_l0")
    add = pm_l0[bl.KEY + ONICE_GOAL_COLS + ["points", "toi_rate_s", "toi_rate_5v5_s"]].rename(
        columns={"toi_rate_s": "_toi_l0", "toi_rate_5v5_s": "_toi5_l0"})
    base = pm_l3.drop(columns=[c for c in ONICE_GOAL_COLS + ["points"] if c in pm_l3.columns])
    out = base.merge(add, on=bl.KEY, how="left", validate="one_to_one")
    matched = out._toi_l0.notna().to_numpy()
    check = {"rows_l3": int(len(pm_l3)), "rows_matched_l0": int(matched.sum()),
             "max_abs_toi_rate_diff_s": float(np.abs(out._toi_l0 - out.toi_rate_s)[matched].max())
             if matched.any() else None,
             "max_abs_toi_rate_5v5_diff_s": float(np.abs(out._toi5_l0 - out.toi_rate_5v5_s)[matched].max())
             if matched.any() else None,
             "points_source": "baselines L0 (только допущенные stints)"}
    filled = out.fillna({c: 0.0 for c in ONICE_GOAL_COLS + ["points"]})
    return filled.drop(columns=["_toi_l0", "_toi5_l0"]), check


# --- цели ----------------------------------------------------------------------------------

def target_frames(pm: pd.DataFrame) -> dict[str, tuple[str, pd.DataFrame]]:
    """Таблицы целей в формате `baselines.MF_COLS`: имя → (подпись, таблица)."""
    out = {}
    for name in TARGET_ORDER:
        metric = IAV_METRIC if name == IAV_METRIC.name else bl.METRICS[name]
        cols = (*metric.plus, *metric.minus, metric.toi_col)
        if all(c in pm.columns for c in cols):
            out[name] = (metric.label, bl.metric_frame(pm, metric))
    if MAIN_TARGET not in out:
        raise KeyError(f"нет колонок основной цели {MAIN_TARGET} (xgf_5v5, xga_5v5, toi_rate_5v5_s)")
    return out


# --- прогноз будущего ----------------------------------------------------------------------

def reference_minus(res: dict, ref: str = REFERENCE) -> dict:
    """Из `compare_ratings(..., baseline=ref)` (разности «X − ref») — разности «ref − X» и их CI."""
    def negate(row: dict) -> dict:
        out = {}
        for s in STATS:
            lo, hi = row[f"{s}_diff_ci"]
            out[f"{s}_diff"] = -row[f"{s}_diff"]
            out[f"{s}_diff_ci"] = [-hi, -lo]
        return out

    return {name: {g: negate(row) for g, row in groups.items()} for name, groups in res["vs_baseline"].items()}


def predict_future(pm: pd.DataFrame, rating_fns: Mapping[str, RatingFn],
                   targets: Mapping[str, tuple[str, pd.DataFrame]], past: set, future: set,
                   min_toi_s: float = va.MIN_TOI_PRED_S, n_boot: int = va.N_BOOT, seed: int = va.SEED,
                   ref: str = REFERENCE) -> dict:
    """Рейтинги по матчам `past` → цели по матчам `future` (множества не пересекаются).

    Игрок участвует, если его экспозиция цели ≥ `min_toi_s` в обеих частях и все рейтинги
    конечны (одни игроки для всех рейтингов — парное сравнение).
    """
    if past & future:
        raise ValueError("predict_future: прошлые и будущие матчи пересекаются")
    if ref not in rating_fns:
        raise KeyError(f"predict_future: нет рейтинга-ориентира {ref!r}")
    pm_past = pm[pm.match_id.isin(past)]
    ratings = {n: fn(pm_past) for n, fn in rating_fns.items()}
    out = {}
    for t, (label, mf) in targets.items():
        t_past = va.target_rates(mf[mf.match_id.isin(past)])
        t_fut = va.target_rates(mf[mf.match_id.isin(future)])
        eligible = t_past.index[t_past.toi_s >= min_toi_s]
        res = va.compare_ratings(ratings, t_fut[t_fut.index.isin(eligible)], ref, min_toi_s, n_boot, seed)
        out[t] = {"label": label, **res, "ref_minus": reference_minus(res, ref)}
    return {"n_past_matches": len(past), "n_future_matches": len(future), "targets": out}


def train_predictive(pm: pd.DataFrame, rating_fns: Mapping[str, RatingFn], seed: int = va.SEED,
                     min_toi_s: float = va.MIN_TOI_PRED_S, n_boot: int = va.N_BOOT) -> dict:
    """Train: половина A → B и B → A на одних случайных половинах матчей."""
    targets = target_frames(pm)
    halves = va.assign_halves(pm.match_id, seed)
    a, b = set(halves.index[halves == 0]), set(halves.index[halves == 1])
    return {"split": "random match halves (train has no time order)", "seed": seed,
            "A->B": predict_future(pm, rating_fns, targets, a, b, min_toi_s, n_boot, seed),
            "B->A": predict_future(pm, rating_fns, targets, b, a, min_toi_s, n_boot, seed)}


def temporal_predictive(pm_test: pd.DataFrame, games: pd.DataFrame, rating_fns: Mapping[str, RatingFn],
                        seq_col: str = "game_seq", min_toi_s: float = va.MIN_TOI_TEMPORAL_S,
                        n_boot: int = va.N_BOOT, seed: int = va.SEED) -> dict:
    """Test: первая половина сезона → вторая; модели заморожены на train. Только диагностика."""
    early, late = temporal_halves(games, seq_col)
    res = predict_future(pm_test, rating_fns, target_frames(pm_test), early, late, min_toi_s, n_boot, seed)
    return {"diagnostic_only": True, "seq_col": seq_col,
            "note": "test просматривался при разработке: результат — диагностика, не для подбора",
            "n_rows_without_seq": int((~pm_test.match_id.isin(early | late)).sum()), **res}


# --- split-half надёжность рейтингов -------------------------------------------------------

def _rel_frame(pm_a: pd.DataFrame, pm_b: pd.DataFrame, rating_fns: Mapping[str, RatingFn],
               min_toi_s: float) -> pd.DataFrame:
    """Игроки с `toi_rate_s` ≥ порога в обеих половинах и конечными рейтингами всех видов."""
    toi_a = pm_a.groupby("player_id").toi_rate_s.sum()
    toi_b = pm_b.groupby("player_id").toi_rate_s.sum()
    ids = toi_a.index[toi_a >= min_toi_s].intersection(toi_b.index[toi_b >= min_toi_s])
    roles = stage_l5.player_roles(pd.concat([pm_a, pm_b]))
    cols = {"role": roles.reindex(ids).to_numpy()}
    for n, fn in rating_fns.items():
        cols[f"a::{n}"] = fn(pm_a).reindex(ids).to_numpy(dtype=float)
        cols[f"b::{n}"] = fn(pm_b).reindex(ids).to_numpy(dtype=float)
    return pd.DataFrame(cols, index=ids).dropna()


def _rel_stats(df: pd.DataFrame, names: list[str], ix: np.ndarray) -> dict:
    groups = df.role.to_numpy().astype(str)[ix]
    return {n: _group_stats(df[f"a::{n}"].to_numpy(float)[ix], df[f"b::{n}"].to_numpy(float)[ix],
                            None, groups) for n in names}


def _rel_summary(point: dict, boots: list[dict], name: str) -> dict:
    out = {}
    for g, st in point[name].items():
        vals = [b[name].get(g, {}).get("spearman", np.nan) for b in boots]
        out[g] = {"n": st["n"], "pearson": st["pearson"], "spearman": st["spearman"],
                  "spearman_ci": _ci(vals), "spearman_sb": va.spearman_brown(st["spearman"]),
                  "spearman_sb_ci": _ci([va.spearman_brown(v) for v in vals]),
                  "pearson_sb": va.spearman_brown(st["pearson"])}
    return out


def _rel_diff(point: dict, boots: list[dict], name: str, ref: str) -> dict:
    def sb(d: dict, n: str) -> float:
        return va.spearman_brown(d[n].get(va.POOLED, {}).get("spearman", np.nan))

    return {"spearman_sb_diff": sb(point, ref) - sb(point, name),
            "spearman_sb_diff_ci": _ci([sb(b, ref) - sb(b, name) for b in boots])}


def rating_reliability(pm: pd.DataFrame, rating_fns: Mapping[str, RatingFn], seed: int = va.SEED,
                       min_toi_s: float = va.MIN_TOI_HALF_S, n_boot: int = va.N_BOOT,
                       ref: str = REFERENCE) -> dict:
    """Split-half рейтингов: каждый рейтинг подогнан отдельно на A и на B, корреляция по игрокам.

    Внутри амплуа и объединённо (перцентильные ранги внутри амплуа), поправка Спирмена–Брауна.
    CI — бутстрэп игроков со стратификацией по амплуа, одни выборки для всех рейтингов; парная
    разность r_SB(ref) − r_SB(X) объединённо. Матчи не перевыбираются: повторная подгонка O1 на
    каждом бутстрэпе матчей дорога, поэтому CI не учитывает общие матчи партнёров.
    """
    pm_a, pm_b, halves = split_halves(pm, seed)
    df = _rel_frame(pm_a, pm_b, rating_fns, min_toi_s)
    names = list(rating_fns)
    point = _rel_stats(df, names, np.arange(len(df)))
    rng = np.random.default_rng(seed + 3)
    groups = df.role.to_numpy().astype(str)
    boots = [_rel_stats(df, names, _boot_indices(groups, rng)) for _ in range(n_boot)] if len(df) else []
    ratings = {}
    for n in names:
        s = _rel_summary(point, boots, n)
        ratings[n] = {"by_role": {g: v for g, v in s.items() if g != va.POOLED}, va.POOLED: s.get(va.POOLED, {})}
    return {"seed": seed, "n_matches": int(len(halves)), "n_players": int(len(df)), "min_toi_s": min_toi_s,
            "n_boot": n_boot, "bootstrap": "players, stratified by role (paired)", "reference": ref,
            "ratings": ratings,
            "ref_minus": {n: _rel_diff(point, boots, n, ref) for n in names if n != ref}}


# --- правило вывода (METHODOLOGY 7.3) ------------------------------------------------------

def acceptance(reliability: dict, predictive: dict, m0: str = M0_RATING, o0: str = "O0",
               target: str = MAIN_TARGET, direction: str = "A->B") -> dict:
    """Условия 7.3: надёжность O1 выше очков/60 (CI разности > 0) и прогноз xGD не хуже O0 (CI ∋ ≥ 0)."""
    rel = reliability["ref_minus"].get(m0, {})
    pred = predictive[direction]["targets"].get(target, {}).get("ref_minus", {}).get(o0, {}).get(va.POOLED, {})
    lo_rel = rel.get("spearman_sb_diff_ci", [np.nan, np.nan])[0]
    hi_pred = pred.get("spearman_diff_ci", [np.nan, np.nan])[1]
    cond1 = bool(np.isfinite(lo_rel) and lo_rel > 0)
    cond2 = bool(np.isfinite(hi_pred) and hi_pred >= 0)
    return {"reliability_above_m0": cond1, "prediction_not_worse_than_o0": cond2,
            "o1_more_useful": cond1 and cond2,
            "rule": "METHODOLOGY 7.3: (1) r_SB(O1) − r_SB(очки/60): нижняя граница CI > 0; "
                    f"(2) Spearman(O1) − Spearman(O0) к {target} ({direction}): верхняя граница CI ≥ 0"}
