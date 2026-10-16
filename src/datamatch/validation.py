"""L7 (часть): проверочный контур для любого рейтинга игроков — устойчивость и прогноз будущего.

Контур не знает, как устроен рейтинг. Ему нужны:

* таблица показателя «игрок × матч» в формате `baselines.MF_COLS`
  (player_id, match_id, role, num, toi_s): ставка = Σ num / Σ toi_s × 3600;
* для прогноза — функции рейтинга `fn(pm_half) -> pd.Series(player_id → rating)`, которые
  получают только матчи «прошлой» половины. Так подключаются и простые базовые строки, и
  O0/O1 пайплайна (функция внутри подгоняет и применяет своё стягивание на этой половине).

Разбиения:

* train не имеет порядка во времени (порядок match_id — не время), поэтому половины — случайное
  назначение матчей с фиксированным seed (`assign_halves`), одно и то же для всех игроков;
* test: `game_seq` — истинный порядок матчей «Авангарда»; `temporal_check` — первая половина
  сезона → вторая. Только отчёт (диагностика), ничего по нему не подбирается.

Статистика:

* split-half: корреляция ставок двух половин внутри амплуа (Pearson и Spearman) с поправкой
  Спирмена–Брауна r_SB = 2r / (1 + r); интервал — бутстрэп матчей (повтор матча = повтор всех
  его строк), перцентильный со сдвигом на смещение бутстрэпа (см. `split_half_reliability`),
  уровень CI_LEVEL;
* прогноз: Spearman и Pearson с весом будущего TOI между рейтингом на половине A и целью на
  половине B; интервал — бутстрэп игроков со стратификацией по амплуа; парное сравнение двух
  рейтингов — разность корреляций на одних и тех же бутстрэп-выборках;
* «объединённо» (pooled) — по всем амплуа после центрирования внутри амплуа (для Spearman —
  перцентильные ранги внутри амплуа), чтобы различия уровней амплуа не создавали корреляцию.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .baselines import MF_COLS, player_rates

SEED = 20261006
N_BOOT = 1000
CI_LEVEL = 0.95
# Минимальный TOI игрока в каждой половине для split-half, сек (60 минут).
MIN_TOI_HALF_S = 60.0 * 60.0
# Минимальный TOI игрока в половине рейтинга и в половине цели для прогноза, сек.
MIN_TOI_PRED_S = 60.0 * 60.0
# То же для хронологической проверки test (половина сезона «Авангарда»), сек.
MIN_TOI_TEMPORAL_S = 60.0 * 60.0
# Минимум игроков в группе для корреляции; меньше — NaN.
MIN_GROUP_PLAYERS = 5
# Доля валидных бутстрэп-повторов, без которой интервал не выдаётся.
MIN_VALID_BOOT_SHARE = 0.5
POOLED = "pooled"
STATS = ("pearson", "spearman", "wpearson")

RatingFn = Callable[[pd.DataFrame], pd.Series]


def _require(df: pd.DataFrame, cols, what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{what}: нет колонок {missing}")


def spearman_brown(r):
    """Надёжность полной выборки по корреляции половин: 2r / (1 + r) при r > 0, иначе 0.

    Отрицательная корреляция половин означает отсутствие обнаруженной надёжности; формула при r < 0
    уходила за шкалу [0, 1] (−2 при r = −0,5). NaN сохраняется.
    """
    r_arr = np.asarray(r, dtype=float)
    out = np.where(np.isfinite(r_arr), 2.0 * np.clip(r_arr, 0.0, None) / (1.0 + np.clip(r_arr, 0.0, None)), np.nan)
    return float(out) if out.ndim == 0 else out


def weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Pearson с весами наблюдений; NaN при нулевой дисперсии или нулевом весе."""
    w = np.asarray(w, dtype=float)
    sw = w.sum()
    if sw <= 0:
        return float("nan")
    dx = x - (w * x).sum() / sw
    dy = y - (w * y).sum() / sw
    vx, vy = (w * dx * dx).sum(), (w * dy * dy).sum()
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float((w * dx * dy).sum() / np.sqrt(vx * vy))


def _ci(samples) -> list[float]:
    a = np.asarray(samples, dtype=float)
    ok = a[np.isfinite(a)]
    if len(a) == 0 or len(ok) < MIN_VALID_BOOT_SHARE * len(a):
        return [float("nan"), float("nan")]
    q = (1.0 - CI_LEVEL) / 2.0
    return [float(np.quantile(ok, q)), float(np.quantile(ok, 1.0 - q))]


def assign_halves(match_ids, seed: int = SEED) -> pd.Series:
    """Случайное назначение матчей половинам 0/1 (фиксированный seed, не зависит от порядка ввода)."""
    ids = np.array(sorted(set(match_ids)), dtype=object)
    perm = np.random.default_rng(seed).permutation(len(ids))
    half = np.zeros(len(ids), dtype=int)
    half[perm[len(ids) // 2:]] = 1
    return pd.Series(half, index=ids, name="half")


# --- корреляции внутри групп и объединённо ---------------------------------------------------

def _centered(v: np.ndarray, groups: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    out = np.empty_like(v, dtype=float)
    for g in np.unique(groups):
        m = groups == g
        ww = np.ones(m.sum()) if w is None else w[m]
        out[m] = v[m] - (ww * v[m]).sum() / ww.sum() if ww.sum() > 0 else 0.0
    return out


def _pct_ranks(v: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.empty_like(v, dtype=float)
    for g in np.unique(groups):
        m = groups == g
        out[m] = (rankdata(v[m]) - 0.5) / m.sum()
    return out


def _corr3(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    if len(x) < MIN_GROUP_PLAYERS:
        return {"n": int(len(x)), **{s: float("nan") for s in STATS}}
    return {"n": int(len(x)), "pearson": weighted_pearson(x, y, np.ones(len(x))),
            "spearman": weighted_pearson(rankdata(x), rankdata(y), np.ones(len(x))),
            "wpearson": weighted_pearson(x, y, w)}


def _group_stats(x: np.ndarray, y: np.ndarray, w: np.ndarray | None, groups: np.ndarray | None) -> dict:
    """Корреляции по группам и объединённо (после центрирования внутри групп)."""
    w = np.ones(len(x)) if w is None else np.asarray(w, dtype=float)
    if groups is None:
        return {POOLED: _corr3(x, y, w)}
    out = {str(g): _corr3(x[groups == g], y[groups == g], w[groups == g]) for g in np.unique(groups)}
    pooled = _corr3(_centered(x, groups, None), _centered(y, groups, None), w)
    if len(x) >= MIN_GROUP_PLAYERS:
        pooled["spearman"] = weighted_pearson(_pct_ranks(x, groups), _pct_ranks(y, groups), np.ones(len(x)))
        pooled["wpearson"] = weighted_pearson(_centered(x, groups, w), _centered(y, groups, w), w)
    out[POOLED] = pooled
    return out


# --- split-half надёжность ------------------------------------------------------------------

def _half_index(mf: pd.DataFrame, halves: pd.Series):
    p_codes, players = pd.factorize(mf.player_id, sort=True)
    m_codes, matches = pd.factorize(mf.match_id, sort=True)
    h = halves.reindex(matches)
    if h.isna().any():
        raise ValueError("split-half: не всем матчам назначена половина")
    return p_codes, players, m_codes, h.to_numpy(dtype=int)


def _half_rates(num, toi, p_codes, half_codes, n_players, w_rows, min_toi):
    idx = p_codes * 2 + half_codes
    s_num = np.bincount(idx, num * w_rows, minlength=2 * n_players).reshape(n_players, 2)
    s_toi = np.bincount(idx, toi * w_rows, minlength=2 * n_players).reshape(n_players, 2)
    keep = (s_toi >= max(min_toi, np.finfo(float).tiny)).all(axis=1)
    return s_num[keep] / s_toi[keep] * 3600.0, keep


def _ci_recentred(est: float, vals: list[float], recentre: bool) -> tuple[list[float], float]:
    """Перцентильный CI; при `recentre` сдвинут на (оценка − медиана бутстрэпа)."""
    ci = _ci(vals)
    ok = np.asarray(vals, dtype=float)
    ok = ok[np.isfinite(ok)]
    if not recentre or not np.isfinite(est) or not len(ok) or not np.isfinite(ci[0]):
        return ci, 0.0
    shift = float(est - np.median(ok))
    return [ci[0] + shift, ci[1] + shift], shift


def _split_summary(point: dict, boots: list[dict], recentre: bool) -> dict:
    out = {}
    for g, st in point.items():
        row = {"n": st["n"], "pearson": st["pearson"], "spearman": st["spearman"]}
        for s in ("pearson", "spearman"):
            vals = [b.get(g, {}).get(s, np.nan) for b in boots]
            row[f"{s}_sb"] = spearman_brown(st[s])
            row[f"{s}_ci"], row[f"{s}_boot_shift"] = _ci_recentred(st[s], vals, recentre)
            row[f"{s}_sb_ci"], _ = _ci_recentred(row[f"{s}_sb"], [spearman_brown(v) for v in vals], recentre)
        out[g] = row
    return out


def _split_boots(stats_fn, n_m: int, n_boot: int, rng: np.random.Generator) -> list[dict]:
    return [stats_fn(np.bincount(rng.integers(0, n_m, n_m), minlength=n_m).astype(float)) for _ in range(n_boot)]


def split_half_reliability(mf: pd.DataFrame, metric: str = "num", toi_col: str = "toi_s",
                           by: str | None = "role", min_toi_half_s: float = MIN_TOI_HALF_S,
                           n_boot: int = N_BOOT, seed: int = SEED, halves: pd.Series | None = None,
                           boot: str = "matches") -> dict:
    """Split-half надёжность ставки Σ metric / Σ toi × 3600 по случайным половинам матчей.

    Игрок участвует, если в каждой половине у него ≥ `min_toi_half_s` секунд. Корреляции —
    внутри `by` (амплуа) и объединённо; поправка Спирмена–Брауна.

    CI: `boot="matches"` — бутстрэп матчей (учитывает общие матчи партнёров). Повторная выборка
    матчей добавляет шум в ставки половин и систематически занижает r*, поэтому перцентильный
    интервал сдвигается на `*_boot_shift` = r̂ − медиана r*. `boot="players"` — бутстрэп игроков
    со стратификацией по амплуа при фиксированных ставках половин (без сдвига).
    """
    if by not in ("role", None):
        raise ValueError(f"split_half_reliability: by должен быть 'role' или None, получено {by!r}")
    if boot not in ("matches", "players"):
        raise ValueError(f"split_half_reliability: boot должен быть 'matches' или 'players', получено {boot!r}")
    _require(mf, ["player_id", "match_id", metric, toi_col] + ([by] if by else []), "split-half")
    d = mf[mf[toi_col].to_numpy(dtype=float) > 0]
    halves = assign_halves(d.match_id, seed) if halves is None else halves
    p_codes, players, m_codes, half_m = _half_index(d, halves)
    groups = np.full(len(players), POOLED) if by is None else d.groupby(p_codes)[by].first().to_numpy().astype(str)
    num, toi = d[metric].to_numpy(dtype=float), d[toi_col].to_numpy(dtype=float)

    def rates_at(w_match: np.ndarray):
        return _half_rates(num, toi, p_codes, half_m[m_codes], len(players), w_match[m_codes], min_toi_half_s)

    def stats(w_match: np.ndarray) -> dict:
        rates, keep = rates_at(w_match)
        return _group_stats(rates[:, 0], rates[:, 1], None, None if by is None else groups[keep])

    n_m = len(half_m)
    point = stats(np.ones(n_m))
    rng = np.random.default_rng(seed + 1)
    if boot == "matches":
        boots = _split_boots(stats, n_m, n_boot, rng)
    else:
        rates, keep = rates_at(np.ones(n_m))
        gk = groups[keep]
        boots = []
        for _ in range(n_boot):
            ix = _boot_indices(gk, rng)
            boots.append(_group_stats(rates[ix, 0], rates[ix, 1], None, None if by is None else gk[ix]))
    summary = _split_summary(point, boots, recentre=boot == "matches")
    return {"metric": metric, "by": by, "seed": seed, "n_boot": n_boot, "min_toi_half_s": min_toi_half_s,
            "bootstrap": boot, "n_matches": n_m,
            "groups": {g: v for g, v in summary.items() if g != POOLED}, POOLED: summary[POOLED]}


# --- прогноз будущего -----------------------------------------------------------------------

def target_rates(mf: pd.DataFrame) -> pd.DataFrame:
    """Цель по игроку: `target` = ставка на 60, `toi_s` — вес (будущая экспозиция), `role`."""
    r = player_rates(mf).set_index("player_id")
    return r.rename(columns={"rate_60": "target"})[["role", "target", "toi_s", "games", "num"]]


def _paired_frame(ratings: Mapping[str, pd.Series], target: pd.DataFrame, min_toi_s: float) -> pd.DataFrame:
    _require(target, ["role", "target", "toi_s"], "target")
    for name, s in ratings.items():
        if not isinstance(s, pd.Series):
            raise TypeError(f"рейтинг {name!r} должен быть pd.Series(player_id → rating)")
    t = target[(target.toi_s >= min_toi_s) & target.target.notna()]
    cols = {f"r::{n}": s.reindex(t.index).to_numpy(dtype=float) for n, s in ratings.items()}
    df = t[["role", "target", "toi_s"]].assign(**cols)
    return df.dropna(subset=list(cols))


def _boot_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = [rng.choice(np.flatnonzero(groups == g), (groups == g).sum(), replace=True) for g in np.unique(groups)]
    return np.concatenate(parts)


def _predictive_core(df: pd.DataFrame, names: list[str], n_boot: int, seed: int) -> tuple[dict, list[dict]]:
    """Точечные корреляции и бутстрэп игроков (стратификация по амплуа) для нескольких рейтингов."""
    groups = df.role.to_numpy().astype(str)
    y, w = df.target.to_numpy(float), df.toi_s.to_numpy(float)
    xs = {n: df[f"r::{n}"].to_numpy(float) for n in names}

    def stats(ix: np.ndarray) -> dict:
        return {n: _group_stats(x[ix], y[ix], w[ix], groups[ix]) for n, x in xs.items()}

    point = stats(np.arange(len(df)))
    rng = np.random.default_rng(seed + 2)
    boots = [stats(_boot_indices(groups, rng)) for _ in range(n_boot)] if len(df) else []
    return point, boots


def _pred_summary(point: dict, boots: list[dict], name: str) -> dict:
    out = {}
    for g, st in point[name].items():
        out[g] = {"n": st["n"], "spearman": st["spearman"], "wpearson": st["wpearson"],
                  "spearman_ci": _ci([b[name].get(g, {}).get("spearman", np.nan) for b in boots]),
                  "wpearson_ci": _ci([b[name].get(g, {}).get("wpearson", np.nan) for b in boots])}
    return out


def _split_pooled(summary: dict) -> dict:
    return {"by_role": {g: v for g, v in summary.items() if g != POOLED}, POOLED: summary.get(POOLED, {})}


def predictive_validity(rating: pd.Series, target: pd.DataFrame, min_toi_s: float = MIN_TOI_PRED_S,
                        n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Spearman и TOI-взвешенный Pearson рейтинга (половина A) с целью (половина B)."""
    df = _paired_frame({"rating": rating}, target, min_toi_s)
    point, boots = _predictive_core(df, ["rating"], n_boot, seed)
    return {"n_players": int(len(df)), "min_toi_s": min_toi_s, "n_boot": n_boot, "seed": seed,
            "bootstrap": "players, stratified by role", **_split_pooled(_pred_summary(point, boots, "rating"))}


def _diff_summary(point: dict, boots: list[dict], name: str, base: str) -> dict:
    out = {}
    for g in point[name]:
        row = {}
        for s in ("spearman", "wpearson"):
            row[f"{s}_diff"] = point[name][g][s] - point[base][g][s]
            row[f"{s}_diff_ci"] = _ci([b[name].get(g, {}).get(s, np.nan) - b[base].get(g, {}).get(s, np.nan)
                                       for b in boots])
        out[g] = row
    return out


def compare_ratings(ratings: Mapping[str, pd.Series], target: pd.DataFrame, baseline: str,
                    min_toi_s: float = MIN_TOI_PRED_S, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Предсказательная сила нескольких рейтингов на одних игроках и парная разность с `baseline`."""
    if baseline not in ratings:
        raise KeyError(f"compare_ratings: baseline {baseline!r} нет среди рейтингов {list(ratings)}")
    names = list(ratings)
    df = _paired_frame(ratings, target, min_toi_s)
    point, boots = _predictive_core(df, names, n_boot, seed)
    return {"n_players": int(len(df)), "baseline": baseline, "min_toi_s": min_toi_s, "n_boot": n_boot,
            "seed": seed, "bootstrap": "players, stratified by role (paired)",
            "ratings": {n: _split_pooled(_pred_summary(point, boots, n)) for n in names},
            "vs_baseline": {n: _diff_summary(point, boots, n, baseline) for n in names if n != baseline}}


def _rate_and_compare(pm_a: pd.DataFrame, t_a: pd.DataFrame, t_b: pd.DataFrame,
                      rating_fns: Mapping[str, RatingFn], baseline: str, min_toi_s: float,
                      n_boot: int, seed: int) -> dict:
    """Рейтинги по прошлой части, цель — по будущей; игроки с TOI ≥ min в обеих частях."""
    eligible = t_a.index[t_a.toi_s >= min_toi_s]
    ratings = {n: fn(pm_a) for n, fn in rating_fns.items()}
    return compare_ratings(ratings, t_b[t_b.index.isin(eligible)], baseline, min_toi_s, n_boot, seed)


def _check_fns(rating_fns: Mapping[str, RatingFn], baseline: str | None) -> str:
    if not rating_fns:
        raise ValueError("нужна хотя бы одна функция рейтинга")
    return next(iter(rating_fns)) if baseline is None else baseline


def half_split_predictive(pm: pd.DataFrame, rating_fns: Mapping[str, RatingFn], target_mf: pd.DataFrame,
                          baseline: str | None = None, min_toi_s: float = MIN_TOI_PRED_S,
                          n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Прогноз между случайными половинами матчей train в обе стороны (A→B и B→A)."""
    _require(pm, ["match_id"], "pm")
    _require(target_mf, MF_COLS, "target_mf")
    baseline = _check_fns(rating_fns, baseline)
    halves = assign_halves(set(pm.match_id) | set(target_mf.match_id), seed)
    h_pm, h_t = pm.match_id.map(halves).to_numpy(), target_mf.match_id.map(halves).to_numpy()
    res = {}
    for a, b, label in ((0, 1, "A->B"), (1, 0, "B->A")):
        res[label] = _rate_and_compare(pm[h_pm == a], target_rates(target_mf[h_t == a]),
                                       target_rates(target_mf[h_t == b]), rating_fns, baseline,
                                       min_toi_s, n_boot, seed)
    mean = {n: {"pooled_spearman": float(np.mean([res[d]["ratings"][n][POOLED]["spearman"] for d in res])),
                "pooled_wpearson": float(np.mean([res[d]["ratings"][n][POOLED]["wpearson"] for d in res]))}
            for n in rating_fns}
    return {"split": "random match halves (train has no time order)", "seed": seed,
            "directions": res, "mean": mean}


def temporal_check(pm: pd.DataFrame, target_mf: pd.DataFrame, games: pd.DataFrame,
                   rating_fns: Mapping[str, RatingFn], baseline: str | None = None,
                   min_toi_s: float = MIN_TOI_TEMPORAL_S, seq_col: str = "game_seq",
                   n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Хронология test: рейтинг по первой половине сезона (game_seq) → цель по второй. Только отчёт."""
    _require(games, ["match_id", seq_col], "games")
    _require(target_mf, MF_COLS, "target_mf")
    baseline = _check_fns(rating_fns, baseline)
    g = games.dropna(subset=[seq_col]).sort_values(seq_col)
    n_early = len(g) // 2
    early, late = set(g.match_id.iloc[:n_early]), set(g.match_id.iloc[n_early:])
    res = _rate_and_compare(pm[pm.match_id.isin(early)], target_rates(target_mf[target_mf.match_id.isin(early)]),
                            target_rates(target_mf[target_mf.match_id.isin(late)]), rating_fns, baseline,
                            min_toi_s, n_boot, seed)
    return {"diagnostic_only": True,
            "note": "test просматривался при разработке: результат — диагностика, не для подбора",
            "seq_col": seq_col, "n_early_matches": len(early), "n_late_matches": len(late),
            "n_rows_without_seq": int((~pm.match_id.isin(early | late)).sum()), "result": res}
