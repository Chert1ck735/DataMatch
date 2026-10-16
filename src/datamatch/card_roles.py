"""Карточка игрока: роли и похожие игроки (METHODOLOGY 11.2). Профиль, не рейтинг.

Признаки — апостериорные компоненты L5 на 60 минут (θ̂ − m_r), стандартизованные внутри амплуа по
обучающему набору (игроки разработки с w ≥ 0,4). Роли — GMM с полной ковариацией, число кластеров —
минимум BIC; устойчивость — ARI назначений по половинам матчей. Похожие — расстояние Махаланобиса на
стандартизованных компонентах, мера сходства — перцентиль близости среди пар пула амплуа.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

BASE_FEATURES = ("comp_shot", "comp_loss", "comp_recovery", "comp_creation")
FEATURES = {"C": BASE_FEATURES + ("comp_faceoff",), "W": BASE_FEATURES, "D": BASE_FEATURES}
# CONFIG: порог |z| центра кластера для названия и число слов в названии (METHODOLOGY 11.2).
NAME_Z = 0.5
NAME_MAX_PARTS = 2
NEUTRAL_NAME = "универсал (около среднего амплуа)"
# CONFIG: направление с собственным значением ≤ NULL_EIG_REL·след ковариации стандартизованных
# компонент считается численно неопределённым и отбрасывается (апостериорные компоненты лежат в
# области значений T_r, у которой есть ≈ нулевые собственные значения; то же обоснование 1e-3, что у
# порога сигнала L5). Найдено после первого расчёта 08.10, METHODOLOGY 11.2 (поправка).
NULL_EIG_REL = 1.0e-3
COVARIANCE_TYPE = "tied"
N_EXAMPLES = 3
NAMES = {"comp_shot": ("бросающий", "мало ценности бросков"),
         "comp_loss": ("надёжный с шайбой", "много потерь"),
         "comp_recovery": ("отбирающий", "мало отборов"),
         "comp_creation": ("созидатель", "мало созидания"),
         "comp_faceoff": ("вбрасывающий", "слабые вбрасывания")}


def role_features(role: str) -> tuple[str, ...]:
    return FEATURES[role]


@dataclass(frozen=True)
class Scaler:
    features: tuple[str, ...]
    mean: np.ndarray
    sd: np.ndarray


def fit_scaler(df: pd.DataFrame, features) -> Scaler:
    """Среднее и SD (ddof = 1) по обучающему набору; нулевой разброс признака — ошибка."""
    X = df[list(features)].to_numpy(dtype=float)
    sd = X.std(axis=0, ddof=1)
    bad = [f for f, s in zip(features, sd) if not np.isfinite(s) or s <= 0]
    if bad:
        raise ValueError(f"fit_scaler: нет разброса у признаков {bad}")
    return Scaler(tuple(features), X.mean(axis=0), sd)


def apply_scaler(sc: Scaler, df: pd.DataFrame) -> np.ndarray:
    return (df[list(sc.features)].to_numpy(dtype=float) - sc.mean) / sc.sd


def alternate_halves(pm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Строки «игрок × матч» по половинам: матчи игрока по порядку match_id через один (как в L5)."""
    p = pm.sort_values(["player_id", "match_id"])
    half = p.groupby("player_id").cumcount().to_numpy() % 2
    return p[half == 0].copy(), p[half == 1].copy()


# ---------------------------------------------------------------------------------------
# Подпространство сигнала

@dataclass(frozen=True)
class Subspace:
    center: np.ndarray       # среднее z обучающего набора
    vectors: np.ndarray      # d × k, собственные векторы по убыванию собственных значений
    eig: np.ndarray          # k собственных значений > NULL_EIG_REL·след


def fit_subspace(Z: np.ndarray, rel: float = NULL_EIG_REL) -> Subspace:
    """Главные направления ковариации z (ddof = 1) без численно нулевых."""
    C = np.cov(np.asarray(Z, dtype=float), rowvar=False)
    val, vec = np.linalg.eigh(C)
    keep = val > rel * float(np.trace(C))
    order = np.argsort(-val[keep])
    return Subspace(np.asarray(Z, dtype=float).mean(axis=0), vec[:, keep][:, order], val[keep][order])


def project(sub: Subspace, Z: np.ndarray) -> np.ndarray:
    """Отбеленные координаты: евклидово расстояние в них = Махаланобис на подпространстве."""
    return (np.asarray(Z, dtype=float) - sub.center) @ sub.vectors / np.sqrt(sub.eig)


def to_z(sub: Subspace, u: np.ndarray) -> np.ndarray:
    """Обратно в z-шкалу признаков (для названий центров кластеров)."""
    return sub.center + sub.vectors @ (np.sqrt(sub.eig) * np.asarray(u, dtype=float))


# ---------------------------------------------------------------------------------------
# Кластеризация

@dataclass(frozen=True)
class GmmSelection:
    k: int
    bic: dict
    model: GaussianMixture


def select_gmm(X: np.ndarray, k_max: int, seed: int, n_init: int) -> GmmSelection:
    """GMM с общей (tied) ковариацией, k = 1…k_max, выбор по минимуму BIC.

    Общая ковариация: кластеры различаются центрами, а не разбросом, — иначе «широкий» компонент
    собирает крайних игроков с обеих сторон и имя по центру им не соответствует (METHODOLOGY 11.2,
    третья поправка; литература — docs/references/references.csv, тема roles).
    """
    models, bic = {}, {}
    for k in range(1, k_max + 1):
        m = GaussianMixture(n_components=k, covariance_type=COVARIANCE_TYPE, n_init=n_init, random_state=seed).fit(X)
        models[k], bic[k] = m, float(m.bic(X))
    best = min(bic, key=bic.get)
    return GmmSelection(best, bic, models[best])


def kmeans_silhouette(X: np.ndarray, ks, seed: int) -> dict:
    """Силуэт k-means — только диагностика числа кластеров."""
    out = {}
    for k in ks:
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        out[k] = float(silhouette_score(X, lab))
    return out


def split_half_ari(model: GaussianMixture, Xa: np.ndarray, Xb: np.ndarray) -> float:
    """ARI назначений одной и той же модели по двум половинам; при одном кластере не определён."""
    if model.n_components < 2:
        return float("nan")
    return float(adjusted_rand_score(model.predict(Xa), model.predict(Xb)))


def cluster_name(center_z: np.ndarray, features) -> str:
    """Название по профилю центра: до двух компонент с |z| ≥ 0,5, по убыванию |z|."""
    z = np.asarray(center_z, dtype=float)
    order = [i for i in np.argsort(-np.abs(z), kind="stable") if abs(z[i]) >= NAME_Z][:NAME_MAX_PARTS]
    if not order:
        return NEUTRAL_NAME
    return ", ".join(NAMES[features[i]][0 if z[i] > 0 else 1] for i in order)


def unique_names(names: list[str]) -> list[str]:
    """Повторяющиеся названия кластеров различаются суффиксом « (2)», « (3)» по порядку."""
    seen: dict[str, int] = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    return out


def cluster_summary(Z: np.ndarray, labels: np.ndarray, centers_z: np.ndarray, ids, features,
                    n_examples: int = N_EXAMPLES) -> list[dict]:
    """По кластеру: размер, разброс z по признакам (выборочное SD) и примеры — игроки, ближайшие к центру."""
    Z, labels, ids = np.asarray(Z, dtype=float), np.asarray(labels), np.asarray(ids, dtype=object)
    out = []
    for c in range(len(centers_z)):
        m = labels == c
        zc = Z[m]
        d = np.linalg.norm(zc - centers_z[c], axis=1)
        order = np.argsort(d, kind="stable")[:n_examples]
        sd = zc.std(axis=0, ddof=1) if len(zc) > 1 else np.full(Z.shape[1], np.nan)
        out.append({"n": int(m.sum()), "sd_z": {f: float(v) for f, v in zip(features, sd)},
                    "examples": ids[m][order].tolist()})
    return out


# ---------------------------------------------------------------------------------------
# Похожие игроки

def _inv(cov: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(cov, dtype=float))


def _mahal(diff: np.ndarray, icov: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, icov, diff))


def nearest(q: np.ndarray, pool: np.ndarray, cov: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Индексы и расстояния Махаланобиса k ближайших строк пула."""
    d = _mahal(pool - q[None, :], _inv(cov))
    idx = np.argsort(d, kind="stable")[:k]
    return idx, d[idx]


def pair_distances(Z: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Расстояния всех пар i < j (опорное распределение для меры сходства)."""
    i, j = np.triu_indices(len(Z), k=1)
    return _mahal(Z[i] - Z[j], _inv(cov))


def closeness_pct(d: float, ref: np.ndarray) -> float:
    """Доля пар пула (в %), которые дальше друг от друга, чем данная пара."""
    ref = np.asarray(ref, dtype=float)
    return float(100.0 * np.mean(ref > d)) if ref.size else float("nan")


def similar_table(players: pd.DataFrame, Z: np.ndarray, cov_by_role: dict, k: int, min_w: float) -> pd.DataFrame:
    """Топ-k похожих в пуле амплуа (w ≥ min_w, без самого игрока): пул 'all' и 'other_teams'.

    players: player_id, role, team_id, w (строки соответствуют строкам Z). У Z общая ширина по
    наибольшему числу признаков; для амплуа читаются первые d столбцов, d — размер его ковариации.
    """
    pid, role = players.player_id.to_numpy(), players.role.to_numpy()
    team, w = players.team_id.to_numpy(), players.w.to_numpy(dtype=float)
    dim = {r: np.asarray(cov_by_role[r]).shape[0] for r in set(role)}
    ref = {}
    for r in set(role):
        pool = (role == r) & (w >= min_w)
        ref[r] = pair_distances(Z[pool, : dim[r]], cov_by_role[r]) if pool.sum() >= 2 else np.empty(0)
    rows = []
    for i in range(len(players)):
        base = (role == role[i]) & (w >= min_w) & (np.arange(len(players)) != i)
        for name, mask in (("all", base), ("other_teams", base & (team != team[i]))):
            cand = np.flatnonzero(mask)
            if not cand.size:
                continue
            d_r = dim[role[i]]
            idx, dist = nearest(Z[i, :d_r], Z[cand, :d_r], cov_by_role[role[i]], k)
            for rank, (j, d) in enumerate(zip(cand[idx], dist), start=1):
                rows.append({"player_id": pid[i], "pool": name, "rank": rank, "similar_id": pid[j],
                             "similar_team_id": team[j], "distance": float(d),
                             "closeness_pct": closeness_pct(d, ref[role[i]]), "similar_w": w[j],
                             "query_unstable": bool(w[i] < min_w)})
    return pd.DataFrame(rows)
