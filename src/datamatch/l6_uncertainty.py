"""L6: неопределённость и надёжность — две разные формулы интервала.

Слой не вводит новой оценки рейтинга: он берёт готовые Overall и апостериорную дисперсию из L5
и отвечает только за ширину интервалов и метку надёжности.

**1. Интервал скрытого эффекта** (идёт в `test_ratings.csv`). Полная дисперсия:

    v_i^tot = E_m[v_i^(m)] + Var_m[θ̂_i^(m)],      θ̂_i ± 1,96·√(v_i^tot)

где m — переобучение моделей ценности (xG, V, Q, π̂) на бутстрэпе матчей train, а данные,
по которым считается рейтинг игрока, фиксированы. Для Overall апостериорная часть — это
1ᵀV_i1 из совместной модели L5; в каждой реплике Overall считается целиком, поэтому ковариации
компонентов учтены и в Var_m. Шум собственных матчей игрока уже сидит в v_i, поэтому бутстрэп
матчей теста в формулу не добавляется (это был бы двойной счёт).

Если бутстрэп моделей ценности не выполнен, Var_m неизвестна, а не равна нулю: тогда
v_tot = 1ᵀV_i1, интервал снизу занижен, и это помечается статусом ``not_run`` с причиной.
Если бутстрэп выполнен, неизвестная или нефинитная Var_m у игрока — ошибка, а не ноль
(аудит L6 06.10.2026). Интервал O0 — явное исключение из L6: только апостериорная дисперсия.

**Ошибки реплик.** Ожидаемые ошибки данных (``EXPECTED_REPLICATE_ERRORS``) записываются и
замещаются дополнительными выборками; их доля выше ``MAX_REPLICATE_FAILURE_SHARE`` даёт статус
``failed``. Остальные исключения (KeyError, TypeError, …) — дефект программы и пробрасываются.

**Гибель процесса пула** (OOM-killer, ревью F3) даёт ``BrokenProcessPool``: пул пересоздаётся
уже (ширина вдвое меньше), неготовые реплики отправляются снова (готовые лежат в кэше реплик и
не пересчитываются); после ``MAX_POOL_RESTARTS`` перезапусков — статус ``failed`` с причиной.
Время каждой реплики пишется в журнал, сводка (готово/всего, среднее, ETA) — в отчёт L6.

**Число повторов.** Интервал нормальный, бутстрэп оценивает только слагаемое дисперсии, а не
хвосты: B = 30–50 — отладка (относительная ошибка оценки дисперсии ≈ 20–25 %), B ≥ 200 —
финальный расчёт. Перцентильные интервалы при B < 1000 не используются, поэтому здесь их нет
вовсе. Бутстрэп матчей одного игрока (2–3 матча) вырожден и не используется.

**2. Предсказательный интервал будущего результата** — отдельная модель:

    y_i^f | данные ≈ N(θ̂_i,  v_i^tot + σ_c²/T_f + ω_c²/M_f + δ²)

δ² — дрейф навыка, состава и роли между периодами. Внутри train дат нет, поэтому δ² из train
не оценивается: задаётся сеткой как анализ чувствительности. В файл этот интервал не идёт.

**Надёжность** по доле собственного сигнала w = 1 − 1ᵀV_i1 / 1ᵀT_r1, обрезанной в [0, 1]:
HIGH при w ≥ 0,7, MEDIUM при 0,4 ≤ w < 0,7, LOW иначе. При 1ᵀT_r1 ≤ ε или τ² ≈ 0 слой L5
отдаёт w = 0 — LOW с пометкой «сигнал не обнаружен».

**Правило:** test не используется ни для подбора σ², δ² и conformal-поправки, ни для выбора
между вариантами интервалов. Все функции здесь — чистые преобразования уже оценённых величин.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import pandas as pd

# Нормальный квантиль 0,975. Архитектура фиксирует именно 1,96, а не 1,959964: граница файла
# должна совпадать с формулой из документа.
Z95 = 1.96

# CONFIG: минимум реплик, при котором выборочная дисперсия Var_m вообще определена (ddof=1).
MIN_BOOT_REPLICATES = 2
# CONFIG: B, ниже которого перцентильные интервалы запрещены архитектурой. Код их не строит
# никогда, константа нужна только для отчёта о соблюдении правила.
PERCENTILE_MIN_B = 1000
# CONFIG: сетка δ² (дрейф между периодами) по умолчанию для предсказательного интервала,
# если в конфигурации нет eval.drift_grid. Единицы — (xG/60)².
DRIFT_GRID_DEFAULT = (0.0, 0.05, 0.1, 0.2)
# CONFIG: допустимая доля реплик, упавших на ожидаемой ошибке данных. Упавшие реплики
# замещаются дополнительными выборками (их не больше floor(доля·B)); превышение — статус failed.
MAX_REPLICATE_FAILURE_SHARE = 0.05
# Ожидаемые ошибки данных реплики: ValueError — неподходящая выборка (мало матчей в фолде,
# нет бросков, вырожденная калибровка; np.linalg.LinAlgError — его подкласс), ArithmeticError —
# FloatingPointError/ZeroDivisionError при вырожденной выборке. KeyError, TypeError,
# AttributeError, ImportError и прочее — дефект программы: они не ловятся.
EXPECTED_REPLICATE_ERRORS: tuple[type[BaseException], ...] = (ValueError, ArithmeticError)
# CONFIG: сколько раз пересоздавать пул после гибели процесса (BrokenProcessPool), прежде чем
# признать бутстрэп проваленным. Каждый перезапуск вдвое сужает пул (меньше памяти).
MAX_POOL_RESTARTS = 2
# Статусы бутстрэпа. not_applicable — определение итога O0 (интервал по архитектуре без Var_m).
STATUS_DONE, STATUS_PARTIAL, STATUS_FAILED = "done", "partial", "failed"
STATUS_NOT_RUN, STATUS_NOT_APPLICABLE = "not_run", "not_applicable"


# --------------------------------------------------------------------------------------------
# 1. Интервал скрытого эффекта
# --------------------------------------------------------------------------------------------
def total_variance(var_post, var_model=None, var_post_boot=None) -> np.ndarray:
    """v_tot = E_m[v^(m)] + Var_m[θ̂^(m)].

    var_post — апостериорная дисперсия точечного расчёта (1ᵀV_i1 из L5);
    var_post_boot — E_m[v^(m)] по репликам, если бутстрэп её вернул (иначе берётся var_post);
    var_model — Var_m[θ̂^(m)]; None означает «бутстрэп не выполнен» (вызывающий код помечает
    занижение ширины). Переданная Var_m обязана быть конечной и неотрицательной у каждого
    игрока: NaN/inf/отрицательное значение — ошибка, а не молчаливый ноль.
    """
    vp = np.asarray(var_post, dtype=float)
    base = vp
    if var_post_boot is not None:
        vb = np.asarray(var_post_boot, dtype=float)
        base = np.where(np.isfinite(vb) & (vb >= 0), vb, vp)
    if var_model is None:
        return np.maximum(base, 0.0)
    vm = np.asarray(var_model, dtype=float)
    bad = ~np.isfinite(vm) | (vm < 0)
    if bad.any():
        raise ValueError(f"Var_m неизвестна или некорректна у {int(bad.sum())} из {len(vm)} строк: "
                         "выполненный бутстрэп обязан дать конечную неотрицательную Var_m каждому игроку")
    return np.maximum(base, 0.0) + vm


def normal_interval(rating, v_tot, z: float = Z95) -> tuple[np.ndarray, np.ndarray]:
    """rating ± z·√v_tot (нормальный интервал; v_tot < 0 считается нулём)."""
    r = np.asarray(rating, dtype=float)
    half = z * np.sqrt(np.maximum(np.asarray(v_tot, dtype=float), 0.0))
    return r - half, r + half


def reliability_labels(w, high: float = 0.7, medium: float = 0.4) -> np.ndarray:
    """HIGH при w ≥ high, MEDIUM при medium ≤ w < high, LOW иначе. NaN -> LOW."""
    if high < medium:
        raise ValueError(f"порог HIGH ({high}) ниже порога MEDIUM ({medium})")
    ww = np.asarray(w, dtype=float)
    ww = np.where(np.isfinite(ww), ww, 0.0)
    return np.where(ww >= high, "HIGH", np.where(ww >= medium, "MEDIUM", "LOW")).astype(object)


def mc_bound_error(var_model, v_tot, b_done: int, z: float = Z95) -> np.ndarray:
    """Монте-Карло-ошибка границы интервала из-за конечного B.

    sd(Var_m) ≈ Var_m·√(2/(B−1)) для нормальных реплик; ошибка границы — та же величина,
    пересчитанная через производную z·√v: z·sd(Var_m)/(2√v_tot).
    """
    vm = np.asarray(var_model, dtype=float)
    v = np.maximum(np.asarray(v_tot, dtype=float), 0.0)
    if b_done is None or b_done < MIN_BOOT_REPLICATES:
        return np.full(vm.shape, np.nan)
    sd = np.abs(vm) * np.sqrt(2.0 / (b_done - 1))          # NaN Var_m -> NaN ошибка, не 0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, z * sd / (2.0 * np.sqrt(v)), np.nan)


# --------------------------------------------------------------------------------------------
# 2. Предсказательный интервал будущего результата (в файл не идёт)
# --------------------------------------------------------------------------------------------
def predictive_variance(v_tot, sigma2: float, omega2: float, toi_future_min: float,
                        games_future: float, delta2: float = 0.0) -> np.ndarray:
    """v_tot + σ²/T_f + ω²/M_f + δ²; T_f — минуты будущей экспозиции, M_f — матчи."""
    if not (toi_future_min > 0) or not (games_future > 0):
        raise ValueError(f"будущая экспозиция должна быть положительной: T_f={toi_future_min}, M_f={games_future}")
    if sigma2 < 0 or omega2 < 0 or delta2 < 0:
        raise ValueError(f"дисперсии не могут быть отрицательными: σ²={sigma2}, ω²={omega2}, δ²={delta2}")
    v = np.maximum(np.asarray(v_tot, dtype=float), 0.0)
    return v + sigma2 / float(toi_future_min) + omega2 / float(games_future) + float(delta2)


def predictive_interval(rating, v_tot, sigma2: float, omega2: float, toi_future_min: float,
                        games_future: float, delta2: float = 0.0, z: float = Z95) -> dict:
    """Предсказательный интервал будущего наблюдаемого результата при заданном δ².

    Возвращает dict с lo/hi, дисперсией и параметрами — он идёт в отчёт и в карточку игрока,
    но не в `test_ratings.csv`.
    """
    vf = predictive_variance(v_tot, sigma2, omega2, toi_future_min, games_future, delta2)
    lo, hi = normal_interval(rating, vf, z)
    return {"lo": lo, "hi": hi, "var": vf, "sigma2": float(sigma2), "omega2": float(omega2),
            "toi_future_min": float(toi_future_min), "games_future": float(games_future),
            "delta2": float(delta2), "z": float(z)}


def predictive_interval_grid(rating, v_tot, sigma2: float, omega2: float, toi_future_min: float,
                             games_future: float, drift_grid=DRIFT_GRID_DEFAULT, z: float = Z95) -> list[dict]:
    """Сетка δ² как анализ чувствительности: δ² из train не оценивается (в train нет дат)."""
    out = []
    for d2 in drift_grid:
        r = predictive_interval(rating, v_tot, sigma2, omega2, toi_future_min, games_future, d2, z)
        r["mean_halfwidth"] = float(np.mean(z * np.sqrt(np.maximum(r["var"], 0.0))))
        out.append(r)
    return out


def predictive_grid_by_role(rating, v_tot, roles, moments: dict, toi_future_min: float, games_future: float,
                            drift_grid=DRIFT_GRID_DEFAULT, z: float = Z95) -> list[dict]:
    """Сетка δ² с σ², ω² амплуа каждого игрока (шум по амплуа); средняя полуширина по всем."""
    rr = np.asarray(roles, dtype=object)
    s2 = np.array([moments[r]["sigma2"] for r in rr], dtype=float)
    o2 = np.array([moments[r]["omega2"] for r in rr], dtype=float)
    out = []
    for d2 in drift_grid:
        vf = np.maximum(np.asarray(v_tot, dtype=float), 0.0) + s2 / toi_future_min + o2 / games_future + d2
        out.append({"delta2": float(d2), "var": vf,
                    "mean_halfwidth": float(np.mean(z * np.sqrt(vf))),
                    "mean_halfwidth_by_role": {r: float(np.mean(z * np.sqrt(vf[rr == r]))) for r in np.unique(rr)}})
    return out


# --------------------------------------------------------------------------------------------
# Моменты шума из отчёта L5 (только чтение опубликованного; test ничего не подбирает)
# --------------------------------------------------------------------------------------------
def _scalar(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) and f >= 0 else None


def overall_noise_moments(report: dict) -> dict | None:
    """σ² и ω² шкалы Overall (суммы компонентов) из отчёта L5, если он их публикует.

    Порядок: одномерная модель суммы (`moments.univariate[role]`), затем явные
    `overall_moments`/`sigma2_overall`, затем 1ᵀΣσ1 и 1ᵀΣω1. Оценивать σ² по тестовой таблице
    запрещено правилом L6, поэтому при отсутствии публикации возвращается None.
    """
    if not isinstance(report, dict):
        return None
    moments = report.get("moments") if isinstance(report.get("moments"), dict) else {}
    uni = moments.get("univariate") if isinstance(moments.get("univariate"), dict) else {}
    pairs = {}
    for role in sorted(uni):
        mm = uni[role]
        if isinstance(mm, dict):
            s, o = _scalar(mm.get("sigma2")), _scalar(mm.get("omega2"))
            if s is not None and o is not None:
                pairs[role] = (s, o)
    if pairs:
        # B5: общая пара σ², ω² существует, только если она одинакова у всех амплуа; иначе — шум по
        # амплуа (predictive_grid_by_role), а не моменты первого амплуа («C»)
        if len(set(pairs.values())) > 1:
            return None
        s, o = next(iter(pairs.values()))
        return {"sigma2": s, "omega2": o, "role": "все",
                "source": "report.moments.univariate (одномерная модель суммы, одинакова по амплуа)"}
    for key in ("overall_moments", "moments_overall", "univariate_overall"):
        m = report.get(key)
        if isinstance(m, dict):
            s, o = _scalar(m.get("sigma2")), _scalar(m.get("omega2"))
            if s is not None and o is not None:
                return {"sigma2": s, "omega2": o, "source": f"report.{key}"}
    s, o = _scalar(report.get("sigma2_overall")), _scalar(report.get("omega2_overall"))
    if s is not None and o is not None:
        return {"sigma2": s, "omega2": o, "source": "report.sigma2_overall/omega2_overall"}
    try:
        s = float(np.asarray(moments["Sigma_sigma"], dtype=float).sum())
        o = float(np.asarray(moments["Sigma_omega"], dtype=float).sum())
    except (KeyError, TypeError, ValueError):
        return None
    if np.isfinite(s) and np.isfinite(o) and s >= 0 and o >= 0:
        return {"sigma2": s, "omega2": o, "source": "report.moments: 1ᵀΣσ1 и 1ᵀΣω1"}
    return None


def role_moments(report: dict) -> dict | None:
    """τ², σ², ω² одномерной модели суммы по амплуа из отчёта L5 (moments.univariate[role])."""
    uni = ((report or {}).get("moments") or {}).get("univariate")
    if not isinstance(uni, dict):
        return None
    out = {}
    for role, mm in uni.items():
        vals = {k: _scalar(mm.get(k)) if isinstance(mm, dict) else None for k in ("tau2", "sigma2", "omega2")}
        if all(v is not None for v in vals.values()):
            out[role] = vals
    return out or None


# --------------------------------------------------------------------------------------------
# Покрытие
# --------------------------------------------------------------------------------------------
def empirical_coverage(theta_true, lo, hi) -> dict:
    """Доля попаданий истинного значения в интервал (для симуляции и для L7)."""
    t = np.asarray(theta_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    ok = np.isfinite(t) & np.isfinite(lo) & np.isfinite(hi)
    n = int(ok.sum())
    hit = int(((t >= lo) & (t <= hi))[ok].sum())
    return {"n": n, "n_hit": hit, "coverage": (hit / n) if n else float("nan")}


def simulate_latent_coverage(tau2: float, sigma2: float, omega2: float, toi_min, games,
                             n_sim: int = 200, seed: int = 0, z: float = Z95) -> dict:
    """Покрытие интервала скрытого эффекта на симуляции с известной дисперсией.

    Скрытое θ не наблюдается, поэтому покрытие проверяется только так: игроки с известным
    θ ~ N(0, τ²) из подогнанной модели, та же структура матчей и TOI, наблюдение
    x = θ + e, Var(e) = σ²/T + ω²/M. Апостериорно θ̂ = w·x, v = τ²s/(τ² + s).
    Возвращает долю попаданий против номинала.
    """
    if tau2 < 0 or sigma2 < 0 or omega2 < 0:
        raise ValueError("τ², σ², ω² не могут быть отрицательными")
    T = np.asarray(toi_min, dtype=float)
    M = np.asarray(games, dtype=float)
    ok = (T > 0) & (M > 0)
    T, M = T[ok], M[ok]
    if len(T) == 0:
        raise ValueError("нет игроков с положительной экспозицией для симуляции")
    rng = np.random.default_rng(int(seed))
    s = sigma2 / T + omega2 / M
    w = tau2 / (tau2 + s) if tau2 > 0 else np.zeros_like(s)
    v = tau2 * s / (tau2 + s) if tau2 > 0 else np.zeros_like(s)
    hit = 0
    tot = 0
    for _ in range(int(n_sim)):
        theta = rng.normal(0.0, np.sqrt(tau2), size=len(T)) if tau2 > 0 else np.zeros(len(T))
        x = theta + rng.normal(0.0, np.sqrt(s))
        th = w * x
        lo, hi = normal_interval(th, v, z)
        c = empirical_coverage(theta, lo, hi)
        hit += c["n_hit"]
        tot += c["n"]
    return {"n": tot, "n_hit": hit, "coverage": hit / tot if tot else float("nan"),
            "nominal": float(2 * _norm_cdf(z) - 1), "mean_w": float(np.mean(w)), "n_players": int(len(T)),
            "n_sim": int(n_sim), "tau2": float(tau2), "sigma2": float(sigma2), "omega2": float(omega2)}


def simulate_posterior_coverage(w, var_post, n_sim: int = 200, seed: int = 0, z: float = Z95) -> dict:
    """Покрытие апостериорного интервала по опубликованным w и v = 1ᵀV_i1.

    Из w = τ²/(τ² + s) и v = τ²s/(τ² + s) однозначно следует τ² = v/(1 − w) и s = v/w, поэтому
    проверка калибровки не требует внутренних величин L5. Симулируется θ_i ~ N(0, τ²_i),
    x_i = θ_i + N(0, s_i), θ̂_i = w_i·x_i, интервал θ̂_i ± z√v_i.
    """
    ww = np.asarray(w, dtype=float)
    vv = np.asarray(var_post, dtype=float)
    ok = np.isfinite(ww) & np.isfinite(vv) & (ww > 0) & (ww < 1) & (vv > 0)
    ww, vv = ww[ok], vv[ok]
    if len(ww) == 0:
        raise ValueError("нет игроков с 0 < w < 1 и положительной апостериорной дисперсией")
    tau2 = vv / (1.0 - ww)
    s = vv / ww
    rng = np.random.default_rng(int(seed))
    hit = tot = 0
    for _ in range(int(n_sim)):
        theta = rng.normal(0.0, np.sqrt(tau2))
        x = theta + rng.normal(0.0, np.sqrt(s))
        lo, hi = normal_interval(ww * x, vv, z)
        c = empirical_coverage(theta, lo, hi)
        hit += c["n_hit"]
        tot += c["n"]
    return {"n": tot, "n_hit": hit, "coverage": hit / tot if tot else float("nan"),
            "nominal": float(2 * _norm_cdf(z) - 1), "n_players": int(len(ww)), "n_sim": int(n_sim),
            "mean_w": float(ww.mean())}


def _simulated_player_match(theta: np.ndarray, toi_min: np.ndarray, games: np.ndarray, roles: np.ndarray,
                            moments: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Таблица «игрок × матч» той же структуры: M_i матчей по T_i/M_i минут.

    Ставка в матче x_im = θ_i + e_im, Var(e_im) = σ²_r/t_im + ω²_r; при равных t_im шум
    средней ставки игрока равен ровно σ²_r/T_i + ω²_r/M_i — модели L5.
    """
    m = games.astype(int)
    pid = np.repeat(np.arange(len(theta)), m)
    t = np.repeat(toi_min / m, m)
    sig = np.array([moments[r]["sigma2"] for r in roles], dtype=float)[pid]
    omg = np.array([moments[r]["omega2"] for r in roles], dtype=float)[pid]
    x = theta[pid] + rng.normal(0.0, 1.0, len(pid)) * np.sqrt(sig / t + omg)
    match = np.concatenate([np.arange(k) for k in m]) if len(m) else np.zeros(0, dtype=int)
    return pd.DataFrame({"player_id": [f"S{i:05d}" for i in pid], "match_id": [f"M{k:04d}" for k in match],
                         "y": x * t / 60.0, "toi_s": t * 60.0})


def simulate_refit_coverage(toi_min, games, roles, moments: dict, n_sim: int = 50, seed: int = 0,
                            min_toi_s: float = 300.0, eps_abs: float = 1e-10, z: float = Z95,
                            noise_by_role: bool = False) -> dict:
    """Покрытие интервала скрытого эффекта с ЗАНОВО оценёнными моментами (раздел L6, таблица).

    Игроки с известным θ_i ~ N(0, τ²_r) по моментам подогнанной модели, та же структура
    матчей и TOI (T_i, M_i; минуты матча поровну), наблюдения по матчам; затем моменты
    τ², σ², ω² оцениваются заново процедурой L5 (`l5_shrink.UnivariateEB`: половины матчей и
    регрессия квадратов разностей), строится θ̂ ± z√v и считается доля попаданий θ.
    В отличие от проверки по опубликованным w/var_post, ошибка оценки моментов здесь видна.
    Шкала — одномерная модель суммы (Overall); Var_m аналога в симуляции не имеет.
    """
    from .l5_shrink import UnivariateEB
    T = np.asarray(toi_min, dtype=float)
    M = np.asarray(games, dtype=float)
    rr = np.asarray(roles, dtype=object)
    ok = (T > 0) & (M >= 1) & np.isin(rr, list(moments))
    T, M, rr = T[ok], np.round(M[ok]), rr[ok]
    if len(T) == 0:
        raise ValueError("нет игроков с положительной экспозицией и известными моментами амплуа")
    for r, mm in moments.items():
        if min(float(mm["tau2"]), float(mm["sigma2"]), float(mm["omega2"])) < 0:
            raise ValueError(f"моменты амплуа {r} не могут быть отрицательными: {mm}")
    rng = np.random.default_rng(int(seed))
    tau = np.sqrt(np.array([moments[r]["tau2"] for r in rr], dtype=float))
    ids = pd.Index([f"S{i:05d}" for i in range(len(T))])
    roles_s = pd.Series(rr, index=ids)
    hit = tot = 0
    for _ in range(int(n_sim)):
        theta = rng.normal(0.0, 1.0, len(T)) * tau
        pm = _simulated_player_match(theta, T, M, rr, moments, rng)
        eb = UnivariateEB(eps_abs, noise_by_role=noise_by_role).fit(pm, "y", roles_s, min_toi_s=min_toi_s)
        sums = pm.groupby("player_id").agg(y=("y", "sum"), toi_s=("toi_s", "sum"),
                                           games=("match_id", "nunique")).reindex(ids)
        th, v, _ = eb.posterior(rr, sums.y.to_numpy(), sums.toi_s.to_numpy(), sums.games.to_numpy())
        lo, hi = normal_interval(th, v, z)
        c = empirical_coverage(theta, lo, hi)
        hit += c["n_hit"]
        tot += c["n"]
    return {"kind": "refit_simulation", "moments_reestimated": True, "estimator": "l5_shrink.UnivariateEB",
            "n": tot, "n_hit": hit, "coverage": hit / tot if tot else float("nan"),
            "nominal": float(2 * _norm_cdf(z) - 1), "n_players": int(len(T)), "n_sim": int(n_sim),
            "toi_structure": "T_i и M_i игроков теста, минуты матча поровну"}


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# --------------------------------------------------------------------------------------------
# Бутстрэп моделей ценности
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BootstrapDraw:
    """Одна реплика m: своя выборка матчей train с возвращением и свой seed."""
    index: int
    seed: int
    matches: np.ndarray


@dataclass(frozen=True)
class BootstrapResult:
    """Итог бутстрэпа моделей ценности.

    status: ``done`` — выполнено ровно запрошенное B; ``partial`` — исчерпан бюджет времени;
    ``failed`` — доля упавших реплик выше порога или нефинитное Overall в реплике;
    ``not_run`` — Var_m не оценена (причина в ``reason``); ``not_applicable`` — итог O0.
    """
    status: str
    reason: str
    b_requested: int
    b_done: int
    var_model: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    var_post_mean: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    n_replicates: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))
    n_failed: int = 0
    n_attempted: int = 0
    failures: tuple = ()
    elapsed_s: float = 0.0
    replicate_seconds: tuple = ()
    pool_restarts: int = 0

    @property
    def available(self) -> bool:
        return self.status in (STATUS_DONE, STATUS_PARTIAL)

    def to_meta(self) -> dict:
        return {"status": self.status, "reason": self.reason, "B_requested": self.b_requested,
                "B_done": self.b_done, "n_players_with_var": int(self.var_model.notna().sum()),
                "n_failed": self.n_failed, "n_attempted": self.n_attempted,
                "failure_share": (self.n_failed / self.n_attempted) if self.n_attempted else 0.0,
                "max_failure_share": MAX_REPLICATE_FAILURE_SHARE,
                "failures": list(self.failures)[:5], "elapsed_s": round(self.elapsed_s, 3),
                "pool_restarts": self.pool_restarts, "max_pool_restarts": MAX_POOL_RESTARTS,
                "progress": progress_summary(min(self.b_done, self.b_requested), self.b_requested,
                                             self.replicate_seconds, self.elapsed_s),
                "percentile_intervals_used": False, "percentile_min_B": PERCENTILE_MIN_B}


def progress_summary(done: int, total: int, seconds, elapsed_s: float) -> dict:
    """Готово/всего, среднее время реплики (включая взятые из кэша) и ETA по фактическому темпу."""
    sec = np.asarray(list(seconds), dtype=float)
    remaining = max(int(total) - int(done), 0)
    eta = (float(elapsed_s) / done * remaining) if done > 0 else None
    return {"done": int(done), "total": int(total),
            "mean_replicate_s": round(float(sec.mean()), 3) if len(sec) else None,
            "max_replicate_s": round(float(sec.max()), 3) if len(sec) else None,
            "elapsed_s": round(float(elapsed_s), 1), "eta_s": None if eta is None else round(eta, 1)}


def bootstrap_draws(match_ids, b: int, seed: int) -> list[BootstrapDraw]:
    """B выборок матчей train с возвращением; seed каждой реплики не зависит от порядка счёта."""
    ids = np.asarray(sorted({str(m) for m in match_ids}), dtype=object)
    if len(ids) == 0:
        raise ValueError("пустой список матчей train — бутстрэп моделей ценности невозможен")
    if int(b) < 0:
        raise ValueError(f"B должно быть неотрицательным, получено {b}")
    out = []
    for i, child in enumerate(np.random.SeedSequence(int(seed)).spawn(int(b))):
        rng = np.random.Generator(np.random.PCG64(child))
        rep_seed = int(rng.integers(1, 2 ** 31 - 1))
        out.append(BootstrapDraw(index=i, seed=rep_seed, matches=ids[rng.integers(0, len(ids), len(ids))]))
    return out


def _as_series(obj) -> tuple[pd.Series, pd.Series | None]:
    """Реплика возвращает Overall (Series/dict/DataFrame) и, опционально, апостериорную дисперсию."""
    var = None
    if isinstance(obj, tuple):
        obj, var = obj[0], obj[1]
    if isinstance(obj, pd.DataFrame):
        if "player_id" not in obj or "overall" not in obj:
            raise TypeError("реплика вернула DataFrame без колонок player_id/overall")
        if var is None and "var_post" in obj:
            var = pd.Series(obj["var_post"].to_numpy(dtype=float), index=obj["player_id"].astype(str))
        obj = pd.Series(obj["overall"].to_numpy(dtype=float), index=obj["player_id"].astype(str))
    s = pd.Series(obj, dtype=float)
    s.index = s.index.astype(str)
    if s.index.duplicated().any():
        raise TypeError("реплика вернула повторяющиеся player_id")
    if var is not None:
        var = pd.Series(var, dtype=float)
        var.index = var.index.astype(str)
    return s, var


def _nonfinite_players(s: pd.Series, v: pd.Series | None) -> list[str]:
    bad = set(s.index[~np.isfinite(s.to_numpy(dtype=float))])
    if v is not None:
        vv = v.to_numpy(dtype=float)
        bad |= set(v.index[~np.isfinite(vv) | (vv < 0)])
    return sorted(bad)


def max_failures(b: int, share: float = MAX_REPLICATE_FAILURE_SHARE) -> int:
    """Сколько упавших реплик допускается (и замещается дополнительными выборками)."""
    return int(math.floor(float(share) * int(b)))


@dataclass
class _Progress:
    """Изменяемое состояние одного прогона бутстрэпа (локально внутри run_model_bootstrap)."""
    overall: dict = field(default_factory=dict)
    varpost: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    seconds: list = field(default_factory=list)
    attempted: int = 0
    restarts: int = 0
    fatal: str = ""


def _absorb(progress: _Progress, done: list, logger=None) -> None:
    for idx, res, err, sec in sorted(done, key=lambda x: x[0]):
        progress.attempted += 1
        progress.seconds.append(float(sec))
        if logger is not None:
            logger.info("L6 реплика %d: %.1f с%s", idx, sec, f" — ошибка данных: {err}" if err else "")
        if err is not None:
            progress.failures.append(f"реплика {idx}: {err}")
            continue
        s, v = res
        bad = _nonfinite_players(s, v)
        if bad:
            progress.fatal = (f"реплика {idx}: нефинитное Overall/var_post у игроков {bad[:5]} — "
                              "бутстрэп провален")
            return
        progress.overall[idx] = s
        if v is not None:
            progress.varpost[idx] = v


class _PoolRunner:
    """Исполнение пачки реплик в пуле с пересозданием пула после гибели процесса."""

    def __init__(self, factory: Callable[[int], Executor], width: int, logger=None,
                 init_error: Callable[[], str] | None = None):
        self.factory, self.width, self.logger = factory, int(width), logger
        # B4: текст исходного исключения инициализатора воркера (пусто — инициализация не падала)
        self.init_error = init_error
        self.restarts = 0
        self.executor = factory(self.width)

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

    def _restart(self, ex: BaseException, n_lost: int) -> str:
        self.close()
        init_msg = self.init_error() if self.init_error is not None else ""
        if init_msg:
            # детерминированная ошибка загрузки контекста: перезапуск не поможет, это не OOM
            return f"инициализация процесса реплик упала: {init_msg}"
        if self.restarts >= MAX_POOL_RESTARTS:
            return (f"пул процессов реплик сломан ({type(ex).__name__}: {ex}) и после {self.restarts} "
                    f"перезапусков (MAX_POOL_RESTARTS={MAX_POOL_RESTARTS}): вероятно, процесс убит "
                    "OOM-killer — уменьшите bootstrap.replicate_workers; готовые реплики в кэше")
        self.restarts += 1
        self.width = max(1, self.width // 2)
        if self.logger is not None:
            self.logger.warning("L6: пул реплик сломан (%s: %s); перезапуск %d/%d шириной %d, повтор %d реплик",
                                type(ex).__name__, ex, self.restarts, MAX_POOL_RESTARTS, self.width, n_lost)
        self.executor = self.factory(self.width)
        return ""

    def map(self, call, batch: list) -> tuple[list, str]:
        """Результаты пачки; неготовые после гибели процесса отправляются в новый пул.

        Возвращает (готовые результаты, причину провала): при исчерпании перезапусков причина
        непуста, а результаты — только то, что успело завершиться.
        """
        results, todo = {}, list(batch)
        while todo:
            futures = [(d, self.executor.submit(call, d)) for d in todo]
            lost, broken = [], None
            for d, fut in futures:
                try:
                    results[d.index] = fut.result()
                except BrokenProcessPool as ex:
                    lost.append(d)
                    broken = ex
            if not lost:
                break
            fatal = self._restart(broken, len(lost))
            if fatal:
                return [results[d.index] for d in batch if d.index in results], fatal
            todo = lost
        return [results[d.index] for d in batch], ""


def _log_progress(logger, progress: _Progress, b: int, t0: float) -> None:
    if logger is None:
        return
    p = progress_summary(len(progress.overall), b, progress.seconds, time.monotonic() - t0)
    eta = "—" if p["eta_s"] is None else f"{p['eta_s']:.0f} с"
    logger.info("L6 бутстрэп: %d/%d реплик (ошибок %d), среднее %s с/реплику, прошло %.0f с, ETA %s",
                p["done"], b, len(progress.failures), p["mean_replicate_s"], p["elapsed_s"], eta)


def _run_batches(replicate, draws, b, runner, t0, time_budget_s, allowed, logger) -> tuple[_Progress, str]:
    progress, stop_reason = _Progress(), ""
    call = partial(_one_replicate, replicate)
    pending = list(draws)
    while pending and len(progress.overall) < b:
        if time_budget_s is not None and (time.monotonic() - t0) > float(time_budget_s):
            stop_reason = (f"исчерпан бюджет времени {float(time_budget_s):.0f} с после "
                           f"{len(progress.overall)} реплик из {b}")
            break
        # Не больше, чем нужно до B: лишние реплики не считаются впрок.
        size = min(runner.width if runner is not None else 1, b - len(progress.overall))
        batch, pending = pending[:size], pending[size:]
        if runner is None:
            done, fatal = [call(d) for d in batch], ""
        else:
            done, fatal = runner.map(call, batch)
            progress.restarts = runner.restarts
        _absorb(progress, done, logger)
        if fatal and not progress.fatal:
            progress.fatal = fatal
        _log_progress(logger, progress, b, t0)
        if progress.fatal or len(progress.failures) > allowed:
            break
    return progress, stop_reason


def run_model_bootstrap(replicate, match_ids, b: int, seed: int, n_jobs: int = 1,
                        time_budget_s: float | None = None, logger=None,
                        executor_factory: Callable[[int], Executor] | None = None,
                        max_failure_share: float = MAX_REPLICATE_FAILURE_SHARE,
                        init_error: Callable[[], str] | None = None) -> BootstrapResult:
    """Переобучение моделей ценности на B бутстрэпах матчей train и разброс Overall.

    ``replicate(matches, index, seed)`` переобучает xG/V (и, если приняты, признаки Ф3) на
    выборке матчей train, пересчитывает δ, компоненты и стягивание и возвращает Overall по
    игрокам теста: Series (player_id -> Overall), DataFrame с player_id/overall[/var_post]
    или пару (Overall, var_post). Данные, по которым считается рейтинг, не ресэмплируются.

    Детерминизм: реплика i всегда получает выборку и seed i-го потомка SeedSequence(seed);
    упавшие реплики замещаются следующими индексами, и в расчёт идут B успешных реплик с
    наименьшими индексами — набор не зависит от порядка и ширины параллельного счёта.
    ``executor_factory(width)`` даёт пул (процессы — l6_replicate.process_pool); без него при
    n_jobs > 1 используются потоки, при n_jobs = 1 — последовательный счёт.

    ``replicate is None`` -> ``not_run``. Исчерпание ``time_budget_s`` -> ``partial`` (или
    ``not_run``, если реплик < 2). Доля ожидаемых ошибок выше порога, нефинитное Overall или
    пул, сломанный (BrokenProcessPool) больше ``MAX_POOL_RESTARTS`` раз, -> ``failed``.
    Неожиданные исключения пробрасываются.
    """
    b = int(b)
    t0 = time.monotonic()
    if replicate is None:
        return BootstrapResult(STATUS_NOT_RUN, "переобучение моделей ценности не подключено "
                                               "(нет функции реплики)", b, 0)
    if b < MIN_BOOT_REPLICATES:
        return BootstrapResult(STATUS_NOT_RUN, f"B={b} < {MIN_BOOT_REPLICATES}: Var_m не определена", b, 0)
    allowed = max_failures(b, max_failure_share)
    draws = bootstrap_draws(match_ids, b + allowed, seed)
    width = max(int(n_jobs or 1), 1)
    factory = executor_factory or (ThreadPoolExecutor if width > 1 else None)
    if factory is None:
        progress, stop_reason = _run_batches(replicate, draws, b, None, t0, time_budget_s, allowed, logger)
    else:
        runner = _PoolRunner(factory, width, logger, init_error)
        try:
            progress, stop_reason = _run_batches(replicate, draws, b, runner, t0, time_budget_s,
                                                 allowed, logger)
        finally:
            runner.close()
    return _summarize(progress, b, allowed, stop_reason, time.monotonic() - t0)


def _summarize(progress: _Progress, b: int, allowed: int, stop_reason: str, elapsed: float) -> BootstrapResult:
    failures = tuple(progress.failures[:5])
    common = dict(n_failed=len(progress.failures), n_attempted=progress.attempted, failures=failures,
                  elapsed_s=elapsed, replicate_seconds=tuple(progress.seconds),
                  pool_restarts=progress.restarts)
    if progress.fatal or len(progress.failures) > allowed:
        reason = progress.fatal or (f"упало {len(progress.failures)} реплик из {progress.attempted} "
                                    f"(допускается {allowed} = {MAX_REPLICATE_FAILURE_SHARE:.0%} от B={b}): "
                                    + "; ".join(progress.failures[:3]))
        return BootstrapResult(STATUS_FAILED, reason, b, len(progress.overall), **common)
    kept = sorted(progress.overall)[:b]
    b_done = len(kept)
    if b_done < MIN_BOOT_REPLICATES:
        reason = stop_reason or f"завершено {b_done} реплик из {b}"
        return BootstrapResult(STATUS_NOT_RUN, reason, b, b_done, **common)
    mat = pd.DataFrame({i: progress.overall[i] for i in kept})
    nrep = mat.notna().sum(axis=1).astype(int)
    # Игрок, отсутствующий хотя бы в одной реплике, получает неизвестную Var_m (NaN), а не
    # дисперсию по неполному набору: гейт L6 n_var_model_known == n_rows это ловит.
    var_model = mat.var(axis=1, ddof=1).where(nrep == b_done)
    vp = {i: progress.varpost[i] for i in kept if i in progress.varpost}
    var_post_mean = pd.DataFrame(vp).mean(axis=1) if vp else pd.Series(dtype=float)
    status = STATUS_DONE if b_done >= b else STATUS_PARTIAL
    reason = "" if status == STATUS_DONE else (stop_reason or f"завершено {b_done} реплик из {b}")
    return BootstrapResult(status, reason, b, b_done, var_model=var_model, var_post_mean=var_post_mean,
                           n_replicates=nrep, **common)


def _one_replicate(replicate, draw: BootstrapDraw):
    """Одна реплика и её время, с; ловятся только ожидаемые ошибки данных (и в процессе пула)."""
    t0 = time.monotonic()
    try:
        res = _as_series(replicate(draw.matches, draw.index, draw.seed))
        return draw.index, res, None, time.monotonic() - t0
    except EXPECTED_REPLICATE_ERRORS as ex:
        return draw.index, None, f"{type(ex).__name__}: {ex}", time.monotonic() - t0
