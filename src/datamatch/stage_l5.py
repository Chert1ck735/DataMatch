"""L5: стягивание компонентов (эмпирический Байес) и итоговая оценка — O1 и страховочный O0.

Вход — `player_match` из L3 (контракт `docs/TIER_A_CONTRACT.md`), выход — латентные таблицы
«одна строка на игрока выборки» для train и test, prior по амплуа и отчёт.

Ставки на 60 минут: x = Σ credit / toi_rate_s × 3600. Знаменатель — именно `toi_rate_*`, то есть
та же допущенная выборка, что и числитель (`valid_for_rate_metrics` в журнале L3). Поэтому
строки «игрок × матч» с нулевой допущенной экспозицией выбрасываются целиком — и из числителя,
и из знаменателя, и из счёта матчей M_i; выброшенная масса credit пишется в отчёт, а не
растворяется в оценке.

Обучаемое только на train: T_r, Σσ, Σω, τ², σ², ω², μ_r. Для применения к test prior-среднее
μ_r берётся по КХЛ-компоненте train; если компоненты нет, берётся train целиком, и это явно
записывается в отчёт (`prior_mu.test_source`). Описательный центр m_r считается на оцениваемой
выборке и в обучении не участвует.

Гейты L5 разделены:
  * программные (`gates_hard`) — конечность, неотрицательность дисперсии, w ∈ [0, 1],
    согласованность Overall с суммой центрированных компонентов, отсутствие тестовых матчей
    в обучении моментов. Их провал — дефект расчёта, он поднимает GateError;
  * статистические (`gates_l5`) — наличие межигрового сигнала, успех Холецкого, доля запасных
    ветвей, объём выборки половин. Их провал не ломает расчёт, а снимает допуск O1:
    `report["o1_usable"] = False`, и L6 по контракту берёт O0.
Жёсткий гейт на пустой таблице не проходит вхолостую (`nonempty_*`), а проверка, которой
не к чему применяться (нет совместных строк), помечается `not_applicable`, а не True.

Для бутстрэпа L6 состояние тестовой совместной модели (μ_r по КХЛ-компоненте, T_r, Σσ, Σω,
одномерная модель суммы, решения о сигнале, амплуа train) сохраняется в кэше L5 объектом
`joint_eb_test` (JSON-совместимый словарь). `load_frozen_overall(cfg, cache)` читает его по
текущему ключу L5, `apply_frozen_overall(state, player_match, positions, cfg)` считает Overall
новой выборки (реплики матчей) с замороженными моментами и пересчитанным на ней m_r — так же,
как точечный L5; на точечной тестовой выборке результат совпадает с `O1.test` побитно.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import l5_o0
from .cache import frame_hash, obj_hash, save_json_with_provenance
from .config import get_logger
from .gates import enforce
from .io import POS_TO_ROLE
from .l5_shrink import (
    METHOD_CHOLESKY_FAILED,
    METHOD_JOINT,
    METHOD_UNIVARIATE,
    MIN_SPLIT_PLAYERS,
    ROLES,
    JointEB,
    reliability,
)

L5_CODE = ("l5_shrink", "l5_o0", "stage_l5")

# CONFIG: категории компонентов вклада L3, входящие в совместную модель Overall (O1).
BASE_CATEGORIES = ["shot", "loss", "recovery", "faceoff", "creation"]
COMP_COLS = [f"c_{c}" for c in BASE_CATEGORIES]

# Схема входа L5 (контракт L3 → L5).
PM_REQUIRED = ["match_id", "player_id", "role", "toi_rate_s"] + COMP_COLS
PM_O0_REQUIRED = ["xgf_5v5", "xga_5v5", "toi_rate_5v5_s"]

# CONFIG: допустимая доля игроков оцениваемой выборки в одномерной запасной ветке Overall.
MAX_FALLBACK_SHARE = 0.5
# CONFIG: уровень замены R — доля TOI (нижняя четверть амплуа) и перцентиль внутри неё.
REPLACEMENT_TOI_QUANTILE = 0.25
REPLACEMENT_PERCENTILE = 20.0
# CONFIG: проверка устойчивости порога R: (квантиль TOI, перцентиль оценки).
REPLACEMENT_STABILITY = ((0.20, 20.0), (0.30, 20.0), (0.25, 15.0), (0.25, 25.0))
# CONFIG: допуск согласованности Overall с суммой центрированных компонентов (xG/60).
OVERALL_SUM_TOL = 1e-9
# Значение гейта, которому не к чему применяться (не True и не False; enforce его не проверяет).
NOT_APPLICABLE = "not_applicable"

# Замороженное состояние тестовой совместной модели для бутстрэпа L6.
FROZEN_STATE_NAME = "joint_eb_test"
FROZEN_STATE_VERSION = 1
FROZEN_COLUMNS = ["player_id", "overall", "var_post", "w", "method", "flag"]
FROZEN_M_R_POLICY = ("m_r пересчитывается на оцениваемой выборке: TOI-взвешенный центр θ̂ амплуа "
                     "у того оценщика, которым посчитана строка (как в точечном L5)")
# Указатель «текущий ключ L5» в кэше: ключ L5 зависит от входов L3, которых у L6 нет.
L5_POINTER_KEY = "l5__current"
L5_POINTER_NAME = "pointer"
# Текст источника μ_r для prior_basis отчёта.
MU_SOURCE_TEXT = {"khl_component": "КХЛ-компоненте train", "train_all": "всему train (КХЛ-компоненты нет)"}


class FrozenStateMissing(RuntimeError):
    """Замороженного состояния L5 нет или оно не соответствует текущим конфигурации/коду/данным."""

NOT_RUN = {
    "model_bootstrap": "not_run — бутстрэп моделей ценности и интервалы считает L6",
    "o2_rapm": "not_run — L4/RAPM не реализован, гейт L4 не пройден, O2 не рассматривается",
    "situational_ratings": "not_run — ситуационные рейтинги 5v5/pp/pk в контракт L5 → L6 не входят",
    "war_g_win": "not_run — константа G_win по восстановленному счёту не калибровалась, WAR не считается",
    "replacement_bootstrap": "not_run — бутстрэп уровня замены R относится к L6",
}


# ---------------------------------------------------------------------------------------
# Подготовка выборки

def check_schema(pm: pd.DataFrame, split: str) -> dict:
    """Проверка схемы входной таблицы; пропуск колонки — дефект контракта, не предположение."""
    need = PM_REQUIRED + PM_O0_REQUIRED
    missing = [c for c in need if c not in pm.columns]
    return {f"pm_schema_{split}": not missing, f"pm_nonempty_{split}": len(pm) > 0,
            "_missing": missing}


def player_roles(pm: pd.DataFrame) -> pd.Series:
    """Амплуа игрока: C/W/D из заявки (L3). Одно значение на игрока, разногласие — по частоте."""
    r = pm[["player_id", "role"]].dropna()
    if r.empty:
        return pd.Series(dtype=object, name="role")
    mode = r.groupby("player_id").role.agg(lambda s: s.value_counts().sort_index().idxmax())
    return mode.rename("role")


def rate_frame(pm: pd.DataFrame, comp_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Строки с допущенной экспозицией: `toi_s` = `toi_rate_s` — один знаменатель с числителем."""
    toi = pm.toi_rate_s.to_numpy(dtype=float)
    keep = toi > 0
    f = pm.loc[keep, ["player_id", "match_id"] + comp_cols].reset_index(drop=True)
    f["toi_s"] = toi[keep]
    dropped = pm.loc[~keep, comp_cols].to_numpy(dtype=float)
    total = np.abs(pm[comp_cols].to_numpy(dtype=float)).sum()
    info = {
        "rows_total": int(len(pm)), "rows_kept": int(keep.sum()), "rows_dropped": int((~keep).sum()),
        "dropped_abs_credit": float(np.abs(dropped).sum()),
        "dropped_abs_credit_share": float(np.abs(dropped).sum() / total) if total > 0 else 0.0,
        "players_total": int(pm.player_id.nunique()),
        "players_without_rate_toi": int(pm.player_id.nunique() - f.player_id.nunique()),
    }
    return f, info


def player_sums(frame: pd.DataFrame, comp_cols: list[str]) -> pd.DataFrame:
    """Суммы компонентов, экспозиция T_i и число матчей M_i по одной и той же выборке строк."""
    return frame.groupby("player_id").agg(**{c: (c, "sum") for c in comp_cols},
                                          toi_s=("toi_s", "sum"), games=("match_id", "nunique"))


# ---------------------------------------------------------------------------------------
# O1: совместная модель

def apply_o1(eb: JointEB, sums: pd.DataFrame, roles: pd.Series, ids: list, cfg) -> tuple[pd.DataFrame, dict]:
    """O1 для оцениваемой выборки; m_r считается здесь же, по этой выборке с весом TOI.

    Игрок без допущенной экспозиции отсутствует в `sums`: для prior-ветки его TOI и M_i равны 0,
    а сам факт отсутствия сохраняется явным признаком `no_rate_exposure`.
    """
    sh = cfg["shrink"]
    s = sums.reindex(ids)
    no_rate = s.toi_s.isna().to_numpy(dtype=bool)
    toi = s.toi_s.fillna(0.0).to_numpy(dtype=float)
    games = s.games.fillna(0).to_numpy(dtype=float)
    vals = s[eb.cols].fillna(0.0).to_numpy(dtype=float)
    rr = np.asarray(roles.reindex(ids).to_numpy(), dtype=object)
    res = eb.overall(ids, rr, vals, toi, games, m_r=None)
    m_r, m_ov = res["m_r"], res["m_r_overall"]
    method = np.asarray(res["method"], dtype=object)
    df = pd.DataFrame({
        "player_id": ids, "role": rr, "games": games.astype(int), "toi_min": toi / 60.0,
        "rating": res["overall"], "var_post": res["var"], "w": np.clip(res["w"], 0.0, 1.0),
        "theta": res["theta_overall"], "definition": "O1",
        "fallback": method == METHOD_UNIVARIATE, "method": method,
        "flag": np.asarray(res["flag"], dtype=object), "no_rate_exposure": no_rate,
    })
    for j, c in enumerate(eb.cols):
        df[f"comp_{c[2:]}" if c.startswith("c_") else f"comp_{c}"] = res["profile"][:, j]
    df["reliability"] = reliability(df.w.to_numpy(), float(sh["reliability_high"]),
                                    float(sh["reliability_medium"]))
    rep = {
        "m_r": {r: [float(v) for v in m_r[r]] for r in ROLES},
        # центр, реально вычтенный из Overall (по оценщику строки; в совместной ветке = Σ m_r)
        "m_r_sum": {r: float(m_ov[r]) for r in ROLES},
        "method_counts": {k: int(v) for k, v in pd.Series(method).value_counts().items()},
        "n_prior_only": int((toi <= 0).sum()),
        "n_no_rate_exposure": int(no_rate.sum()),
        "jitter_log": [dict(e) for e in res["jitter_log"]],
        "cholesky_fallback": [str(p) for p in res["cholesky_fallback"]],
    }
    return df, rep


def comp_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("comp_")]


# ---------------------------------------------------------------------------------------
# Уровень замены и xGAR

def replacement_level(df: pd.DataFrame, toi_q: float = REPLACEMENT_TOI_QUANTILE,
                      pct: float = REPLACEMENT_PERCENTILE) -> dict:
    """R_r — перцентиль стянутой оценки среди игроков нижней четверти TOI амплуа (шкала θ̂)."""
    out = {}
    played = df[(df.toi_min > 0) & (df.games > 0)]
    for r in ROLES:
        g = played[played.role == r]
        if g.empty:
            out[r] = None
            continue
        thr = float(np.quantile(g.toi_min.to_numpy(dtype=float), toi_q))
        low = g[g.toi_min <= thr]
        src = low if len(low) else g
        out[r] = float(np.percentile(src.theta.to_numpy(dtype=float), pct))
    return out


def replacement_report(df: pd.DataFrame) -> dict:
    """R и его устойчивость к выбору квантиля TOI и перцентиля (порог проверяется, не заявляется)."""
    base = replacement_level(df)
    variants = {}
    for q, p in REPLACEMENT_STABILITY:
        variants[f"toi_q={q},pct={p}"] = replacement_level(df, q, p)
    spread = {}
    for r in ROLES:
        vals = [v[r] for v in variants.values() if v[r] is not None]
        spread[r] = (float(max(vals) - min(vals)) if vals and base[r] is not None else None)
    return {"R": base, "variants": variants, "max_spread": spread,
            "definition": "20-й перцентиль θ̂ среди нижней четверти TOI амплуа (train)",
            "scale": "θ̂ без вычитания m_r (xG/60)"}


def add_xgar(df: pd.DataFrame, R: dict) -> pd.DataFrame:
    """xGAR_i = (θ̂_i − R_r) · TOI_i/60; R — уровень замены train, test ничего не настраивает."""
    rv = np.array([R.get(r) if R.get(r) is not None else np.nan for r in df.role], dtype=float)
    out = df.copy()
    out["xgar"] = (out.theta.to_numpy(dtype=float) - rv) * out.toi_min.to_numpy(dtype=float) / 60.0
    return out


# ---------------------------------------------------------------------------------------
# Гейты

def hard_gates(tables: dict, schema: dict) -> dict:
    """Программная корректность: её нарушение — дефект расчёта, а не свойство данных.

    Пустая таблица — провал (`nonempty_*`), а не вхолостую пройденные проверки. Согласованность
    Overall с суммой компонентов без совместных строк не применима: значение `NOT_APPLICABLE`.
    """
    g = dict(schema)
    for name, df in tables.items():
        rating = df.rating.to_numpy(dtype=float)
        var = df.var_post.to_numpy(dtype=float)
        w = df.w.to_numpy(dtype=float)
        g[f"nonempty_{name}"] = bool(len(df) > 0)
        g[f"rating_finite_{name}"] = bool(np.isfinite(rating).all())
        g[f"var_finite_nonneg_{name}"] = bool(np.isfinite(var).all() and (var >= -OVERALL_SUM_TOL).all())
        # w = 0 (prior или «сигнал не обнаружен») не может иметь интервал нулевой ширины
        g[f"var_positive_where_w_zero_{name}"] = bool((var[w <= 0.0] > 0.0).all())
        g[f"w_in_unit_interval_{name}"] = bool(np.isfinite(w).all() and ((w >= 0) & (w <= 1)).all())
        g[f"players_unique_{name}"] = bool(df.player_id.is_unique)
        cc = comp_columns(df)
        if cc:
            joint = (~df.fallback.to_numpy(dtype=bool)) & (df.method.to_numpy() == METHOD_JOINT)
            diff = np.abs(df.loc[joint, cc].to_numpy(dtype=float).sum(axis=1) - rating[joint])
            g[f"components_sum_to_overall_{name}"] = (bool(diff.max() <= OVERALL_SUM_TOL)
                                                      if joint.any() else NOT_APPLICABLE)
    return g


def enforceable(gates: dict) -> dict:
    """Гейты для enforce: неприменимые (`NOT_APPLICABLE`) не проверяются, остальные — как есть."""
    return {k: v for k, v in gates.items() if not (isinstance(v, str) and v == NOT_APPLICABLE)}


def soft_gates(eb: JointEB, o1_test: pd.DataFrame, cfg) -> dict:
    """Статистический допуск O1: провал снимает допуск, но не ломает расчёт (L6 возьмёт O0)."""
    g = {}
    for r in ROLES:
        g[f"signal_present_{r}"] = not bool(eb.no_signal.get(r, True))
        # по сырой 1ᵀT_r1 с относительным порогом — то же решение, что и ветка Overall
        g[f"overall_signal_{r}"] = not bool(eb.overall_no_signal.get(r, True))
    g["cholesky_ok"] = bool(len(o1_test.loc[o1_test.method == METHOD_CHOLESKY_FAILED]) == 0)
    share = float(o1_test.fallback.mean()) if len(o1_test) else 1.0
    g["fallback_share_ok"] = bool(share <= MAX_FALLBACK_SHARE)
    g["split_sample_enough"] = bool(not eb.unstable)
    g["moment_design_conditioned"] = bool(eb.omega_identified)
    return g


# ---------------------------------------------------------------------------------------
# Отчёт

def moments_report(eb: JointEB) -> dict:
    """τ², σ², ω², T_r и диагностика проекций — в единицах (xG/60)² (и ·мин для σ²)."""
    return {
        "components": list(eb.cols),
        "T_r": {r: eb.T[r].tolist() for r in ROLES},
        "T_r_raw": {r: eb.T_raw[r].tolist() for r in ROLES},
        "tau2": {r: [float(v) for v in np.diag(eb.T[r])] for r in ROLES},
        "T_r_trace": {r: float(np.trace(eb.T[r])) for r in ROLES},
        "oneT1": {r: float(eb.T[r].sum()) for r in ROLES},
        "oneT1_raw": {r: float(eb.T_info[r]["oneT1_raw"]) for r in ROLES},
        "overall_threshold": {r: float(eb.overall_threshold[r]) for r in ROLES},
        "overall_no_signal": {r: bool(eb.overall_no_signal[r]) for r in ROLES},
        "Sigma_sigma": eb.Sig_s.tolist(), "Sigma_omega": eb.Sig_o.tolist(),
        "noise_by_role": bool(eb.noise_by_role), "noise_refit": bool(eb.diag.get("noise_refit", False)),
        "noise_converged": bool(eb.diag.get("noise_converged", True)),
        "Sigma_sigma_r": {r: eb.Sig_s_r[r].tolist() for r in ROLES},
        "Sigma_omega_r": {r: eb.Sig_o_r[r].tolist() for r in ROLES},
        "noise_by_role_info": {r: eb.noise_info[r] for r in ROLES},
        "sigma2": [float(v) for v in np.diag(eb.Sig_s)],
        "omega2": [float(v) for v in np.diag(eb.Sig_o)],
        "univariate": {r: {k: (float(v) if not isinstance(v, bool) else v)
                           for k, v in eb.uni[r].items()} for r in ROLES},
        "no_signal": {r: bool(eb.no_signal[r]) for r in ROLES},
        "psd_projection": {"T_r": {r: eb.T_info[r] for r in ROLES},
                           "Sigma_sigma": eb.diag.get("Sigma_sigma_proj"),
                           "Sigma_omega": eb.diag.get("Sigma_omega_proj")},
        "n_split_players": int(eb.diag.get("n_split_players", 0)),
        "min_split_players": MIN_SPLIT_PLAYERS,
        "unstable_joint": bool(eb.unstable),
        "omega_identified": bool(eb.omega_identified),
        "moment_design_cond": float(eb.diag.get("moment_design_cond", float("inf"))),
    }


def eps_log(parts: dict) -> dict:
    """Лог численной защиты: использованный ε Холецкого по игрокам и уходы в prior."""
    out = {}
    for name, rep in parts.items():
        jl = rep.get("jitter_log", [])
        counts: dict[str, int] = {}
        for e in jl:
            counts[str(e["eps"])] = counts.get(str(e["eps"]), 0) + 1
        out[name] = {"n_jitter": len(jl), "eps_counts": counts, "jitter_log": jl[:200],
                     "cholesky_fallback": rep.get("cholesky_fallback", [])}
    return out


# ---------------------------------------------------------------------------------------

def _mu_source(pm_train: pd.DataFrame, comp_cols: list[str]) -> tuple[pd.DataFrame, str, str]:
    """Часть train для μ_r при применении к test: КХЛ-компонента, иначе train целиком."""
    if "is_khl_component" not in pm_train.columns:
        return pm_train, "train_all", "в player_match нет колонки is_khl_component"
    mask = pm_train.is_khl_component.fillna(False).to_numpy(dtype=bool)
    sub = pm_train[mask]
    if not len(sub) or sub.toi_rate_s.sum() <= 0:
        return pm_train, "train_all", "КХЛ-компонента пуста или без допущенной экспозиции"
    return sub, "khl_component", ""


def _tables(key_prefix: str) -> list[str]:
    return [f"{key_prefix}_{s}" for s in ("train", "test")]


def _khl_hash(pm_train: pd.DataFrame) -> str:
    """Хэш колонки КХЛ-компоненты: от неё зависит источник μ_r для test, значит, и ключ кэша."""
    if "is_khl_component" not in pm_train.columns:
        return "absent"
    return frame_hash(pm_train[["is_khl_component"]])


# ---------------------------------------------------------------------------------------
# Замороженное состояние для бутстрэпа L6

def frozen_state(eb_test: JointEB, roles_train: pd.Series, mu_source: str, key: str) -> dict:
    """Состояние, по которому L5 считает test O1: модель, амплуа train и политика m_r (JSON)."""
    return {"version": FROZEN_STATE_VERSION, "definition": "O1", "model": eb_test.to_state(),
            "roles_train": {"player_id": roles_train.index.tolist(), "role": roles_train.tolist()},
            "mu_source": mu_source, "m_r_policy": FROZEN_M_R_POLICY, "cache_key": key}


def _positions_roles(positions: pd.DataFrame | None) -> pd.Series:
    """Амплуа из таблицы позиций: колонка `role` (C/W/D) или `position` (C/LW/RW/D)."""
    if positions is None or not len(positions):
        return pd.Series(dtype=object, name="role")
    if "role" in positions.columns:
        r = positions.set_index("player_id")["role"]
    elif "position" in positions.columns:
        r = positions.set_index("player_id")["position"].map(POS_TO_ROLE)
    else:
        raise KeyError("positions: нужна колонка player_id и role или position")
    r = r[r.isin(ROLES)]
    return r[~r.index.duplicated(keep="first")].rename("role")


def _check_frozen(eb_state: dict, cfg) -> None:
    """Состояние — той же версии и с теми же параметрами shrink, что текущая конфигурация."""
    if eb_state.get("version") != FROZEN_STATE_VERSION or "model" not in eb_state:
        raise FrozenStateMissing(f"L5: состояние версии {eb_state.get('version')!r}, "
                                 f"ожидается {FROZEN_STATE_VERSION}; пересчитайте shrink")
    current = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in dict(cfg["shrink"]).items()}
    frozen = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in eb_state["model"]["shrink"].items()}
    if current != frozen:
        raise ValueError(f"L5: параметры shrink расходятся с замороженными ({frozen} ≠ {current})")


def apply_frozen_overall(eb_state: dict, test_player_match: pd.DataFrame, positions: pd.DataFrame,
                         cfg) -> pd.DataFrame:
    """Overall новой тестовой выборки (реплики матчей) с ЗАМОРОЖЕННЫМИ μ_r, T_r, Σσ, Σω.

    Чистая функция: входы не меняются. Ставки на 60, M_i и prior-ветка — как в `run_l5`
    (`rate_frame` → `player_sums` → `apply_o1`); m_r пересчитывается на этой выборке, поэтому
    шкала совпадает с точечным L5. Строки — игроки `test_player_match` ∪ `positions` (игрок без
    допущенной экспозиции получает prior-ветку). Амплуа: по `role` выборки, затем `positions`,
    затем амплуа train из состояния. Повторённый в реплике матч должен нести свой match_id,
    иначе M_i считает его один раз. Колонки: player_id, overall, var_post, w, method, flag.
    """
    _check_frozen(eb_state, cfg)
    missing = [c for c in PM_REQUIRED if c not in test_player_match.columns]
    if missing:
        raise KeyError(f"apply_frozen_overall: в player_match нет колонок {missing}")
    eb = JointEB.from_state(eb_state["model"], logger=get_logger())
    pos = _positions_roles(positions)
    rt = eb_state["roles_train"]
    roles_train = pd.Series(rt["role"], index=pd.Index(rt["player_id"], name="player_id"), dtype=object)
    roles = player_roles(test_player_match).combine_first(pos).combine_first(roles_train)
    ids = sorted(set(test_player_match.player_id.unique()) | set(pos.index))
    rf, _ = rate_frame(test_player_match, eb.cols)
    df, _ = apply_o1(eb, player_sums(rf, eb.cols), roles, ids, cfg)
    return df.rename(columns={"rating": "overall"})[FROZEN_COLUMNS].reset_index(drop=True)


def _current_key_prefix(cfg, cache) -> str:
    """Префикс ключа L5 без хэша входов: конфигурация shrink, версия данных, код L5."""
    return cache.key("l5", ("shrink",), code=L5_CODE).rsplit("__", 1)[0] + "__"


def load_frozen_overall(cfg, cache) -> dict:
    """Замороженное состояние test O1 из текущего ключа кэша L5; нет или устарело — FrozenStateMissing."""
    ptr = cache.load_obj(L5_POINTER_KEY, L5_POINTER_NAME)
    if not isinstance(ptr, dict) or "key" not in ptr:
        raise FrozenStateMissing("L5: в кэше нет указателя на расчёт shrink — сначала выполните L5 "
                                 "(`datamatch.cli shrink`)")
    key = str(ptr["key"])
    if not key.startswith(_current_key_prefix(cfg, cache)):
        raise FrozenStateMissing(f"L5: последний расчёт ({key}) не соответствует текущим "
                                 "конфигурации shrink, данным или коду L5 — пересчитайте shrink")
    state = cache.load_obj(key, FROZEN_STATE_NAME) if cache.done(key) else None
    if state is None:
        raise FrozenStateMissing(f"L5: в кэше {key} нет объекта {FROZEN_STATE_NAME} — пересчитайте shrink")
    _check_frozen(state, cfg)
    return state


def _save_pointer(cache, key: str, prov: dict) -> None:
    cache.save_obj(L5_POINTER_KEY, L5_POINTER_NAME, {"key": key, "provenance": prov})


# ---------------------------------------------------------------------------------------

def _load_cached(cache, key: str, names: list[str]) -> dict | None:
    """Готовый результат из кэша, если на месте все части, включая замороженное состояние."""
    rep = cache.load_obj(key, "report")
    prior = cache.load_obj(key, "prior")
    state = cache.load_obj(key, FROZEN_STATE_NAME)
    frames = {n: cache.load_df(key, n) for n in names}
    if rep is None or prior is None or state is None or any(f is None for f in frames.values()):
        return None
    return {"O1": {"train": frames["O1_train"], "test": frames["O1_test"]},
            "O0": {"train": frames["O0_train"], "test": frames["O0_test"]},
            "prior": prior, "report": rep}


def _prior_table(eb_test: JointEB, eb0, eb0_test, rep_o1_te: dict, rep_o0_te: dict) -> dict:
    """prior по амплуа для игроков без смен: те же среднее и дисперсия, что у их строк в таблицах."""
    prior = {}
    for r in ROLES:
        prior[r] = {
            "O1": {"mean": float(np.sum(eb_test.mu[r])) - rep_o1_te["m_r_sum"][r],
                   "var": eb_test.prior_var_overall(r)},
            "O0": {"mean": float(eb0_test.mu[r] - rep_o0_te["m_r"][r]), "var": eb0.prior_var(r)},
        }
    return prior


def run_l5(cfg, cache, l3: dict, force: bool = False) -> dict:
    """Стягивание и итоговая оценка: O1 и O0 для train и test, prior по амплуа, отчёт.

    Возвращает ровно контрактный словарь: {"O1": {...}, "O0": {...}, "prior": {...}, "report": {...}}.
    """
    log = get_logger()
    t0 = time.time()
    pm_tr = l3["train"]["player_match"]
    pm_te = l3["test"]["player_match"]
    schema = {}
    for split, pm in (("train", pm_tr), ("test", pm_te)):
        s = check_schema(pm, split)
        missing = s.pop("_missing")
        if missing:
            log.error("L5: в player_match (%s) нет колонок %s", split, missing)
        schema.update(s)
    if not all(schema.values()):
        enforce("l5", schema, cfg.paths.artifacts / "shrink_report.json", {"stage": "l5"})

    prov = cache.provenance(
        "l5", ("shrink",), L5_CODE,
        train_pm=frame_hash(pm_tr[PM_REQUIRED + PM_O0_REQUIRED]),
        test_pm=frame_hash(pm_te[PM_REQUIRED + PM_O0_REQUIRED]),
        train_khl_component=_khl_hash(pm_tr),
        l3_report=obj_hash(l3.get("report", {})))
    key = cache.key("l5", ("shrink",), extra=obj_hash(prov), code=L5_CODE)
    names = _tables("O1") + _tables("O0")
    if not force and cache.done(key):
        cached = _load_cached(cache, key, names)
        if cached is not None:
            log.info("L5: кэш %s", key)
            _save_pointer(cache, key, prov)
            return cached
    cache.invalidate_done(key)

    roles_tr = player_roles(pm_tr)
    roles_te = player_roles(pm_te)
    roles_all = roles_te.combine_first(roles_tr)
    ids_tr = sorted(pm_tr.player_id.unique())
    ids_te = sorted(pm_te.player_id.unique())

    rf_tr, exp_tr = rate_frame(pm_tr, COMP_COLS)
    rf_te, exp_te = rate_frame(pm_te, COMP_COLS)
    sums_tr = player_sums(rf_tr, COMP_COLS)
    sums_te = player_sums(rf_te, COMP_COLS)

    # --- обучение моментов: только train, только допущенная экспозиция
    eb = JointEB(COMP_COLS, cfg["shrink"], logger=log).fit(rf_tr, roles_tr)
    mu_src, mu_source, mu_reason = _mu_source(pm_tr, COMP_COLS)
    if mu_source != "khl_component":
        log.warning("L5: μ_r для test взято по всему train, а не по КХЛ-компоненте: %s", mu_reason)
    rf_mu, exp_mu = rate_frame(mu_src, COMP_COLS)
    eb_test = eb.with_mu(eb.prior_mu(rf_mu, roles_tr))
    state = frozen_state(eb_test, roles_tr, mu_source, key)

    o1_tr, rep_o1_tr = apply_o1(eb, sums_tr, roles_tr, ids_tr, cfg)
    o1_te, rep_o1_te = apply_o1(eb_test, sums_te, roles_all, ids_te, cfg)

    # --- O0: страховка из L0/L1-величин, без компонентов L2/L3
    eb0, rep0_fit = l5_o0.fit_o0(cfg, pm_tr, roles_tr)
    eb0_test = eb0.with_mu(eb0.prior_mu(l5_o0.o0_frame(mu_src)[0], roles_tr))
    o0_tr, rep_o0_tr = l5_o0.rate_o0(eb0, pm_tr, roles_tr, ids_tr, cfg)
    o0_te, rep_o0_te = l5_o0.rate_o0(eb0_test, pm_te, roles_all, ids_te, cfg)

    # --- уровень замены (train) и xGAR в обеих выборках
    repl_o1 = replacement_report(o1_tr)
    repl_o0 = replacement_report(o0_tr)
    o1_tr, o1_te = add_xgar(o1_tr, repl_o1["R"]), add_xgar(o1_te, repl_o1["R"])
    o0_tr, o0_te = add_xgar(o0_tr, repl_o0["R"]), add_xgar(o0_te, repl_o0["R"])
    for df, split in ((o1_tr, "train"), (o1_te, "test"), (o0_tr, "train"), (o0_te, "test")):
        df["split"] = split

    # --- prior по амплуа: игроки без смен (rating = μ_r − m_r, var — как у их строк)
    prior = _prior_table(eb_test, eb0, eb0_test, rep_o1_te, rep_o0_te)

    # --- гейты
    tables = {"O1_train": o1_tr, "O1_test": o1_te, "O0_train": o0_tr, "O0_test": o0_te}
    gh = hard_gates(tables, schema)
    train_matches = set(rf_tr.match_id.unique()) | set(l5_o0.o0_frame(pm_tr)[0].match_id.unique())
    gh["no_test_matches_in_fit"] = bool(not (train_matches & set(pm_te.match_id.unique())))
    gh["roles_known_train"] = bool(o1_tr.role.isin(ROLES).all())
    gh["roles_known_test"] = bool(o1_te.role.isin(ROLES).all())
    # замороженное состояние обязано воспроизводить точечный test O1 (контракт бутстрэпа L6)
    frozen = apply_frozen_overall(state, pm_te, roles_te.rename_axis("player_id").reset_index(), cfg)
    point = o1_te.rename(columns={"rating": "overall"})[FROZEN_COLUMNS].reset_index(drop=True)
    gh["frozen_state_reproduces_o1_test"] = bool(frozen.equals(point))
    gl = soft_gates(eb, o1_te, cfg)
    o1_usable = all(gl.values())

    report = {
        "stage": "l5", "status": "ok" if o1_usable else "degraded",
        "rating_definition": "O1" if o1_usable else "O0",
        "o1_usable": o1_usable, "gates_l5": gl, "gates_hard": gh,
        # Алиас: статистические гейты L5 — это ровно гейты допуска O1. L6 выбирает итог по
        # `rating_definition`, но при его отсутствии читает `gates_O1`; ключ не должен расходиться.
        "gates_O1": gl,
        "moments": moments_report(eb),
        "o0": {"fit": rep0_fit, "train": rep_o0_tr, "test": rep_o0_te},
        "prior_mu": {"train": {r: eb.mu[r].tolist() for r in ROLES},
                     "test": {r: eb_test.mu[r].tolist() for r in ROLES},
                     "test_source": mu_source, "test_source_reason": mu_reason,
                     "test_source_exposure": exp_mu,
                     "o0_train": {r: float(eb0.mu[r]) for r in ROLES},
                     "o0_test": {r: float(eb0_test.mu[r]) for r in ROLES}},
        "m_r": {"train": rep_o1_tr["m_r"], "test": rep_o1_te["m_r"],
                "train_overall": rep_o1_tr["m_r_sum"], "test_overall": rep_o1_te["m_r_sum"],
                "o0_train": rep_o0_tr["m_r"], "o0_test": rep_o0_te["m_r"],
                "basis": "описательный центр оцениваемой выборки с весом TOI по оценщику строки; "
                         "в обучении не участвует"},
        "prior_basis": f"test (μ_r по {MU_SOURCE_TEXT[mu_source]}, m_r по тестовой выборке)",
        "exposure_filter": {"train": exp_tr, "test": exp_te,
                            "rule": "строка учитывается при toi_rate_s > 0; числитель и знаменатель — одна выборка",
                            "no_rate_exposure": {"O1_test": rep_o1_te["n_no_rate_exposure"],
                                                 "O0_test": rep_o0_te["n_no_rate_exposure"]}},
        "fallback": {"O1_train": rep_o1_tr["method_counts"], "O1_test": rep_o1_te["method_counts"],
                     "O1_train_prior_only": rep_o1_tr["n_prior_only"],
                     "O1_test_prior_only": rep_o1_te["n_prior_only"],
                     "O0_test_no_signal": rep_o0_te["n_no_signal"],
                     "univariate_rule": "сырая 1ᵀT_r1 ≤ max(ε_abs, 1e-3·tr(T_r)) или tr(T_r) ≤ ε_abs -> "
                                        "одномерная ветка Overall; в ней сумма компонентов Overall не даёт; "
                                        "без сигнала и там: w = 0, «сигнал не обнаружен», дисперсия — "
                                        "верхняя граница τ² по половинам"},
        "frozen_overall": {"cache_object": FROZEN_STATE_NAME, "version": FROZEN_STATE_VERSION,
                           "m_r_policy": FROZEN_M_R_POLICY,
                           "api": "stage_l5.load_frozen_overall(cfg, cache) -> "
                                  "stage_l5.apply_frozen_overall(state, player_match, positions, cfg)"},
        "eps_log": eps_log({"O1_train": rep_o1_tr, "O1_test": rep_o1_te}),
        "replacement": {"O1": repl_o1, "O0": repl_o0,
                        "applied_from": "train (test не настраивает уровень замены)"},
        "counts": {"players_train": len(ids_tr), "players_test": len(ids_te),
                   "matches_train": int(pm_tr.match_id.nunique()),
                   "matches_test": int(pm_te.match_id.nunique()),
                   "roles_train": {k: int(v) for k, v in o1_tr.role.value_counts().items()},
                   "roles_test": {k: int(v) for k, v in o1_te.role.value_counts().items()}},
        "reliability": {"thresholds": {"high": float(cfg["shrink"]["reliability_high"]),
                                       "medium": float(cfg["shrink"]["reliability_medium"])},
                        "O1_test": {k: int(v) for k, v in o1_te.reliability.value_counts().items()},
                        "O0_test": {k: int(v) for k, v in o0_te.reliability.value_counts().items()}},
        "not_run": dict(NOT_RUN),
        "provenance": prov, "cache_key": key, "seconds": round(time.time() - t0, 2),
    }
    enforce("l5", enforceable(gh), cfg.paths.artifacts / "shrink_report.json", report)
    save_json_with_provenance(cfg.paths.artifacts / "shrink_report.json", report, prov)
    if not o1_usable:
        log.warning("L5: O1 не прошёл статистические гейты %s; итог по контракту — O0",
                    [k for k, v in gl.items() if v is not True])

    out = {"O1": {"train": o1_tr, "test": o1_te}, "O0": {"train": o0_tr, "test": o0_te},
           "prior": prior, "report": report}
    for name, df in (("O1_train", o1_tr), ("O1_test", o1_te), ("O0_train", o0_tr), ("O0_test", o0_te)):
        cache.save_df(key, name, df)
    cache.save_obj(key, "report", report)
    cache.save_obj(key, "prior", prior)
    cache.save_obj(key, FROZEN_STATE_NAME, state)
    cache.mark_done(key)
    _save_pointer(cache, key, prov)
    log.info("L5: O1 train %d / test %d игроков, итог %s, за %.1f с", len(o1_tr), len(o1_te),
             report["rating_definition"], time.time() - t0)
    return out
