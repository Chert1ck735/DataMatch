"""L6: сдаваемый `outputs/test_ratings.csv` с интервалом скрытого эффекта и надёжностью.

Слой ничего не обучает и не вводит новой оценки рейтинга. Он берёт готовую латентную таблицу
L5 (Overall и апостериорную дисперсию 1ᵀV_i1), добавляет неопределённость моделей ценности
через бутстрэп матчей train и записывает файл.

Состав строк (архитектура, «Открытые вопросы»): по заданию нужна строка на игрока data/test,
значит все полевые игроки теста. Источники списка:
  * `test_player_toi` из L0 — кто вышел хотя бы на одну смену;
  * заявки `lineup_*` тестовых матчей (`test_roster` из L0) — кто заявлен;
  * `player_seasons_test.csv` и `players_new.csv` из каталога теста.
Вратари не входят (роль из `io.resolve_roles`: заявки важнее справочника). Игрок с неизвестным
амплуа не выбрасывается молча: со сменами — ошибка сразу, без смен — провал гейта покрытия
(сравнение с полным списком без вратарей). Игрок, которого нет в латентной таблице L5, получает
оценку из prior амплуа rating = 1ᵀ(μ_r − m_r); отсутствие prior амплуа — ошибка.

Интервал в файле — только интервал скрытого эффекта: rating ± 1,96·√(var_post + Var_m).
Предсказательный интервал будущего в файл не идёт (отчёт и карточка игрока).

Бутстрэп моделей ценности (`l6_replicate`): реплика m переобучает замороженные рецепты xG и V на
выборке с возвращением из матчей итоговой подгонки L2 (`stage_l2.final_fit_match_ids`),
пересчитывает δ, компоненты test и Overall с замороженными prior L5. Для итога O0 бутстрэп не
запускается: по архитектуре его интервал — только апостериорная дисперсия.

Режимы и файлы. full (B = bootstrap.final_B ≥ 200) пишет сдаваемый `test_ratings.csv` только
при выполненном бутстрэпе (статус done; для O0 — not_applicable), иначе GateError. fast
(B = bootstrap.debug_B) пишет `test_ratings_fast.csv` и никогда не трогает сдаваемый путь.
Рядом с каждым файлом — meta и детализация (Var_m, Монте-Карло-ошибка границ, флаг L5, число
реплик по игроку). `artifacts/l6_report.json` пишется всегда со статусом; провал прогона
переименовывает прежние файлы своего режима в `*.stale`, чтобы они не выглядели актуальными.

L0 и L5 здесь только читаются. Если готового кэша L0 нет, поднимается `L6InputsNotReady`.
"""
from __future__ import annotations

from functools import partial
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import io, l6_replicate, l6_uncertainty
from .cache import (Cache, atomic_write, code_hash, frame_hash, load_checked_json, obj_hash,
                    save_json_with_provenance)
from .config import Config, get_logger
from .gates import GateError, enforce
from .l6_uncertainty import overall_noise_moments  # noqa: F401  (публичный API L6 для отчёта)

L6_CODE = ("l6_uncertainty", "stage_l6", "l6_replicate")

# Формат сдаваемого файла фиксирован архитектурой: порядок и имена колонок не меняются.
OUTPUT_COLUMNS = ["player_id", "rating", "lo95", "hi95", "reliability", "toi_min", "games", "role"]
# Детализация рядом с файлом: неопределённость по игроку (аудит L6 №10).
DETAIL_COLUMNS = ["player_id", "rating", "lo95", "hi95", "var_post", "var_model", "v_tot", "mc_se_bound",
                  "w", "reliability", "flag", "method", "source", "n_replicates"]
L6_REPORT = "l6_report.json"
STALE_SUFFIX = ".stale"
# Файлы режима: сдаваемый только у full.
MODE_STEMS = {"full": "test_ratings", "fast": "test_ratings_fast"}

# Колонки латентной таблицы L5, без которых L6 не работает (контракт L5 → L6).
REQUIRED_L5_COLUMNS = ("player_id", "role", "games", "toi_min", "rating", "var_post", "w")
OPTIONAL_L5_COLUMNS = ("flag", "method")
FIELD_ROLES = ("C", "W", "D")
MODES = ("fast", "full")
# Секции конфигурации в провенансе L6 (аудит 08.10.2026, B1): seed задаёт выборки бутстрэпа и
# симуляции покрытия, eval.drift_grid — предсказательный интервал отчёта.
L6_PROV_SECTIONS = ("shrink", "bootstrap", "seed", "eval")
GATE_KEYS = ("gates_{d}", "gate_{d}", "gates", "gates_l5", "gates_hard", "gate_l5")
FLAG_PRIOR = "нет оценки L5: prior амплуа"

# CONFIG: бюджет времени на бутстрэп моделей ценности, с. Исчерпание даёт partial (fast) или
# провал (full), а не молчаливо усечённый результат. Для full значение по умолчанию; конфигурация
# задаёт его ключом bootstrap.time_budget_s (см. bootstrap_time_budget).
BOOTSTRAP_TIME_BUDGET_S = {"fast": 1800.0, "full": 21600.0}
TIME_BUDGET_KEY = "time_budget_s"
# CONFIG: симуляции проверки самосогласованности и покрытия с переоценкой моментов (отчёт).
COVERAGE_SIMULATIONS = 200
REFIT_COVERAGE_SIMULATIONS = 50
# CONFIG: эталонная будущая экспозиция для предсказательного интервала в карточке (не в файле).
PREDICTIVE_FUTURE_TOI_MIN = 300.0
PREDICTIVE_FUTURE_GAMES = 20.0
# CONFIG: знаков после запятой в файле — рейтинг и границы в xG/60, TOI в минутах.
RATING_DECIMALS = 6
TOI_DECIMALS = 2
# B финального расчёта по архитектуре.
FINAL_MIN_B = 200


class L6InputsNotReady(RuntimeError):
    """Нужный вход L6 (кэш L0 или латентная таблица L5) отсутствует или неполон."""


@dataclass(frozen=True)
class OutputPaths:
    csv: Path
    meta: Path
    detail: Path


def output_paths(cfg: Config, mode: str) -> OutputPaths:
    """Файлы режима: full — сдаваемый test_ratings.csv, fast — test_ratings_fast.csv."""
    stem = MODE_STEMS[mode]
    out = cfg.paths.outputs
    return OutputPaths(out / f"{stem}.csv", out / f"{stem}_meta.json", out / f"{stem}_detail.csv")


# ---------------------------------------------------------------------------------------
# Состав строк файла: все полевые игроки теста
# ---------------------------------------------------------------------------------------
def _l0_test_tables(cfg: Config, cache: Cache) -> dict:
    """Только три таблицы готового кэша L0: справочник игроков, TOI и заявки теста."""
    from . import stage_l0
    key = cache.key("l0", ("l0",), code=stage_l0.L0_CODE)
    quality = cache.load_obj(key, "quality")
    if not cache.done(key) or quality is None:
        raise L6InputsNotReady(f"L0: нет готового кэша {key}; сначала `datamatch.cli prepare`")
    if quality.get("status") not in ("ok", "stress"):
        raise L6InputsNotReady(f"L0: кэш {key} со статусом {quality.get('status')!r} — использование заблокировано")
    tabs = {name: cache.load_df(key, src) for name, src in
            (("players", "players"), ("toi", "test_toi"), ("roster", "test_roster"))}
    missing = [k for k, v in tabs.items() if v is None]
    if missing:
        raise L6InputsNotReady(f"L0: в кэше {key} нет таблиц {missing}")
    tabs["key"] = key
    tabs["quality_status"] = quality.get("status")
    return tabs


def _avangard_team(cfg: Config) -> str:
    av = cfg.get("avangard_team_id")
    if not av:
        raise ValueError("в конфигурации не задан avangard_team_id: счёт полевых «Авангарда» невозможен")
    return str(av)


def test_field_players(cfg: Config, cache: Cache) -> tuple[pd.DataFrame, dict]:
    """Все полевые игроки теста и провенанс их списка.

    Возвращает (таблицу `player_id, role, toi_min, games, team_id, src_*`, отчёт об источниках).
    `expected_field_ids` в отчёте — полный список без вратарей (включая неизвестное амплуа):
    с ним сверяется покрытие файла.
    """
    av = _avangard_team(cfg)
    l0 = _l0_test_tables(cfg, cache)
    players, toi, roster = l0["players"], l0["toi"], l0["roster"]
    seasons = io.load_player_seasons_test(cfg)
    players_new = pd.read_csv(cfg.paths.test_dir / "players_new.csv")
    played = (toi.assign(player_id=toi.player_id.astype(str))
              .groupby("player_id").agg(toi_s=("toi_s", "sum"), games=("match_id", "nunique")))
    src = {
        "test_player_toi": set(played.index),
        "test_lineup_roster": set(roster.player_id.dropna().astype(str)),
        "player_seasons_test": set(seasons.player_id.dropna().astype(str)),
        "players_new": set(players_new.player_id.dropna().astype(str)),
    }
    ids = sorted(set().union(*src.values()))
    role = players.assign(player_id=players.player_id.astype(str)).set_index("player_id")["role"]
    role = role[~role.index.duplicated()]
    r = role.reindex(ids)
    goalies = sorted(r.index[r.eq("G")])
    unresolved = sorted(r.index[r.isna() | ~r.isin(FIELD_ROLES + ("G",))])
    with_shifts = [i for i in unresolved if i in src["test_player_toi"]]
    if with_shifts:
        raise GateError(f"L6: полевые игроки со сменами в тесте без амплуа: {with_shifts[:10]} — "
                        "их нельзя ни оценить, ни молча выбросить из test_ratings.csv")
    field = [i for i in ids if r.get(i) in FIELD_ROLES]
    df = _members_frame(field, role, played, roster, src)
    report = {
        "n_field": int(len(df)), "n_ids_total": len(ids), "n_goalies_excluded": len(goalies),
        "n_role_unresolved": len(unresolved), "role_unresolved_ids": unresolved[:10],
        "expected_field_ids": sorted(set(ids) - set(goalies)),
        "n_with_shifts": int((df.games > 0).sum()), "n_lineup_only": int((df.games == 0).sum()),
        "n_avangard_field": int((df.team_id == av).sum()),
        "by_role": df.role.value_counts().to_dict(),
        "source_counts": {k: len(v) for k, v in src.items()},
        "source_only_counts": {k: len(src[k] - set().union(*(w for name, w in src.items() if name != k)))
                               for k in src},
        "sources": ["L0 test_player_toi", "L0 test_roster (события lineup_* тестовых матчей)",
                    "player_seasons_test.csv", "players_new.csv"],
        "goalie_policy": "вратари не входят; роль по io.resolve_roles (заявки важнее справочника)",
        "l0_cache_key": l0["key"], "l0_status": l0["quality_status"],
    }
    return df, report


def _members_frame(field: list, role: pd.Series, played: pd.DataFrame, roster: pd.DataFrame,
                   src: dict) -> pd.DataFrame:
    team = (roster.assign(player_id=roster.player_id.astype(str))
            .drop_duplicates("player_id").set_index("player_id")["team_id"])
    df = pd.DataFrame({"player_id": field})
    df["role"] = df.player_id.map(role)
    df["toi_min"] = df.player_id.map(played.toi_s / 60.0).fillna(0.0)
    df["games"] = df.player_id.map(played.games).fillna(0).astype(int)
    df["team_id"] = df.player_id.map(team)
    for name, s in src.items():
        df[f"src_{name}"] = df.player_id.isin(s)
    return df.sort_values("player_id").reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# Выбор определения итога и чтение латентной таблицы L5
# ---------------------------------------------------------------------------------------
def _l5_table(l5: dict, definition: str) -> pd.DataFrame | None:
    part = (l5 or {}).get(definition)
    if not isinstance(part, dict):
        return None
    t = part.get("test")
    return t if isinstance(t, pd.DataFrame) and len(t) else None


def _is_true(v) -> bool:
    return isinstance(v, (bool, np.bool_)) and bool(v)


def _l5_gates_ok(report: dict, definition: str) -> tuple[bool, str]:
    """Гейты L5 пройдены только при явном True каждого; нет гейтов — не пройдены."""
    if report.get("status") == "failed":
        return False, "L5 report.status == failed"
    found = False
    for key in (k.format(d=definition) for k in GATE_KEYS):
        g = report.get(key)
        if not isinstance(g, dict):
            continue
        if not g:
            return False, f"{key}: пустой набор гейтов"
        found = True
        bad = sorted(k for k, v in g.items() if not _is_true(v))
        if bad:
            return False, f"{key}: {bad}"
    return (True, "") if found else (False, "отчёт L5 не содержит гейтов")


def choose_definition(l5: dict) -> tuple[str, str]:
    """O1 только при явно пройденных гейтах L5; иначе страховочный O0."""
    report = (l5 or {}).get("report") or {}
    forced = (report.get("final_definition") or report.get("rating_definition")
              or report.get("definition"))
    ok, why = _l5_gates_ok(report, "O1")
    if forced in ("O1", "O0"):
        if _l5_table(l5, forced) is None:
            raise L6InputsNotReady(f"L5 назначил итог {forced}, но таблицы теста для него нет")
        if forced == "O1" and not ok:
            raise L6InputsNotReady(f"L5 назначил итог O1, но его гейты не пройдены ({why})")
        return forced, f"определение задано L5: {forced}"
    o1, o0 = _l5_table(l5, "O1"), _l5_table(l5, "O0")
    if o1 is not None and ok:
        return "O1", "O1 доступен, гейты L5 явно пройдены"
    if o0 is not None:
        cause = "O1 отсутствует в выходе L5" if o1 is None else f"гейты L5 для O1 не пройдены ({why})"
        return "O0", (f"{cause}; итог — страховочный O0, его интервал по архитектуре строится "
                      "только из апостериорной дисперсии")
    raise L6InputsNotReady("L5 не дал ни O1, ни O0 для теста")


def require_final_definition(definition: str, reason: str, mode: str) -> None:
    """B3: сдаваемый файл (full) пишется со страховочным O0 только по явному назначению L5.

    Молчаливый переход на O0 из-за непройденных гейтов O1 в полном режиме — ошибка с понятной
    причиной (ненулевой код выхода), а не тихая подмена итога; в fast — допустимо (отладка).
    """
    if mode == "full" and definition == "O0" and not reason.startswith("определение задано L5"):
        raise L6InputsNotReady(f"полный режим: итог O1 недоступен ({reason}); O0 пишется в сдаваемый "
                               "файл только по явному назначению в отчёте L5 (final_definition)")


def _validated_table(table: pd.DataFrame, definition: str) -> pd.DataFrame:
    """Таблица с индексом player_id и колонками flag/method (флаг L5 обязателен при w = 0)."""
    t = table.copy()
    if t.index.name == "player_id" and "player_id" not in t.columns:
        t = t.reset_index()
    missing = [c for c in REQUIRED_L5_COLUMNS if c not in t.columns]
    if missing:
        raise L6InputsNotReady(f"латентная таблица L5 ({definition}) без колонок {missing}")
    t["player_id"] = t.player_id.astype(str)
    if t.player_id.duplicated().any():
        dup = sorted(t.player_id[t.player_id.duplicated()].unique())[:5]
        raise L6InputsNotReady(f"латентная таблица L5 ({definition}): повторяющиеся player_id {dup}")
    if "flag" not in t.columns:
        zero = sorted(t.player_id[t.w.to_numpy(dtype=float) == 0])
        if zero:
            raise GateError(f"L6: в таблице L5 ({definition}) нет колонки flag, а у {len(zero)} игроков "
                            f"w = 0 ({zero[:5]}): пометка «сигнал не обнаружен» потеряна")
        t["flag"] = ""
    t["flag"] = t.flag.fillna("").astype(str)
    t["method"] = t.method.fillna("").astype(str) if "method" in t.columns else "l5"
    return t.set_index("player_id")


def _prior_for(prior: dict, role: str, definition: str) -> tuple[float, float]:
    """(mean, var) prior амплуа; отсутствие — ошибка (не рейтинг 0 с нулевой шириной)."""
    p = (prior or {}).get(role)
    q = p.get(definition, p) if isinstance(p, dict) else None
    if not (isinstance(q, dict) and "mean" in q and "var" in q):
        raise GateError(f"L6: в выходе L5 нет prior амплуа {role!r} для {definition}: "
                        "игроков без оценки L5 оценить нечем")
    return float(q["mean"]), float(q["var"])


# ---------------------------------------------------------------------------------------
# Сборка строк
# ---------------------------------------------------------------------------------------
def _point_estimates(members: pd.DataFrame, t: pd.DataFrame, prior: dict, definition: str) -> pd.DataFrame:
    """Точечные величины по строкам: апостериор L5 либо prior амплуа."""
    rows = []
    for m in members.itertuples(index=False):
        pid = str(m.player_id)
        if pid in t.index:
            row = t.loc[pid]
            role = row["role"] if isinstance(row.get("role"), str) and row["role"] in FIELD_ROLES else m.role
            rows.append((pid, float(row["rating"]), float(row["var_post"]), float(row["w"]),
                         float(row["toi_min"]), float(row["games"]), role, "l5_posterior",
                         row["flag"], row["method"], role != m.role))
            continue
        mean, var = _prior_for(prior, m.role, definition)
        # Заявленный без смен: экспозиции нет. Игравший, но отсутствующий в L5 — prior с его TOI.
        source = "prior_lineup_only" if m.games <= 0 else "prior_missing_in_l5"
        rows.append((pid, mean, var, 0.0, float(m.toi_min), float(m.games), m.role, source,
                     FLAG_PRIOR, "prior", False))
    cols = ["player_id", "rating", "var_post", "w", "toi_min", "games", "role", "source", "flag",
            "method", "role_mismatch"]
    return pd.DataFrame(rows, columns=cols)


def build_rows(members: pd.DataFrame, table: pd.DataFrame, prior: dict, definition: str,
               boot: l6_uncertainty.BootstrapResult, high: float, medium: float) -> tuple[pd.DataFrame, dict]:
    """Строки файла по всем полевым игрокам теста и сводка для meta."""
    t = _validated_table(table, definition)
    p = _point_estimates(members, t, prior, definition)
    ids = p.player_id.to_numpy()
    var_model = np.full(len(p), np.nan)
    var_boot = None
    n_rep = np.zeros(len(p), dtype=int)
    if boot.available:
        var_model = boot.var_model.reindex(ids).to_numpy(dtype=float)
        n_rep = boot.n_replicates.reindex(ids).fillna(0).to_numpy(dtype=int)
        if len(boot.var_post_mean):
            var_boot = boot.var_post_mean.reindex(ids).to_numpy(dtype=float)
    vm_arg = var_model if boot.available else None
    v_tot = l6_uncertainty.total_variance(p.var_post.to_numpy(), vm_arg, var_boot)
    lo, hi = l6_uncertainty.normal_interval(p.rating.to_numpy(), v_tot)
    detail = p.drop(columns="role_mismatch").assign(
        lo95=lo, hi95=hi, reliability=l6_uncertainty.reliability_labels(p.w.to_numpy(), high, medium),
        games=p.games.round().astype(int), var_model=var_model, v_tot=v_tot,
        mc_se_bound=l6_uncertainty.mc_bound_error(var_model, v_tot, boot.b_done if boot.available else 0),
        n_replicates=n_rep, team_id=members.set_index(members.player_id.astype(str)).team_id.reindex(ids).to_numpy(),
    ).sort_values("player_id").reset_index(drop=True)
    return detail, _rows_info(detail, t, set(ids), int(p.role_mismatch.sum()))


def _rows_info(detail: pd.DataFrame, t: pd.DataFrame, ids: set, role_mismatch: int) -> dict:
    width = (detail.hi95 - detail.lo95).to_numpy(dtype=float)
    return {
        "n_rows": int(len(detail)),
        "by_source": detail.source.value_counts().to_dict(),
        "n_from_l5": int((detail.source == "l5_posterior").sum()),
        "n_prior_lineup_only": int((detail.source == "prior_lineup_only").sum()),
        "n_prior_missing_in_l5": int((detail.source == "prior_missing_in_l5").sum()),
        "n_role_mismatch_l5_vs_l0": int(role_mismatch),
        "by_reliability": detail.reliability.value_counts().to_dict(),
        "by_role": detail.role.value_counts().to_dict(),
        "by_flag": detail.flag[detail.flag != ""].value_counts().to_dict(),
        "by_method": detail.method.value_counts().to_dict(),
        "n_degenerate_intervals": int((width <= 0).sum()),
        "n_var_model_known": int(np.isfinite(detail.var_model).sum()),
        "mean_halfwidth": float(np.mean(width) / 2.0) if len(width) else None,
        "l5_rows_not_in_test_list": sorted(set(t.index) - ids)[:10],
        "n_l5_rows_not_in_test_list": int(len(set(t.index) - ids)),
    }


def structural_gates(detail: pd.DataFrame, members: pd.DataFrame, file_frame: pd.DataFrame,
                     expected_ids: set, boot: l6_uncertainty.BootstrapResult) -> dict:
    """Проверки формата и состава файла: провал означает дефект L6 или входа, а не свойство данных."""
    fin = np.isfinite(detail[["rating", "lo95", "hi95", "v_tot"]].to_numpy()).all()
    lineup_only = detail.source == "prior_lineup_only"
    return {
        "columns_exact": list(file_frame.columns) == OUTPUT_COLUMNS,
        "rows_nonempty": bool(len(detail) > 0),
        "player_id_unique": not detail.player_id.duplicated().any(),
        # Сверка с полным списком без вратарей: игрок с неизвестным амплуа не выпадает молча.
        "covers_all_field_players": set(detail.player_id) == set(expected_ids) and len(detail) == len(expected_ids),
        "no_goalies": bool(detail.role.isin(FIELD_ROLES).all()
                           and not members.role.astype(str).isin(["G"]).any()),
        "values_finite": bool(fin),
        "interval_ordered": bool((detail.lo95 <= detail.rating).all() and (detail.rating <= detail.hi95).all()),
        # Нулевая ширина — заявление о полной уверенности; запрещена для каждой строки.
        "interval_nondegenerate": bool((detail.lo95 < detail.hi95).all()),
        "var_model_known_all_rows": (not boot.available) or bool(np.isfinite(detail.var_model).all()),
        "exposure_nonnegative": bool((detail.toi_min >= 0).all() and (detail.games >= 0).all()),
        "lineup_only_zero_exposure": bool(((detail.toi_min[lineup_only] == 0)
                                           & (detail.games[lineup_only] == 0)).all()),
        "reliability_in_domain": bool(detail.reliability.isin(["HIGH", "MEDIUM", "LOW"]).all()),
    }


def file_frame(detail: pd.DataFrame) -> pd.DataFrame:
    """Файл ровно в формате архитектуры: строковое форматирование чисел, порядок колонок."""
    return pd.DataFrame({
        "player_id": detail.player_id.astype(str),
        "rating": detail.rating.map(lambda v: f"{v:.{RATING_DECIMALS}f}"),
        "lo95": detail.lo95.map(lambda v: f"{v:.{RATING_DECIMALS}f}"),
        "hi95": detail.hi95.map(lambda v: f"{v:.{RATING_DECIMALS}f}"),
        "reliability": detail.reliability.astype(str),
        "toi_min": detail.toi_min.map(lambda v: f"{v:.{TOI_DECIMALS}f}"),
        "games": detail.games.astype(int),
        "role": detail.role.astype(str),
    })[OUTPUT_COLUMNS]


# ---------------------------------------------------------------------------------------
# План бутстрэпа моделей ценности
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BootstrapPlan:
    """Что и как запускать: реплика, список матчей, пул и отпечаток точки расширения."""
    replicate: object
    matches: tuple | None
    reason: str
    hook: str
    applicable: bool = True
    executor_factory: object = None
    n_jobs: int = 1
    init_error: object = None


def _callable_name(fn) -> str:
    return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', type(fn).__name__)}"


def bootstrap_plan(cfg: Config, cache: Cache, definition: str, replicate=None,
                   bootstrap_matches=None, force: bool = False) -> BootstrapPlan:
    """O0 — без бутстрэпа; подставленная реплика — как есть; иначе l6_replicate (пул процессов).

    force (ratings --force/--force-all) доходит до кэша реплик: они пересчитываются. При пуле
    процессов родитель освобождает свою копию контекста — процессы spawn загружают её сами.
    """
    if definition == "O0":
        return BootstrapPlan(None, None, "итог O0: интервал по архитектуре — только апостериорная "
                                         "дисперсия, бутстрэп моделей ценности не запускается",
                             "not_applicable_o0", applicable=False)
    if replicate is not None:
        name = "override:" + _callable_name(replicate)
        try:
            matches = list(bootstrap_matches) if bootstrap_matches is not None else \
                l6_replicate.bootstrap_match_ids(cfg, cache)
        except l6_replicate.ReplicateUnavailable as ex:
            return BootstrapPlan(None, None, str(ex), name + "|matches_unavailable")
        return BootstrapPlan(replicate, tuple(map(str, matches)), "", name,
                             n_jobs=int(cfg["bootstrap"].get("n_jobs", 1)))
    try:
        matches = l6_replicate.bootstrap_match_ids(cfg, cache)
        fn = l6_replicate.make_replicate(cfg, cache, force=force)
        factory = l6_replicate.executor_factory(cfg, cache)
        workers = l6_replicate.replicate_workers(cfg)
    except l6_replicate.ReplicateUnavailable as ex:
        return BootstrapPlan(None, None, str(ex), "unavailable:" + obj_hash(str(ex)))
    if factory is not None:
        l6_replicate.release_context(cfg, cache)
    return BootstrapPlan(fn, tuple(map(str, matches)), "", "l6_replicate.replicate",
                         executor_factory=factory, n_jobs=workers,
                         init_error=partial(l6_replicate.read_init_error, cache))


def bootstrap_time_budget(cfg: Config, mode: str) -> float:
    """Бюджет времени бутстрэпа, с: full — bootstrap.time_budget_s (по умолчанию 21600), fast — константа."""
    if mode != "full":
        return float(BOOTSTRAP_TIME_BUDGET_S[mode])
    raw = (cfg.get("bootstrap") or {}).get(TIME_BUDGET_KEY, BOOTSTRAP_TIME_BUDGET_S["full"])
    budget = float(raw)
    if not np.isfinite(budget) or budget < 0:
        raise ValueError(f"bootstrap.{TIME_BUDGET_KEY} должно быть конечным и ≥ 0, получено {raw!r}")
    return budget


def _run_bootstrap(cfg: Config, plan: BootstrapPlan, mode: str, b: int, log) -> l6_uncertainty.BootstrapResult:
    if not plan.applicable:
        return l6_uncertainty.BootstrapResult(l6_uncertainty.STATUS_NOT_APPLICABLE, plan.reason, b, 0)
    if plan.replicate is None:
        return l6_uncertainty.BootstrapResult(l6_uncertainty.STATUS_NOT_RUN, plan.reason, b, 0)
    return l6_uncertainty.run_model_bootstrap(
        plan.replicate, list(plan.matches), b, cfg.seed, n_jobs=plan.n_jobs,
        time_budget_s=bootstrap_time_budget(cfg, mode), logger=log,
        executor_factory=plan.executor_factory, init_error=plan.init_error)


def bootstrap_gates(boot: l6_uncertainty.BootstrapResult, mode: str) -> dict:
    """failed — всегда провал; full требует done (или not_applicable для O0)."""
    ok_full = (l6_uncertainty.STATUS_DONE, l6_uncertainty.STATUS_NOT_APPLICABLE)
    return {"model_bootstrap_not_failed": boot.status != l6_uncertainty.STATUS_FAILED,
            # B2: в полном режиме упавшая реплика — не тихая замена дополнительной выборкой
            "full_mode_no_failed_replicates": mode != "full" or int(boot.n_failed) == 0,
            "full_mode_requires_done_bootstrap": mode != "full" or boot.status in ok_full,
            "full_mode_requires_final_B": mode != "full" or not boot.available or boot.b_done >= FINAL_MIN_B}


# ---------------------------------------------------------------------------------------
# Отчётные расчёты: покрытие и предсказательный интервал
# ---------------------------------------------------------------------------------------
def self_consistency_report(detail: pd.DataFrame, seed: int) -> dict:
    """Симуляция по опубликованным w и var_post — НЕ тест покрытия (аудит L6 №9).

    τ²_i = v/(1 − w) и s_i = v/w восстанавливаются из тех же чисел, по которым строится
    интервал, поэтому ≈ 95% получается тождественно при любых (даже неверных) моментах L5.
    Проверяется только арифметика интервала.
    """
    note = ("τ² и s восстановлены из опубликованных w и var_post: проверка воспроизводит собственные "
            "допущения L5 и не способна обнаружить неверные моменты")
    m = (detail.w > 0) & (detail.w < 1) & (detail.var_post > 0) & np.isfinite(detail.var_post)
    if not m.any():
        return {"status": "not_run", "kind": "not_a_coverage_test", "note": note,
                "reason": "нет игроков с 0 < w < 1: апостериорная часть выродилась"}
    out = l6_uncertainty.simulate_posterior_coverage(
        detail.w[m].to_numpy(dtype=float), detail.var_post[m].to_numpy(dtype=float),
        n_sim=COVERAGE_SIMULATIONS, seed=seed)
    return dict(out, status="ok", kind="not_a_coverage_test", note=note)


def refit_coverage_report(detail: pd.DataFrame, l5: dict, cfg: Config) -> dict:
    """Покрытие с переоценкой моментов (симуляция из таблицы L6 архитектуры)."""
    mom = l6_uncertainty.role_moments((l5 or {}).get("report") or {})
    if mom is None:
        return {"status": "not_run", "kind": "refit_simulation",
                "reason": "отчёт L5 не публикует moments.univariate[role] (τ², σ², ω²)"}
    d = detail[(detail.source == "l5_posterior") & (detail.toi_min > 0) & (detail.games >= 1)
               & detail.role.isin(list(mom))]
    if d.empty:
        return {"status": "not_run", "kind": "refit_simulation", "reason": "нет игроков с экспозицией"}
    sh = cfg["shrink"]
    out = l6_uncertainty.simulate_refit_coverage(
        d.toi_min.to_numpy(dtype=float), d.games.to_numpy(dtype=float), d.role.to_numpy(dtype=object), mom,
        n_sim=REFIT_COVERAGE_SIMULATIONS, seed=cfg.seed, min_toi_s=float(sh["min_toi_split_min"]) * 60.0,
        eps_abs=float(sh["eps_abs"]), noise_by_role=bool(sh.get("noise_by_role", False)))
    return dict(out, status="ok", moments=mom,
                scope="апостериорная часть на шкале одномерной модели суммы; Var_m аналога не имеет")


def coverage_report(detail: pd.DataFrame, l5: dict, cfg: Config) -> dict:
    return {"self_consistency": self_consistency_report(detail, cfg.seed),
            "refit_simulation": refit_coverage_report(detail, l5, cfg)}


def predictive_report(detail: pd.DataFrame, l5: dict, cfg: Config, moments: dict | None = None) -> dict:
    """Предсказательный интервал будущего по сетке δ² — в отчёт и в карточку, не в файл."""
    rep = (l5 or {}).get("report") or {}
    by_role = l6_uncertainty.role_moments(rep) if moments is None else None
    if by_role and bool((rep.get("moments") or {}).get("noise_by_role")) and set(detail.role) <= set(by_role):
        grid = tuple((cfg.get("eval") or {}).get("drift_grid") or l6_uncertainty.DRIFT_GRID_DEFAULT)
        res = l6_uncertainty.predictive_grid_by_role(
            detail.rating.to_numpy(dtype=float), detail.v_tot.to_numpy(dtype=float), detail.role.to_numpy(),
            by_role, PREDICTIVE_FUTURE_TOI_MIN, PREDICTIVE_FUTURE_GAMES, grid)
        return {"status": "ok", "in_file": False, "moments": by_role, "noise": "по амплуа",
                "future_exposure": {"toi_min": PREDICTIVE_FUTURE_TOI_MIN, "games": PREDICTIVE_FUTURE_GAMES},
                "drift_grid": list(grid),
                "mean_halfwidth_by_delta2": {str(r["delta2"]): r["mean_halfwidth"] for r in res},
                "mean_halfwidth_by_delta2_role": {str(r["delta2"]): r["mean_halfwidth_by_role"] for r in res},
                "note": "δ² из train не оценивается (в train нет дат): сетка — анализ чувствительности"}
    mom = moments or overall_noise_moments(rep)
    if mom is None and l6_uncertainty.role_moments(rep):
        return {"status": "not_run", "in_file": False,
                "reason": ("моменты шума различаются по амплуа, а шум по амплуа к детализации не применён "
                           "(нет noise_by_role или амплуа без моментов): общей пары σ², ω² нет, запасная "
                           "роль не используется (аудит 08.10.2026, B5)")}
    if mom is None:
        return {"status": "not_run",
                "reason": ("отчёт L5 не публикует σ² и ω² шкалы Overall; оценивать их по тестовой "
                           "таблице запрещено правилом L6 (test не используется для подбора σ²)"),
                "in_file": False}
    grid = tuple((cfg.get("eval") or {}).get("drift_grid") or l6_uncertainty.DRIFT_GRID_DEFAULT)
    res = l6_uncertainty.predictive_interval_grid(
        detail.rating.to_numpy(dtype=float), detail.v_tot.to_numpy(dtype=float),
        mom["sigma2"], mom["omega2"], PREDICTIVE_FUTURE_TOI_MIN, PREDICTIVE_FUTURE_GAMES, grid)
    return {"status": "ok", "in_file": False, "moments": mom,
            "future_exposure": {"toi_min": PREDICTIVE_FUTURE_TOI_MIN, "games": PREDICTIVE_FUTURE_GAMES},
            "drift_grid": list(grid),
            "mean_halfwidth_by_delta2": {str(r["delta2"]): r["mean_halfwidth"] for r in res},
            "note": "δ² из train не оценивается (в train нет дат): сетка — анализ чувствительности"}


# ---------------------------------------------------------------------------------------
# Провенанс, провал и запись
# ---------------------------------------------------------------------------------------
def override_fingerprint(replicate, noise_moments, bootstrap_matches) -> str:
    """Отпечаток точек расширения, подставленных вызывающим кодом (меняют числа в файле)."""
    if replicate is None and noise_moments is None and bootstrap_matches is None:
        return "none"
    return obj_hash({
        "replicate": None if replicate is None else _callable_name(replicate),
        "noise_moments": None if noise_moments is None else obj_hash(noise_moments),
        "bootstrap_matches": None if bootstrap_matches is None else obj_hash(list(bootstrap_matches)),
    })


def _l5_hash_columns(table: pd.DataFrame) -> list[str]:
    t = table.reset_index() if "player_id" not in table.columns else table
    return list(REQUIRED_L5_COLUMNS) + [c for c in OPTIONAL_L5_COLUMNS if c in t.columns]


def _provenance(cfg: Config, cache: Cache, l5: dict, table: pd.DataFrame, members: pd.DataFrame,
                expected_ids: set, definition: str, mode: str, b: int, overrides: str,
                plan: BootstrapPlan) -> dict:
    """Провенанс L6: входы L5, состав, код всей цепочки L0–L6, реплика и список матчей."""
    t = table.reset_index() if "player_id" not in table.columns else table
    return cache.provenance(
        "l6", L6_PROV_SECTIONS, L6_CODE,
        l5_table=frame_hash(t[_l5_hash_columns(table)]),
        l5_prior=obj_hash((l5 or {}).get("prior") or {}),
        l5_report=obj_hash((l5 or {}).get("report") or {}),
        members=frame_hash(members[["player_id", "role", "toi_min", "games"]]),
        expected_ids=obj_hash(sorted(expected_ids)),
        chain_code=code_hash(*l6_replicate.replicate_code()),
        replicate_hook=plan.hook,
        bootstrap_matches=obj_hash(list(plan.matches)) if plan.matches is not None else "none",
        definition=definition, mode=mode, B=b, overrides=overrides)


def _mark_stale(paths: OutputPaths) -> list[str]:
    moved = []
    for p in (paths.csv, paths.meta, paths.detail):
        if p.exists():
            os.replace(p, p.with_name(p.name + STALE_SUFFIX))
            moved.append(p.name)
    return moved


def _record_failure(cfg: Config, paths: OutputPaths, mode: str, ex: BaseException, started: float) -> None:
    """Провал прогона: прежние файлы режима -> *.stale, отчёт со статусом failed."""
    moved = _mark_stale(paths)
    report_path = cfg.paths.artifacts / L6_REPORT
    written_now = report_path.exists() and report_path.stat().st_mtime >= started
    if isinstance(ex, GateError) and written_now:
        rep = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        rep = {"stage": "L6", "status": "failed", "mode": mode, "error": f"{type(ex).__name__}: {ex}"}
    rep["stale_outputs"] = moved
    report_path.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    get_logger().error("L6 (%s) провален: %s; прежние файлы режима помечены %s: %s",
                       mode, ex, STALE_SUFFIX, moved)


def _resolve_members(cfg: Config, cache: Cache, members: pd.DataFrame | None) -> tuple[pd.DataFrame, dict, set]:
    if members is None:
        members, report = test_field_players(cfg, cache)
        return members, report, set(report["expected_field_ids"])
    members = members.copy()
    members["player_id"] = members.player_id.astype(str)
    return members, {"n_field": int(len(members)), "sources": ["передан вызывающим кодом"]}, set(members.player_id)


def run_l6(cfg: Config, cache: Cache, l5: dict, mode: str = "fast", force: bool = False, *,
           replicate=None, members: pd.DataFrame | None = None,
           noise_moments: dict | None = None, bootstrap_matches=None) -> dict:
    """Пишет файлы режима (full — сдаваемый `outputs/test_ratings.csv`) и `artifacts/l6_report.json`.

    `mode`: fast — B = bootstrap.debug_B, файл `test_ratings_fast.csv`; full — B = bootstrap.final_B,
    сдаваемый файл только при выполненном бутстрэпе. `replicate`, `bootstrap_matches`, `members` —
    точки подмены для тестов; `noise_moments` — σ²/ω² шкалы Overall для предсказательного интервала.
    Любой провал помечает прежние файлы режима как устаревшие и пишет отчёт со статусом failed.
    """
    if mode not in MODES:
        raise ValueError(f"режим L6 должен быть одним из {MODES}, получено {mode!r}")
    paths = output_paths(cfg, mode)
    started = time.time()
    try:
        return _run_l6(cfg, cache, l5, mode, force, paths, replicate=replicate, members=members,
                       noise_moments=noise_moments, bootstrap_matches=bootstrap_matches)
    except BaseException as ex:              # фиксируем провал и пробрасываем дальше
        _record_failure(cfg, paths, mode, ex, started)
        raise


def _run_l6(cfg: Config, cache: Cache, l5: dict, mode: str, force: bool, paths: OutputPaths, *,
            replicate, members, noise_moments, bootstrap_matches) -> dict:
    log = get_logger()
    report_path = cfg.paths.artifacts / L6_REPORT
    bs = cfg["bootstrap"]
    b = int(bs["debug_B"] if mode == "fast" else bs["final_B"])
    sh = cfg["shrink"]
    high, medium = float(sh["reliability_high"]), float(sh["reliability_medium"])

    definition, definition_reason = choose_definition(l5)
    require_final_definition(definition, definition_reason, mode)
    table = _validated_table(_l5_table(l5, definition), definition)
    members, members_report, expected_ids = _resolve_members(cfg, cache, members)
    base = {"stage": "L6", "mode": mode, "definition": definition}
    enforce("L6", {"field_players_nonempty": len(members) > 0}, report_path, base)

    plan = bootstrap_plan(cfg, cache, definition, replicate, bootstrap_matches, force=force)
    prov = _provenance(cfg, cache, l5, table, members, expected_ids, definition, mode, b,
                       override_fingerprint(replicate, noise_moments, bootstrap_matches), plan)
    reused = None if force else _reusable(paths, prov, mode)
    if reused is not None:
        log.info("L6: %s актуален (provenance совпадает) — без пересчёта", paths.csv.name)
        return reused

    boot = _run_bootstrap(cfg, plan, mode, b, log)
    base = dict(base, model_bootstrap=boot.to_meta(), replicate_hook=plan.hook)
    enforce("L6", bootstrap_gates(boot, mode), report_path, base)
    if boot.available:
        known = boot.var_model.reindex(sorted(members.player_id)).notna()
        enforce("L6", {"var_model_known_all_rows": bool(known.all())}, report_path,
                dict(base, var_model_unknown=sorted(known.index[~known])[:10]))
    detail, rows_info = build_rows(members, table, (l5 or {}).get("prior") or {}, definition,
                                   boot, high, medium)
    frame = file_frame(detail)
    gates = structural_gates(detail, members, frame, expected_ids, boot)
    meta, report = _meta_and_report(cfg, l5, mode, b, definition, definition_reason, boot, plan,
                                    rows_info, members_report, members, detail, gates, noise_moments,
                                    (high, medium))
    enforce("L6", gates, report_path, report)
    return _write(paths, report_path, frame, detail, meta, report, prov, log)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _reusable(paths: OutputPaths, prov: dict, mode: str) -> dict | None:
    """Файл режима переиспользуется только при совпадении провенанса и sha256 CSV/детализации
    из meta (обрезанный обрывом записи файл не годится); full — только статус ok."""
    if not (paths.csv.exists() and paths.detail.exists()):
        return None
    saved = load_checked_json(paths.meta, prov)
    if saved is None or (mode == "full" and saved.get("status") != "ok"):
        return None
    if (saved.get("csv_sha256") != file_sha256(paths.csv)
            or saved.get("detail_sha256") != file_sha256(paths.detail)):
        get_logger().warning("L6: %s не совпадает с sha256 из meta (оборванная запись?) — пересчёт",
                             paths.csv.name)
        return None
    return {"ratings": pd.read_csv(paths.csv, dtype={"player_id": str}), "detail": None,
            "meta": saved, "report": saved.get("report", {}), "reused": True}


def _meta_and_report(cfg, l5, mode, b, definition, definition_reason, boot, plan, rows_info,
                     members_report, members, detail, gates, noise_moments, thresholds) -> tuple[dict, dict]:
    high, medium = thresholds
    status = "ok" if boot.status in (l6_uncertainty.STATUS_DONE, l6_uncertainty.STATUS_NOT_APPLICABLE) else "partial"
    meta = {
        "status": status, "rating_definition": definition, "definition_reason": definition_reason,
        "rating_units": "xG на 60 минут выше описательного центра амплуа m_r",
        "interval": {"kind": "latent_effect",
                     "formula": "rating ± 1,96·√(v_tot), v_tot = E_m[1ᵀV_i1] + Var_m[Overall_i]",
                     "z": l6_uncertainty.Z95, "var_model_included": bool(boot.available),
                     "m_r_uncertainty_included": False, "predictive_interval_in_file": False},
        "mode": mode, "B": b, "B_done": boot.b_done, "model_bootstrap": boot.to_meta(),
        "replicate_hook": plan.hook, "n_jobs": int(plan.n_jobs),
        "bootstrap_matches": {"n": len(plan.matches) if plan.matches else 0,
                              "source": "stage_l2.final_fit_match_ids (разработка + внешний блок)"},
        "rows": rows_info, "field_players": {k: v for k, v in members_report.items() if k != "expected_field_ids"},
        "reliability": {"thresholds": {"HIGH": high, "MEDIUM": medium},
                        "w": "1 − 1ᵀV_i1 / 1ᵀT_r1, обрезанное в [0, 1] (из L5)",
                        "counts": rows_info["by_reliability"]},
        "columns": OUTPUT_COLUMNS, "detail_columns": DETAIL_COLUMNS,
        "number_format": {"rating/lo95/hi95": f"{RATING_DECIMALS} знаков", "toi_min": f"{TOI_DECIMALS} знака",
                          "games": "целое"},
        "gates": gates,
        "test_usage": "test не используется для подбора σ², δ², conformal-поправки и выбора интервала",
    }
    mc = detail.mc_se_bound.to_numpy(dtype=float)
    report = {
        "stage": "L6", "status": status, "mode": mode, "definition": definition,
        "definition_reason": definition_reason, "n_field_test": int(len(members)),
        "field_players": meta["field_players"], "rows": rows_info, "model_bootstrap": boot.to_meta(),
        "coverage_simulation": coverage_report(detail, l5, cfg),
        "predictive_interval": predictive_report(detail, l5, cfg, noise_moments),
        "mc_bound_error": {"mean": float(np.nanmean(mc)) if np.isfinite(mc).any() else None,
                           "max": float(np.nanmax(mc)) if np.isfinite(mc).any() else None,
                           "per_player": "колонка mc_se_bound файла детализации",
                           "note": "Монте-Карло-ошибка границы из-за конечного B; без бутстрэпа не определена"},
        "limitations": _limitations(boot, definition, rows_info),
        "gates": gates,
    }
    return meta, report


def _write(paths: OutputPaths, report_path: Path, frame: pd.DataFrame, detail: pd.DataFrame, meta: dict,
           report: dict, prov: dict, log) -> dict:
    # CSV и детализация — атомарно (tmp + os.replace); meta с их sha256 — последней: она и есть
    # отметка готовности, а _reusable сверяет хэши (интеграционное ревью №4).
    atomic_write(paths.csv, lambda t: frame.to_csv(t, index=False, lineterminator="\n"))
    atomic_write(paths.detail, lambda t: detail[DETAIL_COLUMNS].to_csv(t, index=False, lineterminator="\n"))
    meta = dict(meta, report=report, csv_sha256=file_sha256(paths.csv),
                detail_sha256=file_sha256(paths.detail))
    save_json_with_provenance(paths.meta, meta, prov)
    save_json_with_provenance(report_path, report, prov)
    log.info("L6: %s — %d строк, итог %s, бутстрэп %s (B=%d/%d), код %s", paths.csv.name, len(frame),
             meta["rating_definition"], meta["model_bootstrap"]["status"], meta["B_done"], meta["B"],
             code_hash(*L6_CODE))
    return {"ratings": frame, "detail": detail, "meta": dict(meta, provenance=prov), "report": report,
            "reused": False}


def _limitations(boot: l6_uncertainty.BootstrapResult, definition: str, rows_info: dict) -> list[str]:
    out = []
    if boot.status == l6_uncertainty.STATUS_NOT_RUN:
        out.append(f"Var_m моделей ценности не оценена (model_bootstrap=not_run: {boot.reason}); "
                   "v_tot = апостериорная дисперсия, поэтому интервалы занижены, а не точны")
    elif boot.status == l6_uncertainty.STATUS_PARTIAL:
        out.append(f"бутстрэп неполон: {boot.b_done} реплик из {boot.b_requested} ({boot.reason})")
    if boot.available and boot.b_done < FINAL_MIN_B:
        out.append(f"B = {boot.b_done} < {FINAL_MIN_B}: это отладочная точность оценки дисперсии "
                   "(20–25 %), а не финальный расчёт")
    if definition == "O0":
        out.append("итог — страховочный O0 (on-ice xGD/60 при 5 на 5): он не отделяет игрока "
                   "от звена и команды; интервал — только апостериорная дисперсия")
    if rows_info["n_prior_missing_in_l5"]:
        out.append(f"{rows_info['n_prior_missing_in_l5']} игроков со сменами отсутствуют в латентной "
                   "таблице L5 и оценены из prior амплуа")
    out.append("проверка по опубликованным w/var_post не является тестом покрытия; покрытие — "
               "coverage_simulation.refit_simulation (моменты оцениваются заново)")
    return out
