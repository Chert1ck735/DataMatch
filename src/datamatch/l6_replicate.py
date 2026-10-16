"""L6: одна реплика бутстрэпа моделей ценности — переобучение замороженных рецептов xG и V.

Смысл (архитектура L6, «интервал скрытого эффекта»). Реплика m — переобучение моделей ценности
на бутстрэпе матчей train при фиксированных данных, по которым считается рейтинг:

  1. выборка с возвращением из ровно тех матчей, на которых L2 подгоняет итоговую процедуру
     (`stage_l2.final_fit_match_ids`: разработка + внешний блок, допустимые схемы). Повторённый
     матч получает новое имя `G0001~1`, а его узлы, отрезки и владения — новые ID: L2 запрещает
     общий ID отрезка у разных матчей (иначе возвраты G склеивались бы через матчи);
  2. замороженные рецепты, выбранные на разработке, переобучаются на выборке: xG —
     `stage_l2.fit_xg(cfg, spec L1, nodes)`, награды — `stage_l2.reward`, V —
     `l2_procedure.ValueProcedure(cfg, spec L2, seed).fit(...)`. Фолды калибровки — по ИСХОДНОМУ
     матчу (`stage_l2.final_fit_folds`), поэтому копии одного матча не попадают по обе стороны фолда;
  3. применение только к узлам test: V(test, нейтральная сила), δ = `l2_value.td_delta`,
     журнал и «игрок × матч» — `stage_l3.contribute` с теми же аргументами, что у L3;
  4. Overall test с ЗАМОРОЖЕННЫМИ prior/моментами L5: `stage_l5.load_frozen_overall` и
     `stage_l5.apply_frozen_overall` (prior в реплике не переоценивается — меняются только δ).

Выход реплики — DataFrame(player_id, overall, var_post) по игрокам теста. Единицы — xG/60 выше
описательного центра m_r, как в латентной таблице L5.

Seed (ревью F6). Реплика m переобучает и xG, и V на seed реплики: `fit_frozen_recipes` передаёт
обеим процедурам конфигурацию с seed = seed реплики (её же получают рефиты xG в фолдах калибровки
V через `ValueProcedure.cfg`). Var_m поэтому согласованно смешивает выборочный шум бутстрэпа
матчей и алгоритмический шум seed обеих моделей, а не только V.

Память (ревью F3). Подгонке нужны не все ~110 колонок узлов L0: перед графом бутстрэпа остаются
только колонки, на которые ссылается код подгонки (`fit_columns`: исходники stage_l2.reward/fit_xg,
l1_xg, l2_procedure, l2_value, l2_gbm, node_graph.subset_nodes) и механика графа. Тест сверяет
реплику на урезанном графе с репликой на полном.

Кэш: каждая реплика сохраняется под ключом, включающим хэши рецептов и входов, хэш выборки
матчей, seed, индекс и код L0–L6, поэтому прерванный прогон продолжает с места остановки.
`ratings --force` пересчитывает реплики (метка прогона, см. `make_replicate`).

Параллельность — пул процессов spawn (ревью F4: fork при живых потоках OpenMP родителя может
зависнуть). Контекст не наследуется: инициализатор процесса один раз загружает его из кэша
(`load_context`) и ограничивает потоки BLAS/OpenMP так, что процессы × потоки ≤ ядер.
"""
from __future__ import annotations

import fcntl
import hashlib
import inspect
import multiprocessing
import os
import pickle
import re
import traceback
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import Cache, obj_hash
from .config import Config

# Разделитель имени копии матча: G0001 -> G0001~1, G0001~2 (первое вхождение — исходное имя).
COPY_SEPARATOR = "~"
# Ссылки графа на ID узла (−1 — нет ссылки); перенумеровываются вместе с узлами.
NODE_REF_COLUMNS = ("next_node", "prev_node", "last_pass_node")
# Глобальные целые ID, обязанные идентифицировать один матч (L2: evaluation_weights).
GLOBAL_INT_ID_COLUMNS = ("segment", "possession")
# Строковые ключи вида "<match_id>:...": префикс заменяется именем копии.
MATCH_PREFIXED_KEY_COLUMNS = ("segment_key", "possession_key", "player_shift_id")
KEY_SEPARATOR = ":"
# Выход реплики (вход l6_uncertainty.run_model_bootstrap).
REPLICATE_COLUMNS = ("player_id", "overall", "var_post")
REPLICATE_TABLE = "overall"
# CONFIG (bootstrap.replicate_workers): процессов по умолчанию. Замер 06.10 (ревью F3): узлы
# итоговой подгонки 1,67 млн × 111 колонок = 1,4 ГБ, граф бутстрэпа столько же, подмножества
# фолдов калибровки V (subset_nodes копирует) — итого ~3–4 ГБ на процесс при всех колонках;
# у каждого процесса spawn ещё своя копия контекста. Отбор колонок (`prune_fit_columns`) режет
# это примерно вдвое, но на машине 15 ГБ ОЗУ безопасно не больше двух процессов.
DEFAULT_REPLICATE_WORKERS = 2
# Способ запуска процессов пула: spawn не копирует потоки OpenMP/BLAS родителя (fork мог зависнуть).
START_METHOD = "spawn"
# Переменные окружения, ограничивающие потоки BLAS/OpenMP в процессе пула (процессы × потоки ≤ ядер).
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
# Подсказка числа потоков моделей для процесса пула. LightGBM в l1_xg/l2_gbm задаёт n_jobs=4 явно,
# и OMP_NUM_THREADS его не ограничивает: чтобы ограничить, модели должны читать эту переменную.
MODEL_THREADS_ENV = "DATAMATCH_MODEL_THREADS"
# Код подгонки xG/V, ссылки которого на колонки узлов определяют нужные колонки графа (F3).
FIT_SOURCES = (("stage_l2", "reward"), ("stage_l2", "fit_xg"), ("l1_xg", None), ("l2_procedure", None),
               ("l2_value", None), ("l2_gbm", None), ("node_graph", "subset_nodes"))
# Механика графа бутстрэпа (копии матчей, перенумерация) — нужна всегда, если колонка есть.
GRAPH_COLUMNS = ("match_id", "node", "source_node") + NODE_REF_COLUMNS + GLOBAL_INT_ID_COLUMNS
# Блокировка в каталоге кэша: процессы пула загружают контекст по очереди — загрузка читает весь
# L0 (пик ~8 ГБ по замеру F3), одновременная загрузка двумя процессами легко исчерпала бы ОЗУ.
CONTEXT_LOAD_LOCK = "_l6_context_load.lock"
# Имя файла метки прогона с --force в каталоге кэша реплики.
RUN_TOKEN = "force_run"
# Секции конфигурации, от которых зависит результат реплики (ключ кэша).
REPLICATE_SECTIONS = ("data", "splits", "xg", "value", "contrib", "shrink", "seed")
L6_REPLICATE_CODE = ("l6_replicate", "l6_uncertainty", "stage_l6")
# Обязательный API замороженного Overall в stage_l5 (добавляет агент L5).
FROZEN_API = ("load_frozen_overall", "apply_frozen_overall")


class ReplicateUnavailable(RuntimeError):
    """Реплику нельзя собрать: нет актуальных L0–L2/L5 или API замороженного Overall."""


class ReplicateDataError(ValueError):
    """Ожидаемая ошибка данных одной реплики (входит в l6_uncertainty.EXPECTED_REPLICATE_ERRORS)."""


class ReplicateContractError(RuntimeError):
    """Нарушен контракт между слоями (не ошибка данных: не ловится бутстрэпом)."""


@dataclass(frozen=True)
class BootGraph:
    """Граф выборки бутстрэпа и исходный матч каждой строки (для фолдов)."""
    nodes: pd.DataFrame
    original_match: np.ndarray


@dataclass(frozen=True)
class ReplicateContext:
    """Всё неизменное между репликами; загружается один раз на процесс (родитель или процесс пула)."""
    final_nodes: pd.DataFrame
    match_ids: tuple
    splits: dict
    xg_spec: dict
    value_spec: dict
    test: dict
    positions: pd.DataFrame
    member_ids: tuple
    team_comp: dict
    khl: int | None
    variant: str
    alpha: float
    frozen: object
    fingerprint: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------
# Матчи финальной подгонки и граф бутстрэпа
# ---------------------------------------------------------------------------------------
def final_fit_matches(splits: dict, schemes: tuple) -> list[str]:
    """Матчи, из которых ресэмплирует бутстрэп: ровно матчи итоговой подгонки L2."""
    from . import stage_l2
    ids = [str(m) for m in stage_l2.final_fit_match_ids(splits, schemes)]
    if not ids:
        raise ReplicateUnavailable("stage_l2.final_fit_match_ids вернул пустой список матчей")
    if len(set(ids)) != len(ids):
        raise ReplicateUnavailable("stage_l2.final_fit_match_ids вернул повторяющиеся матчи")
    return sorted(ids)


def copy_match_id(match_id: str, k: int) -> str:
    """Имя k-й копии матча; k = 0 — исходное имя."""
    return str(match_id) if k == 0 else f"{match_id}{COPY_SEPARATOR}{k}"


def _copy_plan(nodes: pd.DataFrame, matches_sample) -> tuple[np.ndarray, np.ndarray]:
    """Позиции строк исходного графа и номер копии для каждой строки графа бутстрэпа."""
    groups = nodes.groupby("match_id", sort=False).indices
    counts = Counter(str(m) for m in matches_sample)
    unknown = sorted(set(counts) - set(groups))
    if unknown:
        raise ReplicateDataError(f"в выборке матчи вне графа train: {unknown[:5]}")
    pos, copy = [], []
    for mid in sorted(counts):
        idx = groups[mid]
        for k in range(counts[mid]):
            pos.append(idx)
            copy.append(np.full(len(idx), k, dtype=np.int64))
    return np.concatenate(pos), np.concatenate(copy)


def _shift_ids(values: np.ndarray, copy: np.ndarray, stride: int, keep_negative: bool) -> np.ndarray:
    shifted = values + copy * stride
    return np.where(values < 0, values, shifted) if keep_negative else shifted


def _copy_names(original: np.ndarray, copy: np.ndarray) -> np.ndarray:
    """Имена копий без поэлементного цикла: k = 0 — исходное имя, иначе `<match>~k`."""
    names = pd.Series(original, dtype=object)
    dup = copy > 0
    names[dup] = names[dup] + COPY_SEPARATOR + pd.Series(copy[dup]).astype(str).to_numpy()
    return names.to_numpy(dtype=object)


def _rename_keys(keys: pd.Series, original: np.ndarray, new: np.ndarray, copy: np.ndarray,
                 column: str) -> np.ndarray:
    """Префикс `<match_id>:` ключа заменяется именем копии (только у строк копий k ≥ 1)."""
    out = keys.to_numpy(dtype=object).copy()
    rows = np.flatnonzero(keys.notna().to_numpy() & (copy > 0))
    if not len(rows):
        return out
    parts = pd.Series(out[rows]).astype(str).str.split(KEY_SEPARATOR, n=1)
    if not (parts.str[0].to_numpy() == original[rows]).all():
        raise ReplicateDataError(f"{column}: ключ не начинается с match_id своего матча")
    out[rows] = new[rows] + KEY_SEPARATOR + parts.str[1].fillna("").to_numpy(dtype=object)
    return out


def _renumber_nodes(b: pd.DataFrame) -> None:
    """ID узлов -> позиции 0..n−1 и ссылки через них (как node_graph.subset_nodes, без копии кадра)."""
    ids = pd.Index(b.node.to_numpy())
    for c in NODE_REF_COLUMNS:
        if c not in b:
            continue
        old = b[c].to_numpy()
        mapped = ids.get_indexer(old)
        if ((old >= 0) & (mapped < 0)).any():
            raise ReplicateDataError(f"{c}: ссылка на узел вне своего матча")
        b[c] = mapped
    b["node"] = np.arange(len(b))


@lru_cache(maxsize=1)
def _fit_source_text() -> str:
    import importlib
    parts = []
    for module, attr in FIT_SOURCES:
        mod = importlib.import_module(f"{__package__}.{module}")
        parts.append(inspect.getsource(getattr(mod, attr) if attr else mod))
    return "\n".join(parts)


def fit_columns(columns) -> tuple[str, ...]:
    """Колонки узлов, нужные подгонке xG/V реплики, в исходном порядке.

    Колонка нужна, если код подгонки (FIT_SOURCES) ссылается на неё атрибутом (`nodes.x`) или
    строковым литералом (`"t_mono"`, в том числе в списках признаков и проверках `in nodes`),
    либо она входит в механику графа. Правило консервативно: лишнее совпадение только оставляет
    колонку. Необязательные колонки (t_mono, valid_for_transition, …) сохраняются, поэтому ветки
    «если колонка есть» ведут себя так же, как на полном графе.
    """
    text = _fit_source_text()
    keep = []
    for c in map(str, columns):
        pattern = r"(\.{0}\b|[\"']{0}[\"'])".format(re.escape(c))
        if c in GRAPH_COLUMNS or re.search(pattern, text):
            keep.append(c)
    return tuple(keep)


def prune_fit_columns(nodes: pd.DataFrame) -> pd.DataFrame:
    """Узлы только с колонками подгонки; если лишних нет — тот же объект без копии."""
    keep = list(fit_columns(tuple(nodes.columns)))
    return nodes if len(keep) == nodes.shape[1] else nodes[keep]


def bootstrap_graph(nodes: pd.DataFrame, matches_sample) -> BootGraph:
    """Граф выборки с возвращением: копии матчей с новыми именами и уникальными ID.

    Узлы и ссылки k-й копии сдвигаются на k·(max+1), затем перенумеровываются в 0..n−1;
    отрезки и владения сдвигаются так же и остаются уникальными для каждой копии; строковые
    ключи получают префикс имени копии. `source_node` сохраняет исходный ID узла L0.
    Вход не меняется: работа идёт с собственной выборкой строк.
    """
    if not nodes.node.is_unique:
        raise ReplicateDataError("ID узлов исходного графа train не уникальны")
    pos, copy = _copy_plan(nodes, matches_sample)
    b = nodes.iloc[pos].reset_index(drop=True)
    original = b.match_id.astype(str).to_numpy(dtype=object)
    renamed = _copy_names(original, copy)
    if "source_node" not in b:
        b["source_node"] = b.node.to_numpy()
    b["match_id"] = renamed
    b["node"] = _shift_ids(b.node.to_numpy(dtype=np.int64), copy, int(nodes.node.max()) + 1, False)
    for c in NODE_REF_COLUMNS:
        if c in b:
            b[c] = _shift_ids(b[c].to_numpy(dtype=np.int64), copy, int(nodes.node.max()) + 1, True)
    for c in GLOBAL_INT_ID_COLUMNS:
        if c in b:
            b[c] = _shift_ids(b[c].to_numpy(dtype=np.int64), copy, int(nodes[c].max()) + 1, True)
    for c in MATCH_PREFIXED_KEY_COLUMNS:
        if c in b:
            b[c] = _rename_keys(b[c], original, renamed, copy, c)
    _renumber_nodes(b)
    return BootGraph(nodes=b, original_match=original)


def original_match_folds(original_match: np.ndarray, splits: dict) -> np.ndarray:
    """Целоматчевые фолды по ИСХОДНОМУ матчу (правило stage_l2.final_fit_folds)."""
    from . import stage_l2
    return stage_l2.final_fit_folds(pd.DataFrame({"match_id": np.asarray(original_match)}), splits)


# ---------------------------------------------------------------------------------------
# Шаги реплики (тяжёлые; в тестах подменяются)
# ---------------------------------------------------------------------------------------
def replicate_config(cfg: Config, seed: int) -> Config:
    """Конфигурация реплики: seed = seed реплики (xG и его рефиты в фолдах V читают cfg.seed)."""
    return cfg.with_overrides({"seed": int(seed)})


def fit_frozen_recipes(cfg: Config, ctx: ReplicateContext, nodes: pd.DataFrame, folds: np.ndarray,
                       seed: int):
    """Переобучение замороженных рецептов xG (L1) и V (L2) на графе выборки, обе — на seed реплики."""
    from . import l2_procedure, stage_l2
    rcfg = replicate_config(cfg, seed)
    xg = stage_l2.fit_xg(rcfg, ctx.xg_spec, nodes)
    r = stage_l2.reward(nodes, xg)
    proc = l2_procedure.ValueProcedure(rcfg, ctx.value_spec, int(seed)).fit(
        nodes, r, xg_spec=ctx.xg_spec, folds=folds)
    return xg, proc


def apply_to_test(cfg: Config, ctx: ReplicateContext, xg, proc) -> pd.DataFrame:
    """δ на узлах test и «игрок × матч» test (те же аргументы contribute, что у stage_l3)."""
    from . import l2_value, stage_l2, stage_l3
    te = ctx.test
    nodes = te["nodes"]
    r = stage_l2.reward(nodes, xg)
    v = proc.delta_value(nodes, neutral=True)   # та же V для δ, что в точечной оценке (value.delta_from_raw)
    values = nodes[["node", "match_id"]].assign(r=r, value=v, delta=l2_value.td_delta(nodes, r, v))
    part = stage_l3.contribute(nodes, values, te["toi"], te["stints"], te["events"], ctx.positions,
                               ctx.team_comp, ctx.khl, "test", ctx.variant, ctx.alpha)
    bad = sorted(k for k, g in part["report"]["gates"].items() if g is not True)
    if bad:
        raise ReplicateDataError(f"гейты L3 реплики не пройдены: {bad}")
    return part["player_match"]


def frozen_api() -> tuple[Callable, Callable]:
    """Функции замороженного Overall из stage_l5 или ReplicateUnavailable с причиной."""
    from . import stage_l5
    funcs = tuple(getattr(stage_l5, name, None) for name in FROZEN_API)
    missing = [n for n, f in zip(FROZEN_API, funcs) if not callable(f)]
    if missing:
        raise ReplicateUnavailable(f"stage_l5 не предоставляет {missing}: Overall реплики с "
                                   "замороженными prior L5 посчитать нельзя")
    return funcs


def frozen_overall(cfg: Config, ctx: ReplicateContext, pm: pd.DataFrame) -> pd.DataFrame:
    """Overall test с замороженными prior/моментами L5; строка на каждого игрока теста."""
    _, apply = frozen_api()
    # L5 ждёт player_id колонкой (как в run_l5), а stage_l3.player_positions держит его в индексе:
    # без reset_index первая же реплика падала KeyError уже после полного переобучения xG и V.
    positions = (ctx.positions.loc[ctx.positions.index.intersection(list(ctx.member_ids))]
                 .rename_axis("player_id").reset_index())
    out = apply(ctx.frozen, pm, positions, cfg)
    missing = [c for c in REPLICATE_COLUMNS if c not in getattr(out, "columns", ())]
    if missing:
        raise ReplicateContractError(f"stage_l5.apply_frozen_overall вернул таблицу без колонок {missing}")
    return pd.DataFrame({"player_id": out.player_id.astype(str).to_numpy(),
                         "overall": out.overall.to_numpy(dtype=float),
                         "var_post": out.var_post.to_numpy(dtype=float)})


def compute_replicate(cfg: Config, ctx: ReplicateContext, matches_sample, seed: int) -> pd.DataFrame:
    """Реплика без кэша: граф выборки -> переобучение -> test -> замороженный Overall."""
    graph = bootstrap_graph(prune_fit_columns(ctx.final_nodes), matches_sample)
    folds = original_match_folds(graph.original_match, ctx.splits)
    xg, proc = fit_frozen_recipes(cfg, ctx, graph.nodes, folds, seed)
    pm = apply_to_test(cfg, ctx, xg, proc)
    return frozen_overall(cfg, ctx, pm)[list(REPLICATE_COLUMNS)]


# ---------------------------------------------------------------------------------------
# Контекст, кэш и параллельность
# ---------------------------------------------------------------------------------------
_CONTEXTS: dict[str, ReplicateContext] = {}


def replicate_code() -> tuple[str, ...]:
    """Код цепочки L0–L6, от которого зависит реплика (часть ключа кэша и провенанса L6)."""
    from . import stage_l0, stage_l3, stage_l5
    return tuple(sorted(set(stage_l0.L0_CODE + stage_l3.L3_CODE + stage_l5.L5_CODE + L6_REPLICATE_CODE)))


def _state_hash(obj) -> str:
    return hashlib.sha1(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()[:12]


def _load_upstream(cfg: Config, cache: Cache) -> dict:
    from . import stage_l3
    from .gates import GateError
    try:
        l0 = stage_l3._load_l0(cfg, cache)
        sp = stage_l3._load_splits(cfg, cache, l0)
        sel = stage_l3._load_l1_selection(cfg, cache, l0, sp)
        l2 = stage_l3._load_l2(cfg, cache, l0, sp, sel)
    except (stage_l3.UpstreamNotReady, GateError) as ex:
        raise ReplicateUnavailable(f"цепочка L0–L2 не готова: {ex}") from ex
    return {"l0": l0, "splits": sp, "l1": sel, "l2": l2}


def load_context(cfg: Config, cache: Cache) -> ReplicateContext:
    """Контекст реплик из готовых кэшей L0–L2 и замороженного L5 (ничего не пересчитывает)."""
    from . import stage_l3, stage_l6
    load_frozen, _ = frozen_api()
    up = _load_upstream(cfg, cache)
    l0, sp, l2 = up["l0"], up["splits"], up["l2"]
    match_ids = final_fit_matches(sp, tuple(cfg["data"]["train_schemes"]))
    final_nodes = prune_fit_columns(l2["train_nodes"])
    if sorted(set(final_nodes.match_id.astype(str))) != match_ids:
        raise ReplicateUnavailable("узлы итоговой подгонки L2 не совпадают с final_fit_match_ids")
    members, _ = stage_l6.test_field_players(cfg, cache)
    te = l0["test"]
    frozen = load_frozen(cfg, cache)
    team_comp, khl = stage_l3._team_components(l0, sp)
    c = cfg["contrib"]
    ctx_fp = {"l0": l0["key"], "l2": l2["key"], "xg_spec": obj_hash(up["l1"]["spec"]),
              "value_spec": obj_hash(l2["selection"]["spec"]), "matches": obj_hash(match_ids),
              "l5_frozen": _state_hash(frozen), "members": obj_hash(sorted(members.player_id))}
    return ReplicateContext(
        final_nodes=final_nodes, match_ids=tuple(match_ids), splits=sp, xg_spec=up["l1"]["spec"],
        value_spec=l2["selection"]["spec"],
        test={k: te[k] for k in ("nodes", "toi", "stints", "events")},
        positions=stage_l3.player_positions(l0["players"], pd.concat([l0["train"]["roster"],
                                                                      te["roster"]])),
        member_ids=tuple(members.player_id.astype(str)), team_comp=team_comp, khl=khl,
        variant=c["turnover_variant"], alpha=float(c["alpha"]), frozen=frozen, fingerprint=ctx_fp)


def _context_key(cfg: Config, cache: Cache) -> str:
    return f"{cache.root}|{cache.data_hash}|{cfg.section_hash(*REPLICATE_SECTIONS)}"


def context(cfg: Config, cache: Cache) -> ReplicateContext:
    """Контекст процесса: загружается один раз (в процессе пула — инициализатором)."""
    key = _context_key(cfg, cache)
    if key not in _CONTEXTS:
        _CONTEXTS[key] = load_context(cfg, cache)
    return _CONTEXTS[key]


def release_context(cfg: Config, cache: Cache) -> None:
    """Освободить копию контекста в этом процессе (родителю пула spawn нужен только список матчей)."""
    _CONTEXTS.pop(_context_key(cfg, cache), None)


def bootstrap_match_ids(cfg: Config, cache: Cache) -> list[str]:
    """Список матчей, из которого ресэмплирует бутстрэп (final_fit_match_ids)."""
    return list(context(cfg, cache).match_ids)


def replicate_key(cache: Cache, ctx: ReplicateContext, matches_sample, seed: int, index: int) -> str:
    """Ключ кэша реплики: рецепты и входы, выборка матчей (как мультимножество), seed, индекс, код."""
    extra = obj_hash({"context": ctx.fingerprint, "matches": obj_hash(sorted(map(str, matches_sample))),
                      "seed": int(seed), "index": int(index)})
    return cache.key("l6_replicate", REPLICATE_SECTIONS, extra=extra, code=replicate_code())


def _cached(cache: Cache, key: str, run_token: str | None = None) -> pd.DataFrame | None:
    """Готовая реплика; при --force (run_token) — только посчитанная в этом же прогоне."""
    if not cache.done(key):
        return None
    if run_token is not None and cache.load_obj(key, RUN_TOKEN) != run_token:
        return None
    df = cache.load_df(key, REPLICATE_TABLE)
    if df is None or list(df.columns) != list(REPLICATE_COLUMNS):
        cache.invalidate_done(key)
        return None
    return df.assign(player_id=df.player_id.astype(str))


def replicate(cfg: Config, cache: Cache, matches_sample, seed: int, index: int,
              run_token: str | None = None) -> pd.DataFrame:
    """Реплика m с кэшем: DataFrame(player_id, overall, var_post) по игрокам теста.

    run_token задаётся при --force: реплики прежних прогонов пересчитываются, а посчитанные в
    этом прогоне (например, до гибели процесса пула) переиспользуются при повторной отправке.
    """
    ctx = context(cfg, cache)
    key = replicate_key(cache, ctx, matches_sample, seed, index)
    saved = _cached(cache, key, run_token)
    if saved is not None:
        return saved
    cache.invalidate_done(key)
    out = compute_replicate(cfg, ctx, matches_sample, seed)
    cache.save_df(key, REPLICATE_TABLE, out)
    if run_token is not None:
        cache.save_obj(key, RUN_TOKEN, run_token)
    cache.mark_done(key)
    return out


def _replicate_call(cfg: Config, cache: Cache, run_token: str | None, matches, index: int,
                    seed: int) -> pd.DataFrame:
    return replicate(cfg, cache, list(matches), seed, index, run_token)


def make_replicate(cfg: Config, cache: Cache, force: bool = False) -> Callable:
    """Функция реплики для run_model_bootstrap: (matches, index, seed) -> DataFrame (picklable).

    Контекст здесь не загружается: последовательный счёт загрузит его при первой реплике, а
    процессы пула spawn — в инициализаторе. force (ratings --force/--force-all) пересчитывает
    кэшированные реплики прежних прогонов (метка прогона — случайный токен).
    """
    return partial(_replicate_call, cfg, cache, uuid.uuid4().hex if force else None)


def replicate_workers(cfg: Config) -> int:
    """Число процессов пула (bootstrap.replicate_workers, по умолчанию DEFAULT_REPLICATE_WORKERS)."""
    raw = (cfg.get("bootstrap") or {}).get("replicate_workers")
    n = DEFAULT_REPLICATE_WORKERS if raw is None else int(raw)
    if n < 1:
        raise ValueError(f"bootstrap.replicate_workers должно быть ≥ 1, получено {raw!r}")
    return n


def available_cpus() -> int:
    """Ядра, доступные процессу (с учётом affinity)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:                       # нет sched_getaffinity (не Linux)
        return max(1, os.cpu_count() or 1)


def worker_threads(workers: int, cpu: int | None = None) -> int:
    """Потоков BLAS/OpenMP на процесс: процессы × потоки ≤ ядер (не меньше одного потока)."""
    total = available_cpus() if cpu is None else int(cpu)
    return max(1, total // max(1, int(workers)))


def _limit_threadpools(threads: int) -> None:
    """Ограничить уже загруженные библиотеки BLAS/OpenMP (переменные окружения действуют при загрузке)."""
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=int(threads))


@contextmanager
def _context_load_lock(cache: Cache) -> Iterator[None]:
    """Исключительная блокировка загрузки контекста между процессами пула (ждёт, не отказывает)."""
    path = Path(cache.root) / CONTEXT_LOAD_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# B4: файл, куда инициализатор воркера пишет исходное исключение (родитель видит только
# BrokenProcessPool и иначе принял бы детерминированную ошибку за OOM)
INIT_ERROR_FILE = "_l6_worker_init_error.txt"


def _init_error_path(cache) -> Path | None:
    root = getattr(cache, "root", None)
    return None if root is None else Path(root) / INIT_ERROR_FILE


def clear_init_error(cache) -> None:
    p = _init_error_path(cache)
    if p is not None and p.exists():
        p.unlink()


def read_init_error(cache) -> str:
    p = _init_error_path(cache)
    return p.read_text(encoding="utf-8").strip() if p is not None and p.exists() else ""


def _init_worker(threads: int, cfg: Config | None = None, cache: Cache | None = None,
                 loader: Callable | None = None) -> None:
    """Инициализатор процесса пула: один раз загрузить контекст из кэша, затем лимит потоков.

    Порядок важен: переменные потоков входят в отпечаток окружения provenance L1/L2. Выставленные до
    загрузки, они делали принятые L1/L2 «устаревшими» в воркере, и пул ломался (проверка 06.10.2026).
    """
    if loader is not None:
        try:
            with _context_load_lock(cache):
                _CONTEXTS[_context_key(cfg, cache)] = loader(cfg, cache)
        except BaseException as ex:
            p = _init_error_path(cache)
            if p is not None:
                p.write_text(f"{type(ex).__name__}: {ex}\n{traceback.format_exc()}", encoding="utf-8")
            raise
    env = {var: str(int(threads)) for var in THREAD_ENV_VARS}
    os.environ.update(env | {MODEL_THREADS_ENV: str(int(threads))})
    _limit_threadpools(int(threads))


def process_pool(workers: int, cfg: Config | None = None, cache: Cache | None = None,
                 loader: Callable | None = None) -> Callable[[int], Executor]:
    """Фабрика пула процессов spawn; loader(cfg, cache) загружает контекст в каждом процессе."""
    def factory(width: int) -> Executor:
        n = max(1, min(int(width), int(workers)))
        init = partial(_init_worker, worker_threads(n), cfg, cache, loader)
        return ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context(START_METHOD),
                                   initializer=init)
    return factory


def executor_factory(cfg: Config, cache: Cache | None = None) -> Callable[[int], Executor] | None:
    """Пул для реплик по конфигурации; при одном процессе — последовательный счёт в родителе."""
    n = replicate_workers(cfg)
    if n <= 1:
        return None
    clear_init_error(cache)
    return process_pool(n, cfg, cache, load_context if cache is not None else None)
