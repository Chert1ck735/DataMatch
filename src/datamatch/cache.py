"""Файловый кэш промежуточных результатов с проверкой происхождения (provenance).

Ключ этапа включает: имя этапа, хэш относящихся к нему секций конфигурации, хэш версии
данных, хэш исходного кода модулей этапа и (при необходимости) хэш входов — разбиения,
содержимого входной таблицы и т. п. Поэтому изменение признаков, фолдов, кода или данных
даёт новый ключ, а не подмену расчёта старым результатом. Сохранённые артефакты
(json вне кэша) несут словарь provenance и перед использованием сверяются с текущим.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pickle
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow

from .config import PROJECT_ROOT, Config, get_logger

SRC_DIR = PROJECT_ROOT / "src" / "datamatch"

# Повреждения, означающие «файла фактически нет» (оборванная запись), а не ошибку расчёта.
# Смена версии класса даёт AttributeError и ModuleNotFoundError — они сюда намеренно не входят:
# это рассогласование кода, и молчаливый пересчёт его бы замаскировал.
TRUNCATED_OBJECT = (EOFError, pickle.UnpicklingError)
TRUNCATED_TABLE = (pyarrow.ArrowInvalid, OSError)


class ConcurrentRunError(RuntimeError):
    """Другой прогон уже держит этот кэш: одновременная запись смешала бы генерации."""


def atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    """Записывает файл через временный в том же каталоге и заменяет цель одной операцией.

    Прогон может оборваться посреди записи (Ctrl+C, OOM, kill). Прямая запись в целевой
    файл в этот момент оставила бы обрезанный артефакт вместо прежнего пригодного.
    `os.replace` атомарен в пределах файловой системы, поэтому наблюдаемы только два
    состояния: прежнее содержимое целиком или новое целиком.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def code_hash(*modules: str) -> str:
    """Хэш исходного текста модулей пакета (например, 'l0_events', 'l1_xg')."""
    h = hashlib.sha1()
    for m in sorted(set(modules)):
        f = SRC_DIR / f"{m}.py"
        h.update(m.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def frame_hash(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    """Хэш содержимого таблицы (значения, порядок строк и состав столбцов)."""
    d = df if cols is None else df[cols]
    h = hashlib.sha1()
    h.update("|".join(map(str, d.columns)).encode())
    h.update(pd.util.hash_pandas_object(d, index=False).to_numpy().tobytes())
    return h.hexdigest()[:12]


def obj_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


class Cache:
    def __init__(self, cfg: Config, data_version: dict | None = None):
        self.cfg = cfg
        self.root = cfg.paths.cache
        dv = json.dumps(data_version or {}, sort_keys=True)
        self.data_hash = hashlib.sha1(dv.encode()).hexdigest()[:8]

    def key(self, stage: str, sections: tuple[str, ...] = (), extra: str = "", code: tuple[str, ...] = ()) -> str:
        h = self.cfg.section_hash(*sections) if sections else "nocfg"
        ex = hashlib.sha1(extra.encode()).hexdigest()[:8] if extra else "x"
        ch = code_hash(*code) if code else "nocode"
        return f"{stage}__{h}__{self.data_hash}__{ch}__{ex}"

    def provenance(self, stage: str, sections: tuple[str, ...] = (), code: tuple[str, ...] = (), **inputs) -> dict:
        """Словарь происхождения артефакта; сравнивается целиком при повторном использовании."""
        p = {"stage": stage, "data": self.data_hash,
             "config": self.cfg.section_hash(*sections) if sections else "nocfg",
             "code": code_hash(*code) if code else "nocode"}
        p.update({k: str(v) for k, v in inputs.items()})
        return p

    def path(self, key: str, suffix: str) -> Path:
        return self.root / f"{key}{suffix}"

    def save_df(self, key: str, name: str, df: pd.DataFrame):
        atomic_write(self.root / key / f"{name}.parquet", lambda t: df.to_parquet(t, index=False))

    def load_df(self, key: str, name: str) -> pd.DataFrame | None:
        """Таблица из кэша; нечитаемый файл — промах кэша, тот же договор, что и для объектов."""
        f = self.root / key / f"{name}.parquet"
        if not f.exists():
            return None
        try:
            return pd.read_parquet(f)
        except TRUNCATED_TABLE as ex:
            get_logger().warning("кэш %s повреждён (%s: %s) — пересчёт", f, type(ex).__name__, ex)
            return None

    def save_obj(self, key: str, name: str, obj):
        def dump(t: Path):
            with open(t, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        atomic_write(self.root / key / f"{name}.pkl", dump)

    def load_obj(self, key: str, name: str):
        """Объект из кэша; обрезанный файл — промах кэша (пересчёт), а не отказ прогона."""
        f = self.root / key / f"{name}.pkl"
        if not f.exists():
            return None
        try:
            with open(f, "rb") as fh:
                return pickle.load(fh)
        except TRUNCATED_OBJECT as ex:
            get_logger().warning("кэш %s повреждён (%s: %s) — пересчёт", f, type(ex).__name__, ex)
            return None

    @contextmanager
    def run_lock(self):
        """Исключительный доступ к кэшу на время команды.

        Артефакты, кэш и реестр — разделяемое изменяемое состояние без версий. Два прогона
        с одним ключом записывали файлы вперемешку, и результат проходил сверку provenance,
        потому что она сверяет json, а не parquet. В истории проекта был такой случай: второй
        `value --force` того же проекта пришлось останавливать вручную. Отказ быстрый и
        явный: ждать конца чужого прогона (до полутора часов на L1) бессмысленно.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        lock_file = self.root / "_run.lock"
        with open(lock_file, "w", encoding="utf-8") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as ex:
                raise ConcurrentRunError(
                    f"кэш {self.root} уже занят другим прогоном ({lock_file}); "
                    "дождитесь его завершения или используйте отдельный paths.cache") from ex
            fh.write(str(os.getpid()))
            fh.flush()
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def done(self, key: str) -> bool:
        return (self.root / key / "_DONE").exists()

    def invalidate_done(self, key: str):
        """A new generation cannot inherit an earlier successful completion marker."""
        (self.root / key / "_DONE").unlink(missing_ok=True)

    def mark_done(self, key: str):
        atomic_write(self.root / key / "_DONE", lambda t: t.write_text("ok", encoding="utf-8"))
        get_logger().info("кэш сохранён: %s", key)


def load_checked_json(path: Path, provenance: dict) -> dict | None:
    """Загружает json-артефакт только если его provenance совпадает с текущим; иначе None.

    Повреждённый, нечитаемый или не-объектный документ — промах (пересчёт): такой файл
    остаётся после оборванной записи и не должен блокировать прогон.
    """
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as ex:
        get_logger().warning("артефакт %s нечитаем (%s: %s) — пересчёт", path.name,
                             type(ex).__name__, ex)
        return None
    if not isinstance(d, dict):
        get_logger().warning("артефакт %s не является объектом (%s) — пересчёт", path.name,
                             type(d).__name__)
        return None
    saved = d.get("provenance")
    if saved != provenance:
        get_logger().info("артефакт %s устарел (provenance не совпадает) — пересчёт", path.name)
        return None
    return d


def save_json_with_provenance(path: Path, obj: dict, provenance: dict):
    obj = dict(obj)
    obj["provenance"] = provenance
    text = json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default)
    atomic_write(path, lambda t: t.write_text(text, encoding="utf-8"))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
