"""Устойчивость к прерванному прогону: битый артефакт не блокирует запуск и не теряет данные.

Прогон может оборваться между открытием файла и концом записи (Ctrl+C, OOM, kill). После
этого на диске остаётся обрезанный файл. Требование: такой файл трактуется как отсутствующий
кэш (пересчёт), а не как ошибка на каждом следующем запуске; а прерванная запись не
уничтожает ранее сохранённое значение.
"""
import pickle
from pathlib import Path

import pandas as pd
import pytest

from datamatch.cache import Cache, ConcurrentRunError, atomic_write, load_checked_json
from datamatch.config import load_config
from datamatch.registry import Registry


def _cfg(tmp_path):
    return load_config().with_overrides({"paths": {"artifacts": str(tmp_path / "artifacts"),
                                                   "cache": str(tmp_path / "cache")}})


class _FailsMidPickle:
    """Объект, который валит сериализацию уже после записи предшествующих байт."""

    def __reduce__(self):
        raise RuntimeError("запись прервана")


def test_truncated_object_cache_is_a_miss_not_a_crash(tmp_path):
    # Arrange: сохранённый объект обрезан на середине, как после kill
    cache = Cache(_cfg(tmp_path))
    cache.save_obj("k1", "model", {"a": 1})
    f = cache.root / "k1" / "model.pkl"
    f.write_bytes(f.read_bytes()[:3])

    # Act
    got = cache.load_obj("k1", "model")

    # Assert: промах кэша, а не исключение на каждом следующем запуске
    assert got is None


def test_object_write_interrupted_midstream_keeps_previous_value(tmp_path):
    # Arrange: в кэше лежит пригодное значение
    cache = Cache(_cfg(tmp_path))
    cache.save_obj("k2", "model", {"generation": "first"})

    # Act: следующая запись обрывается после частичного вывода
    with pytest.raises(RuntimeError, match="запись прервана"):
        cache.save_obj("k2", "model", {"payload": "x" * 100_000, "boom": _FailsMidPickle()})

    # Assert: прежнее значение уцелело
    assert cache.load_obj("k2", "model") == {"generation": "first"}


def test_registry_with_truncated_file_does_not_block_the_run(tmp_path):
    # Arrange: прерванная запись реестра оставила обрезанный JSON
    cfg = _cfg(tmp_path)
    (cfg.paths.artifacts / "model_registry.json").write_text('{"lgbm": {"candidate": "lgb')

    # Act / Assert: слой запускается, а не падает до начала полезной работы
    reg = Registry(cfg)
    reg.add("ebm", "L1", "A", None, {}, {}, {}, "accepted", "после восстановления реестра")
    assert reg.get("ebm")["status"] == "accepted"


def test_atomic_write_keeps_previous_content_when_the_write_fails(tmp_path):
    """Прямая проверка атомарности: запись валится, записав часть данных в целевой путь."""
    # Arrange
    target = tmp_path / "artifact.json"
    target.write_text("прежнее содержимое", encoding="utf-8")

    def half_written(t):
        t.write_bytes(b"x" * 4096)
        raise RuntimeError("запись прервана")

    # Act
    with pytest.raises(RuntimeError, match="запись прервана"):
        atomic_write(target, half_written)

    # Assert: цель нетронута, временный файл убран
    assert target.read_text(encoding="utf-8") == "прежнее содержимое"
    assert list(tmp_path.glob(".*tmp")) == []


def test_registry_recovers_when_the_corrupt_file_cannot_be_moved_aside(tmp_path, monkeypatch):
    """Гонка двух процессов: файл уже отведён соседом — это не повод валить запуск."""
    # Arrange
    cfg = _cfg(tmp_path)
    (cfg.paths.artifacts / "model_registry.json").write_text('{"lgbm": {"candi', encoding="utf-8")

    def vanished(self, _target):
        raise FileNotFoundError("файл уже отведён другим процессом")

    monkeypatch.setattr(Path, "replace", vanished)

    # Act / Assert: конструктор не падает
    assert Registry(cfg).rows == {}


def test_repeated_corruption_does_not_destroy_the_earlier_evidence(tmp_path):
    # Arrange / Act: два повреждения подряд
    cfg = _cfg(tmp_path)
    f = cfg.paths.artifacts / "model_registry.json"
    f.write_text('{"first": {"candi', encoding="utf-8")
    Registry(cfg)
    f.write_text('{"second": {"candi', encoding="utf-8")
    Registry(cfg)

    # Assert: сохранены обе улики, а не одна поверх другой
    evidence = sorted(p.read_text(encoding="utf-8") for p in cfg.paths.artifacts.glob("*corrupt*"))
    assert evidence == ['{"first": {"candi', '{"second": {"candi']


def test_registry_sets_aside_a_document_that_is_not_an_object(tmp_path):
    # Arrange: валидный JSON, но не словарь
    cfg = _cfg(tmp_path)
    (cfg.paths.artifacts / "model_registry.json").write_text("[1, 2, 3]", encoding="utf-8")

    # Act
    reg = Registry(cfg)

    # Assert: начали заново, но содержимое не потеряно молча
    assert reg.rows == {}
    assert [p.read_text(encoding="utf-8") for p in cfg.paths.artifacts.glob("*corrupt*")] == ["[1, 2, 3]"]


def test_empty_object_cache_file_is_a_miss_not_a_crash(tmp_path):
    # Arrange: нулевой размер — запись не успела начаться
    cache = Cache(_cfg(tmp_path))
    cache.save_obj("k4", "model", {"a": 1})
    (cache.root / "k4" / "model.pkl").write_bytes(b"")

    # Act / Assert
    assert cache.load_obj("k4", "model") is None


def test_truncated_table_cache_is_a_miss_not_a_crash(tmp_path):
    # Arrange: parquet обрезан так же, как pickle
    cache = Cache(_cfg(tmp_path))
    cache.save_df("k5", "values", pd.DataFrame({"node": [1, 2], "value": [0.1, 0.2]}))
    f = cache.root / "k5" / "values.parquet"
    f.write_bytes(f.read_bytes()[:40])

    # Act / Assert: тот же договор, что и для объектов
    assert cache.load_df("k5", "values") is None


def test_checked_json_rejects_a_document_that_is_not_an_object(tmp_path):
    # Arrange: валидный JSON, но не словарь — чужая или повреждённая запись
    f = tmp_path / "xg_selected.json"
    f.write_text("[1, 2, 3]")

    # Act / Assert: промах, а не AttributeError внутри загрузчика
    assert load_checked_json(f, {"stage": "l1"}) is None


def test_a_second_run_on_the_same_cache_is_refused_instead_of_interleaving(tmp_path):
    """Кэш, артефакты и реестр — разделяемое состояние без версий и без блокировок.

    Два прогона с одним ключом писали файлы вперемешку, и результат проходил сверку
    provenance, потому что она сверяет json, а не parquet.
    """
    # Arrange
    cache = Cache(_cfg(tmp_path))

    # Act / Assert: второй захват отказывает сразу, а не ждёт
    with cache.run_lock():
        with pytest.raises(ConcurrentRunError, match="уже занят"):
            with cache.run_lock():
                pass

    # и после освобождения доступ снова открыт
    with cache.run_lock():
        pass


def test_successful_writes_leave_no_temporary_files_behind(tmp_path):
    # Arrange
    cache = Cache(_cfg(tmp_path))

    # Act
    cache.save_obj("k3", "model", {"a": 1})
    cache.mark_done("k3")

    # Assert: в каталоге кэша только конечные файлы
    leftovers = [p.name for p in (cache.root / "k3").iterdir()
                 if p.name not in {"model.pkl", "_DONE"}]
    assert leftovers == []
    assert pickle.loads((cache.root / "k3" / "model.pkl").read_bytes()) == {"a": 1}
