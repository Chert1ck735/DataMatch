"""Провенанс обязан различать расчёты, которые нельзя считать одинаковыми.

Единственное заявленное свойство провенанса — «другие входы дают другой ключ». Здесь
проверяются две дыры, через которые это свойство нарушалось: подмена сырых данных без
изменения размера и времени, и запуск L6 с подменёнными точками расширения.
"""
import json
import os

import pytest

from datamatch import io
from datamatch import stage_l6 as S
from datamatch.cache import Cache
from datamatch.config import load_config
from tests.test_l6_ratings import l5_dict, latent_table, members_frame, replicate_factory


def _data_cfg(tmp_path):
    train, test = tmp_path / "train", tmp_path / "test"
    train.mkdir()
    test.mkdir()
    for name in io.RAW_FILES_TRAIN:
        (train / name).write_bytes(b"A" * 32)
    for name in io.RAW_FILES_TEST:
        (test / name).write_bytes(b"A" * 32)
    return load_config().with_overrides({"paths": {
        "data_dir": str(train), "test_dir": str(test), "cache": str(tmp_path / "cache")}})


def test_data_version_detects_content_change_under_preserved_size_and_mtime(tmp_path):
    """Восстановление данных с сохранением времён (cp -p, rsync -t, распаковка) не должно
    выдавать прежний отпечаток: иначе весь конвейер вернёт результаты на других данных."""
    # Arrange
    cfg = _data_cfg(tmp_path)
    before = io.data_version(cfg)

    # Act: другое содержимое, тот же размер и то же время
    target = cfg.paths.data_dir / io.RAW_FILES_TRAIN[1]
    stat = target.stat()
    target.write_bytes(b"B" * 32)
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    # Assert
    assert io.data_version(cfg) != before


@pytest.fixture
def l6_env(tmp_path):
    cfg = load_config().with_overrides({"paths": {
        "outputs": str(tmp_path / "outputs"), "artifacts": str(tmp_path / "artifacts"),
        "cache": str(tmp_path / "cache"), "logs": str(tmp_path / "logs")}})
    return cfg, Cache(cfg, {})


def _provenance_of(cfg):
    # run_l6 по умолчанию — режим fast: он пишет свой meta, а не сдаваемый test_ratings_meta.json
    return json.loads(S.output_paths(cfg, "fast").meta.read_text())["provenance"]


def test_injected_replicate_is_visible_in_the_published_provenance(l6_env):
    """Точки подмены задают Var_m, то есть границы интервала в сдаваемом файле.

    Пока они не входят в провенанс, файл, собранный с подставленной репликой, неотличим
    от честного и будет переиспользован следующим запуском как актуальный.
    """
    # Arrange
    cfg, cache = l6_env
    table = latent_table()
    members = members_frame(table)

    # Act
    S.run_l6(cfg, cache, l5_dict(table), members=members)
    honest = _provenance_of(cfg)
    S.run_l6(cfg, cache, l5_dict(table), members=members, force=True,
             replicate=replicate_factory(scale=0.05))
    injected = _provenance_of(cfg)

    # Assert
    assert honest != injected
