"""Конфигурация проекта: пути, seed, параметры слоёв.

Пути разрешаются от корня проекта (каталог Solve), а не от текущего каталога.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass
class Paths:
    root: Path
    data_dir: Path
    test_dir: Path
    outputs: Path
    artifacts: Path
    cache: Path
    logs: Path


class Config:
    def __init__(self, raw: dict, source: Path | None = None):
        self.raw = raw
        self.source = source
        p = raw["paths"]
        res = lambda v: (PROJECT_ROOT / v).resolve() if not os.path.isabs(v) else Path(v)
        self.paths = Paths(
            root=PROJECT_ROOT,
            data_dir=res(p["data_dir"]),
            test_dir=res(p["test_dir"]),
            outputs=res(p["outputs"]),
            artifacts=res(p["artifacts"]),
            cache=res(p["cache"]),
            logs=res(p["logs"]),
        )
        for d in (self.paths.outputs, self.paths.artifacts, self.paths.cache, self.paths.logs):
            d.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, k):
        return self.raw[k]

    def get(self, k, default=None):
        return self.raw.get(k, default)

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def section_hash(self, *sections: str) -> str:
        """Хэш выбранных секций конфигурации — часть ключа кэша."""
        sub = {s: self.raw.get(s) for s in sections}
        return hashlib.sha1(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()[:10]

    def with_overrides(self, overrides: dict) -> "Config":
        raw = copy.deepcopy(self.raw)
        _deep_update(raw, overrides)
        return Config(raw, self.source)


def _deep_update(d: dict, u: dict):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v


# Потоки одной модели (LightGBM, EBM). Процесс-реплика L6 выставляет DATAMATCH_MODEL_THREADS,
# чтобы процессы × потоки не превышали число ядер.
DEFAULT_MODEL_THREADS = 4


def model_threads() -> int:
    """Число потоков обучения одной модели: DATAMATCH_MODEL_THREADS или 4, не меньше 1."""
    import os
    raw = os.environ.get("DATAMATCH_MODEL_THREADS")
    if raw is None:
        return DEFAULT_MODEL_THREADS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MODEL_THREADS


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Config(raw, path)
    if overrides:
        cfg = cfg.with_overrides(overrides)
    return cfg


def setup_logging(cfg: Config, name: str) -> logging.Logger:
    logger = logging.getLogger("datamatch")
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith(f"{name}.log") for h in logger.handlers):
        fh = logging.FileHandler(cfg.paths.logs / f"{name}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(fh)
    if not any(type(h) is logging.StreamHandler for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(sh)
    logger.setLevel(logging.INFO)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("datamatch")


def file_fingerprint(path: Path) -> dict[str, Any]:
    st = path.stat()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"file": path.name, "size": st.st_size, "sha256": h.hexdigest()}
