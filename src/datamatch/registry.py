"""Реестр кандидатов: конфигурация, метрики, бюджет, статус и причина решения.

Статусы: not_run (не запускался), error (ошибка расчёта), rejected (отклонён), accepted (принят),
baseline (ориентир), info (диагностика без решения).
"""
from __future__ import annotations

import json
import time

from .cache import atomic_write
from .config import Config, get_logger

STATUSES = {"not_run", "error", "rejected", "accepted", "baseline", "info"}


class Registry:
    def __init__(self, cfg: Config):
        self.f = cfg.paths.artifacts / "model_registry.json"
        self.rows = self._read()

    def _read(self) -> dict:
        """Строки реестра; непригодный файл отводится в сторону, а не блокирует прогон.

        Оборванная запись оставляет обрезанный JSON. Без этой обработки любой следующий
        запуск слоя падал бы в конструкторе до начала полезной работы.
        """
        if not self.f.exists():
            return {}
        try:
            rows = json.loads(self.f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as ex:
            self._set_aside(f"{type(ex).__name__}: {ex}")
            return {}
        if not isinstance(rows, dict):
            self._set_aside(f"документ не является объектом ({type(rows).__name__})")
            return {}
        return rows

    def _set_aside(self, reason: str) -> None:
        """Сохраняет непригодный реестр под свободным именем: улика не удаляется молча.

        Имя выбирается свободное, иначе второе повреждение затёрло бы следы первого.
        Перенос может не удаться (гонка с соседним процессом, права, read-only ФС) — это
        не повод валить запуск, ради которого и сделана вся обработка.
        """
        stem = self.f.with_suffix("").name
        n = 0
        broken = self.f.with_name(f"{stem}.corrupt.json")
        while broken.exists():
            n += 1
            broken = self.f.with_name(f"{stem}.corrupt.{n}.json")
        try:
            self.f.replace(broken)
        except OSError as ex:
            get_logger().warning("реестр %s непригоден (%s); отвести в сторону не удалось (%s), "
                                 "начат заново", self.f.name, reason, ex)
            return
        get_logger().warning("реестр %s непригоден (%s); сохранён как %s, начат заново",
                             self.f.name, reason, broken.name)

    def add(self, cid: str, layer: str, level: str, against: str | None, config: dict, metrics: dict,
            budget: dict, status: str, reason: str, data_part: str = "inner_cv"):
        assert status in STATUSES, status
        self.rows[cid] = {"candidate": cid, "layer": layer, "level": level, "against": against,
                          "config": config, "metrics": metrics, "budget": budget, "status": status,
                          "reason": reason, "data_part": data_part, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.save()

    def clear_layer(self, layer: str):
        """Перед повторным запуском слоя удаляются его прежние строки (без смешивания запусков)."""
        self.rows = {k: v for k, v in self.rows.items() if v.get("layer") != layer}
        self.save()

    def get(self, cid):
        return self.rows.get(cid)

    def save(self):
        text = json.dumps(self.rows, ensure_ascii=False, indent=1, default=float)
        atomic_write(self.f, lambda t: t.write_text(text, encoding="utf-8"))
