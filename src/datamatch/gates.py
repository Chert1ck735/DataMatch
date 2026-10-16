"""Обязательные проверки (гейты): провал блокирует использование результата.

При провале сохраняется диагностический отчёт со статусом "failed" и поднимается GateError;
следующие этапы принимают артефакт только со статусом "ok" (require_ok).
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import get_logger


class GateError(RuntimeError):
    """Провал обязательной проверки: результат этапа нельзя использовать."""


def failed(gates: dict) -> list[str]:
    return sorted(k for k, v in gates.items() if v is not True)


def enforce(stage: str, gates: dict, report_path: Path | None = None, report: dict | None = None):
    """gates: имя -> bool. При любом False пишет отчёт со статусом failed и поднимает GateError."""
    bad = failed(gates)
    if not bad:
        return
    if report_path is not None:
        rep = dict(report or {})
        rep.update({"status": "failed", "stage": stage, "failed_gates": bad, "gates": gates})
        report_path.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    msg = f"{stage}: не пройдены обязательные проверки {bad}" + (f"; отчёт: {report_path}" if report_path else "")
    get_logger().error(msg)
    raise GateError(msg)


def require_ok(artifact: dict | None, name: str) -> dict:
    """Артефакт используется дальше только при статусе ok."""
    if artifact is None:
        raise GateError(f"{name}: артефакт отсутствует или устарел")
    if artifact.get("status") != "ok":
        raise GateError(f"{name}: статус {artifact.get('status')!r}, использование заблокировано "
                        f"(не пройдены: {artifact.get('failed_gates')})")
    return artifact
