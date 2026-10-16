"""Правило отображения меток поверх результата L6 (решение пользователя 07.10.2026).

Игрок с TOI < 60 мин получает в `test_ratings.csv` метку LOW с пометкой «мало минут»; рейтинг и
интервал не меняются. Исходная метка модели (по w из L5) остаётся в `test_ratings_detail.csv`
(`reliability`), рядом — `reliability_display` и `display_flag`.

Обоснование (METHODOLOGY 7.7): на train у игроков с < 60 мин «прошлого» реализованная надёжность
MEDIUM — 0,40 [0,33; 0,47] при обещанных 0,47, у LOW — 0,21: разделение слабое; 60 мин — тот же
порог участия игрока, что в проверках L7. Решение принято по train, test ничего не настраивает.

Модуль не входит ни в ключ кэша L6 (`stage_l6.L6_CODE`), ни в ключ реплик бутстрэпа
(`l6_replicate.replicate_code`): правка правила не делает бутстрэп устаревшим. Файлы L6
переписываются атомарно; meta получает исходные sha256 L6 и новые sha256 файлов, иначе повторный
`ratings` принял бы изменённый файл за оборванную запись. Шаг идемпотентен.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .stage_l6 import file_sha256

MIN_TOI_MIN = 60.0
LOW_MINUTES_FLAG = "мало минут"
RULE = "toi_min < 60 → LOW «мало минут»; рейтинг и интервал не меняются"


class DisplayError(RuntimeError):
    """Файлы не совпадают с meta: правило к ним не применяется (неизвестное происхождение)."""


def apply_low_minutes_rule(csv: pd.DataFrame, detail: pd.DataFrame,
                           min_toi_min: float = MIN_TOI_MIN) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Новые копии CSV и detail с меткой отображения; исходная метка — `detail.reliability`."""
    for name, df, cols in (("csv", csv, ["player_id", "reliability", "toi_min"]),
                           ("detail", detail, ["player_id", "reliability"])):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{name}: нет колонок {missing}")
    model = detail.set_index("player_id").reliability.reindex(csv.player_id).to_numpy()
    if pd.isna(model).any():
        raise DisplayError("detail не содержит исходной метки для части игроков CSV")
    toi = pd.to_numeric(csv.toi_min, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(toi).all():
        # NaN < 60 даёт False: игрок молча сохранил бы метку модели — это ошибка входа, а не «много минут»
        raise DisplayError(f"toi_min не число у {int((~np.isfinite(toi)).sum())} игроков")
    low = toi < float(min_toi_min)
    display = np.where(low, "LOW", model).astype(object)
    demoted = low & (model != "LOW")
    out_csv = csv.copy()
    out_csv["reliability"] = display
    by_id = pd.Series(display, index=csv.player_id.to_numpy())
    flag = pd.Series(np.where(demoted, LOW_MINUTES_FLAG, ""), index=csv.player_id.to_numpy())
    out_det = detail.copy()
    out_det["reliability_display"] = by_id.reindex(detail.player_id).to_numpy()
    out_det["display_flag"] = flag.reindex(detail.player_id).fillna("").to_numpy()
    info = {"rule": RULE, "min_toi_min": float(min_toi_min), "n_demoted": int(demoted.sum()),
            "counts_model": pd.Series(model).value_counts().to_dict(),
            "counts_display": pd.Series(display).value_counts().to_dict()}
    return out_csv, out_det, info


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def apply_to_files(csv_path: Path, detail_path: Path, meta_path: Path, min_toi_min: float = MIN_TOI_MIN) -> dict:
    """Применяет правило к файлам режима L6 и обновляет meta (sha256 до и после)."""
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    if (meta.get("csv_sha256") != file_sha256(csv_path) or meta.get("detail_sha256") != file_sha256(detail_path)):
        raise DisplayError(f"{Path(csv_path).name}: sha256 не совпадает с meta — файл не от L6 или оборван")
    prev = meta.get("display_rule") or {}
    l6_sha = {"l6_csv_sha256": prev.get("l6_csv_sha256", meta["csv_sha256"]),
              "l6_detail_sha256": prev.get("l6_detail_sha256", meta["detail_sha256"])}
    csv = pd.read_csv(csv_path, dtype=str, keep_default_na=False)       # числа — как записал L6
    detail = pd.read_csv(detail_path, dtype=str, keep_default_na=False)
    out_csv, out_det, info = apply_low_minutes_rule(csv, detail, min_toi_min)
    _atomic_write_csv(out_csv, Path(csv_path))
    _atomic_write_csv(out_det, Path(detail_path))
    meta["display_rule"] = {**info, **l6_sha, "decision": "пользователь, 07.10.2026 (METHODOLOGY 7.7)"}
    meta["display_applied"] = True
    meta.setdefault("reliability", {})["counts_display"] = info["counts_display"]
    meta["csv_sha256"], meta["detail_sha256"] = file_sha256(csv_path), file_sha256(detail_path)
    tmp = Path(meta_path).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, meta_path)
    return meta["display_rule"]


def display_status(csv_path: Path, meta_path: Path) -> dict:
    """Применено ли правило к текущим файлам режима (A5, аудит 08.10.2026).

    L6 переписывает meta целиком (без `display_rule` и `display_applied`), поэтому после любого `ratings`
    без шага правила статус — «не применено»; «применено» — если в meta есть `display_rule` (его пишет
    только этот шаг; флаг `display_applied` — для наглядности, у файлов до 08.10 его нет) и sha256 файла
    совпадает с записанным (файл не переписан после шага).
    """
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    sha_ok = meta.get("csv_sha256") == file_sha256(csv_path)
    applied = bool(meta.get("display_rule")) and sha_ok
    return {"applied": applied, "csv_matches_meta": sha_ok,
            "reason": "" if applied else ("файл не совпадает с meta" if not sha_ok
                                          else "правило отображения не применено после ratings")}
