"""Правило отображения меток поверх готового результата L6: TOI < 60 мин → LOW «мало минут».

Запуск из Solve после `datamatch.cli ratings --mode fast|full`:

    PYTHONPATH=src .venv/bin/python scripts/apply_display_rules.py --mode full

Рейтинг и интервал не меняются; исходная метка модели остаётся в *_detail.csv. Шаг идемпотентен и
не входит в ключи кэша L6 и реплик бутстрэпа (см. `datamatch.l6_display`).
"""
from __future__ import annotations

import argparse
import sys

from datamatch import l6_display, stage_l6
from datamatch.config import load_config, setup_logging


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=sorted(stage_l6.MODE_STEMS), default="full")
    ap.add_argument("--check", action="store_true", help="только проверить, что правило применено (код 2 — нет)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    log = setup_logging(cfg, "display_rules")
    paths = stage_l6.output_paths(cfg, args.mode)
    if not paths.csv.exists():
        log.error("нет %s: сначала `datamatch.cli ratings --mode %s`", paths.csv, args.mode)
        return 2
    if args.check:
        st = l6_display.display_status(paths.csv, paths.meta)
        (log.info if st["applied"] else log.error)("правило отображения %s: %s", "применено" if st["applied"]
                                                   else "НЕ применено", st["reason"] or "файл совпадает с meta")
        return 0 if st["applied"] else 2
    try:
        info = l6_display.apply_to_files(paths.csv, paths.detail, paths.meta)
    except l6_display.DisplayError as ex:
        log.error("%s", ex)
        return 2
    log.info("правило отображения: %s; понижено до LOW %d; метки %s", info["rule"], info["n_demoted"],
             info["counts_display"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
