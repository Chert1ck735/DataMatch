"""Командная строка: python -m datamatch.cli <команда> [--config configs/default.yaml] [--force]

Команды:
  prepare   L0 (actions, possessions, stints, player_toi), разбиения, outputs/data_quality.json
  xg        L1: выбор процедуры xG во внутренней валидации
  value     L2: выбор процедуры V, независимые прогнозы и TD
  contrib   L3: журнал вкладов δ и таблица «игрок × матч»
  shrink    L5: стягивание по амплуа, итог O1 и страховочный O0
  ratings   L6: интервалы и надёжность; --mode full -> outputs/test_ratings.csv (+ meta, detail),
            --mode fast -> outputs/test_ratings_fast.csv (отладка, не сдаётся)

prepare/xg/value достраивают предыдущие слои из кэша; contrib/shrink/ratings только читают готовые
входы и при их отсутствии завершаются с кодом 4 (порядок — scripts/run_tier_a.sh).
Коды выхода: 0 — успех, 2 — провален гейт качества, 3 — параллельный запуск, 4 — вход не готов.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import io, splits
from .cache import Cache, ConcurrentRunError
from .config import load_config, setup_logging
from .gates import GateError
from .registry import Registry


def cmd_prepare(cfg, cache, force):
    from .stage_l0 import run_l0
    l0 = run_l0(cfg, cache, force=force)
    sch = l0["train"]["nodes"].groupby("match_id").scheme.first()
    prov = splits.splits_provenance(cache, l0["train"]["games"], l0["test"]["games"], sch)
    sp = None if force else splits.load_splits(cfg, prov)
    if sp is None:
        sp = splits.make_splits(cfg, l0["train"]["games"], l0["test"]["games"], sch)
        splits.save_splits(cfg, sp, prov)
        sp["provenance"] = prov
    q = dict(l0["quality"])
    # ``run_l0`` explicitly marks an admitted incompatible scheme as ``stress``.
    # Keep that provenance in the public quality report: a successful stress run is
    # usable for its declared experiment, but it must not look like an ordinary
    # production-quality L0 result.
    q["status"] = l0["quality"].get("status", "ok")
    q["splits"] = {k: v for k, v in sp.items() if k != "matches"}
    q["data_version"] = io.data_version(cfg)
    (cfg.paths.outputs / "data_quality.json").write_text(json.dumps(q, ensure_ascii=False, indent=1, default=str))
    return l0, sp


def cmd_xg(cfg, cache, force, force_all=False):
    from .stage_l1 import run_l1
    l0, sp = cmd_prepare(cfg, cache, force_all)
    return run_l1(cfg, l0, sp, Registry(cfg), cache, force=force or force_all)


def cmd_value(cfg, cache, force, force_all=False):
    from .stage_l2 import run_l2
    l0, sp = cmd_prepare(cfg, cache, force_all)
    from .stage_l1 import run_l1
    reg = Registry(cfg)
    selection = run_l1(cfg, l0, sp, reg, cache, force=force_all)
    # V needs only node graphs; release large raw-event/shift tables before cross-fits.
    l0 = {'quality': l0['quality'], 'train': {'nodes': l0['train']['nodes']},
          'test': {'nodes': l0['test']['nodes']}}
    return run_l2(cfg, l0, sp, selection, reg, cache, force=force or force_all)


def cmd_contrib(cfg, cache, force, force_all=False):
    """L3. Дочитывает L0–L2 из кэша сам; --force-all пересчитывает их заново."""
    from .stage_l3 import run_l3
    if force_all:
        cmd_value(cfg, cache, True, True)
    return run_l3(cfg, cache, force=force or force_all)


def cmd_shrink(cfg, cache, force, force_all=False):
    from .stage_l5 import run_l5
    l3 = cmd_contrib(cfg, cache, force_all, force_all)
    return run_l5(cfg, cache, l3, force=force or force_all)


def cmd_ratings(cfg, cache, force, force_all=False, mode="fast"):
    from .stage_l6 import run_l6
    l5 = cmd_shrink(cfg, cache, force_all, force_all)
    return run_l6(cfg, cache, l5, mode=mode, force=force or force_all)


COMMANDS = {"prepare": cmd_prepare, "xg": cmd_xg, "value": cmd_value,
            "contrib": cmd_contrib, "shrink": cmd_shrink, "ratings": cmd_ratings}


EXIT_GATE_FAILED = 2
EXIT_CONCURRENT_RUN = 3
EXIT_UPSTREAM_NOT_READY = 4


def _not_ready_errors() -> tuple:
    """Исключения «вход слоя ещё не посчитан»: сообщение и код 4 вместо трассировки."""
    from .l6_replicate import ReplicateUnavailable
    from .stage_l3 import UpstreamNotReady
    from .stage_l5 import FrozenStateMissing
    from .stage_l6 import L6InputsNotReady
    return (UpstreamNotReady, FrozenStateMissing, L6InputsNotReady, ReplicateUnavailable)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="datamatch")
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="пересчитать запрошенный этап (включая его внутренний кэш)")
    ap.add_argument("--force-all", action="store_true", help="пересчитать запрошенный этап и все предшествующие")
    ap.add_argument("--mode", choices=("fast", "full"), default="fast",
                    help="ratings: fast — B из bootstrap.debug_B; full — B из bootstrap.final_B (сдаваемый файл)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    log = setup_logging(cfg, a.command)
    cache = Cache(cfg, io.data_version(cfg))
    t0 = time.time()
    try:
        with cache.run_lock():
            if a.command == "prepare":
                COMMANDS[a.command](cfg, cache, a.force or a.force_all)
            elif a.command == "ratings":
                COMMANDS[a.command](cfg, cache, a.force, a.force_all, mode=a.mode)
            else:
                COMMANDS[a.command](cfg, cache, a.force, a.force_all)
    except ConcurrentRunError as ex:
        log.error("%s", ex)
        return EXIT_CONCURRENT_RUN
    except GateError as ex:
        log.error("остановлено гейтом: %s", ex)
        return EXIT_GATE_FAILED
    except _not_ready_errors() as ex:
        log.error("вход не готов: %s", ex)
        return EXIT_UPSTREAM_NOT_READY
    log.info("команда %s завершена за %.1f с", a.command, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
