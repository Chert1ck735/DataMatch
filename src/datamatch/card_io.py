"""Карточка игрока: входы из готовых кэшей L3/L5, привязанные к сдаваемому файлу.

player_match читается прямо из готового кэша L3 (без загрузки L0–L2) и принимается, только если его
хэш совпадает с provenance сдаваемого расчёта L5. L5 читается по указателю последнего расчёта (`l5__current`), а не по ключу текущего кода: карточка
описывает тот же результат, что и `outputs/test_ratings.csv`, даже если код L5 в рабочем дереве уже
меняется. Привязка проверяется: O1 test из кэша обязан совпасть с `test_ratings_detail.csv`, иначе
`CardInputMismatch` (карточка к другому рейтингу не строится). Ничего не пересчитывается.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cache import frame_hash
from .stage_l5 import L5_POINTER_KEY, L5_POINTER_NAME, PM_O0_REQUIRED, PM_REQUIRED

# CONFIG: допуск совпадения рейтинга кэша L5 и сдаваемого файла (xG/60; файл пишется с float64).
DELIVERED_TOL = 1e-9
L5_TABLES = ("O1_train", "O1_test")
# Колонки player_match, по которым L5 считает хэш входа (provenance train_pm / test_pm).
PM_HASH_COLS = tuple(PM_REQUIRED + PM_O0_REQUIRED)
L3_GLOB = "l3__*"


class CardInputMismatch(RuntimeError):
    """Кэш L5 не соответствует сдаваемому файлу или отсутствует."""


def assert_matches_delivered(o1_test: pd.DataFrame, detail: pd.DataFrame, tol: float = DELIVERED_TOL) -> None:
    """Каждый игрок O1 test есть в файле с тем же рейтингом (игроки только из заявки — лишь в файле)."""
    d = detail.set_index("player_id").rating
    missing = sorted(set(o1_test.player_id) - set(d.index))
    if missing:
        raise CardInputMismatch(f"игроков O1 test нет в test_ratings_detail.csv: {missing[:5]}")
    diff = np.abs(o1_test.rating.to_numpy(dtype=float) - d.reindex(o1_test.player_id).to_numpy(dtype=float))
    bad = o1_test.player_id[diff > tol].tolist()
    if bad:
        raise CardInputMismatch(f"рейтинг кэша L5 расходится со сдаваемым файлом: {bad[:5]}")


def assert_pm_matches_l5(pm_train: pd.DataFrame, pm_test: pd.DataFrame, prov: dict) -> None:
    """player_match train/test — тот самый вход, на котором посчитан L5 (хэш колонок контракта)."""
    cols = list(PM_HASH_COLS)
    got = {"train_pm": frame_hash(pm_train[cols]), "test_pm": frame_hash(pm_test[cols])}
    bad = [k for k, v in got.items() if prov.get(k) != v]
    if bad:
        raise CardInputMismatch(f"player_match не совпадает с входом L5 ({bad})")


def load_pm_bound(cache, prov: dict) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """player_match из готового кэша L3, совпадающий с provenance L5 (без пересчёта и без L0–L2)."""
    for d in sorted(cache.root.glob(L3_GLOB)):
        if not cache.done(d.name):
            continue
        tr, te = cache.load_df(d.name, "train_player_match"), cache.load_df(d.name, "test_player_match")
        if tr is None or te is None:
            continue
        try:
            assert_pm_matches_l5(tr, te, prov)
        except (CardInputMismatch, KeyError):
            continue
        return tr, te, d.name
    raise CardInputMismatch("в кэше нет player_match L3, на котором посчитан сдаваемый L5")


def load_l5_delivered(cache, detail: pd.DataFrame) -> dict:
    """Таблицы O1 и отчёт L5 по указателю последнего расчёта, со сверкой со сдаваемым файлом."""
    ptr = cache.load_obj(L5_POINTER_KEY, L5_POINTER_NAME)
    if not isinstance(ptr, dict) or "key" not in ptr or not cache.done(str(ptr["key"])):
        raise CardInputMismatch("нет готового расчёта L5 (указатель l5__current)")
    key = str(ptr["key"])
    frames = {n: cache.load_df(key, n) for n in L5_TABLES}
    report = cache.load_obj(key, "report")
    if report is None or any(f is None for f in frames.values()):
        raise CardInputMismatch(f"в кэше {key} нет таблиц O1 или отчёта")
    assert_matches_delivered(frames["O1_test"], detail)
    return {"O1": {"train": frames["O1_train"], "test": frames["O1_test"]}, "report": report, "key": key,
            "provenance": ptr.get("provenance", report.get("provenance", {}))}


def load_card_tables(cfg, cache) -> dict:
    """L5 сдаваемого результата и player_match L3, на котором он посчитан (только кэш)."""
    detail = pd.read_csv(cfg.paths.outputs / "test_ratings_detail.csv")
    l5 = load_l5_delivered(cache, detail)
    pm_tr, pm_te, l3_key = load_pm_bound(cache, l5["provenance"])
    return {"pm_train": pm_tr, "pm_test": pm_te, "l3_key": l3_key, "detail": detail, "l5": l5}
