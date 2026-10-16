"""Карточка игрока, пункт 5: физический профиль Ф1–Ф2 (METHODOLOGY 11.5).

Решение «показатель игрока или нет» — по split-half надёжности (Спирмен–Браун) на матчах разработки
train, порог `phys_profile.RELIABILITY_GATE` (0,3, зафиксирован в ветке Ф). Объединённая выборка
train + test — только описательная оценка, решения не меняет. Модуль только импортирует
`phys_profile` и ничего в нём не меняет.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from datamatch import phys_profile as pp

LEVEL_PLAYER = "игрок"
LEVEL_BY_FALLBACK = {"ROLE": "уровень амплуа", "LINE": "характеристика звена"}


def restrict_to_matches(obs: pd.DataFrame, match_ids: set) -> pd.DataFrame:
    """Наблюдения только из перечисленных матчей (новый кадр, вход не меняется)."""
    return obs.loc[obs["match_id"].isin(match_ids)].reset_index(drop=True)


def card_level(gate_passed: bool, fallback: str) -> str:
    """Уровень, на котором показатель показывается в карточке."""
    if fallback not in LEVEL_BY_FALLBACK:
        raise ValueError(f"card_level: неизвестный запасной уровень {fallback!r}")
    return LEVEL_PLAYER if gate_passed else LEVEL_BY_FALLBACK[fallback]


def _passed(rel: dict) -> bool:
    """Порог 0,3 по точечной SB; должен совпасть с флагом `gate_passed` из `phys_profile.reliability`."""
    sb = rel.get("split_half_sb", float("nan"))
    passed = bool(np.isfinite(sb) and sb >= pp.RELIABILITY_GATE)
    if "gate_passed" in rel and bool(rel["gate_passed"]) != passed:
        raise ValueError(f"порог карточки ({passed}) и phys_profile.gate_passed ({rel['gate_passed']}) расходятся")
    return passed


def decision_table(reliability_dev: dict[str, dict], fallback: dict[str, str]) -> pd.DataFrame:
    """Строка на метрику: SB на разработке, порог, прошёл ли, уровень в карточке."""
    rows = []
    for metric, rel in reliability_dev.items():
        passed = _passed(rel)
        rows.append({"metric": metric, "split_half_sb_dev": float(rel.get("split_half_sb", float("nan"))),
                     "n_players_dev": int(rel.get("n_players", 0)), "gate": pp.RELIABILITY_GATE,
                     "passed": passed, "level": card_level(passed, fallback[metric])})
    return pd.DataFrame(rows)


def pooled_descriptive(obs_a: pd.DataFrame, obs_b: pd.DataFrame, roles: pd.Series,
                       spec: pp.MetricSpec) -> dict:
    """Split-half на объединении двух непересекающихся по матчам выборок — только описание."""
    overlap = set(obs_a["match_id"]) & set(obs_b["match_id"])
    if overlap:
        raise ValueError(f"pooled_descriptive: выборки пересекаются по {len(overlap)} матчам")
    rel = pp.reliability(pd.concat([obs_a, obs_b], ignore_index=True), roles, spec, n_boot=0)
    return {"split_half_sb": float(rel["split_half_sb"]), "r_half": float(rel["r_half"]),
            "n_players": int(rel["n_players"]), "per_role": rel["per_role"], "descriptive_only": True}


def apply_level(csv: pd.DataFrame, col: str, level: str, mu_by_role: dict[str, float]) -> pd.DataFrame:
    """Колонки показателя по уровню показа (новый кадр).

    «игрок» — значения как есть; «характеристика звена» — значения остаются (это показатель звена игрока),
    уровень подписан; «уровень амплуа» — вместо значения игрока μ_r его амплуа, сырое значение и вес пусты.
    """
    out = csv.assign(**{f"{col}_level": level})
    if level != LEVEL_BY_FALLBACK["ROLE"]:
        return out
    return out.assign(**{col: out["role"].map(mu_by_role).astype(float),
                         f"{col}_raw": np.nan, f"{col}_weight": np.nan})
