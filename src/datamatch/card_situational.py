"""Карточка игрока: ситуационные рейтинги 5 на 5, большинство, меньшинство (METHODOLOGY 11.3).

Журнал L3 уже разрезан по ситуациям: компоненты `c_*_5v5/_pp/_pk` и экспозиция `toi_rate_*`.
Здесь ситуация превращается во вход того же вида, что у L5 (`c_*`, `toi_rate_s`), и стягивается классом
`JointEB` без правок; моменты — свои для каждой ситуации. Интервал — апостериорный, без бутстрэпа
моделей ценности. Overall не меняется: ситуационные рейтинги — отдельные поля карточки.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .l5_shrink import reliability

BASE = ("shot", "loss", "recovery", "faceoff", "creation")
COMP_COLS = [f"c_{c}" for c in BASE]
# ситуация → (экспозиция в player_match L3, суффикс компонент)
SITUATIONS = {"5v5": ("toi_rate_5v5_s", "5v5"), "pp": ("toi_rate_pp_s", "pp"), "pk": ("toi_rate_sh_s", "pk")}
CARRY_COLS = ("role", "is_khl_component", "team_id")
LOW_MINUTES = "мало минут"


def situation_frame(pm: pd.DataFrame, sit: str) -> pd.DataFrame:
    """player_match ситуации в контракте L5: `toi_rate_s` и `c_*` — значения этой ситуации."""
    toi_col, suffix = SITUATIONS[sit]
    keep = ["match_id", "player_id"] + [c for c in CARRY_COLS if c in pm.columns]
    out = pm[keep].copy()
    out["toi_rate_s"] = pm[toi_col].to_numpy(dtype=float)
    for c in COMP_COLS:
        out[c] = pm[f"{c}_{suffix}"].to_numpy(dtype=float)
    return out


def split_half_reliability(df: pd.DataFrame, min_toi_s: float) -> dict:
    """Split-half (Спирмен–Браун) значения на 60 внутри амплуа.

    df: player_id, match_id, role, toi_s, value. Матчи игрока по порядку match_id через один; игрок
    входит, если в каждой половине не меньше min_toi_s; половины центрируются средним своего амплуа.
    """
    p = df.sort_values(["player_id", "match_id"]).assign(
        half=lambda d: d.groupby("player_id").cumcount() % 2)
    h = p.groupby(["player_id", "half"]).agg(value=("value", "sum"), toi_s=("toi_s", "sum"),
                                            role=("role", "first")).reset_index()
    a = h[h.half == 0].set_index("player_id")
    b = h[h.half == 1].set_index("player_id")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    ok = (a.toi_s >= min_toi_s) & (b.toi_s >= min_toi_s)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return {"r": float("nan"), "r_sb": float("nan"), "n_players": int(len(a))}
    xa = a.value / a.toi_s * 3600.0
    xb = b.value / b.toi_s * 3600.0
    xa = xa - xa.groupby(a.role).transform("mean")
    xb = xb - xb.groupby(b.role).transform("mean")
    r = float(np.corrcoef(xa, xb)[0, 1])
    return {"r": r, "r_sb": float(2 * r / (1 + r)), "n_players": int(len(a))}


def display_label(w: np.ndarray, toi_min: np.ndarray, high: float, medium: float, min_toi_min: float) -> pd.DataFrame:
    """Метка по w (пороги L5) и правило «мало минут» для TOI ситуации."""
    lab = np.asarray(reliability(np.asarray(w, dtype=float), high, medium), dtype=object)
    few = np.asarray(toi_min, dtype=float) < min_toi_min
    return pd.DataFrame({"label": np.where(few, "LOW", lab), "flag": np.where(few, LOW_MINUTES, "")})


def merge_flags(model_flag: np.ndarray, minutes_flag: np.ndarray) -> np.ndarray:
    """Флаг модели L5 (например, «сигнал не обнаружен») и флаг «мало минут» через «; »."""
    a = np.asarray(model_flag, dtype=object)
    b = np.asarray(minutes_flag, dtype=object)
    if len(a) != len(b):
        raise ValueError(f"merge_flags: длины {len(a)} и {len(b)} не совпадают")
    return np.array(["; ".join(str(f) for f in (x, y) if isinstance(f, str) and f) for x, y in zip(a, b)],
                    dtype=object)


def posterior_sd(var_post: np.ndarray) -> np.ndarray:
    """√var_post; численный шум ниже нуля (L5 допускает до −1e-9) → 0."""
    return np.sqrt(np.clip(np.asarray(var_post, dtype=float), 0.0, None))


def consistency(o1: pd.DataFrame, sit_tables: dict, total_toi_min: pd.Series) -> dict:
    """Σ_s (TOI_s/TOI)·rating_s против O1 (диагностика; прочие составы в сумму не входят)."""
    ids = o1.player_id
    tot = total_toi_min.reindex(ids).to_numpy(dtype=float)
    weighted = np.zeros(len(ids))
    for t in sit_tables.values():
        s = t.set_index("player_id").reindex(ids)
        share = np.divide(s.toi_min.fillna(0.0).to_numpy(dtype=float), tot, out=np.zeros(len(ids)), where=tot > 0)
        weighted += share * s.rating.fillna(0.0).to_numpy(dtype=float)
    diff = weighted - o1.rating.to_numpy(dtype=float)
    return {"weighted": pd.Series(weighted, index=ids.to_numpy()), "mae": float(np.mean(np.abs(diff))),
            "mean_diff": float(np.mean(diff)),
            "corr": float(np.corrcoef(weighted, o1.rating)[0, 1]) if len(ids) > 2 else float("nan")}
