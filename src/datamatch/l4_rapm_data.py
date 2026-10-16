"""L4 RAPM: строки «отрезок × атакующая сторона» из готовых кэшей L0 и L2 (только чтение).

Отрезок допускается, если при нём 5 на 5 (`is_5v5`, по 5 полевых и вратарь у каждой стороны),
он допущен к rate-метрикам (`valid_for_rate_metrics`) и длительность > 0. xG стороны — сумма
`r` бросков её команды в отрезке (`r` — xG броска из L2: на train — match-cross-fit, на test —
итоговая модель; тот же источник, что у on-ice xGF/xGA в L3). Счёт — по первому узлу отрезка
(`score_*_before`); у отрезка без узлов событий нет, поэтому его счёт — счёт перед ближайшим
следующим узлом матча (в любом отрезке, в том числе недопущенном), а после последнего узла —
последний известный. Зона — у первого узла, если это
вбрасывание: `zone` дана относительно команды исполнителя и переводится к атакующим; иначе «fly».
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .l4_rapm import ROW_COLS

STINT_COLS = ["match_id", "stint", "duration", "home_sk", "away_sk", "n_home_sk", "n_away_sk",
              "home_g_in", "away_g_in", "is_5v5", "valid_for_rate_metrics", "home_team_id", "away_team_id"]
NODE_COLS = ["match_id", "stint", "seq", "action", "team_id", "zone", "score_home_before", "score_away_before"]
FLIP = {"off": "def", "def": "off", "neutral": "neutral"}
SCORE_CLIP = 2


def _require(df: pd.DataFrame, cols, what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{what}: нет колонок {missing}")


def eligible_stints(stints: pd.DataFrame) -> pd.DataFrame:
    _require(stints, STINT_COLS, "stints")
    m = (stints.is_5v5.fillna(False).astype(bool) & stints.valid_for_rate_metrics.fillna(False).astype(bool)
         & (stints.n_home_sk == 5) & (stints.n_away_sk == 5) & stints.home_g_in.astype(bool)
         & stints.away_g_in.astype(bool) & (stints.duration.astype(float) > 0))
    return stints[m]


def _stint_context(nodes: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Счёт и зона вбрасывания (с точки зрения хозяев) на начало каждого отрезка `stints`.

    `stints` — все отрезки матчей (не только допущенные): иначе гол в недопущенном отрезке
    терялся бы для следующего отрезка без узлов.
    """
    first = nodes.sort_values(["match_id", "seq"]).drop_duplicates(["match_id", "stint"])
    key = ["match_id", "stint"]
    ctx = stints[key + ["home_team_id"]].merge(first, on=key, how="left", suffixes=("", "_n"))
    ctx = ctx.sort_values(key)
    raw = (ctx.score_home_before - ctx.score_away_before).groupby(ctx.match_id)
    diff = raw.bfill().fillna(raw.ffill()).fillna(0.0)
    is_fo = ctx.action.eq("faceoff").to_numpy()
    by_home = ctx.team_id.eq(ctx.home_team_id).to_numpy()
    zone = np.where(is_fo, np.where(by_home, ctx.zone, ctx.zone.map(FLIP)), "fly")
    return pd.DataFrame({"match_id": ctx.match_id.to_numpy(), "stint": ctx.stint.to_numpy(),
                         "score_home": diff.to_numpy(dtype=float), "zone_home": zone})


def stint_rows(nodes: pd.DataFrame, stints: pd.DataFrame, r: np.ndarray) -> pd.DataFrame:
    """Длинная таблица `l4_rapm.ROW_COLS`: две строки на допущенный отрезок 5 на 5."""
    _require(nodes, NODE_COLS, "nodes")
    if len(r) != len(nodes):
        raise ValueError("r должен быть выровнен с узлами")
    st = eligible_stints(stints)
    key = ["match_id", "stint"]
    shots = nodes.assign(r=np.asarray(r, dtype=float))[nodes.action.eq("shot").to_numpy()]
    xg = shots.groupby(key + ["team_id"]).r.sum()
    xg2 = (shots.assign(r2=shots.r ** 2)).groupby(key + ["team_id"]).r2.sum()   # для пуассоновской σ²
    ctx = _stint_context(nodes, stints).set_index(key)
    st = st.set_index(key).join(ctx)
    idx = st.index
    xg_home = xg.reindex(pd.MultiIndex.from_arrays([idx.get_level_values(0), idx.get_level_values(1),
                                                    st.home_team_id])).fillna(0.0).to_numpy()
    xg_away = xg.reindex(pd.MultiIndex.from_arrays([idx.get_level_values(0), idx.get_level_values(1),
                                                    st.away_team_id])).fillna(0.0).to_numpy()
    mi_home = pd.MultiIndex.from_arrays([idx.get_level_values(0), idx.get_level_values(1), st.home_team_id])
    mi_away = pd.MultiIndex.from_arrays([idx.get_level_values(0), idx.get_level_values(1), st.away_team_id])
    xg2_home = xg2.reindex(mi_home).fillna(0.0).to_numpy()
    xg2_away = xg2.reindex(mi_away).fillna(0.0).to_numpy()
    score = np.clip(st.score_home.to_numpy(dtype=float), -SCORE_CLIP, SCORE_CLIP).astype(int)
    zone = st.zone_home.to_numpy(dtype=object)
    home_sk = [tuple(x) for x in st.home_sk]
    away_sk = [tuple(x) for x in st.away_sk]
    base = {"match_id": idx.get_level_values(0).to_numpy(), "stint": idx.get_level_values(1).to_numpy(),
            "dur_s": st.duration.to_numpy(dtype=float)}
    home_rows = pd.DataFrame({**base, "xg": xg_home, "xg2": xg2_home, "att": home_sk, "dfn": away_sk,
                              "att_team": st.home_team_id.to_numpy(), "def_team": st.away_team_id.to_numpy(),
                              "home": 1, "score": score, "zone": zone})
    away_rows = pd.DataFrame({**base, "xg": xg_away, "xg2": xg2_away, "att": away_sk, "dfn": home_sk,
                              "att_team": st.away_team_id.to_numpy(), "def_team": st.home_team_id.to_numpy(),
                              "home": 0, "score": -score,
                              "zone": np.array([FLIP.get(z, z) for z in zone], dtype=object)})
    return pd.concat([home_rows, away_rows], ignore_index=True)[ROW_COLS + ["xg2"]]


def load_rows(cfg, cache) -> dict[str, pd.DataFrame]:
    """Строки train и test из готовых кэшей L0/L2 (загрузчики L3; пересчёт не запускается)."""
    from . import stage_l3
    l0 = stage_l3._load_l0(cfg, cache)
    sp = stage_l3._load_splits(cfg, cache, l0)
    sel = stage_l3._load_l1_selection(cfg, cache, l0, sp)
    l2 = stage_l3._load_l2(cfg, cache, l0, sp, sel)
    tn = l2["train_nodes"]
    tv = stage_l3.align_values(tn, l2["train_values"], check_source=True)
    te = l0["test"]["nodes"]
    ev = stage_l3.align_values(te, l2["test_values"])
    tr_st = l0["train"]["stints"]
    tr_st = tr_st[tr_st.match_id.isin(set(tn.match_id))]
    return {"train": stint_rows(tn, tr_st, tv.r.to_numpy(dtype=float)),
            "test": stint_rows(te, l0["test"]["stints"], ev.r.to_numpy(dtype=float))}
