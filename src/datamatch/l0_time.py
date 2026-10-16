"""L0: единая расчётная шкала времени периода (t_mono) и флаги происхождения времени.

Одна шкала используется для границ составов (stints), индивидуальных смен, владений,
отрезков между остановками и t_seg / t_poss — поэтому время действия всегда лежит внутри
временных границ состава, к которому оно отнесено по seq (P014).

Правило: t_mono = накопленный максимум clock_s событий игры в порядке seq внутри периода
матча. Будущая bfill-импутация не двигает шкалу. Используется только прошлое
(значение на позиции зависит лишь от событий с меньшим
или равным seq). Служебные метки (зоны, передачи гола, goal_error, scoring_chance) шкалу не
двигают: их отметки ненадёжны, а передачи гола стоят в seq раньше гола с его смещённой
отметкой; такие события получают текущее значение шкалы.

Это расчётная шкала, а не восстановленное физическое время: монотонность гарантирует
порядок, но не точность длительностей. Сырой clock_s сохраняется отдельно; флаг
t_behind_scale отмечает события, чья собственная отметка позади шкалы (откат часов или
поздняя отметка предыдущего события).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_EVENTS = {"entry_success", "exit_success", "exit_controlled", "feed_break",
                "assist_primary", "assist_secondary", "goal_error", "scoring_chance"}


def compute_t_mono(e: pd.DataFrame) -> np.ndarray:
    """e — события, отсортированные по match_id, seq; нужен заполненный clock_s."""
    clock = e.clock_s.to_numpy(dtype=float)
    is_label = e.event_type.isin(LABEL_EVENTS).to_numpy()
    # Back-filled leading timestamps come from future events and cannot anchor
    # causal state or exposure. Keep the raw imputation and its flag for diagnostics.
    future_fill = e.get("clock_impute_kind", pd.Series(None, index=e.index)).eq("bfill").to_numpy()
    v = np.where(is_label | future_fill | np.isnan(clock), -np.inf, clock)
    key = pd.factorize(e.match_id.astype(str) + "|" + e.period.astype(str))[0]
    cm = pd.Series(v).groupby(key, sort=False).cummax().to_numpy()
    # метки в начале периода, до первого события игры, — начало периода
    return np.where(np.isfinite(cm), cm, 0.0)


def add_time_columns(e: pd.DataFrame) -> pd.DataFrame:
    """Добавляет t_mono и флаг t_behind_scale (собственная отметка позади шкалы)."""
    e = e.copy()
    e["t_mono"] = compute_t_mono(e)
    e["t_behind_scale"] = (e.t_mono.to_numpy() > e.clock_s.to_numpy(dtype=float) + 1e-9) & \
        ~e.event_type.isin(LABEL_EVENTS).to_numpy()
    return e


def time_scale_checks(nodes: pd.DataFrame, stints: pd.DataFrame, events: pd.DataFrame) -> dict:
    """Независимая проверка согласованности шкал: время узла внутри временных границ его
    состава; события позади шкалы (насколько сырая отметка отстаёт)."""
    st = stints.set_index(["match_id", "stint"])
    j = st.reindex(pd.MultiIndex.from_arrays([nodes.match_id, nodes.stint]))
    t = nodes.t_mono.to_numpy(dtype=float)
    out_lo = t < j.t_start.to_numpy(dtype=float) - 1e-9
    out_hi = t > j.t_end.to_numpy(dtype=float) + 1e-9
    behind = events.t_behind_scale.to_numpy() if "t_behind_scale" in events else np.zeros(len(events), bool)
    lag = (events.t_mono.to_numpy(dtype=float) - events.clock_s.to_numpy(dtype=float))[behind] if behind.any() \
        else np.zeros(0)
    return {"nodes_outside_stint_time": int((out_lo | out_hi).sum()),
            "nodes_outside_stint_time_matches": int(nodes.match_id[out_lo | out_hi].nunique()),
            "events_behind_scale": int(behind.sum()),
            "events_behind_scale_gt_5s": int((lag > 5).sum()),
            "max_behind_scale_s": float(lag.max()) if len(lag) else 0.0}
