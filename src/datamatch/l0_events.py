"""L0: канонические действия с шайбой, отрезки между остановками, владения.

Ключевые правила (PDF L0 + решения проекта, см. docs/METHODOLOGY.md):
* seq — канонический порядок. clock_s, идущий назад, не меняет порядок; метки
  (entry/exit/feed_break) привязываются к ближайшему действию по времени,
  отрицательные Δt обрезаются до 0.
* Период 0 (буллиты) и штрафные броски (so_attempt) исключаются из основной модели.
* Альтернативные метки train -> канонические (configs: l0.label_map).
* touch + pass/steal/shot одного игрока в ту же секунду -> одно действие (узел).
* Пас самому себе -> удержание (hold), флаг self_pass остаётся признаком состояния.
* Ведение (carry) восстанавливается одинаково в train и test: следующий узел того же
  игрока смещён больше чем на d метров.
* Координаты — в системе команды исполнителя; при переводе в систему другой команды
  x -> -x, y -> -y.
* Смена владения определяется по команде следующего действия с шайбой, а не по меткам
  turnover/steal (в 112 матчах train меток turnover почти нет).
* Отрезок между остановками: начинается вбрасыванием/началом периода, заканчивается
  свистком, голом, концом периода. Внутри отрезка узлы образуют цепь; после
  последнего узла — поглощение с ценностью 0.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .l0_time import add_time_columns, compute_t_mono

LABEL_MAP_EXTRA = {"clearance": "clear_out_success", "dump_misc": "dump_in_failed"}
NON_PUCK_DROP = {"error", "intercept_failed", "misc", "fight_end"} | {f"rush_{a}v{b}" for a in range(4) for b in range(4)}

SHOT_TYPES = {"goal", "shot_on_target", "shot_missed", "shot_blocked"}
PUCK_PRIMARY = SHOT_TYPES | {
    "touch", "pass_complete", "pass_failed", "dump_in_success", "dump_in_failed", "clear_out_success",
    "clear_out_failed", "deke_success", "deke_failed", "steal", "faceoff"}
ZONE_LABELS = {"entry_success", "exit_success", "exit_controlled", "feed_break"}
BREAK_BEFORE = {"faceoff", "period_start", "ot_start", "so_attempt", "so_start"}
BREAK_AFTER = {"whistle", "goal", "intermission", "game_end"}

PRIMARY_PRIORITY = {"goal": 0, "shot_on_target": 1, "shot_missed": 2, "shot_blocked": 3,
                    "pass_complete": 4, "pass_failed": 5, "dump_in_success": 6, "dump_in_failed": 7,
                    "clear_out_success": 8, "clear_out_failed": 9, "deke_success": 10, "deke_failed": 11,
                    "faceoff": 12, "steal": 13, "touch": 14}

ACTION_OF_PRIMARY = {"goal": "shot", "shot_on_target": "shot", "shot_missed": "shot", "shot_blocked": "shot",
                     "pass_complete": "pass", "pass_failed": "pass", "dump_in_success": "dump_in",
                     "dump_in_failed": "dump_in", "clear_out_success": "clear_out", "clear_out_failed": "clear_out",
                     "deke_success": "deke", "deke_failed": "deke", "faceoff": "faceoff", "steal": "hold",
                     "touch": "hold"}
ACTIONS = ["pass", "carry", "shot", "dump_in", "clear_out", "deke", "hold"]          # набор pi_hat (+hold)
ARRIVALS = ["faceoff", "after_faceoff", "recovery", "reception", "continuation", "own_rebound", "loose_team"]
MP_CLASSES = ["EV5", "EVo", "PP", "SH", "ENF", "ENA"]
MP_FLIP = {"EV5": "EV5", "EVo": "EVo", "PP": "SH", "SH": "PP", "ENF": "ENA", "ENA": "ENF"}


def visible_angle(dx, dy, half_width):
    """Угол между направлениями из точки броска на штанги (0, ±half_width относительно центра ворот).

    dx — расстояние до линии ворот по оси x (за воротами < 0), dy — поперечная координата.
    atan2(|v1 × v2|, v1 · v2) всегда в [0, π]. С линии ворот и из-за неё створ не виден,
    поэтому там угол 0: иначе точка за воротами получает угол пятачка (аудит 06.10.2026).
    """
    dx = np.asarray(dx, float)
    dy = np.asarray(dy, float)
    half_width = np.asarray(half_width, float)
    v1y = half_width - dy
    v2y = -half_width - dy
    cross = dx * v2y - v1y * dx
    dot = dx * dx + v1y * v2y
    return np.where(dx > 0, np.arctan2(np.abs(cross), dot), 0.0)


def zone_of_x(x, blue):
    """Зона по x в системе своей команды: off (x > синей линии соперника), def (x < своей синей
    линии), иначе neutral. blue — x синей линии зоны соперника из rink_geometry.csv (скаляр или
    массив по строкам). Пропуск x или blue -> None (зона не подставляется)."""
    if np.ndim(x) == 0 and np.ndim(blue) == 0:
        if x is None or blue is None or (isinstance(x, float) and np.isnan(x)) or np.isnan(float(blue)):
            return None
        return "off" if x > blue else ("def" if x < -blue else "neutral")
    x = np.asarray(x, float)
    b = np.broadcast_to(np.asarray(blue, float), x.shape)
    z = np.where(x > b, "off", np.where(x < -b, "def", "neutral")).astype(object)
    z[np.isnan(x) | np.isnan(b)] = None
    return z


def _check_geo(geo: dict):
    if not geo or not all(isinstance(v, dict) for v in geo.values()):
        raise TypeError("нужна геометрия площадок io.rink_geometry(cfg): тип площадки -> словарь линий")


def make_key(*cols) -> np.ndarray:
    """Строковый ключ из нескольких столбцов; устойчив к пустым и разнотипным (object/str) столбцам."""
    if not cols or not len(cols[0]):
        return np.array([], dtype=object)
    # numpy object-сложение зависит от фактического dtype элементов (в полном потоке встречаются
    # float/NA actor_id), поэтому приводим каждую часть к unicode до конкатенации.
    out = pd.Series(cols[0]).fillna("<NA>").astype(str).to_numpy(dtype=str)
    for col in cols[1:]:
        p = pd.Series(col).fillna("<NA>").astype(str).to_numpy(dtype=str)
        out = np.char.add(np.char.add(out, "|"), p)
    return out.astype(object)


def fix_games(games: pd.DataFrame, ev: pd.DataFrame, quality: dict) -> pd.DataFrame:
    """Сверка команд матча с событиями; при однозначном несовпадении — исправление по событиям."""
    g = games.copy()
    teams_ev = ev.dropna(subset=["actor_team_id"]).groupby("match_id").actor_team_id.unique()
    fixes = []
    for i, r in g.iterrows():
        te = set(teams_ev.get(r.match_id, []))
        gs = {r.home_team_id, r.away_team_id}
        if te and te != gs:
            extra = te - gs
            missing = gs - te
            if len(extra) == 1 and len(missing) == 1:
                old = missing.pop()
                new = extra.pop()
                col = "home_team_id" if r.home_team_id == old else "away_team_id"
                g.at[i, col] = new
                fixes.append({"match_id": r.match_id, "column": col, "games_csv": old, "events": new})
            else:
                fixes.append({"match_id": r.match_id, "column": None, "games_csv": sorted(gs), "events": sorted(te),
                              "unresolved": True})
    quality["games_team_fixes"] = fixes
    return g


def preprocess(ev: pd.DataFrame, games: pd.DataFrame, cfg, quality: dict) -> pd.DataFrame:
    """Отображение меток, схема разметки, импутация clock_s. Возвращает события без lineup_*."""
    lm = dict(cfg["l0"]["label_map"])
    lm.update(LABEL_MAP_EXTRA)
    raw_types = ev.event_type
    is_lineup = raw_types.str.startswith("lineup_")
    e = ev[~is_lineup].copy()
    # схема разметки матча: v2 = матчи без touch и с receive/rebound/board_pass
    has_v2 = e.event_type.isin(["receive", "rebound", "board_pass_complete", "carry"]).groupby(e.match_id).any()
    has_touch = (e.event_type == "touch").groupby(e.match_id).any()
    scheme = pd.Series(np.where(has_v2 & ~has_touch, "v2", "v1"), index=has_v2.index)
    e["scheme"] = e.match_id.map(scheme)
    e["raw_type"] = e.event_type
    e["event_type"] = e.event_type.map(lambda t: lm.get(t, t))
    dropped = e.event_type.isin(NON_PUCK_DROP)
    quality["dropped_nonpuck_labels"] = e.loc[dropped, "raw_type"].value_counts().to_dict()
    e = e[~dropped]
    # импутация clock_s: предыдущее значение в периоде матча (seq — канонический порядок)
    miss = e.clock_s.isna()
    quality["clock_missing"] = int(miss.sum())
    quality["clock_missing_by_type"] = e.loc[miss, "event_type"].value_counts().to_dict()
    e["clock_imputed"] = miss.to_numpy()
    grp = [e.match_id, e.period]
    from .l0_time import LABEL_EVENTS
    # Label clocks may describe future outcomes despite their earlier seq. They
    # must not become an indirect causal anchor through a missing game's timestamp.
    anchors = e.clock_s.where(~e.event_type.isin(LABEL_EVENTS))
    ff = anchors.groupby(grp, sort=False).ffill()
    bf = anchors.groupby(grp, sort=False).bfill()
    # bfill не должен пролезать под маской ffill: используем ffill только где он есть,
    # иначе bfill с правильной меткой kind, иначе zero.
    kind = np.where(~miss, None,
                np.where(ff.notna(), "ffill",
                np.where(bf.notna(), "bfill", "zero")))
    e["clock_impute_kind"] = kind
    # Значение: ffill где есть, иначе bfill, иначе 0. kind корректно отражает источник.
    e["clock_s"] = e.clock_s.where(~miss, ff.where(ff.notna(), bf).fillna(0.0))
    # backward clock в порядке seq (по всем событиям, кроме служебных)
    d = e.groupby(grp, sort=False).clock_s.diff()
    back = d < 0
    quality["clock_backward_n"] = int(back.sum())
    quality["clock_backward_matches"] = int(e.loc[back, "match_id"].nunique())
    quality["clock_backward_by_type"] = e.loc[back, "event_type"].value_counts().head(8).to_dict()
    quality["clock_backward_abs_median_s"] = float((-d[back]).median()) if back.any() else 0.0
    quality["clock_backward_abs_max_s"] = float((-d[back]).max()) if back.any() else 0.0
    quality["_clock_backward_values"] = (-d[back]).astype(float).tolist()
    quality["scheme_counts"] = scheme.value_counts().to_dict()
    # единая расчётная шкала времени периода (P014) и флаг «отметка позади шкалы»
    e = add_time_columns(e.reset_index(drop=True))
    return e, scheme, is_lineup


def build_nodes(e: pd.DataFrame, games: pd.DataFrame, stints: pd.DataFrame, geo: dict, cfg, quality: dict):
    """Построение узлов (канонических действий с шайбой) и их состояний.

    geo — геометрия площадок из rink_geometry.csv (io.rink_geometry), единственный источник
    границ, линий ворот, штанг и синих линий (P011)."""
    _check_geo(geo)
    c0 = cfg["l0"]
    gm = games.set_index("match_id")
    e = e.copy()
    e["home_team_id"] = e.match_id.map(gm.home_team_id)
    e["away_team_id"] = e.match_id.map(gm.away_team_id)
    e["rink_type"] = e.match_id.map(gm.rink_type)
    e["stage"] = e.match_id.map(gm.stage)
    et = e.event_type.values
    mid = e.match_id.values
    per = e.period.values

    # --- отрезки между остановками -------------------------------------------------------
    bb = np.isin(et, list(BREAK_BEFORE))
    ba = np.isin(et, list(BREAK_AFTER))
    new_match = np.r_[True, mid[1:] != mid[:-1]]
    new_per = np.r_[True, per[1:] != per[:-1]]
    prev_ba = np.r_[False, ba[:-1]]
    seg_start = bb | prev_ba | new_match | new_per
    e["segment"] = np.cumsum(seg_start)
    # единая расчётная шкала времени периода (l0_time, P014): та же, что у границ составов;
    # сырой clock_s в признаках до действия не меняется
    if "t_mono" not in e:
        e["t_mono"] = compute_t_mono(e)
    tm = e.t_mono.values
    # время остановки каждого отрезка (явная граница для владений): свисток/гол/конец периода
    # внутри отрезка; иначе — начало следующего отрезка того же периода (вбрасывание без
    # свистка); иначе — конец периода (максимум шкалы периода матча)
    sg_all = e.segment.values
    seg_first = pd.DataFrame({"match": mid, "period": per, "clock": tm,
                              "seg": sg_all}).groupby("seg", sort=True).first()
    stop_ba = pd.Series(tm[ba], index=sg_all[ba]).groupby(level=0).first()
    nxt_same = ((seg_first.match.shift(-1) == seg_first.match) & (seg_first.period.shift(-1) == seg_first.period)).values
    pend = pd.Series(tm).groupby([mid, per]).max()
    pend_seg = pend.reindex(pd.MultiIndex.from_arrays([seg_first.match.values, seg_first.period.values])).values
    fallback = np.where(nxt_same, seg_first.clock.shift(-1).values, pend_seg)
    seg_stop = stop_ba.reindex(seg_first.index).values
    seg_stop = np.where(np.isnan(seg_stop), fallback, seg_stop)
    seg_stop_map = pd.Series(seg_stop, index=seg_first.index)
    first_type = e.groupby("segment", sort=True).event_type.first()
    known_boundary = stop_ba.reindex(seg_first.index).notna() | (
        (seg_first.match.shift(-1) == seg_first.match) & first_type.shift(-1).isin(BREAK_BEFORE))
    seg_terminal_map = known_boundary.astype(bool)
    quality["segment_stop_source"] = {"stoppage_event": int(stop_ba.size),
                                      "next_segment_start": int((np.isnan(stop_ba.reindex(seg_first.index).values) & nxt_same).sum()),
                                      "period_end": int((np.isnan(stop_ba.reindex(seg_first.index).values) & ~nxt_same).sum())}
    # штрафные броски / буллиты: отрезок, начинающийся с so_attempt, исключается
    ps_seg = set(e.loc[(e.event_type == "so_attempt"), "segment"])
    e["penalty_shot_seg"] = e.segment.isin(ps_seg)
    quality["penalty_shot_segments_excluded"] = int(len(set(e.loc[e.period > 0].segment) & ps_seg))

    # --- счёт до события ------------------------------------------------------------------
    # голы штрафных бросков в периодах 1–5 входят в счёт; исключается только сама попытка
    # из моделей; период 0 (серия буллитов) в счёт не входит
    is_goal = (e.event_type == "goal") & (e.period > 0)
    e["goal_home"] = (is_goal & (e.actor_team_id == e.home_team_id)).astype(int)
    e["goal_away"] = (is_goal & (e.actor_team_id == e.away_team_id)).astype(int)
    cg = e.groupby("match_id", sort=False)[["goal_home", "goal_away"]].cumsum()
    e["score_home_before"] = cg.goal_home - e.goal_home
    e["score_away_before"] = cg.goal_away - e.goal_away

    # --- доверенное предбросковое состояние: n_home/n_away предыдущего события по seq ------
    grp_mp = [e.match_id, e.period]
    from .l0_time import LABEL_EVENTS
    physical_position = pd.Series(np.arange(len(e)), index=e.index).where(~e.event_type.isin(LABEL_EVENTS))
    prior = physical_position.groupby(grp_mp, sort=False).ffill().groupby(grp_mp, sort=False).shift(1)
    has_prior = prior.notna().to_numpy()
    position = prior.fillna(0).to_numpy(dtype=int)
    e["n_home_prev"] = np.where(has_prior, e.n_home.to_numpy()[position], np.nan)
    e["n_away_prev"] = np.where(has_prior, e.n_away.to_numpy()[position], np.nan)

    # --- первичные события с шайбой -------------------------------------------------------
    # События без исполнителя (P008): при известной и валидной для матча команде — командный
    # узел (node_kind = team; в цепи и владении участвует, личного вклада не получает); без
    # валидной команды — не узел: владелец не выдумывается, следующий узел получает флаг
    # неопределённости цепи chain_gap_before; вбрасывание без исполнителя учитывается как
    # открывающее отрезок при определении класса прибытия.
    is_puck = np.isin(et, list(PUCK_PRIMARY)) & (e.period.values > 0) & ~e.penalty_shot_seg.values
    has_actor = e.actor_id.notna().values
    team_valid = (e.actor_team_id.eq(e.home_team_id) | e.actor_team_id.eq(e.away_team_id)).fillna(False).values
    # Вбрасывание без исполнителя и команды открывает отрезок: до него в отрезке нет цепи,
    # которую оно могло бы разорвать, поэтому оно не барьер и не неопределённость цепи.
    actorless_faceoff = is_puck & ~has_actor & ~team_valid & (et == "faceoff")
    e["unatt_gap"] = is_puck & ~has_actor & ~team_valid & ~actorless_faceoff
    quality["actorless_faceoff_openers"] = int(actorless_faceoff.sum())
    quality["actorless_puck_events"] = int((is_puck & ~has_actor).sum())
    quality["actorless_team_nodes_by_type"] = e.loc[is_puck & ~has_actor & team_valid, "event_type"].value_counts().to_dict()
    quality["actorless_no_team_by_type"] = e.loc[e.unatt_gap.values, "event_type"].value_counts().to_dict()
    pk = e[is_puck & (has_actor | team_valid)].copy()
    e["uncertain_epoch"] = e.unatt_gap.astype(int).groupby(e.segment).cumsum()
    pk["uncertain_epoch"] = e.loc[pk.index, "uncertain_epoch"]
    quality["puck_events"] = int(len(pk))
    # слияние: тот же игрок, тот же момент (clock до 0.1 с), тот же отрезок и тот же состав
    # на льду (stint), подряд в потоке шайбы; замена между событиями запрещает слияние —
    # иначе действие узла отнеслось бы к другому составу
    a = pk.actor_id.values
    ck = np.round(pk.clock_s.values, 1)
    sg = pk.segment.values
    if len(pk) == 0:
        new_node = np.zeros(0, dtype=bool)        # порция/матч без действий с шайбой
    else:
        epoch = pk.uncertain_epoch.to_numpy()
        team_pk = pk.actor_team_id.to_numpy()
        new_node = np.r_[True, (team_pk[1:] != team_pk[:-1]) | (a[1:] != a[:-1]) | (ck[1:] != ck[:-1]) | (sg[1:] != sg[:-1])
                        | (epoch[1:] != epoch[:-1])]
    # вбрасывание всегда отдельный узел; бросок никогда не сливается с событием другого типа
    # (слияние по равенству времени зависело бы от отметки броска, а у голов она смещена);
    # два броска подряд одного игрока в один момент — дубль одной попытки, сливаются
    isfo = pk.event_type.values == "faceoff"
    new_node |= isfo
    new_node[1:] |= isfo[:-1]
    issh = np.isin(pk.event_type.values, list(SHOT_TYPES))
    new_node[1:] |= issh[1:] != issh[:-1]
    # событие без исполнителя — всегда отдельный (командный) узел
    noact = pk.actor_id.isna().values
    new_node |= noact
    new_node[1:] |= noact[:-1]
    # восстановленное время не служит основанием слияния (P009)
    if "clock_imputed" in pk:
        imp = pk.clock_imputed.fillna(False).values.astype(bool)
        imp_split = imp.copy()
        imp_split[1:] |= imp[:-1]
        blocked = imp_split & ~new_node
        quality["merge_blocked_by_imputed_clock"] = int(blocked.sum())
        quality["merge_blocked_by_imputed_clock_by_type"] = pk.event_type[blocked].value_counts().to_dict()
        new_node |= imp_split
    if "stint" in pk:
        stv = pk.stint.values
        stint_split = np.r_[False, stv[1:] != stv[:-1]]
        quality["merge_blocked_by_stint_change"] = int((stint_split & ~new_node).sum())
        new_node |= stint_split
    pk["node"] = np.cumsum(new_node) - 1
    pk["prio"] = pk.event_type.map(PRIMARY_PRIORITY)
    quality["merged_events"] = int(len(pk) - pk.node.nunique())
    grp = pk.groupby("node", sort=False)
    first = grp.head(1).set_index("node")
    prim_idx = pk.sort_values(["node", "prio"], kind="mergesort").groupby("node", sort=False).head(1).set_index("node")
    has_steal = (pk.event_type == "steal").groupby(pk.node.values).any()
    n_merged = grp.size()

    nd = pd.DataFrame(index=first.index)
    nd["match_id"] = first.match_id
    nd["seq"] = first.seq
    nd["primary"] = prim_idx.event_type
    nd["n_events"] = n_merged
    nd["actor_id"] = first.actor_id
    nd["team_id"] = first.actor_team_id
    nd["other_id"] = prim_idx.other_id
    nd["period"] = first.period
    nd["clock"] = first.clock_s
    nd["t_mono"] = first.t_mono
    nd["segment"] = first.segment
    nd["scheme"] = first.scheme
    nd["rink_type"] = first.rink_type
    nd["stage"] = first.stage
    nd["home_team_id"] = first.home_team_id
    nd["away_team_id"] = first.away_team_id
    # координаты: первичного события, иначе любого в группе
    xy = pk[["node", "x", "y"]].dropna().groupby("node").first()
    nd["x"] = prim_idx.x.where(prim_idx.x.notna(), xy.x.reindex(nd.index))
    nd["y"] = prim_idx.y.where(prim_idx.y.notna(), xy.y.reindex(nd.index))
    nd["x2"] = prim_idx.x2
    nd["y2"] = prim_idx.y2
    # у бросков численность берётся на предыдущем событии (состояние до броска)
    is_shot_node = prim_idx.event_type.isin(list(SHOT_TYPES)).reindex(nd.index).values
    nd["n_home"] = np.where(is_shot_node, first.n_home_prev, first.n_home)
    nd["n_away"] = np.where(is_shot_node, first.n_away_prev, first.n_away)
    nd["first_seq"] = first.seq
    nd["last_seq"] = grp.seq.max()
    nd["zone_raw"] = prim_idx.zone
    nd["score_home_before"] = first.score_home_before
    nd["score_away_before"] = first.score_away_before
    nd["has_steal"] = has_steal
    nd["stint"] = first.stint if "stint" in first else -1
    nd["node_kind"] = np.where(first.actor_id.isna().values, "team", "player")
    nd["clock_imputed"] = first.clock_imputed.fillna(False).values.astype(bool) if "clock_imputed" in first else False
    nd["clock_impute_kind"] = first.clock_impute_kind if "clock_impute_kind" in first else None
    nd["t_behind_scale"] = first.t_behind_scale.values if "t_behind_scale" in first else False
    nd = nd.reset_index(drop=True)
    nd["node"] = np.arange(len(nd))

    # --- отсутствующие координаты: исключение с причиной ---------------------------------
    noxy = nd.x.isna() | nd.y.isna()
    quality["nodes_missing_xy"] = int(noxy.sum())
    quality["nodes_missing_xy_by_type"] = nd.loc[noxy, "primary"].value_counts().to_dict()
    # геометрия площадки узла (P011); неизвестный тип площадки — исключение с причиной
    rk = nd.rink_type.astype(str)
    known_rink = rk.isin(list(geo)).values
    quality["nodes_unknown_rink_type"] = int((~known_rink).sum())
    gcol = lambda k: rk.map({r: v[k] for r, v in geo.items()}).astype(float).values
    x_lo, x_hi, y_lo, y_hi = gcol("x_min"), gcol("x_max"), gcol("y_min"), gcol("y_max")
    # координаты за пределами площадки — обрезка по границам этого типа площадки
    out = (nd.x.values < x_lo) | (nd.x.values > x_hi) | (nd.y.values < y_lo) | (nd.y.values > y_hi)
    quality["nodes_xy_clipped"] = int(np.nansum(out & known_rink))
    nd["x"] = np.clip(nd.x.values, x_lo, x_hi)
    nd["y"] = np.clip(nd.y.values, y_lo, y_hi)

    nd["is_home"] = nd.team_id == nd.home_team_id
    nd["opp_team_id"] = np.where(nd.is_home, nd.away_team_id, nd.home_team_id)
    bad_team = ~(nd.team_id.eq(nd.home_team_id) | nd.team_id.eq(nd.away_team_id))
    quality["nodes_team_not_in_match"] = int(bad_team.sum())
    nd["excluded"] = None
    nd.loc[noxy, "excluded"] = "missing_xy"
    nd.loc[bad_team, "excluded"] = "team_not_in_match"
    nd.loc[~known_rink, "excluded"] = "unknown_rink_type"

    # --- состав на льду: вратари из реконструкции -----------------------------------------
    st = stints.set_index(["match_id", "stint"])
    j = st.reindex(pd.MultiIndex.from_arrays([nd.match_id, nd.stint]))
    nd["home_g_in"] = j.home_g_in.fillna(True).values.astype(bool)
    nd["away_g_in"] = j.away_g_in.fillna(True).values.astype(bool)
    rec_h = j.n_home_sk.fillna(5).values + nd.home_g_in.astype(int).values
    rec_a = j.n_away_sk.fillna(5).values + nd.away_g_in.astype(int).values
    nh = nd.n_home.where(nd.n_home.notna(), rec_h).values
    na = nd.n_away.where(nd.n_away.notna(), rec_a).values
    # n_home/n_away включают вратаря, на стороне не больше 6; вне [3, 6] — берём реконструкцию
    bad_n = (nh < 3) | (nh > 6) | (na < 3) | (na > 6)
    quality["nodes_n_out_of_range_replaced"] = int(bad_n.sum())
    nh = np.where(bad_n, rec_h, nh)
    na = np.where(bad_n, rec_a, na)
    own_g = np.where(nd.is_home, nd.home_g_in, nd.away_g_in)
    opp_g = np.where(nd.is_home, nd.away_g_in, nd.home_g_in)
    own_n = np.where(nd.is_home, nh, na)
    opp_n = np.where(nd.is_home, na, nh)
    nd["own_g"] = own_g
    nd["opp_g"] = opp_g
    nd["own_sk"] = (own_n - own_g.astype(int)).astype(int)
    nd["opp_sk"] = (opp_n - opp_g.astype(int)).astype(int)
    mp = np.where(~opp_g, "ENF", np.where(~own_g, "ENA",
                  np.where(nd.own_sk > nd.opp_sk, "PP", np.where(nd.own_sk < nd.opp_sk, "SH",
                           np.where(nd.own_sk == 5, "EV5", "EVo")))))
    nd["mp"] = mp
    nd["score_diff"] = np.where(nd.is_home, nd.score_home_before - nd.score_away_before,
                                nd.score_away_before - nd.score_home_before)

    # --- время периода --------------------------------------------------------------------
    # Наблюдаемый конец периода зависит от будущих событий и годится только для диагностики
    # качества/экспозиции. Состояние ДО действия использует номинальный регламентный конец;
    # иначе изменение будущей отметки intermission меняет признаки прошлых действий (P003).
    nominal = np.where(nd.period <= 3, 1200.0, np.where(nd.stage == "playoff", 1200.0, 300.0))
    period_end_obs = (e[e.period > 0].groupby(["match_id", "period"], sort=False).t_mono.max())
    pkey = pd.MultiIndex.from_arrays([nd.match_id, nd.period])
    observed = period_end_obs.reindex(pkey).to_numpy(dtype=float)
    nd["period_nominal"] = nominal
    nd["period_end_observed"] = observed
    nd["period_end"] = nominal
    nd["period_end_source"] = "nominal_regulation"
    nd["t_beyond_nominal"] = nd.t_mono.values > nominal
    nd["t_remain"] = np.clip(nominal - nd.t_mono.values, 0, None)

    # --- геометрия ------------------------------------------------------------------------
    gx = gcol("goal_line_opp")
    dx = gx - nd.x.values
    dy = nd.y.values
    nd["dist"] = np.sqrt(dx ** 2 + dy ** 2)
    nd["angle_vis"] = visible_angle(dx, dy, gcol("post_half_width"))   # видимый угол ворот, рад, в [0, π]
    nd["angle_abs"] = np.abs(np.arctan2(dy, dx))
    nd["behind_net"] = dx < 0
    nd["zone"] = zone_of_x(nd.x.values, gcol("blue_line_opp"))
    zr = nd.zone_raw.fillna(nd.zone)
    quality["zone_x_disagreement"] = float((zr != nd.zone).mean())

    # --- исключённые узлы разрывают цепь, а не исчезают бесследно (P008) -----------------
    # У них могут быть тип/команда/время, но нет координаты. Сохранить их как полноценный
    # spatial-узел нельзя; соединять действия до и после них тоже нельзя: это создаёт ложные
    # carry, пасовую историю и добивания. В excl остаётся достаточный след для аудита.
    removed = nd[nd.excluded.notna()].copy()
    gaps = e[e.unatt_gap][["match_id", "seq", "event_type", "actor_team_id", "clock_s", "period", "segment"]].rename(
        columns={"event_type": "primary", "actor_team_id": "team_id", "clock_s": "clock"})
    gaps["excluded"] = "unknown_possession"
    barriers_all = pd.concat([removed, gaps], ignore_index=True)
    excl = barriers_all[["match_id", "seq", "primary", "team_id", "clock", "period", "segment", "excluded"]]
    nd = nd[nd.excluded.isna()].reset_index(drop=True)
    nd["node"] = np.arange(len(nd))

    barrier_before = np.zeros(len(nd), dtype=bool)
    barrier_after = np.zeros(len(nd), dtype=bool)
    if len(barriers_all) and len(nd):
        bkeys = barriers_all.groupby(["match_id", "period", "segment"], sort=False).seq.agg(list)
        for _, ii in nd.groupby(["match_id", "period", "segment"], sort=False).groups.items():
            idx = np.asarray(list(ii), dtype=int)
            r = nd.iloc[idx[0]]
            barriers = np.asarray(bkeys.get((r.match_id, r.period, r.segment), []), dtype=np.int64)
            if not len(barriers):
                continue
            barriers.sort()
            prev_seq = -np.inf
            for k, i in enumerate(idx):
                cur_seq = nd.first_seq.iat[i]
                barrier_before[i] = np.any((barriers > prev_seq) & (barriers < cur_seq))
                next_seq = nd.first_seq.iat[idx[k + 1]] if k + 1 < len(idx) else np.inf
                barrier_after[i] = np.any((barriers > nd.last_seq.iat[i]) & (barriers < next_seq))
                prev_seq = nd.last_seq.iat[i]
    quality["chain_barrier_nodes_excluded"] = int(len(barriers_all))
    quality["chain_barrier_before_retained_node"] = int(barrier_before.sum())

    # --- метки зон: привязка к ближайшему узлу по времени того же игрока ------------------
    # Поиск ограничен матчем, периодом, отрезком между остановками и игроком: время каждого
    # периода начинается заново, поэтому без периода/отрезка метка могла прикрепиться к
    # действию другого периода.
    lab = e[e.event_type.isin(ZONE_LABELS) & (e.period > 0) & e.actor_id.notna()][
        ["match_id", "period", "segment", "actor_id", "clock_s", "event_type", "seq"]].copy()
    nd["entry"] = False
    nd["exit"] = False
    if len(lab) and len(nd):
        tol = float(c0["label_attach_window_s"])
        key_n = make_key(nd.match_id, nd.period, nd.segment, nd.actor_id)
        key_l = make_key(lab.match_id, lab.period, lab.segment, lab.actor_id)
        # детерминированный порядок (время, затем seq/номер узла; стабильная сортировка): при
        # равном расстоянии до двух узлов выбор не должен зависеть от состава порции матчей
        left = pd.DataFrame({"k": key_l, "t": lab.clock_s.values, "ltype": lab.event_type.values,
                             "lmatch": lab.match_id.values, "lper": lab.period.values,
                             "lseg": lab.segment.values, "lseq": lab.seq.values}
                            ).sort_values(["t", "lmatch", "lseq"], kind="mergesort")
        right = pd.DataFrame({"k": key_n, "t_node": nd.clock.values, "t": nd.clock.values,
                              "node": nd.node.values, "rmatch": nd.match_id.values,
                              "rseq": nd.first_seq.values}).sort_values(["t", "rmatch", "rseq"], kind="mergesort")
        mm = pd.merge_asof(left, right, on="t", by="k", direction="nearest", tolerance=tol)
        ok = mm.node.notna().values
        # проверка границ после соединения: тот же матч, период, отрезок и |Δt| <= окна
        ni = np.where(ok, mm.node.fillna(-1).astype(int).values, 0)
        same = (ok & (nd.match_id.values[ni] == mm.lmatch.values) & (nd.period.values[ni] == mm.lper.values)
                & (nd.segment.values[ni] == mm.lseg.values)
                & (np.abs(nd.clock.values[ni] - mm.t.values) <= tol + 1e-9))
        quality["zone_labels"] = int(len(left))
        quality["zone_labels_attached"] = float(same.mean())
        quality["zone_labels_rejected_boundary"] = int((ok & ~same).sum())
        mm = mm[same]
        ent = mm[mm.ltype == "entry_success"].node.astype(int).unique()
        ext = mm[mm.ltype.isin(["exit_success", "exit_controlled"])].node.astype(int).unique()
        nd.loc[ent, "entry"] = True
        nd.loc[ext, "exit"] = True
    elif len(lab):
        quality["zone_labels"] = int(len(lab))
        quality["zone_labels_attached"] = 0.0
        quality["zone_labels_rejected_boundary"] = 0

    # --- соседи в цепи --------------------------------------------------------------------
    seg = nd.segment.values
    team = nd.team_id.values
    act = nd.actor_id.values
    same_seg_next = np.r_[(seg[1:] == seg[:-1]) & ~barrier_before[1:], False]
    same_seg_prev = np.r_[False, (seg[1:] == seg[:-1]) & ~barrier_before[1:]]
    nxt = np.where(same_seg_next, np.arange(len(nd)) + 1, -1)
    prv = np.where(same_seg_prev, np.arange(len(nd)) - 1, -1)
    nd["next_node"] = nxt
    nd["prev_node"] = prv
    nxt_team = np.where(nxt >= 0, team[np.clip(nxt, 0, None)], None)
    prv_team = np.where(prv >= 0, team[np.clip(prv, 0, None)], None)
    nd["sigma_next"] = np.where(nxt < 0, 0, np.where(nxt_team == team, 1, -1)).astype(np.int8)
    # Absence of a spatial successor can be censoring, not a real stoppage.
    confirmed_end = nd.segment.map(seg_terminal_map).fillna(False).to_numpy(dtype=bool)
    unknown_transition = barrier_after | ((nxt < 0) & ~confirmed_end)
    nd["terminal"] = (nxt < 0) & ~unknown_transition
    nd["transition_status"] = np.where(unknown_transition, "unknown", np.where(nd.terminal, "terminal", "observed"))
    nd["valid_for_transition"] = ~unknown_transition
    nd["poss_change_next"] = nd.sigma_next == -1
    prim = nd.primary.values
    # пас самому себе
    selfp = (np.isin(prim, ["pass_complete", "pass_failed"])) & (nd.other_id.values == act)
    nd["self_pass"] = selfp

    # --- класс действия ------------------------------------------------------------------
    action = nd.primary.map(ACTION_OF_PRIMARY).values.astype(object)
    action[selfp] = "hold"
    nx = np.clip(nxt, 0, None)
    same_actor_next = (nxt >= 0) & (act[nx] == act)
    disp = np.sqrt((nd.x.values[nx] - nd.x.values) ** 2 + (nd.y.values[nx] - nd.y.values) ** 2)
    # пас самому себе остаётся удержанием независимо от смещения (PDF: не ведение)
    is_carry = (action == "hold") & ~selfp & same_actor_next & (disp > float(c0["carry_min_dist_m"]))
    action[is_carry] = "carry"
    nd["action"] = action
    nd["carry_dist"] = np.where(is_carry, disp, 0.0)

    # --- класс прибытия (состояние до выбора действия) -----------------------------------
    pv = np.clip(prv, 0, None)
    arrival = np.full(len(nd), "loose_team", dtype=object)
    has_prev = prv >= 0
    prev_same_team = has_prev & (team[pv] == team)
    prev_opp = has_prev & ~prev_same_team
    prev_prim = np.where(has_prev, prim[pv], "")
    prev_act = np.where(has_prev, action[pv], "")
    prev_actor = np.where(has_prev, act[pv], None)
    arrival[prev_opp] = "recovery"
    arrival[prev_same_team & (prev_actor == act)] = "continuation"
    arrival[prev_same_team & (prev_actor != act) & (prev_prim == "pass_complete")] = "reception"
    arrival[prev_same_team & (prev_act == "shot")] = "own_rebound"
    arrival[prev_same_team & (prev_prim == "faceoff")] = "after_faceoff"
    arrival[~has_prev] = "loose_team"
    # открывающее событие отрезка (вбрасывание — в т. ч. без исполнителя, начало периода или
    # овертайма), стоящее по seq раньше первого узла: доверенная опора времени (P007) и
    # контекст прибытия (P008)
    seg_open = e.groupby("segment", sort=False).head(1)
    seg_open = seg_open[seg_open.event_type.isin(["faceoff", "period_start", "ot_start"])]
    op_type = nd.segment.map(pd.Series(seg_open.event_type.values, index=seg_open.segment.values)).values
    op_seq = nd.segment.map(pd.Series(seg_open.seq.values, index=seg_open.segment.values)).values.astype(float)
    op_clock = nd.segment.map(pd.Series(seg_open.t_mono.values, index=seg_open.segment.values)).values.astype(float)
    opener_ok = (~has_prev) & ~barrier_before & ~pd.isna(op_type) & (op_seq < nd.first_seq.values.astype(float))
    arrival[opener_ok & (op_type == "faceoff") & (prim != "faceoff")] = "after_faceoff"
    arrival[prim == "faceoff"] = "faceoff"
    nd["arrival"] = arrival
    nd["prev_action"] = np.where(has_prev, prev_act, "none")
    nd["prev_self_pass"] = np.where(has_prev, selfp[pv], False)
    dtp = np.where(has_prev, nd.t_mono.values - nd.t_mono.values[pv], np.nan)
    nd["dt_prev"] = np.clip(dtp, 0, None)
    # предыдущее действие в системе текущей команды
    sgn = np.where(prev_same_team, 1.0, -1.0)
    nd["prev_x"] = np.where(has_prev, sgn * nd.x.values[pv], np.nan)
    nd["prev_y"] = np.where(has_prev, sgn * nd.y.values[pv], np.nan)
    pv2 = np.clip(np.where(has_prev, prv[pv], -1), 0, None)
    has_prev2 = has_prev & (prv[pv] >= 0)
    nd["prev2_action"] = np.where(has_prev2, action[pv2], "none")
    same2 = has_prev2 & (team[pv2] == team)
    nd["prev2_x"] = np.where(has_prev2, np.where(same2, 1.0, -1.0) * nd.x.values[pv2], np.nan)

    # время остановки отрезка узла (явная граница владений) по монотонной шкале; диагностика —
    # сколько остановок по сырому времени раньше последнего действия отрезка
    seg_last_raw = nd.groupby("segment").clock.transform("max").values
    seg_last = nd.groupby("segment").t_mono.transform("max").values
    stop = nd.segment.map(seg_stop_map).values.astype(float)
    stop_raw = nd.segment.map(pd.Series(e.clock_s.values[ba], index=sg_all[ba]).groupby(level=0).first()).values
    quality["segment_stop_before_last_action_raw_clock"] = int(
        pd.Series(stop_raw < seg_last_raw - 1e-9).groupby(nd.segment.values).any().sum())
    nd["seg_stop"] = np.maximum(np.where(np.isnan(stop), seg_last, stop), seg_last)

    # время с начала отрезка и с начала владения: начало — ПЕРВЫЙ по seq узел группы, время —
    # монотонная шкала t_mono (накопленный максимум только по прошлым событиям); минимум по
    # всей группе использовать нельзя: при откате часов позднее событие меняло бы признаки
    # уже прошедших действий
    tmn = nd.t_mono.values
    seg_t0 = nd.groupby("segment", sort=False).t_mono.transform("first").values
    nd["t_seg"] = np.clip(tmn - seg_t0, 0, None)
    poss_start = ~prev_same_team
    nd["possession"] = np.cumsum(poss_start) - 1
    pt0 = nd.groupby("possession", sort=False).t_mono.transform("first").values
    nd["t_poss"] = np.clip(tmn - pt0, 0, None)
    nd["n_in_poss"] = nd.groupby("possession").cumcount()

    # Временные окна признаков считаются от момента ПРЕДЫДУЩЕГО узла (t_ref), а не от
    # отметки текущего: у голов отметка времени систематически позже (сигнал исхода).
    # t_ref — доверенная опора времени ДО действия: время предыдущего узла отрезка; для первого
    # узла — время открывающего события отрезка (вбрасывание, начало периода/овертайма), если
    # оно есть и стоит раньше по seq; иначе опоры нет — окна признаков отключаются
    # (t_ref_source = none). Собственная отметка узла опорой не служит (P007).
    # All elapsed-time windows share the causal monotone clock. A rollback may
    # not revive expired history. Raw clocks remain in nd.clock for diagnostics.
    clock = nd.t_mono.values
    t_ref = np.where(has_prev, clock[pv], np.where(opener_ok, op_clock, np.nan))
    nd["t_ref"] = t_ref
    nd["t_ref_source"] = np.where(has_prev, "prev_node", np.where(opener_ok, "segment_opener", "none"))
    no_ref = np.isnan(t_ref)
    # пас, доставивший шайбу исполнителю: последний точный пас партнёра этому игроку перед
    # непрерывной серией его узлов (приём -> ведение/удержание -> бросок)
    n = len(nd)
    last_pass = np.full(n, -1, dtype=np.int64)
    cont = prev_same_team & (prev_actor == act)
    recv = prev_same_team & (prev_actor != act) & (prev_prim == "pass_complete")
    for i in range(n):
        if recv[i]:
            last_pass[i] = pv[i]
        elif cont[i]:
            last_pass[i] = last_pass[pv[i]]
    lp = np.clip(last_pass, 0, None)
    hp = last_pass >= 0
    y_from = nd.y.values[lp]
    rec_node = np.clip(nxt[lp], 0, None)                 # узел приёма — следующий за пасом
    y_to = np.where(np.isnan(nd.y2.values[lp]), nd.y.values[rec_node], nd.y2.values[lp])
    within = hp & ~no_ref & ((t_ref - clock[lp]) <= float(c0["pass_window_s"]))
    nd["prev_pass_dy"] = np.where(within, np.abs(y_to - y_from), 0.0)
    nd["prev_pass_cross"] = within & (np.sign(y_to) != np.sign(y_from)) & (np.abs(y_to - y_from) > 0)
    nd["last_pass_node"] = last_pass
    # добивание: бросок своей команды в том же отрезке не раньше чем за 3 с до t_ref
    shot_t = pd.Series(np.where(action == "shot", clock, np.nan))
    chain_epoch = np.cumsum(barrier_before)
    key_team_seg = pd.Series(pd.factorize(nd.segment.astype(str) + "|" + nd.team_id.astype(str)
                                          + "|" + pd.Series(chain_epoch).astype(str))[0])
    last_shot = shot_t.groupby(key_team_seg).shift(1).groupby(key_team_seg).ffill()
    nd["dt_last_own_shot"] = t_ref - last_shot.values
    nd["rebound"] = (~no_ref) & (nd.dt_last_own_shot <= float(c0["rebound_window_s"])).fillna(False).values
    # вход в зону своей команды: последняя метка entry_success той же команды в том же матче,
    # периоде и отрезке, стоящая в seq РАНЬШЕ первого события узла; окно k секунд от t_ref.
    # Привязка меток к ближайшему узлу по времени здесь не используется (она зависит от отметки
    # самого узла).
    lab_e = e[(e.event_type == "entry_success") & (e.period > 0) & e.actor_team_id.notna()]
    t_lab = np.full(len(nd), np.nan)
    if len(lab_e) and len(nd):          # матч/порция без меток входа — все recent_entry = False
        # Use the same barrier epochs for labels and nodes. An old entry must
        # stay cleared after the first post-gap node, not reappear on the next.
        barrier_keys = set(zip(barriers_all.match_id, barriers_all.seq))
        is_barrier = np.array([(m, q) in barrier_keys for m, q in zip(e.match_id, e.seq)], dtype=int)
        event_epoch = pd.Series(is_barrier, index=e.index).groupby(e.segment).cumsum()
        epoch_by_seq = pd.Series(event_epoch.to_numpy(), index=pd.MultiIndex.from_arrays([e.match_id, e.seq]))
        lab_epoch = epoch_by_seq.reindex(pd.MultiIndex.from_arrays([lab_e.match_id, lab_e.seq])).to_numpy()
        node_epoch = epoch_by_seq.reindex(pd.MultiIndex.from_arrays([nd.match_id, nd.first_seq])).to_numpy()
        kl = make_key(lab_e.match_id, lab_e.period, lab_e.segment, lab_e.actor_team_id, lab_epoch)
        kn = make_key(nd.match_id, nd.period, nd.segment, nd.team_id, node_epoch)
        L = pd.DataFrame({"k": kl, "seq": lab_e.seq.values.astype(np.int64),
                          "t_lab": lab_e.t_mono.values.astype(float)}).sort_values(["seq", "k"], kind="mergesort")
        R = pd.DataFrame({"k": kn, "seq": nd.first_seq.values.astype(np.int64),
                          "i": np.arange(len(nd))}).sort_values(["seq", "k"], kind="mergesort")
        mm = pd.merge_asof(R, L, on="seq", by="k", direction="backward", allow_exact_matches=False)
        t_lab[mm.i.values] = mm.t_lab.values
    nd["t_last_entry"] = t_lab
    nd["recent_entry"] = (~no_ref) & (~np.isnan(t_lab)) & ~barrier_before & \
        ((np.nan_to_num(t_ref, nan=0.0) - np.nan_to_num(t_lab, nan=-1e9)) <= float(c0["entry_window_s"]))
    # неопределённость цепи (P008): между предыдущим узлом отрезка (или началом отрезка) и этим
    # узлом было событие с шайбой без исполнителя и без валидной команды
    gap = e.unatt_gap.to_numpy().astype(int)
    gcum = pd.Series(gap).groupby(e.match_id.to_numpy()).cumsum().to_numpy()   # включительно
    pos = pd.Series(np.arange(len(e)), index=pd.MultiIndex.from_arrays([e.match_id.to_numpy(), e.seq.to_numpy()]))
    npos = pos.reindex(pd.MultiIndex.from_arrays([nd.match_id.to_numpy(), nd.first_seq.to_numpy()])).to_numpy()
    seg_first_pos = pd.Series(np.arange(len(e))).groupby(e.segment.to_numpy()).transform("min").to_numpy()
    g_now = gcum[npos]
    g_prev = np.where(has_prev, g_now[pv], 0)
    sp = seg_first_pos[npos]
    g_seg0 = gcum[sp] - gap[sp]
    nd["chain_gap_before"] = barrier_before | np.where(has_prev, g_now - g_prev > 0, g_now - g_seg0 > 0)

    # --- исход (только для проверки, не признак состояния) -------------------------------
    nd["is_shot"] = nd.action == "shot"
    nd["is_goal"] = nd.primary == "goal"
    nd["blocked"] = nd.primary == "shot_blocked"
    ecols = ["match_id", "seq", "event_type", "raw_type", "actor_id", "actor_team_id", "other_id", "other_team_id",
             "period", "clock_s", "zone", "x", "y", "x2", "y2", "n_home", "n_away", "scheme", "stint", "segment",
             "penalty_shot_seg", "score_home_before", "score_away_before", "home_team_id", "away_team_id",
             "rink_type", "t_mono", "t_behind_scale", "clock_imputed", "clock_impute_kind"]
    return nd, excl, e[[c for c in ecols if c in e.columns]]


def build_possessions(nd: pd.DataFrame) -> pd.DataFrame:
    g = nd.groupby("possession", sort=False)
    ps = pd.DataFrame({
        "match_id": g.match_id.first(), "segment": g.segment.first(), "team_id": g.team_id.first(),
        "period": g.period.first(), "t_start": g.t_mono.min(), "n_nodes": g.size(),
        "start_arrival": g.arrival.first(), "start_x": g.x.first(),
        "n_shots": g.is_shot.sum(), "n_goals": g.is_goal.sum(),
    })
    # конец владения: начало следующего владения в том же отрезке, иначе ОСТАНОВКА отрезка
    # (свисток, гол, конец периода — явная граница seg_stop), а не время последнего действия;
    # все границы — по монотонной шкале t_mono, поэтому владения не перекрываются и точно
    # покрывают отрезок
    nxt_start = ps.t_start.shift(-1)
    same_seg = ps.segment.shift(-1) == ps.segment
    ps["last_action"] = g.t_mono.max()
    ps["seg_stop"] = g.seg_stop.first()
    ps["t_end"] = np.where(same_seg, nxt_start, ps.seg_stop)
    ps["t_end"] = np.maximum(ps.t_end, ps.t_start)
    ps["duration"] = ps.t_end - ps.t_start
    last = g.tail(1).set_index("possession")
    unknown = ~last.valid_for_transition.reindex(ps.index)
    ps["duration_observed"] = ~unknown
    ps["extended_to_stop"] = (~same_seg) & ~unknown & (ps.seg_stop > ps.last_action + 1e-9)
    ps["end_reason"] = np.where(unknown, "observation_gap",
        np.where(last.sigma_next.reindex(ps.index) == -1, "lost",
                 np.where(last.is_goal.reindex(ps.index), "goal", "stoppage")))
    return ps.reset_index()


def lineup_roster(ev: pd.DataFrame) -> pd.DataFrame:
    """Заявки матчей из событий lineup_<амплуа>_<звено>: match_id, team_id, player_id, slot, line."""
    lu = ev[ev.event_type.str.startswith("lineup_")].copy()
    parts = lu.event_type.str.split("_", expand=True)
    lu["slot"] = parts[1]
    lu["line"] = parts[2].astype(int)
    out = lu[["match_id", "actor_team_id", "actor_id", "slot", "line"]].rename(
        columns={"actor_team_id": "team_id", "actor_id": "player_id"})
    empty = out.player_id.isna()
    out = out[~empty].drop_duplicates(["match_id", "team_id", "player_id", "slot"])
    return out.reset_index(drop=True)
