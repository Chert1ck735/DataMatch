"""L0: восстановление составов на льду, отрезков с неизменным составом (stints) и TOI.

Правила (зафиксированы, см. docs/METHODOLOGY.md):
* Порядок обработки — seq (канонический). Событие относится к составу, который
  действует в момент его позиции в потоке seq. Поэтому при одинаковом clock_s
  действие, стоящее в seq ДО замены, приписывается старому составу, ПОСЛЕ — новому.
* shift_*: actor_id — вышел на лёд, other_id — ушёл. Внутри одной записи сначала
  обрабатывается уход, затем выход.
* shift_G — слот вратаря. Если в слот G выходит полевой (не вратарь по заявке/справочнику),
  это снятие вратаря: полевой добавляется к полевым, вратаря на льду нет (пустые ворота).
  Возврат вратаря — shift_G с вратарём в actor и полевым в other.
* shift_F1..F3, shift_D1..D2 — полевые.
* Смена периода: составы закрываются в момент конца периода (максимальный clock_s
  периода) и очищаются; оставшиеся на льду без записи об уходе логируются.
* Период 0 (буллиты) не входит ни в отрезки, ни в TOI.
* Длительность отрезка = (время начала следующего отрезка) − (время начала);
  время — единая расчётная шкала периода t_mono (l0_time: накопленный максимум clock_s
  событий игры по seq), та же, что у действий и владений, поэтому время действия всегда
  лежит в границах его состава; отрицательные Δt обрезаются до 0.
* Каждое изменение состава ЛЮБОЙ команды открывает новый отрезок; отрезки нулевой
  длительности (серия замен в одну секунду) сохраняются, но TOI не дают.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

SHIFT_TYPES = {"shift_G", "shift_F1", "shift_F2", "shift_F3", "shift_D1", "shift_D2"}


def reconstruct_match(m: pd.DataFrame, home: str, away: str, max_skaters: int = 6, goalies: set | None = None):
    """m — события одного матча (без lineup_*), отсортированные по seq.

    Возвращает (stint_idx по строкам m, список отрезков, счётчик аномалий).
    """
    def _obj(col):
        v = col.to_numpy(dtype=object)
        return np.array([z if isinstance(z, str) else None for z in v], dtype=object)

    et = m.event_type.to_numpy(dtype=object)
    actor = _obj(m.actor_id)
    other = _obj(m.other_id)
    ateam = _obj(m.actor_team_id)
    oteam = _obj(m.other_team_id)
    period = m.period.values
    clock = m.clock_s.values.astype(float)
    if "t_mono" in m:
        tmono = m.t_mono.values.astype(float)
    else:
        from .l0_time import compute_t_mono
        mm = m.copy()
        mm["clock_s"] = mm.clock_s.astype(float).fillna(0.0)
        tmono = compute_t_mono(mm)
    seqv = m.seq.values.astype(np.int64)
    n = len(m)

    # наблюдаемый конец периода — максимум расчётной шкалы периода
    pend = {}
    for p in np.unique(period):
        c = tmono[period == p]
        c = c[~np.isnan(c)]
        pend[p] = float(c.max()) if len(c) else 0.0
    # граница seq конца периода (исключительная): последний seq периода + 1
    pseq_end = {p: int(seqv[period == p].max()) + 1 for p in np.unique(period)}

    goalies = goalies or set()
    sk = {home: set(), away: set()}
    gk = {home: None, away: None}
    anomalies = Counter()
    stints = []
    stint_idx = np.full(n, -1, dtype=np.int64)
    cur = -1
    cur_period = None
    t_run = 0.0

    def open_stint(t, p, q):
        nonlocal cur
        cur += 1
        stints.append({
            "stint": cur, "period": int(p), "t_start": float(t), "seq_start": int(q),
            "home_sk": tuple(sorted(sk[home])), "away_sk": tuple(sorted(sk[away])),
            "home_g": gk[home], "away_g": gk[away],
        })

    for i in range(n):
        p = period[i]
        if p == 0:
            stint_idx[i] = -1
            continue
        if p != cur_period:
            if cur_period is not None:
                left = len(sk[home]) + len(sk[away]) + (gk[home] is not None) + (gk[away] is not None)
                if left:
                    anomalies["carryover_cleared_at_period_end"] += left
            sk = {home: set(), away: set()}
            gk = {home: None, away: None}
            cur_period = p
            t_run = 0.0
            open_stint(0.0, p, seqv[i])
        e = et[i]
        if e in SHIFT_TYPES:
            ci = clock[i]
            if not np.isnan(ci) and ci < tmono[i] - 1e-9:
                anomalies["shift_clock_behind_scale"] += 1
            # время начала нового состава — расчётная шкала на позиции замены
            t_run = max(t_run, tmono[i])
            is_g = e == "shift_G"
            changed = False
            # уход
            o = other[i]
            if o is not None:
                tm = oteam[i] if oteam[i] is not None else ateam[i]
                if tm not in sk:
                    anomalies["shift_unknown_team"] += 1
                elif is_g:
                    if gk[tm] == o:
                        gk[tm] = None
                        changed = True
                    elif o in sk[tm]:
                        sk[tm].discard(o)          # лишний полевой уходит через слот G
                        anomalies["extra_attacker_leave"] += 1
                        changed = True
                    else:
                        anomalies["leave_not_on_ice"] += 1
                else:
                    if o in sk[tm]:
                        sk[tm].discard(o)
                        changed = True
                    elif gk[tm] == o:
                        gk[tm] = None
                        anomalies["skater_slot_leave_was_goalie"] += 1
                        changed = True
                    else:
                        anomalies["leave_not_on_ice"] += 1
            # выход
            a = actor[i]
            if a is not None:
                tm = ateam[i]
                if tm not in sk:
                    anomalies["shift_unknown_team"] += 1
                elif is_g and a not in goalies:
                    # полевой выходит через слот G: снятие вратаря (лишний полевой)
                    if a in sk[tm]:
                        anomalies["enter_already_on_ice"] += 1
                    else:
                        sk[tm].add(a)
                        changed = True
                        anomalies["extra_attacker_enter"] += 1
                        if len(sk[tm]) > max_skaters:
                            anomalies["over_capacity"] += 1
                elif is_g:
                    if gk[tm] == a:
                        anomalies["enter_already_on_ice"] += 1
                    else:
                        if gk[tm] is not None:
                            anomalies["goalie_replaced_without_leave"] += 1
                        if a in sk[tm]:
                            sk[tm].discard(a)
                            anomalies["goalie_enter_was_skater"] += 1
                        gk[tm] = a
                        changed = True
                else:
                    if a in sk[tm]:
                        anomalies["enter_already_on_ice"] += 1
                    else:
                        if gk[tm] == a:
                            gk[tm] = None
                            anomalies["skater_enter_was_goalie"] += 1
                        sk[tm].add(a)
                        changed = True
                        if len(sk[tm]) > max_skaters:
                            anomalies["over_capacity"] += 1
            if changed:
                open_stint(t_run, p, seqv[i])
        stint_idx[i] = cur

    # длительности
    for j, s in enumerate(stints):
        nxt = stints[j + 1] if j + 1 < len(stints) else None
        end = nxt["t_start"] if (nxt is not None and nxt["period"] == s["period"]) else pend.get(s["period"], s["t_start"])
        s["t_end"] = max(end, s["t_start"])
        s["duration"] = s["t_end"] - s["t_start"]
        # принадлежность события составу: [seq_start, seq_end); событие замены с seq = q уже
        # относится к новому составу (ушедший в нём отсутствует, вышедший — присутствует)
        s["seq_end"] = (nxt["seq_start"] if (nxt is not None and nxt["period"] == s["period"])
                        else pseq_end.get(s["period"], s["seq_start"] + 1))
    return stint_idx, stints, anomalies


def match_goalie_set(mid: str, goalies_by_match: dict | None, goalies_global: set | None,
                     skaters_by_match: dict | None) -> set:
    """Вратари КОНКРЕТНОГО матча: заявленные в слот G этого матча плюс вратари по справочнику,
    если они не заявлены в этом матче полевыми (полевой с position=G, выходящий через слот G,
    — снятие вратаря, а не вратарь)."""
    lineup_g = set((goalies_by_match or {}).get(mid, set()))
    lineup_sk = set((skaters_by_match or {}).get(mid, set()))
    return lineup_g | (set(goalies_global or set()) - lineup_sk)


def reconstruct_all(ev: pd.DataFrame, games: pd.DataFrame, max_skaters: int = 6, logger=None,
                    goalies_by_match: dict | None = None, goalies_global: set | None = None,
                    skaters_by_match: dict | None = None):
    """Восстановление по всем матчам. ev — события без lineup_*, отсортированы по match_id, seq."""
    gh = games.set_index("match_id")[["home_team_id", "away_team_id"]]
    stint_col = np.full(len(ev), -1, dtype=np.int64)
    all_stints = []
    anomalies_total = Counter()
    anomalies_by_match = {}
    starts = np.r_[0, np.flatnonzero(ev.match_id.values[1:] != ev.match_id.values[:-1]) + 1, len(ev)]
    for k in range(len(starts) - 1):
        a, b = starts[k], starts[k + 1]
        m = ev.iloc[a:b]
        mid = m.match_id.iat[0]
        home, away = gh.loc[mid, "home_team_id"], gh.loc[mid, "away_team_id"]
        gset = match_goalie_set(mid, goalies_by_match, goalies_global, skaters_by_match)
        idx, st, an = reconstruct_match(m, home, away, max_skaters, gset)
        stint_col[a:b] = idx
        for s in st:
            s["max_skaters"] = max_skaters
            s["match_id"] = mid
            s["home_team_id"] = home
            s["away_team_id"] = away
        all_stints.extend(st)
        anomalies_total.update(an)
        if an:
            anomalies_by_match[mid] = dict(an)
    stints = pd.DataFrame(all_stints)
    stints["n_home_sk"] = stints.home_sk.map(len)
    stints["n_away_sk"] = stints.away_sk.map(len)
    stints["home_g_in"] = stints.home_g.notna()
    stints["away_g_in"] = stints.away_g.notna()
    stints["is_5v5"] = (stints.n_home_sk == 5) & (stints.n_away_sk == 5) & stints.home_g_in & stints.away_g_in
    stints["stint_key"] = stints.match_id + ":" + stints.stint.astype(str)
    if logger:
        logger.info("L0 составы: %d отрезков, аномалии %s", len(stints), dict(anomalies_total))
    return stint_col, stints, anomalies_total, anomalies_by_match


def player_toi(stints: pd.DataFrame) -> pd.DataFrame:
    """TOI полевых игроков по матчам (секунды), всего и при 5 на 5, + по ситуациям.

    toi_valid_s — диагностические секунды надёжного состава.
    toi_rate_s — знаменатель rates, дополнительно исключающий недостаточно покрытые матчи."""
    rows = []
    stints = stints.copy()
    if "valid_for_rates" not in stints:
        stints["valid_for_rates"] = True
    if "valid_for_rate_metrics" not in stints:
        stints["valid_for_rate_metrics"] = False  # no match policy: no approved denominator
    for side, team_col, opp_n, own_n, own_g, opp_g in [
        ("home_sk", "home_team_id", "n_away_sk", "n_home_sk", "home_g_in", "away_g_in"),
        ("away_sk", "away_team_id", "n_home_sk", "n_away_sk", "away_g_in", "home_g_in"),
    ]:
        s = stints[["match_id", team_col, side, "duration", "is_5v5", own_n, opp_n, own_g, opp_g,
                    "valid_for_rates", "valid_for_rate_metrics"]].copy()
        s = s[s.duration > 0]
        s = s.explode(side).dropna(subset=[side])
        s = s.rename(columns={side: "player_id", team_col: "team_id", own_n: "own_n", opp_n: "opp_n",
                              own_g: "own_g", opp_g: "opp_g"})
        rows.append(s)
    s = pd.concat(rows, ignore_index=True)
    s["toi_5v5"] = np.where(s.is_5v5, s.duration, 0.0)
    pp = (s.own_n > s.opp_n) & s.own_g & s.opp_g
    sh = (s.own_n < s.opp_n) & s.own_g & s.opp_g
    s["toi_pp"] = np.where(pp, s.duration, 0.0)
    s["toi_sh"] = np.where(sh, s.duration, 0.0)
    s["toi_valid"] = np.where(s.valid_for_rates.astype(bool), s.duration, 0.0)
    s["toi_rate"] = np.where(s.valid_for_rate_metrics.astype(bool), s.duration, 0.0)
    s["toi_rate_5v5"] = np.where(s.is_5v5, s.toi_rate, 0.0)
    s["toi_rate_pp"] = np.where(pp, s.toi_rate, 0.0)
    s["toi_rate_sh"] = np.where(sh, s.toi_rate, 0.0)
    g = s.groupby(["match_id", "team_id", "player_id"], as_index=False).agg(
        toi_s=("duration", "sum"), toi_5v5_s=("toi_5v5", "sum"), toi_pp_s=("toi_pp", "sum"), toi_sh_s=("toi_sh", "sum"),
        toi_valid_s=("toi_valid", "sum"), toi_rate_s=("toi_rate", "sum"),
        toi_rate_5v5_s=("toi_rate_5v5", "sum"), toi_rate_pp_s=("toi_rate_pp", "sum"),
        toi_rate_sh_s=("toi_rate_sh", "sum"))
    return g


def build_player_shifts(stints: pd.DataFrame) -> pd.DataFrame:
    """Индивидуальные смены полевых игроков из восстановленных составов (одна запись — одна
    непрерывная смена игрока).

    Последовательный проход по stints матча: игрок появился в наборе полевых своей команды —
    смена открывается (seq_start, t_start нового stint); исчез — закрывается (seq_end, t_end =
    начало stint, в котором его нет); остался — смена продолжается. Замены партнёров и
    соперников, свистки, голы и вбрасывания сами смену не завершают. Конец периода закрывает
    все смены (составы очищаются правилом L0). Вратари в наборах полевых отсутствуют; лишний
    полевой, вышедший через слот G, в них есть. Длительность — на той же шкале, что stints,
    поэтому сумма смен игрока равна его TOI.
    """
    rows = []
    for (mid, per), grp in stints.groupby(["match_id", "period"], sort=False):
        grp = grp.sort_values("stint")
        teams = {"home": grp.home_team_id.iat[0], "away": grp.away_team_id.iat[0]}
        active = {"home": {}, "away": {}}
        last_seq_end = int(grp.seq_end.iat[-1])
        last_t_end = float(grp.t_end.iat[-1])
        for r in grp.itertuples(index=False):
            for side, col in (("home", "home_sk"), ("away", "away_sk")):
                cur = set(getattr(r, col))
                act = active[side]
                for pid in [q for q in act if q not in cur]:
                    sh = act.pop(pid)
                    sh["seq_end"], sh["t_end"] = int(r.seq_start), float(r.t_start)
                    rows.append(sh)
                for pid in cur:
                    if pid not in act:
                        act[pid] = {"match_id": mid, "team_id": teams[side], "player_id": pid, "period": int(per),
                                    "seq_start": int(r.seq_start), "t_start": float(r.t_start)}
        for side in ("home", "away"):
            for sh in active[side].values():
                sh["seq_end"], sh["t_end"] = last_seq_end, last_t_end
                rows.append(sh)
    cols = ["match_id", "team_id", "player_id", "period", "seq_start", "seq_end", "t_start", "t_end"]
    ps = pd.DataFrame(rows, columns=cols)
    ps["duration_s"] = ps.t_end - ps.t_start
    ps["player_shift_id"] = (ps.match_id.astype(str) + ":" + ps.team_id.astype(str) + ":" + ps.player_id.astype(str)
                             + ":" + ps.period.astype(str) + ":" + ps.seq_start.astype(str))
    return ps.sort_values(["match_id", "seq_start", "team_id", "player_id"], kind="mergesort").reset_index(drop=True)


OBS_EXCLUDE_PREFIX = ("shift_",)
OBS_EXCLUDE = {"period_start", "intermission", "game_end", "ot_start", "so_start", "so_attempt",
               "entry_success", "exit_success", "exit_controlled", "feed_break", "assist_primary",
               "assist_secondary", "goal_error", "scoring_chance", "penalty_return"}


def structurally_possible_lineups(stints: pd.DataFrame, max_skaters: int = 6) -> np.ndarray:
    """Physical bounds and identities, independent of observation and published flags.

    At least three skaters per side, bounded by the configured skater cap; hockey
    allows at most six total players (sixth skater replaces the goalie).
    Identities cannot occupy two slots or sides.
    """
    st = stints
    cap = st.get("max_skaters", pd.Series(max_skaters, index=st.index))
    possible = ((st.n_home_sk >= 3) & (st.n_away_sk >= 3)
        & (st.n_home_sk <= cap) & (st.n_away_sk <= cap)
        & (st.n_home_sk + st.home_g_in.astype(int) <= 6)
        & (st.n_away_sk + st.away_g_in.astype(int) <= 6)).to_numpy(copy=True)
    for i, r in enumerate(st[["home_sk", "away_sk", "home_g", "away_g",
            "home_g_in", "away_g_in", "n_home_sk", "n_away_sk"]].itertuples(index=False)):
        if not possible[i]:
            continue
        h, a = list(r.home_sk), list(r.away_sk)
        if r.home_g_in:
            h.append(r.home_g)
        if r.away_g_in:
            a.append(r.away_g)
        possible[i] = (len(set(h)) == len(h) and len(set(a)) == len(a)
                      and not set(h).intersection(a)
                      and len(r.home_sk) == r.n_home_sk and len(r.away_sk) == r.n_away_sk)
    return possible


def annotate_lineup_status(stints: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Статус состава каждого отрезка (P004): восстановленная численность сверяется с
    наблюдаемой n_home/n_away игровых событий этого отрезка (независимый источник).

    Статусы: zero_duration; empty (у обеих команд нет полевых); empty_team (у одной);
    insufficient (меньше трёх полевых); unverified (наблюдений нет, численность правдоподобна);
    valid (совпадает с наблюдаемой); mismatch (расходится). Легальные 5×4, 3×3 и снятый вратарь
    — valid, если совпадают с наблюдениями: знаменатель «10 полевых» не используется.

    ``valid_for_rates`` означает пригодность *временной экспозиции*, поэтому нулевой
    отрезок в него не входит. ``valid_for_action_attribution`` независим: действие между
    заменами с одинаковой отметкой времени можно надёжно отнести к составу по seq (P004).
    """
    st = stints.copy()
    ev = events[(events.stint >= 0) & events.n_home.notna() & events.n_away.notna()
                & ~events.event_type.isin(OBS_EXCLUDE)
                & ~events.event_type.astype(str).str.startswith(OBS_EXCLUDE_PREFIX)]
    obs = (ev.groupby(["match_id", "stint", "n_home", "n_away"]).size().rename("n").reset_index()
           .sort_values(["match_id", "stint", "n"], ascending=[True, True, False], kind="mergesort")
           .groupby(["match_id", "stint"], sort=False).head(1).set_index(["match_id", "stint"]))
    n_obs = ev.groupby(["match_id", "stint"]).size()
    key = pd.MultiIndex.from_arrays([st.match_id, st.stint])
    st["obs_n_home"] = obs.n_home.reindex(key).values
    st["obs_n_away"] = obs.n_away.reindex(key).values
    st["obs_events"] = n_obs.reindex(key).fillna(0).astype(int).values
    rec_h = st.n_home_sk + st.home_g_in.astype(int)
    rec_a = st.n_away_sk + st.away_g_in.astype(int)
    has_obs = st.obs_events > 0
    agree = has_obs & (st.obs_n_home.values == rec_h.values) & (st.obs_n_away.values == rec_a.values)
    structurally_possible = structurally_possible_lineups(st)
    status = np.select(
        [st.duration <= 0,
         (st.n_home_sk == 0) & (st.n_away_sk == 0),
         (st.n_home_sk == 0) | (st.n_away_sk == 0),
         (st.n_home_sk < 3) | (st.n_away_sk < 3),
         ~structurally_possible, ~has_obs, agree],
        ["zero_duration", "empty", "empty_team", "insufficient", "impossible", "unverified", "valid"], default="mismatch")
    st["lineup_status"] = status
    st["valid_for_rates"] = np.isin(status, ["valid", "unverified"])
    st["valid_for_action_attribution"] = structurally_possible & (~has_obs | agree)
    return st


def periods_table(events: pd.DataFrame, games: pd.DataFrame, tol_s: float = 1.0) -> pd.DataFrame:
    """Периоды (P003): наблюдаемый конец (максимум расчётной шкалы), время события конца
    периода, номинальная длительность по регламенту (регулярный — 1200 с; овертайм — 1200 с в
    плей-офф, иначе 300 с) и статус. Номинал известен заранее; фактический конец периода в
    признаки до действия не подставляется."""
    ev = events[events.period > 0]
    g = ev.groupby(["match_id", "period"])
    out = pd.DataFrame({"observed_end": g.t_mono.max(), "last_clock": g.clock_s.max()})
    ends = ev[ev.event_type.isin(["intermission", "game_end"])].groupby(["match_id", "period"]).clock_s.max()
    out["end_event_clock"] = ends.reindex(out.index)
    goals = ev[ev.event_type == "goal"].groupby(["match_id", "period"]).size()
    out["n_goals"] = goals.reindex(out.index).fillna(0).astype(int)
    out = out.reset_index()
    stage = out.match_id.map(games.set_index("match_id").stage)
    out["nominal"] = np.where(out.period <= 3, 1200.0, np.where(stage == "playoff", 1200.0, 300.0))
    ot = out.period >= 4
    over = out.observed_end > out.nominal + tol_s
    short = out.observed_end < out.nominal - tol_s
    out["status"] = np.select(
        [over, ot & short & (out.n_goals > 0), short],
        ["overlong", "ot_ended_by_goal", "truncated"], default="complete")
    return out
