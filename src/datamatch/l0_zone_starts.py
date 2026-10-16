"""L0: тип и зона старта индивидуальной смены игрока (PDF: «зона старта смены — зона
вбрасывания, открывающего смену»).

Последовательный проход по ПРЕДОБРАБОТАННОМУ потоку событий матча в порядке seq (вбрасывания
без исполнителя тоже учитываются) с состоянием игры:
* stopped — подтверждённая остановка: после свистка, гола, конца периода/матча и в начале
  периода, открытого событием начала периода, заменой или вбрасыванием;
* live — игра идёт: после вбрасывания или игрового действия;
* unknown — период начинается не с начала периода/замены/вбрасывания (данные с середины игры).

Смена, начавшаяся (seq_start) при stopped, ждёт открывающего вбрасывания. На вбрасывании с
seq = f ожидающие, ещё находящиеся на льду (seq_start ≤ f < seq_end), получают
start_type = faceoff, faceoff_seq = f и зону этого вбрасывания. Ожидание не переносится:
* игрок ушёл до вбрасывания -> unknown, причина left_before_faceoff;
* конец периода -> unknown, period_end_before_faceoff;
* игра возобновилась без зарегистрированного вбрасывания -> unknown, play_resumed_without_faceoff.
Смена, начавшаяся при live, — on_the_fly (зона пустая); при unknown — unknown, game_state_unknown.
Окна по абсолютному времени не используются: только порядок seq.

Зона — с точки зрения команды игрока: x вбрасывания задан в системе команды исполнителя;
если команда события — команда игрока, x_team = x, если соперник — x_team = −x. Неизвестная
или чужая для матча команда события -> зона пустая (faceoff_team_unknown), не «соперник»;
нет x -> faceoff_xy_missing. Победитель вбрасывания не влияет на зону. Определённая зона
сохраняется до конца индивидуальной смены (последующие вбрасывания её не меняют).
Отрезки штрафных бросков не считаются возобновлением игры.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .l0_events import BREAK_AFTER, PUCK_PRIMARY, zone_of_x

PLAY_TYPES = (PUCK_PRIMARY - {"faceoff"}) | {"duel", "body_hit", "stick_check"}
PERIOD_OPENERS = {"period_start", "ot_start", "faceoff"}


def assign_starts(events: pd.DataFrame, shifts: pd.DataFrame, geo: dict) -> pd.DataFrame:
    """events — предобработанные события (с period, seq, event_type, actor_team_id, x, home/away,
    penalty_shot_seg, rink_type); shifts — build_player_shifts; geo — io.rink_geometry
    (синие линии по типу площадки). Возвращает shifts с полями старта."""
    out = shifts.copy()
    n = len(out)
    st_type = np.full(n, None, dtype=object)
    st_zone = np.full(n, None, dtype=object)
    fo_seq = np.full(n, -1, dtype=np.int64)
    reason = np.full(n, None, dtype=object)
    if n == 0:
        for c, v in [("start_type", st_type), ("start_zone", st_zone), ("faceoff_seq", fo_seq), ("unknown_reason", reason)]:
            out[c] = v
        return out
    idx_by_match = {m: g.index.values for m, g in out.groupby("match_id", sort=False)}
    ev_m = events[events.period > 0]
    for mid, ge in ev_m.groupby("match_id", sort=False):
        if mid not in idx_by_match:
            continue
        rows = idx_by_match[mid]
        sh = out.loc[rows]
        starts = {}                                   # seq_start -> индексы смен
        for i, q in zip(rows, sh.seq_start.values):
            starts.setdefault(int(q), []).append(i)
        seq_end = out.seq_end.values
        team_of = out.team_id.values
        ge = ge.sort_values("seq", kind="mergesort")
        home, away = ge.home_team_id.iat[0], ge.away_team_id.iat[0]
        teams = {home, away}
        rink = str(ge.rink_type.iat[0]) if "rink_type" in ge else None
        blue = geo[rink]["blue_line_opp"] if rink in geo else None
        seqs = ge.seq.values.astype(np.int64)
        et = ge.event_type.to_numpy(dtype=object)
        per = ge.period.values
        ateam = ge.actor_team_id.to_numpy(dtype=object)
        xs = ge.x.values.astype(float)
        pss = ge.penalty_shot_seg.values.astype(bool) if "penalty_shot_seg" in ge else np.zeros(len(ge), bool)
        state, cur_per = None, None
        prev_q = None
        pending: list[int] = []

        def fail_pending(why, at_seq):
            for k in pending:
                if seq_end[k] <= at_seq:
                    st_type[k], reason[k] = "unknown", "left_before_faceoff"
                else:
                    st_type[k], reason[k] = "unknown", why
            pending.clear()

        for j in range(len(ge)):
            q, t = seqs[j], et[j]
            if per[j] != cur_per:
                if cur_per is not None:
                    # смены, закрытые границей периода (seq_end = последний seq + 1), — period_end;
                    # закрытые раньше последнего события периода — ушли до вбрасывания
                    fail_pending("period_end_before_faceoff", prev_q)
                cur_per = per[j]
                state = "stopped" if (t in PERIOD_OPENERS or str(t).startswith("shift_")) else "unknown"
            # смены, начавшиеся на этом событии (событие замены), — по состоянию ДО него
            for k in starts.get(int(q), ()):
                if state == "stopped":
                    pending.append(k)
                elif state == "live":
                    st_type[k] = "on_the_fly"
                else:
                    st_type[k], reason[k] = "unknown", "game_state_unknown"
            if t == "faceoff" and not pss[j]:
                fteam = ateam[j] if isinstance(ateam[j], str) else None
                for k in pending:
                    if seq_end[k] <= q:
                        st_type[k], reason[k] = "unknown", "left_before_faceoff"
                        continue
                    st_type[k], fo_seq[k] = "faceoff", q
                    if fteam is None or fteam not in teams:
                        reason[k] = "faceoff_team_unknown"
                    elif np.isnan(xs[j]):
                        reason[k] = "faceoff_xy_missing"
                    elif blue is None:
                        reason[k] = "rink_unknown"
                    else:
                        xt = xs[j] if fteam == team_of[k] else -xs[j]
                        st_zone[k] = zone_of_x(xt, blue)
                pending.clear()
                state = "live"
            else:
                # событие может иметь две роли (обычный гол — и игровое действие, и остановка):
                # сначала роль игрового действия (возобновление без вбрасывания завершает
                # ожидание), затем остановка (P015). Штрафные броски возобновлением не считаются.
                if t in PLAY_TYPES and not pss[j]:
                    if state == "stopped" and pending:
                        fail_pending("play_resumed_without_faceoff", q)
                    state = "live"
                if t in BREAK_AFTER:
                    state = "stopped"
            prev_q = q
        if len(seqs):
            fail_pending("period_end_before_faceoff", seqs[-1])
    # смены без событий начала в потоке (не должно случаться) — явная причина
    miss = np.array([v is None for v in st_type])
    st_type[miss] = "unknown"
    reason[miss & np.array([r is None for r in reason])] = "start_event_not_found"
    out["start_type"] = st_type
    out["start_zone"] = st_zone
    out["faceoff_seq"] = fo_seq
    out["unknown_reason"] = reason
    return out


def link_nodes(nodes: pd.DataFrame, shifts: pd.DataFrame) -> pd.DataFrame:
    """Связь узлов (действий) с индивидуальными сменами по матчу, команде, игроку, периоду и
    интервалу seq [seq_start, seq_end). Зона старта присоединяется только начиная с faceoff_seq."""
    from .l0_events import make_key
    nd = nodes
    out = pd.DataFrame(index=nd.index)
    out["player_shift_id"] = None
    out["shift_start_type"] = None
    out["shift_start_zone"] = None
    out["shift_link"] = "no_shift"
    if len(nd) == 0 or len(shifts) == 0:
        return out
    kn = make_key(nd.match_id, nd.team_id, nd.actor_id, nd.period)
    ks = make_key(shifts.match_id, shifts.team_id, shifts.player_id, shifts.period)
    L = pd.DataFrame({"k": ks, "seq": shifts.seq_start.values.astype(np.int64), "s": np.arange(len(shifts))}
                     ).sort_values(["seq", "k"], kind="mergesort")
    R = pd.DataFrame({"k": kn, "seq": nd.first_seq.values.astype(np.int64), "i": np.arange(len(nd))}
                     ).sort_values(["seq", "k"], kind="mergesort")
    mm = pd.merge_asof(R, L, on="seq", by="k", direction="backward")
    i = mm.i.values
    s = mm.s.values
    ok = ~np.isnan(s)
    si = np.where(ok, s, 0).astype(int)
    inside = ok & (nd.first_seq.values[i] < shifts.seq_end.values[si])
    link = np.where(inside, "linked", "no_shift")
    sid = np.where(inside, shifts.player_shift_id.values[si], None)
    stype = np.where(inside, shifts.start_type.values[si], None)
    fs = shifts.faceoff_seq.values[si]
    zone_ok = inside & (fs >= 0) & (nd.first_seq.values[i] >= fs)
    szone = np.where(zone_ok, shifts.start_zone.values[si], None)
    out.iloc[i, out.columns.get_loc("player_shift_id")] = sid
    out.iloc[i, out.columns.get_loc("shift_start_type")] = stype
    out.iloc[i, out.columns.get_loc("shift_start_zone")] = szone
    out.iloc[i, out.columns.get_loc("shift_link")] = link
    return out
