"""Этап L0 целиком: события -> actions(узлы), possessions, stints, player_toi, отчёт качества.

Обработка порциями матчей (l0.chunk_matches): события читаются фильтром по матчам, каждая
порция проходит предобработку, реконструкцию составов и построение узлов; номера узлов,
отрезков и владений сдвигаются на накопленное смещение, поэтому цепь глобально согласована.
Все правила L0 действуют внутри матча, так что результат не зависит от размера порции
(проверяется тестом). Пик памяти ограничен одной порцией и итоговыми таблицами.

Гейт L0 обязателен: при провале пишется отчёт со статусом failed, кэш не помечается
готовым и поднимается GateError.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from . import io, l0_events, l0_quality, l0_shifts, l0_time, l0_zone_starts
from .cache import Cache
from .config import Config, get_logger
from .gates import enforce

L0_CODE = ("io", "l0_events", "l0_shifts", "l0_quality", "l0_time", "l0_zone_starts", "stage_l0", "gates")
L0_TABLES = ["games", "events", "stints", "toi", "nodes", "possessions", "roster", "excluded", "player_shifts",
             "periods"]
JS_TYPES = {"pass": {"pass_complete", "pass_failed"}, "shot": l0_events.SHOT_TYPES, "steal": {"steal"}}


def _slot_usage(rosters: list[pd.DataFrame], shifts: list[pd.DataFrame]) -> pd.DataFrame:
    """Фактическое использование игрока: заявки вратарём/полевым, самый частый полевой слот,
    выходы через слот G и через полевые слоты (train + test)."""
    r = pd.concat(rosters, ignore_index=True)
    isg = r.slot == "G"
    n_g = isg.groupby(r.player_id).sum()
    n_s = (~isg).groupby(r.player_id).sum()
    sk = r[~isg]
    sk_mode = sk.slot.groupby(sk.player_id).agg(lambda v: v.value_counts().index[0])
    sh = pd.concat(shifts)
    sh = sh.dropna(subset=["actor_id"])
    ng = (sh.event_type == "shift_G").groupby(sh.actor_id).sum()
    ns = (sh.event_type != "shift_G").groupby(sh.actor_id).sum()
    out = pd.DataFrame({"n_lineup_G": n_g, "n_lineup_skater": n_s})
    out = out.join(sk_mode.rename("skater_slot_mode"), how="left")
    out = out.join(ng.rename("n_shift_G"), how="outer").join(ns.rename("n_shift_skater"), how="outer")
    return out.rename_axis("player_id").reset_index()


def role_conflicts(players: pd.DataFrame, rosters: dict, light: dict) -> dict:
    """Конфликты справочника и фактической роли в обе стороны + выходы через слот G."""
    out = {"players_position_G_plays_skater": int(players.conflict_position_G_plays_skater.sum()),
           "players_position_skater_plays_G": int(players.conflict_position_skater_plays_G.sum())}
    pos = players.set_index("player_id").position
    for s, ros in rosters.items():
        rp = ros.player_id.map(pos)
        a = ros[(rp == "G") & (ros.slot != "G")]
        b = ros[(rp.isin(["C", "LW", "RW", "D"])) & (ros.slot == "G")]
        sg = light[s][(light[s].event_type == "shift_G") & light[s].actor_id.notna()]
        k = sg.merge(ros[["match_id", "player_id", "slot"]], left_on=["match_id", "actor_id"],
                     right_on=["match_id", "player_id"], how="left")
        ext = k[k.slot.notna() & (k.slot != "G")]
        out[s] = {"lineup_rows_position_G_as_skater": int(len(a)), "matches": int(a.match_id.nunique()),
                  "lineup_rows_position_skater_as_G": int(len(b)),
                  "shift_G_entries_by_lineup_skaters": int(len(ext)),
                  "shift_G_entries_by_lineup_skaters_with_position_G": int((ext.player_id.map(pos) == "G").sum()),
                  "matches_with_position_G_extra_attacker": int(ext[ext.player_id.map(pos) == "G"].match_id.nunique())}
    return out


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    """Строковые object-столбцы -> компактный строковый тип (как при чтении из parquet)."""
    for c in df.columns:
        if df[c].dtype == object:
            if df[c].map(lambda z: isinstance(z, (tuple, set, list))).any():
                continue
            df[c] = df[c].astype("str")
    return df


WEIGHTED = {"zone_x_disagreement": "n_nodes_chunk", "zone_labels_attached": "zone_labels"}
DERIVED = {"clock_backward_abs_median_s", "clock_backward_abs_max_s"}


def add_local_keys(nodes, events, excluded):
    """One match-local segment map, including segments without usable nodes."""
    reference = events[['match_id', 'segment']].drop_duplicates().copy()
    reference['segment_local'] = reference.groupby('match_id', sort=False).cumcount()
    mapping = reference.set_index(['match_id', 'segment']).segment_local
    for table in (nodes, events, excluded):
        keys = pd.MultiIndex.from_frame(table[['match_id', 'segment']])
        local = mapping.reindex(keys).to_numpy()
        if pd.isna(local).any():
            raise ValueError('Segment absent from physical event reference')
        table['segment_local'] = local.astype(int)
        table['segment_key'] = table.match_id.astype(str) + ':' + table.segment_local.astype(str)
    nodes['possession_local'] = nodes.groupby('match_id', sort=False).possession.transform(
        lambda x: pd.factorize(x)[0])
    nodes['possession_key'] = nodes.match_id.astype(str) + ':' + nodes.possession_local.astype(str)


def merge_quality(parts: list[dict]) -> dict:
    """Объединение счётчиков качества порций: суммы, словари частот, списки; доли — по весам."""
    out = {}
    for q in parts:
        for k, v in q.items():
            if k in WEIGHTED or k in DERIVED:
                continue
            if isinstance(v, bool):
                out[k] = out.get(k, True) and v
            elif isinstance(v, (int, float, np.integer, np.floating)):
                out[k] = out.get(k, 0) + v
            elif isinstance(v, dict):
                c = Counter(out.get(k, {}))
                c.update(v)
                out[k] = dict(c)
            elif isinstance(v, list):
                out[k] = out.get(k, []) + v
            else:
                out[k] = v
    vals = np.asarray(out.pop("_clock_backward_values", []), float)
    out["clock_backward_abs_median_s"] = float(np.median(vals)) if len(vals) else 0.0
    out["clock_backward_abs_max_s"] = float(vals.max()) if len(vals) else 0.0
    for k, wk in WEIGHTED.items():
        w = np.array([q.get(wk, 0) for q in parts], float)
        z = np.array([q.get(k, 0.0) for q in parts], float)
        out[k] = float((w * z).sum() / max(w.sum(), 1))
    out.pop("n_nodes_chunk", None)
    return out


def process_source(cfg: Config, source: str, games_all: pd.DataFrame, roster: pd.DataFrame, goalies_global: set,
                   geo: dict, log=None, load_chunk=None, match_ids=None) -> tuple[dict, dict, dict]:
    """L0 одного источника порциями. load_chunk(ids) -> события порции (по умолчанию io.load_events)."""
    load_chunk = load_chunk or (lambda ids: io.load_events(cfg, source, matches=ids))
    match_ids = sorted(match_ids if match_ids is not None else io.event_match_ids(cfg, source))
    C = int(cfg["l0"].get("chunk_matches", 100))
    tabs = {k: [] for k in ["games", "events", "stints", "toi", "nodes", "possessions", "excluded", "player_shifts",
                            "periods"]}
    parts = []
    ident = {"n": 0, "on": 0.0, "agree": 0.0, "bad_types": Counter()}
    js_counts = {name: {} for name in JS_TYPES}
    node_off = seg_off = poss_off = 0
    gbm = roster[roster.slot == "G"].groupby("match_id").player_id.agg(set).to_dict()
    sbm = roster[roster.slot != "G"].groupby("match_id").player_id.agg(set).to_dict()
    for i in range(0, len(match_ids), C):
        ids = match_ids[i:i + C]
        ev = load_chunk(ids)
        q = {}
        g = l0_events.fix_games(games_all[games_all.match_id.isin(ids)], ev, q)
        e, scheme, _ = l0_events.preprocess(ev, g, cfg, q)
        del ev
        sc, st, an, abm = l0_shifts.reconstruct_all(e, g, cfg["l0"]["max_skaters"], None, gbm, goalies_global, sbm)
        e["stint"] = sc
        q["shift_anomalies"] = dict(an)
        q["shift_anomaly_matches"] = len(abm)
        nd, excl, es = l0_events.build_nodes(e, g, st, geo, cfg, q)
        del e
        # статус состава отрезков (P004) и TOI с ограничением на надёжные отрезки
        st = l0_shifts.annotate_lineup_status(st, es)
        # Любая метрика «на льду» и её знаменатель обязаны видеть этот флаг. Командные
        # узлы также получают статус, но не участвуют в личных метриках.
        stj_rate = st.set_index(["match_id", "stint"])[["lineup_status", "valid_for_rates", "valid_for_action_attribution"]]
        node_stints = stj_rate.reindex(pd.MultiIndex.from_arrays([nd.match_id, nd.stint]))
        nd["lineup_status"] = node_stints.lineup_status.fillna("missing_stint").to_numpy()
        nd["valid_for_rates"] = node_stints.valid_for_rates.fillna(False).astype(bool).to_numpy()
        nd["valid_for_lineup_attribution"] = node_stints.valid_for_action_attribution.fillna(False).astype(bool).to_numpy()
        per = l0_shifts.periods_table(es, g, float(cfg["l0"].get("period_end_tolerance_s", 1.0)))
        q["n_nodes_chunk"] = int(len(nd))
        # индивидуальные смены, тип/зона старта, связь действий со сменами
        pshift = l0_shifts.build_player_shifts(st)
        pshift = l0_zone_starts.assign_starts(es, pshift, geo)
        lk = l0_zone_starts.link_nodes(nd, pshift)
        stj = st.set_index(["match_id", "stint"]).reindex(pd.MultiIndex.from_arrays([nd.match_id, nd.stint]))
        own_g = np.where(nd.is_home.values, stj.home_g.values, stj.away_g.values)
        own_g_in = np.where(nd.is_home.values, stj.home_g_in.values, stj.away_g_in.values).astype(bool)
        is_gk = (lk.shift_link.values != "linked") & (nd.actor_id.values == own_g) & own_g_in
        lk.loc[is_gk, "shift_link"] = "goalie"
        # командные узлы без исполнителя (P008) — отдельный тип связи, личные проверки к ним не
        # применяются
        lk.loc[(nd.node_kind == "team").values, "shift_link"] = "team_node"
        for c in lk.columns:
            nd[c] = lk[c].values
        nd["actor_on_ice_confirmed"] = nd.shift_link.isin(["linked", "goalie"])
        nd["valid_for_action_attribution"] = nd.valid_for_lineup_attribution & nd.actor_on_ice_confirmed
        st, nd = l0_quality.apply_rate_eligibility(
            st, nd, float(cfg["l0"]["gates"]["rate_match_coverage_min"]))
        toi = l0_shifts.player_toi(st)
        es = es.copy()
        excl = excl.copy()
        add_local_keys(nd, es, excl)
        # глобальные смещения номеров узлов, отрезков и владений
        for c in ["node", "next_node", "prev_node", "last_pass_node"]:
            v = nd[c].values
            nd[c] = np.where(v >= 0, v + node_off, v)
        nd["segment"] = nd.segment.values + seg_off
        nd["possession"] = nd.possession.values + poss_off
        es["segment"] = es.segment.values + seg_off
        excl["segment"] = excl.segment.values + seg_off
        ps = l0_events.build_possessions(nd)
        node_off += len(nd)
        seg_off = int(max(es.segment.max(), nd.segment.max() if len(nd) else 0)) + 1
        poss_off = int(nd.possession.max()) + 1 if len(nd) else poss_off
        # частичные проверки по порции
        idc = l0_quality.identity_checks(es, st, g)
        ident["n"] += idc["n_events_checked"]
        ident["on"] += idc["actor_on_ice_share"] * idc["n_events_checked"]
        ident["agree"] += idc["n_counts_agree_share"] * idc["n_events_checked"]
        ident["bad_types"].update(idc["actor_not_on_ice_by_type"])
        for name, types in JS_TYPES.items():
            for k, v in l0_quality.next_event_dist(es, types).items():
                prev = js_counts[name].get(k)
                js_counts[name][k] = v if prev is None else prev.add(v, fill_value=0)
        parts.append(q)
        for k, df in [("games", g), ("events", es), ("stints", st), ("toi", toi), ("nodes", nd),
                      ("possessions", ps), ("excluded", excl), ("player_shifts", pshift), ("periods", per)]:
            tabs[k].append(_compact(df.reset_index(drop=True)))
        if log:
            log.info("L0 %s: матчи %d–%d из %d, узлов %d", source, i + 1, min(i + C, len(match_ids)),
                     len(match_ids), len(nd))
    out = {k: pd.concat(v, ignore_index=True) for k, v in tabs.items()}
    q = merge_quality(parts)
    q["time_scale"] = l0_time.time_scale_checks(out["nodes"], out["stints"], out["events"])
    q["identity"] = {"actor_on_ice_share": ident["on"] / max(ident["n"], 1), "n_events_checked": int(ident["n"]),
                     "n_counts_agree_share": ident["agree"] / max(ident["n"], 1),
                     "actor_not_on_ice_by_type": dict(ident["bad_types"].most_common(10))}
    return out, q, js_counts


def _node_next_label(nd: pd.DataFrame) -> pd.Series:
    """Исход перехода и тип следующего действия (индексы next_node — глобальные = позиции строк)."""
    nx = nd.next_node.to_numpy()
    act = nd.action.to_numpy(dtype=object)
    nxt_act = np.where(nx >= 0, act[np.clip(nx, 0, None)], "")
    pref = np.where(nd.sigma_next.to_numpy() == 1, "same:", "opp:").astype(object)
    labels = np.where(nd.terminal.to_numpy(), "STOP", pref + nxt_act)
    labels = np.where(~nd.valid_for_transition.to_numpy(), "UNKNOWN", labels)
    return pd.Series(labels, index=nd.index)


def run_l0(cfg: Config, cache: Cache, force: bool = False) -> dict:
    log = get_logger()
    key = cache.key("l0", ("l0",), code=L0_CODE)
    if cache.done(key) and not force:
        res = load_l0(cache, key)
        if res is not None:
            _validate_cached_scheme_gate(cfg, res)
            log.info("L0: загрузка из кэша %s", key)
            return res
        log.info("L0: кэш %s неполон (нет таблиц новой версии) — пересчёт", key)

    cache.invalidate_done(key)
    games = {s: io.load_games(cfg, s) for s in ("train", "test")}
    light = {s: io.load_event_columns(cfg, s, ["match_id", "actor_id", "actor_team_id", "other_id"],
                                      ("lineup_", "shift_")) for s in ("train", "test")}
    rosters = {s: l0_events.lineup_roster(light[s]) for s in ("train", "test")}
    shifts = [light[s][light[s].event_type.str.startswith("shift_")][["event_type", "actor_id"]] for s in light]
    players = io.load_players(cfg)
    su = _slot_usage(list(rosters.values()), shifts)
    players = io.resolve_roles(players, su)
    conflicts = role_conflicts(players, rosters, light)
    del light, shifts
    goalies_global = set(players.player_id[players.role == "G"])
    geo = io.rink_geometry(cfg)
    quality = {"players": {
        "n_players": int(len(players)), "position_null": int(players.position.isna().sum()),
        "position_null_slot_G": int((players.position.isna() & (players.lineup_slot_mode == "G")).sum()),
        "role_source": players.role_source.value_counts(dropna=False).to_dict(),
        "role_unresolved": int(players.role.isna().sum()),
        "goalie_conflicts": int(players.goalie_conflict.sum()),
        "role_conflicts": conflicts,
        "player_id_unique": bool(players.player_id.is_unique)}}
    out = {"players": players}
    js_all = {}
    for s in ("train", "test"):
        tabs, q, js_counts = process_source(cfg, s, games[s], rosters[s], goalies_global, geo, log)
        tabs["roster"] = rosters[s]
        nd, st, toi, ps, es, g = (tabs["nodes"], tabs["stints"], tabs["toi"], tabs["possessions"], tabs["events"],
                                  tabs["games"])
        q["n_matches"] = int(g.match_id.nunique())
        q["n_events_nonlineup"] = int(len(es))
        q["n_nodes"] = int(len(nd))
        q["n_nodes_excluded"] = int(len(tabs["excluded"]))
        q["excluded_by_reason"] = tabs["excluded"].excluded.value_counts().to_dict()
        q["transitions"] = l0_quality.transition_checks(nd)
        q["toi"] = l0_quality.toi_checks(st, toi)
        q["possessions"] = l0_quality.possession_checks(ps, st)
        q["score"] = l0_quality.score_checks(es, g)
        q["player_shifts"] = l0_quality.player_shift_checks(tabs["player_shifts"], toi, nd)
        # области подсчёта не смешиваются: train_all (все матчи) и train_v1 отдельно
        scopes = {"all": None}
        if s == "train":
            scopes["v1"] = nd.match_id[nd.scheme == "v1"].unique()
        q["coverage"] = {k: l0_quality.coverage_checks(st, nd, m) for k, m in scopes.items()}
        q["periods"] = {k: l0_quality.period_checks(tabs["periods"], m) for k, m in scopes.items()}
        q["pre_shot_structure"] = {"all": l0_quality.pre_shot_structure_audit(nd)}
        if s == "train":
            q["pre_shot_structure"]["v1"] = l0_quality.pre_shot_structure_audit(nd[nd.scheme == "v1"])
        quality[s] = q
        out[s] = tabs
        js_all[s] = js_counts
        log.info("L0 %s: узлов %d, отрезков составов %d, TOI ratio %.4f, на льду %.4f, неклассиф. %.4f",
                 s, len(nd), len(st), q["toi"]["toi_ratio_overall"], q["identity"]["actor_on_ice_share"],
                 q["transitions"]["unclassified_share"])

    # Дженсен–Шеннон: следующее событие после паса, броска и отбора, train (по схемам) против test
    quality["js_next_event"] = {
        name: {k: l0_quality.js_divergence(v, js_all["test"][name]["all"]) for k, v in js_all["train"][name].items()}
        for name in JS_TYPES}
    # то же на уровне канонических узлов: исход (своя/чужая/остановка) и тип следующего действия
    js_nodes, js_n = {}, {}
    ndtr, ndte = out["train"]["nodes"], out["test"]["nodes"]
    lab_tr, lab_te = _node_next_label(ndtr), _node_next_label(ndte)
    for name, sel_f in [("pass", lambda d: d.action == "pass"), ("shot", lambda d: d.action == "shot"),
                        ("recovery", lambda d: d.arrival == "recovery")]:
        dte = lab_te[sel_f(ndte)].value_counts()
        r, rn = {}, {"test": int(dte.sum())}
        for sch in ["v1", "v2"]:
            m = sel_f(ndtr) & (ndtr.scheme == sch)
            if m.any():
                r[sch] = l0_quality.js_divergence(lab_tr[m].value_counts(), dte)
                rn[sch] = int(m.sum())
        r["all"] = l0_quality.js_divergence(lab_tr[sel_f(ndtr)].value_counts(), dte)
        rn["all"] = int(sel_f(ndtr).sum())
        js_nodes[name], js_n[name] = r, rn
    quality["js_next_node"] = js_nodes
    quality["js_next_node_n"] = js_n
    quality["scheme_gate"] = l0_quality.scheme_gate(js_nodes, js_n, cfg)
    gates = {}
    for s in ("train", "test"):
        quality[s]["gate_l0"] = l0_quality.gate_l0(quality[s], cfg)
        gates.update({f"{s}:{k}": v for k, v in quality[s]["gate_l0"].items()})
    quality["gate_l0_all"] = gates
    gates["train:scheme_compatibility"] = quality["scheme_gate"]["status"] in ("ok", "stress")
    out["quality"] = quality
    # гейт L0 обязателен: при провале — отчёт со статусом failed, кэш не помечается готовым
    enforce("L0", gates, cfg.paths.outputs / "data_quality.json", quality)
    quality["status"] = "stress" if quality["scheme_gate"]["status"] == "stress" else "ok"

    cache.save_df(key, "players", players)
    for s in ("train", "test"):
        for k, v in out[s].items():
            df = v.copy()
            for c in df.columns:
                if df[c].dtype == object and df[c].map(lambda z: isinstance(z, (tuple, set, list))).any():
                    df[c] = df[c].map(lambda z: list(z) if isinstance(z, (tuple, set)) else z)
            cache.save_df(key, f"{s}_{k}", df)
    cache.save_obj(key, "quality", quality)
    cache.mark_done(key)
    return out


def load_l0(cache: Cache, key: str) -> dict:
    out = {"players": cache.load_df(key, "players"), "quality": cache.load_obj(key, "quality")}
    if out["quality"] is None or out["quality"].get("status") not in ("ok", "stress"):
        from .gates import GateError
        raise GateError("L0: кэш без статуса ok — использование заблокировано")
    for s in ("train", "test"):
        out[s] = {}
        for k in L0_TABLES:
            df = cache.load_df(key, f"{s}_{k}")
            if df is None:
                return None                  # неполный (старый) результат — пересчёт
            if k == "stints":
                for c in ["home_sk", "away_sk"]:
                    df[c] = df[c].map(lambda z: tuple(z) if z is not None else tuple())
            out[s][k] = df
    return out


def _validate_cached_scheme_gate(cfg: Config, res: dict) -> None:
    """Повторяет P010 для *текущей* data-конфигурации без пересчёта таблиц L0.

    Таблицы узлов не зависят от списка схем, а решение о допустимости — зависит от
    ``data.train_schemes``, ``data.scheme_gate`` и ``data.stress_mode``. Поэтому ключ L0
    намеренно остаётся привязанным к разделу ``l0``, но кэш нельзя вернуть до этой проверки.
    """
    quality = res["quality"]
    js, n = quality.get("js_next_node"), quality.get("js_next_node_n")
    if js is None or n is None:
        from .gates import GateError
        raise GateError("L0: в кэше нет распределений для проверки схем; требуется пересчёт --force")
    scheme = l0_quality.scheme_gate(js, n, cfg)
    quality["scheme_gate"] = scheme
    gates = dict(quality.get("gate_l0_all", {}))
    gates["train:scheme_compatibility"] = scheme["status"] in ("ok", "stress")
    quality["gate_l0_all"] = gates
    quality["status"] = scheme["status"]
    enforce("L0", gates,
            cfg.paths.outputs / "data_quality.json", quality)
