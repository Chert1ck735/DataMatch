"""L3: журнал вкладов действий и агрегаты «игрок × матч».

Каждый узел t имеет одну прямую TD-ошибку δ_t = r_t + σ V(s_{t+1}) − V(s_t).
Начисления (credit) по узлам:
* обычный переход (шайба у своих или остановка): δ_t целиком исполнителю t;
* переход со сменой владения после действия, не являющегося броском (пара потеря + отбор —
  один переход): потерявшему α·δ_t, вернувшему шайбу (исполнитель t+1) −(1−α)·δ_t
  (вариант 2, «split»; сумма по переходу в согласованной системе знаков = δ_t —
  телескопирование сохраняется). Вариант 1 («full», как VAEP): обоим полная δ со своим знаком
  (переход учитывается дважды, телескопирование нарушается намеренно);
* переход со сменой владения после броска: целиком бросившему (приоритет «бросок»).
Строка журнала = action_id (узел): credit = своя часть + полученная часть за отбор.
Категория строки по приоритету PDF: бросок -> потеря -> отбор -> вбрасывание -> создание.
IAV = Σ credit / TOI × 3600. Категории только раскладывают эту сумму.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ["shot", "loss", "recovery", "faceoff", "creation"]
# Ситуации журнала: метка situation() -> суффикс колонки player_match (контракт уровня A: pk = меньшинство).
SITUATION_COLUMNS = {"5v5": "5v5", "pp": "pp", "pk": "sh"}
SITUATIONS = ["5v5", "pp", "sh", "other", "unknown"]


def journal(nodes: pd.DataFrame, delta: np.ndarray, variant: str = "split", alpha: float = 0.5,
            extra: dict | None = None) -> pd.DataFrame:
    n = len(nodes)
    sig = nodes.sigma_next.values
    is_shot = nodes.action.values == "shot"
    change = sig == -1
    from .l2_value import next_positions
    nxt = next_positions(nodes)
    if variant == "split":
        a_l, a_r = alpha, 1.0 - alpha
    elif variant == "full":
        a_l, a_r = 1.0, 1.0
    else:
        raise ValueError(variant)
    own = np.where(change & ~is_shot, a_l * delta, delta)
    recv = np.zeros(n)
    give = change & ~is_shot
    recv_idx = nxt[give]
    if (recv_idx < 0).any():
        raise ValueError('Turnover credit requires an observed successor')
    # Политика неизвестного: перехода с неизвестной δ нет — нет и передачи доли вернувшему
    # шайбу; его собственная δ при этом не теряется (неизвестное не превращается в ноль чужой δ).
    transfer = -a_r * delta[give]
    known = np.isfinite(transfer)
    np.add.at(recv, recv_idx[known], transfer[known])
    recv_unknown = np.zeros(n, dtype=bool)
    recv_unknown[recv_idx[~known]] = True
    credit = own + recv
    valid_td = np.isfinite(delta) & nodes.get("valid_for_transition", pd.Series(False, index=nodes.index)).fillna(False).to_numpy(dtype=bool)
    attributed = nodes.get("valid_for_action_attribution", pd.Series(False, index=nodes.index)).fillna(False).to_numpy(dtype=bool)
    credit[~(valid_td & attributed)] = np.nan
    # Доля −(1−α)·δ, ушедшая в строку без допуска к вкладу, никому не начислена: явный флаг
    # на строке потерявшего вместо молчаливой потери (аудит 06.10.2026).
    transfer_dropped = np.zeros(n, dtype=bool)
    lost_to_invalid = known & ~(valid_td & attributed)[recv_idx]
    transfer_dropped[np.flatnonzero(give)[lost_to_invalid]] = True
    # категория
    cat = np.full(n, "creation", dtype=object)
    cat[nodes.action.values == "faceoff"] = "faceoff"
    cat[nodes.arrival.values == "recovery"] = "recovery"
    cat[change] = "loss"
    cat[is_shot] = "shot"
    j = pd.DataFrame({
        "action_id": nodes.match_id.values + ":" + nodes.node.astype(str).values,
        "node": nodes.node.values, "match_id": nodes.match_id.values, "player_id": nodes.actor_id.values,
        "team_id": nodes.team_id.values, "period": nodes.period.values, "clock": nodes.clock.values,
        "action": nodes.action.values, "category": cat, "delta": delta, "credit_own": own,
        "credit_recv": recv, "credit": credit, "mp": nodes.mp.values,
        "recv_from_unknown": recv_unknown, "transfer_dropped": transfer_dropped,
    })
    j["valid_for_credit"] = valid_td & attributed
    # Строка вернувшего с неизвестной долей переноса неполна: неизвестное не вменяется нулём,
    # поэтому в ставки она не входит (её собственная δ остаётся в журнале).
    j["valid_for_rate_metrics"] = (nodes.get("valid_for_rate_metrics", pd.Series(False, index=nodes.index))
                                   .fillna(False).to_numpy(dtype=bool) & j.valid_for_credit & ~recv_unknown)
    if extra:
        for k, v in extra.items():
            j[k] = v
    return j


def telescoping_check(nodes: pd.DataFrame, journal_df: pd.DataFrame, r: np.ndarray, v: np.ndarray) -> dict:
    """Σ credit по отрезку в системе знаков хозяев = Σ τ r − τ_0 V(s_0) (вариант split)."""
    tau = np.where(nodes.team_id.values == nodes.home_team_id.values, 1.0, -1.0)
    seg = nodes.segment.values
    observed_seg = pd.Series(journal_df.valid_for_credit.to_numpy()).groupby(seg).all()
    lhs = pd.Series(tau * journal_df.credit.values).groupby(seg).sum()
    first = ~pd.Series(seg).duplicated().values
    rhs = pd.Series(tau * r).groupby(seg).sum() - pd.Series(np.where(first, tau * v, 0.0)).groupby(seg).sum()
    d = (lhs - rhs).abs()[observed_seg]
    return {"max_abs_diff": float(d.max()) if len(d) else None, "n_segments": int(len(d)),
            "excluded_segments": int((~observed_seg).sum())}


def onice_incidence(nodes: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Для каждого броска при 5×5: полевые на льду атакующей (side=+1) и обороняющейся (−1) команды."""
    sh = np.flatnonzero(nodes.action.values == "shot")
    sn = nodes.iloc[sh]
    st = stints.set_index(["match_id", "stint"])
    j = st.reindex(pd.MultiIndex.from_arrays([sn.match_id.values, sn.stint.values]))
    is5 = j.is_5v5.fillna(False).to_numpy(dtype=bool)
    # On-ice счёт следует подтверждённому составу stint, а не атрибуции бросающего: бросок с
    # неподтверждённым исполнителем всё равно произошёл при этих десяти на льду. Нулевая
    # длительность stint атрибуцию не отменяет (правило проекта; решение по броскам из нулевых stint).
    # Матч должен быть допущен к ставкам: иначе числитель on-ice разошёлся бы с toi_rate (= 0).
    is5 = (is5 & j.valid_for_action_attribution.fillna(False).to_numpy(dtype=bool)
           & j.rate_match_eligible.fillna(False).to_numpy(dtype=bool))
    ish = sn.team_id.values == sn.home_team_id.values
    rows_i, rows_p, rows_s = [], [], []
    hs_all, as_all = j.home_sk.values, j.away_sk.values
    for k in np.flatnonzero(is5):
        f, a = (hs_all[k], as_all[k]) if ish[k] else (as_all[k], hs_all[k])
        for p in f:
            rows_i.append(sh[k]); rows_p.append(p); rows_s.append(1)
        for p in a:
            rows_i.append(sh[k]); rows_p.append(p); rows_s.append(-1)
    inc = pd.DataFrame({"node_idx": np.asarray(rows_i, dtype=np.int64), "player_id": rows_p,
                        "side": np.array(rows_s, dtype=np.int8)})
    inc["match_id"] = nodes.match_id.values[inc.node_idx.values]
    return inc


def static_player_match(nodes: pd.DataFrame, toi: pd.DataFrame, stints: pd.DataFrame,
                        events: pd.DataFrame, extra_keys: pd.DataFrame | None = None) -> pd.DataFrame:
    """Не зависящие от моделей агрегаты (матч, игрок): TOI, голы, передачи, очки, +/− (равные составы).

    extra_keys — дополнительные пары (match_id, player_id, team_id), например исполнители
    действий в stint нулевой длительности: строка создаётся с нулевым TOI, а не теряется.
    """
    cols = ["team_id", "toi_s", "toi_5v5_s", "toi_pp_s", "toi_sh_s", "toi_valid_s",
            "toi_rate_s", "toi_rate_5v5_s", "toi_rate_pp_s", "toi_rate_sh_s"]
    base = toi.set_index(["match_id", "player_id"])[cols].copy()
    if extra_keys is not None and len(extra_keys):
        ek = extra_keys.drop_duplicates(["match_id", "player_id"]).set_index(["match_id", "player_id"])
        ek = ek[~ek.index.isin(base.index)]
        if len(ek):
            add = pd.DataFrame(0.0, index=ek.index, columns=cols[1:])
            add.insert(0, "team_id", ek.team_id.to_numpy())
            base = pd.concat([base, add])
    # голы основного и дополнительного времени (периоды 1–5), включая штрафные броски, входят в
    # очки; серия буллитов (период 0) — нет. В «+/−» голы штрафных бросков не входят.
    ev = events[events.period > 0]
    g = ev[ev.event_type == "goal"]
    a = ev[ev.event_type.isin(["assist_primary", "assist_secondary"])]
    pts = pd.concat([g[["match_id", "actor_id"]].assign(goal=1, ast=0),
                     a[["match_id", "actor_id"]].assign(goal=0, ast=1)])
    pts = pts.rename(columns={"actor_id": "player_id"}).groupby(["match_id", "player_id"]).sum()
    pm = base.join(pts, how="left").fillna({"goal": 0, "ast": 0})
    pm["points"] = pm.goal + pm.ast
    st = stints.set_index(["match_id", "stint"])
    g = g[~g.penalty_shot_seg.astype(bool)]
    gj = st.reindex(pd.MultiIndex.from_arrays([g.match_id.values, g.stint.values]))
    rows = []
    for i in range(len(g)):
        if not (gj.n_home_sk.values[i] == gj.n_away_sk.values[i] and bool(gj.home_g_in.values[i])
                and bool(gj.away_g_in.values[i])):
            continue
        home_scored = g.actor_team_id.values[i] == g.home_team_id.values[i]
        for p in gj.home_sk.values[i]:
            rows.append((g.match_id.values[i], p, 1 if home_scored else -1))
        for p in gj.away_sk.values[i]:
            rows.append((g.match_id.values[i], p, -1 if home_scored else 1))
    pmv = pd.DataFrame(rows, columns=["match_id", "player_id", "pm"]).groupby(["match_id", "player_id"]).sum()
    pm = pm.join(pmv, how="left")
    pm["pm"] = pm.pm.fillna(0.0)
    # пустые таблицы голов/+/− дают object-столбцы; счётчики всегда числовые
    for c in ["goal", "ast", "points", "pm"]:
        pm[c] = pm[c].astype(float)
    return pm.reset_index()


def dynamic_player_match(static_pm: pd.DataFrame, nodes: pd.DataFrame, jr: pd.DataFrame, inc: pd.DataFrame,
                         xg_node: np.ndarray, extra_cols: dict | None = None) -> pd.DataFrame:
    """Добавляет к статической таблице зависящие от моделей величины: компоненты Σcredit по
    категориям, IAV-сумму, индивидуальный xG и on-ice 5×5 xGF/xGA (замороженный xG-скорер)."""
    key = ["match_id", "player_id"]
    jr = jr[jr.valid_for_rate_metrics].copy()
    comp = jr.pivot_table(index=key, columns="category", values="credit", aggfunc="sum", fill_value=0.0)
    for c in CATEGORIES:
        if c not in comp:
            comp[c] = 0.0
    comp = comp[CATEGORIES].add_prefix("c_")
    pm = static_pm.set_index(key).join(comp, how="left")
    pm[[f"c_{c}" for c in CATEGORIES]] = pm[[f"c_{c}" for c in CATEGORIES]].fillna(0.0)
    if extra_cols:
        for name, col in extra_cols.items():
            e = jr.groupby(key)[col].sum().rename(name)
            pm = pm.join(e, how="left")
            pm[name] = pm[name].fillna(0.0)
    sh = (nodes.action.values == "shot") & nodes.valid_for_rate_metrics.to_numpy(dtype=bool)
    ind = pd.DataFrame({"match_id": nodes.match_id.values[sh], "player_id": nodes.actor_id.values[sh],
                        "ixg": xg_node[sh], "igoals": nodes.is_goal.values[sh].astype(float), "ishots": 1.0})
    ind = ind.groupby(key).sum()
    pm = pm.join(ind, how="left").fillna({"ixg": 0.0, "igoals": 0.0, "ishots": 0.0})
    x = xg_node[inc.node_idx.values]
    oi = pd.DataFrame({"match_id": inc.match_id.values, "player_id": inc.player_id.values,
                       "xgf_5v5": np.where(inc.side.values > 0, x, 0.0), "xga_5v5": np.where(inc.side.values < 0, x, 0.0)})
    oi = oi.groupby(key).sum()
    pm = pm.join(oi, how="left").fillna({"xgf_5v5": 0.0, "xga_5v5": 0.0})
    pm["iav_sum"] = pm[[f"c_{c}" for c in CATEGORIES]].sum(axis=1)
    if "situation" in jr:
        # разрез тех же сумм по ситуациям (журнал по ситуациям L3; IAV от него не меняется)
        for sit_name, sit_label in SITUATION_COLUMNS.items():
            js = jr[jr.situation == sit_label]
            cs = js.pivot_table(index=key, columns="category", values="credit", aggfunc="sum", fill_value=0.0)
            cs = cs.reindex(columns=CATEGORIES, fill_value=0.0)
            cs.columns = [f"c_{c}_{sit_name}" for c in CATEGORIES]
            pm = pm.join(cs, how="left")
            pm[list(cs.columns)] = pm[list(cs.columns)].fillna(0.0)
    n_act = jr.groupby(key).size().rename("n_actions")
    pm = pm.join(n_act, how="left")
    pm["n_actions"] = pm.n_actions.fillna(0).astype(int)
    return pm.reset_index()


# ---------------------------------------------------------------------------------------
# Журнал по ситуациям и программные гейты журнала (стадия L3)


def situation(nodes: pd.DataFrame, stints: pd.DataFrame) -> np.ndarray:
    """Численность на льду из stints L0 с точки зрения команды исполнителя.

    Определения совпадают с TOI L0 (l0_shifts): 5v5 = is_5v5; большинство (pp) / меньшинство (sh) —
    больше/меньше полевых при обоих вратарях на льду; остальное — other; узел без stint — unknown.
    Ситуация только раскладывает журнал и не меняет категорий и IAV.
    """
    st = stints.drop_duplicates(["match_id", "stint"]).set_index(["match_id", "stint"])
    j = st.reindex(pd.MultiIndex.from_arrays([nodes.match_id.to_numpy(), nodes.stint.to_numpy()]))
    found = j.n_home_sk.notna().to_numpy()
    home = nodes.team_id.to_numpy() == nodes.home_team_id.to_numpy()
    nh = j.n_home_sk.fillna(-1).to_numpy()
    na = j.n_away_sk.fillna(-1).to_numpy()
    own, opp = np.where(home, nh, na), np.where(home, na, nh)

    def flag(c):
        return j[c].astype("boolean").fillna(False).to_numpy(dtype=bool)

    goalies = flag("home_g_in") & flag("away_g_in")
    lab = np.full(len(nodes), "other", dtype=object)
    lab[goalies & (own > opp)] = "pp"
    lab[goalies & (own < opp)] = "sh"
    lab[flag("is_5v5")] = "5v5"
    lab[~found] = "unknown"
    return lab


def journal_gates(jr: pd.DataFrame, pm: pd.DataFrame, tele: dict | None, tol: float = 1e-8) -> dict:
    """Одна категория на action_id; категории раскладывают сумму без пересечений и потерь.

    pm — агрегаты по всем исполнителям журнала (до отбора полевых), чтобы сверка была полной.
    """
    ok_rows = jr.valid_for_credit.to_numpy(dtype=bool)
    rate = jr.valid_for_rate_metrics.to_numpy(dtype=bool)
    total = float(np.nansum(jr.credit.to_numpy()[rate]))
    comp = float(pm[[f"c_{c}" for c in CATEGORIES]].to_numpy().sum())
    gates = {
        "action_id_unique": bool(jr.action_id.is_unique),
        "one_known_category_per_action": bool(jr.category.isin(CATEGORIES).all()),
        "credited_rows_finite": bool(np.isfinite(jr.credit.to_numpy()[ok_rows]).all()),
        "components_sum_to_journal": bool(abs(comp - total) <= tol * max(1.0, float(rate.sum()))),
        "iav_equals_component_sum": bool(np.allclose(
            pm.iav_sum.to_numpy(), pm[[f"c_{c}" for c in CATEGORIES]].sum(axis=1).to_numpy(), atol=1e-12)),
    }
    if tele is not None:
        gates["telescoping_split"] = telescoping_ok(tele)
    return gates


TELESCOPING_TOL = 1e-6   # ожидаемые голы: допуск сверки Σcredit по отрезку (вариант split)


def telescoping_ok(tele: dict) -> bool:
    """Сверка телескопирования проходит только при хотя бы одном проверенном отрезке."""
    return bool(tele.get("n_segments", 0) > 0 and tele.get("max_abs_diff") is not None
                and tele["max_abs_diff"] <= TELESCOPING_TOL)
