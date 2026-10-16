"""L3: журнал вклада δ и таблица «игрок × матч» по выбранной L2-процедуре (уровень A).

Смысл. Для каждого узла (action_id) TD-ошибка δ = r + σ V(s') − V(s) берётся из L2 без
пересчёта: для train — match-cross-fit OOF (train_values: каждый матч оценён моделями xG→V,
обученными без него), для test — итоговая процедура, переобученная на всём допустимом train
(test_values, решение 06.10.2026 по правилу holdout L7). l3_contrib.journal распределяет
δ по исполнителям (вариант split/full из раздела contrib) и даёт каждому action_id одну категорию
(бросок → потеря → отбор → вбрасывание → создание). Ситуация строки журнала — численность из
stints L0. player_match — суммы credit по категориям и ситуациям, TOI L0, очки, ixG и on-ice xG
при 5 на 5, амплуа и компонента связности. Ставки на 60 здесь не считаются (это L5).

Единицы: credit, iav_sum, ixg, xgf/xga — ожидаемые голы (знаковая сумма xG); TOI — секунды.
Train — все допустимые матчи train (разработка и внешний блок, схемы data.train_schemes), для
которых L2 сохранил OOF-δ. Ничего не обучается: L3 только агрегирует.

L0–L2 здесь только читаются: если их актуального кэша нет, поднимается UpstreamNotReady и
ничего не пересчитывается (запуск prepare/xg/value — отдельные команды).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import l3_contrib, stage_l2
from .cache import Cache, load_checked_json, obj_hash
from .config import Config, get_logger
from .gates import enforce, require_ok

L3_CODE = tuple(sorted(set(stage_l2.L2_CODE + ("l3_contrib", "stage_l3"))))
# CONFIG: позиции полевых и их амплуа для L5 (архитектура: амплуа по позиции из заявки)
POSITION_ROLE = {"C": "C", "LW": "W", "RW": "W", "D": "D"}
# CONFIG: слот заявки -> позиция (запасной источник, если позиция справочника не полевая/пуста)
SLOT_POSITION = {"C": "C", "LW": "LW", "RW": "RW", "LD": "D", "RD": "D"}

PM_KEY = ["match_id", "player_id"]
# Таблицы L0, которые действительно читает L3 (остальные выходы L0 освобождаются после загрузки)
L0_USED = ("nodes", "stints", "toi", "events", "roster", "games")


class UpstreamNotReady(RuntimeError):
    """Нужный предшествующий слой (L0/L1/L2) ещё не рассчитан или устарел в кэше."""


# ---------------------------------------------------------------------------------------
# Загрузка L0–L2 только из готового кэша (без пересчёта)

def _load_l0(cfg: Config, cache: Cache) -> dict:
    from . import stage_l0
    key = cache.key("l0", ("l0",), code=stage_l0.L0_CODE)
    complete = cache.done(key) and all((cache.root / key / f"{s}_{k}.parquet").exists()
                                       for s in ("train", "test") for k in stage_l0.L0_TABLES)
    if not complete:
        raise UpstreamNotReady(f"L0: нет готового кэша {key}; сначала `datamatch.cli prepare`")
    res = stage_l0.run_l0(cfg, cache, force=False)  # при готовом кэше только загрузка + P010
    # L3 нужны только узлы, составы, TOI, события (очки и +/−), заявка и матчи; остальные
    # таблицы L0 освобождаются, чтобы пик памяти не зависел от неиспользуемых выходов L0.
    for s in ("train", "test"):
        for k in set(stage_l0.L0_TABLES) - set(L0_USED):
            res[s].pop(k, None)
    res["key"] = key
    return res


def _load_splits(cfg: Config, cache: Cache, l0: dict) -> dict:
    from . import splits
    sch = l0["train"]["nodes"].groupby("match_id").scheme.first()
    prov = splits.splits_provenance(cache, l0["train"]["games"], l0["test"]["games"], sch)
    sp = splits.load_splits(cfg, prov)
    if sp is None:
        raise UpstreamNotReady("Разбиения: artifacts/splits.json отсутствует или устарел; требуется `prepare`")
    sp["provenance"] = prov
    return sp


def _load_l1_selection(cfg: Config, cache: Cache, l0: dict, sp: dict) -> dict:
    from . import stage_l1
    s, _ = stage_l1.dev_shots(l0, sp, tuple(cfg["data"]["train_schemes"]))
    prov = stage_l1.l1_provenance(cache, s, sp)
    key = cache.key("l1_dev_model", ("xg", "data", "eval", "seed"), extra=obj_hash(prov), code=stage_l1.L1_CODE)
    sel = load_checked_json(cfg.paths.artifacts / "xg_selected.json", prov)
    if sel is None or not cache.done(key):
        raise UpstreamNotReady(f"L1: нет актуального выбора xG (ключ {key}); требуется `datamatch.cli xg`")
    require_ok(sel, "L1 xg_selected.json")
    return sel


def _load_l2(cfg: Config, cache: Cache, l0: dict, sp: dict, sel: dict) -> dict:
    schemes = tuple(cfg["data"]["train_schemes"])
    nodes = stage_l2.development_nodes(l0["train"]["nodes"], sp, schemes)
    train_nodes = stage_l2.final_fit_nodes(l0["train"]["nodes"], sp, schemes)
    prov = stage_l2.provenance(cfg, cache, nodes, sp, sel, l0["test"]["nodes"], train_nodes)
    key = cache.key("l2_dev_model", ("value", "data", "eval", "seed"), extra=obj_hash(prov), code=stage_l2.L2_CODE)
    result = load_checked_json(cfg.paths.artifacts / "value_selected.json", prov)
    if result is None or not cache.done(key):
        raise UpstreamNotReady(f"L2: нет актуального выбора V (ключ {key}); требуется `datamatch.cli value`")
    require_ok(result, "L2 value_selected.json")
    train, test = cache.load_df(key, "train_values"), cache.load_df(key, "test_values")
    if train is None or test is None:
        raise UpstreamNotReady(f"L2: в кэше {key} нет train_values/test_values")
    return {"selection": result, "train_nodes": train_nodes, "train_values": train, "test_values": test, "key": key}


# ---------------------------------------------------------------------------------------
# Чистые функции (тестируются на синтетике)

def align_values(nodes: pd.DataFrame, values: pd.DataFrame, check_source: bool = False) -> pd.DataFrame:
    """Значения L2 строго по ID узлов графа; любое несовпадение покрытия — ошибка, не подстановка."""
    if values.node.duplicated().any() or set(values.node) != set(nodes.node):
        raise ValueError("L2 values must cover each graph node exactly once")
    v = values.set_index("node").loc[nodes.node.to_numpy()].reset_index()
    if not np.array_equal(v.match_id.astype(str).to_numpy(), nodes.match_id.astype(str).to_numpy()):
        raise ValueError("L2 values disagree with node match identities")
    if check_source and "source_node" in v and "source_node" in nodes:
        if not np.array_equal(v.source_node.to_numpy(), nodes.source_node.to_numpy()):
            raise ValueError("L2 dev values disagree with original L0 node identities")
    return v


def player_positions(players: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Позиция C/LW/RW/D и амплуа C/W/D полевого игрока; вратари (role G) исключаются.

    Источник — позиция справочника; если она не полевая или пуста — самый частый полевой слот
    заявки (LD/RD -> D). Амплуа: LW/RW -> W. Позиция не зависит от рейтинга и известна до матча.
    """
    p = players.set_index("player_id")
    sk = roster[roster.slot.isin(SLOT_POSITION)]
    slot_mode = sk.groupby("player_id").slot.agg(lambda v: v.value_counts().sort_index().idxmax())
    ids = pd.Index(sorted(set(roster.player_id) | set(p.index)))
    pos = p.position.reindex(ids)
    pos = pos.where(pos.isin(POSITION_ROLE), slot_mode.reindex(ids).map(SLOT_POSITION))
    out = pd.DataFrame({"position": pos, "role": pos.map(POSITION_ROLE)}, index=ids)
    goalie = p.role.reindex(ids).eq("G")
    out = out[out.role.notna() & ~goalie]
    return out.rename_axis("player_id")


def build_player_match(nodes: pd.DataFrame, jr: pd.DataFrame, r: np.ndarray, toi: pd.DataFrame,
                       stints: pd.DataFrame, events: pd.DataFrame, positions: pd.DataFrame,
                       team_component: dict, khl_component: int | None,
                       split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (player_match полевых, агрегаты по всем исполнителям — для сверки гейтов).

    Строка (матч, игрок) есть, если у игрока допущенная экспозиция L0 или действие в журнале:
    действие в stint нулевой длительности даёт строку с нулевым TOI (атрибуция не теряется).
    Игроки только из заявки строк не получают — их строки в test_ratings добавляет L6 из prior.
    """
    acted = jr[jr.valid_for_rate_metrics.to_numpy(dtype=bool) & jr.player_id.notna()]
    extra = acted[["match_id", "player_id", "team_id"]].drop_duplicates(PM_KEY)
    static = l3_contrib.static_player_match(nodes, toi, stints, events, extra_keys=extra)
    inc = l3_contrib.onice_incidence(nodes, stints)
    full = l3_contrib.dynamic_player_match(static, nodes, jr, inc, r)
    field = full.player_id.isin(positions.index)
    pm = full[field].copy()
    pm.insert(2, "split", split)
    pm["position"] = pm.player_id.map(positions.position)
    pm["role"] = pm.player_id.map(positions.role)
    comp = pm.team_id.map(team_component)
    pm["component"] = comp.fillna(-1).astype(int)
    pm["is_khl_component"] = (pm.component == khl_component) if khl_component is not None else False
    pm["rated"] = (pm.toi_rate_s > 0) | (pm.n_actions > 0)
    return pm.sort_values(PM_KEY).reset_index(drop=True), full


def contribute(nodes: pd.DataFrame, values: pd.DataFrame, toi, stints, events, positions,
               team_component, khl_component, split: str, variant: str, alpha: float) -> dict:
    """Журнал и player_match одной части (train или test) по готовым δ из L2."""
    delta = values.delta.to_numpy(dtype=float)
    r = values.r.to_numpy(dtype=float)
    jr = l3_contrib.journal(nodes, delta, variant, alpha)
    jr["situation"] = l3_contrib.situation(nodes, stints)
    jr["split"] = split
    pm, full = build_player_match(nodes, jr, r, toi, stints, events, positions,
                                  team_component, khl_component, split)
    tele = (l3_contrib.telescoping_check(nodes, jr, r, values.value.to_numpy(dtype=float))
            if variant == "split" else None)
    gates = l3_contrib.journal_gates(jr, full, tele)
    if tele is not None:
        gates["telescoping_split"] = l3_contrib.telescoping_ok(tele)
    rate = jr.valid_for_rate_metrics.to_numpy(dtype=bool)
    credit = jr.credit.to_numpy()
    report = {
        "n_nodes": int(len(jr)), "n_matches": int(jr.match_id.nunique()),
        # Диагностика контракта L2 (единичные веса): на OOF средняя δ допущенных узлов ≈ 0.
        "mean_delta_valid": (float(np.nanmean(jr.delta.to_numpy()[jr.valid_for_credit.to_numpy(dtype=bool)]))
                             if jr.valid_for_credit.any() else None),
        "transfer_dropped": int(jr.transfer_dropped.sum()),
        "valid_for_credit_share": float(jr.valid_for_credit.mean()) if len(jr) else None,
        "valid_for_rate_metrics_share": float(rate.mean()) if len(jr) else None,
        "nonfinite_delta": int((~np.isfinite(delta)).sum()),
        "recv_from_unknown_transition": int(jr.recv_from_unknown.sum()),
        "telescoping": tele,
        "category_counts": jr.loc[rate, "category"].value_counts().to_dict(),
        "category_credit": {c: float(np.nansum(credit[rate & (jr.category.to_numpy() == c)]))
                            for c in l3_contrib.CATEGORIES},
        "situation_counts": jr.loc[rate, "situation"].value_counts().to_dict(),
        "journal_credit_total": float(np.nansum(credit[rate])),
        "field_credit_total": float(pm.iav_sum.sum()),
        "non_field_actor_credit": float(full.loc[~full.player_id.isin(positions.index), "iav_sum"].sum()),
        "n_player_match_rows": int(len(pm)), "n_players": int(pm.player_id.nunique()),
        "n_rows_zero_toi_with_actions": int(((pm.toi_s <= 0) & (pm.n_actions > 0)).sum()),
        "n_rows_rate_ineligible": int((~pm.rated).sum()),
        "role_counts_players": pm.drop_duplicates("player_id").role.value_counts().to_dict(),
        "gates": gates,
    }
    return {"journal": jr, "player_match": pm, "report": report}


# ---------------------------------------------------------------------------------------

def _team_components(l0: dict, sp: dict) -> tuple[dict, int | None]:
    from .splits import team_components
    comp = team_components(l0["train"]["games"])
    return comp, sp.get("khl_component")


def run_l3(cfg: Config, cache: Cache, force: bool = False) -> dict:
    log = get_logger()
    l0 = _load_l0(cfg, cache)
    sp = _load_splits(cfg, cache, l0)
    sel = _load_l1_selection(cfg, cache, l0, sp)
    l2 = _load_l2(cfg, cache, l0, sp, sel)
    c = cfg["contrib"]
    variant, alpha = c["turnover_variant"], float(c["alpha"])
    from .splits import splits_hash
    key = cache.key("l3", ("contrib",), code=L3_CODE,
                    extra=obj_hash({"l0": l0["key"], "l2": l2["key"], "splits": splits_hash(sp)}))
    names = [f"{s}_{t}" for s in ("train", "test") for t in ("journal", "player_match")]
    if not force and cache.done(key) and all((cache.root / key / f"{n}.parquet").exists() for n in names):
        report = cache.load_obj(key, "report")
        if report is not None and report.get("status") == "ok":
            log.info("L3: загрузка из кэша %s", key)
            out = {s: {t: cache.load_df(key, f"{s}_{t}") for t in ("journal", "player_match")}
                   for s in ("train", "test")}
            out["report"] = report
            return out
    cache.invalidate_done(key)
    positions = player_positions(l0["players"], pd.concat([l0["train"]["roster"], l0["test"]["roster"]]))
    team_comp, khl = _team_components(l0, sp)
    train_nodes = l2["train_nodes"]
    train_matches = set(train_nodes.match_id)
    tr = l0["train"]
    sub = {k: tr[k][tr[k].match_id.isin(train_matches)] for k in ("toi", "stints", "events")}
    parts = {}
    parts["train"] = contribute(train_nodes, align_values(train_nodes, l2["train_values"], check_source=True),
                                sub["toi"], sub["stints"], sub["events"], positions,
                                team_comp, khl, "train", variant, alpha)
    te = l0["test"]
    parts["test"] = contribute(te["nodes"], align_values(te["nodes"], l2["test_values"]), te["toi"], te["stints"],
                               te["events"], positions, team_comp, khl, "test", variant, alpha)
    gates = {f"{s}:{k}": v for s in parts for k, v in parts[s]["report"]["gates"].items()}
    report = {
        "status": "pending", "stage": "L3", "cache_key": key, "variant": variant, "alpha": alpha,
        "inputs": {"l0": l0["key"], "l2": l2["key"], "l2_selected": l2["selection"].get("selected"),
                   "l2_spec": l2["selection"].get("spec"), "splits": splits_hash(sp)},
        "delta_source": {"train": "L2 train_values: match-cross-fit OOF of the whole xG->V recipe over all allowed train",
                         "test": "L2 test_values: final procedure refitted on all allowed train, neutral strength",
                         # какой V считался δ (METHODOLOGY 3, поправка 08.10): читаемо из артефакта, а не только из хэша
                         "value_for_delta": ("uncalibrated Markov V (value.delta_from_raw: true)"
                                             if bool(cfg["value"].get("delta_from_raw", False))
                                             else "calibrated V (value.delta_from_raw: false)")},
        "train_scope": "all allowed train matches (development + outer block, data.train_schemes) with OOF delta",
        "khl_component": khl,
        "units": {"credit": "expected signed goals (sum, not per 60)", "toi": "seconds"},
        "rate_policy": "numerator rows valid_for_rate_metrics; denominator toi_rate_*; zero-TOI actions keep rows",
        "unknown_transfer_policy": "turnover with unknown delta transfers nothing; recoverer keeps own delta",
        "train": parts["train"]["report"], "test": parts["test"]["report"], "gates": gates,
    }
    enforce("L3", gates, cfg.paths.artifacts / "contrib_report.json", report)
    report["status"] = "ok"
    for s in parts:
        for t in ("journal", "player_match"):
            cache.save_df(key, f"{s}_{t}", parts[s][t])
    cache.save_obj(key, "report", report)
    cache.mark_done(key)
    (cfg.paths.artifacts / "contrib_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str))
    log.info("L3: train %d строк player_match / test %d; телескопирование train %s, test %s",
             len(parts["train"]["player_match"]), len(parts["test"]["player_match"]),
             parts["train"]["report"]["telescoping"], parts["test"]["report"]["telescoping"])
    out = {s: {t: parts[s][t] for t in ("journal", "player_match")} for s in parts}
    out["report"] = report
    return out
