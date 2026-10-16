"""Диагностики аудита 08.10.2026 (A6), только чтение готовых кэшей, модули не меняются.

(а) L2: средний δ по типу шага на OOF train — последний в отрезке (терминальный), неизвестный переход,
    смена владения, своё продолжение. При беллмановской V средний δ каждого типа ≈ 0 только в сумме;
    систематический сдвиг одного типа — признак смещения награды/цели.
(б) L3: доля действий игрока с неизвестным переходом (передача кредита не известна) и её корреляция
    Спирмена с рейтингом O1 (train) — проверка, не зависит ли рейтинг от качества разметки.
(в) L0: «длинные» регулярные периоды (наблюдаемый конец > 1200 с): сколько, где кончаются, есть ли
    события после 1200 с, какую долю игрового времени их матчей дают секунды сверх 1200.

Кэши ищутся по префиксу (в кэше по одной актуальной папке L0/L3, L5 — по указателю l5__current).
Запуск: PYTHONPATH=src .venv/bin/python scripts/diagnostics_audit.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
OUT = ROOT / "outputs"
REGULAR_PERIOD_S = 1200.0
MIN_TOI_S = 3600.0


def _one(prefix: str) -> Path:
    dirs = sorted(p for p in CACHE.glob(f"{prefix}__*") if (p / "_DONE").exists())
    if len(dirs) != 1:
        raise SystemExit(f"ожидалась одна готовая папка {prefix}__*, найдено {len(dirs)}: {[d.name for d in dirs]}")
    return dirs[0]


def _l5_dir() -> Path:
    ptr = pickle.load(open(CACHE / "l5__current" / "pointer.pkl", "rb"))
    name = ptr.get("key") if isinstance(ptr, dict) else ptr
    return CACHE / str(name)


def graph_nodes() -> pd.DataFrame:
    """Узлы графа L2 итоговой подгонки — те же номера `node`, что в журнале L3 (только чтение кэша)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import validation_report as vr

    from datamatch import io, stage_l2, stage_l3
    from datamatch.config import load_config
    cfg = load_config(None)
    cache = vr.ReadOnlyCache(cfg, io.data_version(cfg))
    l0 = stage_l3._load_l0(cfg, cache)
    sp = stage_l3._load_splits(cfg, cache, l0)
    nodes = stage_l2.final_fit_nodes(l0["train"]["nodes"], sp, tuple(cfg["data"]["train_schemes"]))
    return nodes[["match_id", "node", "action", "transition_status", "poss_change_next"]]


def step_types(l3: Path) -> dict:
    nodes = graph_nodes()
    jr = pd.read_parquet(l3 / "train_journal.parquet", columns=["match_id", "node", "delta", "valid_for_credit"])
    jr = jr[jr.valid_for_credit.to_numpy(dtype=bool)].drop_duplicates(["match_id", "node"])
    m = jr.merge(nodes, on=["match_id", "node"], how="left", validate="one_to_one")
    st = m.transition_status.fillna("нет узла L0")
    kind = np.where(st == "terminal", "последний в отрезке", np.where(st == "unknown", "неизвестный переход",
                    np.where(m.poss_change_next.fillna(False).astype(bool), "смена владения", "своё продолжение")))
    kind = np.where(st == "нет узла L0", "нет узла L0", kind)
    m = m.assign(kind=kind, is_shot=m.action.eq("shot"))
    g = m.groupby("kind").delta.agg(["size", "mean", "std", "sum"])
    g["se"] = g["std"] / np.sqrt(g["size"])
    by_shot = m.groupby(["kind", "is_shot"]).delta.agg(["size", "mean"]).reset_index()
    return {"n_nodes": int(len(m)), "unmatched_nodes": int((st == "нет узла L0").sum()),
            "mean_delta_all": float(m.delta.mean()),
            "by_kind": {k: {c: float(v) for c, v in r.items()} for k, r in g.iterrows()},
            "by_kind_shot": by_shot.to_dict(orient="records")}


def unknown_share(l3: Path, l5: Path) -> dict:
    """Доля действий игрока без известной δ (неизвестный переход: valid_for_credit = False)."""
    jr = pd.read_parquet(l3 / "train_journal.parquet", columns=["player_id", "valid_for_credit"])
    jr = jr[jr.player_id.notna()]
    share = (~jr.valid_for_credit.astype(bool)).groupby(jr.player_id).mean().rename("unknown_share")
    pm = pd.read_parquet(l3 / "train_player_match.parquet", columns=["player_id", "toi_rate_s"])
    toi = pm.groupby("player_id").toi_rate_s.sum()
    # без неизвестных переходов у игрока корреляция не определена — оставляем в выборке (доля 0)
    o1 = pd.read_parquet(l5 / "O1_train.parquet", columns=["player_id", "rating", "role"]).set_index("player_id")
    df = o1.join(share, how="inner").join(toi.rename("toi_s"), how="inner")
    df = df[df.toi_s >= MIN_TOI_S]
    rho = float(df.rating.corr(df.unknown_share, method="spearman"))
    by_role = {r: float(g.rating.corr(g.unknown_share, method="spearman")) for r, g in df.groupby("role")}
    q = df.unknown_share.quantile([0.1, 0.5, 0.9]).to_dict()
    return {"n_players": int(len(df)), "min_toi_s": MIN_TOI_S, "spearman_rating_vs_unknown_share": rho,
            "spearman_by_role": by_role, "unknown_share_quantiles": {str(k): float(v) for k, v in q.items()},
            "mean_unknown_share": float(df.unknown_share.mean())}


def long_periods(l0: Path) -> dict:
    per = pd.read_parquet(l0 / "train_periods.parquet")
    per = pd.concat([per.assign(split="train"), pd.read_parquet(l0 / "test_periods.parquet").assign(split="test")])
    reg = per[per.period <= 3]
    long = reg[reg.status.eq("overlong")]        # статус L0: конец регулярного периода заметно позже 1200 с
    ends = long.observed_end.round(0).value_counts().head(8)
    after = (long.end_event_clock.fillna(0) > REGULAR_PERIOD_S).sum() if "end_event_clock" in long else None
    excess = (long.observed_end - REGULAR_PERIOD_S).clip(lower=0)
    games = per[per.match_id.isin(long.match_id)].groupby("match_id").observed_end.sum()
    return {"n_regular_periods": int(len(reg)), "n_long": int(len(long)),
            "by_split": long.split.value_counts().to_dict(),
            "observed_end_top": {str(int(k)): int(v) for k, v in ends.items()},
            "n_with_event_after_1200": None if after is None else int(after),
            "status_counts": long.status.value_counts().to_dict() if "status" in long else {},
            "excess_seconds_total": float(excess.sum()),
            "excess_share_of_their_matches_time": float(excess.sum() / games.sum()) if len(games) else 0.0,
            "max_observed_end": float(long.observed_end.max()) if len(long) else None}


def main() -> int:
    l0, l3, l5 = _one("l0"), _one("l3"), _l5_dir()
    rep = {"caches": {"l0": l0.name, "l3": l3.name, "l5": l5.name},
           "a_l2_delta_by_step": step_types(l3), "b_l3_unknown_transition": unknown_share(l3, l5),
           "c_l0_long_periods": long_periods(l0)}
    OUT.mkdir(exist_ok=True)
    (OUT / "diagnostics_audit.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    a, b, c = rep["a_l2_delta_by_step"], rep["b_l3_unknown_transition"], rep["c_l0_long_periods"]
    md = ["# Диагностики аудита 08.10.2026 (A6)", "", f"Кэши: {rep['caches']}", "",
          f"## (а) L2: средний δ по типу шага (OOF train, {a['n_nodes']} узлов; без узла L0 — {a['unmatched_nodes']})", "",
          "| тип шага | n | средний δ | ст. ошибка | Σδ |", "|:--|--:|--:|--:|--:|"]
    md += [f"| {k} | {int(r['size'])} | {r['mean']:+.5f} | {r['se']:.5f} | {r['sum']:+.2f} |" for k, r in a["by_kind"].items()]
    md += ["", f"Средний δ всех допущенных узлов: {a['mean_delta_all']:+.5f}.", "",
           f"## (б) L3: доля действий с неизвестным переходом (игроки train с ≥ 60 мин, n = {b['n_players']})", "",
           f"Средняя доля {b['mean_unknown_share']:.3f}; квантили 10/50/90%: {b['unknown_share_quantiles']}.",
           f"Спирмен с рейтингом O1: **{b['spearman_rating_vs_unknown_share']:+.3f}**; по амплуа: "
           + ", ".join(f"{r} {v:+.3f}" for r, v in b["spearman_by_role"].items()) + ".", "",
           f"## (в) L0: длинные регулярные периоды (> 1200 с): {c['n_long']} из {c['n_regular_periods']}", "",
           f"По частям: {c['by_split']}; самые частые концы: {c['observed_end_top']}; с событием после 1200 с: "
           f"{c['n_with_event_after_1200']}; статусы: {c['status_counts']}.",
           f"Секунд сверх 1200: {c['excess_seconds_total']:.0f} — {100 * c['excess_share_of_their_matches_time']:.2f}% "
           f"игрового времени этих матчей; максимум конца периода {c['max_observed_end']}.", ""]
    (OUT / "diagnostics_audit.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
