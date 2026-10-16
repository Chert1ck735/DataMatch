"""Разбиения матчей train: внешний блок 20% и K-fold внутренней валидации на 80%.

Стратификация по компонентам связности графа «команда — матч» (лиги неизвестны,
компонента — их наблюдаемый заменитель). Разбиение по матчам: события одного матча
никогда не делятся между частями. Фиксируется до настройки моделей и сохраняется в
artifacts/splits.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import Config, get_logger


def team_components(games: pd.DataFrame) -> dict[str, int]:
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    games = games.sort_values("match_id", kind="stable")
    for h, a in zip(games.home_team_id, games.away_team_id):
        ra, rb = find(h), find(a)
        if ra != rb:
            parent[ra] = rb
    roots = {t: find(t) for t in parent}
    # Equal-sized components use their sorted team identities, never encounter order.
    members = {}
    for team, root in roots.items():
        members.setdefault(root, []).append(str(team))
    counts = games.home_team_id.map(roots).value_counts().to_dict()
    order = sorted(members, key=lambda r: (-counts[r], tuple(sorted(members[r]))))
    rid = {r: i for i, r in enumerate(order)}
    return {t: rid[r] for t, r in roots.items()}


def make_splits(cfg: Config, games_train: pd.DataFrame, games_test: pd.DataFrame, scheme: pd.Series) -> dict:
    rng = np.random.default_rng(cfg.seed)
    comp = team_components(games_train)
    g = games_train.sort_values("match_id", kind="stable").reset_index(drop=True).copy()
    g["component"] = g.home_team_id.map(comp)
    sizes = g.component.value_counts()
    # маленькие компоненты (< 10 матчей) объединяются в один страт
    g["stratum"] = np.where(g.component.map(sizes) >= 10, g.component, -1)
    outer_frac = float(cfg["splits"]["outer_frac"])
    K = int(cfg["splits"]["inner_k"])
    g["outer"] = False
    g["fold"] = -1
    for s, sub in g.groupby("stratum"):
        ids = sub.sort_values("match_id", kind="stable").match_id.to_numpy()
        ids = ids[rng.permutation(len(ids))]
        n_out = int(round(outer_frac * len(ids)))
        out_ids = set(ids[:n_out])
        rest = ids[n_out:]
        g.loc[g.match_id.isin(out_ids), "outer"] = True
        # K-fold по оставшимся: циклическое назначение после перемешивания
        folds = np.arange(len(rest)) % K
        g.loc[g.match_id.isin(rest), "fold"] = g.loc[g.match_id.isin(rest), "match_id"].map(dict(zip(rest, folds)))
    g["scheme"] = g.match_id.map(scheme).fillna("v1")
    # компонента КХЛ: компонента train, содержащая больше всего команд теста
    test_teams = set(games_test.home_team_id) | set(games_test.away_team_id)
    tc = pd.Series({t: comp.get(t) for t in test_teams})
    tc_counts = tc.dropna().value_counts()
    if tc_counts.empty:
        raise ValueError("No test team belongs to a training component")
    khl = int(sorted(tc_counts.index, key=lambda c: (-tc_counts[c], c))[0])
    res = {
        "seed": cfg.seed,
        "outer_frac": outer_frac,
        "inner_k": K,
        "khl_component": khl,
        "test_teams": len(test_teams),
        "test_teams_in_khl_component": int((tc == khl).sum()),
        "test_teams_not_in_train": int(tc.isna().sum()),
        "khl_component_teams": int(sum(1 for v in comp.values() if v == khl)),
        "n_components": int(len(set(comp.values()))),
        "component_sizes_matches": {int(k): int(v) for k, v in sizes.items()},
        "matches": g[["match_id", "component", "stratum", "outer", "fold", "scheme"]].to_dict(orient="list"),
    }
    from .cache import frame_hash
    res["assignment_hash"] = frame_hash(g[["match_id", "component", "stratum", "outer", "fold", "scheme"]])
    return res


def splits_provenance(cache, games_train: pd.DataFrame, games_test: pd.DataFrame, scheme: pd.Series) -> dict:
    """Разбиение зависит от матчей train (команды), команд test, схем разметки, seed и параметров."""
    from .cache import frame_hash
    gt = games_train[["match_id", "home_team_id", "away_team_id"]].sort_values("match_id").reset_index(drop=True)
    te = games_test[["home_team_id", "away_team_id"]].sort_values(["home_team_id", "away_team_id"]).reset_index(drop=True)
    sc = scheme.sort_index().rename("scheme").rename_axis("match_id").reset_index()
    return cache.provenance("splits", ("splits", "seed"), ("splits",), games=frame_hash(gt), test=frame_hash(te),
                            scheme=frame_hash(sc))


def save_splits(cfg: Config, splits: dict, provenance: dict):
    from .cache import save_json_with_provenance
    f = cfg.paths.artifacts / "splits.json"
    save_json_with_provenance(f, splits, provenance)
    get_logger().info("разбиения сохранены: %s", f)


def load_splits(cfg: Config, provenance: dict | None = None) -> dict | None:
    """С provenance — только актуальное разбиение; без — как есть (для диагностических скриптов)."""
    from .cache import load_checked_json
    f = cfg.paths.artifacts / "splits.json"
    if provenance is not None:
        return load_checked_json(f, provenance)
    return json.loads(f.read_text()) if f.exists() else None


def splits_hash(splits: dict) -> str:
    """Хэш разбиения в JSON-представлении: свежее и загруженное из splits.json дают одно значение
    (int-ключи словарей после JSON становятся строками; проверка 06.10.2026)."""
    import json

    from .cache import _json_default, obj_hash
    body = {k: v for k, v in splits.items() if k != "provenance"}
    return obj_hash(json.loads(json.dumps(body, default=_json_default)))


def split_frame(splits: dict) -> pd.DataFrame:
    return pd.DataFrame(splits["matches"])


def matches_for(splits: dict, part: str, fold: int | None = None, schemes=("v1",)) -> list[str]:
    """part: 'dev' (80%), 'outer' (20%), 'train_all', 'inner_train' (dev без fold), 'inner_val' (fold)."""
    f = split_frame(splits)
    f = f[f.scheme.isin(schemes)]
    if part == "dev":
        m = ~f.outer
    elif part == "outer":
        m = f.outer
    elif part == "train_all":
        m = pd.Series(True, index=f.index)
    elif part == "inner_train":
        m = (~f.outer) & (f.fold != fold)
    elif part == "inner_val":
        m = (~f.outer) & (f.fold == fold)
    else:
        raise ValueError(part)
    return f.match_id[m].tolist()
