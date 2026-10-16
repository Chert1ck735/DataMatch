"""Чтение исходных данных (только чтение, исходные файлы не изменяются)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import atomic_write
from .config import Config, file_fingerprint, get_logger

RAW_FILES_TRAIN = ["events.parquet", "games.csv", "players.csv", "player_teams.csv", "teams.csv", "rink_geometry.csv"]
RAW_FILES_TEST = ["events_test.parquet", "games_test.csv", "player_seasons_test.csv", "players_new.csv"]

EVENT_COLS = ["match_id", "seq", "event_type", "actor_id", "actor_team_id", "other_id", "other_team_id",
              "period", "clock_s", "zone", "x", "y", "x2", "y2", "n_home", "n_away", "net_x", "net_y"]


def data_version(cfg: Config) -> dict:
    """Идентификаторы версий данных (размер + sha256); содержимое читается всегда.

    Отпечаток входит в ключ кэша каждого слоя, и единственное заявленное свойство провенанса —
    «другие данные дают другой ключ». Прежняя оптимизация возвращала сохранённый sha256, если
    совпали размер и mtime, поэтому восстановление данных с сохранением времён (`cp -p`,
    `rsync --times`, распаковка архива) оставляло ключ прежним, и весь конвейер отдавал
    результаты, посчитанные на других данных. Чтение всех сырых файлов занимает доли секунды
    (около 60 МБ), что несопоставимо со стоимостью такой ошибки.
    """
    cache_f = cfg.paths.cache / "data_version.json"
    files = [cfg.paths.data_dir / f for f in RAW_FILES_TRAIN] + [cfg.paths.test_dir / f for f in RAW_FILES_TEST]
    version = {f.name: file_fingerprint(f) for f in files}
    stamp = {str(f): [f.stat().st_size, f.stat().st_mtime] for f in files}
    atomic_write(cache_f, lambda t: t.write_text(
        json.dumps({"stamp": stamp, "version": version}, ensure_ascii=False, indent=1),
        encoding="utf-8"))
    return version


def events_path(cfg: Config, source: str) -> Path:
    return cfg.paths.data_dir / "events.parquet" if source == "train" else cfg.paths.test_dir / "events_test.parquet"


def event_match_ids(cfg: Config, source: str) -> list[str]:
    import pyarrow.parquet as pq
    t = pq.read_table(events_path(cfg, source), columns=["match_id"])
    return sorted(set(t.column("match_id").to_pylist()))


def load_event_columns(cfg: Config, source: str, columns: list[str], event_prefixes: tuple[str, ...] = ()) -> pd.DataFrame:
    """Лёгкое чтение нескольких столбцов (опционально только события с заданными префиксами типа)."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    cols = list(dict.fromkeys(["event_type"] + list(columns)))
    t = pq.read_table(events_path(cfg, source), columns=cols)
    if event_prefixes:
        m = None
        for p in event_prefixes:
            mm = pc.starts_with(t.column("event_type"), p)
            m = mm if m is None else pc.or_(m, mm)
        t = t.filter(m)
    return t.to_pandas()


def load_events(cfg: Config, source: str, matches=None) -> pd.DataFrame:
    """События (все или только матчей `matches`); чтение порцией через фильтр pyarrow."""
    import pyarrow.dataset as ds
    f = events_path(cfg, source)
    if matches is None:
        ev = pd.read_parquet(f, columns=EVENT_COLS)
    else:
        d = ds.dataset(f, format="parquet")
        ev = d.to_table(columns=EVENT_COLS, filter=ds.field("match_id").isin(list(matches))).to_pandas()
    ev = ev[EVENT_COLS].copy()
    for c in ["match_id", "event_type", "actor_id", "actor_team_id", "other_id", "other_team_id", "zone"]:
        ev[c] = ev[c].astype(object).where(ev[c].notna(), None)
    ev["n_home"] = ev["n_home"].astype("float64")
    ev["n_away"] = ev["n_away"].astype("float64")
    ev["period"] = ev["period"].astype(np.int16)
    ev = ev.sort_values(["match_id", "seq"], kind="mergesort").reset_index(drop=True)
    return ev


def load_games(cfg: Config, source: str) -> pd.DataFrame:
    if source == "train":
        g = pd.read_csv(cfg.paths.data_dir / "games.csv")
    else:
        g = pd.read_csv(cfg.paths.test_dir / "games_test.csv")
    for c in g.columns:
        if g[c].dtype != object and str(g[c].dtype).startswith("str"):
            g[c] = g[c].astype(object)
    g["match_id"] = g["match_id"].astype(str)
    return g


def load_rink(cfg: Config) -> pd.DataFrame:
    return pd.read_csv(cfg.paths.data_dir / "rink_geometry.csv")


class GeometryError(ValueError):
    """Неполная или противоречивая геометрия площадки в rink_geometry.csv."""


def rink_geometry(cfg: Config) -> dict[str, dict]:
    """Геометрия площадок из rink_geometry.csv — единственный источник (P011).

    Для каждого типа: границы (x_min, x_max, y_min, y_max), линии ворот и синие линии своей
    и чужой зоны, полуширина ворот по штангам. Проверяются полнота, порядок линий, симметрия
    и правдоподобность единиц (метры); при нарушении — GeometryError с причиной.
    """
    r = load_rink(cfg)
    out = {}
    for rt, g in r.groupby("rink_type"):
        def val(obj, kind, col):
            m = g[(g.object == obj) & (g.kind == kind)][col].dropna()
            if len(m) != 1:
                raise GeometryError(f"{rt}: нет однозначного значения {obj}/{kind}/{col}")
            return float(m.iat[0])
        posts = g[g.object.isin(["goal_opp_post_y_plus", "goal_opp_post_y_minus"])]
        if len(posts) != 2:
            raise GeometryError(f"{rt}: нет двух штанг ворот")
        d = {"x_min": val("rink_boundary", "x_min", "x"), "x_max": val("rink_boundary", "x_max", "x"),
             "y_min": val("rink_boundary", "y_min", "y"), "y_max": val("rink_boundary", "y_max", "y"),
             "goal_line_own": val("goal_line_own", "x", "x"), "goal_line_opp": val("goal_line_opp", "x", "x"),
             "blue_line_own": val("blue_line_own", "x", "x"), "blue_line_opp": val("blue_line_opp", "x", "x"),
             "center_line": val("center_line", "x", "x"),
             "post_half_width": float((posts.y.max() - posts.y.min()) / 2)}
        order = [d["x_min"], d["goal_line_own"], d["blue_line_own"], d["center_line"], d["blue_line_opp"],
                 d["goal_line_opp"], d["x_max"]]
        if not all(a < b for a, b in zip(order, order[1:])):
            raise GeometryError(f"{rt}: нарушен порядок линий по x {order}")
        if not (d["y_min"] < 0 < d["y_max"]):
            raise GeometryError(f"{rt}: границы по y не охватывают ось")
        if abs(d["goal_line_own"] + d["goal_line_opp"]) > 1e-6 or abs(d["blue_line_own"] + d["blue_line_opp"]) > 1e-6:
            raise GeometryError(f"{rt}: линии своей и чужой зоны несимметричны")
        if not (20 <= d["x_max"] <= 40 and 10 <= d["y_max"] <= 20 and 0.5 <= d["post_half_width"] <= 1.5):
            raise GeometryError(f"{rt}: размеры вне правдоподобного диапазона для метров {d}")
        out[rt] = d
    return out


def goal_line_x(cfg: Config) -> dict[str, float]:
    return {k: v["goal_line_opp"] for k, v in rink_geometry(cfg).items()}


def load_players(cfg: Config) -> pd.DataFrame:
    p1 = pd.read_csv(cfg.paths.data_dir / "players.csv")
    p2 = pd.read_csv(cfg.paths.test_dir / "players_new.csv")
    p1["source"] = "players"
    p2["source"] = "players_new"
    p = pd.concat([p1, p2], ignore_index=True)
    for c in ["player_id", "position", "lineup_slot_mode", "shoots", "source"]:
        p[c] = p[c].astype(object).where(p[c].notna(), None)
    assert p.player_id.is_unique, "player_id не уникален в справочниках"
    return p


def load_player_teams(cfg: Config) -> pd.DataFrame:
    return pd.read_csv(cfg.paths.data_dir / "player_teams.csv")


def load_player_seasons_test(cfg: Config) -> pd.DataFrame:
    return pd.read_csv(cfg.paths.test_dir / "player_seasons_test.csv")


POS_TO_ROLE = {"C": "C", "LW": "W", "RW": "W", "D": "D", "G": "G"}
SLOT_TO_ROLE = {"C": "C", "LW": "W", "RW": "W", "LD": "D", "RD": "D", "G": "G"}


def resolve_roles(players: pd.DataFrame, slot_usage: pd.DataFrame | None = None) -> pd.DataFrame:
    """Амплуа игрока: C / W / D / G — по фактическому использованию, справочник — запасной источник.

    slot_usage (по событиям train и test): player_id, n_lineup_G, n_lineup_skater (сколько раз
    заявлен вратарём / полевым), skater_slot_mode (самый частый полевой слот), n_shift_G,
    n_shift_skater (выходы через слот G / полевые слоты).

    Правило:
    1. Есть заявки: вратарь, если заявок в слот G больше, чем полевых. Если справочник
       (position) с этим расходится — конфликт фиксируется в обе стороны, роль — по заявкам;
       амплуа полевого — position, если это полевая позиция, иначе самый частый полевой слот.
    2. Заявок нет: position -> lineup_slot_mode (пустой position при lineup_slot_mode = G -> вратарь)
       -> смены (вратарь, если выходов через слот G больше, чем полевых).
    Роль игрока в КОНКРЕТНОМ матче (для слота G при реконструкции составов) определяется
    отдельно — по заявке этого матча (l0_shifts.match_goalies).
    """
    p = players.copy()
    pos_role = p["position"].map(POS_TO_ROLE)
    p["role"] = pos_role
    p["role_source"] = np.where(p["role"].notna(), "position", None)
    m = p["role"].isna() & p["lineup_slot_mode"].notna()
    p.loc[m, "role"] = p.loc[m, "lineup_slot_mode"].map(SLOT_TO_ROLE)
    p.loc[m, "role_source"] = "lineup_slot_mode"
    p["conflict_position_G_plays_skater"] = False
    p["conflict_position_skater_plays_G"] = False
    p["goalie_conflict"] = False
    if slot_usage is not None:
        su = slot_usage.set_index("player_id")
        nG = p.player_id.map(su.get("n_lineup_G", pd.Series(dtype=float))).fillna(0).values
        nS = p.player_id.map(su.get("n_lineup_skater", pd.Series(dtype=float))).fillna(0).values
        has_lu = (nG + nS) > 0
        lu_goalie = nG > nS
        skater_slot = p.player_id.map(su.get("skater_slot_mode", pd.Series(dtype=object))).map(SLOT_TO_ROLE)
        is_g_now = (p["role"] == "G").values
        c1 = has_lu & ~lu_goalie & is_g_now                         # справочник G, играет полевым
        c2 = has_lu & lu_goalie & ~is_g_now & p["role"].notna().values  # справочник полевой, заявлен вратарём
        p["conflict_position_G_plays_skater"] = c1
        p["conflict_position_skater_plays_G"] = c2
        p["goalie_conflict"] = c1 | c2
        # роль по заявкам
        p.loc[c1, "role"] = skater_slot[c1].values
        p.loc[c1, "role_source"] = "lineup_usage_skater"
        p.loc[c2, "role"] = "G"
        p.loc[c2, "role_source"] = "lineup_usage_goalie"
        # без справочника — по заявкам
        m = p["role"].isna() & has_lu
        p.loc[m, "role"] = np.where(lu_goalie[m.values], "G", skater_slot[m].values)
        p.loc[m, "role_source"] = "lineup_usage"
        # без заявок — по сменам
        g = p.player_id.map(su.get("n_shift_G", pd.Series(dtype=float))).fillna(0) > \
            p.player_id.map(su.get("n_shift_skater", pd.Series(dtype=float))).fillna(0)
        m2 = p["role"].isna() & g
        p.loc[m2, "role"] = "G"
        p.loc[m2, "role_source"] = "shift_G"
    p["is_goalie"] = p["role"] == "G"
    return p
