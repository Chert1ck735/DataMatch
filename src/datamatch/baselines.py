"""Базовые показатели полевых игроков (M0/M1 турнира L7) и простое стягивание по амплуа.

Смысл и единицы
---------------
Таблица «игрок × матч» (`player_match_frame`) — суммы за матч и экспозиция в секундах:

* личные: `ishots` (попытки бросков, включая заблокированные и мимо), `igoals`, `a1`/`a2`
  (первая/вторая передача), `points` = igoals + a1 + a2, `ixg` (если дан xG);
* на льду, все ситуации: `gf`/`ga` (голы), `cf`/`ca` (попытки бросков), `xgf`/`xga` (если дан xG);
* на льду при 5 на 5: те же с суффиксом `_5v5`;
* экспозиция: `toi_rate_s`, `toi_rate_5v5_s` из L0 (`toi`), т.е. только stints с
  `valid_for_rate_metrics`.

Одинаковая выборка числителя и TOI (правило проекта): бросок/гол/передача учитываются, только если
их stint допущен к rate-метрикам (`valid_for_rate_metrics` таблицы stints — тот же флаг, по
которому L0 считает `toi_rate_*`); для 5 на 5 дополнительно `is_5v5` stint — как `toi_rate_5v5_s`.
On-ice атрибуция: каждый бросок приписывается всем полевым из `home_sk`/`away_sk` его stint:
своей команде со знаком «за», соперникам — «против». Вратари в списках полевых отсутствуют и
дополнительно исключаются по амплуа. Серия буллитов (period 0) и штрафные броски в узлах L0
отсутствуют; передачи берутся из событий с period > 0.

xG: необязательный `pd.Series` с MultiIndex (match_id, seq) по узлам-броскам. Без него
xG-колонки не создаются, а показатели «xG» явно заменяются попытками бросков (`cd_5v5_60`,
подпись «fallback»), чтобы контур работал до пересчёта L1.

Ставка на 60 минут: rate = Σ num / Σ toi_s × 3600 по одной и той же выборке строк
(строки с нулевой экспозицией исключаются целиком, как в L5/O0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .l5_shrink import ROLES

SECONDS_PER_HOUR = 3600.0
FIELD_ROLES = tuple(ROLES)
KEY = ["match_id", "player_id"]
# Формат таблицы показателя «игрок × матч» — общий вход проверочного контура (validation.py).
MF_COLS = ["player_id", "match_id", "role", "num", "toi_s"]
SHOT_ACTION = "shot"
ASSIST_PRIMARY, ASSIST_SECONDARY = "assist_primary", "assist_secondary"
# Минимальная экспозиция игрока для участия в оценке моментов стягивания (сек; 60 минут).
MIN_TOI_SHRINK_S = 60.0 * 60.0
# Матчи короче этого порога не входят в оценку σ² (num²/h неустойчиво при малом h), сек.
MIN_MATCH_TOI_MOMENTS_S = 60.0
# Минимум игроков амплуа для отдельных моментов; иначе — объединённые по всем амплуа.
MIN_ROLE_PLAYERS_SHRINK = 10
# Порог «межигровой дисперсии нет».
TAU2_EPS = 1e-12

NODE_COLUMNS = ["match_id", "seq", "stint", "period", "action", "actor_id", "team_id", "is_goal"]
STINT_COLUMNS = ["match_id", "stint", "home_sk", "away_sk", "home_team_id", "is_5v5",
                 "valid_for_rate_metrics"]
TOI_COLUMNS = ["match_id", "player_id", "team_id", "toi_rate_s", "toi_rate_5v5_s"]
EVENT_COLUMNS = ["match_id", "stint", "period", "event_type", "actor_id"]
COUNT_COLS = ["ishots", "igoals", "a1", "a2", "gf", "ga", "cf", "ca",
              "gf_5v5", "ga_5v5", "cf_5v5", "ca_5v5"]
XG_COLS = ["ixg", "xgf", "xga", "xgf_5v5", "xga_5v5"]


def _require(df: pd.DataFrame, cols, what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{what}: нет колонок {missing}")


def _stint_flags(stints: pd.DataFrame) -> pd.DataFrame:
    _require(stints, STINT_COLUMNS, "stints")
    st = stints[["match_id", "stint", "is_5v5", "valid_for_rate_metrics"]]
    return st.astype({"is_5v5": bool, "valid_for_rate_metrics": bool})


def shot_table(nodes: pd.DataFrame, stints: pd.DataFrame, xg: pd.Series | None = None) -> pd.DataFrame:
    """Узлы-броски с флагами stint: `valid` (rate-допуск), `is_5v5`, `is_home`, `xg` (если дан)."""
    _require(nodes, NODE_COLUMNS, "nodes")
    s = nodes.loc[(nodes.action == SHOT_ACTION) & (nodes.period > 0), NODE_COLUMNS]
    flags = _stint_flags(stints)
    home = stints[["match_id", "stint", "home_team_id"]]
    s = s.merge(flags, on=["match_id", "stint"], how="left").merge(home, on=["match_id", "stint"], how="left")
    s = s.assign(valid=s.valid_for_rate_metrics.fillna(False).astype(bool),
                 is_5v5=s.is_5v5.fillna(False).astype(bool),
                 is_home=(s.team_id == s.home_team_id).to_numpy(),
                 is_goal=s.is_goal.astype(bool))
    s = s.drop(columns=["valid_for_rate_metrics", "home_team_id"]).sort_values(["match_id", "seq"])
    s = s.reset_index(drop=True)
    if xg is None:
        return s
    vals = xg.reindex(pd.MultiIndex.from_arrays([s.match_id, s.seq])).to_numpy(dtype=float)
    if np.isnan(vals[s.valid.to_numpy()]).any():
        raise ValueError("xG: нет значения для части допущенных бросков (ключ match_id, seq)")
    return s.assign(xg=np.nan_to_num(vals))


def onice_incidence(shots: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Полевые на льду для допущенных бросков: side = +1 атакующие, −1 обороняющиеся."""
    v = shots[shots.valid].reset_index().rename(columns={"index": "shot_row"})
    sk = stints[["match_id", "stint", "home_sk", "away_sk"]]
    v = v.merge(sk, on=["match_id", "stint"], how="left")
    parts = []
    for col, home_side in (("home_sk", True), ("away_sk", False)):
        e = v[["shot_row", "match_id", "is_home", "is_5v5", col]].explode(col).dropna(subset=[col])
        side = np.where(e.is_home.to_numpy() == home_side, 1, -1).astype(np.int8)
        parts.append(pd.DataFrame({"shot_row": e.shot_row.to_numpy(), "match_id": e.match_id.to_numpy(),
                                   "player_id": e[col].to_numpy(), "side": side,
                                   "is_5v5": e.is_5v5.to_numpy(dtype=bool)}))
    return pd.concat(parts, ignore_index=True)


def onice_sums(shots: pd.DataFrame, inc: pd.DataFrame) -> pd.DataFrame:
    """On-ice суммы «игрок × матч»: голы, попытки и xG за/против, все ситуации и 5 на 5."""
    goal = shots.is_goal.to_numpy(dtype=float)[inc.shot_row.to_numpy()]
    f = (inc.side.to_numpy() > 0).astype(float)
    a = 1.0 - f
    s5 = inc.is_5v5.to_numpy(dtype=float)
    cols = {"gf": goal * f, "ga": goal * a, "cf": f, "ca": a}
    if "xg" in shots:
        x = shots.xg.to_numpy(dtype=float)[inc.shot_row.to_numpy()]
        cols.update({"xgf": x * f, "xga": x * a})
    cols.update({f"{k}_5v5": v * s5 for k, v in list(cols.items())})
    out = pd.DataFrame({"match_id": inc.match_id.to_numpy(), "player_id": inc.player_id.to_numpy(), **cols})
    return out.groupby(KEY).sum()


def assist_table(events: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Передачи к голу (period > 0) в допущенных stints: `a1`, `a2` по (match_id, player_id)."""
    _require(events, EVENT_COLUMNS, "events")
    e = events.loc[events.event_type.isin([ASSIST_PRIMARY, ASSIST_SECONDARY]) & (events.period > 0)]
    e = e.merge(_stint_flags(stints), on=["match_id", "stint"], how="left")
    e = e[e.valid_for_rate_metrics.fillna(False).astype(bool)]
    out = pd.DataFrame({"match_id": e.match_id.to_numpy(), "player_id": e.actor_id.to_numpy(),
                        "a1": (e.event_type == ASSIST_PRIMARY).to_numpy(dtype=float),
                        "a2": (e.event_type == ASSIST_SECONDARY).to_numpy(dtype=float)})
    return out.groupby(KEY).sum()


def individual_sums(shots: pd.DataFrame) -> pd.DataFrame:
    """Личные броски, голы и ixG по допущенным броскам."""
    v = shots[shots.valid]
    cols = {"ishots": np.ones(len(v)), "igoals": v.is_goal.to_numpy(dtype=float)}
    if "xg" in v:
        cols["ixg"] = v.xg.to_numpy(dtype=float)
    out = pd.DataFrame({"match_id": v.match_id.to_numpy(), "player_id": v.actor_id.to_numpy(), **cols})
    return out.groupby(KEY).sum()


def player_roles(players: pd.DataFrame) -> pd.Series:
    """Амплуа полевых из players.parquet (C/W/D); вратари и неизвестные исключаются."""
    _require(players, ["player_id", "role"], "players")
    p = players[players.role.isin(FIELD_ROLES)]
    return pd.Series(p.role.to_numpy(), index=p.player_id.to_numpy(), name="role")


def player_match_frame(nodes: pd.DataFrame, stints: pd.DataFrame, toi: pd.DataFrame,
                       events: pd.DataFrame, roles: pd.Series, xg: pd.Series | None = None) -> pd.DataFrame:
    """Таблица «игрок × матч» полевых: TOI (rate-допуск) и все суммы из модульного docstring."""
    _require(toi, TOI_COLUMNS, "toi")
    shots = shot_table(nodes, stints, xg)
    parts = [onice_sums(shots, onice_incidence(shots, stints)), individual_sums(shots),
             assist_table(events, stints)]
    base = toi[TOI_COLUMNS].set_index(KEY)
    pm = base.join(parts, how="outer")
    num_cols = [c for c in pm.columns if c != "team_id"]
    pm[num_cols] = pm[num_cols].fillna(0.0).astype(float)
    pm = pm.reindex(columns=["team_id", "toi_rate_s", "toi_rate_5v5_s"] + COUNT_COLS
                    + [c for c in XG_COLS if c in pm.columns], fill_value=0.0)
    pm["points"] = pm.igoals + pm.a1 + pm.a2
    pm = pm.reset_index()
    pm.insert(2, "role", pm.player_id.map(roles))
    return pm[pm.role.isin(FIELD_ROLES)].reset_index(drop=True)


@dataclass(frozen=True)
class RateMetric:
    """Показатель на 60: (Σ plus − Σ minus) / Σ toi_col × 3600."""
    name: str
    plus: tuple[str, ...]
    minus: tuple[str, ...]
    toi_col: str
    label: str
    needs_xg: bool = False


METRICS: dict[str, RateMetric] = {m.name: m for m in [
    RateMetric("points_60", ("points",), (), "toi_rate_s", "очки/60 (все ситуации)"),
    RateMetric("igoals_60", ("igoals",), (), "toi_rate_s", "личные голы/60"),
    RateMetric("ishots_60", ("ishots",), (), "toi_rate_s", "личные попытки бросков/60"),
    RateMetric("ixg_60", ("ixg",), (), "toi_rate_s", "личный xG/60", True),
    RateMetric("gd_60", ("gf",), ("ga",), "toi_rate_s", "on-ice GD/60 (все ситуации)"),
    RateMetric("gd_5v5_60", ("gf_5v5",), ("ga_5v5",), "toi_rate_5v5_s", "on-ice GD/60 5×5"),
    RateMetric("cd_5v5_60", ("cf_5v5",), ("ca_5v5",), "toi_rate_5v5_s",
               "on-ice разность попыток бросков/60 5×5 (fallback вместо xGD, пока нет xG)"),
    RateMetric("xgd_60", ("xgf",), ("xga",), "toi_rate_s", "on-ice xGD/60 (все ситуации)", True),
    RateMetric("xgd_5v5_60", ("xgf_5v5",), ("xga_5v5",), "toi_rate_5v5_s", "on-ice xGD/60 5×5 (M1)", True),
]}


def available_metrics(pm: pd.DataFrame) -> list[str]:
    """Показатели, для которых в таблице есть все колонки (xG-показатели — только с xG)."""
    return [n for n, m in METRICS.items() if all(c in pm.columns for c in (*m.plus, *m.minus, m.toi_col))]


def metric_frame(pm: pd.DataFrame, metric: RateMetric) -> pd.DataFrame:
    """Таблица показателя в формате MF_COLS; строки с нулевой экспозицией исключаются целиком."""
    _require(pm, ["player_id", "match_id", "role", *metric.plus, *metric.minus, metric.toi_col], metric.name)
    num = pm[list(metric.plus)].sum(axis=1).to_numpy(dtype=float)
    if metric.minus:
        num = num - pm[list(metric.minus)].sum(axis=1).to_numpy(dtype=float)
    toi = pm[metric.toi_col].to_numpy(dtype=float)
    keep = toi > 0
    return pd.DataFrame({"player_id": pm.player_id.to_numpy()[keep], "match_id": pm.match_id.to_numpy()[keep],
                         "role": pm.role.to_numpy()[keep], "num": num[keep], "toi_s": toi[keep]})


def player_rates(mf: pd.DataFrame) -> pd.DataFrame:
    """Суммы и ставка на 60 по игроку: num, toi_s, games, rate_60 (одна выборка строк)."""
    _require(mf, MF_COLS, "metric frame")
    g = mf.groupby("player_id", sort=True).agg(role=("role", "first"), num=("num", "sum"),
                                               toi_s=("toi_s", "sum"), games=("match_id", "nunique"))
    g["rate_60"] = np.where(g.toi_s > 0, g.num / g.toi_s.where(g.toi_s > 0) * SECONDS_PER_HOUR, np.nan)
    return g.reset_index()


# --- эмпирический Байес по амплуа ----------------------------------------------------------

@dataclass(frozen=True)
class ShrinkParams:
    """Моменты стягивания по амплуа: μ_r, τ²_r (межигровая), σ²_r (шум на час игры)."""
    mu: dict
    tau2: dict
    sigma2: dict
    n: dict
    pooled: dict = field(default_factory=dict)


def _sigma2_terms(mf: pd.DataFrame) -> tuple[float, float]:
    """Σ_i Σ_m h_im (x_im − x̄_i)² и Σ_i (M_i − 1); x — ставка на час, h — часы."""
    m = mf[mf.toi_s >= MIN_MATCH_TOI_MOMENTS_S]
    h = m.toi_s.to_numpy() / SECONDS_PER_HOUR
    q = pd.DataFrame({"p": m.player_id.to_numpy(), "n2h": m.num.to_numpy() ** 2 / h,
                      "num": m.num.to_numpy(), "h": h, "k": 1.0}).groupby("p").sum()
    q = q[q.k >= 2]
    ss = float((q.n2h - q.num ** 2 / q.h).sum())
    return ss, float((q.k - 1).sum())


def _moments(mf: pd.DataFrame, min_toi_s: float) -> dict:
    """μ, σ², τ² одной группы (формулы — в `fit_role_shrinkage`)."""
    ss, dof = _sigma2_terms(mf)
    sigma2 = ss / dof if dof > 0 else 0.0
    r = player_rates(mf)
    r = r[r.toi_s >= max(min_toi_s, np.finfo(float).tiny)]
    h = r.toi_s.to_numpy() / SECONDS_PER_HOUR
    x = r.rate_60.to_numpy()
    n = len(r)
    if n < 2:
        mu = float((x * h).sum() / h.sum()) if n else 0.0
        return {"mu": mu, "sigma2": sigma2, "tau2": 0.0, "n": n}
    mu = float((x * h).sum() / h.sum())
    q = float((h * (x - mu) ** 2).sum())
    denom = float(h.sum() - (h ** 2).sum() / h.sum())
    tau2 = max(0.0, (q - (n - 1) * sigma2) / denom) if denom > 0 else 0.0
    return {"mu": mu, "sigma2": sigma2, "tau2": tau2, "n": n}


def fit_role_shrinkage(mf: pd.DataFrame, min_toi_s: float = MIN_TOI_SHRINK_S) -> ShrinkParams:
    """Моменты эмпирического Байеса по амплуа (метод моментов, веса — TOI в часах h).

    Модель: x_i = θ_i + e_i, θ_i ~ (μ_r, τ²_r), Var(e_i) = σ²_r / h_i, x_i — ставка на 60.

    * μ_r = Σ h_i x_i / Σ h_i;
    * σ²_r = Σ_i Σ_m h_im (x_im − x̄_i)² / Σ_i (M_i − 1) — разброс матчей внутри игрока
      (несмещённо при Var(x_im) = σ²/h_im; матчи короче MIN_MATCH_TOI_MOMENTS_S исключены);
    * Q = Σ h_i (x_i − μ_r)², E[Q] = (N − 1) σ² + τ² (Σh − Σh²/Σh) ⇒
      τ²_r = max(0, (Q − (N − 1) σ²_r) / (Σh − Σh²/Σh)) (DerSimonian–Laird с весами h_i).

    В Q входят игроки с TOI ≥ `min_toi_s`. Амплуа с < MIN_ROLE_PLAYERS_SHRINK игроков получает
    объединённые по всем амплуа моменты (`pooled[r] = True`).
    """
    _require(mf, MF_COLS, "metric frame")
    bad = sorted(set(mf.role.dropna()) - set(FIELD_ROLES))
    if bad or mf.role.isna().any():
        raise ValueError(f"fit_role_shrinkage: неизвестные role {bad} (ожидаются {FIELD_ROLES})")
    pooled_m = _moments(mf, min_toi_s)
    out = {"mu": {}, "tau2": {}, "sigma2": {}, "n": {}, "pooled": {}}
    for r in FIELD_ROLES:
        m = _moments(mf[mf.role == r], min_toi_s)
        use_pool = m["n"] < MIN_ROLE_PLAYERS_SHRINK
        src = pooled_m if use_pool else m
        for k in ("mu", "tau2", "sigma2", "n"):
            out[k][r] = src[k]
        out["pooled"][r] = bool(use_pool)
    return ShrinkParams(**out)


def apply_role_shrinkage(rates: pd.DataFrame, params: ShrinkParams) -> pd.DataFrame:
    """θ̂_i = μ_r + B_i (x_i − μ_r), B_i = τ²_r / (τ²_r + σ²_r / h_i); без TOI или τ² ≈ 0 — θ̂ = μ_r.

    Возвращает копию `rates` с колонками `mu_r`, `b`, `rate_shrunk` (вход не изменяется).
    """
    _require(rates, ["player_id", "role", "toi_s", "rate_60"], "rates")
    roles = rates.role.to_numpy(dtype=object)
    missing = sorted(set(roles) - set(params.mu))
    if missing:
        raise ValueError(f"apply_role_shrinkage: нет моментов для role {missing}")
    mu = np.array([params.mu[r] for r in roles], dtype=float)
    tau2 = np.array([params.tau2[r] for r in roles], dtype=float)
    sigma2 = np.array([params.sigma2[r] for r in roles], dtype=float)
    h = rates.toi_s.to_numpy(dtype=float) / SECONDS_PER_HOUR
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where((h > 0) & (tau2 > TAU2_EPS), tau2 / (tau2 + sigma2 / np.where(h > 0, h, 1.0)), 0.0)
    x = np.nan_to_num(rates.rate_60.to_numpy(dtype=float), nan=0.0)
    return rates.assign(mu_r=mu, b=b, rate_shrunk=mu + b * (x - mu))


def shrunk_rates(mf: pd.DataFrame, params: ShrinkParams | None = None,
                 min_toi_s: float = MIN_TOI_SHRINK_S) -> pd.DataFrame:
    """Сырые и стянутые ставки игроков; моменты подгоняются на `mf`, если не заданы."""
    p = fit_role_shrinkage(mf, min_toi_s) if params is None else params
    return apply_role_shrinkage(player_rates(mf), p)


# --- загрузка L0 (только нужные колонки) ---------------------------------------------------

def _read_stints(path: Path, keys: pd.DataFrame) -> pd.DataFrame:
    """Stints только для пар (match_id, stint) из `keys`: списки составов не грузятся целиком."""
    tbl = pq.read_table(path, columns=STINT_COLUMNS)
    idx = pd.MultiIndex.from_arrays([tbl.column("match_id").to_numpy(), tbl.column("stint").to_numpy()])
    need = pd.MultiIndex.from_frame(keys[["match_id", "stint"]].drop_duplicates())
    return tbl.filter(idx.isin(need)).to_pandas()


def load_l0_split(cache_dir: str | Path, split: str) -> dict[str, pd.DataFrame]:
    """Чтение таблиц L0 одной части (train/test) с явным списком колонок и фильтрами."""
    d = Path(cache_dir)
    nodes = pd.read_parquet(d / f"{split}_nodes.parquet", columns=NODE_COLUMNS,
                            filters=[("action", "==", SHOT_ACTION)])
    events = pd.read_parquet(d / f"{split}_events.parquet", columns=EVENT_COLUMNS,
                             filters=[("event_type", "in", [ASSIST_PRIMARY, ASSIST_SECONDARY])])
    keys = pd.concat([nodes[["match_id", "stint"]], events[["match_id", "stint"]]])
    return {"nodes": nodes, "events": events, "stints": _read_stints(d / f"{split}_stints.parquet", keys),
            "toi": pd.read_parquet(d / f"{split}_toi.parquet", columns=TOI_COLUMNS)}


def load_player_match(cache_dir: str | Path, split: str, roles: pd.Series,
                      xg: pd.Series | None = None) -> pd.DataFrame:
    """`player_match_frame` по кэшу L0 одной части."""
    t = load_l0_split(cache_dir, split)
    return player_match_frame(t["nodes"], t["stints"], t["toi"], t["events"], roles, xg=xg)
