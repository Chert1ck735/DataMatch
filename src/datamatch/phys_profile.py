"""Ветка Ф: физический профиль игрока — Ф1 (скорость с шайбой) и Ф2 (темп перехода, скорость
шайбы в передачах). Ф3 (модель запаса сил) и Ф4 (огибающая) здесь не реализованы.

Смысл и границы применимости (docs/METHODOLOGY.md, физический профиль):
- личные показатели — описательный профиль для карточки L8; в модель ценности, xG и Overall не
  входят и xG/V не используют;
- трекинга нет: положение игрока известно только в моменты событий с шайбой, поэтому Ф1 —
  нижняя граница скорости С ШАЙБОЙ (путь между событиями не обязательно прямой), а не скорость
  катания; Ф2-скорость паса — тоже нижняя граница (время до следующего события получателя);
- координаты узлов L0 — метры в атакующей системе команды исполнителя (бросок: x ≈ +16,7 м во
  всех периодах). Оба события пары Ф1 принадлежат одному игроку и одной команде, а пас — точный,
  то есть в системе одной команды; смещения считаются без перевода систем, смена команды внутри
  пары запрещена явно.

Источник — узлы L0 (действия с шайбой; touch включает receive/carry/rebound, steal — intercept)
и события duel (в узлы L0 не входят, берутся из таблицы событий отдельным фильтром). Время —
причинная монотонная шкала t_mono; отрезок segment — промежуток между остановками.

Надёжность (L7, строка «Ф»): матчи делятся на две половины случайно с фиксированным seed
(порядок match_id в train — не время); split-half внутри амплуа (центрирование по амплуа,
затем Пирсон), поправка Спирмена–Брауна, парный бутстрэп по матчам; базовые строки — «только
амплуа» (внутри амплуа постоянна: её надёжность 0, гейт ≥ 0,3 сравнивает именно с ней) и
«on-ice звено» (взвешенное по совместным наблюдениям среднее показателя партнёров на льду в той
же половине предсказывает другую половину игрока). Стягивание к амплуа — модель L5:
шум σ²/n + ω²/M по разностям половин (регрессия без свободного члена), τ²_r — ковариация половин.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROLES = ("C", "W", "D")
SHOT_TYPES = frozenset({"goal", "shot_on_target", "shot_missed", "shot_blocked"})
PASS_TYPES = frozenset({"pass_complete", "pass_failed"})
# Ф1: предыдущее событие пары — шайба у игрока (touch ⊇ receive/carry/rebound; steal ⊇ intercept;
# duel). Пас, бросок, вброс/выброс и deke как предыдущее событие означают, что шайба ушла.
PAIR_PREV_TYPES = frozenset({"touch", "steal", "duel"})
PAIR_NEXT_TYPES = PAIR_PREV_TYPES | SHOT_TYPES | PASS_TYPES
# Метки на линиях (entry/exit/dump_in_*/clear_out_*) координату игрока не несут: entry/exit в узлы
# L0 не входят, dump/clear не входят в списки типов выше.

# Разрешение clock_s — 0,1 с (гейт L0 для ветки Ф): нижняя граница Δt = 0,5 с держит ошибку
# округления времени не выше ±20%; верхняя 3 с — дальше путь игрока с шайбой заметно
# непрямой и оценка теряет смысл «скорости». Обе границы — из спецификации Ф1.
PAIR_DT_MIN_S = 0.5            # строго больше
PAIR_DT_MAX_S = 3.0            # включительно
# Выше 13 м/с с шайбой (≈ 47 км/ч; рекордные отрезки забегов ~ 40 км/ч) — артефакт разметки.
SPEED_MAX_MPS = 13.0
# Показатель игрока Ф1 — p80 его пар: «потолок» скорости с шайбой, а не тактическое среднее.
SPEED_QUANTILE = 0.8
# Допуск сравнения Δt: t_mono кратно 0,1 с, разность в float даёт 0,5000000000000018.
DT_EPS_S = 1e-6

# Ф2, скорость шайбы: Δt ≥ 0,3 с — ошибка округления ≤ ±33% при разрешении 0,1 с (короче —
# в основном слитые/одновременные отметки); ≤ 3 с — дальше «следующее событие получателя»
# уже не приём; выше 40 м/с — сильнее самых жёстких передач (броски щелчком ~ 45 м/с): артефакт.
PASS_DT_MIN_S = 0.3
PASS_DT_MAX_S = 3.0
PASS_SPEED_MAX_MPS = 40.0

# Ф2, переход: возврат шайбы (steal/intercept) в своей зоне в первом узле владения; конец —
# первый узел владения в чужой зоне, с меткой входа или бросок. Только 5 на 5: в большинстве и
# меньшинстве темп определяется численностью, а не звеном (адаптация, см. отчёт).
TRANSITION_STRENGTH = "EV5"
# Доля быстрых атак: бросок не позже k с после возврата; k выбирается на train по сетке.
FAST_ATTACK_K_GRID_S = (4.0, 6.0, 8.0, 10.0, 12.0)

# Опровергающая проверка Ф1: p80 игрока не должна объясняться плотностью его пар (пар на матч)
# и долей коротких Δt (≤ 1 с — где ошибка округления времени даёт ложные высокие скорости).
# Порог: больше половины внутриамплуа-дисперсии p80 объяснено этими двумя величинами — показатель
# считается артефактом выборки пар. Игроки с < 10 пар не участвуют (квантиль слишком шумен).
SHORT_DT_S = 1.0
CONFOUND_R2_MAX = 0.5
CONFOUND_MIN_PAIRS = 10

RELIABILITY_GATE = 0.3          # split-half (Спирмен–Браун) внутри амплуа, спецификация L7/Ф
MIN_SPLIT_PLAYERS = 30          # меньше игроков с двумя половинами — надёжность не оценивается
MIN_ROLE_PLAYERS = 20           # меньше — амплуа не получает отдельной r и τ² (берётся общая)
MAX_MOMENT_COND = 1.0e8         # обусловленность плана σ²/ω²; выше — ω² не идентифицируется
HALF_SEED = 20261006            # фиксированное случайное разбиение матчей на половины
CI_LEVEL = 0.95
WEIGHT_HIGH = 0.7               # пороги веса стягивания, как в L5/L6
WEIGHT_MEDIUM = 0.4

PAIR_REQUIRED = ("match_id", "seq", "actor_id", "team_id", "period", "segment", "t_mono", "x", "y", "primary")
PASS_REQUIRED = PAIR_REQUIRED + ("other_id", "x2", "y2")
TRANSITION_REQUIRED = ("match_id", "seq", "possession", "team_id", "t_mono", "primary", "zone",
                       "has_steal", "entry", "is_shot", "valid_for_transition", "mp")


@dataclass(frozen=True)
class MetricSpec:
    """Агрегат наблюдений игрока: agg = "quantile" (с q) или "mean"; min_obs_half — минимум
    наблюдений в каждой половине для split-half (и в полной выборке для μ_r)."""
    name: str
    agg: str = "quantile"
    q: float = 0.5
    min_obs_half: int = 5

    def __post_init__(self):
        if self.agg not in ("quantile", "mean"):
            raise ValueError(f"MetricSpec.agg: ожидалось quantile|mean, получено {self.agg!r}")
        if not 0.0 <= self.q <= 1.0:
            raise ValueError(f"MetricSpec.q вне [0, 1]: {self.q}")


def _require(df: pd.DataFrame, cols, what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what}: нет обязательных колонок {missing}")


def _flag(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return df[col].fillna(False).to_numpy(dtype=bool)


OPTIONAL_COLS = ("clock_imputed", "t_behind_scale", "chain_gap_before", "skaters")


def _sorted(ev: pd.DataFrame, cols, drop_duel: bool = False) -> pd.DataFrame:
    """Копия только нужных колонок (обязательные + имеющиеся необязательные) в порядке (матч, seq):
    подмножество колонок держит пиковую память ниже полной копии таблицы узлов. drop_duel —
    оставить только узлы L0 (события duel узлами не являются)."""
    keep = list(dict.fromkeys([*cols, *(c for c in OPTIONAL_COLS if c in ev.columns)]))
    e = ev[keep]
    if drop_duel:
        e = e[e["primary"].ne("duel").to_numpy()]
    return e.sort_values(["match_id", "seq"], kind="mergesort").reset_index(drop=True)


# --- Ф1: пары соседних событий игрока ------------------------------------------------------

def candidate_pairs(ev: pd.DataFrame) -> pd.DataFrame:
    """Все пары соседних событий одного игрока внутри матча, периода и отрезка (без остановки).

    Возвращает кадр пар с Δt, смещением, скоростью и флагами для filter_speed_pairs; вход не
    меняется. interrupted — между событиями был узел с шайбой другого исполнителя или разрыв
    цепи L0 (chain_gap_before): путь не был ведением шайбы. duel чужого игрока цепь не рвёт."""
    _require(ev, PAIR_REQUIRED, "candidate_pairs")
    e = _sorted(ev, PAIR_REQUIRED)
    is_puck = e["primary"].ne("duel").to_numpy()
    puck_cum = pd.Series(is_puck.astype(np.int64)).groupby(e["match_id"].to_numpy()).cumsum().to_numpy()
    pos = pd.Series(np.arange(len(e), dtype=float))
    keys = [e["match_id"], e["period"], e["segment"], e["actor_id"]]
    prev = pos.groupby(keys, sort=False, dropna=True).shift(1).to_numpy()
    k = np.flatnonzero(~np.isnan(prev))
    j = prev[k].astype(np.int64)
    between = (puck_cum[k] - is_puck[k]) - puck_cum[j]
    cur, prv = e.iloc[k].reset_index(drop=True), e.iloc[j].reset_index(drop=True)
    imputed, behind = _flag(e, "clock_imputed"), _flag(e, "t_behind_scale")
    t, x, y = (e[c].to_numpy(dtype=float) for c in ("t_mono", "x", "y"))
    dt = t[k] - t[j]
    dist = np.hypot(x[k] - x[j], y[k] - y[j])
    speed = np.full(len(k), np.nan)
    np.divide(dist, dt, out=speed, where=dt > 0)
    # Колонки берутся срезом кадра (iloc), а не через to_numpy: строки остаются в компактном
    # arrow-представлении, без миллионов объектов Python.
    out = pd.DataFrame({
        "match_id": cur["match_id"], "period": cur["period"], "segment": cur["segment"],
        "player_id": cur["actor_id"], "team_id": cur["team_id"], "seq1": prv["seq"], "seq2": cur["seq"],
        "type1": prv["primary"], "type2": cur["primary"], "t1": t[j], "t2": t[k], "dt": dt,
        "dist": dist, "speed": speed,
        "interrupted": (between > 0) | _flag(e, "chain_gap_before")[k],
        "team_mismatch": prv["team_id"].ne(cur["team_id"]).to_numpy(dtype=bool),
        "time_untrusted": imputed[j] | imputed[k] | behind[j] | behind[k]})
    if "skaters" in e.columns:
        out["skaters"] = prv["skaters"]
    return out


def _sequential_filter(df: pd.DataFrame, reasons: list[tuple[str, np.ndarray]]) -> tuple[pd.DataFrame, dict]:
    """Последовательное отсеивание: каждая строка учитывается в первой сработавшей причине."""
    alive = np.ones(len(df), dtype=bool)
    counts = {"candidates": int(len(df))}
    for name, bad in reasons:
        counts[name] = int((alive & bad).sum())
        alive &= ~bad
    counts["kept"] = int(alive.sum())
    return df[alive].reset_index(drop=True), counts


def filter_speed_pairs(pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Фильтры Ф1 (типы, непрерывность владения, доверие ко времени, 0,5 < Δt ≤ 3 с, v < 13 м/с)."""
    dt = pairs["dt"].to_numpy(dtype=float)
    type_ok = pairs["type1"].isin(PAIR_PREV_TYPES).to_numpy() & pairs["type2"].isin(PAIR_NEXT_TYPES).to_numpy()
    dt_ok = (dt > PAIR_DT_MIN_S + DT_EPS_S) & (dt <= PAIR_DT_MAX_S + DT_EPS_S)
    speed = pairs["speed"].to_numpy(dtype=float)
    reasons = [
        ("event_type", ~type_ok),
        ("interrupted", pairs["interrupted"].to_numpy(dtype=bool) | pairs["team_mismatch"].to_numpy(dtype=bool)),
        ("time_untrusted", pairs["time_untrusted"].to_numpy(dtype=bool)),
        ("missing_xy", ~np.isfinite(pairs["dist"].to_numpy(dtype=float))),
        ("dt_out_of_range", ~dt_ok),
        ("speed_implausible", ~(speed < SPEED_MAX_MPS)),
    ]
    return _sequential_filter(pairs, reasons)


def _mates(skaters: pd.Series, players: pd.Series) -> list[list]:
    return [[q for q in (s if isinstance(s, (list, tuple, np.ndarray)) else []) if q != p]
            for s, p in zip(skaters, players)]


def to_obs(df: pd.DataFrame, value_col: str, player_col: str = "player_id") -> pd.DataFrame:
    """Кадр наблюдений для надёжности: match_id, player_id, value[, mates — партнёры на льду]."""
    out = pd.DataFrame({"match_id": df["match_id"].to_numpy(), "player_id": df[player_col].to_numpy(),
                        "value": df[value_col].to_numpy(dtype=float)})
    if "skaters" in df.columns:
        out["mates"] = _mates(df["skaters"], df[player_col])
    return out


# --- Ф2: скорость шайбы в передаче ----------------------------------------------------------

def pass_speed_obs(ev: pd.DataFrame, return_counts: bool = False):
    """Точный пас → следующий узел того же отрезка, если его исполнитель — получатель паса.

    Длина — от точки паса до (x2, y2) (точка приёма = положение получателя), без x2 — до
    положения следующего события получателя; время — до этого события. Атрибуция — пасующему."""
    _require(ev, PASS_REQUIRED, "pass_speed_obs")
    e = _sorted(ev, PASS_REQUIRED, drop_duel=True)
    grp = e.groupby(["match_id", "period", "segment"], sort=False)
    nxt = {c: grp[c].shift(-1) for c in ("actor_id", "team_id", "t_mono", "x", "y")}
    untrusted = _flag(e, "clock_imputed") | _flag(e, "t_behind_scale")
    nxt_untrusted = pd.Series(untrusted).groupby([e["match_id"], e["period"], e["segment"]]).shift(-1)
    nxt_gap = pd.Series(_flag(e, "chain_gap_before")).groupby([e["match_id"], e["period"], e["segment"]]).shift(-1)
    has_x2 = e["x2"].notna() & e["y2"].notna()
    tx = e["x2"].where(has_x2, nxt["x"]).to_numpy(dtype=float)
    ty = e["y2"].where(has_x2, nxt["y"]).to_numpy(dtype=float)
    dt = (nxt["t_mono"] - e["t_mono"]).to_numpy(dtype=float)
    dist = np.hypot(tx - e["x"].to_numpy(dtype=float), ty - e["y"].to_numpy(dtype=float))
    speed = np.full(len(e), np.nan)
    np.divide(dist, dt, out=speed, where=dt > 0)
    is_pass = e["primary"].eq("pass_complete").to_numpy() & e["other_id"].notna().to_numpy() \
        & e["other_id"].ne(e["actor_id"]).to_numpy()
    reasons = [
        ("receiver_not_next", ~(nxt["actor_id"].eq(e["other_id"]) & nxt["team_id"].eq(e["team_id"])).to_numpy()),
        ("time_untrusted", untrusted | nxt_untrusted.fillna(True).to_numpy(dtype=bool)
         | nxt_gap.fillna(True).to_numpy(dtype=bool)),
        ("missing_xy", ~np.isfinite(dist)),
        ("dt_out_of_range", ~((dt >= PASS_DT_MIN_S - DT_EPS_S) & (dt <= PASS_DT_MAX_S + DT_EPS_S))),
        ("speed_implausible", ~(speed < PASS_SPEED_MAX_MPS)),
    ]
    cand = e.assign(speed=speed, dist=dist, dt=dt)[is_pass]
    kept, counts = _sequential_filter(cand, [(n, m[is_pass]) for n, m in reasons])
    kept = kept.rename(columns={"actor_id": "player_id"})
    cols = ["match_id", "seq", "player_id", "team_id", "speed", "dist", "dt"] + (["skaters"] if "skaters" in kept else [])
    return (kept[cols], counts) if return_counts else kept[cols]


# --- Ф2: темп перехода ---------------------------------------------------------------------

def transition_table(nodes: pd.DataFrame) -> pd.DataFrame:
    """Владения 5 на 5, начатые возвратом шайбы (steal/intercept) в своей зоне.

    t_transition — от возврата до первого узла владения в чужой зоне / с меткой входа / броска
    (NaN, если не дошли); t_first_shot — до первого броска владения; t_observed — до последнего
    узла владения. censored: владение оборвано разрывом наблюдения (valid_for_transition = False
    у последнего узла) — исход после t_observed неизвестен, а не «перехода/броска не было»."""
    _require(nodes, TRANSITION_REQUIRED, "transition_table")
    n = _sorted(nodes, TRANSITION_REQUIRED, drop_duel=True)
    key = [n["match_id"], n["possession"]]
    t = n["t_mono"]
    reach = n["zone"].eq("off") | n["entry"].fillna(False).astype(bool) | n["is_shot"].fillna(False).astype(bool)
    t_reach = t.where(reach).groupby(key, sort=False).transform("min")
    t_shot = t.where(n["is_shot"].fillna(False).astype(bool)).groupby(key, sort=False).transform("min")
    censored = ~n["valid_for_transition"].fillna(False).astype(bool).groupby(key, sort=False).transform("last")
    first = ~n.duplicated(["match_id", "possession"])
    start_ok = (n["primary"].eq("steal") | n["has_steal"].fillna(False).astype(bool)) & n["zone"].eq("def") \
        & n["mp"].eq(TRANSITION_STRENGTH)
    t_last = t.groupby(key, sort=False).transform("max")
    rows = first & start_ok
    out = pd.DataFrame({"match_id": n["match_id"], "seq": n["seq"], "possession": n["possession"],
                        "team_id": n["team_id"],
                        "t_start": t, "reached": t_reach.notna(), "t_transition": t_reach - t,
                        "t_first_shot": t_shot - t, "t_observed": t_last - t, "censored": censored})
    if "skaters" in n.columns:
        out["skaters"] = n["skaters"]
    return out[rows.to_numpy()].reset_index(drop=True)


def fast_attack_values(tr: pd.DataFrame, k_s: float) -> pd.Series:
    """1, если бросок не позже k с после возврата; 0, если его не было и владение наблюдалось до
    конца или дольше k; NaN — владение оборвано разрывом наблюдения раньше k без броска."""
    shot_in_k = tr["t_first_shot"].le(k_s + DT_EPS_S)
    known = shot_in_k | ~tr["censored"] | tr["t_observed"].ge(k_s - DT_EPS_S) | tr["t_first_shot"].notna()
    return shot_in_k.astype(float).where(known)


def expand_onice(poss: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Наблюдение владения → по одному на каждого полевого своей команды на льду (on-ice уровень)."""
    _require(poss, ("match_id", "skaters", value_col), "expand_onice")
    p = poss[poss[value_col].notna()]
    lists = [list(s) if isinstance(s, (list, tuple, np.ndarray)) else [] for s in p["skaters"]]
    lens = np.fromiter((len(s) for s in lists), dtype=np.int64, count=len(lists))
    players = [q for s in lists for q in s]
    mates = [[m for m in s if m != q] for s in lists for q in s]
    return pd.DataFrame({"match_id": np.repeat(p["match_id"].to_numpy(), lens), "player_id": players,
                         "value": np.repeat(p[value_col].to_numpy(dtype=float), lens), "mates": mates})


def onice_skaters(ev: pd.DataFrame, stints: pd.DataFrame) -> pd.Series:
    """Полевые команды исполнителя на льду в момент события (по составу stint L0)."""
    _require(ev, ("match_id", "stint", "team_id", "home_team_id"), "onice_skaters")
    _require(stints, ("match_id", "stint", "home_sk", "away_sk"), "onice_skaters/stints")
    st = stints.set_index(["match_id", "stint"])[["home_sk", "away_sk"]]
    j = st.reindex(pd.MultiIndex.from_arrays([ev["match_id"], ev["stint"]]))
    is_home = (ev["team_id"] == ev["home_team_id"]).to_numpy()
    picked = np.where(is_home, j["home_sk"].to_numpy(dtype=object), j["away_sk"].to_numpy(dtype=object))
    return pd.Series([list(s) if isinstance(s, (list, tuple, np.ndarray)) else [] for s in picked],
                     index=ev.index, dtype=object)


# --- надёжность: половины, групповые агрегаты, бутстрэп -------------------------------------

def spearman_brown(r):
    """Надёжность полной выборки по корреляции половин (общая реализация в validation)."""
    from .validation import spearman_brown as _sb
    return _sb(r)


def assign_halves(match_ids, seed: int = HALF_SEED) -> pd.Series:
    """Матч → половина 0/1: случайная перестановка отсортированных id с фиксированным seed,
    затем чередование (половины равны). Не зависит от порядка входа."""
    uniq = np.array(sorted(set(match_ids)), dtype=object)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    half = np.empty(len(uniq), dtype=np.int64)
    half[perm] = np.arange(len(uniq)) % 2
    return pd.Series(half, index=uniq, name="half")


@dataclass(frozen=True)
class HalfData:
    """Наблюдения, отсортированные по (игрок-половина, значение), и индексы для бутстрэпа."""
    values: np.ndarray
    ph: np.ndarray
    mcode: np.ndarray
    n_matches: int
    player_ids: np.ndarray
    player_role: np.ndarray
    pm_ph: np.ndarray
    pm_m: np.ndarray
    mate_ph: np.ndarray | None
    mate_player: np.ndarray | None
    mate_m: np.ndarray | None
    mate_cnt: np.ndarray | None

    @property
    def n_ph(self) -> int:
        return 2 * len(self.player_ids)


def _mate_table(o: pd.DataFrame, ph: np.ndarray, mcode: np.ndarray, players: pd.Index, n_m: int):
    """Совместные наблюдения (игрок-половина, партнёр, матч) → число наблюдений."""
    lens = np.fromiter((len(s) for s in o["mates"]), dtype=np.int64, count=len(o))
    flat = [q for s in o["mates"] for q in s]
    mate = players.get_indexer(pd.Index(flat, dtype=object)) if flat else np.zeros(0, dtype=np.int64)
    rph, rm = np.repeat(ph, lens), np.repeat(mcode, lens)
    ok = mate >= 0
    n_p = len(players)
    key = (rph[ok].astype(np.int64) * n_p + mate[ok]) * n_m + rm[ok]
    uk, cnt = np.unique(key, return_counts=True)
    return (uk // n_m) // n_p, (uk // n_m) % n_p, uk % n_m, cnt.astype(float)


def prepare_half_data(obs: pd.DataFrame, halves: pd.Series, roles: pd.Series) -> HalfData:
    """Наблюдения полевых игроков с конечным значением → HalfData."""
    _require(obs, ("match_id", "player_id", "value"), "prepare_half_data")
    role = obs["player_id"].map(roles)
    o = obs[role.isin(ROLES).to_numpy() & np.isfinite(obs["value"].to_numpy(dtype=float))]
    pcode, players = pd.factorize(o["player_id"])
    mcode, matches = pd.factorize(o["match_id"])
    half_m = halves.reindex(matches)
    if half_m.isna().any():
        raise ValueError("prepare_half_data: у части матчей нет половины")
    ph = pcode * 2 + half_m.to_numpy(dtype=np.int64)[mcode]
    vals = o["value"].to_numpy(dtype=float)
    order = np.lexsort((vals, ph))
    n_m = len(matches)
    pm = np.unique(ph.astype(np.int64) * n_m + mcode)
    mates = _mate_table(o, ph, mcode, pd.Index(players), n_m) if "mates" in o.columns else (None,) * 4
    return HalfData(values=vals[order], ph=ph[order], mcode=mcode[order], n_matches=n_m,
                    player_ids=np.asarray(players, dtype=object),
                    player_role=roles.reindex(players).to_numpy(dtype=object),
                    pm_ph=pm // n_m, pm_m=pm % n_m, mate_ph=mates[0], mate_player=mates[1],
                    mate_m=mates[2], mate_cnt=mates[3])


def _group_values(hd: HalfData, w_match: np.ndarray, spec: MetricSpec):
    """Агрегат по игроку-половине при весах матчей (кратности бутстрэпа): значение, n, M."""
    w = w_match[hd.mcode].astype(np.int64)
    n = np.bincount(hd.ph, weights=w, minlength=hd.n_ph)
    m = np.bincount(hd.pm_ph, weights=w_match[hd.pm_m], minlength=hd.n_ph)
    val = np.full(hd.n_ph, np.nan)
    if spec.agg == "mean":
        s = np.bincount(hd.ph, weights=w * hd.values, minlength=hd.n_ph)
        np.divide(s, n, out=val, where=n > 0)
    else:
        idx = np.repeat(np.arange(len(hd.values)), w)       # сортировка внутри группы сохраняется
        v, g = hd.values[idx], hd.ph[idx]
        cnt = np.bincount(g, minlength=hd.n_ph)
        has = cnt > 0
        start = np.cumsum(cnt) - cnt
        pos = spec.q * (cnt[has] - 1)
        lo, hi = np.floor(pos).astype(np.int64), np.ceil(pos).astype(np.int64)
        base = start[has]
        val[has] = v[base + lo] + (pos - lo) * (v[base + hi] - v[base + lo])
    val[n < spec.min_obs_half] = np.nan
    return val, n, m


def _line_values(hd: HalfData, w_match: np.ndarray, val: np.ndarray) -> np.ndarray:
    """Базовая строка «on-ice звено»: среднее показателя партнёров в той же половине, вес —
    число совместных наблюдений игрока с партнёром на льду."""
    if hd.mate_ph is None:
        return np.full(hd.n_ph, np.nan)
    w = hd.mate_cnt * w_match[hd.mate_m]
    xv = val[hd.mate_player * 2 + hd.mate_ph % 2]
    ok = np.isfinite(xv) & (w > 0)
    num = np.bincount(hd.mate_ph[ok], weights=w[ok] * xv[ok], minlength=hd.n_ph)
    den = np.bincount(hd.mate_ph[ok], weights=w[ok], minlength=hd.n_ph)
    out = np.full(hd.n_ph, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def _center_by_role(a: np.ndarray, b: np.ndarray, roles: np.ndarray, ok: np.ndarray):
    """Центрирование обеих половин по амплуа (внутриамплуа сравнение) для строк ok."""
    ca, cb, cr = [], [], []
    for r in ROLES:
        sel = ok & (roles == r)
        if sel.sum() == 0:
            continue
        ca.append(a[sel] - a[sel].mean())
        cb.append(b[sel] - b[sel].mean())
        cr.append(np.full(sel.sum(), r, dtype=object))
    if not ca:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=object)
    return np.concatenate(ca), np.concatenate(cb), np.concatenate(cr)


def _within_role_r(a: np.ndarray, b: np.ndarray, roles: np.ndarray, ok: np.ndarray) -> tuple[float, dict, int]:
    """Пирсон по центрированным внутри амплуа половинам: общий и по каждому амплуа."""
    ca, cb, cr = _center_by_role(a, b, roles, ok)
    pooled = _pearson(ca, cb) if len(ca) >= MIN_SPLIT_PLAYERS else float("nan")
    per_role = {}
    for r in ROLES:
        sel = cr == r
        per_role[r] = {"n_players": int(sel.sum()),
                       "r_half": _pearson(ca[sel], cb[sel]) if sel.sum() >= MIN_ROLE_PLAYERS else float("nan")}
        per_role[r]["split_half_sb"] = spearman_brown(per_role[r]["r_half"])
    return pooled, per_role, int(len(ca))


def _point(hd: HalfData, w_match: np.ndarray, spec: MetricSpec) -> dict:
    """r половин внутри амплуа, r базовой строки звена и разница на одних и тех же игроках."""
    val, _, _ = _group_values(hd, w_match, spec)
    xa, xb = val[0::2], val[1::2]
    ok = np.isfinite(xa) & np.isfinite(xb)
    r, per_role, n_pl = _within_role_r(xa, xb, hd.player_role, ok)
    line = _line_values(hd, w_match, val)
    la, lb = line[0::2], line[1::2]
    common = ok & np.isfinite(la) & np.isfinite(lb)
    r_own_c, _, n_c = _within_role_r(xa, xb, hd.player_role, common)
    r_l1, _, _ = _within_role_r(la, xb, hd.player_role, common)
    r_l2, _, _ = _within_role_r(lb, xa, hd.player_role, common)
    r_line = float(np.mean([r_l1, r_l2])) if np.isfinite(r_l1) and np.isfinite(r_l2) else float("nan")
    return {"r_half": r, "split_half_sb": spearman_brown(r), "n_players": n_pl, "per_role": per_role,
            "r_line": r_line, "r_own_common": r_own_c, "n_players_line": n_c,
            "delta_own_minus_line": r_own_c - r_line}


def _ci_shifted(x: list[float], point: float) -> tuple[list[float], float]:
    """Перцентильный интервал, сдвинутый на смещение бутстрэпа (медиана − точка).

    Повторы матчей уменьшают число различных матчей в выборке (~63%), поэтому бутстрэп
    систематически занижает надёжность половин; без сдвига интервал мог бы не содержать
    точечную оценку. Возвращает (интервал, смещение)."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0 or not np.isfinite(point):
        return [float("nan"), float("nan")], float("nan")
    bias = float(np.median(a)) - float(point)
    tail = (1.0 - CI_LEVEL) / 2.0 * 100.0
    return [float(np.percentile(a, tail)) - bias, float(np.percentile(a, 100.0 - tail)) - bias], bias


def bootstrap(hd: HalfData, spec: MetricSpec, n_boot: int, seed: int, point: dict) -> dict:
    """Парный бутстрэп по матчам: одна выборка матчей для игрока, его половин и базовой строки."""
    rng = np.random.default_rng(seed)
    keys = ("split_half_sb", "r_line", "delta_own_minus_line")
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, hd.n_matches, hd.n_matches), minlength=hd.n_matches).astype(float)
        p = _point(hd, w, spec)
        for k in keys:
            draws[k].append(p[k])
    out = {"n_boot": int(n_boot)}
    for k, name in zip(keys, ("boot_sb", "boot_r_line", "boot_delta_own_minus_line")):
        out[f"{name}_ci"], out[f"{name}_bias"] = _ci_shifted(draws[k], point[k])
    return out


def reliability(obs: pd.DataFrame, roles: pd.Series, spec: MetricSpec, seed: int = HALF_SEED,
                n_boot: int = 200, boot_seed: int | None = None) -> dict:
    """Split-half внутри амплуа (Спирмен–Браун) с бутстрэп-интервалом и строкой «звено».

    gate_passed: SB ≥ RELIABILITY_GATE по точечной оценке (как в спецификации); иначе показатель
    остаётся на уровне амплуа/звена. beats_line: нижняя граница интервала (r_own − r_line) > 0."""
    hd = prepare_half_data(obs, assign_halves(obs["match_id"].unique(), seed), roles)
    res = {"metric": spec.name, "agg": spec.agg, "q": spec.q, "min_obs_half": spec.min_obs_half,
           "half_seed": int(seed), "n_obs": int(len(hd.values)), "n_matches": int(hd.n_matches),
           "role_only_within_role_reliability": 0.0}
    res.update(_point(hd, np.ones(hd.n_matches), spec))
    res["gate"] = RELIABILITY_GATE
    res["gate_passed"] = bool(np.isfinite(res["split_half_sb"]) and res["split_half_sb"] >= RELIABILITY_GATE)
    if n_boot > 0:
        res.update(bootstrap(hd, spec, n_boot, seed + 1 if boot_seed is None else boot_seed, res))
        res["beats_line"] = bool(res["boot_delta_own_minus_line_ci"][0] > 0)
    return res


# --- стягивание к амплуа (модель L5) -------------------------------------------------------

def player_summary(obs: pd.DataFrame, spec: MetricSpec) -> pd.DataFrame:
    """Показатель игрока по всем наблюдениям: value, n_obs, n_matches."""
    _require(obs, ("match_id", "player_id", "value"), "player_summary")
    o = obs[np.isfinite(obs["value"].to_numpy(dtype=float))]
    g = o.groupby("player_id", sort=True)
    value = g["value"].mean() if spec.agg == "mean" else g["value"].quantile(spec.q)
    return pd.DataFrame({"value": value, "n_obs": g.size(), "n_matches": g["match_id"].nunique()}) \
        .reset_index()


def _noise_regression(d2: np.ndarray, x_n: np.ndarray, x_m: np.ndarray) -> tuple[float, float, bool]:
    """E[d²] = σ²·x_n + ω²·x_m без свободного члена, неотрицательно; при коллинеарности — только σ²."""
    X = np.column_stack([x_n, x_m])
    identified = bool(np.linalg.matrix_rank(X) == 2 and np.linalg.cond(X) < MAX_MOMENT_COND)
    if identified:
        coef, _ = nnls(X, d2)
        return float(coef[0]), float(coef[1]), True
    coef, _ = nnls(x_n[:, None], d2)
    return float(coef[0]), 0.0, False


def fit_shrinkage(obs: pd.DataFrame, roles: pd.Series, spec: MetricSpec, seed: int = HALF_SEED) -> dict:
    """μ_r, τ²_r (ковариация половин внутри амплуа, пол 0), σ², ω² по разностям половин."""
    hd = prepare_half_data(obs, assign_halves(obs["match_id"].unique(), seed), roles)
    val, n, m = _group_values(hd, np.ones(hd.n_matches), spec)
    xa, xb = val[0::2], val[1::2]
    ok = np.isfinite(xa) & np.isfinite(xb)
    if ok.sum() < MIN_SPLIT_PLAYERS:
        raise ValueError(f"fit_shrinkage[{spec.name}]: игроков с двумя половинами {int(ok.sum())} "
                         f"< {MIN_SPLIT_PLAYERS}")
    na, nb, ma, mb = n[0::2][ok], n[1::2][ok], m[0::2][ok], m[1::2][ok]
    sigma2, omega2, ident = _noise_regression((xa[ok] - xb[ok]) ** 2, 1 / na + 1 / nb, 1 / ma + 1 / mb)
    ca, cb, cr = _center_by_role(xa, xb, hd.player_role, ok)
    pooled = max(float(np.mean(ca * cb)), 0.0)
    tau2 = {r: (max(float(np.mean(ca[cr == r] * cb[cr == r])), 0.0)
                if (cr == r).sum() >= MIN_ROLE_PLAYERS else pooled) for r in ROLES}
    summ = player_summary(obs, spec)
    summ = summ[summ.n_obs >= spec.min_obs_half].assign(role=lambda d: d.player_id.map(roles))
    mu = {r: float(summ.loc[summ.role == r, "value"].mean()) for r in ROLES}
    return {"metric": spec.name, "mu": mu, "tau2": tau2, "tau2_pooled": pooled, "sigma2": sigma2,
            "omega2": omega2, "omega_identified": ident, "n_players_split": int(ok.sum())}


def apply_shrinkage(summary: pd.DataFrame, roles: pd.Series, params: dict) -> pd.DataFrame:
    """θ̂ = μ_r + w (x − μ_r), w = τ²_r / (τ²_r + σ²/n + ω²/M); метка HIGH/MEDIUM/LOW по w."""
    out = summary.assign(role=summary["player_id"].map(roles))
    mu = out["role"].map(params["mu"]).to_numpy(dtype=float)
    tau2 = out["role"].map(params["tau2"]).to_numpy(dtype=float)
    n = out["n_obs"].to_numpy(dtype=float)
    m = out["n_matches"].to_numpy(dtype=float)
    s2 = np.full(len(out), np.inf)
    has = (n > 0) & (m > 0)
    s2[has] = params["sigma2"] / n[has] + params["omega2"] / m[has]
    w = np.zeros(len(out))
    np.divide(tau2, tau2 + s2, out=w, where=np.isfinite(s2) & (tau2 + s2 > 0))
    x = out["value"].to_numpy(dtype=float)
    shrunk = np.where(np.isfinite(x), mu + w * (np.nan_to_num(x) - mu), mu)
    label = np.where(w >= WEIGHT_HIGH, "HIGH", np.where(w >= WEIGHT_MEDIUM, "MEDIUM", "LOW"))
    return out.assign(weight=w, shrunk=shrunk, label=label)


def speed_confound_check(pairs: pd.DataFrame, roles: pd.Series) -> dict:
    """Внутри амплуа: p80 игрока против пар на матч и доли коротких Δt (Пирсон и общий R²)."""
    _require(pairs, ("match_id", "player_id", "speed", "dt"), "speed_confound_check")
    d = pairs.assign(short=pairs["dt"].to_numpy(dtype=float) <= SHORT_DT_S + DT_EPS_S)
    g = d.groupby("player_id", sort=True)
    pl = pd.DataFrame({"p80": g["speed"].quantile(SPEED_QUANTILE), "n": g.size(),
                       "m": g["match_id"].nunique(), "short": g["short"].mean()})
    pl = pl[(pl.n >= CONFOUND_MIN_PAIRS) & pl.index.map(roles).isin(ROLES)]
    pl = pl.assign(density=pl.n / pl.m, role=pl.index.map(roles))
    cols = ["p80", "density", "short"]
    c = pl[cols] - pl.groupby("role")[cols].transform("mean")
    if len(c) < MIN_SPLIT_PLAYERS:
        return {"n_players": int(len(c)), "refuted": None}
    X = np.column_stack([c["density"].to_numpy(), c["short"].to_numpy()])
    yv = c["p80"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    r2 = float(1.0 - ((yv - X @ beta) ** 2).sum() / (yv ** 2).sum()) if (yv ** 2).sum() > 0 else float("nan")
    return {"n_players": int(len(c)), "r_density": _pearson(yv, X[:, 0]), "r_short_dt": _pearson(yv, X[:, 1]),
            "r2_within_role": r2, "r2_max": CONFOUND_R2_MAX,
            "refuted": bool(np.isfinite(r2) and r2 >= CONFOUND_R2_MAX)}
