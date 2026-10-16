# Контракт уровня A: L3 → L5 → L6 → test_ratings.csv

Общий интерфейс таблиц между слоями. Архитектура — `docs/ARCHITECTURE.md`, методика — `docs/METHODOLOGY.md`.

## Владение файлами

Каждый агент правит только свои файлы. Остальные читает, но не меняет.

| Агент | Свои файлы |
|---|---|
| L3 | `src/datamatch/stage_l3.py` (новый), `src/datamatch/l3_contrib.py`, `tests/test_l3_*.py` |
| L5 | `src/datamatch/stage_l5.py` (новый), `src/datamatch/l5_shrink.py`, `src/datamatch/l5_o0.py` (новый), `tests/test_l5_*.py` |
| L6 | `src/datamatch/stage_l6.py` (новый), `src/datamatch/l6_uncertainty.py` (новый), `tests/test_l6_*.py` |
| Интегратор | `src/datamatch/cli.py`, `configs/default.yaml` (новые разделы), `tests/test_tier_a_e2e.py`, README |

Запрещено всем (ключи кэша L0–L2): `io`, `l0_*`, `stage_l0`, `gates`, `l1_*`, `stage_l1`, `splits`,
`config.py`, `cache.py`, `l2_*`, `stage_l2`, `node_graph`; разделы конфига `l0, xg, value, data, eval, seed`.
Новые параметры слоя — константы модуля с комментарием `# CONFIG:` или чтение из уже существующих
разделов `shrink`, `bootstrap`; интегратор переносит их в конфиг.

## Входы от L0–L2 (уже есть)

- L0 кэш: `train_nodes`, `test_nodes`, `train_stints`, `test_stints`, `train_player_toi`,
  `test_player_toi`, `train_events`, `test_events`, `players`, `quality` (см. `stage_l0.run_l0`).
- L2 кэш: `dev_values` (OOF по train: `node`, `match_id`, `r`, `value`, `delta`, …),
  `test_values` (`node`, `match_id`, `r`, `value`, `delta`), объекты `procedure`, `xg`.
- Загрузка через существующие `run_l0/run_l1/run_l2(cfg, cache, force=False)` — при готовом кэше они
  не пересчитывают.

## L3 → L5: `stage_l3.run_l3(cfg, cache, force=False) -> dict`

```
{"train": {"journal": DataFrame, "player_match": DataFrame},
 "test":  {"journal": DataFrame, "player_match": DataFrame},
 "report": dict}           # телескопирование, доли valid_for_credit, число игроков/матчей
```

`journal` — выход `l3_contrib.journal` (+ колонка `situation`).
`player_match` — одна строка на (match_id, player_id) полевого игрока; колонки:

| Колонка | Смысл |
|---|---|
| `match_id`, `player_id`, `team_id`, `split` (`train`/`test`) | ключи |
| `position` (C/LW/RW/D), `role` (`C`, `W`, `D`) | амплуа из заявки; `role`: LW/RW → W |
| `toi_s`, `toi_5v5_s`, `toi_pp_s`, `toi_sh_s`, `toi_rate_s`, `toi_rate_5v5_s`, `toi_rate_pp_s`, `toi_rate_sh_s` | из `static_player_match` |
| `goal`, `ast`, `points`, `pm` | очки и +/− |
| `c_shot`, `c_loss`, `c_recovery`, `c_faceoff`, `c_creation`, `iav_sum` | Σ credit по категориям (не на 60) |
| `c_<cat>_<sit>` для `sit` ∈ `5v5`, `pp`, `pk` | то же по ситуациям (для ситуационных рейтингов) |
| `ixg`, `igoals`, `ishots`, `xgf_5v5`, `xga_5v5` | из `dynamic_player_match` |
| `component` | id компоненты связности команды (train); `is_khl_component` — bool |

Ставки на 60 считает L5: `x = Σ / toi_rate_s × 3600`. Знаменатель — `toi_rate_*` (та же выборка,
что числитель `valid_for_rate_metrics`).

## L5 → L6: `stage_l5.run_l5(cfg, cache, l3, force=False) -> dict`

```
{"O1": {"train": DataFrame, "test": DataFrame},
 "O0": {"train": DataFrame, "test": DataFrame},
 "prior": {role: {"O1": {"mean": float, "var": float}, "O0": {...}}},
 "report": dict}           # τ², σ², ω², T_r, запасные ветки, eps-логи, гейты L5
```

Латентная таблица (одна строка на игрока выборки):

| Колонка | Смысл |
|---|---|
| `player_id`, `role`, `games` (M_i), `toi_min` (T_i) | |
| `rating` | Overall, центрированный на описательный центр m_r (xG/60) |
| `var_post` | апостериорная дисперсия 1ᵀV_i1 (или v_i в одномерной ветке) |
| `w` | доля собственного сигнала, обрезанная в [0, 1] |
| `comp_<cat>` | центрированные стянутые компоненты θ̂_ic − m_rc (только O1) |
| `definition` | `O1` или `O0`; `fallback` — bool (одномерная запасная ветка) |

O0 = on-ice (xgf_5v5 − xga_5v5)/toi_rate_5v5 × 3600, одномерное стягивание по `role`.
`prior[role]` — среднее и дисперсия для игроков без смен (rating = μ_r − m_r, var = 1ᵀT_r1). Если сигнал Overall не обнаружен (сырая 1ᵀT_r1 ≤ max(ε_abs, 10⁻³·tr T_r), решение 06.10.2026), w = 0, флаг «сигнал не обнаружен», а дисперсия — консервативная верхняя граница τ² по половинам (max(τ̂², 0) + 1,96·SE), чтобы интервал LOW-игрока не схлопывался в точку.
Всё обучаемое (μ_r, T_r, Σ, τ², σ², ω²) — только по train (`split == train`).

## L6 → файл: `stage_l6.run_l6(cfg, cache, l5, mode="fast", force=False) -> dict`

Пишет `outputs/test_ratings.csv`:

```
player_id,rating,lo95,hi95,reliability,toi_min,games,role
```

- строки — ВСЕ полевые игроки теста (из `test_player_toi`, заявок `lineup_*` тестовых матчей,
  `player_seasons_test.csv`, `players_new.csv`), без вратарей; игроки без смен — из `prior`,
  `toi_min = 0`, `games = 0`, `reliability = LOW`;
- `lo95/hi95 = rating ± 1,96·√v_tot`, `v_tot = var_post + Var_m` (бутстрэп моделей ценности);
- `reliability`: HIGH при w ≥ 0,7, MEDIUM при 0,4–0,7, LOW иначе (пороги из `shrink`);
- определение итога: O1, если он есть и прошёл гейты L5; иначе O0.

Пишет `outputs/test_ratings_meta.json`: `rating_definition`, `mode` (`fast`/`full`), `B`,
`model_bootstrap` (`done` / `not_run` с причиной), число строк по типам, хэши входов (провенанс).
Режим fast: B = `bootstrap.debug_B`; full: B = `bootstrap.final_B`.
