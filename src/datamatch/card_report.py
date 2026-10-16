"""Карточка игрока: сборка сводной таблицы «Авангарда» из готовых блоков (без пересчёта; METHODOLOGY 11)."""
from __future__ import annotations

import pandas as pd

KEY = "player_id"
COLUMNS = {
    "war": ["player_id", "role", "rating", "reliability", "toi_min", "war", "war_lo95", "war_hi95"],
    "roles": ["player_id", "role_name", "own_profile", "flag"],
    "situational": ["player_id", "rating_5v5", "reliability_5v5", "rating_pp", "reliability_pp",
                    "rating_pk", "reliability_pk"],
    "penalties": ["player_id", "drawn_per60", "taken_per60"],
    "phys": ["player_id", "speed_p80", "speed_p80_level"],
}


def _block(df: pd.DataFrame, name: str) -> pd.DataFrame:
    part = df[COLUMNS[name]]
    if part[KEY].duplicated().any():
        raise ValueError(f"avangard_table: в блоке {name} повторяются игроки")
    return part.rename(columns={"flag": "role_flag"}) if name == "roles" else part


def avangard_table(war: pd.DataFrame, roles: pd.DataFrame, situational: pd.DataFrame,
                   penalties: pd.DataFrame, phys: pd.DataFrame) -> pd.DataFrame:
    """Игроки «Авангарда» (по таблице WAR) со всеми блоками карточки, по убыванию рейтинга O1."""
    base = _block(war[war["is_avangard"].astype(bool)], "war")
    for name, df in (("roles", roles), ("situational", situational), ("penalties", penalties), ("phys", phys)):
        base = base.merge(_block(df, name), on=KEY, how="left", validate="one_to_one")
    return base.sort_values("rating", ascending=False, kind="stable").reset_index(drop=True)


def markdown_table(df: pd.DataFrame) -> str:
    """Таблица markdown без внешних зависимостей (tabulate не входит в зафиксированные зависимости)."""
    def cell(v) -> str:
        return "" if pd.isna(v) else str(v).replace("|", "/")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])
