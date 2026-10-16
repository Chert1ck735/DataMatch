"""Карточка для клуба: понятная шкала и текст по шаблону, без LLM (METHODOLOGY 11.7).

Литература (docs/REFERENCES.md, «Шкала и подача результата»): числа и диапазоны понятнее слов; интервал подписывается
явно; метка надёжности — словами и отдельно от уровня игрока; WAR — шкала, понятная клубам. Текст собирается
детерминированно из чисел карточки — воспроизводим и не может «приукрасить».
"""
from __future__ import annotations

import math

SEASON_GAMES = 82
NAME_Z = 0.5                              # как у названий ролей (METHODOLOGY 11.2)
ROLE_TITLE = {"C": "Центральный нападающий", "W": "Крайний нападающий", "D": "Защитник"}
RELIABILITY_WORDS = {"HIGH": "надёжная", "MEDIUM": "предварительная", "LOW": "мало данных"}
RELIABILITY_PHRASE = {"HIGH": "оценка надёжная", "MEDIUM": "оценка предварительная", "LOW": "данных мало"}
VERDICT_ABOVE, VERDICT_BELOW = "выше среднего амплуа", "ниже среднего амплуа"
VERDICT_UNCONFIRMED = "отличие от среднего амплуа не подтверждено"
PCT_SHOWN_MIN, PCT_SHOWN_MAX = 1, 99      # в тексте перцентиль не пишется как 0% или 100%
# признак профиля → (сильная сторона при z ≥ 0,5, слабая при z ≤ −0,5); у потерь знак: + = меньше потерь
PROFILE_WORDS = {"shot": ("бросок", "мало ценности бросков"), "creation": ("созидание", "мало созидания"),
                 "recovery": ("отбор", "мало отборов"), "loss": ("надёжность с шайбой", "много потерь"),
                 "faceoff": ("вбрасывание", "слабые вбрасывания")}


def percentile(rating: float, tau: float) -> float:
    """Место относительно распределения истинных уровней амплуа N(0, τ²): 100·Φ(rating/τ)."""
    return 50.0 * (1.0 + math.erf(rating / (tau * math.sqrt(2.0))))


def verdict(lo95: float, hi95: float) -> str:
    """По 95%-интервалу против нуля (среднее амплуа оценено; его неопределённость в интервал не входит)."""
    if lo95 > 0:
        return VERDICT_ABOVE
    if hi95 < 0:
        return VERDICT_BELOW
    return VERDICT_UNCONFIRMED


def _low_minutes(flag: str) -> bool:
    return bool(flag) and "мало минут" in flag


def reliability_words(label: str, flag: str = "") -> str:
    words = RELIABILITY_WORDS[label]
    return f"{words} (меньше 60 минут на льду)" if _low_minutes(flag) else words


def reliability_phrase(label: str, flag: str = "") -> str:
    """Надёжность для текста карточки: «оценка надёжная / предварительная» или «данных мало»."""
    phrase = RELIABILITY_PHRASE[label]
    return f"{phrase} (меньше 60 минут на льду)" if _low_minutes(flag) else phrase


def show_profile(label: str) -> bool:
    """Сильные и слабые стороны называются только при достаточном объёме данных (не LOW)."""
    return label != "LOW"


def strengths(z: dict) -> tuple[list[str], list[str]]:
    """Сильные (z ≥ 0,5) и слабые (z ≤ −0,5) стороны профиля, по убыванию |z|."""
    items = sorted(((k, v) for k, v in z.items() if v is not None and not math.isnan(v)), key=lambda kv: -abs(kv[1]))
    return ([PROFILE_WORDS[k][0] for k, v in items if v >= NAME_Z],
            [PROFILE_WORDS[k][1] for k, v in items if v <= -NAME_Z])


def war_per_season(war_per60: float, toi_min_per_game: float, games: int = SEASON_GAMES) -> float:
    """Победы за сезон при типичном времени амплуа: WAR/60 × TOI/матч × 82 / 60."""
    return war_per60 * toi_min_per_game * games / 60.0


def _num(x: float, digits: int = 1) -> str:
    s = f"{x:.{digits}f}".replace(".", ",")
    return s.replace("-", "−")


def _pct(x: float) -> str:
    return f"{min(max(math.floor(x + 0.5), PCT_SHOWN_MIN), PCT_SHOWN_MAX)}%"   # арифметическое округление


def player_text(role: str, pct: float, pct_lo: float, pct_hi: float, verdict: str, label: str, flag: str,
                strengths: list[str], weaknesses: list[str]) -> str:
    """Текст по шаблону. Без подтверждённого отличия от среднего — диапазон места, а не точка.

    WAR за сезон в текст не входит: это условный пересчёт (G_win по смеси лиг train, уровень замены, типичное
    время амплуа), он остаётся числовыми колонками `club_ratings.csv` (METHODOLOGY 11.7).
    """
    if verdict == VERDICT_UNCONFIRMED:
        place = f"место среди игроков амплуа — от {_pct(pct_lo)} до {_pct(pct_hi)}"
    else:
        place = f"лучше {_pct(pct)} игроков амплуа (диапазон {_pct(pct_lo)[:-1]}–{_pct(pct_hi)})"
    parts = [f"{ROLE_TITLE[role]}: {place}, {verdict}; {reliability_phrase(label, flag)}."]
    if not show_profile(label):
        parts.append("Сильные и слабые стороны по такому объёму данных не называем.")
        return " ".join(parts)
    if strengths:
        parts.append(f"Сильные стороны: {', '.join(strengths)}.")
    if weaknesses:
        parts.append(f"Слабые стороны: {', '.join(weaknesses)}.")
    if not strengths and not weaknesses:
        parts.append("Выраженных сильных и слабых сторон профиль не показывает.")
    return " ".join(parts)
