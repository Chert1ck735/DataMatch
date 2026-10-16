"""Карточка игрока: итоговый отчёт card_report.md и сводная таблица «Авангарда» (METHODOLOGY 11).

Запуск из Solve после всех блоков карточки (только чтение outputs/card):

    PYTHONPATH=src .venv/bin/python scripts/card_report.py

Числа берутся из JSON-отчётов блоков без пересчёта. Выход: outputs/card/card_report.md, avangard_summary.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from datamatch import card_report as cr

CARD = Path(__file__).resolve().parents[1] / "outputs" / "card"


def _j(name: str) -> dict:
    return json.loads((CARD / name).read_text())


def _ci(v) -> str:
    return f"[{v[0]:.3f}; {v[1]:.3f}]"


def sections() -> list[str]:
    war, roles, sit = _j("war_report.json"), _j("roles_report.json"), _j("situational_report.json")
    pen, phys = _j("penalties_report.json"), _j("phys_card_report.json")
    g = war["g_win"]
    lines = ["# Карточка игрока: составляющие вне рейтинга", "",
             "Методы, пороги и критерии записаны до расчёта (METHODOLOGY 11, поправки датированы). Всё, что "
             "выбирается, — только по матчам разработки train; test только применяется; итог O1 и "
             "`test_ratings.csv` не меняются; каждое поле — отдельная колонка, в Overall не складывается.", "",
             "## 1. WAR", "",
             f"- G_win = **{g['g_win']:.2f}** гола на победу, 95% CI {_ci(g['ci95_team_bootstrap'])} (бутстрэп по командам); "
             f"{g['n_matches']} матчей разработки; сверка счёта {war['scores']['known_scores_match']}/"
             f"{war['scores']['known_scores_checked']}.",
             "- WAR = xGAR / G_win; интервал — Монте-Карло (θ̂, R, G_win). Порядок игроков = порядок O1 при равном TOI.",
             "- Ограничения: R — по всему train (итоговая подгонка L5); θ̂, R и G_win в Монте-Карло независимы.", "",
             "## 2. Роли и похожие игроки", ""]
    for r, v in roles["roles"].items():
        lines.append(f"- {r}: k = {v['k']} (BIC, общая ковариация), ARI половин {v['ari_split_half']:.2f} "
                     f"(n = {v['n_ari_players']}) → роли {'приняты' if v['roles_accepted'] else '**не приняты** (порог 0,3)'}.")
    lines += ["- Не принятые роли показываются как «тяготеет к» с пометкой «роль неустойчива»; у каждого игрока — "
              "описание собственного профиля. Похожие — расстояние в отбеленном подпространстве.", "",
              "## 3. Ситуационные рейтинги", ""]
    for s, v in sit["situations"].items():
        sh = v["split_half_dev"]
        lines.append(f"- {s}: надёжность половин (Спирмен–Браун) {sh['r_sb']:.2f} (n = {sh['n_players']}); "
                     f"метки test {v['test_labels']}.")
    c = sit["consistency_vs_o1"]
    lines += [f"- Σ ситуаций (вес TOI) против O1: корреляция {c['corr']:.2f}, MAE {c['mae']:.3f}. "
              "Интервал — апостериорный, без бутстрэпа моделей ценности.", "", "## 4. Удаления", "",
              f"- Двойного счёта нет: численность меняется в следующем отрезке ({pen['manpower_timing_dev']['later_segment']} "
              f"из {sum(pen['manpower_timing_dev'].values())}).",
              f"- ν = **{pen['nu']['nu_xg']:.3f}** xG на нарушение, CI {_ci(pen['nu']['nu_xg_ci95'])}."]
    for k, a in pen["acceptance"].items():
        o = a["observed"]
        lines.append(f"- {k}: r половин {o['r']:.3f} {_ci([o['lo'], o['hi']])}, модель {a['model_implied_half_r']:.3f} → "
                     f"стягивание {'принято' if a['accepted'] else '**не принято** (только частоты)'}.")
    lines += ["", "## 5. Физический профиль Ф1–Ф2", ""]
    for d in phys["decision_dev"]:
        lines.append(f"- {d['metric']}: надёжность половин на разработке {d['split_half_sb_dev']:.2f} "
                     f"(порог 0,3) → {d['level']}.")
    p = phys["speed_p80_pooled_dev_test_descriptive"]
    lines += [f"- Скорость с шайбой на разработке + test (описательно): {p['split_half_sb']:.2f}.",
              "- Ф3 (модель запаса сил) не делалась: нужны данные по сменам (длина, отдых, накопленный TOI); "
              "оценка — 1–2 дня работы, план развития.", ""]
    fo = _j("faceoffs_report.json")
    sh = fo["split_half_dev"]
    lines += ["## 6. Вбрасывания (Bradley–Terry)", "",
              f"- τ² = {fo['tau2']} (кросс-валидация по матчам разработки); контекст: своя зона {fo['beta_dev']['def']:+.3f}, "
              f"чужая {fo['beta_dev']['off']:+.3f}, площадка {fo['beta_dev']['home']:+.3f} (логиты).",
              f"- Надёжность половин {sh['r_sb']:.2f} (n = {sh['n_players']}, порог 0,3) → "
              + ("умение с поправкой на соперника показывается." if sh["passed"]
                 else "**не прошла**: показываются только сырая доля побед и число вбрасываний."), ""]
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=CARD)
    args = ap.parse_args(argv)
    read = {n: pd.read_csv(CARD / f"{n}.csv") for n in ("war", "roles", "situational", "penalties", "phys")}
    table = cr.avangard_table(read["war"], read["roles"], read["situational"], read["penalties"], read["phys"])
    table.to_csv(args.out / "avangard_summary.csv", index=False)
    club = pd.read_csv(CARD / "club_ratings.csv")
    texts = club[club.is_avangard.astype(bool)].sort_values("rating", ascending=False)
    md = sections() + ["## Игроки «Авангарда» простыми словами (шаблон без LLM, METHODOLOGY 11.7)", "",
                       *[f"- **{r.player_id}.** {r.text}" for r in texts.itertuples()], "",
                       f"## Сводная таблица «Авангарда» ({len(table)} игроков)", "",
                       cr.markdown_table(table.round(3)), ""]
    (args.out / "card_report.md").write_text("\n".join(md))
    print(f"card_report.md: {len(md)} строк; «Авангард»: {len(table)} игроков")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
