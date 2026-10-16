"""Карта оценки ситуации с шайбой по клеткам площадки (марковская V из L2) — визуал для презентации.

Запуск из Solve после L2 (кэш только читается):

    PYTHONPATH=src .venv/bin/python scripts/value_map.py [--kind raw|calibrated|delta] [--pass X1 Y1 X2 Y2]

V задана на состояниях «клетка × интервал времени до конца периода × численный состав». Для карты значение клетки —
среднее по интервалам времени с весом числа событий в состоянии (n_state модели); калибровка нелинейна и
применяется к каждому состоянию до усреднения. Координаты — в системе команды с шайбой: атака всегда в сторону +x.
Клетки с числом событий меньше силы сглаживания модели (κ) помечаются: их значение в основном взято у более грубой
клетки. Выход (только агрегаты по клеткам): outputs/value_map/value_map.csv и value_map.png.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from datamatch.config import load_config
from datamatch.l0_events import MP_CLASSES

ROOT = Path(__file__).resolve().parents[1]
PANELS = (("EV5", "5 на 5"), ("PP", "Большинство"))
KIND_TITLE = {"raw": "исходная марковская V", "calibrated": "калиброванная V"}


def cell_table(model, mp_index: int, calibration=None) -> pd.DataFrame:
    """Значение каждой клетки для состава `mp_index`: среднее V по интервалам времени с весом n_state."""
    g, nt, nm = model.g, model.nt, model.nm
    cells = np.arange(g.n)
    states = (cells[:, None] * nt + np.arange(nt)[None, :]) * nm + mp_index      # клетка × интервал времени
    n = np.asarray(model.n_state, float)[states]
    v = np.asarray(model.V, float)[states]
    v_cal = calibration(v.ravel(), np.zeros(v.size, bool)).reshape(v.shape) if calibration is not None else v
    total = n.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        v_raw = np.where(total > 0, (n * v).sum(axis=1) / total, np.nan)
        v_c = np.where(total > 0, (n * v_cal).sum(axis=1) / total, np.nan)
    ix, iy = cells // g.ny, cells % g.ny
    return pd.DataFrame({"mp": MP_CLASSES[mp_index] if mp_index < len(MP_CLASSES) else str(mp_index),
                         "cell": cells, "ix": ix, "iy": iy, "x0": g.xe[ix], "x1": g.xe[ix + 1],
                         "y0": g.ye[iy], "y1": g.ye[iy + 1], "n_events": total, "v_raw": v_raw,
                         "v_calibrated": v_c, "low_data": total < float(model.kappa)})


def _calibration(proc):
    cal = proc.calibration
    if isinstance(cal, tuple):
        slope, offset = cal
        return lambda raw, isfo: slope * np.asarray(raw) + np.where(isfo, 0.0, offset)
    return cal


def load_procedure(cfg):
    sel = json.loads((cfg.paths.artifacts / "value_selected.json").read_text())
    if sel.get("selected") != "markov":
        raise SystemExit(f"карта строится только для марковской V, выбрано: {sel.get('selected')}")
    path = Path(cfg.paths.cache) / sel["cache_key"] / "procedure_full.pkl"
    with open(path, "rb") as fh:
        return pickle.load(fh), sel["cache_key"]


def _fmt(v: float) -> str:
    return "" if not np.isfinite(v) else f"{v:+.2f}".replace(".", ",").replace("-", "−")


def plot(tables: dict, column: str, title: str, out: Path, pass_xy=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import FancyArrowPatch, Rectangle

    vals = np.concatenate([t[column].to_numpy() for t in tables.values()])
    lim = float(np.nanmax(np.abs(vals))) or 1.0
    norm, cmap = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), plt.get_cmap("RdBu_r")
    fig, axes = plt.subplots(1, len(tables), figsize=(7.2 * len(tables), 4.6), constrained_layout=True)
    for ax, (label, t) in zip(np.atleast_1d(axes), tables.items()):
        for r in t.itertuples(index=False):
            v = getattr(r, column)
            face = cmap(norm(v)) if np.isfinite(v) else (0.85, 0.85, 0.85, 1)
            ax.add_patch(Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0, facecolor=face, edgecolor="white",
                                   linewidth=0.8, hatch="///" if r.low_data else None))
            ax.text((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, _fmt(v), ha="center", va="center", fontsize=7)
        ax.axvline(0, color="0.3", linestyle="--", linewidth=0.8)
        if pass_xy is not None:
            x1, y1, x2, y2 = pass_xy
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, color="black",
                                         linewidth=2))
        ax.set_xlim(t.x0.min(), t.x1.max())
        ax.set_ylim(t.y0.min(), t.y1.max())
        ax.set_aspect("equal")
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("направление атаки →")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes, shrink=0.8, label="ожидаемая разница голов\nдо остановки игры")
    fig.suptitle(title, fontsize=10)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", choices=["raw", "calibrated", "delta"], default="delta",
                    help="delta — та V, по которой считаются вклады (value.delta_from_raw)")
    ap.add_argument("--pass", dest="pass_xy", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
                    help="учебный пас: показать стрелку и значения клеток начала и конца (5 на 5)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "value_map")
    args = ap.parse_args(argv)
    cfg = load_config()
    proc, key = load_procedure(cfg)
    kind = args.kind
    if kind == "delta":
        kind = "raw" if bool(cfg["value"].get("delta_from_raw", False)) else "calibrated"
    column = "v_raw" if kind == "raw" else "v_calibrated"
    cal = _calibration(proc)
    tables = {label: cell_table(proc.model, MP_CLASSES.index(mp), cal) for mp, label in PANELS}
    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat(tables.values(), ignore_index=True).to_csv(args.out / "value_map.csv", index=False)
    plot(tables, column, f"Оценка ситуации с шайбой при заданных условиях — {KIND_TITLE[kind]}\n"
                         "среднее по времени до конца периода с весом числа событий; штриховка — мало событий",
         args.out / "value_map.png", args.pass_xy)
    print(f"L2: {key}; V: {KIND_TITLE[kind]}; клеток с малым числом событий: "
          + ", ".join(f"{k} {int(t.low_data.sum())}/{len(t)}" for k, t in tables.items()))
    if args.pass_xy is not None:
        x1, y1, x2, y2 = args.pass_xy
        g, t = proc.model.g, tables["5 на 5"].set_index("cell")
        a, b = int(g.cell(np.array([x1]), np.array([y1]))[0]), int(g.cell(np.array([x2]), np.array([y2]))[0])
        print(f"пас 5 на 5: клетка {a} V = {_fmt(t.loc[a, column])} (событий {int(t.loc[a, 'n_events'])}) → "
              f"клетка {b} V = {_fmt(t.loc[b, column])} (событий {int(t.loc[b, 'n_events'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
