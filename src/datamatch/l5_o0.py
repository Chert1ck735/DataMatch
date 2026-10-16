"""L5/O0 — страховочный кандидат итога: on-ice xGD/60 при 5 на 5 с одномерным стягиванием.

Определение (архитектура, таблица кандидатов итога):

    x_i = (Σ xgf_5v5 − Σ xga_5v5) / Σ toi_rate_5v5_s × 3600,

одномерное стягивание по амплуа (τ², σ², ω² — метод моментов по чётным/нечётным матчам
только на train), апостериорный интервал без бутстрэпа моделей ценности.

O0 нужен как страховка: он считается из готовых L0–L1 величин (замороженный xG-скорер и
on-ice экспозиция из L0) и не использует ни ценность состояния L2, ни компоненты вклада L3.
Поэтому модуль требует от таблицы «игрок × матч» только ключи, амплуа, on-ice xG и
5v5-экспозицию; наличие колонок `c_*` не проверяется и не используется.

Знаменатель — `toi_rate_5v5_s`: та же допущенная выборка матчей, что и у числителя
(матч, исключённый из rate-метрик, не даёт ни xGF/xGA, ни экспозиции). Строки с нулевой
5v5-экспозицией из оценки исключаются целиком (числитель и знаменатель одной выборкой),
а игрок без допущенной 5v5-экспозиции получает prior-ветку (θ̂ = μ_r, v = τ²) и явный признак
`no_rate_exposure` (нулевые toi_min/games в выходе — следствие контракта, а не наблюдение нуля).
Без межигрового сигнала (τ² ≤ ε) θ̂ = μ_r, w = 0, «сигнал не обнаружен», а дисперсия — верхняя
граница τ² по половинам (`UnivariateEB.prior_var`), а не ноль: нулевая ширина интервала у LOW-строки
утверждала бы точное знание.

O0 — не определение итога по умолчанию: он заменяет O1 только если O1 не прошёл гейты L5
(см. контракт L5 → L6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .l5_shrink import NO_SIGNAL_FLAG, ROLES, UnivariateEB, reliability

# Колонка показателя O0 внутри модуля (xG-разница на льду при 5 на 5, сумма за матч).
O0_COL = "xgd_5v5"
# Требуемая схема входа: только L0/L1-величины, без L2/L3.
O0_REQUIRED = ["match_id", "player_id", "xgf_5v5", "xga_5v5", "toi_rate_5v5_s"]


def o0_frame(pm: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Таблица «игрок × матч» для O0: xGD при 5 на 5 и 5v5-экспозиция как `toi_s`."""
    missing = [c for c in O0_REQUIRED if c not in pm.columns]
    if missing:
        raise KeyError(f"O0: во входной таблице нет колонок {missing}")
    toi = pm.toi_rate_5v5_s.to_numpy(dtype=float)
    xgd = pm.xgf_5v5.to_numpy(dtype=float) - pm.xga_5v5.to_numpy(dtype=float)
    keep = toi > 0
    f = pd.DataFrame({"player_id": pm.player_id.to_numpy()[keep],
                      "match_id": pm.match_id.to_numpy()[keep],
                      O0_COL: xgd[keep], "toi_s": toi[keep]})
    info = {"rows_total": int(len(pm)), "rows_kept": int(keep.sum()),
            "rows_dropped_zero_5v5_toi": int((~keep).sum()),
            "dropped_abs_xgd": float(np.abs(xgd[~keep]).sum())}
    return f, info


def player_sums(frame: pd.DataFrame) -> pd.DataFrame:
    """Суммы по игроку: числитель, экспозиция и число матчей — одна и та же выборка."""
    return frame.groupby("player_id").agg(**{O0_COL: (O0_COL, "sum")},
                                          toi_s=("toi_s", "sum"), games=("match_id", "nunique"))


def fit_o0(cfg, pm_train: pd.DataFrame, roles: pd.Series, mu_pm: pd.DataFrame | None = None):
    """Моменты O0 только на train. `mu_pm` — часть train для μ_r (для test: КХЛ-компонента)."""
    sh = cfg["shrink"]
    frame, info = o0_frame(pm_train)
    mu_frame = None if mu_pm is None else o0_frame(mu_pm)[0]
    eb = UnivariateEB(float(sh["eps_abs"]), noise_by_role=bool(sh.get("noise_by_role", False))).fit(
        frame, O0_COL, roles, min_toi_s=float(sh["min_toi_split_min"]) * 60.0, mu_pm=mu_frame)
    return eb, {"exposure": info, "n_split_players": eb.n}


def rate_o0(eb: UnivariateEB, pm_eval: pd.DataFrame, roles: pd.Series, ids, cfg,
            m_r: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """O0 для оцениваемой выборки: стянутая оценка минус описательный центр амплуа m_r."""
    sh = cfg["shrink"]
    ids = list(ids)
    frame, info = o0_frame(pm_eval)
    s = player_sums(frame).reindex(ids)
    # нет допущенной 5v5-экспозиции: строки нет в сумме; NaN -> 0 только для prior-ветки, признак явный
    no_rate = s.toi_s.isna().to_numpy(dtype=bool)
    toi = s.toi_s.fillna(0.0).to_numpy(dtype=float)
    games = s.games.fillna(0).to_numpy(dtype=float)
    vals = s[O0_COL].fillna(0.0).to_numpy(dtype=float)
    rr = np.asarray(roles.reindex(ids).to_numpy(), dtype=object)
    th, v, w = eb.posterior(rr, vals, toi, games)
    if m_r is None:
        m_r = {}
        for r in ROLES:
            m = rr == r
            ww = toi[m]
            if ww.sum() > 0:
                m_r[r] = float((th[m] * ww).sum() / ww.sum())
            else:
                m_r[r] = float(th[m].mean()) if m.any() else 0.0
    centre = np.array([m_r.get(r, 0.0) for r in rr], dtype=float)
    no_signal = np.array([eb.no_signal(r) for r in rr], dtype=bool)
    df = pd.DataFrame({
        "player_id": ids, "role": rr, "games": games.astype(int), "toi_min": toi / 60.0,
        "rating": th - centre, "var_post": v, "w": np.clip(w, 0.0, 1.0),
        "theta": th, "definition": "O0", "fallback": no_signal,
        "method": np.where(no_signal, "no_signal", np.where(toi > 0, "univariate", "prior")),
        "flag": np.where(no_signal, NO_SIGNAL_FLAG, ""),
        "no_rate_exposure": no_rate,
    })
    df["reliability"] = reliability(df.w.to_numpy(), float(sh["reliability_high"]),
                                    float(sh["reliability_medium"]))
    rep = {"exposure": info, "m_r": {r: float(m_r.get(r, 0.0)) for r in ROLES},
           "tau2": {r: float(eb.tau2.get(r, 0.0)) for r in ROLES},
           "tau2_upper": {r: float(eb.tau2_upper.get(r, 0.0)) for r in ROLES},
           "sigma2": float(eb.sigma2), "omega2": float(eb.omega2), "noise_refit": bool(eb.noise_refit),
           "noise_by_role": bool(eb.noise_by_role),
           "sigma2_r": {r: float(eb.sigma2_r[r]) for r in ROLES},
           "omega2_r": {r: float(eb.omega2_r[r]) for r in ROLES},
           "omega_identified": bool(eb.omega_identified), "design_cond": float(eb.design_cond),
           "pooled_roles": {r: bool(eb.pooled.get(r, False)) for r in ROLES},
           "mu": {r: float(eb.mu.get(r, 0.0)) for r in ROLES},
           "n_no_signal": int(no_signal.sum()), "n_prior_only": int((toi <= 0).sum()),
           "n_no_rate_exposure": int(no_rate.sum())}
    return df, rep
