"""L5: совместное стягивание компонентов (эмпирический Байес) и итог O1.

Модель: x_i = θ_i + e_i,  Cov(θ_i) = T_r,  Cov(e_i) = S_i = Σσ / T_i + Σω / M_i
(x — компоненты на 60 минут, T_i — TOI в минутах, M_i — число матчей).
Апостериорно: θ̂_i = μ_r + T_r (T_r + S_i)^{-1} (x_i − μ_r),  V_i = T_r − T_r (T_r + S_i)^{-1} T_r.
Overall_i = 1ᵀ(θ̂_i − m_r), Var = 1ᵀ V_i 1. Центрированные компоненты θ̂_ic − m_rc в сумме дают Overall.

Оценка моментов по половинам матчей игрока (чередование его матчей по порядку match_id):
  T_r = симметризованная ковариация половин по игрокам амплуа r;
  E[d dᵀ] = Σσ (1/T_a + 1/T_b) + Σω (1/M_a + 1/M_b), d = x_a − x_b (регрессия без свободного члена).
Численная защита (PDF): проекция T_r, Σσ, Σω на положительно определённые с полом
max(1e-6·tr/k, ε_abs); V_i без обращения T_r и S_i; система T_r + S_i — Холецкий, при неудаче
диагональ + ε·max(tr/k, ε_abs), ε от 1e-6 ×10 до 1e-2, ε пишется в лог для игрока.
tr(T_r) ≤ ε_abs -> межигрового сигнала нет, сразу запасной вариант. 1ᵀT_r1 ≤ ε -> Overall
на одномерный запасной вариант; если и там τ² ≈ 0, w = 0, LOW, «сигнал не обнаружен».
Решение «нет сигнала в сумме» принимается по СЫРОЙ 1ᵀT_r1 (до проекции, она равна одномерному τ²)
с относительным порогом max(ε_abs, REL·tr(T_r)): после проекции 1ᵀT_r1 не меньше пола проекции,
и решение по ней принималось бы внутри численного пола (отрицательная сырая сумма давала joint).
Строка «сигнал не обнаружен» получает θ̂ = μ_r и дисперсию prior одномерной модели — верхнюю
границу τ² по половинам (τ² ≥ 0 неизвестно, но ограничено сверху), а не max(τ̂², 0) = 0.
Игроки без игрового времени: апостериорное = prior (θ̂ = μ_r, V = T_r), без деления на TOI.
Центр Overall m_r считается по тому оценщику, которым строка посчитана: в одномерной ветке —
по одномерным θ̂ суммы, в совместной — по 1ᵀθ̂ совместной модели.

Идентифицируемость σ² и ω²: столбцы плана (1/T_н + 1/T_ч) и (1/M_н + 1/M_ч) коллинеарны,
когда TOI на матч одинаково у всех игроков (в частности, при одном матче на половину).
Тогда ω² из моментов не извлекается: оценивается только σ², ω² = 0, и это пишется в диагностику
(`omega_identified`), а не выдаётся за оценку шума на матч.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

ROLES = ["C", "W", "D"]

# CONFIG: минимум игроков с двумя непустыми половинами для совместной оценки моментов
MIN_SPLIT_PLAYERS = 30
# CONFIG: минимум игроков амплуа для отдельной T_r/τ² (иначе амплуа объединяются, с пометкой)
MIN_ROLE_PLAYERS = 20
# CONFIG: предел обусловленности плана моментов; выше — ω² считается неидентифицируемым
MAX_MOMENT_COND = 1.0e8
# CONFIG: относительный порог «в сумме компонентов нет межигрового сигнала»:
# 1ᵀT_r1 (сырая) ≤ max(ε_abs, REL·tr(T_r)). Обоснование: проекция поднимает собственные значения
# до 1e-6·tr/k, поэтому спроецированная 1ᵀT_r1 бывает порядка 1e-6·tr при нулевом сигнале суммы.
# REL = 1e-3 на три порядка выше пола (решение не попадает в численный шум проекции) и означает,
# что на сумму приходится < 0,1% межигровой дисперсии компонентов: w Overall как отношение
# двух почти нулевых величин тогда не определена.
OVERALL_SIGNAL_REL = 1.0e-3
# CONFIG: z верхней границы τ² для строк «сигнал не обнаружен»: max(τ̂², 0) + z·SE(τ̂²),
# односторонний уровень ≈ 97,5%. SE — разброс взвешенных произведений половин по игрокам.
TAU2_UPPER_Z = 1.96
NO_SIGNAL_FLAG = "сигнал не обнаружен"
# Методы строк Overall.
METHOD_JOINT = "joint"
METHOD_UNIVARIATE = "univariate_fallback"
METHOD_CHOLESKY_FAILED = "prior_cholesky_failed"


def overall_signal_threshold(T_raw: np.ndarray, eps_abs: float) -> float:
    """Порог для сырой 1ᵀT_r1: max(ε_abs, REL·tr(T_r)) (отрицательный след — только ε_abs)."""
    return max(float(eps_abs), OVERALL_SIGNAL_REL * max(float(np.trace(T_raw)), 0.0))


def overall_no_signal(T_raw: np.ndarray, eps_abs: float) -> bool:
    """В сумме компонентов нет межигрового сигнала: сырая 1ᵀT_r1 не выше относительного порога."""
    return bool(float(np.sum(T_raw)) <= overall_signal_threshold(T_raw, eps_abs))


def weighted_cross_cov(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """τ̂² = Σ w (a − ā)(b − b̄) по половинам и его верхняя граница max(τ̂², 0) + z·SE.

    w нормированы (Σ w = 1). SE = sqrt(Σ w² (p − τ̂²)²), p — произведения центрированных половин:
    дисперсия взвешенного среднего независимых по игрокам слагаемых (оценка центров не учитывается).
    """
    if not len(a):
        return 0.0, 0.0
    p = (a - (w * a).sum()) * (b - (w * b).sum())
    tau2 = float((w * p).sum())
    se = float(np.sqrt(((w * (p - tau2)) ** 2).sum()))
    return tau2, max(tau2, 0.0) + TAU2_UPPER_Z * se


def _wls(X: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Взвешенная регрессия без свободного члена; наименьшие квадраты вместо обращения XᵀWX."""
    sw = np.sqrt(np.maximum(w, 0.0))
    beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    out = np.zeros(2)
    out[: len(beta)] = beta
    return out


def psd_project(A: np.ndarray, eps_abs: float, floor_rel: float = 1e-6) -> tuple[np.ndarray, dict]:
    A = 0.5 * (A + A.T)
    k = A.shape[0]
    w, U = np.linalg.eigh(A)
    tr = float(np.trace(A))
    floor = max(floor_rel * max(tr, 0.0) / k, eps_abs)
    n_raised = int((w < floor).sum())
    w2 = np.maximum(w, floor)
    return (U * w2) @ U.T, {"eig_min_raw": float(w.min()), "n_eig_raised": n_raised, "floor": floor, "trace_raw": tr}


def offdiag_shrink(ca: np.ndarray, cb: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, dict]:
    """Кросс-ковариация половин C = Σ w_i (a_i b_iᵀ + b_i a_iᵀ)/2 со стягиванием недиагонали к нулю.

    Цель — диагональ C (Schäfer–Strimmer, вариант Ledoit–Wolf для признаков разного масштаба):
    C* = diag(C) + (1 − ρ)·offdiag(C), ρ = Σ_{j≠k} Var(C_jk) / Σ_{j≠k} C_jk², обрезка в [0, 1];
    Var(C_jk) ≈ Σ w_i² (z_ijk − C_jk)², z_ijk — вклад игрока. Минимальное собственное значение не
    уменьшается (выпуклая комбинация C и diag(C)). METHODOLOGY 5.2, поправка 08.10.
    """
    ca, cb, w = np.asarray(ca, float), np.asarray(cb, float), np.asarray(w, float)
    z = 0.5 * (ca[:, :, None] * cb[:, None, :] + cb[:, :, None] * ca[:, None, :])
    C = np.einsum("i,ijk->jk", w, z)
    var = np.einsum("i,ijk->jk", w ** 2, (z - C[None]) ** 2)
    off = ~np.eye(C.shape[0], dtype=bool)
    den = float((C[off] ** 2).sum())
    rho = float(np.clip(var[off].sum() / den, 0.0, 1.0)) if den > 0 else 0.0
    shrunk = np.where(off, (1.0 - rho) * C, C)
    return shrunk, {"rho": rho, "C_raw": C}


# CONFIG: предел итераций переоценки Σσ/Σω после проекции (блочный спуск, P055) и порог сходимости
# 5000: на синтетике с вырожденной Σω (k = 3) спуску требовалось до 3442 итераций (аудит 08.10.2026)
NOISE_REFIT_MAX_ITER = 5000
NOISE_REFIT_TOL = 1e-12


def _sym_from_upper(vals: np.ndarray, k: int) -> np.ndarray:
    M = np.zeros((k, k))
    M[np.triu_indices(k)] = vals
    return M + np.triu(M, 1).T


def _project_noise(A: np.ndarray, eps_abs: float, floor_rel: float, zero_floor: bool) -> tuple[np.ndarray, dict]:
    """Проекция матрицы шума на PSD; скаляр — обрезка снизу (σ² ≥ ε_abs, ω² ≥ 0)."""
    if A.shape == (1, 1):
        raw = float(A[0, 0])
        floor = 0.0 if zero_floor else eps_abs
        return np.array([[max(raw, floor)]]), {"eig_min_raw": raw, "n_eig_raised": int(raw < floor),
                                                "floor": floor, "trace_raw": raw}
    return psd_project(A, eps_abs, floor_rel)


def fit_noise_moments(d: np.ndarray, A1: np.ndarray, A2: np.ndarray, omega_identified: bool,
                      eps_abs: float, floor_rel: float, reweight: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    """Σσ, Σω из E[d dᵀ] = Σσ·A1 + Σω·A2 (A1 = 1/T_a + 1/T_b, A2 = 1/M_a + 1/M_b), d — (n, k).

    Шаг 1 — поэлементная WLS без ограничений (веса 1/A1², при `reweight` — две итерации с весами
    по предсказанному следу, как раньше в `JointEB.fit`). Если обе оценки уже PSD, результат прежний.
    Шаг 2 (P055) — иначе блочный спуск с ограничением PSD: проекция одной матрицы, переоценка другой
    при фиксированной первой, проекция, и так до сходимости. Целевая функция — Σ_n w_n‖d dᵀ − Σσ·A1 −
    Σω·A2‖²_F по полным матрицам (внедиагональный элемент учтён дважды, B6 аудита 08.10.2026), веса w —
    последней свободной подгонки. При общем весе строки шаг по одной матрице — поэлементный МНК,
    затем проекция Фробениуса (обрезка собственных значений), то есть точный условный минимум;
    блочный спуск выпуклой задачи сходится к минимуму с ограничением. `info["converged"]` — достигнут
    ли порог изменения до лимита итераций; `info["weights"]` — веса целевой функции.
    Раньше отрицательная ω² (или собственные значения Σω) обрезалась, а σ² из той же регрессии
    оставалась завышенной; при отрицательной σ² — наоборот.
    """
    n, k = d.shape
    iu = np.triu_indices(k)
    P = d[:, iu[0]] * d[:, iu[1]]
    X = np.column_stack([A1, A2]) if omega_identified else A1[:, None]
    wts = 1.0 / A1 ** 2
    Ss, So = np.zeros((k, k)), np.zeros((k, k))
    fit_w = wts
    for _ in range(2 if reweight else 1):
        fit_w = wts
        sw = np.sqrt(np.maximum(fit_w, 0.0))
        beta = np.linalg.lstsq(X * sw[:, None], P * sw[:, None], rcond=None)[0]
        Ss = _sym_from_upper(beta[0], k)
        So = _sym_from_upper(beta[1], k) if omega_identified else np.zeros((k, k))
        if reweight:
            pred_tr = A1 * max(float(np.trace(Ss)), 1e-12) + A2 * max(float(np.trace(So)), 0.0)
            wts = 1.0 / np.maximum(pred_tr, 1e-12) ** 2
    info = {"refit": False, "n_iter": 0, "Sigma_sigma_raw": Ss.copy(), "Sigma_omega_raw": So.copy()}
    Ss_p, i1 = _project_noise(Ss, eps_abs, floor_rel, zero_floor=False)
    So_p, i2 = _project_noise(So, eps_abs, floor_rel, zero_floor=True)
    if not omega_identified or (i1["n_eig_raised"] == 0 and i2["n_eig_raised"] == 0):
        return Ss_p, So_p, {**info, "converged": True, "weights": np.maximum(fit_w, 0.0),
                            "Sigma_sigma_proj": i1, "Sigma_omega_proj": i2}
    sw2 = np.maximum(fit_w, 0.0)   # веса последней подгонки свободной регрессии

    def given(other: np.ndarray, a_other: np.ndarray, a_self: np.ndarray) -> np.ndarray:
        target = P - np.outer(a_other, other[iu])
        return _sym_from_upper((sw2 * a_self) @ target / float((sw2 * a_self * a_self).sum()), k)

    s_first = i1["n_eig_raised"] > 0
    converged = False
    for it in range(1, NOISE_REFIT_MAX_ITER + 1):
        prev = (Ss_p.copy(), So_p.copy())
        if s_first:
            So_p, i2 = _project_noise(given(Ss_p, A1, A2), eps_abs, floor_rel, zero_floor=True)
            Ss_p, i1 = _project_noise(given(So_p, A2, A1), eps_abs, floor_rel, zero_floor=False)
        else:
            Ss_p, i1 = _project_noise(given(So_p, A2, A1), eps_abs, floor_rel, zero_floor=False)
            So_p, i2 = _project_noise(given(Ss_p, A1, A2), eps_abs, floor_rel, zero_floor=True)
        change = max(np.abs(Ss_p - prev[0]).max(), np.abs(So_p - prev[1]).max())
        if change <= NOISE_REFIT_TOL * max(1.0, np.abs(Ss_p).max(), np.abs(So_p).max()):
            converged = True
            break
    return Ss_p, So_p, {**info, "refit": True, "n_iter": it, "converged": converged, "weights": sw2,
                        "Sigma_sigma_proj": i1, "Sigma_omega_proj": i2}


def _design_identified(A1: np.ndarray, A2: np.ndarray) -> tuple[bool, float]:
    """ω² идентифицируем, если обусловленность плана (1/T, 1/M) не выше MAX_MOMENT_COND."""
    if len(A1) < 2:
        return False, float("inf")
    X = np.column_stack([A1, A2])
    cond = float(np.linalg.cond(X.T @ X))
    return bool(np.isfinite(cond) and cond <= MAX_MOMENT_COND), cond


def halves(pm: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Две половины матчей каждого игрока: чередование матчей игрока по порядку match_id."""
    p = pm.sort_values(["player_id", "match_id"]).copy()
    p["half"] = p.groupby("player_id").cumcount() % 2
    agg = {c: "sum" for c in value_cols}
    agg.update({"toi_s": "sum", "match_id": "nunique"})
    h = p.groupby(["player_id", "half"]).agg(agg).rename(columns={"match_id": "games"})
    return h


def halves_split(pm: pd.DataFrame, value_cols: list[str], min_toi_s: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Половины с общими игроками и достаточным TOI в каждой.

    Пустой результат — допустимое состояние данных (например, один матч на игрока: второй
    половины нет вовсе), поэтому отсутствие половины не ошибка индексации, а пустая выборка.
    """
    cols = list(value_cols) + ["toi_s", "games"]
    empty = pd.DataFrame(columns=cols, index=pd.Index([], name="player_id"), dtype=float)
    if not len(pm):
        return empty, empty
    h = halves(pm, value_cols)
    lv = h.index.get_level_values("half")
    a = h[lv == 0].droplevel("half")
    b = h[lv == 1].droplevel("half")
    if not len(a) or not len(b):
        return empty, empty
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    ok = (a.toi_s >= min_toi_s) & (b.toi_s >= min_toi_s)
    return a[ok], b[ok]


def per60(sum_vals: np.ndarray, toi_s: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(toi_s[:, None] > 0, sum_vals / toi_s[:, None] * 3600.0, 0.0)


class JointEB:
    def __init__(self, comp_cols: list[str], cfg_shrink: dict, logger=None):
        self.cols = comp_cols
        self.k = len(comp_cols)
        self.c = cfg_shrink
        self.eps_abs = float(cfg_shrink["eps_abs"])
        self.log = logger
        self.diag = {}

    # --------------------------------------------------------------------------------
    def fit(self, pm: pd.DataFrame, roles: pd.Series, mu_pm: pd.DataFrame | None = None):
        """pm — таблица «игрок × матч» обучающей части; roles — амплуа по player_id.
        mu_pm — часть, по которой оцениваются prior-средние μ_r (для test — КХЛ-компонента)."""
        mt = float(self.c["min_toi_split_min"]) * 60
        a, b = halves_split(pm, self.cols, mt)
        xa = per60(a[self.cols].values, a.toi_s.values)
        xb = per60(b[self.cols].values, b.toi_s.values)
        Ta, Tb = a.toi_s.values / 60, b.toi_s.values / 60
        Ma, Mb = a.games.values.astype(float), b.games.values.astype(float)
        rr = np.asarray(roles.reindex(a.index).values, dtype=object)
        self.diag["n_split_players"] = int(len(a))
        # Малая выборка половин: совместная оценка не считается устойчивой (флаг для гейта L5).
        self.unstable = bool(len(a) < MIN_SPLIT_PLAYERS)
        self.diag["unstable_joint"] = self.unstable
        # Σσ, Σω — регрессия внешних произведений разностей (по всем амплуа)
        d = xa - xb
        A1 = 1 / Ta + 1 / Tb if len(a) else np.zeros(0)
        A2 = 1 / np.maximum(Ma, 1.0) + 1 / np.maximum(Mb, 1.0) if len(a) else np.zeros(0)
        self.omega_identified, cond = _design_identified(A1, A2)
        self.diag["moment_design_cond"] = cond
        self.diag["omega_identified"] = self.omega_identified
        # Неидентифицируемый ω²: план сводится к одному столбцу 1/T, ω² остаётся нулём.
        # Σσ, Σω по всем амплуа (с переоценкой после проекции, P055) и, при noise_by_role, по амплуа.
        hw = (Ta * Tb) / (Ta + Tb) if len(a) else np.zeros(0)
        self.noise_by_role = bool(self.c.get("noise_by_role", False))
        self._fit_noise(d, A1, A2, rr, hw)
        self.T, self.T_raw, self.T_info, self.no_signal = {}, {}, {}, {}
        self.overall_no_signal, self.overall_threshold = {}, {}
        for r in ROLES:
            m, pooled = self._role_mask(rr, r, hw)
            offdiag_rho = None
            if not m.any():
                C = np.zeros((self.k, self.k))
            else:
                w = hw[m] / hw[m].sum()
                ca = xa[m] - (w[:, None] * xa[m]).sum(0)
                cb = xb[m] - (w[:, None] * xb[m]).sum(0)
                C = (ca * w[:, None]).T @ cb
                C = 0.5 * (C + C.T)
                if bool(self.c.get("t_offdiag_shrink", False)):
                    C, sh = offdiag_shrink(ca, cb, w)
                    offdiag_rho = sh["rho"]
            self.T_raw[r] = C
            if float(np.trace(C)) <= self.eps_abs:
                self.no_signal[r] = True
                self.T[r], info = psd_project(np.zeros_like(C), self.eps_abs, self.c["eps_floor_rel"])
            else:
                self.no_signal[r] = False
                self.T[r], info = psd_project(C, self.eps_abs, self.c["eps_floor_rel"])
            info["n_players"] = int(m.sum())
            info["pooled_roles"] = bool(pooled)
            info["oneT1_raw"] = float(C.sum())
            if offdiag_rho is not None:
                info["offdiag_rho"] = offdiag_rho
            self.T_info[r] = info
            # «нет сигнала в сумме» — по сырой 1ᵀT_r1, а не по спроецированной (см. OVERALL_SIGNAL_REL)
            self.overall_threshold[r] = overall_signal_threshold(C, self.eps_abs)
            self.overall_no_signal[r] = bool(self.no_signal[r] or overall_no_signal(C, self.eps_abs))
        # одномерная модель суммы (запасной вариант)
        ta, tb = xa.sum(1), xb.sum(1)
        self.uni = {}
        uni_noise = self._uni_noise((ta - tb)[:, None], A1, A2, rr, hw)
        for r in ROLES:
            s2u, o2u = uni_noise[r]
            m, pooled = self._role_mask(rr, r, hw)
            w = hw[m] / hw[m].sum() if m.any() else np.zeros(0)
            tau2, upper = weighted_cross_cov(ta[m], tb[m], w)
            # верхняя граница строго положительна: пол ε_abs — только для полностью вырожденных данных
            self.uni[r] = {"tau2": tau2, "tau2_upper": max(upper, self.eps_abs),
                           "sigma2": float(s2u), "omega2": float(o2u),
                           "pooled_roles": bool(pooled), "n_players": int(m.sum())}
        # prior-средние μ_r по разрешённой части (с весом TOI)
        self.mu = self.prior_mu(pm if mu_pm is None else mu_pm, roles)
        self.diag["mu"] = {r: self.mu[r].tolist() for r in ROLES}
        return self

    def _noise(self, d, A1, A2, reweight: bool) -> tuple[np.ndarray, np.ndarray, dict]:
        """Σσ, Σω по строкам `d` (с проверкой идентифицируемости ω² по этим же строкам)."""
        k = d.shape[1]
        if not len(d):
            Ss, i1 = _project_noise(np.zeros((k, k)), self.eps_abs, self.c["eps_floor_rel"], zero_floor=False)
            So, i2 = _project_noise(np.zeros((k, k)), self.eps_abs, self.c["eps_floor_rel"], zero_floor=True)
            return Ss, So, {"refit": False, "converged": True, "Sigma_sigma_proj": i1, "Sigma_omega_proj": i2,
                            "omega_identified": False, "design_cond": float("inf")}
        ident, cond = _design_identified(A1, A2)
        Ss, So, info = fit_noise_moments(d, A1, A2, ident and self.omega_identified, self.eps_abs,
                                         self.c["eps_floor_rel"], reweight)
        return Ss, So, {**info, "omega_identified": ident, "design_cond": cond}

    def _role_rows(self, rr: np.ndarray, r: str, hw: np.ndarray) -> np.ndarray | None:
        """Строки амплуа для шума по амплуа; None — шум общий (флаг выключен или мало игроков)."""
        if not self.noise_by_role:
            return None
        m, pooled = self._role_mask(rr, r, hw)
        return None if pooled else m

    def _fit_noise(self, d, A1, A2, rr, hw) -> None:
        Ss, So, info = self._noise(d, A1, A2, reweight=True)
        self.Sig_s_raw, self.Sig_o_raw = info.get("Sigma_sigma_raw", Ss), info.get("Sigma_omega_raw", So)
        self.Sig_s, self.Sig_o = Ss, So
        self.diag["Sigma_sigma_proj"] = info["Sigma_sigma_proj"]
        self.diag["Sigma_omega_proj"] = info["Sigma_omega_proj"]
        self.diag["noise_refit"] = bool(info["refit"])
        self.diag["noise_converged"] = bool(info["converged"])
        self.Sig_s_r, self.Sig_o_r, self.noise_info = {}, {}, {}
        for r in ROLES:
            m = self._role_rows(rr, r, hw)
            if m is None:
                self.Sig_s_r[r], self.Sig_o_r[r] = Ss, So
                self.noise_info[r] = {"source": "pooled", "refit": bool(info["refit"]),
                                      "converged": bool(info["converged"])}
                continue
            Sr, Or, ir = self._noise(d[m], A1[m], A2[m], reweight=True)
            self.Sig_s_r[r], self.Sig_o_r[r] = Sr, Or
            self.noise_info[r] = {"source": "role", "n_players": int(m.sum()), "refit": bool(ir["refit"]),
                                  "converged": bool(ir["converged"]), "n_iter": int(ir.get("n_iter", 0)),
                                  "omega_identified": bool(ir["omega_identified"]),
                                  "design_cond": float(ir["design_cond"]),
                                  "Sigma_sigma_proj": ir["Sigma_sigma_proj"],
                                  "Sigma_omega_proj": ir["Sigma_omega_proj"]}

    def _uni_noise(self, dsum, A1, A2, rr, hw) -> dict:
        """σ², ω² одномерной модели суммы: общие или по амплуа (те же правила, что у матриц)."""
        Ss, So, _ = self._noise(dsum, A1, A2, reweight=False)
        out = {}
        for r in ROLES:
            m = self._role_rows(rr, r, hw)
            if m is not None:
                Sr, Or, _ = self._noise(dsum[m], A1[m], A2[m], reweight=False)
                out[r] = (float(Sr[0, 0]), float(Or[0, 0]))
            else:
                out[r] = (float(Ss[0, 0]), float(So[0, 0]))
        return out

    @staticmethod
    def _role_mask(rr: np.ndarray, role: str, hw: np.ndarray) -> tuple[np.ndarray, bool]:
        """Маска игроков амплуа; при недостатке игроков или нулевом весе амплуа объединяются."""
        m = rr == role
        if m.sum() >= MIN_ROLE_PLAYERS and hw[m].sum() > 0:
            return m, False
        pooled = np.ones(len(rr), dtype=bool)
        if pooled.any() and hw[pooled].sum() > 0:
            return pooled, True
        return np.zeros(len(rr), dtype=bool), True

    def prior_mu(self, src: pd.DataFrame, roles: pd.Series) -> dict:
        """μ_r — prior-среднее амплуа с весом TOI; обучаемая величина, только обучающая часть.

        `src` задаёт разрешённую выборку: для применения к test — КХЛ-компонента train.
        """
        g = src.groupby("player_id")[self.cols + ["toi_s"]].sum()
        g["role"] = roles.reindex(g.index).values
        mu = {}
        for r in ROLES:
            gg = g[g.role == r]
            tot = float(gg.toi_s.sum())
            mu[r] = (gg[self.cols].sum().to_numpy(dtype=float) / tot * 3600.0 if tot > 0
                     else np.zeros(self.k))
        return mu

    def with_mu(self, mu: dict) -> "JointEB":
        """Копия подогнанной модели с другим prior-средним (T_r, Σσ, Σω не меняются)."""
        other = copy.copy(self)
        other.mu = {r: np.asarray(mu[r], dtype=float) for r in ROLES}
        other.diag = dict(self.diag)
        other.diag["mu"] = {r: other.mu[r].tolist() for r in ROLES}
        return other

    # --------------------------------------------------------------------------------
    def _solve(self, A: np.ndarray, B: np.ndarray, pid: str, jitter_log: list):
        """Решение (T_r + S_i) X = B Холецким; при неудаче — диагональ + ε·max(tr/k, ε_abs).

        Ошибка формы — дефект программы, а не численная неудача: поднимается ValueError и не
        уходит в ветку с ε. В лог пишется только ε, при котором разложение и решение прошли.
        """
        if A.ndim != 2 or A.shape[0] != A.shape[1] or B.ndim != 2 or B.shape[0] != A.shape[0]:
            raise ValueError(f"система для {pid}: форма A {A.shape}, B {B.shape} несовместима")
        k = A.shape[0]
        if not (np.isfinite(A).all() and np.isfinite(B).all()):
            raise np.linalg.LinAlgError(f"нечисловая система для {pid}")
        try:
            return cho_solve(cho_factor(A, lower=True), B), 0.0
        except np.linalg.LinAlgError:
            pass
        tr = float(np.trace(A))
        eps = float(self.c["jitter_start"])
        while eps <= float(self.c["jitter_max"]) + 1e-15:
            try:
                sol = cho_solve(cho_factor(A + np.eye(k) * eps * max(tr / k, self.eps_abs), lower=True), B)
            except np.linalg.LinAlgError:
                eps *= 10
                continue
            jitter_log.append({"player_id": pid, "eps": eps})
            return sol, eps
        raise np.linalg.LinAlgError(f"Холецкий не прошёл для {pid}")

    def posterior(self, ids, roles, sums: np.ndarray, toi_s: np.ndarray, games: np.ndarray):
        """sums — суммы компонентов (не на 60), toi_s — секунды, games — матчи. Возвращает dict."""
        n = len(ids)
        th = np.zeros((n, self.k))
        Vd = np.zeros((n, self.k, self.k))
        jl, fallback = [], []
        prior_only = []
        for i in range(n):
            r = roles[i]
            T = self.T[r]
            mu = self.mu[r]
            if toi_s[i] <= 0 or games[i] <= 0:
                # отдельная ветка без деления на TOI: апостериорное = prior
                th[i] = mu
                Vd[i] = T
                prior_only.append(ids[i])
                continue
            Tm = toi_s[i] / 60.0
            x = sums[i] / toi_s[i] * 3600.0
            S = self.Sig_s_r[r] / Tm + self.Sig_o_r[r] / games[i]
            try:
                sol, eps = self._solve(T + S, np.column_stack([x - mu, T]), ids[i], jl)
            except np.linalg.LinAlgError:
                fallback.append(ids[i])
                th[i] = mu
                Vd[i] = T
                continue
            th[i] = mu + T @ sol[:, 0]
            Vd[i] = T - T @ sol[:, 1:]
            Vd[i] = 0.5 * (Vd[i] + Vd[i].T)
        return {"theta": th, "V": Vd, "jitter_log": jl, "cholesky_fallback": fallback,
                "prior_only": prior_only}

    def _uni_no_signal(self, r: str) -> bool:
        """Одномерная модель суммы тоже без сигнала: τ² (= сырая 1ᵀT_r1) не выше порога суммы."""
        thr = self.overall_threshold.get(r, self.eps_abs)
        return bool(self.uni[r]["tau2"] <= thr or self.T_info[r]["oneT1_raw"] <= thr)

    def prior_var_overall(self, r: str) -> float:
        """Дисперсия Overall игрока без смен — та же, что получает строка prior-ветки в overall()."""
        if not self.overall_no_signal[r]:
            return float(self.T[r].sum())
        u = self.uni[r]
        return float(u["tau2_upper"] if self._uni_no_signal(r) else u["tau2"])

    def _univariate_overall(self, r: str, sum_i: float, toi_i: float, games_i: float) -> tuple:
        """Одномерная ветка суммы: (θ̂ суммы, дисперсия, w, пометка) для одной строки."""
        u = self.uni[r]
        mu = float(self.mu[r].sum())
        if self._uni_no_signal(r):
            # сигнала нет: оценка — prior, её ошибка имеет дисперсию prior τ² ≤ верхней границы
            return mu, float(u["tau2_upper"]), 0.0, NO_SIGNAL_FLAG
        if toi_i <= 0:
            return mu, float(u["tau2"]), 0.0, ""
        s_i = u["sigma2"] / (toi_i / 60) + u["omega2"] / max(games_i, 1.0)
        wi = u["tau2"] / (u["tau2"] + s_i)
        xs = sum_i / toi_i * 3600.0
        return mu + wi * (xs - mu), u["tau2"] * s_i / (u["tau2"] + s_i), wi, ""

    def overall(self, ids, roles, sums, toi_s, games, m_r: dict | None = None, toi_weights=None):
        """Overall и компоненты профиля; m_r — описательный центр (если None — считается по выборке).

        Центр Overall `m_r_overall` берётся у того оценщика, которым посчитаны строки амплуа:
        у одномерного — TOI-взвешенное среднее одномерных θ̂ суммы, у совместного — Σ m_r.
        Профиль (компоненты) центрируется векторным m_r совместной модели.
        """
        post = self.posterior(ids, roles, sums, toi_s, games)
        th, V = post["theta"], post["V"]
        roles = np.asarray(roles, dtype=object)
        toi_s = np.asarray(toi_s, dtype=float)
        games = np.asarray(games, dtype=float)
        sums = np.asarray(sums, dtype=float)
        wv = toi_s if toi_weights is None else np.asarray(toi_weights, dtype=float)
        given_centre = m_r is not None
        if m_r is None:
            m_r = {r: _weighted_centre(th[roles == r], wv[roles == r], np.zeros(self.k)) for r in ROLES}
        M = np.vstack([m_r[r] for r in roles]) if len(roles) else np.zeros((0, self.k))
        prof = th - M
        theta_ov = th.sum(1)
        var = V.sum(axis=(1, 2))
        oneT1 = np.array([self.T[r].sum() for r in roles])
        w = np.clip(1 - var / np.maximum(oneT1, 1e-300), 0, 1)
        method = np.full(len(ids), METHOD_JOINT, dtype=object)
        flag = np.full(len(ids), "", dtype=object)
        # нет сигнала в сумме (сырая 1ᵀT_r1) или в T_r -> одномерный запасной вариант для Overall
        bad = np.array([bool(self.overall_no_signal[r]) for r in roles], dtype=bool)
        for i in np.flatnonzero(bad):
            theta_ov[i], var[i], w[i], flag[i] = self._univariate_overall(
                roles[i], float(sums[i].sum()), float(toi_s[i]), float(games[i]))
            method[i] = METHOD_UNIVARIATE
        pos = {pid: i for i, pid in enumerate(ids)}
        for pid in post["cholesky_fallback"]:
            if not bad[pos[pid]]:
                method[pos[pid]] = METHOD_CHOLESKY_FAILED
        m_ov = {}
        for r in ROLES:
            sel = roles == r
            own = (not given_centre) and sel.any() and bool(self.overall_no_signal[r])
            m_ov[r] = (float(_weighted_centre(theta_ov[sel], wv[sel], np.sum(m_r[r])))
                       if own else float(np.sum(m_r[r])))
        centre = np.array([m_ov[r] for r in roles], dtype=float)
        return {"overall": theta_ov - centre, "theta_overall": theta_ov, "var": var, "w": w,
                "profile": prof, "theta": th, "V": V, "m_r": m_r, "m_r_overall": m_ov,
                "method": method, "flag": flag, "jitter_log": post["jitter_log"],
                "cholesky_fallback": post["cholesky_fallback"], "prior_only": post["prior_only"]}

    # --------------------------------------------------------------------------------
    def to_state(self) -> dict:
        """Сериализуемое (JSON) состояние для применения к новой выборке с замороженными моментами."""
        return {
            "cols": list(self.cols), "shrink": {k: _plain(v) for k, v in dict(self.c).items()},
            "T": {r: self.T[r].tolist() for r in ROLES},
            "T_info": {r: {k: _plain(v) for k, v in self.T_info[r].items()} for r in ROLES},
            "no_signal": {r: bool(self.no_signal[r]) for r in ROLES},
            "overall_no_signal": {r: bool(self.overall_no_signal[r]) for r in ROLES},
            "overall_threshold": {r: float(self.overall_threshold[r]) for r in ROLES},
            "uni": {r: {k: _plain(v) for k, v in self.uni[r].items()} for r in ROLES},
            "mu": {r: self.mu[r].tolist() for r in ROLES},
            "Sigma_sigma": self.Sig_s.tolist(), "Sigma_omega": self.Sig_o.tolist(),
            "noise_by_role": bool(self.noise_by_role),
            "Sigma_sigma_r": {r: self.Sig_s_r[r].tolist() for r in ROLES},
            "Sigma_omega_r": {r: self.Sig_o_r[r].tolist() for r in ROLES},
        }

    @classmethod
    def from_state(cls, state: dict, logger=None) -> "JointEB":
        """Модель из `to_state()`: те же T_r, Σσ, Σω, μ_r, одномерная модель и решения о сигнале."""
        eb = cls(list(state["cols"]), dict(state["shrink"]), logger=logger)
        eb.T = {r: np.asarray(state["T"][r], dtype=float) for r in ROLES}
        eb.T_info = {r: dict(state["T_info"][r]) for r in ROLES}
        eb.no_signal = {r: bool(state["no_signal"][r]) for r in ROLES}
        eb.overall_no_signal = {r: bool(state["overall_no_signal"][r]) for r in ROLES}
        eb.overall_threshold = {r: float(state["overall_threshold"][r]) for r in ROLES}
        eb.uni = {r: dict(state["uni"][r]) for r in ROLES}
        eb.mu = {r: np.asarray(state["mu"][r], dtype=float) for r in ROLES}
        eb.Sig_s = np.asarray(state["Sigma_sigma"], dtype=float)
        eb.Sig_o = np.asarray(state["Sigma_omega"], dtype=float)
        # состояние до шума по амплуа: общий шум у всех амплуа
        eb.noise_by_role = bool(state.get("noise_by_role", False))
        eb.Sig_s_r = {r: np.asarray(state.get("Sigma_sigma_r", {}).get(r, state["Sigma_sigma"]), dtype=float)
                      for r in ROLES}
        eb.Sig_o_r = {r: np.asarray(state.get("Sigma_omega_r", {}).get(r, state["Sigma_omega"]), dtype=float)
                      for r in ROLES}
        return eb


def _plain(v):
    """Значение для JSON: numpy-скаляры -> float/int/bool, остальное как есть."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _weighted_centre(values: np.ndarray, weights: np.ndarray, default):
    """Среднее с весом TOI; без веса — простое среднее; без строк — `default`."""
    if weights.sum() > 0:
        return (values * (weights if values.ndim == 1 else weights[:, None])).sum(0) / weights.sum()
    if len(values):
        return values.mean(0)
    return default


def reliability(w: np.ndarray, hi: float = 0.7, med: float = 0.4) -> np.ndarray:
    return np.where(w >= hi, "HIGH", np.where(w >= med, "MEDIUM", "LOW"))


class UnivariateEB:
    """Одномерное стягивание одного показателя (для профиля и ориентиров)."""

    def __init__(self, eps_abs=1e-10, noise_by_role: bool = False):
        self.eps = eps_abs
        self.noise_by_role = bool(noise_by_role)

    def fit(self, pm: pd.DataFrame, col: str, roles: pd.Series, min_toi_s: float = 300, mu_pm=None):
        self.col = col
        a, b = halves_split(pm, [col], min_toi_s)
        xa = a[col].values / a.toi_s.values * 3600 if len(a) else np.zeros(0)
        xb = b[col].values / b.toi_s.values * 3600 if len(b) else np.zeros(0)
        Ta, Tb = a.toi_s.values / 60, b.toi_s.values / 60
        A1 = 1 / Ta + 1 / Tb if len(a) else np.zeros(0)
        A2 = (1 / np.maximum(a.games.values, 1.0) + 1 / np.maximum(b.games.values, 1.0)
              if len(a) else np.zeros(0))
        self.omega_identified, self.design_cond = _design_identified(A1, A2)
        rr = np.asarray(roles.reindex(a.index).values, dtype=object)
        hw = Ta * Tb / (Ta + Tb) if len(a) else np.zeros(0)
        d = (xa - xb)[:, None] if len(a) else np.zeros((0, 1))
        self.sigma2, self.omega2, self.noise_refit = self._noise(d, A1, A2, self.omega_identified)
        self.sigma2_r, self.omega2_r = {}, {}
        for r in ROLES:
            m, pooled = JointEB._role_mask(rr, r, hw) if len(a) else (None, True)
            if self.noise_by_role and not pooled:
                ident, _ = _design_identified(A1[m], A2[m])
                self.sigma2_r[r], self.omega2_r[r], _ = self._noise(d[m], A1[m], A2[m], ident and self.omega_identified)
            else:
                self.sigma2_r[r], self.omega2_r[r] = self.sigma2, self.omega2
        self.tau2, self.tau2_upper, self.pooled = {}, {}, {}
        for r in ROLES:
            m, pooled = JointEB._role_mask(rr, r, hw)
            self.pooled[r] = bool(pooled)
            w = hw[m] / hw[m].sum() if m.any() else np.zeros(0)
            self.tau2[r], upper = weighted_cross_cov(xa[m], xb[m], w)
            self.tau2_upper[r] = max(upper, self.eps)
        self.mu = self.prior_mu(pm if mu_pm is None else mu_pm, roles)
        self.n = int(len(a))
        return self

    def _noise(self, d: np.ndarray, A1: np.ndarray, A2: np.ndarray, ident: bool) -> tuple[float, float, bool]:
        """σ², ω² одной регрессии (σ² ≥ ε, ω² ≥ 0, переоценка после обрезки — P055)."""
        if not len(d):
            return self.eps, 0.0, False
        Ss, So, info = fit_noise_moments(d, A1, A2, ident, self.eps, 1e-6, reweight=False)
        return float(Ss[0, 0]), float(So[0, 0]), bool(info["refit"])

    def prior_mu(self, src: pd.DataFrame, roles: pd.Series) -> dict:
        """μ_r — prior-среднее амплуа с весом TOI на разрешённой выборке (только обучающая часть)."""
        g = src.groupby("player_id")[[self.col, "toi_s"]].sum()
        g["role"] = roles.reindex(g.index).values
        mu = {}
        for r in ROLES:
            gg = g[g.role == r]
            tot = float(gg.toi_s.sum())
            mu[r] = float(gg[self.col].sum() / tot * 3600) if tot > 0 else 0.0
        return mu

    def with_mu(self, mu: dict) -> "UnivariateEB":
        """Копия подогнанной модели с другим prior-средним (τ², σ², ω² не меняются)."""
        other = copy.copy(self)
        other.mu = {r: float(mu[r]) for r in ROLES}
        return other

    def no_signal(self, r: str) -> bool:
        """τ² ≈ 0: межигрового сигнала нет, w = 0, «сигнал не обнаружен»."""
        return bool(self.tau2[r] <= self.eps)

    def prior_var(self, r: str) -> float:
        """Дисперсия оценки θ̂ = μ_r (prior-ветка); без сигнала — верхняя граница τ², не ноль."""
        return float(self.tau2_upper[r] if self.no_signal(r) else self.tau2[r])

    def posterior(self, roles, sums, toi_s, games):
        """θ̂ = μ + w(x − μ), v = τ²s/(τ² + s); prior-ветка и «нет сигнала» — θ̂ = μ, v = prior_var."""
        roles = np.asarray(roles, dtype=object)
        toi_s = np.asarray(toi_s, dtype=float)
        games = np.asarray(games, dtype=float)
        sums = np.asarray(sums, dtype=float)
        th = np.zeros(len(roles))
        v = np.zeros(len(roles))
        w = np.zeros(len(roles))
        for i, r in enumerate(roles):
            t2 = self.tau2[r]
            if toi_s[i] <= 0 or self.no_signal(r):
                th[i], v[i], w[i] = self.mu[r], self.prior_var(r), 0.0
                continue
            s = self.sigma2_r[r] / (toi_s[i] / 60) + self.omega2_r[r] / max(games[i], 1.0)
            w[i] = t2 / (t2 + s)
            th[i] = self.mu[r] + w[i] * (sums[i] / toi_s[i] * 3600 - self.mu[r])
            v[i] = t2 * s / (t2 + s)
        return th, v, w
