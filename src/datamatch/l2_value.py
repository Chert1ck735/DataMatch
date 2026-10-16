"""L2: ценность состояния V(s) — ожидаемая знаковая сумма xG от события до остановки (γ = 1).

V(s_t) = E[ Σ_{k=t..T} σ_{t,k} r_k | s_t ],  r_k = xG_k для попытки броска, иначе 0;
σ_{t,k} = +1, если действие k совершает команда узла t, иначе −1; T — последний узел
перед свистком, голом или концом периода (поглощение, V = 0).

M-Markov: состояние = клетка × номинальный остаток времени × манпауэр.
Спорный arrival исключён контрактом P005. Вес события — единица (аудит 06.10.2026).
Знаковая цепь: V = R + (P_своя − P_чужая) V  ->  (I − Q) V = R, Q = P_своя − P_чужая.
Решается линейной системой (без явного обращения). Строки редких состояний сглаживаются
к родительскому состоянию (клетка × манпауэр) с κ псевдонаблюдениями.

Вбрасывание: узел победителя; ценность до исхода V_fo(клетка, манпауэр) оценивается
по обеим перспективам: (победитель, +V_после) и (проигравший, зеркальная клетка,
перевёрнутый манпауэр, −V_после).

Дисконт по времени (вызов в турнире): Q_γ = переходы × γ^{Δt}.
Короткий вариант «только своё владение»: потеря — поглощение с V = 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .l0_events import MP_CLASSES, MP_FLIP



def transition_observed(nodes: pd.DataFrame) -> np.ndarray:
    """Known outgoing transition, including confirmed absorption; missing is not zero."""
    if "valid_for_transition" in nodes:
        return nodes.valid_for_transition.fillna(False).to_numpy(dtype=bool)
    if "terminal" in nodes:
        return (nodes.next_node.to_numpy() >= 0) | nodes.terminal.to_numpy(dtype=bool)
    # Compatibility with legacy fixtures: a gap before i censors the preceding node.
    ok = np.ones(len(nodes), dtype=bool)
    if "chain_gap_before" in nodes:
        same = nodes.segment.to_numpy()[1:] == nodes.segment.to_numpy()[:-1]
        ok[:-1] &= ~(same & nodes.chain_gap_before.to_numpy(dtype=bool)[1:])
    return ok


def next_positions(nodes):
    raw = nodes.next_node.to_numpy()
    positions = pd.Index(nodes.node).get_indexer(raw)
    if ((raw >= 0) & (positions < 0)).any():
        raise ValueError("Unresolved successor: pass a complete match graph")
    return positions


def realized_return(nodes: pd.DataFrame, r: np.ndarray, gamma_per_s: float | None = None,
                    own_poss_only: bool = False) -> np.ndarray:
    """Signed full-segment return; NaN if its future contains a censored transition."""
    tau = np.where(nodes.team_id.values == nodes.home_team_id.values, 1.0, -1.0)
    seg = nodes.possession.to_numpy() if own_poss_only else nodes.segment.to_numpy()
    G = np.zeros(len(nodes))
    if gamma_per_s is None:
        x = tau * r
        # обратная накопленная сумма внутри отрезка
        rev = x[::-1]
        seg_rev = seg[::-1]
        cs = pd.Series(rev).groupby(seg_rev).cumsum().values[::-1]
        G = tau * cs
    else:
        clock = nodes.t_mono.to_numpy() if "t_mono" in nodes else nodes.clock.to_numpy()
        acc = 0.0
        for i in range(len(nodes) - 1, -1, -1):
            if i == len(nodes) - 1 or seg[i + 1] != seg[i]:
                acc = 0.0
                t_next = clock[i]
            dt = max(0.0, t_next - clock[i])
            acc = tau[i] * r[i] + (gamma_per_s ** dt) * acc
            t_next = clock[i]
            G[i] = tau[i] * acc
    # A full-horizon return is unavailable for every prefix ending at an unknown
    # transition. The independently observed suffix remains usable.
    unknown = ~transition_observed(nodes) | ~np.isfinite(r)
    # Every earlier horizon through ANY unknown transition is unavailable.
    # Reverse grouped accumulation also avoids scanning all nodes per segment.
    censored = pd.Series(unknown[::-1]).groupby(seg[::-1]).cummax().to_numpy()[::-1]
    G[censored] = np.nan
    return G


class Grid:
    def __init__(self, x_edges, y_edges):
        self.xe = np.asarray(x_edges, float)
        self.ye = np.asarray(y_edges, float)
        self.nx = len(self.xe) - 1
        self.ny = len(self.ye) - 1
        self.n = self.nx * self.ny

    def cell(self, x, y):
        cx = np.clip(np.searchsorted(self.xe, x, side="right") - 1, 0, self.nx - 1)
        cy = np.clip(np.searchsorted(self.ye, y, side="right") - 1, 0, self.ny - 1)
        return cx * self.ny + cy


class MarkovV:
    name = "M-Markov"

    def __init__(self, grid: Grid, kappa: float = 20.0, gamma_per_s: float | None = None, own_poss_only: bool = False):
        self.g = grid
        self.kappa = kappa
        self.gamma = gamma_per_s
        self.own_only = own_poss_only
        self.na = 1  # disputed arrival excluded consistently with P005
        self.time_edges = np.array([0., 30., 120., 300., 600., 1200., np.inf])
        self.nt = len(self.time_edges)
        self.nm = len(MP_CLASSES)
        self.S = self.g.n * self.nt * self.nm

    # --- индексация ------------------------------------------------------------------
    def state(self, nodes: pd.DataFrame) -> np.ndarray:
        c = self.g.cell(nodes.x.values, nodes.y.values)
        ti = self.time_index(nodes)
        m = pd.Categorical(nodes.mp.values, categories=MP_CLASSES).codes
        if (m < 0).any():
            raise ValueError("Unknown manpower state")
        return (c * self.nt + ti) * self.nm + m

    def time_index(self, nodes):
        # Small synthetic fixtures without temporal context retain an unknown bin.
        if "period_nominal" not in nodes:
            return np.full(len(nodes), self.nt - 1, dtype=int)
        # Share the pre-action clock contract with GBM: a shot's own timestamp
        # is outcome-sensitive, and the first action after a gap has no anchor.
        from .l2_gbm import v_features
        remain = v_features(nodes, False).t_remain_pre.to_numpy()
        return np.where(np.isnan(remain), self.nt-1,
                        np.clip(np.searchsorted(self.time_edges, remain, side="right")-1, 0, self.nt-2))

    def parent(self, s: np.ndarray) -> np.ndarray:
        m = s % self.nm
        c = s // (self.nt * self.nm)
        return c * self.nm + m

    # --- обучение --------------------------------------------------------------------
    def fit(self, nodes: pd.DataFrame, r: np.ndarray, w: np.ndarray | None = None):
        # Переходы считаются по событиям без веса 1/N: он зависит от длины отрезка (будущего
        # исхода) и смещает оценку вероятности продолжения, ломая уравнение Беллмана.
        w = np.ones(len(nodes)) if w is None else np.asarray(w, float)
        is_fo = nodes.action.values == "faceoff"
        s = self.state(nodes)
        nxt = next_positions(nodes)
        sig = nodes.sigma_next.values.astype(float)
        has = nxt >= 0
        nxi = np.clip(nxt, 0, None)
        s_next = s[nxi]
        if self.gamma is not None:
            dt = np.clip((nodes.t_mono.to_numpy() if "t_mono" in nodes else nodes.clock.to_numpy())[nxi] -
                         (nodes.t_mono.to_numpy() if "t_mono" in nodes else nodes.clock.to_numpy()), 0, None)
            disc = np.where(has, self.gamma ** dt, 0.0)
        else:
            disc = np.where(has, 1.0, 0.0)
        if self.own_only:
            disc = np.where(sig < 0, 0.0, disc)   # потеря = поглощение с V = 0
        eligible = transition_observed(nodes) & np.isfinite(r) & np.isfinite(w) & (w > 0)
        chain = ~is_fo & eligible
        S = self.S
        # счётчики переходов: строка s, столбец s_next, вес w * σ * disc
        rows = s[chain & has]
        cols = s_next[chain & has]
        vals = (w * sig * disc)[chain & has]
        n_s = np.bincount(s[chain], weights=w[chain], minlength=S)
        R_s = np.bincount(s[chain], weights=(w * r)[chain], minlength=S)
        M = np.zeros((S, S))
        np.add.at(M, (rows, cols), vals)
        # сглаживание к родителю
        par = self.parent(np.arange(S))
        P = self.g.n * self.nm
        Mp = np.zeros((P, S))
        np.add.at(Mp, par, M)
        n_p = np.bincount(par, weights=n_s, minlength=P)
        R_p = np.bincount(par, weights=R_s, minlength=P)
        k = self.kappa
        with np.errstate(invalid="ignore", divide="ignore"):
            rowp = np.where(n_p[:, None] > 0, Mp / np.maximum(n_p, 1e-12)[:, None], 0.0)
            rp = np.where(n_p > 0, R_p / np.maximum(n_p, 1e-12), 0.0)
        denom = n_s + k
        Q = (M + k * rowp[par]) / denom[:, None]
        Rv = (R_s + k * rp[par]) / denom
        # состояния без данных и без данных родителя: Q = 0, R = 0 -> V = 0
        A = np.eye(S) - Q
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import spsolve
        self.V = spsolve(csr_matrix(A), Rv)
        if not np.isfinite(self.V).all() or np.max(np.abs(A @ self.V - Rv)) > 1e-8:
            raise ValueError("Markov Bellman system is singular or inconsistent")
        self.n_state = n_s
        self.R = Rv
        # ценность вбрасывания до исхода — по обеим перспективам
        fo = np.flatnonzero(is_fo & eligible)
        v_next = np.where(has[fo], sig[fo] * disc[fo] * self.V[s_next[fo]], 0.0) + r[fo]
        c_w = self.g.cell(nodes.x.values[fo], nodes.y.values[fo])
        c_l = self.g.cell(-nodes.x.values[fo], -nodes.y.values[fo])
        mp_w = pd.Categorical(nodes.mp.values[fo], categories=MP_CLASSES).codes
        mp_l = pd.Categorical(pd.Series(nodes.mp.values[fo]).map(MP_FLIP).values, categories=MP_CLASSES).codes
        ti = self.time_index(nodes)[fo]
        keys = np.r_[(c_w*self.nt+ti)*self.nm+mp_w, (c_l*self.nt+ti)*self.nm+mp_l]
        Pfo = self.g.n*self.nt*self.nm
        tgt = np.r_[v_next, -v_next]
        ww = np.r_[w[fo]/2, w[fo]/2]
        nfo = np.bincount(keys, weights=ww, minlength=Pfo)
        sfo = np.bincount(keys, weights=ww * tgt, minlength=Pfo)
        cells = keys // (self.nt*self.nm)
        nc = np.bincount(cells, weights=ww, minlength=self.g.n)
        sc = np.bincount(cells, weights=ww * tgt, minlength=self.g.n)
        vc = np.where(nc > 0, sc / np.maximum(nc, 1e-12), 0.0)
        cell_of = np.arange(Pfo) // (self.nt*self.nm)
        self.V_fo = (sfo + k * vc[cell_of]) / (nfo + k)
        return self

    def value(self, nodes: pd.DataFrame) -> np.ndarray:
        s = self.state(nodes)
        v = self.V[s]
        is_fo = nodes.action.values == "faceoff"
        if is_fo.any():
            c = self.g.cell(nodes.x.values[is_fo], nodes.y.values[is_fo])
            m = pd.Categorical(nodes.mp.values[is_fo], categories=MP_CLASSES).codes
            v = v.copy()
            v[is_fo] = self.V_fo[(c*self.nt+self.time_index(nodes)[is_fo])*self.nm+m]
        return v


def td_delta(nodes: pd.DataFrame, r: np.ndarray, v: np.ndarray, gamma_per_s: float | None = None,
             own_poss_only: bool = False) -> np.ndarray:
    """δ = r + σ γ^Δt V(next) − V; terminal continuation is 0, unknown is NaN."""
    nxt = next_positions(nodes)
    has = nxt >= 0
    nxi = np.clip(nxt, 0, None)
    sig = nodes.sigma_next.values.astype(float)
    disc = np.ones(len(nodes))
    if gamma_per_s is not None:
        dt = np.clip((nodes.t_mono.to_numpy() if "t_mono" in nodes else nodes.clock.to_numpy())[nxi] -
                         (nodes.t_mono.to_numpy() if "t_mono" in nodes else nodes.clock.to_numpy()), 0, None)
        disc = gamma_per_s ** dt
    if own_poss_only:
        disc = np.where(sig < 0, 0.0, disc)
    v_next = np.where(has, sig * disc * v[nxi], 0.0)
    delta = r + v_next - v
    delta[~transition_observed(nodes)] = np.nan
    return delta


def evaluation_weights(nodes, G):
    """Вес события в оценке V: вероятность перспективы (вбрасывание 0,5/0,5), если цель наблюдаема.

    Каждое физическое событие входит один раз, без веса 1/N и без случайной выборки одного
    события на отрезок: такая выборка снова даёт вес 1/N и оценивает E_w[G|s], а не V.
    Зависимость событий одного матча учитывает бутстрэп целых матчей в cluster_interval.
    """
    if (nodes.groupby("segment").match_id.nunique() > 1).any():
        raise ValueError("Global segment IDs must identify one match; assign unique IDs to bootstrap replicas")
    keep = np.isfinite(G)
    if not keep.any():
        raise ValueError("No observed finite value targets for evaluation")
    prob = nodes.get("evaluation_probability", pd.Series(1., index=nodes.index)).to_numpy(dtype=float)
    if not np.isfinite(prob).all() or (prob <= 0).any():
        raise ValueError("Evaluation probabilities must be finite and positive")
    return np.where(keep, prob, 0.)


def cluster_interval(nodes, numerator, denominator, B=1000, seed=0, level=0.95):
    """Ratio bootstrap of whole matches; events of one match are never IID.

    Ключи lo95/hi95 сохранены для совместимости потребителей; фактический уровень — в `level`.
    """
    if not 0 < level < 1:
        raise ValueError("Interval level must lie in (0, 1)")
    ids, inv = np.unique(nodes.match_id.to_numpy(), return_inverse=True)
    num = np.bincount(inv, weights=numerator, minlength=len(ids))
    den = np.bincount(inv, weights=denominator, minlength=len(ids))
    n_matches = int((den > 0).sum())
    point = float(num.sum()/den.sum()) if den.sum() > 0 else None
    if n_matches < 2:
        return {"mean": point, "lo95": None, "hi95": None, "level": level,
                "n_matches": n_matches, "status": "inconclusive"}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(B, len(ids)))
    total = den[draws].sum(axis=1)
    boot = np.divide(num[draws].sum(axis=1), total, out=np.full(B, np.nan), where=total > 0)
    valid = boot[np.isfinite(boot)]
    if not len(valid):
        return {"mean": point, "lo95": None, "hi95": None, "level": level,
                "n_matches": n_matches, "status": "inconclusive"}
    tail = (1 - level) / 2
    lo, hi = np.quantile(valid, [tail, 1 - tail])
    return {"mean": point, "lo95": float(lo), "hi95": float(hi), "level": level,
            "n_matches": n_matches, "status": "estimated"}


GATE_FAMILY_LEVEL = 0.95   # совместный уровень гейта калибровки по всем децилям


def eval_value_model(nodes, G, v, const, seed=0, bins=10, B=1000):
    keep = np.isfinite(G)
    if not np.isfinite(v[keep]).all():
        raise ValueError("Nonfinite prediction on an observed target")
    v = np.where(keep, v, 0.)
    w = evaluation_weights(nodes, G)
    ok = w > 0
    y = np.where(keep, G, 0.)
    mse = float(np.sum(w*(y-v)**2)/w.sum())
    mse0 = float(np.sum(w*(y-const)**2)/w.sum())
    # Десять проверок децилей образуют одно семейство: поправка Бонферрони даёт совместный
    # уровень 95%; без неё идеальная модель проваливала гейт примерно в 40% случаев.
    level = 1 - (1 - GATE_FAMILY_LEVEL) / bins
    # Quantile atoms (especially identical table values) share their mass across
    # boundaries. Every event with the same prediction receives the same fraction;
    # no arbitrary event ordering and no empty bins caused solely by ties.
    values, inverse = np.unique(v[ok], return_inverse=True)
    mass = np.bincount(inverse, weights=w[ok], minlength=len(values))
    end = np.cumsum(mass)
    start = end-mass
    deciles = []
    for k in range(bins):
        low, high = w.sum()*k/bins, w.sum()*(k+1)/bins
        overlap = np.maximum(0., np.minimum(end, high)-np.maximum(start, low))
        fraction = overlap/mass
        wk = np.zeros(len(w))
        wk[ok] = w[ok]*fraction[inverse]
        interval = cluster_interval(nodes, wk*(y-v), wk, B, seed, level)
        passed = interval["lo95"] is not None and interval["lo95"] <= 0 <= interval["hi95"]
        deciles.append({"pred": float(np.sum(wk*v)/wk.sum()) if wk.sum() else None,
            "obs": float(np.sum(wk*y)/wk.sum()) if wk.sum() else None,
            "residual_interval": interval, "weight": float(wk.sum()), "ok": bool(passed)})
    return {"mse": mse, "mse_const": mse0, "r2": 1-mse/mse0 if mse0 else None,
            "n_segments": int(nodes.loc[keep, "segment"].nunique()),
            "excluded_nodes": int((~keep).sum()), "deciles": deciles,
            "calibration_all_deciles_ok": all(x["ok"] for x in deciles),
            "deciles_ok": sum(x["ok"] for x in deciles),
            "decile_interval_level": level,
            "calibration_method": "all physical events with unit weight (faceoff perspectives 0.5/0.5); censored "
                                  "targets excluded without renormalization; whole-match residual bootstrap; "
                                  "Bonferroni-simultaneous decile intervals; fractional quantile ties"}


def per_segment_pick_errors(nodes, G, v):
    w = evaluation_weights(nodes, G)
    y = np.where(np.isfinite(G), G, 0.)
    v = np.where(np.isfinite(G), v, 0.)
    if not np.isfinite(v).all():
        raise ValueError("Nonfinite prediction on an observed target")
    err = pd.DataFrame({"seg": nodes.segment, "match": nodes.match_id, "e": w*(y-v)**2})
    err = err[w > 0].groupby("seg").agg(e=("e", "sum"), match=("match", "first"))
    return err.e.to_numpy(), err.match.to_numpy()


def paired_value_difference(nodes, G, a, b, seed=0, B=1000):
    w = evaluation_weights(nodes, G)
    y = np.where(np.isfinite(G), G, 0.)
    a, b = np.where(np.isfinite(G), a, 0.), np.where(np.isfinite(G), b, 0.)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Nonfinite prediction on an observed target")
    return cluster_interval(nodes, w*((y-a)**2-(y-b)**2), w, B, seed)


class MonotoneCalibration:
    """Монотонная поправка условного среднего V, обученная на кросс-фитах внутри train.

    Аффинная поправка `a·v + b` правит только наклон и сдвиг. На данных после неё остаётся
    систематический изгиб: на краях прогноза модель занижает фактическую знаковую сумму xG,
    в середине завышает. Одна прямая такую форму исправить не может, и гейт калибровки по
    децилям не проходит. Изотоническая регрессия правит форму, не меняя порядок состояний,
    поэтому ранжирование действий и игроков по вкладу сохраняется.

    Вбрасывания: V до исхода антисимметрична — то же состояние глазами соперника стоит −V.
    Произвольная монотонная g это свойство нарушила бы и дала бы вбрасыванию сдвиг, поэтому
    к ним применяется нечётная часть поправки g_odd(v) = (g(v) − g(−v)) / 2.

    Аффинная ветка сохраняется как запасная: при малой выборке изотоническая оценка
    переобучается, и тогда берётся прямая.
    """

    MIN_POINTS = 500          # меньше — изотоническая оценка ненадёжна, остаётся аффинная
    MIN_DISTINCT = 50         # число различных прогнозов, иначе ступенька вырождена

    def __init__(self, affine, iso=None):
        self.affine = (float(affine[0]), float(affine[1]))
        self.iso = iso

    @property
    def kind(self):
        return "isotonic" if self.iso is not None else "affine"

    @classmethod
    def fit(cls, p, y, w, isfo):
        """p — сырой прогноз, y — реализованная знаковая сумма, w — вес, isfa — вбрасывание.

        Свободный член аффинной ветки оценивается только по невбрасываниям: у вбрасываний
        он обязан быть нулевым, иначе ломается антисимметрия.
        """
        design = np.column_stack([p, (~isfo).astype(float)])
        normal = design.T @ (w[:, None] * design) + np.eye(2) * 1e-8
        affine = tuple(np.linalg.solve(normal, design.T @ (w * y)))
        ok = w > 0
        if ok.sum() < cls.MIN_POINTS or len(np.unique(p[ok])) < cls.MIN_DISTINCT:
            return cls(affine)
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(p[ok], y[ok], sample_weight=w[ok])
        return cls(affine, iso)

    def __call__(self, raw, isfo):
        raw = np.asarray(raw, dtype=float)
        isfo = np.asarray(isfo, dtype=bool)
        if self.iso is None:
            slope, offset = self.affine
            return slope * raw + np.where(isfo, 0.0, offset)
        out = self.iso.predict(raw)
        if isfo.any():
            out = np.where(isfo, .5 * (self.iso.predict(raw) - self.iso.predict(-raw)), out)
        return out
