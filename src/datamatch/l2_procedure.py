"""Fold-local V fitting; nuisance strength is cross-fitted by match."""
import numpy as np
import pandas as pd

from . import l2_gbm, l2_value


def event_weights(nodes):
    """Вес физического события в обучении, калибровке и оценке V — единица.

    Прежний вес 1/N зависел от длины отрезка, то есть от будущего исхода, и оценивал
    E_w[G|s] вместо беллмановской V (аудит 06.10.2026). Корреляцию событий одного отрезка
    учитывает бутстрэп целых матчей, а не перевзвешивание цели.
    """
    return np.ones(len(nodes))


def fitting_folds(nodes, supplied, seed, calibrate):
    """Validate whole-match folds before fitting; invalid input is never a fallback."""
    ids = np.unique(nodes.match_id.to_numpy())
    if supplied is None:
        rng = np.random.default_rng(seed)
        assignment = dict(zip(ids, rng.permutation(len(ids)) % min(3, len(ids))))
        f = nodes.match_id.map(assignment).to_numpy()
    else:
        f = np.asarray(supplied)
    if (f.ndim != 1 or len(f) != len(nodes) or
            not np.issubdtype(f.dtype, np.number) or not np.isfinite(f).all() or
            (f < 0).any() or (f != np.floor(f)).any()):
        raise ValueError('Folds must provide one finite nonnegative integer per node')
    f = f.astype(int, copy=True)
    grouped = pd.DataFrame({'match': nodes.match_id.to_numpy(), 'fold': f}).groupby('match').fold
    if (grouped.nunique() != 1).any():
        raise ValueError('Folds must keep each complete match together')
    match_folds = grouped.first()
    if match_folds.nunique() < 2:
        raise ValueError('Folds require at least two nonempty groups')
    minimum_train = 2 if calibrate else 1
    if any((match_folds != k).sum() < minimum_train for k in match_folds.unique()):
        raise ValueError('Folds leave insufficient training matches for fitting/calibration')
    return f


class ValueProcedure:
    def __init__(self, cfg, spec, seed=0):
        self.cfg, self.spec, self.seed = cfg, dict(spec), seed

    def _strength(self, nodes, G):
        w = event_weights(nodes)
        valid = np.isfinite(G)
        x = pd.DataFrame({'team': nodes.team_id.to_numpy(), 'sum': w*np.where(valid, G, 0.),
                          'w': w*valid})
        t = x.groupby('team')[['sum', 'w']].sum()
        return (t['sum']/(t.w+20.)).to_dict()

    def fit(self, nodes, r, calibrate=True, xg_spec=None, folds=None):
        if nodes.match_id.nunique() < (3 if calibrate else 2):
            raise ValueError("At least three matches for calibrated value fitting; two for base fitting")
        f = fitting_folds(nodes, folds, self.seed, calibrate)
        self.training_folds_ = f.copy()
        self.gamma = self.spec.get('gamma_per_s')
        G = l2_value.realized_return(nodes, r, self.gamma, self.spec.get("own_poss_only", False))
        self.strength = self._strength(nodes, G)
        fold_data = {}
        def fold_reward(k):
            from .node_graph import subset_nodes
            tr = f != k
            ntr, nva = subset_nodes(nodes, tr), subset_nodes(nodes, ~tr)
            if k not in fold_data:
                if xg_spec is None:
                    rt, rv = r[tr], r[~tr]
                else:
                    # Calibration and nuisance cross-fits refit the reward model too.
                    from .stage_l2 import fit_xg, reward
                    xg = fit_xg(self.cfg, xg_spec, ntr)
                    rt, rv = reward(ntr, xg), reward(nva, xg)
                fold_data[k] = rt, rv
            return ntr, nva, *fold_data[k]
        if self.spec['kind'] == 'markov':
            c = self.cfg['value']['markov']
            self.model = l2_value.MarkovV(l2_value.Grid(c['x_edges'], c['y_edges']),
                c['prior_strength'], self.gamma, self.spec.get("own_poss_only", False)).fit(nodes, r)
        else:
            strength = np.zeros(len(nodes))
            for k in np.unique(f):
                tr = f != k
                ntr, _, rt, _ = fold_reward(k)
                gt = l2_value.realized_return(ntr, rt, self.gamma, self.spec.get("own_poss_only", False))
                mapping = self._strength(ntr, gt)
                strength[~tr] = .5*(nodes.loc[~tr, 'team_id'].map(mapping).fillna(0.).to_numpy() -
                    nodes.loc[~tr, 'opp_team_id'].map(mapping).fillna(0.).to_numpy())
            self.training_strength_ = strength.copy()
            n = nodes.assign(team_strength=strength)
            self.model = l2_gbm.GBMV(self.cfg, self.spec['kind'], self.seed, self.spec).fit(n, r)
        self.calibration = (1., 0.)
        if calibrate:
            predictions, targets, weights, fo_flags = [], [], [], []
            for k in np.unique(f):
                tr = f != k
                ntr, nva, rt, rv = fold_reward(k)
                sub = ValueProcedure(self.cfg, self.spec, self.seed).fit(
                    ntr, rt, calibrate=False, xg_spec=xg_spec,
                    folds=None if folds is None else f[tr])
                gv = l2_value.realized_return(nva, rv, self.gamma, self.spec.get("own_poss_only", False))
                # Калибровка учится на нейтральных прогнозах: именно они экспортируются в δ (L3).
                ev, yv, pv = evaluation_view(nva, gv, sub, neutral=True)
                base = event_weights(nva)
                fova = nva.action.eq('faceoff').to_numpy()
                wv = np.r_[base*np.where(fova, .5, 1.), base[fova]*.5]
                good = np.isfinite(yv)
                predictions.append(pv[good]); targets.append(yv[good]); weights.append(wv[good])
                fo_flags.append(ev.action.eq('faceoff').to_numpy()[good])
            p, y, w, isfo = map(np.concatenate, [predictions, targets, weights, fo_flags])
            self.calibration = l2_value.MonotoneCalibration.fit(p, y, w, isfo)
        return self

    def _raw_value(self, nodes, neutral=False):
        """V модели до калибровки (марковская V согласована с переходами по Беллману)."""
        strength = np.zeros(len(nodes)) if neutral else .5*(nodes.team_id.map(self.strength).fillna(0.).to_numpy() -
            nodes.opp_team_id.map(self.strength).fillna(0.).to_numpy())
        raw = self.model.value(nodes.assign(team_strength=strength))
        # Pre-outcome faceoff value must change sign exactly when perspectives swap.
        fo = nodes.action.eq("faceoff").to_numpy()
        if fo.any():
            mirrored = mirror_faceoff_nodes(nodes, self.cfg)
            mirrored["team_strength"] = -strength[fo]
            raw[fo] = .5*(raw[fo] - self.model.value(mirrored))
        return raw, fo

    def delta_value(self, nodes, neutral=False):
        """V для δ = r + σV(s') − V(s). При value.delta_from_raw — без калибровки: изотоническая g
        нелинейна и даёт систематический сдвиг δ на действие (METHODOLOGY 3, поправка 08.10)."""
        if bool(self.cfg["value"].get("delta_from_raw", False)):
            return self._raw_value(nodes, neutral)[0]
        return self.value(nodes, neutral)

    def value(self, nodes, neutral=False):
        raw, fo = self._raw_value(nodes, neutral)
        if isinstance(self.calibration, tuple):      # аффинная поправка, заданная напрямую
            slope, offset = self.calibration
            return slope*raw + np.where(fo, 0., offset)
        return self.calibration(raw, fo)


def mirror_faceoff_nodes(nodes, cfg=None):
    from . import io
    from .l0_events import MP_FLIP, visible_angle
    from .config import load_config
    f = nodes[nodes.action.eq('faceoff')].copy()
    f['node'] = np.arange(len(f)) + (int(nodes.node.max())+1 if len(nodes) else 0)
    f['prev_node'] = -1
    f['x'], f['y'] = -f.x, -f.y
    f['team_id'], f['opp_team_id'] = f.opp_team_id.to_numpy(), f.team_id.to_numpy()
    f['is_home'] = ~f.is_home
    for own, opp in [('own_sk', 'opp_sk'), ('own_g', 'opp_g')]:
        f[own], f[opp] = f[opp].to_numpy(), f[own].to_numpy()
    f['mp'] = f.mp.map(MP_FLIP)
    f['score_diff'] = -f.score_diff
    geo = io.rink_geometry(load_config() if cfg is None else cfg)
    gx = f.rink_type.map({k: v['goal_line_opp'] for k, v in geo.items()}).to_numpy()
    ph = f.rink_type.map({k: v['post_half_width'] for k, v in geo.items()}).to_numpy()
    f['dist'] = np.hypot(gx-f.x.to_numpy(), f.y.to_numpy())
    f['angle_vis'] = visible_angle(gx-f.x.to_numpy(), f.y.to_numpy(), ph)
    return f


def evaluation_view(nodes, G, proc, neutral=False):
    """Faceoffs are evaluated before the outcome, in both team perspectives.

    neutral=True оценивает экспортируемые значения (нейтральная сила команд), как их видит L3.
    """
    f = nodes.action.eq('faceoff').to_numpy()
    mirrored = mirror_faceoff_nodes(nodes, proc.cfg)
    nd = pd.concat([nodes, mirrored], ignore_index=True)
    nd['evaluation_probability'] = np.r_[np.where(f, .5, 1.), np.full(f.sum(), .5)]
    target = np.r_[G, -G[f]]
    prediction = np.r_[proc.value(nodes, neutral=neutral), proc.value(mirrored, neutral=neutral)]
    return nd, target, prediction


class TableDistillation:
    """Weighted cell/time/manpower approximation of a frozen GBM, with no refitting to labels."""
    def __init__(self, cfg):
        self.cfg = cfg
        c = cfg['value']['markov']
        self.indexer = l2_value.MarkovV(l2_value.Grid(c['x_edges'], c['y_edges']))

    def fit(self, nodes, teacher):
        mirrored = mirror_faceoff_nodes(nodes, self.cfg)
        fo = nodes.action.eq('faceoff').to_numpy()
        view = pd.concat([nodes, mirrored], ignore_index=True)
        y = np.r_[teacher.value(nodes, neutral=True), teacher.value(mirrored, neutral=True)]
        base = event_weights(nodes)
        w = np.r_[base*np.where(fo, .5, 1.), base[fo]*.5]
        state = self.indexer.state(view)
        keys = state*2 + view.action.eq('faceoff').to_numpy()
        size = self.indexer.S*2
        mass = np.bincount(keys, weights=w, minlength=size)
        sums = np.bincount(keys, weights=w*y, minlength=size)
        parents = self.indexer.parent(np.arange(self.indexer.S)).repeat(2)*2 + np.tile([0, 1], self.indexer.S)
        pm = np.bincount(parents, weights=mass)
        ps = np.bincount(parents, weights=sums)
        fallback = ps/np.maximum(pm, 1e-12)
        self.table = np.divide(sums, mass, out=fallback[parents].copy(), where=mass > 0)
        return self

    def value(self, nodes, neutral=False):
        fo = nodes.action.eq('faceoff').to_numpy()
        raw = self.table[self.indexer.state(nodes)*2 + fo]
        if fo.any():
            mirror = mirror_faceoff_nodes(nodes, self.cfg)
            raw[fo] = .5*(raw[fo]-self.table[self.indexer.state(mirror)*2+1])
        return raw
