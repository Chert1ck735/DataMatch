"""L2: V согласована с Беллманом — события без весов 1/N, одновременные интервалы децилей.

Решение пользователя 06.10.2026 по аудиту L2: вес 1/N зависит от длины отрезка, то есть от
будущего исхода, и оценивает E_w[G|s] вместо беллмановской V. Корреляцию событий одного
отрезка учитывает только бутстрэп целых матчей.
"""
import numpy as np
import pandas as pd

from datamatch import l2_gbm, l2_procedure, l2_value

P_CONTINUE = (0.5, 0.9)
REWARD = 0.03


def _known_chain(n_matches=120, segments_per_match=120, seed=1):
    rng = np.random.default_rng(seed)
    rows, node, seg = [], 0, 0
    for match in range(n_matches):
        for _ in range(segments_per_match):
            state, states = rng.integers(2), []
            while True:
                states.append(state)
                if rng.random() > P_CONTINUE[state]:
                    break
                state = rng.integers(2)
            for i, st in enumerate(states):
                last = i == len(states) - 1
                rows.append(dict(node=node, match_id=match, segment=seg, next_node=-1 if last else node + 1,
                                 prev_node=-1 if i == 0 else node - 1, sigma_next=0 if last else 1,
                                 action="shot", x=-10.0 if st == 0 else 10.0, y=0.0, mp="EV5", team_id=1,
                                 home_team_id=1, valid_for_transition=True, terminal=last, possession=seg))
                node += 1
            seg += 1
    nodes = pd.DataFrame(rows)
    state = (nodes.x > 0).astype(int).to_numpy()
    p = np.array(P_CONTINUE)
    v_true = np.linalg.solve(np.eye(2) - np.outer(p, [0.5, 0.5]), np.full(2, REWARD))
    return nodes, np.full(len(nodes), REWARD), state, v_true


def test_event_weights_are_unit_per_event():
    nodes, _, _, _ = _known_chain(n_matches=2, segments_per_match=5)
    assert np.array_equal(l2_procedure.event_weights(nodes), np.ones(len(nodes)))


def test_markov_recovers_bellman_value_on_known_chain():
    nodes, r, state, v_true = _known_chain()
    markov = l2_value.MarkovV(l2_value.Grid([-31, 0, 31], [-16, 16]), kappa=1e-6).fit(nodes, r)
    v = markov.value(nodes)
    for k in (0, 1):
        assert abs(v[state == k][0] - v_true[k]) < 0.004


def test_calibration_gate_accepts_true_conditional_mean():
    nodes, r, state, v_true = _known_chain()
    G = l2_value.realized_return(nodes, r)
    report = l2_value.eval_value_model(nodes, G, v_true[state], float(np.mean(G)), B=300)
    assert report["calibration_all_deciles_ok"]


def test_decile_intervals_are_simultaneous_bonferroni():
    nodes, r, state, v_true = _known_chain(n_matches=40, segments_per_match=60)
    G = l2_value.realized_return(nodes, r)
    report = l2_value.eval_value_model(nodes, G, v_true[state], float(np.mean(G)), bins=10, B=200)
    assert report["decile_interval_level"] == 1 - 0.05 / 10
    assert all(d["residual_interval"]["level"] == 1 - 0.05 / 10 for d in report["deciles"])


def test_gbm_value_trains_with_unit_event_weights(monkeypatch):
    seen = {}

    class Recorder:
        def fit(self, X, y, sample_weight=None):
            seen["w"] = np.asarray(sample_weight)
            return self

    nodes, r, _, _ = _known_chain(n_matches=2, segments_per_match=10)
    model = l2_gbm.GBMV.__new__(l2_gbm.GBMV)
    model.opts, model.hist, model.geo = {}, False, None
    monkeypatch.setattr(model, "_model", lambda: Recorder())
    monkeypatch.setattr(l2_gbm, "v_features", lambda n, hist, geo: pd.DataFrame({"x": n.x.to_numpy()}))
    monkeypatch.setattr(l2_gbm, "mirror_faceoffs", lambda X, y, w, is_fo: (X, y, w))
    model._fit_one(nodes, r, None)
    assert np.allclose(seen["w"], 1.0)
