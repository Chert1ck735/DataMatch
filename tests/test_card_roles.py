"""Карточка, роли и похожие игроки (METHODOLOGY 11.2): ожидания — синтетика и ручной расчёт."""
import numpy as np
import pandas as pd
import pytest

from datamatch import card_roles as cr


def _blobs(seed=0, n=150):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
    X = np.vstack([c + rng.standard_normal((n, 2)) * 0.6 for c in centers])
    return X, np.repeat([0, 1, 2], n)


def test_features_faceoff_only_for_centers():
    assert "comp_faceoff" in cr.role_features("C")
    assert "comp_faceoff" not in cr.role_features("W")
    assert "comp_faceoff" not in cr.role_features("D")
    with pytest.raises(KeyError):
        cr.role_features("G")


def test_standardizer_hand_values_and_apply():
    df = pd.DataFrame({"a": [1.0, 3.0, 5.0], "b": [2.0, 2.0, 8.0]})
    sc = cr.fit_scaler(df, ["a", "b"])
    assert sc.mean.tolist() == [3.0, 4.0]
    assert sc.sd == pytest.approx([2.0, np.sqrt(12.0)])             # SD с ddof = 1
    z = cr.apply_scaler(sc, pd.DataFrame({"a": [5.0], "b": [4.0]}))
    assert z[0] == pytest.approx([1.0, 0.0])


def test_standardizer_rejects_constant_feature():
    with pytest.raises(ValueError, match="b"):
        cr.fit_scaler(pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 1.0]}), ["a", "b"])


def test_bic_selects_three_separated_clusters():
    X, _ = _blobs()
    sel = cr.select_gmm(X, k_max=6, seed=1, n_init=3)
    assert sel.k == 3
    assert set(sel.bic) == {1, 2, 3, 4, 5, 6}
    assert min(sel.bic, key=sel.bic.get) == 3


def test_bic_selects_one_for_single_gaussian():
    rng = np.random.default_rng(4)
    sel = cr.select_gmm(rng.standard_normal((400, 3)), k_max=4, seed=1, n_init=3)
    assert sel.k == 1


def test_split_half_ari_identical_halves_is_one_and_independent_is_near_zero():
    X, _ = _blobs()
    sel = cr.select_gmm(X, k_max=4, seed=1, n_init=3)
    assert cr.split_half_ari(sel.model, X, X) == pytest.approx(1.0)
    rng = np.random.default_rng(9)
    noise = rng.standard_normal((len(X), 2)) * 4
    assert abs(cr.split_half_ari(sel.model, noise, rng.permutation(noise))) < 0.1


def test_cluster_names_from_profile():
    feats = ["comp_shot", "comp_loss", "comp_recovery", "comp_creation"]
    assert cr.cluster_name(np.array([1.2, -0.2, 0.1, 0.6]), feats) == "бросающий, созидатель"
    assert cr.cluster_name(np.array([0.1, -0.8, 0.2, 0.0]), feats) == "много потерь"
    assert cr.cluster_name(np.array([0.3, 0.1, -0.4, 0.2]), feats) == cr.NEUTRAL_NAME
    # не больше двух и по убыванию |z|
    assert cr.cluster_name(np.array([0.6, 0.9, -1.5, 0.0]), feats) == "мало отборов, надёжный с шайбой"


def test_mahalanobis_nearest_identity_cov_hand_case():
    pool = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0], [10.0, 10.0]])
    idx, dist = cr.nearest(np.array([0.0, 0.0]), pool, np.eye(2), k=3)
    assert idx.tolist() == [1, 2, 0]
    assert dist.tolist() == pytest.approx([1.0, 2.0, 5.0])


def test_mahalanobis_uses_covariance():
    # при дисперсии 4 по первой оси точка (2, 0) ближе, чем (0, 1.5)
    pool = np.array([[2.0, 0.0], [0.0, 1.5]])
    idx, dist = cr.nearest(np.zeros(2), pool, np.diag([4.0, 1.0]), k=2)
    assert idx.tolist() == [0, 1]
    assert dist.tolist() == pytest.approx([1.0, 1.5])


def test_closeness_percentile_hand_values():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    assert cr.closeness_pct(2.5, ref) == pytest.approx(50.0)
    assert cr.closeness_pct(0.5, ref) == pytest.approx(100.0)
    assert cr.closeness_pct(4.0, ref) == pytest.approx(0.0)


def test_pair_distances_count_and_values():
    Z = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
    d = cr.pair_distances(Z, np.eye(2))
    assert sorted(d.tolist()) == pytest.approx([1.0, np.sqrt(18.0), 5.0])


def test_similar_table_excludes_self_low_pool_and_other_role():
    players = pd.DataFrame({"player_id": ["q", "a", "b", "c", "d"], "role": ["D"] * 4 + ["W"],
                            "team_id": ["T1", "T1", "T2", "T2", "T2"], "w": [0.2, 0.9, 0.5, 0.1, 0.9]})
    Z = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.1, 0.0], [0.0, 0.0]])
    out = cr.similar_table(players, Z, {"D": np.eye(2), "W": np.eye(2)}, k=5, min_w=0.4)
    q = out[(out.player_id == "q") & (out.pool == "all")]
    assert q.similar_id.tolist() == ["a", "b"]                     # c — LOW, d — другое амплуа
    assert q.query_unstable.all()                                  # сам q — LOW
    other = out[(out.player_id == "q") & (out.pool == "other_teams")]
    assert other.similar_id.tolist() == ["b"]                      # без своей команды


def test_alternate_halves_follow_player_match_order():
    pm = pd.DataFrame({"player_id": ["p"] * 4 + ["r"] * 3,
                       "match_id": ["M3", "M1", "M4", "M2", "M9", "M5", "M7"]})
    a, b = cr.alternate_halves(pm)
    assert sorted(a[a.player_id == "p"].match_id) == ["M1", "M3"]
    assert sorted(b[b.player_id == "p"].match_id) == ["M2", "M4"]
    assert sorted(a[a.player_id == "r"].match_id) == ["M5", "M9"]
    assert b[b.player_id == "r"].match_id.tolist() == ["M7"]
    assert len(a) + len(b) == len(pm)


def test_similar_table_uses_role_dimension_of_shared_matrix():
    # общая матрица шириной 3 (у C три признака), у D — два: третий столбец D (нули-заполнители) не читается
    players = pd.DataFrame({"player_id": ["x", "y", "z"], "role": ["D"] * 3, "team_id": ["T"] * 3,
                            "w": [0.9, 0.9, 0.9]})
    Z = np.array([[0.0, 0.0, 9.0], [3.0, 4.0, 0.0], [0.0, 1.0, 0.0]])
    out = cr.similar_table(players, Z, {"D": np.eye(2)}, k=2, min_w=0.4)
    x = out[(out.player_id == "x") & (out.pool == "all")]
    assert x.similar_id.tolist() == ["z", "y"]
    assert x.distance.tolist() == pytest.approx([1.0, 5.0])


def test_signal_subspace_drops_null_direction_and_whitens():
    # z3 = z1 + z2: ранг 2 → нулевое направление (1, 1, −1)/√3 отбрасывается
    rng = np.random.default_rng(3)
    a = rng.standard_normal((500, 2))
    Z = np.column_stack([a[:, 0], a[:, 1], a[:, 0] + a[:, 1]])
    sub = cr.fit_subspace(Z)
    assert sub.vectors.shape == (3, 2)
    U = cr.project(sub, Z)
    assert np.cov(U, rowvar=False) == pytest.approx(np.eye(2), abs=1e-9)   # отбелено на наборе
    shifted = Z[:5] + 7.0 * np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)
    assert cr.project(sub, shifted) == pytest.approx(U[:5])                 # сдвиг вдоль нуля не виден


def test_signal_subspace_keeps_full_rank_data():
    rng = np.random.default_rng(5)
    sub = cr.fit_subspace(rng.standard_normal((400, 4)) * np.array([1.0, 2.0, 0.5, 1.5]))
    assert sub.vectors.shape == (4, 4)


def test_center_back_to_z_scale():
    # центр в отбеленных координатах → z: z = V·diag(√λ)·c
    rng = np.random.default_rng(8)
    Z = rng.standard_normal((300, 3)) * np.array([2.0, 1.0, 0.5])
    sub = cr.fit_subspace(Z)
    c = cr.project(sub, np.array([[1.0, -2.0, 0.5]]))[0]
    assert cr.to_z(sub, c) == pytest.approx([1.0, -2.0, 0.5])



def test_select_gmm_uses_tied_covariance():
    # Литература (Consensus, roles.md): для интерпретируемых ролей — общая ковариация кластеров.
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(c, 0.3, (150, 2)) for c in ([0, 0], [3, 0], [0, 3])])
    sel = cr.select_gmm(X, k_max=5, seed=1, n_init=3)
    assert sel.model.covariance_type == "tied"
    assert sel.k == 3


def test_tied_gmm_components_differ_by_centers_not_spread():
    # Один центр, два разброса (узкий и широкий). Полная ковариация даёт компоненты с почти общим
    # центром (различие только в разбросе); общая — компоненты, разнесённые по положению.
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal(0, 0.3, (300, 2)), rng.normal(0, 2.0, (300, 2))])
    sel = cr.select_gmm(X, k_max=4, seed=1, n_init=3)
    means = sel.model.means_
    if sel.k > 1:
        gaps = [np.linalg.norm(a - b) for i, a in enumerate(means) for b in means[i + 1:]]
        assert min(gaps) > 1.0


def test_unique_names_suffix_duplicates():
    assert cr.unique_names(["a", "a", "b", "a"]) == ["a", "a (2)", "b", "a (3)"]


def test_cluster_summary_spread_and_exemplars_hand_case():
    # Кластер 0: z = (0,0), (2,0), (1,0) → центр (1,0); SD по признакам (1, 0); ближайший к центру — "p3".
    Z = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [10.0, 10.0]])
    labels = np.array([0, 0, 0, 1])
    centers = np.array([[1.0, 0.0], [10.0, 10.0]])
    out = cr.cluster_summary(Z, labels, centers, ["p1", "p2", "p3", "p4"], ["comp_shot", "comp_loss"], n_examples=2)
    assert out[0]["sd_z"] == pytest.approx({"comp_shot": 1.0, "comp_loss": 0.0})
    assert out[0]["examples"] == ["p3", "p1"]
    assert out[0]["n"] == 3 and out[1]["examples"] == ["p4"]
