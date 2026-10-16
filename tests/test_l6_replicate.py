"""L6: реплика бутстрэпа моделей ценности (l6_replicate) на крошечных синтетических графах.

Тяжёлые шаги (переобучение xG/V, применение к test, замороженный L5) подменяются: здесь
проверяется только механика реплики — уникальность ID копий матчей, фолды по исходному матчу,
детерминизм по seed, возобновление из кэша и учёт ошибок.
"""
import numpy as np
import pandas as pd
import pytest

from datamatch import l6_replicate as R
from datamatch import l6_uncertainty as U
from datamatch.cache import Cache
from datamatch.config import load_config


# --------------------------------------------------------------------------------------
# Синтетика: граф узлов трёх матчей с глобальными ID отрезков/владений и строковыми ключами
# --------------------------------------------------------------------------------------
def tiny_graph() -> pd.DataFrame:
    rows = []
    node = 0
    seg = 0
    for mid, n in (("G0001", 3), ("G0002", 2), ("G0003", 4)):
        for j in range(n):
            last = j == n - 1
            rows.append({
                "match_id": mid, "node": node,
                "next_node": -1 if last else node + 1, "prev_node": -1 if j == 0 else node - 1,
                "last_pass_node": node - 1 if j >= 1 else -1,
                "segment": seg + j // 2, "possession": seg + j // 2,
                "segment_key": f"{mid}:{j // 2}", "possession_key": f"{mid}:{j // 2}",
                "player_shift_id": f"{mid}:T1:P{j:05d}:1:{j}" if j % 2 == 0 else None,
                "stint": j, "x": float(j),
            })
            node += 1
        seg += (n + 1) // 2
    return pd.DataFrame(rows).assign(source_node=lambda d: d.node + 1000)


def tiny_splits() -> dict:
    return {"matches": {"match_id": ["G0001", "G0002", "G0003"], "outer": [False, False, True],
                        "fold": [0, 1, -1], "scheme": ["v1"] * 3}}


def tiny_context(**overrides) -> R.ReplicateContext:
    base = dict(
        final_nodes=tiny_graph(), match_ids=("G0001", "G0002", "G0003"), splits=tiny_splits(),
        xg_spec={"family": "fake"}, value_spec={"kind": "markov"},
        test={"nodes": pd.DataFrame(), "toi": pd.DataFrame(), "stints": pd.DataFrame(),
              "events": pd.DataFrame()},
        positions=pd.DataFrame({"position": ["C", "D"], "role": ["C", "D"]},
                               index=pd.Index(["P1", "P2"], name="player_id")),
        member_ids=("P1", "P2"), team_comp={}, khl=None, variant="split", alpha=0.5,
        frozen={"state": 1}, fingerprint={"recipe": "r1"})
    base.update(overrides)
    return R.ReplicateContext(**base)


@pytest.fixture
def env(tmp_path):
    cfg = load_config().with_overrides({"paths": {
        "outputs": str(tmp_path / "outputs"), "artifacts": str(tmp_path / "artifacts"),
        "cache": str(tmp_path / "cache"), "logs": str(tmp_path / "logs")}})
    return cfg, Cache(cfg, {})


@pytest.fixture
def fake_chain(monkeypatch):
    """Подмена тяжёлых шагов: результат зависит только от seed и числа узлов выборки."""
    calls = {"fit": 0}

    def fit(cfg, ctx, nodes, folds, seed):
        calls["fit"] += 1
        return ("xg", seed), ("proc", seed, len(nodes))

    def apply(cfg, ctx, xg, proc):
        return pd.DataFrame({"seed": [proc[1]], "n": [proc[2]]})

    def overall(cfg, ctx, pm):
        rng = np.random.default_rng(int(pm.seed.iloc[0]))
        return pd.DataFrame({"player_id": list(ctx.member_ids),
                             "overall": rng.normal(0, 0.1, len(ctx.member_ids)) + pm.n.iloc[0] * 1e-3,
                             "var_post": np.full(len(ctx.member_ids), 0.01)})
    monkeypatch.setattr(R, "fit_frozen_recipes", fit)
    monkeypatch.setattr(R, "apply_to_test", apply)
    monkeypatch.setattr(R, "frozen_overall", overall)
    return calls


# --------------------------------------------------------------------------------------
# Граф бутстрэпа: копии матчей и перенумерация ID
# --------------------------------------------------------------------------------------
def test_duplicated_matches_get_unique_match_node_segment_and_possession_ids():
    # Arrange
    nodes = tiny_graph()
    before = nodes.copy()

    # Act
    g = R.bootstrap_graph(nodes, ["G0001", "G0003", "G0001", "G0001"])

    # Assert
    b = g.nodes
    pd.testing.assert_frame_equal(nodes, before)                       # вход не изменён
    assert sorted(set(b.match_id)) == ["G0001", "G0001~1", "G0001~2", "G0003"]
    assert len(b) == 3 * 3 + 4
    assert b.node.is_unique and list(b.node) == list(range(len(b)))
    assert (b.groupby("segment").match_id.nunique() == 1).all()
    assert (b.groupby("possession").match_id.nunique() == 1).all()
    for key in ("segment_key", "possession_key", "player_shift_id"):
        k = b[key].dropna()
        assert (k.str.split(":", n=1).str[0] == b.loc[k.index, "match_id"]).all(), key
        assert (b.dropna(subset=[key]).groupby(key).match_id.nunique() == 1).all(), key
    assert list(g.original_match) == [m.split(R.COPY_SEPARATOR)[0] for m in b.match_id]


def test_graph_references_stay_inside_their_own_copy():
    nodes = tiny_graph()
    g = R.bootstrap_graph(nodes, ["G0001", "G0001"])
    b = g.nodes.set_index("node")
    for col in ("next_node", "prev_node", "last_pass_node"):
        ref = b[col]
        linked = ref[ref >= 0]
        assert (b.loc[linked.to_numpy(), "match_id"].to_numpy() == b.loc[linked.index, "match_id"].to_numpy()).all()
    # source_node сохраняет исходный ID узла L0 у обеих копий
    assert sorted(b.source_node) == sorted(list(nodes.source_node[:3]) * 2)


def test_graph_rejects_unknown_matches_and_broken_string_keys():
    nodes = tiny_graph()
    with pytest.raises(R.ReplicateDataError):
        R.bootstrap_graph(nodes, ["G9999"])
    bad = nodes.assign(segment_key=nodes.segment_key.where(nodes.match_id != "G0002", "X:0"))
    with pytest.raises(R.ReplicateDataError):
        R.bootstrap_graph(bad, ["G0002", "G0002"])


def test_folds_are_assigned_by_original_match_so_copies_never_split():
    # Arrange
    g = R.bootstrap_graph(tiny_graph(), ["G0001", "G0001", "G0002", "G0003", "G0003"])

    # Act
    folds = R.original_match_folds(g.original_match, tiny_splits())

    # Assert
    frame = pd.DataFrame({"orig": g.original_match, "fold": folds})
    assert (frame.groupby("orig").fold.nunique() == 1).all()
    assert frame.groupby("orig").fold.first().to_dict() == {"G0001": 0, "G0002": 1, "G0003": 2}


# --------------------------------------------------------------------------------------
# Реплика: детерминизм, кэш и возобновление
# --------------------------------------------------------------------------------------
def test_replicate_is_deterministic_by_seed(env, fake_chain):
    cfg, _ = env
    ctx = tiny_context()
    a = R.compute_replicate(cfg, ctx, ["G0001", "G0002", "G0002"], seed=7)
    b = R.compute_replicate(cfg, ctx, ["G0001", "G0002", "G0002"], seed=7)
    c = R.compute_replicate(cfg, ctx, ["G0001", "G0002", "G0002"], seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a.overall, c.overall)
    assert list(a.columns) == list(R.REPLICATE_COLUMNS)


def test_replicate_resumes_from_cache_without_refitting(env, fake_chain, monkeypatch):
    # Arrange
    cfg, cache = env
    monkeypatch.setattr(R, "context", lambda cfg_, cache_: tiny_context())
    sample = ["G0003", "G0001", "G0003"]

    # Act
    first = R.replicate(cfg, cache, sample, seed=11, index=4)
    second = R.replicate(cfg, cache, list(reversed(sample)), seed=11, index=4)

    # Assert
    assert fake_chain["fit"] == 1                                # вторая — из кэша
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))


def test_replicate_cache_key_changes_with_recipe_seed_index_and_matches(env):
    _, cache = env
    ctx = tiny_context()
    k = R.replicate_key(cache, ctx, ["G0001"], 1, 0)
    assert k != R.replicate_key(cache, ctx, ["G0002"], 1, 0)
    assert k != R.replicate_key(cache, ctx, ["G0001"], 2, 0)
    assert k != R.replicate_key(cache, ctx, ["G0001"], 1, 1)
    assert k != R.replicate_key(cache, tiny_context(fingerprint={"recipe": "r2"}), ["G0001"], 1, 0)
    assert k == R.replicate_key(cache, ctx, ["G0001"], 1, 0)


def test_frozen_overall_contract_violation_is_not_a_data_error(env, monkeypatch):
    from datamatch import stage_l5
    cfg, _ = env
    monkeypatch.setattr(stage_l5, "apply_frozen_overall",
                        lambda eb, pm, pos, cfg_: pd.DataFrame({"player_id": ["P1"]}), raising=False)
    with pytest.raises(R.ReplicateContractError):
        R.frozen_overall(cfg, tiny_context(), pd.DataFrame())
    assert not issubclass(R.ReplicateContractError, U.EXPECTED_REPLICATE_ERRORS)


def test_frozen_overall_uses_the_l5_api_with_member_positions(env, monkeypatch):
    from datamatch import stage_l5
    cfg, _ = env
    seen = {}

    def apply(eb, pm, positions, cfg_):
        seen["eb"], seen["pos"] = eb, list(positions.player_id)   # контракт L5: player_id — колонка
        return pd.DataFrame({"player_id": ["P1", "P2"], "overall": [0.1, 0.2], "var_post": [0.01, 0.02],
                             "w": [0.5, 0.5], "method": "joint", "flag": ""})
    monkeypatch.setattr(stage_l5, "apply_frozen_overall", apply, raising=False)
    out = R.frozen_overall(cfg, tiny_context(), pd.DataFrame())
    assert seen == {"eb": {"state": 1}, "pos": ["P1", "P2"]}
    assert list(out.columns) == list(R.REPLICATE_COLUMNS)


def test_missing_l5_frozen_api_makes_the_replicate_unavailable(env, monkeypatch):
    from datamatch import stage_l5
    monkeypatch.delattr(stage_l5, "load_frozen_overall", raising=False)
    with pytest.raises(R.ReplicateUnavailable):
        R.frozen_api()


# --------------------------------------------------------------------------------------
# Параллельность и учёт ошибок
# --------------------------------------------------------------------------------------
def _module_level_replicate(matches, index, seed):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 0.05, 3), index=["P1", "P2", "P3"])


def _data_error_replicate(matches, index, seed):
    if index == 2:
        raise R.ReplicateDataError("в выборке нет бросков")
    return _module_level_replicate(matches, index, seed)


def test_process_pool_gives_the_same_variance_as_serial():
    ids = [f"M{i:02d}" for i in range(20)]
    serial = U.run_model_bootstrap(_module_level_replicate, ids, b=8, seed=3, n_jobs=1)
    pooled = U.run_model_bootstrap(_module_level_replicate, ids, b=8, seed=3, n_jobs=2,
                                   executor_factory=R.process_pool(2))
    assert pooled.status == "done"
    assert serial.var_model.to_numpy() == pytest.approx(pooled.var_model.to_numpy())


def test_replicate_data_error_is_counted_and_replaced_within_threshold():
    ids = [f"M{i:02d}" for i in range(20)]
    res = U.run_model_bootstrap(_data_error_replicate, ids, b=40, seed=3)
    assert res.status == "done" and res.b_done == 40 and res.n_failed == 1
    assert "нет бросков" in res.failures[0]


def test_worker_count_is_configurable_with_default_two(env):
    cfg, _ = env
    assert R.replicate_workers(cfg.with_overrides({"bootstrap": {"replicate_workers": None}})) == \
        R.DEFAULT_REPLICATE_WORKERS == 2
    assert R.replicate_workers(cfg.with_overrides({"bootstrap": {"replicate_workers": 2}})) == 2
    with pytest.raises(ValueError):
        R.replicate_workers(cfg.with_overrides({"bootstrap": {"replicate_workers": 0}}))


def test_frozen_overall_passes_player_id_as_column_to_real_l5_role_lookup(monkeypatch):
    # Регрессия (ревью 06.10.2026): позиции из stage_l3.player_positions держат player_id в индексе,
    # а настоящий stage_l5._positions_roles делает set_index("player_id") — KeyError на первой реплике.
    from types import SimpleNamespace

    from datamatch import l6_replicate as R
    from datamatch import stage_l5

    positions = pd.DataFrame({"position": ["C", "D"], "role": ["C", "D"]},
                             index=pd.Index(["P1", "P2"], name="player_id"))

    def apply_with_real_role_lookup(frozen, pm, pos, cfg):
        roles = stage_l5._positions_roles(pos)
        return pd.DataFrame({"player_id": list(roles.index), "overall": 0.0, "var_post": 1.0})

    monkeypatch.setattr(R, "frozen_api", lambda: (None, apply_with_real_role_lookup))
    ctx = SimpleNamespace(positions=positions, member_ids=("P1", "P2"), frozen={})
    out = R.frozen_overall(None, ctx, pd.DataFrame())
    assert sorted(out.player_id) == ["P1", "P2"]


# --------------------------------------------------------------------------------------
# Ревью F3/F4/F6, интеграция №1: память, пул, потоки, seed, --force
# --------------------------------------------------------------------------------------
def _fit_graph(n_matches: int = 12) -> pd.DataFrame:
    """Настоящий граф схемы L0 (tests.test_l2) плюс колонки, которые подгонке не нужны."""
    from tests.test_l2 import many_nodes
    nd = many_nodes(n_matches)
    goals = nd.action.eq("shot") & nd.match_id.isin(["M001", "M004", "M007", "M010"])
    nd.loc[goals, "is_goal"] = True
    return nd.assign(junk_text=[f"j{i}" for i in range(len(nd))], junk_num=np.arange(len(nd)) * 1.5,
                     segment_key=nd.match_id + ":" + nd.segment.astype(str),
                     possession_key=nd.match_id + ":" + nd.possession.astype(str),
                     player_shift_id=nd.match_id + ":T1:P00001:1:0", lineup_status="confirmed")


def _fit_splits(ids) -> dict:
    n = len(ids)
    return {"matches": {"match_id": list(ids), "outer": [i >= n - 3 for i in range(n)],
                        "fold": [i % 3 if i < n - 3 else -1 for i in range(n)], "scheme": ["v1"] * n}}


def test_pruned_fit_columns_keep_model_inputs_and_drop_unused_columns():
    # Arrange
    from datamatch import l1_xg
    nd = _fit_graph(3)

    # Act
    pruned = R.prune_fit_columns(nd)

    # Assert
    dropped = set(nd.columns) - set(pruned.columns)
    assert {"junk_text", "junk_num", "segment_key", "possession_key", "player_shift_id",
            "lineup_status"} <= dropped
    raw_xg = [c for c in l1_xg.FEATURES if c in nd.columns]
    needed = {"match_id", "node", "next_node", "prev_node", "last_pass_node", "segment", "possession",
              "action", "x", "y", "t_mono", "is_goal", "valid_for_transition", "period_nominal",
              "t_ref_source", "team_id", "opp_team_id", "home_team_id", "sigma_next", "terminal"}
    assert needed | set(raw_xg) <= set(pruned.columns)
    assert list(pruned.columns) == [c for c in nd.columns if c in pruned.columns]     # порядок сохранён
    assert R.prune_fit_columns(pruned) is pruned                                       # повтор — без копии


@pytest.mark.parametrize("kind,xg_family", [("markov", "logit_spline"), ("markov", "lgbm_mono"),
                                            ("gbm", "logit_spline"), ("gbm_hist", "logit_spline")])
def test_replicate_on_pruned_graph_equals_replicate_on_full_graph(env, monkeypatch, kind, xg_family):
    # Arrange: настоящие fit_xg/reward/ValueProcedure; test — узлы того же графа
    from datamatch import l1_xg, stage_l2
    cfg, _ = env
    cfg = cfg.with_overrides({"value": {"gbm": {"n_estimators": 3},
                                        "markov": {"x_edges": [-31, 31], "y_edges": [-16, 16]}}})
    nd = _fit_graph()
    ids = sorted(nd.match_id.unique())
    params = {"n_estimators": 5, "learning_rate": 0.1, "num_leaves": 4, "min_child_samples": 1}
    xg_spec = {"family": xg_family, "l2": 1.0, "params": params, "features": list(l1_xg.FEATURES)}
    ctx = tiny_context(final_nodes=nd, match_ids=tuple(ids), splits=_fit_splits(ids), xg_spec=xg_spec,
                       value_spec={"kind": kind}, test={"nodes": nd})
    seen_columns = []
    real_fit = R.fit_frozen_recipes

    def recording_fit(cfg_, ctx_, nodes, folds, seed):
        seen_columns.append(set(nodes.columns))
        return real_fit(cfg_, ctx_, nodes, folds, seed)

    def apply(cfg_, ctx_, xg, proc):
        test = ctx_.test["nodes"]
        return pd.DataFrame({"r": stage_l2.reward(test, xg), "v": proc.value(test, neutral=True)})

    def overall(cfg_, ctx_, pm):
        s = float(pm.r.sum()) + float(pm.v.sum())
        return pd.DataFrame({"player_id": list(ctx_.member_ids), "overall": [s, 2 * s], "var_post": [0.1, 0.2]})

    monkeypatch.setattr(R, "fit_frozen_recipes", recording_fit)
    monkeypatch.setattr(R, "apply_to_test", apply)
    monkeypatch.setattr(R, "frozen_overall", overall)
    sample = ids[:6] + ids[:3] + ids[6:9]

    # Act
    pruned = R.compute_replicate(cfg, ctx, sample, seed=5)
    with monkeypatch.context() as mp:
        mp.setattr(R, "prune_fit_columns", lambda nodes: nodes)
        full = R.compute_replicate(cfg, ctx, sample, seed=5)

    # Assert
    pd.testing.assert_frame_equal(pruned, full)
    assert np.isfinite(pruned.overall).all()
    assert "junk_text" not in seen_columns[0] and "junk_text" in seen_columns[1]


def test_replicate_refits_xg_and_v_with_the_replicate_seed(env, monkeypatch):
    # F6: и xG (включая его рефиты в фолдах V через cfg процедуры), и V — на seed реплики.
    from datamatch import l2_procedure, stage_l2
    cfg, _ = env
    seen = []

    def fit_xg(cfg_, spec, nodes):
        seen.append(("xg", cfg_.seed))
        return "xg"

    class Proc:
        def __init__(self, cfg_, spec, seed):
            seen.append(("v", cfg_.seed, int(seed)))

        def fit(self, nodes, r, **kwargs):
            return self

    monkeypatch.setattr(stage_l2, "fit_xg", fit_xg)
    monkeypatch.setattr(stage_l2, "reward", lambda nodes, xg: np.zeros(len(nodes)))
    monkeypatch.setattr(l2_procedure, "ValueProcedure", Proc)
    rep_seed = cfg.seed + 12345

    R.fit_frozen_recipes(cfg, tiny_context(), tiny_graph(), np.zeros(9, dtype=int), rep_seed)

    assert seen == [("xg", rep_seed), ("v", rep_seed, rep_seed)]
    assert cfg.seed != rep_seed                                     # исходная конфигурация не изменена


def test_force_recomputes_cached_replicates_once_per_run(env, fake_chain, monkeypatch):
    # Интеграция №1: ratings --force/--force-all доходит до кэша реплик.
    cfg, cache = env
    monkeypatch.setattr(R, "context", lambda cfg_, cache_: tiny_context())
    sample = ["G0003", "G0001", "G0003"]
    R.replicate(cfg, cache, sample, seed=11, index=4)
    assert fake_chain["fit"] == 1

    R.make_replicate(cfg, cache)(sample, 4, 11)                      # без force — из кэша
    assert fake_chain["fit"] == 1

    forced = R.make_replicate(cfg, cache, force=True)
    forced(sample, 4, 11)                                            # force — пересчёт
    assert fake_chain["fit"] == 2
    forced(sample, 4, 11)                                            # тот же прогон (повтор после сбоя пула)
    assert fake_chain["fit"] == 2
    R.make_replicate(cfg, cache, force=True)(sample, 4, 11)          # новый прогон с force
    assert fake_chain["fit"] == 3


@pytest.mark.parametrize("workers,cpu,expected", [(1, 8, 8), (2, 8, 4), (3, 8, 2), (8, 8, 1), (16, 8, 1)])
def test_worker_thread_limit_keeps_workers_times_threads_within_cpu_count(workers, cpu, expected):
    threads = R.worker_threads(workers, cpu)
    assert threads == expected
    assert threads * workers <= cpu or threads == 1


def test_worker_initializer_sets_thread_limits(monkeypatch):
    import os
    for var in R.THREAD_ENV_VARS + (R.MODEL_THREADS_ENV,):
        monkeypatch.delenv(var, raising=False)
    limited = []
    monkeypatch.setattr(R, "_limit_threadpools", limited.append)

    R._init_worker(3)

    assert {v: os.environ[v] for v in R.THREAD_ENV_VARS} == {v: "3" for v in R.THREAD_ENV_VARS}
    assert {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"} <= set(R.THREAD_ENV_VARS)
    assert limited == [3]


def test_worker_context_load_is_serialized_by_a_file_lock(env, monkeypatch):
    # Загрузка контекста читает весь L0 (пик ~8 ГБ по замеру F3): процессы грузят его по очереди.
    import fcntl
    cfg, cache = env
    monkeypatch.setattr(R, "_CONTEXTS", {})
    monkeypatch.setattr(R, "_limit_threadpools", lambda threads: None)
    for var in R.THREAD_ENV_VARS + (R.MODEL_THREADS_ENV,):
        monkeypatch.delenv(var, raising=False)
    held = []

    def loader(cfg_, cache_):
        with open(cache.root / R.CONTEXT_LOAD_LOCK, "a") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fh, fcntl.LOCK_UN)
                held.append(False)
            except BlockingIOError:
                held.append(True)
        return tiny_context()

    R._init_worker(2, cfg, cache, loader)

    assert held == [True]
    assert R.context(cfg, cache).match_ids == ("G0001", "G0002", "G0003")


def _tiny_loader(cfg, cache):
    """Загрузчик контекста для процессов spawn: пишет метку (pid, лимит потоков) и отдаёт синтетику."""
    import os
    import uuid
    from pathlib import Path
    marks = Path(cfg.paths.artifacts) / "context_loads"
    marks.mkdir(parents=True, exist_ok=True)
    (marks / f"{os.getpid()}-{uuid.uuid4().hex}").write_text(os.environ.get("OMP_NUM_THREADS", ""))
    return tiny_context()


def _context_replicate(cfg, cache, matches, index, seed):
    ctx = R.context(cfg, cache)        # без инициализатора настоящий загрузчик здесь упал бы
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 0.05, len(ctx.member_ids)), index=list(ctx.member_ids))


def test_spawn_pool_loads_the_context_once_per_worker_in_the_initializer(env, monkeypatch):
    # Arrange
    import os
    from functools import partial
    cfg, cache = env
    cfg = cfg.with_overrides({"bootstrap": {"replicate_workers": 2}})
    monkeypatch.setattr(R, "_CONTEXTS", {})
    monkeypatch.setattr(R, "load_context", _tiny_loader)
    ids = [f"M{i:02d}" for i in range(20)]
    fn = partial(_context_replicate, cfg, cache)

    # Act
    pooled = U.run_model_bootstrap(fn, ids, b=6, seed=3, n_jobs=2, executor_factory=R.executor_factory(cfg, cache))

    # Assert
    assert R.START_METHOD in ("spawn", "forkserver")
    marks = list((cfg.paths.artifacts / "context_loads").iterdir())
    pids = {m.name.split("-")[0] for m in marks}
    assert 1 <= len(marks) <= 2 and len(pids) == len(marks)             # один раз на процесс, не на задачу
    assert str(os.getpid()) not in pids                                  # не в родителе
    # контекст грузится в окружении родителя, лимит потоков — после загрузки (проверка 06.10.2026)
    assert {m.read_text() for m in marks} == {os.environ.get("OMP_NUM_THREADS", "")}
    serial = U.run_model_bootstrap(fn, ids, b=6, seed=3)
    assert pooled.status == serial.status == "done"
    assert pooled.var_model.to_numpy() == pytest.approx(serial.var_model.to_numpy())


def test_pool_plan_releases_the_parent_copy_of_the_context(env, monkeypatch):
    # spawn: родителю нужен только список матчей; контекст держат процессы пула.
    from datamatch import stage_l6 as S
    cfg, cache = env
    cfg = cfg.with_overrides({"bootstrap": {"replicate_workers": 2}})
    monkeypatch.setattr(R, "_CONTEXTS", {})
    monkeypatch.setattr(R, "load_context", lambda cfg_, cache_: tiny_context())

    plan = S.bootstrap_plan(cfg, cache, "O1")

    assert plan.executor_factory is not None and plan.n_jobs == 2
    assert plan.matches == ("G0001", "G0002", "G0003")
    assert R._CONTEXTS == {}


def _dies_once(marker_dir, matches, index, seed):
    import os
    import signal
    from pathlib import Path
    mark = Path(marker_dir) / "died"
    if index == 1 and not mark.exists():
        mark.write_text("killed")
        os.kill(os.getpid(), signal.SIGKILL)          # как OOM-killer
    return _module_level_replicate(matches, index, seed)


def _always_dies(matches, index, seed):
    import os
    import signal
    os.kill(os.getpid(), signal.SIGKILL)


def test_pool_recovers_after_a_worker_is_killed(tmp_path):
    from functools import partial
    ids = [f"M{i:02d}" for i in range(20)]
    pooled = U.run_model_bootstrap(partial(_dies_once, str(tmp_path)), ids, b=6, seed=3, n_jobs=2,
                                   executor_factory=R.process_pool(2))
    serial = U.run_model_bootstrap(_module_level_replicate, ids, b=6, seed=3)
    assert (tmp_path / "died").exists()
    assert pooled.status == "done" and pooled.b_done == 6 and pooled.pool_restarts == 1
    assert pooled.var_model.to_numpy() == pytest.approx(serial.var_model.to_numpy())


def test_pool_that_keeps_dying_fails_with_a_clear_status_after_capped_restarts():
    ids = [f"M{i:02d}" for i in range(20)]
    res = U.run_model_bootstrap(_always_dies, ids, b=4, seed=3, n_jobs=2, executor_factory=R.process_pool(2))
    assert res.status == "failed" and res.pool_restarts == U.MAX_POOL_RESTARTS
    assert "BrokenProcessPool" in res.reason and "replicate_workers" in res.reason
    assert res.to_meta()["pool_restarts"] == U.MAX_POOL_RESTARTS


def test_worker_loads_context_before_setting_thread_limits(monkeypatch, tmp_path):
    # Интеграционная проверка 06.10.2026: переменные потоков входят в отпечаток окружения provenance L1/L2.
    # Выставленные до загрузки, они делали L1/L2 «устаревшими» в воркере -> UpstreamNotReady -> пул ломался.
    import contextlib
    import os
    from types import SimpleNamespace

    from datamatch import l6_replicate as R

    for var in R.THREAD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(R, "_context_load_lock", lambda cache: contextlib.nullcontext())
    monkeypatch.setattr(R, "_context_key", lambda cfg, cache: "k")
    monkeypatch.setattr(R, "_limit_threadpools", lambda n: None)
    seen = {}

    def loader(cfg, cache):
        seen.update({var: os.environ.get(var) for var in R.THREAD_ENV_VARS})
        return "ctx"

    R._init_worker(2, SimpleNamespace(), SimpleNamespace(root=tmp_path), loader)
    assert all(v is None for v in seen.values())
    assert all(os.environ.get(var) == "2" for var in R.THREAD_ENV_VARS)
