"""L1: правила приёма кандидатов и устойчивость кэша OOF (аудит 06.10.2026)."""
import json

import numpy as np
import pandas as pd

from datamatch import l1_xg, stage_l1
from datamatch.cache import code_hash, frame_hash, obj_hash
from tests.test_l1 import _synthetic


def _calibrated_block(n_matches=300, shots_per_match=100, bias=1.0, seed=0):
    rng = np.random.default_rng(seed)
    mids = np.repeat(np.arange(n_matches), shots_per_match)
    p = rng.beta(1.0, 18.0, size=len(mids))
    y = (rng.random(len(mids)) < p).astype(float)
    return mids, y, np.clip(p * bias, 1e-6, 1 - 1e-6)


def test_xg_sum_interval_accepts_calibrated_model_beyond_two_percent_noise():
    # Идеальная модель: Σp/Σy выходит за ±2% из-за шума числа голов, но интервал по матчам
    # накрывает 1 почти всегда — жёсткий порог ±2% отклонял её примерно в 40% выборок.
    accepted = 0
    outside_two_percent = 0
    for seed in range(40):
        mids, y, p = _calibrated_block(seed=seed)
        iv = l1_xg.xg_sum_interval(mids, y, p, B=300, seed=seed)
        accepted += stage_l1.xg_sum_acceptable(iv)
        outside_two_percent += abs(iv["ratio"] - 1.0) > 0.02
    assert outside_two_percent >= 5
    assert accepted >= 36


def test_xg_sum_interval_rejects_systematic_ten_percent_bias():
    mids, y, p = _calibrated_block(bias=1.10, seed=1)
    iv = l1_xg.xg_sum_interval(mids, y, p, B=300, seed=1)
    assert iv["status"] == "estimated"
    assert iv["lo95"] > 1.0
    assert not stage_l1.xg_sum_acceptable(iv)


def test_optuna_preference_is_decided_on_tuning_folds_only():
    # Сравнение на поисковых фолдах решает; блок сравнения (фолды 3–4) в выборе не участвует.
    worse_on_tuning = {"diff": 0.002, "lo95": 0.001, "hi95": 0.003, "status": "estimated"}
    tied_on_tuning = {"diff": 0.0001, "lo95": -0.0005, "hi95": 0.0007, "status": "estimated"}
    assert not stage_l1.prefer_optuna(worse_on_tuning, n_trials=8, n_grid=12)
    assert stage_l1.prefer_optuna(tied_on_tuning, n_trials=8, n_grid=12)
    assert not stage_l1.prefer_optuna(tied_on_tuning, n_trials=12, n_grid=12)


def test_truncated_oof_cache_file_is_a_cache_miss(tmp_path):
    s = _synthetic(2000)
    s["blocked"] = False
    s["fold"] = s.match_id.str[1:].astype(int) % 5
    spec = {"family": "logit_spline", "l2": 1.0, "features": ["dist", "angle_vis", "mp"]}
    prov = {"spec": json.dumps(spec, sort_keys=True, default=str), "seed": 0,
            "shots": frame_hash(s, stage_l1.SHOT_HASH_COLS), "code": code_hash(*stage_l1.L1_CODE),
            "evaluation_folds": None, "runtime": stage_l1.numerical_environment()}
    target = tmp_path / f"oof_{obj_hash(prov)}.pkl"
    target.write_bytes(b"\x80\x05\x95")
    saved = dict(stage_l1._OOF_CACHE)
    stage_l1._OOF_CACHE.update(dir=tmp_path, force=False)
    try:
        p, _, _ = stage_l1.oof(spec, s, 0)
    finally:
        stage_l1._OOF_CACHE.clear()
        stage_l1._OOF_CACHE.update(saved)
    assert np.isfinite(p).all()
    assert target.stat().st_size > 3
    assert not list(tmp_path.glob("*.tmp*"))
    assert isinstance(pd.read_pickle(target), tuple)


def test_saved_selection_with_numpy_scalars_matches_published_json():
    # Выбор хранится в кэше как pickle, а публикуется как JSON: np.int64 фолдов превращается в int.
    # Сравнение по сырому obj_hash давало вечный промах кэша и пересчёт турнира L1 (06.10.2026).
    saved = {"validation_design": {"comparison_folds": [np.int64(3), np.int64(4)], "share": np.float64(.5)},
             "flag": np.bool_(True)}
    published = json.loads(json.dumps({"validation_design": {"comparison_folds": [3, 4], "share": .5},
                                       "flag": True}))
    assert stage_l1.same_selection(saved, published)
    assert not stage_l1.same_selection(saved, dict(published, flag=False))


def test_bootstrap_copies_of_a_match_share_the_opponent_strength_fold():
    # Копия матча (G~1) в бутстрэпе L6 — тот же матч: оказавшись в другом фолде, она училась бы
    # на идентичных бросках «близнеца». Точная копия получает ту же кодировку, что оригинал.
    s = _synthetic(3000)
    s["blocked"] = False
    s["opp_team_id"] = "T" + (s.match_id.str[1:].astype(int) % 7).astype(str)
    copy = s.assign(match_id=s.match_id + "~1")
    both = pd.concat([s, copy], ignore_index=True)
    enc = l1_xg.crossfit_strength(both, ["dist", "angle_vis", "mp"], seed=0)[0]
    np.testing.assert_allclose(enc[:len(s)], enc[len(s):])
    assert l1_xg.original_match(pd.Series(["G1", "G1~2", "G10~1"])).tolist() == ["G1", "G1", "G10"]


def test_outer_audit_xg_sum_gate_uses_match_bootstrap_interval():
    # Решение 06.10.2026: внешний аудит судит сумму xG тем же правилом, что L1 (P049), а не ±5%:
    # на ~175 матчах шум отношения ~3–3,5%, и жёсткий порог валил идеальную модель в 10–15% случаев.
    from scripts import audit_frozen_layers as audit
    mids, y, p = _calibrated_block(n_matches=175, shots_per_match=110, seed=3)
    shots = pd.DataFrame({"match_id": mids, "goal": y})
    ok, interval = audit.outer_xg_sum_gate(shots, p, B=300, seed=0)
    assert interval["status"] == "estimated" and ok
    biased_ok, _ = audit.outer_xg_sum_gate(shots, np.clip(p * 1.15, 1e-6, 1 - 1e-6), B=300, seed=0)
    assert not biased_ok


def test_lgbm_neutral_mixture_equals_the_per_level_loop():
    # Ускорение 06.10.2026: деревья без разбиений по opp_def считаются один раз, на каждом уровне силы
    # соперника пересчитываются только зависящие от неё. Результат — тот же, что у полного перебора.
    import yaml
    s = _synthetic(4000)
    s["blocked"] = False
    rng = np.random.default_rng(5)
    s["opp_def"] = rng.normal(0, .3, len(s))
    s["goal"] = ((rng.random(len(s)) < np.clip(0.05 + 0.08 * s.opp_def, 0.005, 0.5))).astype(int)
    params = dict(yaml.safe_load(open("configs/default.yaml"))["xg"]["lgbm"])
    m = l1_xg.LGBMxG(params, 0, ["dist", "angle_vis", "mp", "opp_def"]).fit(s)
    values = np.array([-0.4, -0.1, 0.0, 0.2, 0.5])
    weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    loop = np.zeros(len(s))
    for v, w in zip(values, weights):
        loop += w * m.predict(s.assign(opp_def=v))
    fast = m.neutral_mixture(s, values, weights)
    np.testing.assert_allclose(fast, loop, rtol=0, atol=1e-12)
    assert 0 < m.dynamic_tree_count() <= m.model.booster_.num_trees()


def test_splits_hash_is_stable_across_json_round_trip():
    # Интеграционная проверка 06.10.2026: int-ключи component_sizes_matches после JSON становились строками,
    # хэш свежего разбиения не совпадал с загруженным -> --force-all ломал цепочку (UpstreamNotReady).
    from datamatch.splits import splits_hash
    fresh = {"khl_component": 3, "component_sizes_matches": {"10": 5, "3": 7, "2": 1},
             "matches": {"match_id": ["A", "B"], "fold": [0, 1]}}
    as_int_keys = dict(fresh, component_sizes_matches={10: 5, 3: 7, 2: 1})
    assert splits_hash(as_int_keys) == splits_hash(json.loads(json.dumps(as_int_keys)))
    assert splits_hash(fresh) == splits_hash(as_int_keys)
