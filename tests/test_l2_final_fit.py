"""L2: финальная процедура переобучается на всём допустимом train (решение 06.10.2026).

Спецификация L7, правило holdout: после выбора на разработке «процедура переобучается на всём
train и применяется к test». Модель разработки сохраняется для замороженного внешнего аудита.
"""
import numpy as np
import pandas as pd

from datamatch import stage_l2
from datamatch.cache import Cache
from datamatch.config import load_config
from datamatch.registry import Registry
from tests.test_l2 import many_nodes


class FixedXG:
    spec = {"family": "logit_spline", "features": ["dist", "angle_vis"]}

    def predict(self, shots):
        return np.full(len(shots), .2)


def _splits(match_ids):
    ids = sorted(match_ids)
    n = len(ids)
    outer = np.arange(n) >= n - 3                     # последние три матча — внешний блок
    fold = np.where(outer, -1, np.arange(n) % 5)
    scheme = np.where(np.arange(n) == 0, "v2", "v1")  # матч другой схемы в подгонку не входит
    return {"matches": pd.DataFrame({"match_id": ids, "fold": fold, "outer": outer,
                                     "scheme": scheme}).to_dict(orient="list")}


def test_final_fit_matches_are_dev_and_outer_of_allowed_schemes():
    sp = _splits([f"M{i:02d}" for i in range(15)])
    ids = stage_l2.final_fit_match_ids(sp, ("v1",))
    assert ids == [f"M{i:02d}" for i in range(1, 15)]


def test_final_fit_folds_keep_dev_folds_and_give_outer_its_own_fold():
    sp = _splits([f"M{i:02d}" for i in range(15)])
    nodes = pd.DataFrame({"match_id": ["M01", "M01", "M05", "M13", "M14"]})
    folds = stage_l2.final_fit_folds(nodes, sp)
    assert folds.tolist() == [1, 1, 0, 5, 5]


def test_run_l2_refits_final_procedure_on_all_allowed_train(tmp_path, monkeypatch):
    cfg = load_config().with_overrides({"paths": {"artifacts": str(tmp_path / "artifacts"),
        "cache": str(tmp_path / "cache")}, "value": {"markov": {"x_edges": [-31, 31], "y_edges": [-16, 16]}}})
    nd = many_nodes(18)
    sp = _splits(nd.match_id.unique())
    fitted = []

    def fake_fit(self, train, r, **kwargs):
        self.train_matches_ = set(train.match_id)
        fitted.append(self.train_matches_)
        return self

    monkeypatch.setattr(stage_l2, "fit_xg", lambda *args: FixedXG())
    monkeypatch.setattr(stage_l2.l2_procedure.ValueProcedure, "fit", fake_fit)
    monkeypatch.setattr(stage_l2.l2_procedure.ValueProcedure, "value",
                        lambda self, validation, neutral=False: np.zeros(len(validation)))
    monkeypatch.setattr(stage_l2, "enforce", lambda *args, **kwargs: None)   # проверяем шаг после гейта
    cache = Cache(cfg)
    sel = {"status": "ok", "spec": FixedXG.spec}
    l0 = {"quality": {"status": "ok"}, "train": {"nodes": nd}, "test": {"nodes": nd}}
    result = stage_l2.run_l2(cfg, l0, sp, sel, Registry(cfg), cache, force=True)

    all_train = set(stage_l2.final_fit_match_ids(sp, ("v1",)))
    dev = set(sp["matches"]["match_id"][i] for i in range(18)
              if not sp["matches"]["outer"][i] and sp["matches"]["scheme"][i] == "v1")
    key = result["cache_key"]
    assert cache.load_obj(key, "procedure_full").train_matches_ == all_train
    assert cache.load_obj(key, "procedure").train_matches_ == dev          # для внешнего аудита
    train_values = cache.load_df(key, "train_values")
    assert set(train_values.match_id) == all_train
    assert cache.load_df(key, "test_values").estimate_kind.eq("full_train_fit").all()
    assert result["final_fit"]["matches"] == len(all_train)
    assert result["final_fit"]["outer_block_in_final_fit"] is True


def test_previous_l2_generation_stays_done_during_long_final_fits(tmp_path, monkeypatch):
    # Ревью 06.10.2026: DONE снимался до долгих итоговых подгонок; выключение машины в этом окне
    # оставляло L2 без пригодной генерации. Снимается только перед первой записью.
    from datamatch.cache import obj_hash
    cfg = load_config().with_overrides({"paths": {"artifacts": str(tmp_path / "artifacts"),
        "cache": str(tmp_path / "cache")}, "value": {"markov": {"x_edges": [-31, 31], "y_edges": [-16, 16]}}})
    nd = many_nodes(18)
    sp = _splits(nd.match_id.unique())
    cache = Cache(cfg)
    sel = {"status": "ok", "spec": FixedXG.spec}
    schemes = tuple(cfg["data"]["train_schemes"])
    dev = stage_l2.development_nodes(nd, sp, schemes)
    full = stage_l2.final_fit_nodes(nd, sp, schemes)
    prov = stage_l2.provenance(cfg, cache, dev, sp, sel, nd, full)
    key = cache.key("l2_dev_model", ("value", "data", "eval", "seed"), extra=obj_hash(prov), code=stage_l2.L2_CODE)
    cache.mark_done(key)
    done_during_fits = []

    def fake_fit(self, train, r, **kwargs):
        done_during_fits.append(cache.done(key))
        return self

    monkeypatch.setattr(stage_l2, "fit_xg", lambda *args: FixedXG())
    monkeypatch.setattr(stage_l2.l2_procedure.ValueProcedure, "fit", fake_fit)
    monkeypatch.setattr(stage_l2.l2_procedure.ValueProcedure, "value",
                        lambda self, validation, neutral=False: np.zeros(len(validation)))
    monkeypatch.setattr(stage_l2, "enforce", lambda *args, **kwargs: None)
    l0 = {"quality": {"status": "ok"}, "train": {"nodes": nd}, "test": {"nodes": nd}}
    stage_l2.run_l2(cfg, l0, sp, sel, Registry(cfg), cache, force=True)
    assert done_during_fits and all(done_during_fits)
    assert cache.done(key)
