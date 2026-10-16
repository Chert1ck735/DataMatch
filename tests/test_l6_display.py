"""Правило отображения меток (решение пользователя 07.10.2026): TOI < 60 мин → LOW «мало минут».

Контракт: рейтинг и интервал не меняются; исходная метка модели остаётся в detail (`reliability`);
в CSV — метка отображения; шаг идемпотентен; meta хранит исходные sha256 L6 и обновлённые sha256
файлов (иначе повторный `ratings` счёл бы файл оборванной записью); модуль не входит в ключи
кэша L6 и реплик бутстрэпа.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from datamatch import l6_display as disp
from datamatch import l6_replicate, stage_l6
from datamatch.stage_l6 import file_sha256


def frames():
    csv = pd.DataFrame({"player_id": ["a", "b", "c", "d"], "rating": [0.1, -0.2, 0.3, 0.0],
                        "lo95": [-0.1, -0.5, 0.1, -0.3], "hi95": [0.3, 0.1, 0.5, 0.3],
                        "reliability": ["MEDIUM", "HIGH", "MEDIUM", "LOW"],
                        "toi_min": [7.0, 59.9, 60.0, 0.0], "games": [2, 5, 6, 0], "role": ["C", "D", "W", "C"]})
    detail = csv[["player_id", "rating", "lo95", "hi95", "reliability"]].assign(flag="", w=[0.44, 0.7, 0.5, 0.0])
    return csv, detail


def test_rule_demotes_low_minutes_only():
    csv, detail = frames()
    c2, d2, info = disp.apply_low_minutes_rule(csv, detail, disp.MIN_TOI_MIN)
    assert c2.reliability.tolist() == ["LOW", "LOW", "MEDIUM", "LOW"]
    assert d2.reliability.tolist() == ["MEDIUM", "HIGH", "MEDIUM", "LOW"]          # исходная метка
    assert d2.reliability_display.tolist() == ["LOW", "LOW", "MEDIUM", "LOW"]
    assert d2.display_flag.tolist() == [disp.LOW_MINUTES_FLAG, disp.LOW_MINUTES_FLAG, "", ""]
    pd.testing.assert_frame_equal(c2[["rating", "lo95", "hi95", "toi_min"]], csv[["rating", "lo95", "hi95", "toi_min"]])
    assert info["n_demoted"] == 2


def test_rule_is_idempotent():
    csv, detail = frames()
    c2, d2, _ = disp.apply_low_minutes_rule(csv, detail, disp.MIN_TOI_MIN)
    c3, d3, info = disp.apply_low_minutes_rule(c2, d2, disp.MIN_TOI_MIN)
    pd.testing.assert_frame_equal(c2, c3)
    pd.testing.assert_frame_equal(d2, d3)
    assert info["n_demoted"] == 2


def test_files_and_meta_sha_updated(tmp_path):
    csv, detail = frames()
    p_csv, p_det, p_meta = tmp_path / "r.csv", tmp_path / "r_detail.csv", tmp_path / "r_meta.json"
    csv.to_csv(p_csv, index=False)
    detail.to_csv(p_det, index=False)
    l6_sha = file_sha256(p_csv)
    p_meta.write_text(json.dumps({"csv_sha256": l6_sha, "detail_sha256": file_sha256(p_det),
                                  "reliability": {"counts": {"LOW": 1, "MEDIUM": 2, "HIGH": 1}}}))
    disp.apply_to_files(p_csv, p_det, p_meta)
    meta = json.loads(p_meta.read_text())
    assert meta["csv_sha256"] == file_sha256(p_csv) != l6_sha
    assert meta["detail_sha256"] == file_sha256(p_det)
    assert meta["display_rule"]["l6_csv_sha256"] == l6_sha
    assert meta["reliability"]["counts_display"] == {"LOW": 3, "MEDIUM": 1}
    disp.apply_to_files(p_csv, p_det, p_meta)          # повтор: ничего не меняется, исходный sha L6 сохранён
    meta2 = json.loads(p_meta.read_text())
    assert meta2["display_rule"]["l6_csv_sha256"] == l6_sha
    assert meta2["csv_sha256"] == meta["csv_sha256"]


def test_module_outside_l6_cache_keys():
    assert "l6_display" not in l6_replicate.replicate_code()
    assert "l6_display" not in stage_l6.L6_CODE


def test_missing_meta_sha_fails(tmp_path):
    csv, detail = frames()
    p_csv, p_det, p_meta = tmp_path / "r.csv", tmp_path / "r_detail.csv", tmp_path / "r_meta.json"
    csv.to_csv(p_csv, index=False)
    detail.to_csv(p_det, index=False)
    p_meta.write_text(json.dumps({"csv_sha256": "deadbeef", "detail_sha256": "x"}))
    with pytest.raises(disp.DisplayError):
        disp.apply_to_files(p_csv, p_det, p_meta)


def test_missing_toi_is_an_error_not_silent_keep():
    """Пустой или нечисловой TOI — ошибка (раньше NaN < 60 давало False, и метка молча не понижалась)."""
    csv, detail = frames()
    csv = csv.astype({"toi_min": object})
    csv.loc[0, "toi_min"] = ""
    with pytest.raises(disp.DisplayError):
        disp.apply_low_minutes_rule(csv, detail, disp.MIN_TOI_MIN)


@pytest.mark.parametrize("toi, expected", [(59.99, "LOW"), (60.0, "MEDIUM"), (60.01, "MEDIUM")])
def test_rule_boundary_at_sixty_minutes(toi, expected):
    """A3(г), аудит 08.10.2026: граница строгая — LOW только при toi_min < 60."""
    csv, detail = frames()
    csv = csv.assign(toi_min=[toi, 100.0, 100.0, 100.0])
    out, _, _ = disp.apply_low_minutes_rule(csv, detail, disp.MIN_TOI_MIN)
    assert out.reliability.iloc[0] == expected


def test_rule_nan_toi_is_error():
    csv, detail = frames()
    csv = csv.assign(toi_min=[float("nan"), 100.0, 100.0, 100.0])
    with pytest.raises(disp.DisplayError):
        disp.apply_low_minutes_rule(csv, detail, disp.MIN_TOI_MIN)


def test_display_status_marks_fresh_l6_output_as_not_applied(tmp_path):
    """A5, аудит 08.10.2026: свежий выход L6 (meta без display_applied) — правило не применено;
    после apply — применено; повторный ratings (новая meta) снова даёт «не применено»."""
    csv, detail = frames()
    p_csv, p_det, p_meta = tmp_path / "r.csv", tmp_path / "r_detail.csv", tmp_path / "r_meta.json"
    csv.to_csv(p_csv, index=False)
    detail.to_csv(p_det, index=False)
    fresh = {"csv_sha256": file_sha256(p_csv), "detail_sha256": file_sha256(p_det)}
    p_meta.write_text(json.dumps(fresh))
    st = disp.display_status(p_csv, p_meta)
    assert not st["applied"]
    disp.apply_to_files(p_csv, p_det, p_meta)
    assert disp.display_status(p_csv, p_meta)["applied"]
    assert json.loads(p_meta.read_text())["display_applied"] is True
    csv.to_csv(p_csv, index=False)                       # «повторный ratings» переписал файлы и meta
    p_meta.write_text(json.dumps({"csv_sha256": file_sha256(p_csv), "detail_sha256": fresh["detail_sha256"]}))
    assert not disp.display_status(p_csv, p_meta)["applied"]
