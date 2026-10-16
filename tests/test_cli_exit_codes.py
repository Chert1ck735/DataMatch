"""CLI: понятные коды выхода вместо трассировки (ревью 06.10.2026)."""
import pytest

from datamatch import cli, l6_replicate, stage_l3, stage_l5, stage_l6


@pytest.mark.parametrize("error", [stage_l3.UpstreamNotReady, stage_l5.FrozenStateMissing,
                                   stage_l6.L6InputsNotReady, l6_replicate.ReplicateUnavailable])
def test_missing_upstream_exits_with_code_4_not_a_traceback(monkeypatch, tmp_path, error):
    def not_ready(*args, **kwargs):
        raise error("нет готового входа")

    monkeypatch.setitem(cli.COMMANDS, "contrib", not_ready)
    # Свой кэш и журналы: блокировка настоящего кэша занята идущим расчётом.
    from datamatch.config import load_config
    cfg = load_config().with_overrides({"paths": {"cache": str(tmp_path / "cache"),
                                                  "logs": str(tmp_path / "logs")}})
    monkeypatch.setattr(cli, "load_config", lambda *_: cfg)
    assert cli.main(["contrib"]) == cli.EXIT_UPSTREAM_NOT_READY == 4
