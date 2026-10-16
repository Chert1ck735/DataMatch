"""Exercise real L0 cache status transitions via CLI, using an isolated cache copy."""
import gc
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from datamatch import cli, io, stage_l0
from datamatch.cache import Cache
from datamatch.config import load_config


def verify():
    cfg = load_config()
    data = io.data_version(cfg)
    original = Cache(cfg, data)
    key = original.key("l0", ("l0",), code=stage_l0.L0_CODE)
    if not original.done(key):
        raise RuntimeError("Current L0 calculation has not completed")
    results = []
    with TemporaryDirectory(prefix="datamatch-l0-status-") as directory:
        base = Path(directory)
        isolated = cfg.with_overrides({"paths": {"cache": str(base / "cache"),
            "outputs": str(base / "outputs"), "artifacts": str(base / "artifacts")}})
        destination = isolated.paths.cache / key
        destination.mkdir()
        for file in (original.root / key).iterdir():
            if file.suffix == ".pkl":
                shutil.copy2(file, destination / file.name)  # mutable objects MUST NOT share an inode
            else:
                os.link(file, destination / file.name)  # tables are read-only in this check
        cache = Cache(isolated, data)
        stress = isolated.with_overrides({"data": {"train_schemes": ["v1", "v2"], "stress_mode": True}})
        assert cache.load_obj(key, "quality")["status"] == "ok"
        res, _ = cli.cmd_prepare(stress, Cache(stress, data), False)
        report = json.loads((stress.paths.outputs / "data_quality.json").read_text())
        assert report["status"] == report["scheme_gate"]["status"] == "stress"
        results.append({"from": "ok", "to": report["status"], "json_matches_scheme_gate": True})
        # Save the admitted stress result only in the isolated cache, then reload it.
        cache.save_obj(key, "quality", res["quality"])
        del res
        gc.collect()
        assert cache.load_obj(key, "quality")["status"] == "stress"
        normal = isolated.with_overrides({"data": {"train_schemes": ["v1"], "stress_mode": False}})
        res, _ = cli.cmd_prepare(normal, Cache(normal, data), False)
        report = json.loads((normal.paths.outputs / "data_quality.json").read_text())
        assert report["status"] == report["scheme_gate"]["status"] == "ok"
        results.append({"from": "stress", "to": report["status"], "json_matches_scheme_gate": True})
        del res
    assert original.load_obj(key, "quality")["status"] == "ok"
    output = {"cache_key": key, "transitions": results, "original_cache_unchanged": True}
    (cfg.paths.outputs / "l0_cache_status_verification.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    verify()
