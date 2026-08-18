"""Report assembly, serialization and the command-line entry point."""

import json
import pathlib

import numpy as np
import pytest

import authoring
import factories
from sim_env_builder.asset_checks import cli, config, report
from sim_env_builder.asset_checks.checks import registry
from sim_env_builder.asset_checks.render import turntable


def _results():
    return [
        registry.CheckResult("mesh.watertight", registry.PASS, {"open_components": 0}),
        registry.CheckResult(
            "mesh.ngons", registry.FAIL, {"ngons": np.int64(3)}, "three n-gons"
        ),
        registry.CheckResult("sweep.rigid_invariance", registry.NOT_APPLICABLE, {}),
        registry.CheckResult(
            "materials.baked_lighting",
            registry.FAIL,
            {"shell_gradient": np.float64(0.7)},
            "ramped",
        ),
    ]


def test_report_structure_and_section_counts():
    asset = factories.asset([factories.part()])
    document = report.build(asset, _results(), {"status": "skipped"})

    assert document["schema_version"] == report.SCHEMA_VERSION
    assert document["asset"]["format"] == "usd"
    assert document["asset"]["triangles"] == 12
    assert document["sections"]["mesh"] == {
        "pass": 1,
        "fail": 1,
        "not_applicable": 0,
        "info": 0,
    }
    assert document["sections"]["sweep"]["not_applicable"] == 1


def test_messages_appear_only_where_meaningful():
    entries = {
        entry["id"]: entry
        for entry in report.build(
            factories.asset([factories.part()]), _results(), {}
        )["checks"]
    }
    assert "message" not in entries["mesh.watertight"]
    assert entries["mesh.ngons"]["message"] == "three n-gons"


def test_numpy_metrics_serialize(tmp_path):
    document = report.build(
        factories.asset([factories.part()]), _results(), {"status": "skipped"}
    )
    path = tmp_path / "report.json"
    report.write(path, document)

    reloaded = json.loads(path.read_text())
    assert reloaded["checks"][1]["metrics"]["ngons"] == 3
    assert isinstance(reloaded["checks"][3]["metrics"]["shell_gradient"], float)


def test_unserializable_metric_is_rejected(tmp_path):
    document = report.build(
        factories.asset([factories.part()]),
        [registry.CheckResult("mesh.ngons", registry.PASS, {"bad": object()})],
        {},
    )
    with pytest.raises(TypeError):
        report.write(tmp_path / "report.json", document)


def test_check_order_follows_registration():
    asset = factories.asset([factories.part()], [factories.proxy()])
    results = registry.run_all(factories.context(asset))
    document = report.build(asset, results, {})
    assert [entry["id"] for entry in document["checks"]] == [
        entry.check_id for entry in registry.ordered()
    ]


def test_summarize_lists_every_section():
    document = report.build(
        factories.asset([factories.part()]), _results(), {"status": "skipped"}
    )
    text = report.summarize(document)
    assert "mesh" in text and "sweep" in text and "materials" in text


def test_thresholds_override_partially(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"poly_budget": 7}))
    loaded = config.load(path)
    assert loaded.poly_budget == 7
    assert loaded.uv_shells_per_part == config.Thresholds().uv_shells_per_part


def test_unknown_threshold_key_is_rejected(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"nonexistent": 1}))
    with pytest.raises(ValueError, match="unknown threshold keys"):
        config.load(path)


def test_cli_writes_a_report(tmp_path):
    asset = authoring.hinged_stage(tmp_path / "hinge.usda")
    out = tmp_path / "out"
    assert cli.main([asset, "--out", str(out)]) == 0

    document = json.loads((out / "report.json").read_text())
    assert document["asset"]["articulated"] is True
    assert document["render"] == {"status": "skipped"}
    assert len(document["checks"]) == len(registry.REGISTRY)


def test_cli_reports_the_reversed_limit_as_a_failure(tmp_path):
    asset = authoring.hinged_stage(tmp_path / "hinge.usda")
    out = tmp_path / "out"
    cli.main([asset, "--out", str(out)])

    document = json.loads((out / "report.json").read_text())
    statuses = {entry["id"]: entry["status"] for entry in document["checks"]}
    assert statuses["articulation.limits_ordered"] == registry.FAIL


def test_render_command_is_well_formed(tmp_path):
    command = turntable.command("blender", "asset.usdz", tmp_path, "preview")
    assert command[:4] == ["blender", "-b", "--factory-startup", "--python"]
    assert command[4].endswith("studio_loop.py")
    assert command[5] == "--"
    assert "--input" in command and "asset.usdz" in command
    assert command[command.index("--mode") + 1] == "preview"


def test_render_skipped_and_missing_binary_do_not_raise(tmp_path, monkeypatch):
    assert turntable.render("a.usdz", tmp_path, "off", None) == {"status": "skipped"}

    monkeypatch.setattr(turntable, "find_blender", lambda explicit: None)
    result = turntable.render("a.usdz", tmp_path, "preview", None)
    assert result["status"] == "not_run"


def test_render_script_is_packaged():
    assert pathlib.Path(turntable.SCRIPT).is_file()


def test_render_survives_a_nonexistent_blender_binary(tmp_path):
    result = turntable.render("a.usdz", tmp_path, "preview", "/no/such/blender")
    assert result["status"] == "failed"


def test_render_survives_a_timeout(tmp_path, monkeypatch):
    import subprocess

    def explode(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="blender", timeout=1)

    monkeypatch.setattr(turntable, "find_blender", lambda explicit: "blender")
    monkeypatch.setattr(turntable.subprocess, "run", explode)
    result = turntable.render("a.usdz", tmp_path, "preview", None)
    assert result["status"] == "failed"


def test_render_survives_a_malformed_manifest(tmp_path, monkeypatch):
    class Done:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(turntable, "find_blender", lambda explicit: "blender")
    monkeypatch.setattr(turntable.subprocess, "run", lambda *a, **k: Done())
    (tmp_path / "manifest.json").write_text("{not json")
    result = turntable.render("a.usdz", tmp_path, "preview", None)
    assert result["status"] == "failed"
    assert "manifest" in result["reason"]
