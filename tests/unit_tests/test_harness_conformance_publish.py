# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish measured failures while rejecting stale, partial, or contradictory evidence."""

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from scripts.harness_conformance import publish


SHA = "a" * 40
RUN = {"id": 12, "run_attempt": 2, "head_sha": SHA, "updated_at": "2026-10-12T08:34:00Z"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "repo"
    page = root / publish.PAGE
    page.parent.mkdir(parents=True)
    page.write_text(f"Introduction\n{publish.BEGIN}\nold measurement\n{publish.END}\nATIF reference\n")
    artifacts = tmp_path / "artifacts"
    for harness in publish.HARNESSES:
        probes = artifacts / harness / "probes"
        write_json(probes / "ci-result.json", {"harness": harness, "sha": SHA, "status": "unavailable"})
    probes = artifacts / "pi" / "probes"
    write_json(probes / "ci-result.json", {"harness": "pi", "sha": SHA, "status": "failed"})
    manifest = publish.SCENARIOS
    write_json(
        probes / "suite.json",
        {
            "version": publish.SUITE,
            "harnesses": ["pi"],
            "scenarios": [{"name": s.name} for s in manifest],
        },
    )
    scenarios = []
    for scenario in manifest:
        runtime = probes / "pi" / scenario.name / "runtime.json"
        write_json(runtime, {"version": "test-runtime", "adapter_sha256": "adapter-source"})
        scenarios.append(
            {
                "scenario": scenario.name,
                "verdict": "not_fulfilled",
                "checks": [{"id": "calls.outcome", "kind": "schema", "tier": "P0", "status": "fail"}],
                "hashes": {"runtime.json": publish.digest(runtime)},
            }
        )
    write_json(
        probes / "conformance_summary.json",
        {
            "schema_version": "harness-probe-report/v1",
            "runner_status": "completed",
            "full_suite": True,
            "suite": publish.SUITE,
            "suite_sha256": publish.digest(probes / "suite.json"),
            "limits": [],
            "harnesses": {"pi": {"verdict": "not_fulfilled", "scenarios": scenarios}},
        },
    )
    # Retained Markdown cannot override individual statuses in the measured JSON.
    (probes / "conformance_report.md").write_text("invented pass\n")
    (artifacts / "pi" / "tests.xml").write_text(
        "<testsuites><testsuite><testcase/><testcase><skipped/></testcase></testsuite></testsuites>"
    )
    return root, artifacts, probes, page


def test_failed_and_unavailable_results_publish_with_provenance(evidence):
    root, artifacts, _, page = evidence
    publish.publish(root=root, artifacts=artifacts, run=RUN, repo="NVIDIA-NeMo/Gym")
    text = page.read_text()
    assert text.startswith("Introduction\n") and text.endswith("\nATIF reference\n")
    assert "`pi` | FAILED | 0/7" in text
    assert "`hermes` | Unavailable | —" in text
    assert "`calls.outcome` | ✗" in text and "invented pass" not in text
    assert "1 passed, 1 skipped" in text and "runs/12/attempts/2" in text
    (asset,) = (root / publish.ASSETS).glob("*.json")
    report = json.loads(asset.read_text())
    assert asset.name in text and report["gym_commit"] == SHA
    assert report["harnesses"]["pi"]["scenarios"][0]["checks"][0]["status"] == "fail"
    assert report["harnesses"]["pi"]["runtime"]["version"] == "test-runtime"
    # A repeat publication of identical evidence has one content-addressed asset.
    publish.publish(root=root, artifacts=artifacts, run=RUN, repo="NVIDIA-NeMo/Gym")
    assert page.read_text() == text and len(list((root / publish.ASSETS).glob("*.json"))) == 1


@pytest.mark.parametrize("problem", ["missing", "sha", "partial", "truncated", "hash", "runtime", "verdict", "tests"])
def test_invalid_evidence_preserves_previous_publication(evidence, problem):
    root, artifacts, probes, page = evidence
    original = page.read_bytes()
    ci = json.loads((probes / "ci-result.json").read_text())
    summary = json.loads((probes / "conformance_summary.json").read_text())
    if problem == "missing":
        (artifacts / "hermes" / "probes" / "ci-result.json").unlink()
    elif problem == "sha":
        ci["sha"] = "b" * 40
    elif problem == "partial":
        summary["full_suite"] = False
    elif problem == "truncated":
        summary["harnesses"]["pi"]["scenarios"].pop()
    elif problem == "hash":
        summary["suite_sha256"] = "wrong"
    elif problem == "runtime":
        (probes / "pi" / publish.SCENARIOS[0].name / "runtime.json").write_text("{}")
    elif problem == "verdict":
        ci["status"] = "passed"
    else:
        (artifacts / "pi" / "tests.xml").write_text(
            "<testsuites><testsuite><testcase><failure/></testcase></testsuite></testsuites>"
        )
    write_json(probes / "ci-result.json", ci)
    write_json(probes / "conformance_summary.json", summary)
    with pytest.raises((OSError, ValueError)):
        publish.publish(root=root, artifacts=artifacts, run=RUN, repo="NVIDIA-NeMo/Gym")
    assert page.read_bytes() == original and not (root / publish.ASSETS).exists()


def test_legacy_artifacts_report_absent_counts_explicitly(evidence):
    root, artifacts, _, page = evidence
    (artifacts / "pi" / "tests.xml").unlink()
    publish.publish(root=root, artifacts=artifacts, run=RUN, repo="NVIDIA-NeMo/Gym")
    assert "counts not retained by older CI" in page.read_text()


def test_latest_main_run_includes_failed_measurements_but_excludes_prs_and_forks(monkeypatch):
    trusted = {
        **RUN,
        "status": "completed",
        "conclusion": "failure",
        "head_branch": "main",
        "event": "schedule",
        "head_repository": {"full_name": "NVIDIA-NeMo/Gym"},
        "run_started_at": "2026-10-12T08:00:00Z",
    }
    newer = {**trusted, "id": 14, "run_started_at": "2026-10-12T09:00:00Z"}
    forks = {**newer, "head_repository": {"full_name": "elsewhere/Gym"}}
    prs = {**newer, "head_branch": "pull-request/123"}
    running = {**newer, "status": "in_progress"}
    monkeypatch.setattr(
        publish.subprocess,
        "check_output",
        lambda *a, **kw: json.dumps(
            {
                "workflow_runs": [trusted, forks, prs, running],
            }
        ),
    )
    assert publish.latest_run("NVIDIA-NeMo/Gym")["id"] == 12


def test_download_uses_exact_attempt_names_and_fails_on_missing_artifacts(tmp_path, monkeypatch):
    calls = []

    def download(command, *, check):
        calls.append(command)
        assert check and "--name" in command
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(publish.subprocess, "run", download)
    with pytest.raises(subprocess.CalledProcessError):
        publish.download(RUN, repo="NVIDIA-NeMo/Gym", output=tmp_path / "artifacts")
    assert "harness-conformance-opencode-12-2" in calls[0]


def test_fern_publish_reuses_main_ci_and_keeps_probe_permissions_read_only():
    root = Path(__file__).resolve().parents[2]
    text = (root / ".github/workflows/publish-fern-docs.yml").read_text()
    workflow = yaml.safe_load(text)
    jobs = workflow["jobs"]
    assert "workflow_run:" in text and "types: [completed]" in text and "branches: [main]" in text
    assert "head_repository.full_name == github.repository" in jobs["publish"]["if"]
    assert "schedule" in jobs["publish"]["if"] and "workflow_dispatch" in jobs["publish"]["if"]
    assert "conclusion" not in jobs["publish"]["if"]
    # Ineligible workflow_run events must not cancel an active publication.
    assert "concurrency" not in workflow and jobs["publish"]["concurrency"]["group"] == "fern-publish"
    steps = jobs["publish"]["steps"]
    overlay = next(i for i, step in enumerate(steps) if "scripts.harness_conformance.publish" in step.get("run", ""))
    deploy = next(i for i, step in enumerate(steps) if step.get("name") == "Publish Docs")
    assert overlay < deploy
    probes = yaml.safe_load((root / ".github/workflows/harness-conformance.yml").read_text())
    assert probes["permissions"] == {"contents": "read"}
    assert "--junitxml" in str(probes["jobs"]["probe"]["steps"])
