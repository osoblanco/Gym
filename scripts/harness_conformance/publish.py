# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Overlay main documentation with the latest scheduled/manual main CI measurements.

Uses Python's standard library and gh; no Gym installation or probe execution.
The retained check statuses supply the individual check matrices. A compact,
content-addressed report keeps their provenance after Actions artifacts expire.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from nemo_gym.harness_reporting import render_matrices

from .registry import HARNESSES
from .scenarios import SCENARIOS, SUITE


ROOT = Path(__file__).resolve().parents[2]
PAGE = Path("fern/versions/latest/pages/reference/trajectory-capabilities.mdx")
ASSETS = Path("fern/assets/trajectory-capabilities")
BEGIN = "{/* BEGIN GENERATED TE TABLE */}"
END = "{/* END GENERATED TE TABLE */}"


def digest(path: Path) -> str:
    """Hash retained input bytes, rather than their reserialized representation."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_counts(path: Path) -> dict[str, int] | None:
    """Read JUnit gates; older CI artifacts have no structured test counts."""
    if not path.exists():
        return None
    cases = ET.parse(path).findall(".//testcase")
    skipped = sum(case.find("skipped") is not None for case in cases)
    if (
        not cases
        or skipped == len(cases)
        or any(case.find("failure") is not None or case.find("error") is not None for case in cases)
    ):
        raise ValueError(f"validation did not establish a passing test suite: {path}")
    return {"passed": len(cases) - skipped, "skipped": skipped}


def latest_run(repo: str) -> dict:
    """Select a completed trusted main measurement independently of its conclusion."""
    runs = []
    for event in ("schedule", "workflow_dispatch"):
        response = subprocess.check_output(
            ["gh", "api", f"repos/{repo}/actions/workflows/cicd-main.yml/runs?event={event}&per_page=20"],
            text=True,
        )
        runs.extend(json.loads(response)["workflow_runs"])
    candidates = [
        run
        for run in runs
        if run["status"] == "completed"
        and run["head_branch"] == "main"
        and run["event"] in ("schedule", "workflow_dispatch")
        and run["head_repository"]["full_name"] == repo
    ]
    if not candidates:
        raise ValueError("no completed scheduled/manual main conformance run is available")
    return max(candidates, key=lambda run: (run["run_started_at"], run["id"]))


def download(run: dict, *, repo: str, output: Path) -> None:
    """Download exact artifacts from one attempt; partial reruns cannot reuse old evidence."""
    output.mkdir(parents=True, exist_ok=False)
    for harness in HARNESSES:
        subprocess.run(
            [
                "gh",
                "run",
                "download",
                str(run["id"]),
                "--repo",
                repo,
                "--name",
                f"harness-conformance-{harness}-{run['id']}-{run['run_attempt']}",
                "--dir",
                str(output / harness),
            ],
            check=True,
        )


def measurement(harness: str, *, directory: Path, sha: str) -> tuple[dict, str]:
    """Validate one CI result and retain portable checks, runtime pins, and input hashes."""
    probes = directory / "probes"
    ci_path = probes / "ci-result.json"
    ci = json.loads(ci_path.read_text())
    if (
        ci.get("harness") != harness
        or ci.get("sha") != sha
        or ci.get("status") not in ("passed", "failed", "unavailable")
    ):
        raise ValueError(f"{harness}: missing/invalid result or evidence from another commit")
    result = {"status": ci["status"], "ci_result_sha256": digest(ci_path)}
    if ci["status"] == "unavailable":
        return result, ""
    counts = test_counts(directory / "tests.xml")
    summary_path = probes / "conformance_summary.json"
    summary = json.loads(summary_path.read_text())
    manifest_path = probes / "suite.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        summary["schema_version"] != "harness-probe-report/v1"
        or summary["runner_status"] != "completed"
        or summary["full_suite"] is not True
        or set(summary["harnesses"]) != {harness}
        or summary["suite_sha256"] != digest(manifest_path)
        or summary["suite"] != manifest["version"]
        or summary["suite"] != SUITE
        or manifest["harnesses"] != [harness]
    ):
        raise ValueError(f"{harness}: incomplete or inconsistent probe suite")
    row = summary["harnesses"][harness]
    names = [scenario["name"] for scenario in manifest["scenarios"]]
    if names != [s.name for s in SCENARIOS] or [s["scenario"] for s in row["scenarios"]] != names:
        raise ValueError(f"{harness}: scenario coverage differs from the suite manifest")
    scenarios = []
    runtimes = []
    for scenario in row["scenarios"]:
        name = scenario["scenario"]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"{harness}: invalid scenario name")
        checks = scenario["checks"]
        if (
            not checks
            or len({check["id"] for check in checks}) != len(checks)
            or any(
                check["status"] not in ("pass", "fail", "not_applicable")
                or check["kind"] not in ("schema", "semantic", "behavioral", "health")
                or check["tier"] not in ("P0", "P1", "P2")
                for check in checks
            )
        ):
            raise ValueError(f"{harness}/{name}: invalid or empty check results")
        p0 = [check for check in checks if check["tier"] == "P0" and check["status"] != "not_applicable"]
        passed = bool(p0) and all(check["status"] == "pass" for check in p0)
        expected = "fulfilled" if passed else "not_fulfilled"
        if scenario["verdict"] != expected:
            raise ValueError(f"{harness}/{name}: verdict disagrees with individual P0 checks")
        runtime_path = probes / harness / name / "runtime.json"
        if scenario["hashes"].get("runtime.json") != digest(runtime_path):
            raise ValueError(f"{harness}/{name}: runtime hash mismatch")
        runtime = json.loads(runtime_path.read_text())
        runtime = {key: value for key, value in runtime.items() if key not in ("source", "executable")}
        if not (runtime.get("version") or runtime.get("sha256")) or not runtime.get("adapter_sha256"):
            raise ValueError(f"{harness}/{name}: runtime identity is unavailable")
        runtimes.append(runtime)
        scenarios.append(
            {
                "scenario": name,
                "verdict": expected,
                "checks": [{key: check[key] for key in ("id", "kind", "tier", "status")} for check in checks],
            }
        )
    if any(runtime != runtimes[0] for runtime in runtimes):
        raise ValueError(f"{harness}: runtime changed between scenarios")
    passed = all(scenario["verdict"] == "fulfilled" for scenario in scenarios)
    if ci["status"] != ("passed" if passed else "failed") or row["verdict"] != (
        "fulfilled" if passed else "not_fulfilled"
    ):
        raise ValueError(f"{harness}: CI status disagrees with individual P0 checks")
    report_path = probes / "conformance_report.md"
    result.update(
        suite=summary["suite"],
        scenarios=scenarios,
        runtime=runtimes[0],
        tests=counts,
        limits=summary["limits"],
        summary_sha256=digest(summary_path),
        suite_sha256=digest(manifest_path),
        markdown_sha256=digest(report_path),
        tests_sha256=digest(directory / "tests.xml") if counts else None,
    )
    # Render retained statuses without reevaluating old evidence with a newer checker.
    markdown = render_matrices({harness: row}).replace("# Harness checks by scenario\n", "", 1)
    markdown = markdown.replace(f"## {harness}\n", f"### `{harness}`\n", 1)
    return result, markdown


def atomic_write(path: Path, content: str) -> None:
    """Replace one output only after its complete contents are available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def publish(*, root: Path, artifacts: Path, run: dict, repo: str) -> None:
    """Update only the main page's generated block after validating all four artifacts."""
    sha = run["head_sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("invalid source commit or repository")
    if (
        not isinstance(run["id"], int)
        or run["id"] <= 0
        or not isinstance(run["run_attempt"], int)
        or run["run_attempt"] <= 0
    ):
        raise ValueError("invalid CI run identity")
    measured = datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00"))
    if measured.tzinfo is None:
        raise ValueError("CI measurement time must include its timezone")
    url = f"https://github.com/{repo}/actions/runs/{run['id']}/attempts/{run['run_attempt']}"
    rows, matrices = {}, []
    for harness in HARNESSES:
        row, markdown = measurement(harness, directory=artifacts / harness, sha=sha)
        rows[harness] = row
        if markdown:
            matrices.append(markdown)
    suites = {row["suite"] for row in rows.values() if row["status"] != "unavailable"}
    scenarios = {
        tuple(s["scenario"] for s in row["scenarios"]) for row in rows.values() if row["status"] != "unavailable"
    }
    if len(suites) > 1 or len(scenarios) > 1:
        raise ValueError("harnesses were measured with different suites/scenarios")
    portable = {
        "schema_version": "trajectory-capabilities-ci/v1",
        "gym_commit": sha,
        "measured_at": measured.isoformat(),
        "run_id": run["id"],
        "run_attempt": run["run_attempt"],
        "run_url": url,
        "harnesses": rows,
    }
    serialized = json.dumps(portable, sort_keys=True, indent=2, allow_nan=False) + "\n"
    asset = hashlib.sha256(serialized.encode()).hexdigest() + ".json"
    block = [
        BEGIN,
        "",
        f"Measured at **{measured.isoformat()}**. Updated after scheduled CI (every four hours).",
        f"Tested Gym commit: [`{sha[:12]}`](https://github.com/{repo}/commit/{sha}).",
        f"[CI run]({url}) · [Generation report](../../../../assets/trajectory-capabilities/{asset}).",
        "",
        "| Harness | P0 | Passing scenarios | Runtime | Validation |",
        "| --- | --- | --- | --- | --- |",
    ]
    for harness, row in rows.items():
        if row["status"] == "unavailable":
            block.append(f"| `{harness}` | Unavailable | — | — | Setup, tests, execution, or collection failed |")
            continue
        passed = sum(s["verdict"] == "fulfilled" for s in row["scenarios"])
        runtime = row["runtime"].get("version") or ("source SHA-256 " + row["runtime"]["sha256"])
        runtime = runtime.replace("`", "").replace("|", "\\|").replace("\n", " ")
        counts = row["tests"]
        tests = (
            f"{counts['passed']} passed, {counts['skipped']} skipped"
            if counts
            else "Passed; counts not retained by older CI"
        )
        block.append(
            f"| `{harness}` | {row['status'].upper()} | {passed}/{len(row['scenarios'])} | `{runtime}` | {tests} |"
        )
    block += [
        "",
        "P0 requires every applicable individual P0 check to pass. Unavailable is a missing measurement,",
        "not a passing or failing capability claim. The matrices below retain the tested revision's check definitions.",
        "",
        *matrices,
        "",
        END,
    ]
    page = root / PAGE
    original = page.read_text()
    if original.count(BEGIN) != 1 or original.count(END) != 1 or original.index(BEGIN) >= original.index(END):
        raise ValueError("reference page needs exactly one ordered pair of generated-table markers")
    updated = original[: original.index(BEGIN)] + "\n".join(block) + original[original.index(END) + len(END) :]
    atomic_write(root / ASSETS / asset, serialized)
    atomic_write(page, updated)
    print(f"Rendered {url} into {page}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="NVIDIA-NeMo/Gym")
    args = parser.parse_args()
    try:
        run = latest_run(args.repo)
        with tempfile.TemporaryDirectory(prefix="conformance-docs-") as directory:
            artifacts = Path(directory) / "artifacts"
            download(run, repo=args.repo, output=artifacts)
            publish(root=ROOT, artifacts=artifacts, run=run, repo=args.repo)
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError, subprocess.CalledProcessError) as exc:
        print(f"Documentation measurement unavailable; publication stopped: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
