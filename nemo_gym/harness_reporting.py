# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free Markdown rendering for retained harness check results."""


def render_matrices(harnesses: dict) -> str:
    """Render saved check results; never infer a pass from an absent failure message."""
    labels = {"pass": "✓", "fail": "✗", "not_applicable": "—"}
    lines = [
        "# Harness checks by scenario",
        "",
        "✓ Pass · ✗ Requirement not demonstrated (invalid, missing, blocked, or no result) · — Not applicable.",
        "",
    ]
    for harness, row in harnesses.items():
        scenarios = row["scenarios"]
        keys = list(dict.fromkeys(c["id"] for s in scenarios for c in s.get("checks", [])))
        lines += [
            f"## {harness}",
            "",
            "| Check | " + " | ".join(s["scenario"] for s in scenarios) + " |",
            "|---|" + "---|" * len(scenarios),
        ]
        for kind in ("schema", "semantic", "behavioral", "health"):
            group = []
            for key in keys:
                found = [next((c for c in s.get("checks", []) if c["id"] == key), None) for s in scenarios]
                if next(c["kind"] for c in found if c) != kind:
                    continue
                cells = [labels[c["status"]] if c else "✗" for c in found]
                group.append(f"| `{key}` | " + " | ".join(cells) + " |")
            if group:
                lines.append(f"| **{kind.title()}** |" + " |" * len(scenarios))
                lines.extend(group)
        if not keys:
            lines.append("| No check results available | " + " | ".join("✗" for _ in scenarios) + " |")
        lines.append("")
    return "\n".join(lines)
