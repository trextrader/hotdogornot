"""9-class confusion report.

Pure stdlib so it runs on the local CPU PC and inside CI without torch.
Surfaces the subtle pairs the customer cares about: adjacent precision
sizes and male/female within a size.
"""
from __future__ import annotations

from typing import Sequence


def _base_and_gender(name: str) -> tuple[str, str]:
    # "2.92mm-M" -> ("2.92mm", "M"); "SMA-F" -> ("SMA", "F")
    base, _, g = name.rpartition("-")
    return (base or name), g


def build_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> dict:
    n = len(class_names)
    confusion = [[0 for _ in range(n)] for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    total = len(y_true)
    correct = sum(confusion[i][i] for i in range(n))

    per_class: dict[str, dict[str, float]] = {}
    for i, name in enumerate(class_names):
        tp = confusion[i][i]
        support = sum(confusion[i])
        pred_pos = sum(confusion[r][i] for r in range(n))
        recall = tp / support if support else 0.0
        precision = tp / pred_pos if pred_pos else 0.0
        per_class[name] = {
            "support": support,
            "recall": recall,
            "precision": precision,
        }

    notable: dict[str, int] = {}
    for i in range(n):
        for j in range(n):
            if i == j or confusion[i][j] == 0:
                continue
            bi, gi = _base_and_gender(class_names[i])
            bj, gj = _base_and_gender(class_names[j])
            same_size_diff_gender = (bi == bj and gi != gj)
            diff_size = (bi != bj)
            if same_size_diff_gender or diff_size:
                notable[f"{class_names[i]} -> {class_names[j]}"] = confusion[i][j]

    return {
        "overall_accuracy": (correct / total) if total else 0.0,
        "confusion": confusion,
        "class_names": list(class_names),
        "per_class": per_class,
        "notable_confusions": notable,
    }


def render_markdown(report: dict) -> str:
    names = report["class_names"]
    lines = [
        "# 9-Class Evaluation",
        "",
        f"- Overall accuracy: **{report['overall_accuracy']:.4f}**",
        "",
        "## Per-class",
        "",
        "| class | support | precision | recall |",
        "|---|---:|---:|---:|",
    ]
    for name in names:
        pc = report["per_class"][name]
        lines.append(
            f"| {name} | {pc['support']} | {pc['precision']:.3f} | {pc['recall']:.3f} |"
        )
    lines += ["", "## Notable confusions (adjacent size / M-vs-F)", ""]
    if report["notable_confusions"]:
        for pair, cnt in sorted(
            report["notable_confusions"].items(), key=lambda kv: -kv[1]
        ):
            lines.append(f"- {pair}: {cnt}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"
