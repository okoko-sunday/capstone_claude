"""Confusion matrix and the per-class numbers you compute at the board.

Pure NumPy so the web API can use it without loading TensorFlow.
Convention (matches the brief): rows = TRUE label, columns = PREDICTED label.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from catdog.config import LABELS


@dataclass
class ClassReport:
    label: str
    support: int  # how many images truly were this class
    predicted: int  # how many images the system called this class
    precision: float | None  # of everything called <label>, how much really was?
    recall: float | None  # of the real <label>s, how many did it find?
    f1: float | None


@dataclass
class EvaluationReport:
    labels: list[str]
    matrix: list[list[int]]
    total: int
    correct: int
    accuracy: float | None
    per_class: list[ClassReport]
    macro_f1: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = LABELS) -> np.ndarray:
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        if t in index and p in index:  # "error" rows are reported separately
            matrix[index[t], index[p]] += 1
    return matrix


def _safe_div(a: float, b: float) -> float | None:
    return a / b if b else None


def evaluate(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = LABELS) -> EvaluationReport:
    m = confusion_matrix(y_true, y_pred, labels)
    per_class = []
    for i, label in enumerate(labels):
        tp, support, predicted = int(m[i, i]), int(m[i, :].sum()), int(m[:, i].sum())
        precision, recall = _safe_div(tp, predicted), _safe_div(tp, support)
        f1 = _safe_div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
        per_class.append(ClassReport(label, support, predicted, precision, recall, f1))
    f1s = [c.f1 or 0.0 for c in per_class if c.support]
    total, correct = int(m.sum()), int(np.trace(m))
    return EvaluationReport(
        labels=list(labels),
        matrix=m.tolist(),
        total=total,
        correct=correct,
        accuracy=_safe_div(correct, total),
        per_class=per_class,
        macro_f1=float(np.mean(f1s)) if f1s else None,
    )


def format_report(report: EvaluationReport) -> str:
    """Render the matrix exactly as you would draw it on the board."""
    pct = lambda v: "  n/a" if v is None else f"{v * 100:5.1f}%"  # noqa: E731
    w = max(len(label) for label in report.labels) + 2
    header = " " * (w + 6) + "".join(f"{label:>{w}}" for label in report.labels)
    lines = [" " * (w + 6) + "PREDICTED".center(w * len(report.labels)), header]
    for i, label in enumerate(report.labels):
        prefix = "TRUE  " if i == len(report.labels) // 2 else "      "
        row = "".join(f"{v:>{w}}" for v in report.matrix[i])
        lines.append(f"{prefix}{label:>{w}}{row}   = {sum(report.matrix[i])}")
    lines.append("")
    lines.append(f"Accuracy: {report.correct}/{report.total} = {pct(report.accuracy).strip()}")
    lines.append(f"{'class':<10}{'precision':>11}{'recall':>9}{'f1':>9}{'support':>9}")
    for c in report.per_class:
        lines.append(f"{c.label:<10}{pct(c.precision):>11}{pct(c.recall):>9}{pct(c.f1):>9}{c.support:>9}")
    return "\n".join(lines)


def plot_confusion_matrix(report: EvaluationReport, path: str | Path | None = None, title: str = "Confusion matrix"):
    """Save/return a matplotlib figure of the matrix (optional dependency)."""
    import matplotlib.pyplot as plt

    m = np.array(report.matrix)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=150)
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(report.labels)), report.labels)
    ax.set_yticks(range(len(report.labels)), report.labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    acc = "n/a" if report.accuracy is None else f"{report.accuracy:.1%}"
    ax.set_title(f"{title}\naccuracy {acc} on {report.total} images", fontsize=10)
    threshold = m.max() / 2 if m.size else 0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, m[i, j], ha="center", va="center", color="white" if m[i, j] > threshold else "#1f2937")
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig
