"""Evidence for the "neither" strategy.

Two questions, answered with numbers instead of opinions:

1. Approach B (two-class model + confidence threshold): if we only trained
   on cats and dogs, how well does "max probability < t => neither" separate
   real pets from everything else? `threshold_study` sweeps t and reports,
   for each value, how many negatives would land in the wrong pile.

2. Approach A (three-class model): the softmax already has a "neither"
   output, but we add a safety net — if the strongest animal probability is
   below t, call it "neither". `tune_animal_threshold` picks t on validation
   data by macro-F1.
"""

from __future__ import annotations

import numpy as np

from catdog.config import CAT, DOG, LABELS, NEITHER, DecisionPolicy
from catdog.evaluation.metrics import evaluate
from catdog.inference.predictor import decide

THRESHOLDS = np.round(np.arange(0.50, 0.995, 0.01), 2)


def threshold_study(pet_max_probs: np.ndarray, negative_max_probs: np.ndarray) -> dict:
    """Approach B. Inputs: the 2-class model's top probability on real pets
    and on negative images. Returns a sweep plus the best achievable point."""
    pet_max_probs, negative_max_probs = np.asarray(pet_max_probs), np.asarray(negative_max_probs)
    rows = []
    for t in THRESHOLDS:
        pets_kept = float((pet_max_probs >= t).mean())  # pets still called cat/dog
        negs_rejected = float((negative_max_probs < t).mean())  # negatives correctly called neither
        rows.append(
            {
                "threshold": float(t),
                "pet_recall": pets_kept,
                "neither_recall": negs_rejected,
                "balanced_accuracy": (pets_kept + negs_rejected) / 2,
            }
        )
    best = max(rows, key=lambda r: r["balanced_accuracy"])
    return {
        "sweep": rows,
        "best": best,
        "auroc": _auroc(pet_max_probs, negative_max_probs),
        "negatives_confident_over_90": float((negative_max_probs >= 0.9).mean()),
        "n_pets": int(len(pet_max_probs)),
        "n_negatives": int(len(negative_max_probs)),
    }


def tune_animal_threshold(probs: np.ndarray, y_true: np.ndarray) -> tuple[float, list[dict]]:
    """Approach A safety net: choose the threshold that maximises macro-F1."""
    true_labels = [LABELS[i] for i in y_true]
    sweep = []
    for t in np.round(np.arange(0.34, 0.91, 0.02), 2):
        policy = DecisionPolicy(animal_threshold=float(t))
        pred_labels = [LABELS[decide(p, policy)[0]] for p in probs]
        report = evaluate(true_labels, pred_labels)
        sweep.append({"threshold": float(t), "macro_f1": report.macro_f1, "accuracy": report.accuracy})
    # Prefer the lowest threshold among (near-)ties: it overrides the model least.
    best_f1 = max(r["macro_f1"] for r in sweep)
    best = next(r for r in sweep if r["macro_f1"] >= best_f1 - 1e-3)
    return best["threshold"], sweep


def animal_max_prob(probs: np.ndarray) -> np.ndarray:
    return np.maximum(probs[:, CAT], probs[:, DOG])


def _auroc(positives: np.ndarray, negatives: np.ndarray) -> float | None:
    """Probability that a random pet scores higher than a random negative."""
    if not len(positives) or not len(negatives):
        return None
    scores = np.concatenate([positives, negatives])
    ranks = scores.argsort().argsort() + 1.0
    pos_rank_sum = ranks[: len(positives)].sum()
    return float((pos_rank_sum - len(positives) * (len(positives) + 1) / 2) / (len(positives) * len(negatives)))


__all__ = ["threshold_study", "tune_animal_threshold", "animal_max_prob", "NEITHER"]
