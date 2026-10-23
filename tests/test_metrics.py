import numpy as np
import pytest

from catdog.config import DecisionPolicy
from catdog.evaluation.metrics import confusion_matrix, evaluate, format_report
from catdog.evaluation.threshold import threshold_study, tune_animal_threshold
from catdog.inference.predictor import decide


def test_confusion_matrix_rows_are_truth():
    m = confusion_matrix(["dog", "dog", "cat", "neither"], ["dog", "cat", "cat", "dog"])
    assert m.tolist() == [[1, 0, 0], [1, 1, 0], [0, 1, 0]]  # labels: cat, dog, neither


def test_board_example():
    # 5 dogs, 5 cats, 5 neither — like assessment day.
    y_true = ["dog"] * 5 + ["cat"] * 5 + ["neither"] * 5
    y_pred = ["dog"] * 4 + ["neither"] + ["cat"] * 5 + ["neither"] * 3 + ["dog"] * 2
    r = evaluate(y_true, y_pred)
    assert (r.correct, r.total) == (12, 15)
    assert r.accuracy == pytest.approx(0.8)
    dog = next(c for c in r.per_class if c.label == "dog")
    assert dog.recall == pytest.approx(4 / 5)
    assert dog.precision == pytest.approx(4 / 6)
    assert "Accuracy: 12/15" in format_report(r)


def test_error_rows_are_ignored_and_empty_class_is_none():
    r = evaluate(["cat", "cat"], ["cat", "error"])
    assert r.total == 1
    assert next(c for c in r.per_class if c.label == "dog").precision is None


def test_decision_policy_threshold_override():
    policy = DecisionPolicy(animal_threshold=0.6)
    assert decide(np.array([0.5, 0.2, 0.3]), policy) == (2, "threshold")  # weak cat → neither
    assert decide(np.array([0.7, 0.2, 0.1]), policy) == (0, "model")
    assert decide(np.array([0.1, 0.2, 0.7]), policy) == (2, "model")


def test_threshold_study_perfect_separation():
    study = threshold_study(np.full(10, 0.99), np.full(10, 0.55))
    assert study["auroc"] == pytest.approx(1.0)
    assert study["best"]["balanced_accuracy"] == pytest.approx(1.0)


def test_threshold_study_confident_and_wrong():
    # Negatives as confident as pets: no threshold can separate them.
    study = threshold_study(np.full(10, 0.97), np.full(10, 0.97))
    assert study["negatives_confident_over_90"] == 1.0
    assert study["best"]["balanced_accuracy"] == pytest.approx(0.5)


def test_tune_animal_threshold_returns_value_in_range(rng):
    probs = rng.dirichlet([1, 1, 1], size=60)
    y = probs.argmax(1)
    t, sweep = tune_animal_threshold(probs, y)
    assert 0.34 <= t <= 0.9 and len(sweep) > 10
