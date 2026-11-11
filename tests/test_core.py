"""Numerical parity and independent direct-solve verification of the PA core."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from panelaugment import balanced_weights, conditional_order, training_moments


FIXTURE = Path(__file__).parent / "fixtures" / "frozen_core_golden.json"


@pytest.fixture
def golden():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_reference_fixture_is_bound_to_provenance():
    provenance = json.loads(FIXTURE.with_name("frozen_core_sha256.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == provenance["golden_fixture"]["sha256"]
    assert json.loads(FIXTURE.read_text())["source_sha256"] == provenance["upstream"]["sha256"]


def test_frozen_training_moments_and_weights(golden):
    data = golden["input"]
    actual = training_moments(data["X"], data["y"], data["groups"])
    expected = golden["moments"]
    np.testing.assert_allclose(
        balanced_weights(data["y"], data["groups"]), golden["weights"],
        atol=1e-15, rtol=1e-14,
    )
    for key in ("means", "raw_means", "covariance", "deltas", "scale"):
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(actual["classes"], expected["classes"])
    np.testing.assert_array_equal(actual["pairs"], expected["pairs"])
    assert actual["pair_support"] == expected["pair_support"]


def test_frozen_selection_and_step_trace(golden):
    moments = golden["moments"]
    for case in golden["selections"]:
        order, trace = conditional_order(
            moments["deltas"], moments["covariance"], case["budget"],
            ridge=case["ridge"], diagonal=case["diagonal"], initial=case["initial"],
        )
        assert list(order) == case["indices"]
        assert len(trace) == len(case["trace"])
        for actual, expected in zip(trace, case["trace"]):
            for key in ("index", "step", "compulsory"):
                assert actual[key] == expected[key]
            for key in ("objective", "gain", "pair_increments"):
                np.testing.assert_allclose(actual[key], expected[key], atol=1e-11, rtol=1e-11)


def direct_distances(deltas, matrix, subset):
    """Independent objective evaluation; no incremental state or Schur updates."""
    if not subset:
        return np.zeros(len(deltas))
    contrast = deltas[:, subset]
    return np.einsum(
        "ij,ji->i", contrast,
        np.linalg.solve(matrix[np.ix_(subset, subset)], contrast.T),
    )


@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("seed", [0, 11, 2026])
def test_incremental_search_matches_direct_solve_oracle(seed, diagonal):
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(9, 13))
    covariance = factor @ factor.T / 13
    deltas = rng.normal(size=(6, 9))
    ridge = 0.7
    matrix = (np.diag(np.diag(covariance)) if diagonal else covariance) + ridge * np.eye(9)
    base = [5, 1]
    actual, trace = conditional_order(deltas, covariance, 8, ridge=ridge, diagonal=diagonal, initial=base)
    expected = []
    previous = np.zeros(6)
    for step in range(8):
        if step < len(base):
            selected = base[step]
        else:
            candidates = [j for j in range(9) if j not in expected]
            objectives = [np.log1p(direct_distances(deltas, matrix, expected + [j])).sum() for j in candidates]
            selected = candidates[int(np.argmax(objectives))]
        expected.append(selected)
        current = direct_distances(deltas, matrix, expected)
        objective = np.log1p(current).sum()
        assert trace[step]["index"] == selected
        np.testing.assert_allclose(trace[step]["objective"], objective, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(trace[step]["pair_increments"], current - previous, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(trace[step]["gain"], objective - np.log1p(previous).sum(), atol=1e-11, rtol=1e-11)
        previous = current
    assert list(actual) == expected


def test_class_and_observed_group_equal_weighting():
    y = np.array(["A", "A", "A", "A", "B", "B", "B"])
    groups = np.array(["x", "x", "x", "y", "x", "z", "z"])
    w = balanced_weights(y, groups)
    np.testing.assert_allclose(w.sum(), 1.0)
    for label in np.unique(y):
        np.testing.assert_allclose(w[y == label].sum(), 0.5)
        for group in np.unique(groups[y == label]):
            np.testing.assert_allclose(w[(y == label) & (groups == group)].sum(), 0.25)


def test_pair_differences_use_only_shared_groups():
    X = np.array([[0.], [2.], [10.], [14.], [100.]])
    y = ["A", "B", "A", "B", "A"]
    groups = ["g1", "g1", "g2", "g2", "g3"]
    moments = training_moments(X, y, groups)
    np.testing.assert_allclose(moments["deltas"][0], -3.0 / moments["scale"])
    assert moments["pair_support"] == [
        {"label_a": "A", "label_b": "B", "groups": ["g1", "g2"], "n_groups": 2}
    ]
    assert not np.allclose(moments["deltas"][0], moments["means"][0] - moments["means"][1])


def test_missing_shared_group_raises():
    with pytest.raises(ValueError):
        training_moments([[1., 2.], [3., 4.]], ["A", "B"], ["g1", "g2"])


def test_tie_breaking_base_order_and_zero_budget():
    deltas, covariance = np.zeros((3, 5)), np.eye(5)
    order, trace = conditional_order(deltas, covariance, 5, initial=[3, 1])
    assert list(order) == [3, 1, 0, 2, 4]
    assert [step["compulsory"] for step in trace] == [True, True, False, False, False]
    assert all(step["objective"] == step["gain"] == 0 for step in trace)
    assert conditional_order(deltas, covariance, 0) == ([], [])


def test_selection_is_prefix_consistent(golden):
    moments = golden["moments"]
    largest, largest_trace = conditional_order(moments["deltas"], moments["covariance"], 8, initial=[4, 0])
    for budget in (2, 3, 5, 7):
        order, trace = conditional_order(moments["deltas"], moments["covariance"], budget, initial=[4, 0])
        assert list(order) == list(largest[:budget])
        np.testing.assert_allclose([v["objective"] for v in trace], [v["objective"] for v in largest_trace[:budget]], atol=1e-12)


@pytest.mark.parametrize("budget, initial, ridge", [
    (-1, [], 1), (4, [], 1), (1, [0, 1], 1), (2, [0, 0], 1),
    (2, [-1], 1), (2, [3], 1), (2, [], 0), (2, [], -1),
    (2, [], np.nan), (2, [], np.inf), (1.5, [], 1), (2, [0.5], 1),
])
def test_invalid_selection_parameters(budget, initial, ridge):
    with pytest.raises((ValueError, TypeError)):
        conditional_order(np.ones((1, 3)), np.eye(3), budget, initial=initial, ridge=ridge)


@pytest.mark.parametrize("deltas,covariance", [
    (np.ones((1, 3)), np.eye(2)),
    (np.array([[np.nan, 1.]]), np.eye(2)),
    (np.ones((1, 2)), np.array([[1., np.inf], [np.inf, 1.]])),
    (np.ones(3), np.eye(3)),
])
def test_invalid_selection_matrices(deltas, covariance):
    with pytest.raises((ValueError, TypeError)):
        conditional_order(deltas, covariance, 1)


@pytest.mark.parametrize("complex_part", ["deltas", "covariance"])
def test_complex_search_input_is_not_silently_cast_to_real(complex_part):
    deltas, covariance = np.ones((1, 2)), np.eye(2)
    if complex_part == "deltas":
        deltas = deltas.astype(complex)
        deltas[0, 0] += 2j
    else:
        covariance = covariance.astype(complex)
        covariance[0, 1] = 0.5j
        covariance[1, 0] = -0.5j
    with pytest.raises((ValueError, TypeError)):
        conditional_order(deltas, covariance, 1)


def test_complex_training_expression_is_not_silently_cast_to_real():
    with pytest.raises((ValueError, TypeError)):
        training_moments(np.array([[1. + 2j], [2. + 3j]]), ["A", "B"], ["g", "g"])


@pytest.mark.parametrize("X,y,groups", [
    ([[1.], [np.nan]], ["A", "B"], ["g", "g"]),
    ([[1.], [np.inf]], ["A", "B"], ["g", "g"]),
    ([[1.], [2.]], ["A"], ["g", "g"]),
    ([[1.], [2.]], ["A", "B"], ["g"]),
    ([[1.], [2.]], ["A", "A"], ["g", "g"]),
    ([[1.], [2.]], ["A", None], ["g", "g"]),
    ([[1.], [2.]], ["A", "B"], ["g", None]),
    ([[1.], [2.]], ["A", float("nan")], ["g", "g"]),
    ([[1.], [2.]], ["A", "B"], ["g", float("nan")]),
])
def test_invalid_training_inputs(X, y, groups):
    with pytest.raises((ValueError, TypeError)):
        training_moments(X, y, groups)


def test_missing_group_is_rejected_even_if_shared_by_both_classes():
    with pytest.raises((ValueError, TypeError)):
        training_moments([[1.], [2.], [3.], [4.]], ["A", "B", "A", "B"],
                         ["g", "g", float("nan"), float("nan")])


def test_literal_string_nan_is_an_explicit_valid_identity():
    # Reject a missing float value without prohibiting users' actual string IDs.
    moments = training_moments([[1.], [2.]], ["A", "B"], ["nan", "nan"])
    assert moments["pair_support"][0]["groups"] == ["nan"]


def test_constant_genes_and_inputs_are_preserved():
    X = np.ones((4, 3))
    original = X.copy()
    moments = training_moments(X, ["A", "B", "A", "B"], ["g1", "g1", "g2", "g2"])
    np.testing.assert_array_equal(X, original)
    np.testing.assert_allclose(moments["scale"], 1e-4)
    np.testing.assert_array_equal(moments["covariance"], np.zeros((3, 3)))
    np.testing.assert_array_equal(moments["deltas"], np.zeros((1, 3)))

