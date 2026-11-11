"""User-facing panel contracts and explicit normalization semantics."""

import json
from pathlib import Path

import numpy as np
import pytest

from panelaugment import PanelAugmenter, augment_panel, log_normalize


@pytest.fixture
def data():
    path = Path(__file__).parent / "fixtures" / "frozen_core_golden.json"
    fixture = json.loads(path.read_text(encoding="utf-8"))
    return fixture["input"]


def test_public_interface_preserves_base_and_matches_frozen_selection(data):
    model = PanelAugmenter().fit(data["X"], data["y"], data["groups"], gene_names=data["genes"])
    result = model.select(base_genes=["gene_4", "gene_0"], n_added=3)
    assert list(result.base_genes) == ["gene_4", "gene_0"]
    assert list(result.genes[:2]) == list(result.base_genes)
    assert len(result.genes) == len(set(result.genes)) == 5
    assert list(result.added_genes) == list(result.genes[2:])
    assert result.ridge == 1.0
    assert list(result.genes) == [data["genes"][j] for j in result.indices]
    assert len(result.trace) == 5
    assert len(result.pair_support) == 3
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert payload["genes"] == list(result.genes)
    one_shot = augment_panel(data["X"], data["y"], data["groups"], data["genes"], ["gene_4", "gene_0"], 3)
    assert list(result.genes) == list(one_shot.genes)


def test_prefixes_count_added_genes_and_allow_zero(data):
    model = PanelAugmenter().fit(data["X"], data["y"], data["groups"], gene_names=data["genes"])
    result = model.select(base_genes=["gene_4", "gene_0"], n_added=5)
    for k in (0, 1, 3, 5):
        prefix = result.prefix(k)
        direct = model.select(base_genes=["gene_4", "gene_0"], n_added=k)
        assert list(prefix.genes) == list(direct.genes) == list(result.genes[:2 + k])
        assert len(prefix.added_genes) == k
        assert len(prefix.trace) == 2 + k
    for invalid in (-1, 6, 1.5):
        with pytest.raises((ValueError, TypeError)):
            result.prefix(invalid)


def test_empty_base_and_zero_added(data):
    result = augment_panel(data["X"], data["y"], data["groups"], data["genes"], [], 0)
    assert list(result.genes) == list(result.indices) == list(result.trace) == []


@pytest.mark.parametrize("base,n_added", [
    (["missing"], 1), (["gene_0", "gene_0"], 1), (["gene_0"], -1),
    (["gene_0"], 8), (["gene_0"], 1.5),
])
def test_invalid_panel_requests(data, base, n_added):
    model = PanelAugmenter().fit(data["X"], data["y"], data["groups"], gene_names=data["genes"])
    with pytest.raises((ValueError, TypeError)):
        model.select(base_genes=base, n_added=n_added)


def test_gene_axis_validation(data):
    for genes in (data["genes"][:-1], ["duplicate"] * len(data["genes"]), data["genes"][:-1] + [7]):
        with pytest.raises(ValueError):
            PanelAugmenter().fit(data["X"], data["y"], data["groups"], gene_names=genes)


def test_unfitted_model_fails_clearly():
    with pytest.raises((ValueError, RuntimeError)):
        PanelAugmenter().select(base_genes=[], n_added=1)


def test_results_are_independent_and_failed_refit_invalidates_model(data):
    model = PanelAugmenter().fit(data["X"], data["y"], data["groups"], gene_names=data["genes"])
    first = model.select(base_genes=["gene_0"], n_added=2)
    original_genes = list(first.genes)
    record = first.to_dict()
    record["genes"].append("not-a-gene")
    record["trace"][0]["objective"] = -1
    assert list(first.genes) == original_genes
    assert first.trace[0]["objective"] >= 0
    second = model.select(base_genes=["gene_0"], n_added=2)
    assert list(second.genes) == original_genes
    with pytest.raises(ValueError):
        model.fit([[1.]], ["A"], ["g"], gene_names=["gene_0"])
    with pytest.raises((ValueError, RuntimeError)):
        model.select(base_genes=[], n_added=1)


def test_full_library_denominator_survives_candidate_slicing():
    full = np.array([[2., 3., 5.], [0., 0., 0.], [7., 1., 2.]])
    before = full.copy()
    subset = full[:, [0, 2]]
    normalized_full = log_normalize(full)
    normalized_subset = log_normalize(subset, library_totals=full.sum(axis=1))
    np.testing.assert_allclose(normalized_subset, normalized_full[:, [0, 2]], atol=1e-14)
    np.testing.assert_array_equal(normalized_subset[1], [0., 0.])
    np.testing.assert_array_equal(full, before)
    assert not np.allclose(log_normalize(subset)[0], normalized_subset[0])


@pytest.mark.parametrize("counts,totals", [
    ([[1., -1.]], None), ([[np.inf, 1.]], None),
    ([[np.nan, 1.]], None), ([[1., 2.]], [2.]),
    ([[1., 2.]], [-1.]), ([[1., 2.]], [np.nan]),
    ([[1., 2.]], [3., 3.]),
])
def test_invalid_normalization_inputs(counts, totals):
    with pytest.raises((ValueError, TypeError)):
        log_normalize(counts, library_totals=totals)


def test_complex_fit_expression_is_not_silently_cast_to_real(data):
    X = np.asarray(data["X"], dtype=complex)
    X[0, 0] += 1j
    with pytest.raises((ValueError, TypeError)):
        PanelAugmenter().fit(X, data["y"], data["groups"], gene_names=data["genes"])


@pytest.mark.parametrize("complex_part", ["counts", "library_totals"])
def test_complex_normalization_input_is_not_silently_cast_to_real(complex_part):
    counts, totals = np.array([[1., 2.]]), np.array([5.])
    if complex_part == "counts":
        counts = counts.astype(complex) + 1j
    else:
        totals = totals.astype(complex) + 1j
    with pytest.raises((ValueError, TypeError)):
        log_normalize(counts, library_totals=totals)


def test_complex_sparse_expression_is_not_silently_cast_to_real(data):
    sparse = pytest.importorskip("scipy.sparse")
    complex_counts = sparse.csr_matrix([[1. + 1j, 0.], [2., 3.]])
    with pytest.raises((ValueError, TypeError)):
        log_normalize(complex_counts)
    X = sparse.csr_matrix(np.asarray(data["X"], dtype=complex) + 1j)
    with pytest.raises((ValueError, TypeError)):
        PanelAugmenter().fit(X, data["y"], data["groups"], gene_names=data["genes"])


@pytest.mark.parametrize("format", ["coo", "csr"])
def test_negative_sparse_duplicate_entries_cannot_be_hidden_by_summing(format):
    sparse = pytest.importorskip("scipy.sparse")
    if format == "coo":
        counts = sparse.coo_matrix(([2., -1.], ([0, 0], [0, 0])), shape=(1, 2))
    else:
        counts = sparse.csr_matrix(([2., -1.], [0, 0], [0, 2]), shape=(1, 2))
    assert not counts.has_canonical_format
    # The represented matrix is nonnegative, but the supplied count entries are invalid.
    np.testing.assert_array_equal(counts.toarray(), [[1., 0.]])
    with pytest.raises(ValueError):
        log_normalize(counts)


def test_sparse_normalization_and_fit_parity(data):
    sparse = pytest.importorskip("scipy.sparse")
    counts = sparse.csr_matrix([[2., 0., 3.], [0., 0., 0.], [1., 2., 0.]])
    original = counts.copy()
    normalized = log_normalize(counts, library_totals=[10., 0., 8.])
    assert sparse.issparse(normalized)
    np.testing.assert_allclose(normalized.toarray(), log_normalize(counts.toarray(), library_totals=[10., 0., 8.]))
    np.testing.assert_array_equal(counts.toarray(), original.toarray())
    dense = PanelAugmenter().fit(data["X"], data["y"], data["groups"], gene_names=data["genes"])
    with pytest.warns(UserWarning, match="densif"):
        sparse_model = PanelAugmenter().fit(sparse.csr_matrix(data["X"]), data["y"], data["groups"], gene_names=data["genes"])
    assert list(dense.select(base_genes=[], n_added=5).genes) == list(sparse_model.select(base_genes=[], n_added=5).genes)

