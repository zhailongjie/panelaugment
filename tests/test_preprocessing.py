import itertools
import json

import numpy as np
import pytest

from panelaugment.preprocessing import (
    CandidatePreprocessor, log_normalize, measurement_consistency_screen,
)


def test_qc_fits_only_training_and_keeps_full_library_denominator():
    counts = np.array([[1, 0, 9], [3, 0, 7], [0, 10000, 0]], dtype=float)
    model = CandidatePreprocessor(min_detected=2)
    result = model.fit_transform(counts, ["a", "test_only", "c"],
                                 fit_mask=np.array([True, True, False]))
    assert model.genes_ == ("a", "c")
    np.testing.assert_allclose(result, log_normalize(counts)[:, [0, 2]])
    assert model.detection_counts_.tolist() == [2, 0, 2]
    assert model.fit_cell_count_ == 2
    assert model.to_dict()["measurement_screen"]["status"] == "not_evaluated"
    changed = counts.copy()
    changed[2] = [1000000, 10000000, 800000]
    other = CandidatePreprocessor(min_detected=2).fit(changed, ["a", "test_only", "c"],
                                                     fit_mask=np.array([True, True, False]))
    np.testing.assert_array_equal(model.variances_, other.variances_)
    assert json.loads(json.dumps(model.to_dict()))["fit_cell_count"] == 2


def test_mandatory_union_exclusions_and_transform_axis_alignment():
    counts = np.array([[1, 0, 2, 0], [3, 0, 1, 0]])
    totals = np.array([100, 100])
    genes = ["a", "base", "exclude", "low"]
    model = CandidatePreprocessor(min_detected=2).fit(
        counts, genes, fit_mask=np.ones(2, dtype=bool), library_totals=totals,
        mandatory_genes=["base"], excluded_genes=["exclude"])
    assert model.genes_ == ("a", "base")
    assert model.to_dict()["qc_failed_mandatory"] == ["base"]
    expected = np.log1p(counts[:, :2] * 100)
    np.testing.assert_array_equal(model.transform(counts[:, ::-1], genes[::-1],
                                                 library_totals=totals), expected)
    with pytest.raises(ValueError, match="denominator policy"):
        model.transform(counts, genes)
    with pytest.raises(ValueError, match="complete fitted"):
        model.transform(counts[:, :2], genes[:2], library_totals=totals)


def test_stable_variance_ranking_and_exact_qc_boundaries():
    counts = np.array([[1, 1, 0], [3, 3, 10], [2, 2, 0]])
    totals = np.full(3, 20)
    model = CandidatePreprocessor(min_detected=2, n_top_genes=1).fit(
        counts, ["first", "tied", "rare"], fit_mask=np.ones(3, dtype=bool), library_totals=totals)
    assert model.genes_ == ("first",)
    boundary = CandidatePreprocessor(min_detected=2, min_variance=model.variances_[0])
    with pytest.raises(ValueError, match="No candidate"):
        boundary.fit(counts, ["first", "tied", "rare"], fit_mask=np.ones(3, dtype=bool),
                     library_totals=totals)


def test_sparse_candidate_qc_matches_dense_without_densifying():
    sparse = pytest.importorskip("scipy.sparse")
    x = np.array([[0, 1, 5], [3, 0, 8], [2, 4, 9]], dtype=float)
    kwargs = dict(fit_mask=np.ones(3, dtype=bool), library_totals=np.full(3, 30))
    dense_model = CandidatePreprocessor(min_detected=2, n_top_genes=2)
    sparse_model = CandidatePreprocessor(min_detected=2, n_top_genes=2)
    dense_result = dense_model.fit_transform(x, ["a", "b", "c"], **kwargs)
    sparse_result = sparse_model.fit_transform(sparse.coo_matrix(x), ["a", "b", "c"], **kwargs)
    assert sparse.isspmatrix_csr(sparse_result)
    assert dense_model.genes_ == sparse_model.genes_
    np.testing.assert_allclose(sparse_result.toarray(), dense_result, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("mask", [[1, 1], [False, False], [True], [[True], [True]]])
def test_fit_partition_must_be_explicit_boolean_nonempty(mask):
    with pytest.raises(ValueError, match="fit_mask"):
        CandidatePreprocessor().fit([[1, 2], [2, 1]], ["a", "b"], fit_mask=mask)


def test_input_and_policy_contracts():
    x = np.array([[1, 2], [2, 1]])
    mask = np.ones(2, dtype=bool)
    model = CandidatePreprocessor(min_detected=1)
    for genes, kwargs in [(["a", "a"], {}), (["a", "b"], {"mandatory_genes": ["missing"]}),
                           (["a", "b"], {"mandatory_genes": ["a"], "excluded_genes": ["a"]})]:
        with pytest.raises(ValueError):
            model.fit(x, genes, fit_mask=mask, **kwargs)
    with pytest.raises(ValueError, match="cannot be combined"):
        CandidatePreprocessor(n_top_genes=1).fit(x, ["a", "b"], fit_mask=mask,
                                                 mandatory_genes=["a"])
    with pytest.raises(ValueError, match="Fewer QC"):
        CandidatePreprocessor(min_detected=1, n_top_genes=3).fit(x, ["a", "b"], fit_mask=mask)
    with pytest.raises(ValueError, match="fitted"):
        CandidatePreprocessor().to_dict()
    model.fit(x, ["a", "b"], fit_mask=mask)
    with pytest.raises(ValueError):
        model.fit(x, ["a", "a"], fit_mask=mask)
    with pytest.raises(ValueError, match="fitted"):
        model.transform(x, ["a", "b"])


def test_measurement_screen_strict_boundaries_and_not_applicable():
    positives = np.array([[4, 5, 4], [4, 4, 4], [91, 91, 90], [91, 91, 91]])
    result = measurement_consistency_screen(positives, [100] * 4, ["hit", "at5", "at90"])
    assert result["hit_mask"] == [True, False, False]
    assert result["excluded_genes"] == ["hit"]
    small = measurement_consistency_screen(positives[:2], [100] * 2, ["hit", "at5", "at90"])
    assert small["status"] == "not_applicable"
    assert small["hit_mask"] is None
    assert small["excluded_genes"] == []


def test_measurement_screen_matches_independent_partition_enumeration():
    rng = np.random.default_rng(19)
    support = np.array([43, 101, 97, 59, 103, 89])
    positives = np.floor(rng.uniform(size=(6, 500)) * (support[:, None] + 1)).astype(int)
    positives[:, 0] = [0, 0, 97, 59, 0, 0]
    expected = []
    for gene in positives.T:
        expected.append(any(all(int(gene[i]) * 20 < int(support[i]) for i in low)
                            and all(int(gene[i]) * 10 > 9 * int(support[i]) for i in high)
                            for low in itertools.combinations(range(6), 2)
                            for high in itertools.combinations([i for i in range(6) if i not in low], 2)))
    result = measurement_consistency_screen(positives, support, [str(i) for i in range(500)])
    assert result["hit_mask"] == expected
    huge = np.iinfo(np.uint64).max
    assert measurement_consistency_screen(np.array([[0], [0], [huge], [huge]], dtype=np.uint64),
                                           np.full(4, huge, dtype=np.uint64), ["x"])["hit_mask"] == [True]
    # Mixed signed/unsigned NumPy comparison can round these values to float64.
    with pytest.raises(ValueError):
        measurement_consistency_screen(np.array([[2**63]], dtype=np.uint64),
                                        np.array([2**63 - 1], dtype=np.int64), ["x"])


@pytest.mark.parametrize("positive,support", [([[0.0], [1.0]], [1, 1]),
                                               ([[True], [False]], [1, 1]),
                                               ([[2], [1]], [1, 1]),
                                               ([[0], [0]], [0, 1])])
def test_measurement_screen_rejects_invalid_aggregates(positive, support):
    with pytest.raises(ValueError):
        measurement_consistency_screen(positive, support, ["a"])


def test_candidate_pipeline_applies_gate_after_qc_and_preserves_normalization():
    counts = np.array([[1, 1, 0], [3, 2, 0], [2, 4, 0]])
    model = CandidatePreprocessor(min_detected=2)
    # Aggregate columns are reordered and omit a gene discarded by training QC.
    result = model.fit_transform(
        counts, ["flagged", "retained", "qc_fail"], fit_mask=np.ones(3, dtype=bool),
        measurement_genes=["retained", "flagged"], measurement_cell_counts=[100] * 4,
        measurement_positive_counts=[[50, 4], [50, 4], [50, 91], [50, 91]])
    assert model.genes_ == ("retained",)
    np.testing.assert_array_equal(result, log_normalize(counts)[:, [1]])
    report = model.to_dict()
    assert report["measurement_screen"]["status"] == "evaluable"
    assert report["measurement_screen"]["excluded_genes"] == ["flagged"]
    assert report["pre_screen_genes"] == ["flagged", "retained"]
    report["measurement_screen"]["excluded_genes"].append("mutated")
    assert model.measurement_screen_["excluded_genes"] == ["flagged"]
    np.testing.assert_array_equal(model.transform(counts[::-1], ["flagged", "retained", "qc_fail"]),
                                  result[::-1])


def test_candidate_gate_missing_coverage_and_mandatory_conflicts():
    x = np.array([[1, 2], [2, 1]])
    args = dict(fit_mask=np.ones(2, dtype=bool), measurement_cell_counts=[100] * 4,
                measurement_positive_counts=[[4, 50], [4, 50], [91, 50], [91, 50]])
    with pytest.raises(ValueError, match="mandatory gene"):
        CandidatePreprocessor(min_detected=1).fit(x, ["a", "b"], mandatory_genes=["a"], **args)
    with pytest.raises(ValueError, match="every pre-gate"):
        CandidatePreprocessor(min_detected=1).fit(x, ["a", "b"], measurement_genes=["a", "other"], **args)
    with pytest.raises(ValueError, match="together"):
        CandidatePreprocessor(min_detected=1).fit(x, ["a", "b"], fit_mask=np.ones(2, dtype=bool),
                                                 measurement_cell_counts=[100] * 4)
    with pytest.raises(ValueError, match="requires measurement"):
        CandidatePreprocessor(min_detected=1).fit(x, ["a", "b"], fit_mask=np.ones(2, dtype=bool),
                                                 measurement_genes=["a", "b"])


def test_candidate_gate_inapplicable_and_no_top_n_refill():
    x = np.array([[1, 1, 0], [3, 3, 10], [2, 2, 0]])
    totals = np.full(3, 20)
    args = dict(fit_mask=np.ones(3, dtype=bool), library_totals=totals)
    small = CandidatePreprocessor(min_detected=2).fit(
        x, ["first", "tied", "rare"], measurement_cell_counts=[100, 100],
        measurement_positive_counts=[[4, 91, 0], [4, 91, 0]], **args)
    assert small.genes_ == ("first", "tied")
    assert small.measurement_screen_["status"] == "not_applicable"
    assert small.measurement_screen_["hit_mask"] is None
    ranked = CandidatePreprocessor(min_detected=1, n_top_genes=2).fit(
        x, ["first", "tied", "rare"], measurement_cell_counts=[100] * 4,
        measurement_positive_counts=[[50, 50, 4], [50, 50, 4], [50, 50, 91], [50, 50, 91]], **args)
    assert ranked.pre_screen_genes_ == ("rare", "first")
    assert ranked.genes_ == ("first",)  # no refill from the lower-ranked tied gene
