"""Explicit normalization, training-only candidate QC, and measurement screening."""

from copy import deepcopy

import numpy as np

from .core import _positive


def log_normalize(counts, *, library_totals=None, target_sum=10000.0):
    """Normalize counts to target_sum and log1p, preserving zero-count rows.

    Without library_totals, counts MUST contain the full measured universe.
    For a candidate subset, pass totals computed before subsetting. Sparse
    input requires SciPy and returns CSR; dense input returns an ndarray.
    Fractional nonnegative count-like measurements are accepted.
    """
    target_sum = _positive(target_sum, "target_sum")
    try:
        from scipy import sparse
    except ImportError:
        sparse = None
    is_sparse = sparse is not None and sparse.issparse(counts)
    if np.iscomplexobj(counts) or np.iscomplexobj(library_totals):
        raise ValueError("counts and library_totals must contain real values")
    if is_sparse:
        # Inspect stored entries before COO/CSR conversion can sum duplicates.
        entries = counts.tocoo(copy=False).data
        if not np.isfinite(entries).all() or (entries < 0).any():
            raise ValueError("counts must be finite and nonnegative")
        x = sparse.csr_matrix(counts, dtype=np.float64, copy=True)
        x.sum_duplicates()
        values = x.data
    else:
        x = np.array(counts, dtype=np.float64, copy=True)
        values = x
    if x.ndim != 2 or min(x.shape) == 0:
        raise ValueError("counts must be a nonempty 2D matrix")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("counts must be finite and nonnegative")
    observed = np.asarray(x.sum(axis=1)).ravel()
    if library_totals is None:
        totals = observed
    else:
        totals = np.asarray(library_totals, dtype=np.float64)
        if totals.shape != (x.shape[0],) or not np.isfinite(totals).all() or (totals < 0).any():
            raise ValueError("library_totals must be finite nonnegative totals for each cell")
        if np.any(totals + 1e-10 * np.maximum(observed, 1) < observed):
            raise ValueError("library_totals cannot be smaller than the supplied row sums")
    if not np.isfinite(observed).all():
        raise ValueError("Count row totals overflowed")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        factors = target_sum / np.maximum(totals, 1)
        if is_sparse:
            x = (sparse.diags(factors) @ x).tocsr()
            x.data = np.log1p(x.data)
        else:
            x = np.log1p(x * factors[:, None])
    return x


def _gene_axis(genes, size, name="genes"):
    if isinstance(genes, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of gene identifiers")
    values = tuple(genes)
    if len(values) != size or any(not isinstance(g, str) or not g.strip() for g in values):
        raise ValueError(f"{name} must contain one nonempty string per gene")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique; resolve gene mappings before preprocessing")
    return values


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class CandidatePreprocessor:
    """Fit a candidate axis on explicitly marked training cells.

    QC uses raw detection counts >= ``min_detected`` and *unweighted population*
    variance of full-library CP10K/log1p values > ``min_variance``. With
    ``n_top_genes``, retain that many QC-passing genes by descending variance,
    breaking ties by the input axis. Otherwise preserve input order. Mandatory
    genes bypass QC; exclusions never do. Combining a top-N budget and mandatory
    genes is deliberately unsupported because the manuscript does not define it.

    ``fit_mask`` is required, even for an entirely training-only input. All input
    columns must describe the normalization universe, or ``library_totals`` must
    be supplied from the full universe before slicing. Transformation freezes the
    candidate decision and never estimates QC from new cells. Measurement
    consistency gating is part of this preprocessing pipeline: when target
    donor aggregates are supplied, it runs after training QC and automatically
    removes flagged candidates. Missing aggregates are explicitly recorded as
    ``not_evaluated``; fewer than four donors are ``not_applicable``.
    """

    def __init__(self, *, min_detected=20, min_variance=1e-10,
                 n_top_genes=None, target_sum=10000.0):
        self.min_detected = _integer(min_detected, "min_detected")
        if isinstance(min_variance, (bool, np.bool_)) or np.iscomplexobj(min_variance):
            raise ValueError("min_variance must be finite and nonnegative")
        self.min_variance = float(min_variance)
        if not np.isfinite(self.min_variance) or self.min_variance < 0:
            raise ValueError("min_variance must be finite and nonnegative")
        self.n_top_genes = None if n_top_genes is None else _integer(n_top_genes, "n_top_genes", 1)
        self.target_sum = _positive(target_sum, "target_sum")

    def fit(self, counts, genes, *, fit_mask, library_totals=None,
            mandatory_genes=(), excluded_genes=(), measurement_positive_counts=None,
            measurement_cell_counts=None, measurement_genes=None):
        """Fit training QC, then apply the measurement gate on donor aggregates.

        ``measurement_genes`` describes aggregate columns; it defaults to the
        count-matrix gene axis. Aggregates must cover every candidate surviving
        training QC, ranking, and explicit exclusions. This permits the frozen
        experiment's post-QC aggregate axes without requiring discarded genes.
        """
        # A failed refit must not leave an earlier candidate decision usable.
        for attribute in tuple(self.__dict__):
            if attribute.endswith("_"):
                delattr(self, attribute)
        # log_normalize validates all counts and denominators, including rows not
        # used to fit QC. Normalization itself is an independent per-cell operation.
        normalized = log_normalize(counts, library_totals=library_totals,
                                   target_sum=self.target_sum)
        axis = _gene_axis(genes, normalized.shape[1])
        mask = np.asarray(fit_mask)
        if mask.dtype != np.dtype(bool) or mask.shape != (normalized.shape[0],) or not mask.any():
            raise ValueError("fit_mask must be a boolean vector selecting at least one training cell")
        mandatory = _gene_axis(mandatory_genes, len(mandatory_genes), "mandatory_genes")
        excluded = _gene_axis(excluded_genes, len(excluded_genes), "excluded_genes")
        if not set(mandatory).issubset(axis) or not set(excluded).issubset(axis):
            raise ValueError("Mandatory and excluded genes must exist on the input axis; no zero filling")
        if set(mandatory) & set(excluded):
            raise ValueError("A mandatory gene cannot also be excluded")
        if mandatory and self.n_top_genes is not None:
            raise ValueError("n_top_genes and mandatory_genes cannot be combined")
        try:
            from scipy import sparse
        except ImportError:
            sparse = None
        is_sparse = sparse is not None and sparse.issparse(normalized)
        if is_sparse:
            raw = sparse.csr_matrix(counts, dtype=np.float64, copy=True)
            raw.sum_duplicates()
            detected = np.asarray((raw[mask] > 0).sum(axis=0)).ravel()
            train = normalized[mask]
            mean = np.asarray(train.mean(axis=0)).ravel()
            variance = np.asarray(train.power(2).mean(axis=0)).ravel() - mean * mean
        else:
            raw = np.asarray(counts, dtype=np.float64)
            detected = (raw[mask] > 0).sum(axis=0)
            train = normalized[mask]
            mean = train.mean(axis=0)
            variance = np.square(train).mean(axis=0) - mean * mean
        if not np.isfinite(variance).all():
            raise ValueError("Nonfinite training variance")
        passed = (detected >= self.min_detected) & (variance > self.min_variance)
        mandatory_mask = np.isin(axis, mandatory)
        excluded_mask = np.isin(axis, excluded)
        keep = (passed | mandatory_mask) & ~excluded_mask
        selected = np.flatnonzero(keep)
        if self.n_top_genes is not None:
            if len(selected) < self.n_top_genes:
                raise ValueError("Fewer QC-passing eligible genes than n_top_genes")
            selected = selected[np.argsort(-variance[selected], kind="stable")[:self.n_top_genes]]
        if len(selected) == 0:
            raise ValueError("No candidate genes remain after QC and exclusions")
        before_screen = tuple(axis[i] for i in selected)
        if (measurement_positive_counts is None) != (measurement_cell_counts is None):
            raise ValueError("Provide measurement_positive_counts and measurement_cell_counts together")
        if measurement_positive_counts is None:
            if measurement_genes is not None:
                raise ValueError("measurement_genes requires measurement detection aggregates")
            screen = {
                "schema": "panelaugment_measurement_consistency_v1",
                "status": "not_evaluated", "reason": "measurement_aggregates_not_supplied",
                "donor_count": None, "gene_count": len(before_screen),
                "genes": list(before_screen), "hit_mask": None, "excluded_genes": [],
                "rule": "two donors positive*20<n and two disjoint donors positive*10>9*n",
                "boundary": "strict", "training_condition_used": False,
                "class_condition_used": False, "gene_name_condition_used": False,
                "scope": "PA measurement preprocessing; status is not a negative gate decision",
            }
        else:
            aggregate_genes = axis if measurement_genes is None else measurement_genes
            # Validate the entire supplied aggregate before selecting its columns.
            supplied_screen = measurement_consistency_screen(
                measurement_positive_counts, measurement_cell_counts, aggregate_genes)
            aggregate_axis = supplied_screen["genes"]
            aggregate_lookup = {g: i for i, g in enumerate(aggregate_axis)}
            if not set(before_screen).issubset(aggregate_lookup):
                raise ValueError("Measurement aggregates must cover every pre-gate candidate; no zero filling")
            reordered = np.asarray(measurement_positive_counts)[:,
                         [aggregate_lookup[g] for g in before_screen]]
            screen = measurement_consistency_screen(reordered, measurement_cell_counts, before_screen)
            if set(screen["excluded_genes"]) & set(mandatory):
                raise ValueError("Measurement gate flags a mandatory gene; resolve the base-panel conflict explicitly")
            hit_genes = set(screen["excluded_genes"])
            selected = np.asarray([i for i in selected if axis[i] not in hit_genes], dtype=int)
            if not len(selected):
                raise ValueError("No candidate genes remain after the measurement gate")
        # Publish state only after all validation succeeds.
        self.input_genes_ = axis
        self.genes_ = tuple(axis[i] for i in selected)
        self.selected_indices_ = selected.copy()
        self.detection_counts_ = detected.astype(np.int64)
        self.variances_ = variance.copy()
        self.qc_pass_ = passed.copy()
        self.fit_cell_count_ = int(mask.sum())
        self.mandatory_genes_ = mandatory
        self.excluded_genes_ = excluded
        self.pre_screen_genes_ = before_screen
        self.measurement_screen_ = screen
        self.external_library_totals_ = library_totals is not None
        return self

    def transform(self, counts, genes, *, library_totals=None):
        """Normalize and slice new cells using the frozen candidate axis.

        The same complete input gene set is required, but its order may differ.
        External full-library totals are required on every transform if they were
        used during fit. Missing genes are errors, never silent zeros.
        """
        self._require_fitted()
        if self.external_library_totals_ != (library_totals is not None):
            raise ValueError("Use the same full-library denominator policy as during fit")
        normalized = log_normalize(counts, library_totals=library_totals,
                                   target_sum=self.target_sum)
        axis = _gene_axis(genes, normalized.shape[1])
        if set(axis) != set(self.input_genes_):
            raise ValueError("Transform requires the complete fitted input gene set; no zero filling")
        lookup = {gene: i for i, gene in enumerate(axis)}
        return normalized[:, [lookup[gene] for gene in self.genes_]].copy()

    def fit_transform(self, counts, genes, *, fit_mask, library_totals=None,
                      mandatory_genes=(), excluded_genes=(), measurement_positive_counts=None,
                      measurement_cell_counts=None, measurement_genes=None):
        """Fit on marked rows and transform all rows; callers select training rows."""
        self.fit(counts, genes, fit_mask=fit_mask, library_totals=library_totals,
                 mandatory_genes=mandatory_genes, excluded_genes=excluded_genes,
                 measurement_positive_counts=measurement_positive_counts,
                 measurement_cell_counts=measurement_cell_counts, measurement_genes=measurement_genes)
        return self.transform(counts, genes, library_totals=library_totals)

    def _require_fitted(self):
        if not hasattr(self, "genes_"):
            raise ValueError("CandidatePreprocessor must be fitted before use")

    def to_dict(self):
        """Return a JSON-compatible audit record without cell or donor identities."""
        self._require_fitted()
        return {
            "schema": "panelaugment_candidate_preprocessing_v1",
            "parameters": {"min_detected": self.min_detected,
                           "min_variance": self.min_variance,
                           "n_top_genes": self.n_top_genes, "target_sum": self.target_sum},
            "fit_cell_count": self.fit_cell_count_,
            "normalization": "full_library_CP10K_log1p" if self.target_sum == 10000 else "full_library_target_sum_log1p",
            "external_library_totals": self.external_library_totals_,
            "variance": "unweighted_population_second_moment_minus_squared_mean",
            "candidate_order": "descending_variance_stable_input_order" if self.n_top_genes is not None else "input_order",
            "input_genes": list(self.input_genes_), "genes": list(self.genes_),
            "selected_indices": self.selected_indices_.tolist(),
            "detection_counts": self.detection_counts_.tolist(),
            "variances": self.variances_.tolist(), "qc_pass": self.qc_pass_.tolist(),
            "mandatory_genes": list(self.mandatory_genes_),
            "excluded_genes": list(self.excluded_genes_),
            "measurement_screen": deepcopy(self.measurement_screen_),
            "pre_screen_genes": list(self.pre_screen_genes_),
            "qc_failed_mandatory": [gene for gene, passed in zip(self.input_genes_, self.qc_pass_)
                                    if gene in set(self.mandatory_genes_) and not passed],
        }


def measurement_consistency_screen(positive_counts, cell_counts, genes):
    """Screen exact donor-by-gene detection counts with the manuscript's rule.

    A gene is flagged if at least two independent target donors have detection
    strictly below 5% and at least two other donors strictly above 90%. Integer
    comparisons preserve the exact boundaries. Fewer than four donors produces
    ``status='not_applicable'`` and ``hit_mask=None``, not a negative decision.

    Supply aggregated counts from independent biological donors, not sections or
    batches from the same donor. No class labels, reference expression, or gene
    names affect the decision. This target-informed screen does not establish
    independent external validation. CandidatePreprocessor calls this function
    as its measurement-gating step and applies its exclusions automatically.
    """
    positive = np.asarray(positive_counts)
    support = np.asarray(cell_counts)
    if positive.ndim != 2 or min(positive.shape) == 0:
        raise ValueError("positive_counts must be a nonempty donor-by-gene matrix")
    if support.shape != (positive.shape[0],):
        raise ValueError("cell_counts must contain one support count per donor")
    if positive.dtype.kind not in "iu" or support.dtype.kind not in "iu":
        raise ValueError("Detection and support counts must be exact integers, not floats or booleans")
    if ((positive < 0).any() or (support <= 0).any()
            or any(int(p) > int(n) for row, n in zip(positive, support) for p in row)):
        raise ValueError("Require positive donor support and 0 <= positive_counts <= cell_counts")
    axis = _gene_axis(genes, positive.shape[1])
    # Python integers prevent overflow even for int64/uint64 inputs at their limit.
    low = np.asarray([[int(p) * 20 < int(n) for p in row]
                      for row, n in zip(positive, support)], dtype=bool).sum(axis=0)
    high = np.asarray([[int(p) * 10 > int(n) * 9 for p in row]
                       for row, n in zip(positive, support)], dtype=bool).sum(axis=0)
    applicable = len(support) >= 4
    hit = (low >= 2) & (high >= 2) if applicable else None
    return {
        "schema": "panelaugment_measurement_consistency_v1",
        "status": "evaluable" if applicable else "not_applicable",
        "reason": None if applicable else "fewer_than_four_independent_donors",
        "donor_count": len(support), "gene_count": len(axis),
        "genes": list(axis), "hit_mask": hit.tolist() if applicable else None,
        "excluded_genes": [g for g, flag in zip(axis, hit) if flag] if applicable else [],
        "low_donor_counts": low.tolist(), "high_donor_counts": high.tolist(),
        "rule": "two donors positive*20<n and two disjoint donors positive*10>9*n",
        "boundary": "strict", "training_condition_used": False,
        "class_condition_used": False, "gene_name_condition_used": False,
        "scope": "PA measurement preprocessing; target-derived inputs do not establish independent validation",
    }
