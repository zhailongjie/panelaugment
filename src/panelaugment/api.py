"""Reusable, study-independent public interface to PanelAugment."""

from copy import deepcopy
from dataclasses import dataclass
import warnings

import numpy as np

from .core import _axis, _integer, _positive, conditional_order, training_moments


def _genes(values, *, allow_empty=False):
    if isinstance(values, (str, bytes)):
        raise ValueError("Gene names must be a sequence of strings")
    values = list(values)
    if allow_empty and not values:
        return []
    if any(not isinstance(g, (str, np.str_)) for g in values):
        raise ValueError("Gene names must be strings")
    array = _axis(values, "gene_names")
    if any(not isinstance(g, (str, np.str_)) for g in array):
        raise ValueError("Gene names must be strings")
    result = array.tolist()
    if len(result) != len(set(result)):
        raise ValueError("Gene names must be unique")
    return result


@dataclass(frozen=True)
class SelectionResult:
    """An ordered fixed-base panel and its audit trace (no expression data)."""

    genes: list
    base_genes: list
    added_genes: list
    indices: list
    trace: list
    pair_support: list
    ridge: float
    method: str

    def to_dict(self):
        """Return an independent JSON-compatible record."""
        return deepcopy(dict(schema_version=1, method=self.method, ridge=self.ridge,
                             genes=self.genes, base_genes=self.base_genes,
                             added_genes=self.added_genes, indices=self.indices,
                             n_added=len(self.added_genes), total_size=len(self.genes),
                             objective=self.trace[-1]["objective"] if self.trace else 0.0,
                             trace=self.trace, pair_support=self.pair_support))

    def prefix(self, n_added):
        """Reuse a smaller budget from this same fitted search and ordered base."""
        n_added = _integer(n_added, "n_added")
        if n_added > len(self.added_genes):
            raise ValueError("Requested prefix exceeds the searched added budget")
        total = len(self.base_genes) + n_added
        return SelectionResult(genes=self.genes[:total], base_genes=self.base_genes.copy(),
                               added_genes=self.added_genes[:n_added], indices=self.indices[:total],
                               trace=deepcopy(self.trace[:total]), pair_support=deepcopy(self.pair_support),
                               ridge=self.ridge, method=self.method)


class PanelAugmenter:
    """Fit moments once; select multiple fixed-base panels from training data.

    The input is already normalized expression. Sparse inputs are densified:
    this implementation uses O(NP + P² + QP + PT) numerical storage.
    """

    def __init__(self, ridge=1.0):
        self.ridge = _positive(ridge, "ridge")

    def fit(self, X, y, groups, *, gene_names):
        """Estimate moments on the selection training role only; return self."""
        # A failed refit must not silently leave a previous fitted model usable.
        self.__dict__.pop("moments_", None)
        self.__dict__.pop("gene_names_", None)
        names = _genes(gene_names)
        try:
            from scipy import sparse
        except ImportError:
            sparse = None
        if np.iscomplexobj(X):
            raise ValueError("X must contain real values, not complex numbers")
        if sparse is not None and sparse.issparse(X):
            warnings.warn("PA densifies sparse expression and stores a dense gene covariance; "
                          "check available memory before fitting.", UserWarning, stacklevel=2)
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(names):
            raise ValueError("gene_names must match the columns of X")
        moments = training_moments(X, y, groups)
        self.gene_names_ = names
        self.moments_ = moments
        self.n_cells_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        return self

    def select(self, base_genes, n_added, *, diagonal=False):
        """Retain ordered base_genes and greedily add exactly n_added genes."""
        if not hasattr(self, "moments_"):
            raise ValueError("Call fit before select")
        base = _genes(base_genes, allow_empty=True)
        n_added = _integer(n_added, "n_added")
        positions = {gene: index for index, gene in enumerate(self.gene_names_)}
        unknown = [gene for gene in base if gene not in positions]
        if unknown:
            raise ValueError(f"Base genes are absent from the candidate axis: {unknown}")
        selected, trace = conditional_order(self.moments_["deltas"], self.moments_["covariance"],
                                            len(base) + n_added, ridge=self.ridge,
                                            diagonal=diagonal, initial=[positions[g] for g in base])
        names = [self.gene_names_[j] for j in selected]
        return SelectionResult(genes=names, base_genes=base, added_genes=names[len(base):],
                               indices=selected, trace=trace,
                               pair_support=deepcopy(self.moments_["pair_support"]),
                               ridge=self.ridge, method="diagonal" if diagonal else "PA")


def augment_panel(X, y, groups, genes, base_genes, n_added, *, ridge=1.0, diagonal=False):
    """One-shot selection from already normalized training expression."""
    return PanelAugmenter(ridge=ridge).fit(X, y, groups, gene_names=genes).select(
        base_genes=base_genes, n_added=n_added, diagonal=diagonal)
