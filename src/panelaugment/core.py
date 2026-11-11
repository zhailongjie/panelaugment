"""Donor-aware moments and the frozen PA conditional greedy search.

Arithmetic follows the manuscript's frozen implementation; validation is stricter.
No study-specific inputs, global configuration, or normalization are hidden here.
"""

from itertools import combinations
from numbers import Integral, Real

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _axis(values, name, length=None):
    # Validate original values before NumPy can coerce NaN or numeric identities
    # into strings in mixed Python sequences.
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a nonempty 1D axis of matching length")
    kinds = set()
    for value in raw:
        if value is None or not isinstance(value, (str, Real, np.integer)):
            raise ValueError(f"{name} contains missing or unsupported values")
        if isinstance(value, Real) and not np.isfinite(value):
            raise ValueError(f"{name} contains missing or nonfinite values")
        kinds.add("str" if isinstance(value, str) else "number")
    if len(kinds) > 1:
        raise ValueError(f"{name} must not mix string and numeric identities")
    a = np.asarray(values)
    if a.ndim != 1 or not len(a) or (length is not None and len(a) != length):
        raise ValueError(f"{name} must be a nonempty 1D axis of matching length")
    if a.dtype.kind in "fc" and not np.isfinite(a).all():
        raise ValueError(f"{name} contains missing or nonfinite values")
    if a.dtype.kind == "O":
        # Do not silently turn missing values into the literal label 'nan'.
        for value in a:
            if value is None or not isinstance(value, (str, Real, np.integer)):
                raise ValueError(f"{name} contains missing or unsupported values")
            if isinstance(value, Real) and not np.isfinite(value):
                raise ValueError(f"{name} contains missing or nonfinite values")
        kinds = {"str" if isinstance(v, str) else "number" for v in a}
        if len(kinds) > 1:
            raise ValueError(f"{name} must not mix string and numeric identities")
    if a.dtype.kind not in "biufUO":
        raise ValueError(f"{name} must contain strings or real numeric identities")
    if any(not str(v).strip() for v in a):
        raise ValueError(f"{name} contains empty identities")
    return a


def balanced_weights(y, groups):
    """Equal class mass, then equal mass per observed group within each class."""
    y = _axis(y, "y")
    groups = _axis(groups, "groups", len(y))
    weights = np.zeros(len(y), dtype=float)
    classes = np.unique(y)
    for cls in classes:
        present = np.unique(groups[y == cls])
        for group in present:
            mask = (y == cls) & (groups == group)
            weights[mask] = 1.0 / (len(classes) * len(present) * mask.sum())
    return weights


def training_moments(X, y, groups):
    """Estimate shared within-group/class covariance and shared-group contrasts.

    X is a finite, dense, already normalized cells-by-genes array. This function
    does not normalize counts, select candidates, or filter training cells.
    """
    if np.iscomplexobj(X):
        raise ValueError("X must contain real values, not complex numbers")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite 2D matrix")
    y = _axis(y, "y", len(X))
    groups = _axis(groups, "groups", len(X))
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("At least two training classes are required")
    w = balanced_weights(y, groups)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        center = w @ X
        total_var = w @ ((X - center) ** 2)
        scale = np.sqrt(np.maximum(total_var, 1e-8))
        Z = X / scale
        residual = Z.copy()
        means, raw_means = [], []
        group_class_means = {}
        for cls in classes:
            pieces, raw_pieces = [], []
            for group in np.unique(groups[y == cls]):
                mask = (y == cls) & (groups == group)
                mu = Z[mask].mean(axis=0)
                residual[mask] -= mu
                group_class_means[(group, cls)] = mu
                pieces.append(mu)
                raw_pieces.append(X[mask].mean(axis=0))
            means.append(np.mean(pieces, axis=0))
            raw_means.append(np.mean(raw_pieces, axis=0))
        covariance = (residual.T * w) @ residual
        covariance = (covariance + covariance.T) / 2
        pairs = list(combinations(range(len(classes)), 2))
        deltas, pair_support = [], []
        for a, b in pairs:
            ca, cb = classes[a], classes[b]
            shared = [g for g in np.unique(groups)
                      if (g, ca) in group_class_means and (g, cb) in group_class_means]
            if not shared:
                raise ValueError(f"Classes {ca!r} and {cb!r} have no shared training group")
            deltas.append(np.mean([group_class_means[(g, ca)] - group_class_means[(g, cb)]
                                   for g in shared], axis=0))
            pair_support.append(dict(label_a=str(ca), label_b=str(cb),
                                     groups=[str(g) for g in shared], n_groups=len(shared)))
        deltas = np.stack(deltas)
    if not np.isfinite(covariance).all() or not np.isfinite(deltas).all():
        raise FloatingPointError("Nonfinite training moments; inspect expression scale")
    return dict(classes=classes, pairs=pairs, means=np.asarray(means),
                raw_means=np.asarray(raw_means), covariance=covariance,
                deltas=deltas, scale=scale, pair_support=pair_support)


def conditional_order(deltas, covariance, budget, ridge=1.0, diagonal=False, initial=()):
    """Return ordered indices and trace; budget INCLUDES compulsory initial genes.

    Maximize sum_ab log(1 + delta_ab,S.T (cov_SS + ridge I)^-1 delta_ab,S).
    Covariance must be symmetric positive semidefinite. Positive Schur complements
    are checked during search; no O(P^3) full eigendecomposition is performed.
    """
    if np.iscomplexobj(deltas) or np.iscomplexobj(covariance):
        raise ValueError("deltas and covariance must contain real values")
    delta = np.asarray(deltas, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    if delta.ndim != 2 or min(delta.shape) == 0 or not np.isfinite(delta).all():
        raise ValueError("deltas must be a nonempty finite 2D matrix")
    p = delta.shape[1]
    budget = _integer(budget, "budget")
    ridge = _positive(ridge, "ridge")
    if not isinstance(diagonal, (bool, np.bool_)):
        raise ValueError("diagonal must be a boolean")
    initial = [_integer(j, "initial index") for j in initial]
    if len(set(initial)) != len(initial) or any(j >= p for j in initial):
        raise ValueError("Invalid compulsory gene indices")
    if not len(initial) <= budget <= p:
        raise ValueError("Require len(initial) <= budget <= n_genes")
    if cov.shape != (p, p) or not np.isfinite(cov).all():
        raise ValueError("Invalid covariance shape or nonfinite entries")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    C = np.diag(np.diag(cov)) if diagonal else cov.copy()
    C.flat[::p + 1] += ridge
    residual = delta.copy()
    variance = np.diag(C).copy()
    factors = np.zeros((p, budget), dtype=float)
    distances = np.zeros(delta.shape[0], dtype=float)
    available = np.ones(p, dtype=bool)
    selected, trace = [], []
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for step in range(budget):
            if not np.isfinite(variance[available]).all() or np.any(variance[available] <= 0):
                raise FloatingPointError("Nonpositive/nonfinite Schur complement; inspect covariance/ridge")
            increments = residual ** 2 / np.maximum(variance, 1e-15)[None, :]
            gains = np.log1p(increments / (1 + distances[:, None])).sum(axis=0)
            if not np.isfinite(gains[available]).all():
                raise FloatingPointError("Nonfinite candidate gains; inspect inputs")
            gains[~available] = -np.inf
            gene = initial[step] if step < len(initial) else int(np.argmax(gains))
            denom = np.sqrt(variance[gene])
            column = C[:, gene] - factors[:, :step] @ factors[gene, :step]
            column /= denom
            z = residual[:, gene].copy() / denom
            old_objective = np.log1p(distances).sum()
            distances += z * z
            residual -= z[:, None] * column[None, :]
            factors[:, step] = column
            variance -= column * column
            available[gene] = False
            selected.append(gene)
            trace.append(dict(step=step + 1, index=gene, compulsory=step < len(initial),
                              objective=float(np.log1p(distances).sum()),
                              gain=float(np.log1p(distances).sum() - old_objective),
                              pair_increments=(z * z).tolist()))
    return selected, trace
