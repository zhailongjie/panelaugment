# Preprocessing and candidate eligibility

`CandidatePreprocessor` implements PA's data-preprocessing pipeline: reference
normalization, training-side candidate QC, and cross-donor measurement gating.
It runs the gate automatically when detection aggregates are supplied, then
passes the resulting candidate axis to the PA conditional selector.
`measurement_consistency_screen` also exposes the gate calculation directly.

## Inputs and candidate preparation

```python
import numpy as np
from panelaugment.preprocessing import CandidatePreprocessor

prep = CandidatePreprocessor(min_detected=20, min_variance=1e-10)
prep.fit(raw_counts, genes, fit_mask=training_mask,
         library_totals=full_library_totals,
         measurement_positive_counts=donor_by_gene_detection_counts,
         measurement_cell_counts=cells_per_independent_donor,
         measurement_genes=measurement_gene_axis)
training_X = prep.transform(raw_counts[training_mask], genes,
                            library_totals=full_library_totals[training_mask])
candidate_genes = prep.genes_
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `target_sum` | `10000.0` | Per-cell normalization target before `log1p` |
| `min_detected` | `20` | Minimum number of training cells with positive counts |
| `min_variance` | `1e-10` | Exclusive lower bound on normalized training variance |
| `n_top_genes` | `None` | Retain all eligible genes; an integer enables variance ranking |

The input is a cell-by-gene matrix of finite, nonnegative raw/count-like values.
Dense NumPy and sparse SciPy inputs are supported. Gene identifiers must be
unique, explicit strings: mapping and cross-platform identifier reconciliation
must happen before this step. Missing genes cause errors; they are not filled
with zeros.

`fit_mask` is required and must be a boolean vector selecting at least one
training cell. For an entirely training-only matrix, use
`np.ones(raw_counts.shape[0], dtype=bool)`. Detection counts and variance use only
these rows. Measurement gating separately uses the supplied target-donor
aggregates. Transforming held-out cells reuses the frozen candidate decision.
`fit_transform` returns **all input rows**; callers must apply the training mask
before passing its result to `PanelAugmenter.fit`. The CLI `preprocess` command
saves only the selected training rows.

The sequence is:

1. Normalize each cell to 10,000 using its full reference-library total, then
   apply `log1p`. With no `library_totals`, every input column participates in
   the denominator; the caller must supply the full normalization universe.
   When columns already represent a subset, supply totals calculated **before**
   slicing. A fitted model requires the same denominator policy on transform.
2. Count positive raw values in the training rows. Require detection in at
   least 20 cells and unweighted population variance strictly greater than
   `1e-10` in the normalized training matrix. Variance is the second moment
   minus the squared mean.
3. With `n_top_genes`, rank eligible genes by decreasing variance, with stable
   input-axis order for ties; require enough passing genes for the requested
   count. Otherwise retain input-axis order. `mandatory_genes` explicitly
   bypass QC. `excluded_genes` explicitly remove genes. Unknown identifiers and
   mandatory/excluded conflicts are errors. A top-N budget and mandatory genes
   cannot be combined.
4. Run measurement gating over the pre-gate candidate axis. A supplied
   `measurement_genes` axis may be reordered or omit genes already discarded by
   training QC, but it must cover every remaining candidate; extra genes are
   allowed. It defaults to the input gene axis. Flagged genes are automatically
   removed. A flagged mandatory gene is an error requiring an explicit
   base-panel decision. Top-N selections are not refilled after gating.
   Missing aggregates produce `not_evaluated`; fewer than four independent
   donors produce `not_applicable`. Neither status is a negative/pass decision.
5. Transform new cells by normalizing their full input universe and slicing the
   frozen ordered axis. A reordered complete input gene set is accepted; a
   shortened or expanded set is rejected.

Zero-total cells remain zero under the `log_normalize` convention
(`max(total, 1)`). This preprocessing does not automatically discard cells,
infer donor roles, select labels, perform batch correction, or fit a classifier.
`to_dict()` records thresholds, training
cell count, gene-wise QC statistics, the gate result under `measurement_screen`,
and the ordered candidate decision without including cell or donor identifiers.
Detection and support arrays must either both be supplied or both omitted.

## Dataset configurations

| Study | Training input | Normalization denominator | Candidate decision |
| --- | --- | --- | --- |
| DLPFC | 5,500 design cells, 26,060 common measured genes | Sum over the specified 26,060-gene common axis | Detection/variance QC, then stable descending-variance top 2,000 |
| Caudate | 5,568 gradient cells, 464 measured genes | Full 36,601-gene reference library totals retained before slicing | QC OR union of six base panels, in measured-axis order: 452 genes |
| MERFISH reference | 2,027 gradient cells; shared mapped axis of 3,491 genes | Full 50,281-gene exonic reference library totals | Training QC gives 3,225 genes in input-axis order; measurement gating gives 3,224 genes |

The Caudate mandatory union contains 396 distinct genes. MERFISH excludes its
independent development donor from the QC fit. Measurement gating flags
`MALAT1` among the 3,225 MERFISH candidates and no genes among the 452 Caudate
candidates. The DLPFC gate is `not_applicable` because there are two target
donors. Gate decisions depend on detection counts rather than gene names.
Exclusion preserves full reference-library denominators. MERFISH structure
evaluation uses a 3,225-gene reference universe.

Selection-space normalization is distinct from classifier preprocessing. The
experimental classifier uses panel-total CP10K/log1p followed by standardization
estimated on its training cells. PA itself uses its weighted training
scale; the candidate variance above is unweighted. This module does not fit or
substitute either classifier standardization or PA scaling.

## Cross-donor measurement gating

```python
from panelaugment.preprocessing import measurement_consistency_screen

screen = measurement_consistency_screen(
    positive_counts=donor_by_gene_detection_counts,
    cell_counts=cells_per_independent_donor,
    genes=candidate_genes,
)
# The same calculation runs inside CandidatePreprocessor.fit().
print(screen["status"], screen["excluded_genes"])
```

Supply exact integer detection aggregates with one row per independent target
donor and one column per candidate gene, plus positive cell counts for each
donor. Sections from one donor are not separate independent donors. A gene is
flagged when at least two donors satisfy `positive * 20 < cells` and at least two
other donors satisfy `positive * 10 > 9 * cells`. Thus the 5% and 90% boundaries
are strict. Python integer comparisons avoid overflow and floating-point
boundary changes. Low and high donor sets cannot overlap under these thresholds.
No training-detection condition, class-label condition, or named-gene exception
is used. With fewer than four donors, the result is `not_applicable` and
`hit_mask` is `None`; this must not be described as a negative screen.

`CandidatePreprocessor.fit()` applies the gate's exclusions automatically.
Calling `measurement_consistency_screen` alone returns a result dictionary and
does not mutate an expression matrix. The gate uses target-donor measurements;
evaluation should report this use of target information.

## Runnable example

After installing PanelAugment, run:

```bash
python examples/preprocessing_demo.py
```

The [example](../examples/preprocessing_demo.py) generates synthetic counts and
donor aggregates locally. It fits candidate QC on explicitly marked training
cells, applies measurement gating, selects an augmented panel, and transforms
held-out cells with the fitted gene axis. It requires no data download.
