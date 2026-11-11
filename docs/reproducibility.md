# Input data and reproducible runs

## Selection input

PA uses an expression matrix with cells in rows and genes in columns. Supply one class label and one training-group identifier per row, plus unique gene identifiers in column order. Groups represent the study's biological or sample units. At least two classes are required, with a shared training group for each class pair.

NPZ inputs contain `X`, `y`, `groups` and `genes`. Store metadata as Unicode or UTF-8 byte arrays without pickle. H5AD inputs require explicit label and group keys; use `--layer` to select a matrix and `--library-total-key` when full-library totals are stored in observation metadata.

The selector accepts normalized expression. For raw counts, PA preprocessing uses full-library CP10K/log1p. When the input contains only a measured subset, supply totals computed over the original full library before subsetting.

## Candidate preprocessing and measurement data

Declare training rows using a boolean mask, or explicitly mark the entire input as training. Detection and population-variance QC use only those rows. The fitted gene axis is reused when transforming new cells.

The integrated measurement-consistency gate uses donor-level detection aggregates: integer `positive_counts` with shape donors × genes, integer `cell_counts` with one positive count per donor, and the corresponding unique `genes` axis. This axis must cover all candidates surviving QC and can have a different order. Use independent biological donors, pooling sections from the same donor before computing detection rates.

The gate requires at least four donors. It excludes a gene when at least two donors have detection strictly below 5% and at least two other donors have detection strictly above 90%. Missing inputs are `not_evaluated`; fewer than four donors are `not_applicable`. Supply the measurement data from the intended deployment setting and record their source and role. See [preprocessing](preprocessing.md) for exact options and examples.

## Fixed bases

A base is an ordered list of unique gene identifiers contained in the candidate axis. PA preserves the complete base as the output prefix. `n_added` is the number of new genes. The bundled [BP catalog](base-panels.md) provides 24 ordered bases with dataset, construction settings and checksums.

## Outputs and run records

The preprocessing command writes `training.npz`, `genes.txt` and `preprocessing.json`. Only declared training rows are exported. The record includes candidate parameters, denominator policy, measurement-gate status, excluded genes and input checksums.

Selection writes `panels.json`, `genes.csv`, `trace.json`, `pair_support.json` and `run.json`. These record the ordered genes, base/addition roles, objective trace, shared-group support, parameters, input/axis checksums and environment. Output directories must be new.

Retain the input representation, full-library denominator, row roles, gene identifiers, ordered base, measurement inputs, budget, ridge setting and software environment with each result. Keep participant-level records according to the source data's access conditions.

## Training and evaluation roles

PA moments, candidate detection and variance statistics use the declared training rows. Refit training-dependent steps within each fold when choosing a budget by cross-validation. Fix bases, budgets and settings independently of evaluation outcomes.

Measurement gating uses the declared donor observations, including target observations when specified by the data protocol. Record this dependency when interpreting evaluation results. No class labels are needed for the gate.

Full-library selection normalization, panel-total classifier normalization and common-candidate structure normalization have different denominators. Choose the appropriate computation for each analysis.
