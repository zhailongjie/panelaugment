# Python and command-line usage

PanelAugment extends an ordered base panel by a specified number of genes.
The selector receives normalized training expression; PA preprocessing prepares
raw counts through normalization, training-fitted candidate QC, and integrated
measurement-consistency gating.

See the [method](method.md) for the objective and numerical updates,
[preprocessing](preprocessing.md) for dataset-specific settings, and
[input data and reproducible runs](reproducibility.md) for data roles and records.

## Installation and offline examples

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install .
panelaugment demo --out demo-output
```

The demo generates synthetic data locally and writes a selected panel. For a
complete synthetic example including candidate QC, measurement gating, selection,
and transformation of held-out cells:

```bash
python examples/preprocessing_demo.py
```

The core requires NumPy. Install `python -m pip install '.[sparse]'` for sparse
Python inputs, or `python -m pip install '.[h5ad]'` for AnnData/H5AD input.
These commands install the local checkout. Synthetic examples demonstrate usage;
their outputs are not biological performance estimates.

## Input contract

| Input | Required structure |
| --- | --- |
| `X` or counts | Cells × genes; finite real values, with nonnegative values for raw/count-like input |
| `y` | One nonempty class label per cell |
| `groups` | One nonempty training-group identifier per cell |
| `genes` | Unique, nonempty gene identifiers in matrix-column order |
| Base panel | An ordered list of unique genes present in the candidate axis |
| `n_added` | Number of new genes, excluding the base; a nonnegative integer |

Groups represent the biological or sample units defined by the study. At least
two classes are required, and every class pair must share at least one training
group. Missing group × class combinations are allowed; the software does not
fill them with synthetic observations. Gene identifiers are case-sensitive and
are not automatically converted between symbols and Ensembl IDs.

The NPZ reader requires `X`, `y`, `groups`, and `genes`. Expression must be a
dense real numeric array. Store metadata as NumPy Unicode or UTF-8 byte arrays,
not object arrays requiring pickle. The Python API and H5AD reader also accept
sparse expression when the corresponding dependency is installed.

## Generate runnable example inputs

This block creates synthetic full-library counts, a candidate-only version,
normalized expression, an ordered base, a training mask, and donor measurement
aggregates. Run it in a working directory where `usage-inputs` does not yet exist.
The following examples use these files and Python variables.

```python
from pathlib import Path
import numpy as np
from panelaugment import log_normalize

root = Path("usage-inputs")
root.mkdir(exist_ok=False)
rng = np.random.default_rng(2026)
y = np.tile(np.repeat(["type_a", "type_b", "type_c"], 10), 3)
groups = np.repeat(["donor_1", "donor_2", "donor_3"], 30)
genes = np.array([f"gene_{j}" for j in range(30)])
rates = np.ones((len(y), len(genes)))
for i, label in enumerate(["type_a", "type_b", "type_c"]):
    rates[y == label, i * 5:(i + 1) * 5] = 5.0
counts = rng.poisson(rates)
totals = counts.sum(axis=1)  # Compute before restricting the gene axis.
X = log_normalize(counts)
base_genes = ["gene_0", "gene_5"]
fit_mask = groups != "donor_3"

np.savez(root / "training.npz", X=X, y=y, groups=groups, genes=genes)
np.savez(root / "training-counts.npz", X=counts, y=y, groups=groups, genes=genes)
np.savez(root / "candidate-counts.npz", X=counts[:, :20],
         y=y, groups=groups, genes=genes[:20])
np.save(root / "totals.npy", totals)
np.save(root / "training-mask.npy", fit_mask)
(root / "base.txt").write_text("\n".join(base_genes) + "\n", encoding="utf-8")

# Four synthetic independent measurement donors; gene_19 triggers the gate.
positive_counts = np.full((4, len(genes)), 50, dtype=int)
positive_counts[:, 19] = [4, 4, 91, 91]
cell_counts = np.full(4, 100, dtype=int)
np.savez(root / "donor-detection.npz", positive_counts=positive_counts,
         cell_counts=cell_counts, genes=genes)
```

The direct-selector examples below treat all rows as selection-training input.
The preprocessing examples instead use `fit_mask` to reserve `donor_3` as
held-out reference rows. In an actual analysis, establish these roles before
selection and use the same intended roles throughout the workflow.

## Python selection API

`PanelAugmenter.fit` accepts already normalized expression. It does not
automatically normalize counts or run candidate preprocessing.

```python
from panelaugment import PanelAugmenter

model = PanelAugmenter(ridge=1.0).fit(X, y, groups, gene_names=genes)
result = model.select(base_genes=base_genes, n_added=5)

print(result.genes)        # Ordered base, followed by five additions.
print(result.added_genes)
smaller = result.prefix(n_added=2)
record = result.to_dict()
```

For a single selection, the convenience function exposes the same operation:

```python
from panelaugment import augment_panel

result = augment_panel(
    X, y, groups, genes,
    base_genes=base_genes, n_added=5, ridge=1.0,
)
```

The reusable estimator computes training statistics once. Call `select` again
for another base or addition budget. `n_added=0` retains only the base; an empty
base allows selection from an empty starting panel. A base must fit within the
candidate axis, and the requested total size cannot exceed that axis.

Use `diagonal=True` in `select` or `augment_panel` for the diagonal-covariance
ablation. The default ridge is `1.0` and must be positive. `prefix(n_added=...)`
reuses a shorter prefix of the **same** fitted search, ordered base, and
parameters. It cannot exceed the original addition budget or replace refitting
within a validation fold. A different base needs a new search.

`SelectionResult` provides `genes`, `base_genes`, `added_genes`, `indices`,
`trace`, `pair_support`, `ridge`, and `method`. `indices` are zero-based positions
on the candidate axis supplied to `fit`. `to_dict()` returns an independent,
JSON-compatible record. The trace includes compulsory base-incorporation steps.

The lower-level functions `balanced_weights`, `training_moments`, and
`conditional_order` are also exported. The latter's `budget` means **total
panel size**, including its `initial` indices. The high-level API uses
`n_added` to distinguish the addition budget.

## Normalization choices

For selection from raw counts, use full-library CP10K followed by `log1p` before
restricting to candidate genes. When only candidates are supplied, preserve
totals measured over the original full library:

```python
candidate_X = log_normalize(counts[:, :20], library_totals=totals)
```

If `library_totals` is omitted, the supplied matrix's row sums define the
denominator. Use that mode only when those columns constitute the intended
full normalization universe. Candidate-only counts cannot recover omitted
library counts. Zero-total rows remain zero under the `max(total, 1)` convention.
`log_normalize` accepts `target_sum`, with default `10000.0`.

For the `select` command, choose exactly one normalization mode:

| Flag | Input meaning |
| --- | --- |
| `--normalized` | Expression already has the intended selection normalization |
| `--full-library` | Raw/count-like columns form the complete normalization universe |
| `--library-totals totals.npy` | Raw/count-like columns use external full-library totals, one per row |
| `--library-total-key full_library_total` | H5AD counts use totals from the named `adata.obs` column |

Do not normalize an already normalized layer again. `select` can normalize
counts through these flags, but candidate QC and measurement gating run through
`CandidatePreprocessor` or the `preprocess` command. Use the preprocessing
workflow when those steps are needed.

## PA preprocessing in Python

`CandidatePreprocessor` first normalizes the full input universe, then fits
ordinary candidate QC only on rows selected by a boolean `fit_mask`. Defaults
require detection in at least 20 training cells and unweighted population
variance strictly above `1e-10`. Measurement gating is integrated into this
preprocessing pipeline and runs automatically when donor aggregates are supplied.

```python
from panelaugment import CandidatePreprocessor

prep = CandidatePreprocessor(min_detected=20, min_variance=1e-10)
prepared_X = prep.fit_transform(
    counts, genes, fit_mask=fit_mask, library_totals=totals,
    mandatory_genes=base_genes,
    measurement_positive_counts=positive_counts,
    measurement_cell_counts=cell_counts,
    measurement_genes=genes,
)
model = PanelAugmenter().fit(
    prepared_X[fit_mask], y[fit_mask], groups[fit_mask], gene_names=prep.genes_,
)
result = model.select(base_genes=base_genes, n_added=5)

held_out_X = prep.transform(
    counts[~fit_mask], genes, library_totals=totals[~fit_mask],
)
held_out_panel_X = held_out_X[:, result.indices]
print(prep.to_dict()["measurement_screen"]["status"])
```

`fit_transform` returns **all input rows** on the fitted candidate axis; select
the training rows before fitting PA moments. `transform` reuses the candidate
decision without fitting QC on new cells. It requires the complete fitted input
gene set, permits a different column order, and retains the denominator policy
used during fitting. Missing genes are errors and are never filled with zeros.
The transformed values remain in selection normalization; classifier
preprocessing, when used, has its own denominator and training-scale definition.

For entirely training-only input, use `np.ones(len(y), dtype=bool)` as the mask.
`mandatory_genes` bypass ordinary QC, while `excluded_genes` remove explicitly
listed genes. A gene cannot be both mandatory and excluded. `n_top_genes`
enables descending-variance ranking with stable input-order ties; otherwise
input order is retained. Mandatory retention and a top-N cap cannot be combined.
Use the cap appropriate to the data protocol rather than assuming every study
uses the same candidate count.

### Measurement inputs and gate states

The gate uses integer `positive_counts` with shape donors × genes, integer
`cell_counts` with one positive count per independent donor, and a unique gene
axis. Every detection count must lie between zero and that donor's cell count.
The aggregate axis must cover all pre-gate candidates and may be reordered or
contain extra genes. Python arguments are `measurement_positive_counts`,
`measurement_cell_counts`, and `measurement_genes`; the first two must either
both be supplied or both be omitted. The gene axis defaults to the count-matrix
axis when `measurement_genes` is omitted.

A candidate is removed when detection is strictly below 5% in at least two
donors and strictly above 90% in at least two other donors. Exactly 5% and 90%
do not satisfy these conditions. Use independent biological donors; sections
from the same donor are not independent donors. No gene name or class label
determines a gate hit.

| Status | Meaning |
| --- | --- |
| `evaluable` | At least four donors supplied; inspect the hit mask and excluded genes |
| `not_applicable` | Fewer than four donors; no negative/pass decision is made |
| `not_evaluated` | Measurement aggregates were not supplied |

Hits are removed after candidate QC and any top-N ranking without replacing
them. A hit in a mandatory base raises an error, requiring an explicit base
decision. The gate preserves the full-library normalization denominator.

Measurement gating uses the supplied donor observations, including target
observations when specified by the data protocol. Record their source and role.
The absence of target class labels does not make a target-informed candidate
axis independent of those target data.

## Command-line preprocessing and selection

Using the generated inputs, run candidate preprocessing with the declared
training mask and measurement aggregates, then select from its normalized output:

```bash
panelaugment preprocess --input usage-inputs/training-counts.npz \
  --full-library --fit-mask usage-inputs/training-mask.npy \
  --mandatory-genes usage-inputs/base.txt \
  --measurement-counts usage-inputs/donor-detection.npz \
  --out prepared-output
panelaugment select --input prepared-output/training.npz --normalized \
  --base usage-inputs/base.txt --n-added 5 --out prepared-selection-output
```

`preprocess` exports only the training rows, with their aligned labels, groups,
and selected genes. Use `--all-training` instead of `--fit-mask` when every row
belongs to the selection-training role. For a candidate-only raw matrix, use
`--library-totals usage-inputs/totals.npy` instead of `--full-library`.

Additional options are `--min-detected`, `--min-variance`, `--target-sum`,
`--top-genes`, and `--exclude-genes`. `--mandatory-base-id` retains a bundled
base through ordinary QC and can be repeated to retain a union. It has the same
measurement-gate conflict behavior as `--mandatory-genes`.

To inspect the same gate calculation without preparing an expression matrix:

```bash
panelaugment screen --input usage-inputs/donor-detection.npz --out screen-output
```

This writes `screen.json` and `excluded_genes.txt`. The `screen` command itself
does not modify expression or run selection. The `preprocess` command applies
gate hits automatically when `--measurement-counts` is supplied.

### Direct NPZ selection

If the input already contains the intended training rows and candidate axis,
select directly from normalized expression:

```bash
panelaugment select --input usage-inputs/training.npz \
  --base usage-inputs/base.txt --n-added 5 --normalized --out normalized-output
```

For candidate-only raw counts with the full-library denominator saved before
subsetting:

```bash
panelaugment select --input usage-inputs/candidate-counts.npz \
  --base usage-inputs/base.txt --n-added 5 \
  --library-totals usage-inputs/totals.npy --out candidate-counts-output
```

For the complete raw normalization universe:

```bash
panelaugment select --input usage-inputs/training-counts.npz \
  --base usage-inputs/base.txt --n-added 5 --full-library --out full-counts-output
```

A UTF-8 text base has one gene per line. A JSON list of strings is also accepted;
blank text lines are ignored. An empty file or JSON list selects from an empty
base. Existing output directories are rejected, so choose a new path for each run.

### H5AD input

After installing the H5AD extra, the following block creates a file from the
synthetic variables above, with explicit expression layers and observation keys:

```python
import anndata as ad
import pandas as pd

adata = ad.AnnData(
    X=counts.copy(),
    obs=pd.DataFrame({"cell_type": y, "donor": groups,
                      "full_library_total": totals},
                     index=[f"cell_{i}" for i in range(len(y))]),
    var=pd.DataFrame(index=genes),
)
adata.layers["counts"] = counts.copy()
adata.layers["log1p_full_library"] = X.copy()
adata.write_h5ad("usage-inputs/training.h5ad")
```

Choose label and group keys explicitly when selecting from a normalized layer:

```bash
panelaugment select --input usage-inputs/training.h5ad \
  --label-key cell_type --group-key donor --layer log1p_full_library \
  --normalized --base usage-inputs/base.txt --n-added 5 --out h5ad-output
```

For raw counts with totals stored in `adata.obs`, choose
`--library-total-key full_library_total` instead of `--normalized`. H5AD input
also works with integrated preprocessing:

```bash
panelaugment preprocess --input usage-inputs/training.h5ad \
  --label-key cell_type --group-key donor --layer counts \
  --library-total-key full_library_total \
  --fit-mask usage-inputs/training-mask.npy --mandatory-genes usage-inputs/base.txt \
  --measurement-counts usage-inputs/donor-detection.npz --out h5ad-prepared-output
```

Omit `--layer` to read `adata.X`. The software does not infer normalization from
a layer name. Cell and gene identifiers must be unique, and label/group columns
must have no missing values. NPZ input does not accept H5AD observation or layer
arguments. Use `panelaugment select --help` and `panelaugment preprocess --help`
for the full argument lists.

## Bundled bases

```bash
panelaugment panels list
panelaugment panels list --dataset merfish --n-genes 500 --format json
panelaugment panels export --id BP-11010522 --out BP-11010522.txt
```

```python
from panelaugment import load_base_panel, list_base_panels

base = load_base_panel("BP-11010522")
print(base.dataset, base.n_genes)
print(base.genes)     # Immutable tuple in the supplied base order.
print(base.metadata)
```

With compatible expression, `select --base-id BP-11010522` replaces
`--base base.txt`; supply exactly one of the two. The input must contain every
base gene with matching identifiers. These real identifiers do not match the
synthetic `gene_0`-style inputs above. The selected registry metadata is recorded
in `run.json`. The loader verifies file and ordered-gene checksums. See the
[BP catalog](base-panels.md) for all IDs and available list filters.

## Outputs and interpretation

| Selection file | Contents |
| --- | --- |
| `genes.csv` | `rank` (one-based), `gene`, `candidate_index` (zero-based), and `role` (`base` or `added`) |
| `panels.json` | Ordered genes, base/additions, candidate indices, budget, method, ridge, objective, trace, and pair support |
| `trace.json` | Step, selected index, compulsory-base flag, objective, gain, and per-pair increments |
| `pair_support.json` | Shared training groups and group count for each class pair |
| `run.json` | Input/axis checksums, dimensions, configuration, normalization, versions, and timings |

The CLI saves these files automatically. Python selection returns a
`SelectionResult` without writing files. Compulsory base steps in the trace are
not greedy choices. `pair_support.json` can contain supplied group identifiers;
handle it according to the input data's access conditions.

Preprocessing writes `training.npz`, `genes.txt`, and `preprocessing.json`.
The record contains the training-row count, thresholds, candidate axis,
denominator policy, input checksums, and the gate result under
`preprocessing.measurement_screen`. Keep both preprocessing and selection
records when chaining the two commands.

PA maximizes a training discrimination objective. It does not optimize probe
chemistry, discover unknown classes, or directly minimize classifier error.
The log objective discounts further gains for class pairs that are already
well separated. A larger set has a nondecreasing mathematical objective for
fixed statistics; this alone does not establish better downstream accuracy,
structure preservation, general submodularity, or a global optimum.

Labels and training-group identities must come from the permitted
selection-training role. Fix bases, candidates, normalization, and budgets
without consulting external evaluation scores. If cross-validation chooses a
budget, fit training-dependent QC and PA statistics and select panels within
each training fold. An already selected full-development panel cannot stand in
for independent fold-level selections. Report the separate data dependency of
measurement gating when target observations supply its aggregates.

The implementation estimates a dense gene × gene covariance matrix. Sparse
input is densified for PA moment estimation, so sparse storage does not imply
a sparse covariance algorithm. Large candidate pools can require substantial
memory and compute. Selection-space normalization, panel-total classifier
normalization, and common-candidate structure normalization use different
denominators; the toolkit's selector does not implement those downstream
evaluation pipelines. See [method details and complexity](method.md).
