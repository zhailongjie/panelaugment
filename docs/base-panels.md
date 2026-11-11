# Numbered benchmark base panels

The package contains 24 fixed starting panels, `BP-11010501`–`BP-11010524`.
Each identifier denotes an ordered starting gene list. Download an individual
list from the table or load it by ID with the Python API.

There are nine DLPFC panels, six human caudate panels and nine cortical MERFISH
panels. Applying the six addition budgets (10, 20, 30, 40, 50 and 100) gives
144 positive-budget conditions.

See [data sources](data.md) for the reference-expression and spatial datasets.

| Dataset | Base panel IDs (TSV) | Genes |
| --- | --- | ---: |
| DLPFC | [BP-11010501](../src/panelaugment/resources/base_panels/genes/BP-11010501.tsv), [BP-11010502](../src/panelaugment/resources/base_panels/genes/BP-11010502.tsv), [BP-11010503](../src/panelaugment/resources/base_panels/genes/BP-11010503.tsv) | 140 |
| DLPFC | [BP-11010504](../src/panelaugment/resources/base_panels/genes/BP-11010504.tsv), [BP-11010505](../src/panelaugment/resources/base_panels/genes/BP-11010505.tsv), [BP-11010506](../src/panelaugment/resources/base_panels/genes/BP-11010506.tsv) | 300 |
| DLPFC | [BP-11010507](../src/panelaugment/resources/base_panels/genes/BP-11010507.tsv), [BP-11010508](../src/panelaugment/resources/base_panels/genes/BP-11010508.tsv), [BP-11010509](../src/panelaugment/resources/base_panels/genes/BP-11010509.tsv) | 500 |
| Human caudate | [BP-11010510](../src/panelaugment/resources/base_panels/genes/BP-11010510.tsv), [BP-11010511](../src/panelaugment/resources/base_panels/genes/BP-11010511.tsv), [BP-11010512](../src/panelaugment/resources/base_panels/genes/BP-11010512.tsv) | 140 |
| Human caudate | [BP-11010513](../src/panelaugment/resources/base_panels/genes/BP-11010513.tsv), [BP-11010514](../src/panelaugment/resources/base_panels/genes/BP-11010514.tsv), [BP-11010515](../src/panelaugment/resources/base_panels/genes/BP-11010515.tsv) | 300 |
| Cortical MERFISH | [BP-11010516](../src/panelaugment/resources/base_panels/genes/BP-11010516.tsv), [BP-11010517](../src/panelaugment/resources/base_panels/genes/BP-11010517.tsv), [BP-11010518](../src/panelaugment/resources/base_panels/genes/BP-11010518.tsv) | 140 |
| Cortical MERFISH | [BP-11010519](../src/panelaugment/resources/base_panels/genes/BP-11010519.tsv), [BP-11010520](../src/panelaugment/resources/base_panels/genes/BP-11010520.tsv), [BP-11010521](../src/panelaugment/resources/base_panels/genes/BP-11010521.tsv) | 300 |
| Cortical MERFISH | [BP-11010522](../src/panelaugment/resources/base_panels/genes/BP-11010522.tsv), [BP-11010523](../src/panelaugment/resources/base_panels/genes/BP-11010523.tsv), [BP-11010524](../src/panelaugment/resources/base_panels/genes/BP-11010524.tsv) | 500 |

## Data files

All resources are installed with the Python package under
[`src/panelaugment/resources/base_panels/`](../src/panelaugment/resources/base_panels/).
The directory contains:

- [`base_panel_registry.csv`](../src/panelaugment/resources/base_panels/base_panel_registry.csv)
  and [`base_panel_registry.json`](../src/panelaugment/resources/base_panels/base_panel_registry.json):
  mappings from BP ID to dataset, gene count, constructor, original experiment
  panel ID, source identifier, requested/effective constructor seeds and hashes.
- [`genes/`](../src/panelaugment/resources/base_panels/genes/): one TSV per BP ID,
  with `gene_index` (one-based) and `gene`. Row order is the compulsory base order.
- [`constructor_settings.json`](../src/panelaugment/resources/base_panels/constructor_settings.json):
  construction recipes and parameters.

DLPFC lists use the frozen Ensembl gene IDs; caudate and MERFISH lists use gene
symbols. The loader preserves these strings exactly. Gene names are not converted,
sorted, replaced or dropped when loading a panel. When applying a bundled panel to
new expression, its identifiers must be present on the input candidate axis.

The nine MERFISH panels use the measurement-filtered candidate universe and
contain no `MALAT1`.

The 24 TSV files contain 6,960 gene entries in their required base order.
Ordered-list hashes use SHA256 of a compact UTF-8 JSON array with
`ensure_ascii=False`. The independent TSV byte hash also detects changes to ranks,
line endings or file contents. Source run identifiers and source-file hashes
identify each list's source gene file. `registry_version` identifies the
distributed data format.

## Read and use a panel

```python
from panelaugment.base_panels import list_base_panels, load_base_panel

panel = load_base_panel("BP-11010510")
print(panel.dataset, panel.n_genes, panel.panel_id)
# caudate 140 BP-11010510
print(panel.genes[:3])
# ('XIST', 'CLU', 'CAMK2D')
metadata = panel.metadata  # JSON-compatible provenance dictionary.

for candidate in list_base_panels(dataset="merfish", n_genes=300):
    print(candidate.dataset, candidate.n_genes, candidate.panel_id)

# `model` is a PanelAugmenter fitted on appropriate, already normalized
# selection-training expression with a compatible gene axis.
result = model.select(base_genes=panel.genes, n_added=20)
assert tuple(result.genes[:panel.n_genes]) == panel.genes
```

`BasePanel` is immutable and `genes` is a tuple. `list_base_panels` returns a tuple
of panels in BP-ID order. Dataset filters accept `dlpfc`, `caudate` (alias `cah`)
and `merfish`, case-insensitively; `n_genes` filters by base size. Unknown names
or IDs raise `ValueError`.
The loader verifies each bundled gene file and its ordered-list hash before
returning panel records.

Export a plain ordered list for other software without the TSV header:

```python
from pathlib import Path
from panelaugment.base_panels import load_base_panel

panel = load_base_panel("BP-11010516")
with Path("BP-11010516.txt").open("x", encoding="utf-8", newline="\n") as handle:
    handle.write("\n".join(panel.genes) + "\n")
```

After installing the package, run the self-contained reading and selection
example from the repository root:

```bash
python examples/bundled_base_demo.py --panel-id BP-11010510
```

This example uses the numbered gene identifiers with synthetic counts to
demonstrate loading, selection and ordered-base retention.

The same lists can be inspected and exported from the command line:

```bash
panelaugment panels list --dataset merfish --n-genes 300
panelaugment panels export --id BP-11010516 --out BP-11010516.txt
```

## Construction settings

ReconST uses a fixed epoch-20 gated autoencoder ranking with seed 2026, hidden
size 200, dropout 0.05, `lk=0.3`, learning rate 0.001, weight decay 0.00001,
gate penalty 0.0001 and batch size 256. Stable positive-gate ordering supplies the
different sizes. Loss monitoring does not choose a checkpoint or stopping epoch.

Spapros uses an independent cold `ProbesetSelector` run for each size, no
preselected genes, and `n_pca_genes=max(100, ceil(1.25*n))`. The requested seed is
2026, but the implementation's effective seed is **0**, with four
jobs. The registry records both values.

Wilcoxon bases use pooled-cell one-versus-rest rank-sum z scores with average
ranks, tie correction and no continuity correction. The construction cycles
through sorted classes, taking unused positive-scoring genes from each stable
ranking. It is deterministic on the frozen candidate order and has no random
seed. These base-construction scores are distinct from the Marker augmentation
comparator.

Reconstructing a panel from expression requires its reference partitions,
eligible gene axis, construction parameters and constructor software version.
