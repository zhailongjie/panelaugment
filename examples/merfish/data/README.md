# Example data sources

## Reference counts

`training-counts.npz` and `library-totals.npy` are a compact research subset
derived from the Allen Institute for Brain Science's **Human MTG SMART-seq**
data, `human_MTG_gene_expression_matrices_2018-06-14.zip` (download ID 694416044).
The subset preserves the PA study's reference training rows, ordered candidate
genes and original full exonic-library totals. Counts are losslessly stored as
unsigned 32-bit integers. Labels and donor groups are supplied for selection;
other cell metadata are omitted.

- [Official dataset](https://brain-map.org/our-research/cell-types-taxonomies/cell-types-database-rna-seq-data/human-mtg-smart-seq)
- [Original expression archive](https://celltypes.brain-map.org/api/v2/well_known_file_download/694416044)
- Hodge, R. D., Bakken, T. E., Miller, J. A. et al. Conserved cell types with
  divergent features in human versus mouse cortex. *Nature* **573**, 61–68
  (2019). [doi:10.1038/s41586-019-1506-7](https://doi.org/10.1038/s41586-019-1506-7).

Source data remain subject to the
[Allen Institute Terms of Use](https://alleninstitute.org/legal/terms-of-use)
and [Citation Policy](https://alleninstitute.org/legal/citation-policy).
The supplied subset is for research and other noncommercial purposes; its
inclusion does not transfer ownership or grant commercial redistribution rights.
Credit the Allen Institute for Brain Science and the source publication when
using these data.

## Measurement aggregates

`measurement-counts.npz` contains per-donor cell counts and per-gene positive
detection counts derived from the human cortical MERFISH data of Fang et al.
It covers four donors, pooling sections within donor, and includes no class
labels or cell-level expression. These target measurement aggregates provide
the measurement-consistency input to PA preprocessing.

Fang, R. (2022). *Conservation and divergence of cortical cell organization in
human and mouse revealed by MERFISH* [Dataset]. Dryad.
[doi:10.5061/dryad.x3ffbg7mw](https://doi.org/10.5061/dryad.x3ffbg7mw).
The source data are distributed through Dryad under its
[CC0 data terms](https://datadryad.org/terms). The associated article is
[Fang et al., Science (2022)](https://doi.org/10.1126/science.abm1741).

[`input.json`](input.json) gives file checksums, dimensions, training groups,
measurement donors and cell counts. These source-data terms apply to the data
files described here, separately from the PanelAugment software.
