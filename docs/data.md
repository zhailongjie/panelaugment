# Data sources

The benchmark combines reference expression with spatial measurements. The
[base-panel catalogue](base-panels.md) provides the ordered starting gene lists;
the source datasets are available from the providers below.

## DLPFC

Training and development use the HumanPilot Visium data from
[Maynard et al. (2021)](https://doi.org/10.1038/s41593-020-00787-0).
The training sections are `151507–151510` and `151669–151672`; development
uses `151673–151676`. Obtain the processed object with
`spatialLIBD::fetch_data(type = "spe")` through
[spatialLIBD / ExperimentHub](https://research.libd.org/spatialLIBD/).
The [raw-data page](https://research.libd.org/spatialLIBD/#raw-data) provides
per-section downloads and the Globus collection `jhpce#HumanPilot10x`.

External spatial evaluation uses the separate
[spatialDLPFC collection](https://github.com/LieberInstitute/spatialDLPFC#r-objects)
from [Huuki-Myers et al. (2024)](https://doi.org/10.1126/science.adh1938):
`Br6522_ant`, `Br6522_mid` and `Br8667_post`. Obtain its official processed
object with `spatialLIBD::fetch_data(type = "spatialDLPFC_Visium")`.
Its raw-data archive is [NDA Study 2619](https://doi.org/10.15154/7893-6778).

## Human caudate

The [SEA-AD data portal](https://brain-map.org/consortia/sea-ad/our-data) provides
the caudate reference and spatial measurements. The single-nucleus reference is
[CaH_all_nuclei.2025-12-29.h5ad](https://sea-ad-single-cell-profiling.s3.amazonaws.com/Caudate_Nucleus/CaH_all_nuclei.2025-12-29.h5ad);
the spatial target is the Xenium object
[CaH_Xenium.2026-01-07.h5ad](https://sea-ad-spatial-transcriptomics.s3.amazonaws.com/caudate_nucleus/combined_anndata_object/CaH_Xenium.2026-01-07.h5ad).

## Cortical MERFISH

The reference is the Allen Institute's
[Human MTG SMART-seq](https://brain-map.org/our-research/cell-types-taxonomies/cell-types-database-rna-seq-data/human-mtg-smart-seq)
dataset: `human_MTG_gene_expression_matrices_2018-06-14.zip`,
[download ID 694416044](https://celltypes.brain-map.org/api/v2/well_known_file_download/694416044).
Selection uses gene-level exonic read counts from this archive.
Reference publication: [Hodge et al., Nature (2019)](https://doi.org/10.1038/s41586-019-1506-7).

The spatial target uses the human MTG/STG 4,000-gene MERFISH files from
[Fang et al. (2022), Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.x3ffbg7mw),
DOI `10.5061/dryad.x3ffbg7mw`, also available in
[Zenodo record 6819663](https://zenodo.org/records/6819663).
These spatial measurements are distinct from the SMART-seq reference.

## Attribution

Cite the original datasets and their associated publications when using these
resources. Allen Institute data remain subject to its
[Terms of Use](https://alleninstitute.org/legal/terms-of-use) and
[Citation Policy](https://alleninstitute.org/legal/citation-policy); consult each
source repository for its data terms.
