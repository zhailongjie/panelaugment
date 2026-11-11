"""Load the frozen, numbered benchmark base panels without expression data."""

import csv
from dataclasses import dataclass
import hashlib
from importlib.resources import files
import io
import json
from numbers import Integral


@dataclass(frozen=True)
class BasePanel:
    """An immutable benchmark panel whose ``genes`` retain the frozen order."""

    panel_id: str
    dataset: str
    construction: str
    genes: tuple[str, ...]
    registry_version: str
    original_panel_id: str
    gene_identifier_type: str
    ordered_genes_sha256: str
    gene_file_sha256: str
    source_file_sha256: str
    source_id: str
    constructor_seed_requested: int | None
    constructor_seed_effective: int | None

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    @property
    def metadata(self) -> dict:
        """Return independent, JSON-compatible provenance without gene rows."""
        return {
            "panel_id": self.panel_id,
            "dataset": self.dataset,
            "construction": self.construction,
            "n_genes": self.n_genes,
            "registry_version": self.registry_version,
            "original_panel_id": self.original_panel_id,
            "gene_identifier_type": self.gene_identifier_type,
            "ordered_genes_sha256": self.ordered_genes_sha256,
            "gene_file_sha256": self.gene_file_sha256,
            "source_file_sha256": self.source_file_sha256,
            "source_id": self.source_id,
            "constructor_seed_requested": self.constructor_seed_requested,
            "constructor_seed_effective": self.constructor_seed_effective,
        }


def _ordered_sha256(genes):
    raw = json.dumps(genes, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_panels(root=None):
    if root is None:
        root = files("panelaugment").joinpath("resources", "base_panels")
    try:
        registry = json.loads(root.joinpath("base_panel_registry.json").read_text(encoding="utf-8"))
        if registry["schema"] != "panelaugment.base-panels.v1":
            raise ValueError("unsupported registry schema")
        records = registry["panels"]
        expected_ids = [f"BP-{11010500 + i}" for i in range(1, 25)]
        if registry["count"] != 24 or [r["panel_id"] for r in records] != expected_ids:
            raise ValueError("unexpected panel identifiers or order")
        panels = []
        for row in records:
            panel_id = row["panel_id"]
            # Do not allow registry metadata to redirect resource reads.
            if row["gene_file"] != f"genes/{panel_id}.tsv":
                raise ValueError(f"unexpected gene resource for {panel_id}")
            raw = root.joinpath("genes", f"{panel_id}.tsv").read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["gene_file_sha256"]:
                raise ValueError(f"gene file checksum mismatch for {panel_id}")
            reader = csv.DictReader(io.StringIO(raw.decode("utf-8")), delimiter="\t")
            if reader.fieldnames != ["gene_index", "gene"]:
                raise ValueError(f"invalid gene columns for {panel_id}")
            gene_rows = list(reader)
            genes = [g["gene"] for g in gene_rows]
            n = row["n_genes"]
            if (
                len(genes) != n
                or len(set(genes)) != n
                or any(not g or g != g.strip() for g in genes)
                or [int(g["gene_index"]) for g in gene_rows] != list(range(1, n + 1))
            ):
                raise ValueError(f"invalid ordered gene list for {panel_id}")
            if _ordered_sha256(genes) != row["ordered_genes_sha256"]:
                raise ValueError(f"ordered gene checksum mismatch for {panel_id}")
            if _ordered_sha256(sorted(genes)) != row["unordered_gene_set_sha256"]:
                raise ValueError(f"gene set checksum mismatch for {panel_id}")
            panels.append(BasePanel(
                panel_id=panel_id, dataset=row["dataset"], construction=row["construction"],
                genes=tuple(genes), registry_version=registry["registry_version"],
                original_panel_id=row["original_panel_id"], gene_identifier_type=row["gene_identifier_type"],
                ordered_genes_sha256=row["ordered_genes_sha256"], gene_file_sha256=row["gene_file_sha256"],
                source_file_sha256=row["source_file_sha256"], source_id=row["source_id"],
                constructor_seed_requested=row["constructor_seed_requested"],
                constructor_seed_effective=row["constructor_seed_effective"],
            ))
        return tuple(panels)
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        raise RuntimeError(f"Bundled base-panel registry is missing or invalid: {exc}") from exc


def load_base_panel(panel_id: str) -> BasePanel:
    """Load one exact BP identifier and verify its frozen ordered gene list.

    ``panel_id`` must be one of BP-11010501 through BP-11010524. These
    dataset-specific panels do not imply that every gene exists in a new input;
    the selector still checks the candidate axis when the panel is used.
    """
    if not isinstance(panel_id, str):
        raise ValueError("panel_id must be a BP identifier string")
    for panel in _read_panels():
        if panel.panel_id == panel_id:
            return panel
    raise ValueError(f"Unknown base panel {panel_id!r}; use list_base_panels() to inspect available IDs")


def list_base_panels(*, dataset=None, n_genes=None, construction=None) -> tuple[BasePanel, ...]:
    """List numbered panels, optionally filtering by dataset, size or constructor.

    Datasets are ``dlpfc``, ``caudate`` (alias ``cah``), and ``merfish``.
    Constructors are ``reconst``, ``spapros``, and ``wilcoxon``. Names are
    case-insensitive. Valid filters with no matching combination return an
    empty tuple; unknown filter values raise ``ValueError``.
    """
    if dataset is not None:
        if not isinstance(dataset, str):
            raise ValueError("dataset must be dlpfc, caudate or merfish")
        dataset = dataset.lower()
        if dataset == "cah":
            dataset = "caudate"
        if dataset not in {"dlpfc", "caudate", "merfish"}:
            raise ValueError("dataset must be dlpfc, caudate or merfish")
    if construction is not None:
        if not isinstance(construction, str) or construction.lower() not in {"reconst", "spapros", "wilcoxon"}:
            raise ValueError("construction must be reconst, spapros or wilcoxon")
        construction = construction.lower()
    if n_genes is not None and (isinstance(n_genes, bool) or not isinstance(n_genes, Integral) or n_genes < 1):
        raise ValueError("n_genes must be a positive integer")
    return tuple(p for p in _read_panels()
                 if (dataset is None or p.dataset == dataset)
                 and (n_genes is None or p.n_genes == n_genes)
                 and (construction is None or p.construction == construction))
