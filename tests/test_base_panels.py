"""Frozen BP identities, ordered genes, resource integrity and public loading."""

from collections import Counter
from dataclasses import FrozenInstanceError
import hashlib
from importlib.resources import files
import json
import shutil

import numpy as np
import pytest

from panelaugment import PanelAugmenter, log_normalize
from panelaugment.base_panels import _read_panels, list_base_panels, load_base_panel


def test_registry_matches_the_current_24_panel_grid_and_known_identities():
    panels = list_base_panels()
    assert [p.panel_id for p in panels] == [f"BP-{11010500 + i}" for i in range(1, 25)]
    assert Counter(p.dataset for p in panels) == {"dlpfc": 9, "caudate": 6, "merfish": 9}
    assert sum(p.n_genes for p in panels) == 6960
    assert len({(p.dataset, p.n_genes, p.construction) for p in panels}) == 24
    assert len({p.ordered_genes_sha256 for p in panels}) == 24
    assert load_base_panel("BP-11010501").ordered_genes_sha256 == "1f0c24da71d07bc30d772bfea226400c5e949cc487af8495d9ab8a117e1be306"
    assert load_base_panel("BP-11010510").ordered_genes_sha256 == "1bba4f1e51d9d577f44ca97c797824b37416587ecffd5d89843473fe8c2e896a"
    assert load_base_panel("BP-11010516").ordered_genes_sha256 == "d71e4ba24492111b57ba044c7993567ef227298ee923be4455310b949d649e8e"
    assert load_base_panel("BP-11010501").genes[:3] == ("ENSG00000211592", "ENSG00000110484", "ENSG00000197971")
    assert all("MALAT1" not in p.genes for p in list_base_panels(dataset="merfish"))


def test_registry_filters_seed_semantics_and_unknown_ids():
    assert len(list_base_panels(dataset="DLPFC", n_genes=140)) == 3
    assert len(list_base_panels(dataset="cah")) == 6
    assert list_base_panels(dataset="caudate", n_genes=500) == ()
    assert [p.panel_id for p in list_base_panels(dataset="merfish", n_genes=300, construction="ReconST")] == ["BP-11010519"]
    for p in list_base_panels(construction="spapros"):
        assert p.constructor_seed_requested == 2026
        assert p.constructor_seed_effective == 0
    for p in list_base_panels(construction="wilcoxon"):
        assert p.constructor_seed_requested is None and p.constructor_seed_effective is None
    for value in ("BP-11010525", "../method.md", "bp-11010501", None):
        with pytest.raises(ValueError):
            load_base_panel(value)
    for kwargs in ({"dataset": "starmap"}, {"construction": "PA"}, {"n_genes": True}, {"n_genes": 1.5}, {"n_genes": 0}):
        with pytest.raises(ValueError):
            list_base_panels(**kwargs)


def test_metadata_and_gene_records_are_independent():
    panel = load_base_panel("BP-11010510")
    assert isinstance(panel.genes, tuple)
    with pytest.raises(FrozenInstanceError):
        panel.dataset = "changed"
    record = panel.metadata
    record["dataset"] = "changed"
    assert panel.metadata["dataset"] == "caudate"
    assert json.loads(json.dumps(panel.metadata))["n_genes"] == 140


@pytest.mark.parametrize("damage,expected", [
    ("bytes", "file checksum"),
    ("reorder", "ordered gene checksum"),
    ("ranks", "invalid ordered gene list"),
    ("path", "unexpected gene resource"),
])
def test_loading_rejects_corrupt_gene_resources(tmp_path, damage, expected):
    source = files("panelaugment").joinpath("resources", "base_panels")
    root = tmp_path / "base_panels"
    shutil.copytree(str(source), root)
    registry_file = root / "base_panel_registry.json"
    registry = json.loads(registry_file.read_text())
    record = registry["panels"][0]
    gene_file = root / record["gene_file"]
    lines = gene_file.read_text().splitlines()
    if damage == "bytes":
        gene_file.write_text("changed\n")
    elif damage == "path":
        record["gene_file"] = "../outside.tsv"
    elif damage == "reorder":
        first, second = lines[1].split("\t")[1], lines[2].split("\t")[1]
        lines[1], lines[2] = f"1\t{second}", f"2\t{first}"
        gene_file.write_text("\n".join(lines) + "\n")
        record["gene_file_sha256"] = hashlib.sha256(gene_file.read_bytes()).hexdigest()
    else:
        lines[1] = lines[1].replace("1\t", "2\t", 1)
        gene_file.write_text("\n".join(lines) + "\n")
        record["gene_file_sha256"] = hashlib.sha256(gene_file.read_bytes()).hexdigest()
    registry_file.write_text(json.dumps(registry))
    with pytest.raises(RuntimeError, match=expected):
        _read_panels(root)


def test_loaded_base_order_survives_a_different_candidate_axis():
    panel = load_base_panel("BP-11010510")
    genes = list(reversed(panel.genes)) + ["synthetic_extra_A", "synthetic_extra_B"]
    rng = np.random.default_rng(18)
    X = log_normalize(rng.poisson(3, (24, len(genes))))
    labels = np.tile(np.repeat(["A", "B"], 6), 2)
    groups = np.repeat(["group1", "group2"], 12)
    fitted = PanelAugmenter().fit(X, labels, groups, gene_names=genes)
    selected = fitted.select(panel.genes, n_added=1)
    assert tuple(selected.base_genes) == panel.genes
    assert tuple(selected.genes[:panel.n_genes]) == panel.genes
    assert selected.added_genes[0] in {"synthetic_extra_A", "synthetic_extra_B"}
