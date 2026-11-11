"""CLI contracts for bundled bases, training-only preparation and measurement gates."""

from collections import Counter
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from panelaugment.base_panels import load_base_panel
from panelaugment.io import file_sha256


def cli(*args):
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        [source_root, *([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])]
    )
    return subprocess.run(
        [sys.executable, "-m", "panelaugment", *map(str, args)],
        capture_output=True, text=True, env=environment, timeout=30,
    )


def success(*args):
    result = cli(*args)
    assert result.returncode == 0, result.stderr
    return result


def rejected(*args):
    result = cli(*args)
    assert result.returncode == 2, result.stderr
    assert "Traceback" not in result.stderr
    return result


def write_training(path, counts, genes, *, y=None, groups=None):
    counts = np.asarray(counts)
    if y is None:
        y = np.asarray(["class_a" if i % 2 == 0 else "class_b"
                        for i in range(len(counts))])
    if groups is None:
        groups = np.repeat("training_group", len(counts))
    np.savez(path, X=counts, genes=np.asarray(genes), y=y, groups=groups)
    return path


def read_training(path):
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"X", "genes", "y", "groups"}
        return {key: archive[key] for key in archive.files}


def test_panels_list_outputs_current_registry_and_filters_as_machine_readable_data():
    result = success("panels", "list", "--format", "json")
    records = json.loads(result.stdout)
    assert [r["panel_id"] for r in records] == [f"BP-{11010500 + i}" for i in range(1, 25)]
    assert Counter(r["dataset"] for r in records) == {"dlpfc": 9, "caudate": 6, "merfish": 9}
    assert all(r["registry_version"] and len(r["ordered_genes_sha256"]) == 64 for r in records)

    filtered = success("panels", "list", "--dataset", "CaH", "--n-genes", 140,
                       "--construction", "wilcoxon")
    assert list(csv.DictReader(io.StringIO(filtered.stdout), delimiter="\t")) == [
        {"panel_id": "BP-11010512", "dataset": "caudate", "n_genes": "140",
         "construction": "wilcoxon"},
    ]
    rejected("panels", "list", "--dataset", "unregistered_dataset")


@pytest.mark.parametrize("suffix", ["txt", "json"])
def test_panels_export_preserves_order_and_refuses_to_replace_an_existing_file(tmp_path, suffix):
    panel = load_base_panel("BP-11010510")
    destination = tmp_path / "new_parent" / f"base.{suffix}"
    success("panels", "export", "--id", panel.panel_id, "--out", destination)
    raw = destination.read_bytes()
    genes = json.loads(raw) if suffix == "json" else raw.decode("utf-8").splitlines()
    assert genes == list(panel.genes)
    assert b"\r" not in raw
    rejected("panels", "export", "--id", "BP-11010501", "--out", destination)
    assert destination.read_bytes() == raw


def test_select_by_bp_id_records_registry_and_uses_base_order_not_input_axis_order(tmp_path):
    panel = load_base_panel("BP-11010510")
    genes = ["synthetic_extra", *reversed(panel.genes)]
    counts = np.random.default_rng(31).uniform(0, 3, (12, len(genes)))
    source = write_training(tmp_path / "training.npz", counts, genes)
    destination = tmp_path / "selected"
    success("select", "--input", source, "--base-id", panel.panel_id,
            "--normalized", "--n-added", 1, "--out", destination)

    result = json.loads((destination / "panels.json").read_text(encoding="utf-8"))
    assert result["genes"] == [*panel.genes, "synthetic_extra"]
    receipt = json.loads((destination / "run.json").read_text(encoding="utf-8"))
    assert receipt["configuration"]["base_registry"] == panel.metadata
    assert receipt["configuration"]["base_genes"] == list(panel.genes)
    assert receipt["input"]["input_sha256"] == file_sha256(source)
    assert "base_sha256" not in receipt["input"]

    base = tmp_path / "base.txt"
    base.write_text("synthetic_extra\n", encoding="utf-8")
    invalid = tmp_path / "invalid"
    rejected("select", "--input", source, "--base-id", panel.panel_id, "--base", base,
             "--normalized", "--n-added", 0, "--out", invalid)
    assert not invalid.exists()
    rejected("select", "--input", source, "--base-id", "BP-11010599",
             "--normalized", "--n-added", 0, "--out", invalid)
    assert not invalid.exists()


@pytest.mark.parametrize("external_totals", [False, True])
def test_preprocess_all_training_exports_explicit_denominator_and_aligned_npz(tmp_path, external_totals):
    counts = np.array([[1, 0, 99], [3, 1, 96], [2, 0, 98], [4, 0, 96]])
    source = write_training(tmp_path / "counts.npz", counts, ["signal", "rare", "rest"])
    totals = np.array([200, 300, 400, 500]) if external_totals else counts.sum(axis=1)
    flags = ["--full-library"]
    totals_path = tmp_path / "totals.npy"
    if external_totals:
        np.save(totals_path, totals)
        flags = ["--library-totals", totals_path]
    destination = tmp_path / "prepared"
    success("preprocess", "--input", source, "--all-training", "--min-detected", 2,
            "--out", destination, *flags)

    output = read_training(destination / "training.npz")
    assert output["genes"].tolist() == ["signal", "rest"]
    expected = np.log1p(counts[:, [0, 2]] * (10000.0 / totals[:, None]))
    np.testing.assert_allclose(output["X"], expected, rtol=1e-14, atol=1e-14)
    original = read_training(source)
    np.testing.assert_array_equal(output["y"], original["y"])
    np.testing.assert_array_equal(output["groups"], original["groups"])
    assert (destination / "genes.txt").read_text(encoding="utf-8").splitlines() == output["genes"].tolist()
    report_text = (destination / "preprocessing.json").read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["input_rows"] == report["exported_training_rows"] == 4
    assert report["sources"]["input_sha256"] == file_sha256(source)
    assert report["normalization"]["mode"] == ("provided_full_library_totals" if external_totals else "full_library")
    if external_totals:
        assert report["sources"]["library_totals_sha256"] == file_sha256(totals_path)
    assert str(tmp_path) not in report_text
    assert {p.name for p in destination.iterdir()} == {"training.npz", "genes.txt", "preprocessing.json"}


def test_preprocess_fit_mask_excludes_heldout_from_qc_and_every_exported_axis(tmp_path):
    counts = np.array([[1, 0, 9], [3, 0, 7], [0, 10000, 0]])
    y = np.array(["class_a", "class_b", "heldout_label"])
    groups = np.array(["training_group", "training_group", "heldout_group"])
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, np.array([True, True, False]))
    outputs, decisions = [], []
    for index, heldout in enumerate([[0, 10000, 0], [999999, 888888, 777777]]):
        changed = counts.copy()
        changed[2] = heldout
        source = write_training(tmp_path / f"counts{index}.npz", changed,
                                ["a", "heldout_only", "c"], y=y, groups=groups)
        destination = tmp_path / f"prepared{index}"
        success("preprocess", "--input", source, "--fit-mask", mask_path,
                "--full-library", "--min-detected", 2, "--out", destination)
        exported = read_training(destination / "training.npz")
        assert exported["genes"].tolist() == ["a", "c"]
        assert exported["X"].shape == (2, 2)
        np.testing.assert_array_equal(exported["y"], y[:2])
        np.testing.assert_array_equal(exported["groups"], groups[:2])
        report_text = (destination / "preprocessing.json").read_text(encoding="utf-8")
        report = json.loads(report_text)
        assert report["input_rows"] == 3 and report["exported_training_rows"] == 2
        assert report["sources"]["fit_mask_sha256"] == file_sha256(mask_path)
        assert "heldout_label" not in report_text and "heldout_group" not in report_text
        decisions.append(report["preprocessing"])
        outputs.append(exported)
    assert decisions[0] == decisions[1]
    for key in outputs[0]:
        np.testing.assert_array_equal(outputs[0][key], outputs[1][key])


@pytest.mark.parametrize("mask", [
    np.array([1, 1]), np.array([False, False]), np.array([True]),
    np.array([[True], [True]]), np.array([True, True], dtype=object),
])
def test_preprocess_rejects_invalid_masks_without_creating_output(tmp_path, mask):
    source = write_training(tmp_path / "counts.npz", [[1, 2], [2, 1]], ["a", "b"])
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, mask)
    destination = tmp_path / "rejected"
    rejected("preprocess", "--input", source, "--fit-mask", mask_path,
             "--full-library", "--min-detected", 1, "--out", destination)
    assert not destination.exists()


def test_preprocess_requires_one_training_declaration_and_one_denominator(tmp_path):
    source = write_training(tmp_path / "counts.npz", [[1, 2], [2, 1]], ["a", "b"])
    mask, totals = tmp_path / "mask.npy", tmp_path / "totals.npy"
    np.save(mask, np.ones(2, dtype=bool))
    np.save(totals, np.full(2, 3))
    policies = [[], ["--all-training"], ["--full-library"],
                ["--all-training", "--fit-mask", mask, "--full-library"],
                ["--all-training", "--full-library", "--library-totals", totals]]
    for index, flags in enumerate(policies):
        destination = tmp_path / f"invalid_policy{index}"
        rejected("preprocess", "--input", source, "--min-detected", 1,
                 "--out", destination, *flags)
        assert not destination.exists()


def test_preprocess_preserves_existing_output_and_can_force_a_registry_base_through_qc(tmp_path):
    panel = load_base_panel("BP-11010510")
    genes = [*reversed(panel.genes), "varying_extra"]
    counts = np.zeros((4, len(genes)), dtype=int)
    counts[:, -1] = [1, 2, 3, 4]
    source = write_training(tmp_path / "counts.npz", counts, genes)
    totals = tmp_path / "totals.npy"
    np.save(totals, np.full(4, 100))
    destination = tmp_path / "prepared"
    args = ["preprocess", "--input", source, "--all-training", "--library-totals", totals,
            "--mandatory-base-id", panel.panel_id, "--mandatory-base-id", panel.panel_id,
            "--min-detected", 1, "--out", destination]
    success(*args)
    output = read_training(destination / "training.npz")
    assert output["genes"].tolist() == genes
    report = json.loads((destination / "preprocessing.json").read_text(encoding="utf-8"))
    assert report["preprocessing"]["mandatory_genes"] == list(panel.genes)
    assert set(report["preprocessing"]["qc_failed_mandatory"]) == set(panel.genes)
    before = {p.name: p.read_bytes() for p in destination.iterdir()}
    rejected(*args)
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == before


def test_screen_cli_serializes_thresholds_and_supports_an_explicit_exclusion_file(tmp_path):
    # Exact 5% and 90% boundaries must not be classified as <5% and >90%.
    positive = np.array([[4, 5, 4], [4, 4, 4], [91, 91, 90], [91, 91, 91]])
    genes = np.array(["hit", "at5", "at90"])
    aggregate = tmp_path / "aggregated.npz"
    np.savez(aggregate, positive_counts=positive, cell_counts=np.full(4, 100), genes=genes)
    screened = tmp_path / "screened"
    success("screen", "--input", aggregate, "--out", screened)
    report = json.loads((screened / "screen.json").read_text(encoding="utf-8"))
    assert report["status"] == "evaluable"
    assert report["hit_mask"] == [True, False, False]
    assert report["excluded_genes"] == ["hit"]
    assert report["input_sha256"] == file_sha256(aggregate)
    assert (screened / "excluded_genes.txt").read_text(encoding="utf-8") == "hit\n"
    before = {p.name: p.read_bytes() for p in screened.iterdir()}
    rejected("screen", "--input", aggregate, "--out", screened)
    assert {p.name: p.read_bytes() for p in screened.iterdir()} == before

    counts = np.vstack([(np.arange(100)[:, None] < row).astype(int) for row in positive])
    source = write_training(tmp_path / "counts.npz", counts, genes,
                            groups=np.repeat(["donor_a", "donor_b", "donor_c", "donor_d"], 100))
    totals = tmp_path / "totals.npy"
    np.save(totals, np.full(400, 100))
    for name, exclusions, expected in [
        ("without_screen", [], genes.tolist()),
        ("with_explicit_exclusion", ["--exclude-genes", screened / "excluded_genes.txt"], ["at5", "at90"]),
    ]:
        destination = tmp_path / name
        success("preprocess", "--input", source, "--all-training", "--library-totals", totals,
                "--out", destination, *exclusions)
        assert read_training(destination / "training.npz")["genes"].tolist() == expected
        assert not (destination / "screen.json").exists()


@pytest.mark.parametrize("donors", [2, 3])
def test_screen_with_fewer_than_four_donors_exports_no_negative_decision(tmp_path, donors):
    source = tmp_path / "aggregated.npz"
    np.savez(source, positive_counts=np.array([[0], [0], [100]])[:donors],
             cell_counts=np.full(donors, 100), genes=np.array(["gene_a"]))
    destination = tmp_path / "not_applicable"
    success("screen", "--input", source, "--out", destination)
    report = json.loads((destination / "screen.json").read_text(encoding="utf-8"))
    assert report["status"] == "not_applicable"
    assert report["reason"] == "fewer_than_four_independent_donors"
    assert report["hit_mask"] is None
    assert report["excluded_genes"] == []
    assert (destination / "excluded_genes.txt").read_bytes() == b""


@pytest.mark.parametrize("payload", [b"", b"PK\x03\x04truncated", b"not numpy"])
def test_screen_malformed_archive_is_a_cli_error_without_partial_output(tmp_path, payload):
    source = tmp_path / "broken.npz"
    source.write_bytes(payload)
    destination = tmp_path / "failed"
    rejected("screen", "--input", source, "--out", destination)
    assert not destination.exists()


@pytest.mark.parametrize("donors", [None, 3, 4])
def test_preprocess_measurement_input_controls_gate_status_and_automatic_exclusion(tmp_path, donors):
    # Aggregate columns deliberately differ from the reference-expression axis.
    # Boundary genes must survive, and gate exclusions must not alter library totals.
    genes = ["hit", "at5", "at90"]
    counts = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2], [1, 3, 2]])
    source = write_training(tmp_path / "counts.npz", counts, genes)
    totals_path = tmp_path / "totals.npy"
    np.save(totals_path, np.full(len(counts), 100))
    positive = np.array([[4, 5, 4], [4, 4, 4], [91, 91, 90], [91, 91, 91]])
    flags = []
    if donors is not None:
        measurement_path = tmp_path / "measurement.npz"
        np.savez(measurement_path, genes=np.array(genes)[[2, 0, 1]],
                 positive_counts=positive[:donors, [2, 0, 1]], cell_counts=np.full(donors, 100))
        flags = ["--measurement-counts", measurement_path]
    destination = tmp_path / "prepared"
    success("preprocess", "--input", source, "--all-training", "--library-totals", totals_path,
            "--min-detected", 1, "--out", destination, *flags)
    output = read_training(destination / "training.npz")
    columns = [1, 2] if donors == 4 else [0, 1, 2]
    assert output["genes"].tolist() == [genes[i] for i in columns]
    np.testing.assert_allclose(output["X"], np.log1p(counts[:, columns] * 100), rtol=1e-14)
    report = json.loads((destination / "preprocessing.json").read_text(encoding="utf-8"))
    gate = report["preprocessing"]["measurement_screen"]
    expected_status = {None: "not_evaluated", 3: "not_applicable", 4: "evaluable"}[donors]
    assert gate["status"] == expected_status
    assert gate["hit_mask"] == ([True, False, False] if donors == 4 else None)
    assert gate["excluded_genes"] == (["hit"] if donors == 4 else [])
    if donors is not None:
        assert report["sources"]["measurement_counts_sha256"] == file_sha256(measurement_path)
    else:
        assert "measurement_counts_sha256" not in report["sources"]


def test_preprocess_gate_collision_with_mandatory_gene_leaves_no_output(tmp_path):
    source = write_training(tmp_path / "counts.npz", [[1, 2], [2, 1]], ["hit", "keep"])
    measurement = tmp_path / "measurement.npz"
    np.savez(measurement, positive_counts=np.array([[4, 50], [4, 50], [91, 50], [91, 50]]),
             cell_counts=np.full(4, 100), genes=np.array(["hit", "keep"]))
    mandatory = tmp_path / "mandatory.txt"
    mandatory.write_text("hit\n", encoding="utf-8")
    destination = tmp_path / "conflict"
    result = rejected("preprocess", "--input", source, "--all-training", "--full-library",
                      "--min-detected", 1, "--measurement-counts", measurement,
                      "--mandatory-genes", mandatory, "--out", destination)
    assert "mandatory" in result.stderr.lower()
    assert not destination.exists()


def test_preprocess_gate_accepts_candidate_only_coverage_and_does_not_refill_top_cap(tmp_path):
    genes = ["hit", "qc_fail", "keep", "below_top_cap"]
    counts = np.array([[1, 0, 1, 3], [20, 0, 1, 4], [1, 0, 15, 3], [20, 0, 15, 4]])
    source = write_training(tmp_path / "counts.npz", counts, genes)
    totals = tmp_path / "totals.npy"
    np.save(totals, np.full(4, 100))
    measurement = tmp_path / "measurement.npz"
    # Only the two candidates entering the gate require coverage. Extra aggregate
    # columns are harmless and must not change the reference denominator.
    np.savez(measurement, genes=np.array(["extra_target_gene", "keep", "hit"]),
             positive_counts=np.array([[1, 50, 4], [1, 50, 4], [99, 50, 91], [99, 50, 91]]),
             cell_counts=np.full(4, 100))
    destination = tmp_path / "gated_top"
    success("preprocess", "--input", source, "--all-training", "--library-totals", totals,
            "--min-detected", 1, "--top-genes", 2, "--measurement-counts", measurement,
            "--out", destination)
    output = read_training(destination / "training.npz")
    assert output["genes"].tolist() == ["keep"]
    np.testing.assert_allclose(output["X"], np.log1p(counts[:, [2]] * 100), rtol=1e-14)
    report = json.loads((destination / "preprocessing.json").read_text(encoding="utf-8"))
    assert report["preprocessing"]["pre_screen_genes"] == ["hit", "keep"]
    gate = report["preprocessing"]["measurement_screen"]
    assert gate["genes"] == ["hit", "keep"]
    assert gate["excluded_genes"] == ["hit"]


@pytest.mark.parametrize("measurement_genes", [["a", "a"], ["a", "unknown"], ["a"]])
def test_preprocess_rejects_ambiguous_or_incomplete_measurement_axis(tmp_path, measurement_genes):
    source = write_training(tmp_path / "counts.npz", [[1, 2], [2, 1]], ["a", "b"])
    measurement = tmp_path / "measurement.npz"
    np.savez(measurement, positive_counts=np.full((4, len(measurement_genes)), 50),
             cell_counts=np.full(4, 100), genes=np.array(measurement_genes))
    destination = tmp_path / "invalid_axis"
    rejected("preprocess", "--input", source, "--all-training", "--full-library",
             "--min-detected", 1, "--measurement-counts", measurement, "--out", destination)
    assert not destination.exists()
