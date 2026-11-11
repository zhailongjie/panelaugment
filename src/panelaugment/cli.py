"""Command-line selection with explicit normalization and reproducible inputs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np

from . import PanelAugmenter, __version__, log_normalize
from .io import TrainingData, axis_sha256, file_sha256, load_h5ad, load_library_totals, load_npz, read_base, write_result


def synthetic_data(seed: int = 2026) -> TrainingData:
    """Small generated counts, containing no study or participant data."""
    rng = np.random.default_rng(seed)
    genes = np.asarray([f"Gene{i:02d}" for i in range(12)])
    y = np.tile(np.repeat(["TypeA", "TypeB", "TypeC"], 20), 3)
    groups = np.repeat(["Group1", "Group2", "Group3"], 60)
    rates = np.full((len(y), len(genes)), 2.0)
    for index, label in enumerate(["TypeA", "TypeB", "TypeC"]):
        rates[y == label, index * 3:(index + 1) * 3] += 7.0
    rates[groups == "Group2"] *= 1.2
    rates[groups == "Group3"] *= 0.8
    return TrainingData(rng.poisson(rates), y, groups, genes)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="panelaugment",
        description="PA: augment an ordered fixed gene panel from training expression.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="select genes using NPZ or optional H5AD input")
    select.add_argument("--input", type=Path, required=True)
    base = select.add_mutually_exclusive_group(required=True)
    base.add_argument("--base", type=Path, help="ordered base: text lines or JSON list")
    base.add_argument("--base-id", help="bundled BP-110105xx base; input must contain every base gene")
    select.add_argument("--n-added", type=int, required=True, help="number of NEW genes (not total size)")
    select.add_argument("--out", type=Path, required=True, help="new output directory")
    select.add_argument("--ridge", type=float, default=1.0)
    select.add_argument("--diagonal", action="store_true", help="run the diagonal-covariance control")
    select.add_argument("--label-key", help="required H5AD obs label column")
    select.add_argument("--group-key", help="required H5AD obs donor/sample-group column")
    select.add_argument("--layer", help="H5AD expression layer (default: X)")
    normalization = select.add_mutually_exclusive_group(required=True)
    normalization.add_argument("--normalized", action="store_true", help="X already has selection normalization")
    normalization.add_argument("--full-library", action="store_true", help="counts include the full measured gene universe")
    normalization.add_argument("--library-totals", type=Path, help="NPY array of full-library totals aligned to rows")
    normalization.add_argument("--library-total-key", help="H5AD obs column with full-library totals")
    demo = commands.add_parser("demo", help="run a deterministic synthetic example")
    demo.add_argument("--out", type=Path, required=True, help="new output directory")
    panels = commands.add_parser("panels", help="inspect or export the frozen manuscript base panels")
    panel_commands = panels.add_subparsers(dest="panel_command", required=True)
    panel_list = panel_commands.add_parser("list", help="list BP IDs, datasets and construction methods")
    panel_list.add_argument("--dataset", help="dlpfc, caudate (or cah), or merfish")
    panel_list.add_argument("--n-genes", type=int)
    panel_list.add_argument("--construction", help="reconst, spapros, or wilcoxon")
    panel_list.add_argument("--format", choices=["tsv", "json"], default="tsv")
    panel_export = panel_commands.add_parser("export", help="export one ordered gene list")
    panel_export.add_argument("--id", required=True, dest="panel_id")
    panel_export.add_argument("--out", required=True, type=Path, help="new .txt or .json file")
    preprocess = commands.add_parser("preprocess", help="fit candidate QC on declared training rows and export training NPZ")
    preprocess.add_argument("--input", type=Path, required=True, help="raw counts in NPZ or H5AD")
    preprocess.add_argument("--out", type=Path, required=True, help="new output directory")
    preprocess.add_argument("--label-key")
    preprocess.add_argument("--group-key")
    preprocess.add_argument("--layer", help="raw-count H5AD layer (default: X)")
    fitting = preprocess.add_mutually_exclusive_group(required=True)
    fitting.add_argument("--all-training", action="store_true", help="declare every input row a selection-training row")
    fitting.add_argument("--fit-mask", type=Path, help="boolean NPY mask; only True rows fit QC and enter exported training.npz")
    denominator = preprocess.add_mutually_exclusive_group(required=True)
    denominator.add_argument("--full-library", action="store_true", help="input columns define the full normalization library")
    denominator.add_argument("--library-totals", type=Path, help="NPY full-library totals for every input row")
    denominator.add_argument("--library-total-key", help="H5AD obs column of full-library totals")
    preprocess.add_argument("--min-detected", type=int, default=20)
    preprocess.add_argument("--min-variance", type=float, default=1e-10)
    preprocess.add_argument("--top-genes", type=int, help="stable highest-variance candidate cap; omitted preserves input order")
    preprocess.add_argument("--target-sum", type=float, default=10000.0)
    preprocess.add_argument("--mandatory-genes", type=Path, help="ordered text or JSON gene list to retain through QC")
    preprocess.add_argument("--mandatory-base-id", action="append", default=[], help="retain a bundled base; repeat for a base union")
    preprocess.add_argument("--exclude-genes", type=Path, help="additional explicitly declared excluded genes")
    preprocess.add_argument("--measurement-counts", type=Path,
                            help="NPZ donor detection counts for the PA measurement gate; genes must cover all pre-gate candidates")
    screen = commands.add_parser("screen", help="inspect the PA preprocessing measurement gate on donor-level counts")
    screen.add_argument("--input", required=True, type=Path, help="NPZ: integer positive_counts, cell_counts, and string genes")
    screen.add_argument("--out", required=True, type=Path, help="new output directory")
    return parser


def _run_panels(args: argparse.Namespace) -> None:
    from .base_panels import list_base_panels, load_base_panel

    if args.panel_command == "list":
        panels = list_base_panels(dataset=args.dataset, n_genes=args.n_genes,
                                  construction=args.construction)
        if args.format == "json":
            print(json.dumps([panel.metadata for panel in panels], ensure_ascii=False, indent=2))
        else:
            writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
            writer.writerow(["panel_id", "dataset", "n_genes", "construction"])
            for panel in panels:
                writer.writerow([panel.panel_id, panel.dataset, panel.n_genes, panel.construction])
        return
    panel = load_base_panel(args.panel_id)
    if args.out.suffix.lower() not in {".txt", ".json"}:
        raise ValueError("Panel export --out must end in .txt or .json")
    payload = (json.dumps(list(panel.genes), ensure_ascii=False, indent=2) + "\n"
               if args.out.suffix.lower() == ".json" else "\n".join(panel.genes) + "\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    print(f"Saved {panel.panel_id} ({panel.n_genes} genes) to {args.out}")


def _run_preprocess(args: argparse.Namespace) -> Path:
    from .preprocessing import CandidatePreprocessor

    if args.out.exists():
        raise ValueError("Output directory already exists; choose a new --out path")
    data = _read_input(args)
    fit_mask = (np.ones(data.X.shape[0], dtype=bool) if args.all_training
                else load_library_totals(args.fit_mask))
    if fit_mask.dtype.kind != "b" or fit_mask.shape != (data.X.shape[0],):
        raise ValueError("--fit-mask must contain one boolean value per input row")
    totals = (load_library_totals(args.library_totals) if args.library_totals
              else data.library_totals)
    mandatory = read_base(args.mandatory_genes) if args.mandatory_genes else []
    if args.mandatory_base_id:
        from .base_panels import load_base_panel
        for panel_id in args.mandatory_base_id:
            mandatory.extend(load_base_panel(panel_id).genes)
    mandatory = list(dict.fromkeys(mandatory))
    excluded = read_base(args.exclude_genes) if args.exclude_genes else []
    measurement = {}
    if args.measurement_counts:
        positive, support, measurement_genes = _read_measurement_counts(args.measurement_counts)
        measurement = {
            "measurement_positive_counts": positive,
            "measurement_cell_counts": support,
            "measurement_genes": measurement_genes,
        }
    model = CandidatePreprocessor(min_detected=args.min_detected,
                                  min_variance=args.min_variance,
                                  n_top_genes=args.top_genes,
                                  target_sum=args.target_sum)
    X = model.fit_transform(data.X, data.genes, fit_mask=fit_mask,
                            library_totals=totals, mandatory_genes=mandatory,
                            excluded_genes=excluded, **measurement)
    X = X[fit_mask]
    if hasattr(X, "toarray"):
        X = X.toarray()
    provenance = {"input_sha256": file_sha256(args.input)}
    for key in ("library_totals", "fit_mask", "mandatory_genes", "exclude_genes", "measurement_counts"):
        value = getattr(args, key)
        if value is not None:
            provenance[f"{key}_sha256"] = file_sha256(value)
    report = {
        "schema": "panelaugment.preprocess-run.v1",
        "toolkit_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing": model.to_dict(),
        "sources": provenance,
        "input_rows": int(data.X.shape[0]),
        "exported_training_rows": int(np.count_nonzero(fit_mask)),
        "mandatory_base_ids": args.mandatory_base_id,
        "normalization": {"mode": "full_library" if args.full_library else "provided_full_library_totals",
                          "target_sum": args.target_sum},
        "scope": "PA preprocessing: candidate QC fits only on declared training rows; the measurement gate uses the supplied donor aggregates and automatically excludes hits. Its status is recorded when inputs are absent or inapplicable. Only training rows are exported. This is not a classifier evaluation.",
    }
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        np.savez_compressed(args.out / "training.npz", X=X, y=data.y[fit_mask],
                            groups=data.groups[fit_mask], genes=np.asarray(model.genes_, dtype=str))
        (args.out / "genes.txt").write_text("\n".join(model.genes_) + "\n", encoding="utf-8")
        (args.out / "preprocessing.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                              allow_nan=False) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(args.out)
        raise
    return args.out


def _read_measurement_counts(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        archive = np.load(path, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError("Measurement input must be an NPZ archive")
        with archive:
            if not {"positive_counts", "cell_counts", "genes"} <= set(archive.files):
                raise ValueError("Measurement NPZ requires positive_counts, cell_counts and genes")
            positive, support, genes = archive["positive_counts"], archive["cell_counts"], archive["genes"]
            if genes.ndim != 1 or genes.dtype.kind not in {"U", "S"}:
                raise ValueError("Measurement genes must be a one-dimensional string array")
            if genes.dtype.kind == "S":
                genes = np.char.decode(genes, "utf-8")
            return positive, support, genes
    except (EOFError, zipfile.BadZipFile, UnicodeError) as exc:
        raise ValueError("Cannot read measurement NPZ: invalid or incomplete archive") from exc


def _run_screen(args: argparse.Namespace) -> Path:
    from .preprocessing import measurement_consistency_screen

    if args.out.exists():
        raise ValueError("Output directory already exists; choose a new --out path")
    report = measurement_consistency_screen(*_read_measurement_counts(args.input))
    report["input_sha256"] = file_sha256(args.input)
    report["toolkit_version"] = __version__
    report["scope"] = "Standalone inspection of the PA preprocessing measurement gate. The preprocess command applies its hits automatically when donor aggregates are supplied. Measurement inputs and their role must be reported; this is not an independent validation result."
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        (args.out / "screen.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                      allow_nan=False) + "\n", encoding="utf-8")
        excluded = report["excluded_genes"]
        (args.out / "excluded_genes.txt").write_text("\n".join(excluded) + ("\n" if excluded else ""),
                                                    encoding="utf-8")
    except BaseException:
        shutil.rmtree(args.out)
        raise
    return args.out


def _read_input(args: argparse.Namespace) -> TrainingData:
    suffix = args.input.suffix.lower()
    if suffix == ".h5ad":
        if not args.label_key or not args.group_key:
            raise ValueError("H5AD input requires --label-key and --group-key")
        return load_h5ad(args.input, label_key=args.label_key, group_key=args.group_key,
                        layer=args.layer, library_total_key=args.library_total_key)
    if suffix != ".npz":
        raise ValueError("Input extension must be .npz or .h5ad")
    if any((args.label_key, args.group_key, args.layer, args.library_total_key)):
        raise ValueError("--label-key, --group-key, --layer, and --library-total-key require H5AD input")
    return load_npz(args.input)


def _run(args: argparse.Namespace) -> Path:
    if args.out.exists():
        raise ValueError("Output directory already exists; choose a new --out path")
    started = time.perf_counter()
    sources = {}
    if args.command == "demo":
        data = synthetic_data()
        base, n_added, ridge, diagonal = ["Gene00", "Gene03"], 4, 1.0, False
        X = log_normalize(data.X)
        normalization = {"mode": "full_library", "target_sum": 10000.0,
                         "note": "Synthetic counts: all 12 measured genes define the denominator."}
        config = {"synthetic_seed": 2026}
    else:
        data = _read_input(args)
        base_record = None
        if args.base_id:
            from .base_panels import load_base_panel
            panel = load_base_panel(args.base_id)
            base, base_record = list(panel.genes), panel.metadata
        else:
            base = read_base(args.base)
        n_added, ridge, diagonal = args.n_added, args.ridge, args.diagonal
        sources = {"input_sha256": file_sha256(args.input)}
        if args.base:
            sources["base_sha256"] = file_sha256(args.base)
        config = {"label_key": args.label_key, "group_key": args.group_key, "layer": args.layer}
        if base_record is not None:
            config["base_registry"] = base_record
        if args.normalized:
            X = data.X
            normalization = {"mode": "already_normalized", "note": "As declared by the caller; not inferred."}
        else:
            totals = data.library_totals
            if args.library_totals:
                totals = load_library_totals(args.library_totals)
                sources["library_totals_sha256"] = file_sha256(args.library_totals)
            X = log_normalize(data.X, library_totals=totals)
            normalization = {"mode": "full_library" if args.full_library else "provided_full_library_totals",
                             "target_sum": 10000.0, "library_total_key": args.library_total_key}
    prepared_seconds = time.perf_counter() - started
    moment_start = time.perf_counter()
    model = PanelAugmenter(ridge=ridge).fit(X, data.y, data.groups, gene_names=data.genes)
    moments_seconds = time.perf_counter() - moment_start
    selection_start = time.perf_counter()
    result = model.select(base_genes=base, n_added=n_added, diagonal=diagonal)
    selection_seconds = time.perf_counter() - selection_start
    run = {
        "schema_version": 1,
        "toolkit_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": args.command,
        "synthetic": args.command == "demo",
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "input": {"n_cells": int(data.X.shape[0]), "n_genes": int(data.X.shape[1]),
                  "gene_axis_sha256": axis_sha256(data.genes),
                  "label_axis_sha256": axis_sha256(data.y),
                  "group_axis_sha256": axis_sha256(data.groups), **sources},
        "configuration": {"n_added": n_added, "base_genes": base, "ridge": ridge,
                          "diagonal": diagonal, **config},
        "normalization": normalization,
        "timing_seconds": {"input_and_normalization": prepared_seconds,
                           "training_moments": moments_seconds, "selection": selection_seconds},
        "scope": "Selection from caller-supplied training rows; no evaluation or automatic budget tuning.",
    }
    return write_result(result, args.out, run=run)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "panels":
            _run_panels(args)
            return 0
        if args.command == "preprocess":
            output = _run_preprocess(args)
            print(f"Saved selection-training preprocessing to {output}")
            return 0
        if args.command == "screen":
            output = _run_screen(args)
            print(f"Saved PA preprocessing measurement gate to {output}")
            return 0
        output = _run(args)
    except (ValueError, OSError, ImportError, FloatingPointError, TypeError) as exc:
        print(f"panelaugment: error: {exc}", file=sys.stderr)
        return 2
    print(f"Saved PA selection to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
