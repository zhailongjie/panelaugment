"""Explicit, pickle-free input readers and portable result exports."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any
import zipfile

import numpy as np


@dataclass(frozen=True)
class TrainingData:
    """A cells-by-genes matrix and aligned label, group, and gene axes."""

    X: Any
    y: np.ndarray
    groups: np.ndarray
    genes: np.ndarray
    library_totals: np.ndarray | None = None


def _string_axis(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "US":
        raise ValueError(f"{name} must be a one-dimensional string array")
    if array.dtype.kind == "S":
        try:
            array = np.char.decode(array, "utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{name} must use UTF-8 strings") from exc
    if any(not value.strip() for value in array.tolist()):
        raise ValueError(f"{name} contains empty identifiers")
    return array


def _validate_axes(data: TrainingData) -> TrainingData:
    if len(data.X.shape) != 2 or 0 in data.X.shape:
        raise ValueError("X must be a nonempty cells-by-genes matrix")
    if len(data.y) != data.X.shape[0] or len(data.groups) != data.X.shape[0]:
        raise ValueError("y and groups must match the number of cells in X")
    if len(data.genes) != data.X.shape[1]:
        raise ValueError("genes must match the number of columns in X")
    if len(set(data.genes.tolist())) != len(data.genes):
        raise ValueError("genes must contain unique identifiers")
    return data


def load_npz(path: str | Path) -> TrainingData:
    """Read X/y/groups/genes from an NPZ without enabling pickle.

    ``X`` is dense numeric data. All three identifier arrays must be NumPy
    Unicode or UTF-8 byte arrays, not object arrays. Row order defines alignment.
    No normalization or candidate filtering is performed.
    """
    try:
        archive = np.load(path, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError("Input must be an NPZ archive, not an NPY array")
        with archive:
            required = {"X", "y", "groups", "genes"}
            if not required <= set(archive.files):
                raise ValueError("NPZ must contain X, y, groups, and genes")
            X = archive["X"]
            if X.dtype.kind not in "iuf":
                raise ValueError("X must be a real numeric array")
            data = TrainingData(
                X=X,
                y=_string_axis(archive["y"], "y"),
                groups=_string_axis(archive["groups"], "groups"),
                genes=_string_axis(archive["genes"], "genes"),
            )
    except (OSError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError("Could not read NPZ input; check the file exists and is a complete NPZ archive") from exc
    return _validate_axes(data)


def load_library_totals(path: str | Path) -> np.ndarray:
    """Read a single pickle-free NPY array; semantic validation is in normalization."""
    try:
        totals = np.load(path, allow_pickle=False)
    except (OSError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError("Could not read library totals; provide a complete NPY array") from exc
    if not isinstance(totals, np.ndarray):
        totals.close()
        raise ValueError("--library-totals requires a .npy array")
    return totals


def load_h5ad(
    path: str | Path,
    *,
    label_key: str,
    group_key: str,
    layer: str | None = None,
    library_total_key: str | None = None,
) -> TrainingData:
    """Read an AnnData file with explicit metadata keys and optional layer.

    Requires the ``anndata`` extra. Missing labels/groups, duplicate gene or
    cell identifiers, and absent layers are errors. No input is modified.
    Sparse input stays sparse until the method's dense moment estimation.
    """
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("H5AD input requires: pip install 'panelaugment[h5ad]'") from exc

    adata = ad.read_h5ad(path)
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("H5AD cell and gene identifiers must be unique")
    for key in (label_key, group_key):
        if key not in adata.obs:
            raise ValueError(f"Missing H5AD obs key: {key}")
        if adata.obs[key].isna().any():
            raise ValueError(f"Missing values in H5AD obs key: {key}")
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"Missing H5AD layer: {layer}")
    totals = None
    if library_total_key is not None:
        if library_total_key not in adata.obs:
            raise ValueError(f"Missing H5AD obs key: {library_total_key}")
        totals = adata.obs[library_total_key].to_numpy(dtype=float, copy=True)
    X = adata.X if layer is None else adata.layers[layer]
    if X is None:
        raise ValueError("H5AD does not contain an expression matrix")
    return _validate_axes(TrainingData(
        X=X.copy(),
        y=_string_axis(adata.obs[label_key].astype(str).to_numpy(dtype=str), "y"),
        groups=_string_axis(adata.obs[group_key].astype(str).to_numpy(dtype=str), "groups"),
        genes=_string_axis(adata.var_names.to_numpy(dtype=str), "genes"),
        library_totals=totals,
    ))


def read_base(path: str | Path) -> list[str]:
    """Read a JSON string list or a UTF-8 file with one exact gene ID per line.

    Blank text lines are skipped. Gene order is preserved. An empty file/list
    denotes de novo selection. Identifiers are case-sensitive.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        values = json.loads(text)
        if not isinstance(values, list) or not all(isinstance(g, str) for g in values):
            raise ValueError("Base JSON must contain a list of gene strings")
    else:
        values = [line.strip() for line in text.splitlines() if line.strip()]
    if any(not g.strip() or g != g.strip() for g in values):
        raise ValueError("Base identifiers cannot be empty or have surrounding whitespace")
    if len(values) != len(set(values)):
        raise ValueError("Base genes must be unique")
    return values


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def axis_sha256(values: Any) -> str:
    payload = json.dumps(np.asarray(values).tolist(), ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_result(result: Any, output: str | Path, *, run: dict[str, Any]) -> Path:
    """Export a selection to a new directory; never overwrite prior results."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        payloads = {
            "panels.json": result.to_dict(),
            "trace.json": result.trace,
            "pair_support.json": result.pair_support,
            "run.json": run,
        }
        for name, payload in payloads.items():
            (output / name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        with (output / "genes.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["rank", "gene", "candidate_index", "role"])
            for rank, (gene, index) in enumerate(zip(result.genes, result.indices), 1):
                writer.writerow([rank, gene, index,
                                 "base" if rank <= len(result.base_genes) else "added"])
    except BaseException:
        shutil.rmtree(output)
        raise
    return output
