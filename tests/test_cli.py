"""Exercise the installed-style CLI on generated data and invalid requests."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from panelaugment import log_normalize
from panelaugment.cli import synthetic_data
from panelaugment.io import file_sha256


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def cli(self, *args):
        return subprocess.run([sys.executable, "-m", "panelaugment", *map(str, args)],
                              capture_output=True, text=True, env=os.environ.copy())

    def inputs(self, normalized=False):
        data = synthetic_data()
        input_path = self.root / ("normalized.npz" if normalized else "counts.npz")
        X = log_normalize(data.X) if normalized else data.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        np.savez(input_path, X=X, y=data.y, groups=data.groups, genes=data.genes)
        base = self.root / "base.txt"
        base.write_text("Gene00\nGene03\n", encoding="utf-8")
        return input_path, base

    def test_demo_exports_reproducible_selection_and_provenance(self):
        outputs = [self.root / "one", self.root / "two"]
        panels = []
        for output in outputs:
            run = self.cli("demo", "--out", output)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(set(p.name for p in output.iterdir()),
                             {"panels.json", "genes.csv", "trace.json", "pair_support.json", "run.json"})
            panel = json.loads((output / "panels.json").read_text())
            panels.append(panel)
            self.assertEqual(panel["genes"][:2], ["Gene00", "Gene03"])
            self.assertEqual(len(panel["genes"]), 6)
            receipt = json.loads((output / "run.json").read_text())
            self.assertTrue(receipt["synthetic"])
            self.assertEqual(receipt["input"]["n_cells"], 180)
            self.assertIn("training_moments", receipt["timing_seconds"])
            self.assertNotIn(str(self.root), (output / "run.json").read_text())
        self.assertEqual(panels[0], panels[1])
        before = (outputs[0] / "panels.json").read_bytes()
        rerun = self.cli("demo", "--out", outputs[0])
        self.assertEqual(rerun.returncode, 2)
        self.assertEqual((outputs[0] / "panels.json").read_bytes(), before)

    def test_npz_normalization_routes_agree(self):
        counts, base = self.inputs()
        normalized, _ = self.inputs(normalized=True)
        totals = self.root / "totals.npy"
        np.save(totals, synthetic_data().X.sum(axis=1))
        panels = []
        for index, (input_path, flags) in enumerate([
            (counts, ["--full-library"]),
            (counts, ["--library-totals", totals]),
            (normalized, ["--normalized"]),
        ]):
            output = self.root / f"route{index}"
            result = self.cli("select", "--input", input_path, "--base", base,
                              "--n-added", 4, "--out", output, *flags)
            self.assertEqual(result.returncode, 0, result.stderr)
            panels.append(json.loads((output / "panels.json").read_text())["genes"])
            receipt = json.loads((output / "run.json").read_text())
            self.assertEqual(len(receipt["input"]["input_sha256"]), 64)
        self.assertEqual(panels[0], panels[1])
        self.assertEqual(panels[0], panels[2])

    def test_normalization_is_required_and_mutually_exclusive(self):
        input_path, base = self.inputs()
        args = ["select", "--input", input_path, "--base", base,
                "--n-added", 2, "--out", self.root / "result"]
        for flags in [[], ["--normalized", "--full-library"]]:
            result = self.cli(*args, *flags)
            self.assertEqual(result.returncode, 2)
            self.assertFalse((self.root / "result").exists())

    def test_invalid_budget_or_h5ad_options_create_no_results(self):
        input_path, base = self.inputs()
        for index, flags in enumerate([["--n-added", -1], ["--n-added", 100],
                                       ["--n-added", 1, "--layer", "log1p"]]):
            output = self.root / f"invalid{index}"
            result = self.cli("select", "--input", input_path, "--base", base,
                              "--normalized", "--out", output, *flags)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(output.exists())

    def test_zero_budget_returns_exact_ordered_base(self):
        input_path, base = self.inputs(normalized=True)
        base.write_text("Gene03\nGene00\n", encoding="utf-8")
        output = self.root / "zero"
        result = self.cli("select", "--input", input_path, "--base", base,
                          "--n-added", 0, "--normalized", "--out", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((output / "panels.json").read_text())["genes"], ["Gene03", "Gene00"])

    def test_malformed_inputs_fail_without_traceback_or_output(self):
        _, base = self.inputs()
        input_path = self.root / "bad.npz"
        payloads = [b"", b"PK\x03\x04truncated", b"not numpy"]
        with input_path.open("wb") as stream:
            np.save(stream, np.ones((4, 3)))
        payloads.append(input_path.read_bytes())
        for index, payload in enumerate(payloads):
            input_path.write_bytes(payload)
            output = self.root / f"malformed{index}"
            result = self.cli("select", "--input", input_path, "--base", base,
                              "--n-added", 1, "--normalized", "--out", output)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(output.exists())

    @unittest.skipUnless(importlib.util.find_spec("anndata"), "optional anndata is not installed")
    def test_h5ad_layer_and_total_column_routes_agree_without_mutation(self):
        import anndata as ad
        import pandas as pd
        from scipy import sparse

        data = synthetic_data()
        totals = data.X.sum(axis=1) + 100  # Omitted measured genes contribute to the denominator.
        a = ad.AnnData(sparse.csr_matrix(data.X),
                       obs=pd.DataFrame({"label": data.y, "group": data.groups,
                                         "full_total": totals}, index=[f"c{i}" for i in range(len(data.y))]),
                       var=pd.DataFrame(index=data.genes))
        a.layers["normalized"] = log_normalize(a.X, library_totals=totals)
        input_path = self.root / "training.h5ad"
        a.write_h5ad(input_path)
        original_hash = file_sha256(input_path)
        base = self.root / "base.json"
        base.write_text('["Gene00", "Gene03"]', encoding="utf-8")
        selections = []
        for index, flags in enumerate([["--layer", "normalized", "--normalized"],
                                       ["--library-total-key", "full_total"]]):
            output = self.root / f"h5ad{index}"
            result = self.cli("select", "--input", input_path, "--base", base,
                              "--label-key", "label", "--group-key", "group",
                              "--n-added", 4, "--out", output, *flags)
            self.assertEqual(result.returncode, 0, result.stderr)
            selections.append(json.loads((output / "panels.json").read_text()))
            self.assertEqual(file_sha256(input_path), original_hash)
        self.assertEqual(selections[0], selections[1])
        for index, flags in enumerate([["--normalized", "--layer", "missing"],
                                       ["--library-total-key", "missing"]]):
            output = self.root / f"h5ad_missing{index}"
            result = self.cli("select", "--input", input_path, "--base", base,
                              "--label-key", "label", "--group-key", "group",
                              "--n-added", 4, "--out", output, *flags)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(output.exists())
            self.assertEqual(file_sha256(input_path), original_hash)


if __name__ == "__main__":
    unittest.main()
