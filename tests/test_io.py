"""Input alignment, pickle policy, optional H5AD, and export integrity."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from panelaugment.io import axis_sha256, file_sha256, load_h5ad, load_library_totals, load_npz, read_base, write_result


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.arrays = dict(X=np.arange(12).reshape(4, 3),
                           y=np.array(["a", "b", "a", "b"]),
                           groups=np.array(["one", "one", "two", "two"]),
                           genes=np.array(["G1", "G2", "G3"]))

    def tearDown(self):
        self.temp.cleanup()

    def save(self, **overrides):
        path = self.root / "input.npz"
        np.savez(path, **(self.arrays | overrides))
        return path

    def test_npz_preserves_axes_and_values(self):
        data = load_npz(self.save())
        np.testing.assert_array_equal(data.X, self.arrays["X"])
        np.testing.assert_array_equal(data.genes, self.arrays["genes"])
        np.testing.assert_array_equal(data.groups, self.arrays["groups"])

    def test_npz_rejects_object_arrays_and_axis_errors(self):
        for overrides in [dict(y=np.array(["a", "b", "a", "b"], dtype=object)),
                          dict(y=np.array(["a", "b"])),
                          dict(genes=np.array(["G1", "G1", "G3"])),
                          dict(groups=np.array(["one", "", "two", "two"])),
                          dict(X=np.ones((4, 3), dtype=complex)),
                          dict(X=np.ones(4))]:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    load_npz(self.save(**overrides))

    def test_npz_rejects_missing_required_keys(self):
        path = self.root / "missing.npz"
        np.savez(path, X=self.arrays["X"])
        with self.assertRaisesRegex(ValueError, "X, y, groups, and genes"):
            load_npz(path)

    def test_npz_rejects_npy_and_corrupt_archives_cleanly(self):
        path = self.root / "disguised.npz"
        with path.open("wb") as stream:
            np.save(stream, self.arrays["X"])
        with self.assertRaisesRegex(ValueError, "NPZ archive"):
            load_npz(path)
        for payload in [b"", b"PK\x03\x04truncated", b"not a numpy archive"]:
            path.write_bytes(payload)
            with self.assertRaises(ValueError):
                load_npz(path)

    def test_library_totals_require_single_npy_array(self):
        totals = self.root / "totals.npy"
        np.save(totals, np.array([100, 200, 300, 400]))
        np.testing.assert_array_equal(load_library_totals(totals), [100, 200, 300, 400])
        for payload in [b"", b"PK\x03\x04truncated"]:
            totals.write_bytes(payload)
            with self.assertRaises(ValueError):
                load_library_totals(totals)
        archive = self.save()
        with self.assertRaisesRegex(ValueError, ".npy array"):
            load_library_totals(archive)

    def test_base_order_and_empty_base(self):
        path = self.root / "base.txt"
        path.write_text("G3\n\nG1\n", encoding="utf-8")
        self.assertEqual(read_base(path), ["G3", "G1"])
        path.write_text("", encoding="utf-8")
        self.assertEqual(read_base(path), [])
        path = self.root / "base.json"
        path.write_text('["G3", "G1"]', encoding="utf-8")
        self.assertEqual(read_base(path), ["G3", "G1"])
        for text in ['["G1", "G1"]', '[1]', '{"gene": "G1"}', '[" G1"]']:
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_base(path)

    def test_axis_hash_is_order_sensitive(self):
        self.assertNotEqual(axis_sha256(["G1", "G2"]), axis_sha256(["G2", "G1"]))
        self.assertEqual(axis_sha256(["G1", "G2"]), axis_sha256(np.array(["G1", "G2"])))

    def test_export_refuses_overwrite_and_cleans_failed_new_output(self):
        output = self.root / "result"
        result = SimpleNamespace(genes=["G2", "G1"], indices=[1, 0], base_genes=["G2"],
                                 trace=[], pair_support=[], to_dict=lambda: {"genes": ["G2", "G1"]})
        write_result(result, output, run={"version": "test"})
        self.assertEqual(json.loads((output / "panels.json").read_text()), {"genes": ["G2", "G1"]})
        self.assertIn("1,G2,1,base", (output / "genes.csv").read_text())
        with self.assertRaises(FileExistsError):
            write_result(result, output, run={})
        failed = self.root / "failed"
        with self.assertRaises(ValueError):
            write_result(result, failed, run={"invalid": float("nan")})
        self.assertFalse(failed.exists())

    @unittest.skipUnless(importlib.util.find_spec("anndata"), "optional anndata is not installed")
    def test_h5ad_explicit_layer_and_metadata(self):
        import anndata as ad
        import pandas as pd
        from scipy import sparse

        a = ad.AnnData(sparse.csr_matrix(self.arrays["X"]),
                       obs=pd.DataFrame({"celltype": self.arrays["y"], "donor": self.arrays["groups"],
                                         "totals": [100, 200, 300, 400]}, index=["c0", "c1", "c2", "c3"]),
                       var=pd.DataFrame(index=self.arrays["genes"]))
        a.layers["normalized"] = a.X.astype(float) / 2
        path = self.root / "input.h5ad"
        a.write_h5ad(path)
        original_hash = file_sha256(path)
        data = load_h5ad(path, label_key="celltype", group_key="donor", layer="normalized",
                         library_total_key="totals")
        np.testing.assert_array_equal(data.X.toarray(), self.arrays["X"] / 2)
        np.testing.assert_array_equal(data.library_totals, [100, 200, 300, 400])
        with self.assertRaisesRegex(ValueError, "Missing H5AD layer"):
            load_h5ad(path, label_key="celltype", group_key="donor", layer="wrong")
        with self.assertRaisesRegex(ValueError, "Missing H5AD obs"):
            load_h5ad(path, label_key="wrong", group_key="donor")
        with self.assertRaisesRegex(ValueError, "Missing H5AD obs"):
            load_h5ad(path, label_key="celltype", group_key="donor", library_total_key="wrong")
        self.assertEqual(file_sha256(path), original_hash)
        for axis in ["cells", "genes"]:
            duplicate = a.copy()
            if axis == "cells":
                duplicate.obs_names = ["duplicate"] * duplicate.n_obs
            else:
                duplicate.var_names = ["duplicate"] * duplicate.n_vars
            duplicate_path = self.root / f"duplicate_{axis}.h5ad"
            duplicate.write_h5ad(duplicate_path)
            before = file_sha256(duplicate_path)
            with self.assertRaisesRegex(ValueError, "must be unique"):
                load_h5ad(duplicate_path, label_key="celltype", group_key="donor")
            self.assertEqual(file_sha256(duplicate_path), before)
        missing = a.copy()
        missing.obs.loc["c0", "celltype"] = None
        missing_path = self.root / "missing_label.h5ad"
        missing.write_h5ad(missing_path)
        before = file_sha256(missing_path)
        with self.assertRaisesRegex(ValueError, "Missing values"):
            load_h5ad(missing_path, label_key="celltype", group_key="donor")
        self.assertEqual(file_sha256(missing_path), before)


if __name__ == "__main__":
    unittest.main()
