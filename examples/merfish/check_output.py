"""Compare this example's ordered output with the accompanying reference panel."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path, help="panels.json from the selection run")
    args = parser.parse_args()
    expected = json.loads(Path(__file__).with_name("expected_panel.json").read_text())
    actual = json.loads(args.panel.read_text())
    for key in ("base_genes", "added_genes", "genes", "n_added", "total_size", "ridge"):
        if actual.get(key) != expected[key]:
            raise ValueError(f"Output differs from the example for {key}")
    print("500 base + 10 added genes: exact match, including order.")


if __name__ == "__main__":
    main()
