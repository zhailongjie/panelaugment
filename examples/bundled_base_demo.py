"""Read a numbered base and use it with explicitly synthetic expression.

The bundled gene identifiers are real benchmark identifiers. The generated
counts below are synthetic and do not reproduce a manuscript experiment.
"""

import argparse
import json

import numpy as np

from panelaugment import PanelAugmenter, log_normalize
from panelaugment.base_panels import list_base_panels, load_base_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-id", default="BP-11010510")
    args = parser.parse_args()
    panel = load_base_panel(args.panel_id)
    print(json.dumps(panel.metadata, indent=2))
    print(f"Available numbered bases: {len(list_base_panels())}")

    # These extra names denote synthetic candidate genes, not measured genes.
    genes = list(panel.genes) + [f"synthetic_candidate_{i}" for i in range(8)]
    rng = np.random.default_rng(2026)
    labels = np.tile(np.repeat(["type_a", "type_b"], 10), 3)
    groups = np.repeat(["group_a", "group_b", "group_c"], 20)
    rates = np.full((len(labels), len(genes)), 2.0)
    rates[labels == "type_a", -4:] = 6.0
    counts = rng.poisson(rates)
    X = log_normalize(counts)
    result = PanelAugmenter().fit(X, labels, groups, gene_names=genes).select(panel.genes, n_added=3)
    assert tuple(result.genes[:panel.n_genes]) == panel.genes
    print(json.dumps({"base_panel_id": panel.panel_id, "base_genes_retained": panel.n_genes,
                      "synthetic_added_genes": result.added_genes}, indent=2))


if __name__ == "__main__":
    main()
