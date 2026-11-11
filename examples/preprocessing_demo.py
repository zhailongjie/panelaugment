"""Offline synthetic example: explicit training QC, then fixed-base PA selection.

Run after installing PanelAugment: python examples/preprocessing_demo.py
These synthetic genes and donors are unrelated to the manuscript BP registry.
"""

import json

import numpy as np

from panelaugment import PanelAugmenter
from panelaugment.preprocessing import CandidatePreprocessor


def main():
    rng = np.random.default_rng(2026)
    genes = [f"toy_gene_{i}" for i in range(8)]
    groups = np.repeat(["train_a", "train_b", "held_out"], 40)
    labels = np.tile(np.repeat(["class_a", "class_b"], 20), 3)
    means = np.full((120, 8), 2.0)
    means[labels == "class_a", :2] += 6
    means[labels == "class_b", 2:4] += 6
    counts = rng.poisson(means)
    fit_mask = groups != "held_out"

    # Synthetic target measurement aggregates, one row per independent donor.
    # The gate flags toy_gene_7; no class labels enter that decision.
    measurement_positive = np.full((4, len(genes)), 50, dtype=int)
    measurement_positive[:, 7] = [4, 4, 91, 91]

    # Counts contain the complete normalization universe in this toy example.
    # For a measured subset, pass full library totals saved before subsetting.
    preprocessor = CandidatePreprocessor().fit(counts, genes, fit_mask=fit_mask,
        mandatory_genes=["toy_gene_0"], measurement_positive_counts=measurement_positive,
        measurement_cell_counts=[100] * 4, measurement_genes=genes)
    training = preprocessor.transform(counts[fit_mask], genes)
    selector = PanelAugmenter().fit(training, labels[fit_mask], groups[fit_mask],
                                    gene_names=preprocessor.genes_)
    panel = selector.select(["toy_gene_0"], n_added=2)

    # Apply the frozen axis to new cells. No held-out labels or expressions fitted
    # the training QC or PA moments. The measurement gate above separately uses
    # target detection aggregates, so the complete pipeline is target-informed.
    # This is selection-space normalization;
    # classifier preprocessing, if performed, needs its own panel-total contract.
    held_out = preprocessor.transform(counts[~fit_mask], genes)
    held_out_selected = held_out[:, panel.indices]
    print(json.dumps({"training_cells": int(fit_mask.sum()),
                      "candidate_genes": list(preprocessor.genes_),
                      "measurement_gate": preprocessor.measurement_screen_["status"],
                      "measurement_exclusions": preprocessor.measurement_screen_["excluded_genes"],
                      "selected_panel": panel.genes,
                      "held_out_selected_shape": list(held_out_selected.shape)}, indent=2))

if __name__ == "__main__":
    main()
