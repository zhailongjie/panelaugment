"""Run after installing panelaugment; this uses generated counts only."""

from panelaugment import PanelAugmenter, log_normalize
from panelaugment.cli import synthetic_data


def main():
    data = synthetic_data()
    # Here all generated genes are measured, so the full input defines totals.
    normalized = log_normalize(data.X)
    model = PanelAugmenter().fit(normalized, data.y, data.groups, gene_names=data.genes)
    result = model.select(base_genes=["Gene00", "Gene03"], n_added=4)
    print("Full panel:", result.genes)
    print("Added genes:", result.added_genes)
    print("Smaller budget:", result.prefix(2).genes)


if __name__ == "__main__":
    main()
