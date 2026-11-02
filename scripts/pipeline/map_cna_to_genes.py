#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(
        PROJECT_ROOT
        / "src"
    ),
)

from cancer_st_sim.cna import (  # noqa: E402
    write_gene_level_cna,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        required=True,
    )

    args = parser.parse_args()

    run_dir = Path(
        args.run_dir
    ).resolve()

    result = write_gene_level_cna(
        genes_path=(
            PROJECT_ROOT
            / "data"
            / "annotations"
            / "gencode"
            / "v50"
            / "processed"
            / "genes_autosomes.tsv"
        ),

        clone_bin_cna_path=(
            run_dir
            / "cna"
            / "processed"
            / "clone_bin_cna.tsv"
        ),

        output_dir=(
            run_dir
            / "cna"
            / "gene_level"
        ),
    )

    summary = result.summary

    print()
    print("=" * 80)
    print("GENE-LEVEL CNA MAPPING COMPLETE")
    print("=" * 80)

    print(
        "Input GENCODE genes:",
        f'{summary["n_input_genes"]:,}',
    )

    print(
        "Mapped genes:",
        f'{summary["n_mapped_genes"]:,}',
    )

    print(
        "Unmapped genes:",
        f'{summary["n_unmapped_genes"]:,}',
    )

    print(
        "Lineage nodes:",
        summary[
            "n_lineage_nodes"
        ],
    )

    print(
        "Clone-gene rows:",
        f'{summary["clone_gene_cna_rows"]:,}',
    )

    print(
        "Genes spanning >1 CNA bin:",
        f'{summary["genes_spanning_multiple_bins"]:,}',
    )

    print(
        "Genes varying across malignant clones:",
        f'{summary["genes_with_cna_variation_across_observed_clones"]:,}',
    )

    print(
        "Malignant cell-weighted mean gene CN:",
        f'{summary["mean_total_cn_across_malignant_cell_gene_pairs"]:.4f}',
    )

    print()


if __name__ == "__main__":
    main()
