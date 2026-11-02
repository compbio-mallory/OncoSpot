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
    write_canonical_cnasim_output,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Map raw CNAsim profiles back to "
            "biological DEMON malignant subclones."
        )
    )

    parser.add_argument(
        "--run-dir",
        required=True,
    )

    args = parser.parse_args()

    run_dir = Path(
        args.run_dir
    ).resolve()

    result = write_canonical_cnasim_output(
        cnasim_dir=(
            run_dir
            / "cna"
            / "cnasim_raw"
        ),
        node_mapping_path=(
            run_dir
            / "cna"
            / "bridge"
            / "node_mapping.tsv"
        ),
        cells_path=(
            run_dir
            / "cells"
            / "cells_with_cell_types.tsv"
        ),
        output_dir=(
            run_dir
            / "cna"
            / "processed"
        ),
    )

    summary = result.summary

    burden = summary[
        "population_weighted_cna_burden"
    ]

    print()
    print("=" * 80)
    print("CNASIM OUTPUT ADAPTER COMPLETE")
    print("=" * 80)

    print(
        "CNAsim lineage nodes:",
        summary[
            "n_cnasim_nodes"
        ],
    )

    print(
        "Observed malignant clones:",
        summary[
            "n_observed_malignant_clones"
        ],
    )

    print(
        "Ancestor-only nodes:",
        summary[
            "n_ancestor_only_nodes"
        ],
    )

    print(
        "Genome bins / node:",
        f'{summary["n_genome_bins_per_node"]:,}',
    )

    print(
        "Malignant cells mapped:",
        f'{summary["malignant_cells_with_cna_profile"]:,}',
    )

    print(
        "All malignant cells mapped:",
        summary[
            "all_malignant_cells_mapped"
        ],
    )

    print()
    print("Population-weighted CNA burden:")

    print(
        "  altered bins:",
        f'{burden["altered_bin_fraction"]:.4f}',
    )

    print(
        "  deletions:",
        f'{burden["deletion_bin_fraction"]:.4f}',
    )

    print(
        "  amplifications:",
        f'{burden["amplification_bin_fraction"]:.4f}',
    )

    print(
        "  CN=0:",
        f'{burden["homozygous_deletion_bin_fraction"]:.4f}',
    )

    print(
        "  mean total CN:",
        f'{burden["mean_total_cn"]:.4f}',
    )

    print()


if __name__ == "__main__":
    main()
