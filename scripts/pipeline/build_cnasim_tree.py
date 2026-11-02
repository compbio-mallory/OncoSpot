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
    write_malignant_cnasim_tree,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the malignant-only CNAsim tree "
            "from the canonical DEMON lineage."
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

    clone_tree_path = (
        run_dir
        / "scaffold"
        / "clone_tree.tsv"
    )

    cells_path = (
        run_dir
        / "cells"
        / "cells_with_cell_types.tsv"
    )

    output_dir = (
        run_dir
        / "cna"
        / "bridge"
    )

    result = write_malignant_cnasim_tree(
        clone_tree_path=clone_tree_path,
        cells_path=cells_path,
        output_dir=output_dir,
    )

    summary = result.summary

    print()
    print("=" * 80)
    print("MALIGNANT CNASIM TREE COMPLETE")
    print("=" * 80)

    print(
        "Original DEMON tree nodes:",
        summary[
            "original_demon_tree_nodes"
        ],
    )

    print(
        "Observed malignant subclones:",
        summary[
            "observed_malignant_subclones"
        ],
    )

    print(
        "Retained CNAsim tree nodes:",
        summary[
            "retained_cnasim_tree_nodes"
        ],
    )

    print(
        "Ancestor-only nodes:",
        summary[
            "ancestor_only_nodes"
        ],
    )

    print(
        "Pruned irrelevant nodes:",
        summary[
            "pruned_irrelevant_nodes"
        ],
    )

    print(
        "Malignant cells:",
        f'{summary["malignant_cells"]:,}',
    )

    print(
        "Population conserved:",
        summary[
            "malignant_population_conserved"
        ],
    )

    print(
        "All malignant clones mapped:",
        summary[
            "all_observed_malignant_clones_mapped"
        ],
    )

    print()
    print(
        "Output:",
        output_dir,
    )
    print()


if __name__ == "__main__":
    main()
