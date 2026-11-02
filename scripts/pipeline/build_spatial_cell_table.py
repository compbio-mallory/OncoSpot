#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import yaml


CELL_TYPE_COLORS = {
    "malignant": "#B2182B",
    "T_cell": "#2166AC",
    "myeloid_macrophage": "#4D9221",
    "fibroblast": "#D95F02",
    "B_plasma": "#762A83",
    "endothelial": "#1B9E77",
    "normal_epithelial": "#E6AB02",
}

NONMALIGNANT_CLONE_COLOR = "#D9D9D9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def deterministic_clone_color(clone_id: str) -> str:
    """
    Deterministic RGB color derived from clone ID.

    Avoid extremely dark or extremely pale colors by mapping
    hash bytes into approximately 60..210.
    """

    digest = hashlib.sha256(
        clone_id.encode("utf-8")
    ).digest()

    rgb = [
        60 + (digest[i] % 151)
        for i in range(3)
    ]

    return "#{:02X}{:02X}{:02X}".format(
        *rgb
    )


def main() -> None:

    args = parse_args()

    run_dir = args.run_dir.resolve()

    input_file = (
        run_dir
        / "cells"
        / "cells_with_cell_types.tsv"
    )

    latent_cells_file = (
        run_dir
        / "expression"
        / "latent"
        / "cells.tsv"
    )

    output_dir = (
        run_dir
        / "spatial"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_cells = (
        output_dir
        / "cells_spatial.tsv"
    )

    cell_type_palette_file = (
        output_dir
        / "cell_type_palette.tsv"
    )

    clone_palette_file = (
        output_dir
        / "subclone_palette.tsv"
    )

    summary_file = (
        output_dir
        / "summary.yaml"
    )

    print()
    print("=" * 80)
    print("BUILDING SPATIAL CELL TABLE")
    print("=" * 80)

    cells = pd.read_csv(
        input_file,
        sep="\t",
        low_memory=False,
    )

    required = {
        "cell_id",
        "deme_id",
        "deme_x",
        "deme_y",
        "x",
        "y",
        "subclone_id",
        "compartment",
        "is_malignant",
        "cell_type",
    }

    missing = (
        required
        - set(cells.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing required cell columns: "
            + ", ".join(sorted(missing))
        )

    if cells["cell_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate cell IDs in canonical cell table."
        )

    #
    # ------------------------------------------------------------
    # Validate exact alignment with latent expression object.
    # ------------------------------------------------------------
    #

    if not latent_cells_file.exists():
        raise FileNotFoundError(
            f"Latent cells file not found: {latent_cells_file}"
        )

    latent_cells = pd.read_csv(
        latent_cells_file,
        sep="\t",
        usecols=["cell_id"],
    )

    if len(latent_cells) != len(cells):
        raise RuntimeError(
            f"Spatial cells={len(cells):,}, "
            f"latent cells={len(latent_cells):,}."
        )

    if not (
        latent_cells["cell_id"].astype(str).to_numpy()
        ==
        cells["cell_id"].astype(str).to_numpy()
    ).all():
        raise RuntimeError(
            "Canonical spatial cell order does not match "
            "latent expression cell order."
        )

    cells.insert(
        0,
        "matrix_row_index",
        range(len(cells)),
    )

    #
    # ------------------------------------------------------------
    # Cell-type colors.
    # ------------------------------------------------------------
    #

    observed_types = set(
        cells["cell_type"]
        .dropna()
        .astype(str)
        .unique()
    )

    unknown_types = (
        observed_types
        - set(CELL_TYPE_COLORS)
    )

    if unknown_types:
        raise RuntimeError(
            "No configured color for cell types: "
            + ", ".join(sorted(unknown_types))
        )

    cells["cell_type_color"] = (
        cells["cell_type"]
        .astype(str)
        .map(CELL_TYPE_COLORS)
    )

    #
    # ------------------------------------------------------------
    # Malignant subclone colors.
    # ------------------------------------------------------------
    #

    malignant_mask = (
        cells["is_malignant"]
        .astype(bool)
    )

    malignant_clones = sorted(
        cells.loc[
            malignant_mask,
            "subclone_id",
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    clone_colors = {
        clone_id:
            deterministic_clone_color(
                clone_id
            )
        for clone_id in malignant_clones
    }

    cells["subclone_color"] = (
        NONMALIGNANT_CLONE_COLOR
    )

    cells.loc[
        malignant_mask,
        "subclone_color",
    ] = (
        cells.loc[
            malignant_mask,
            "subclone_id",
        ]
        .astype(str)
        .map(clone_colors)
    )

    #
    # ------------------------------------------------------------
    # Useful plotting labels.
    # ------------------------------------------------------------
    #

    cells["malignancy_label"] = (
        cells["is_malignant"]
        .astype(bool)
        .map(
            {
                True: "malignant",
                False: "nonmalignant",
            }
        )
    )

    cells["clone_plot_label"] = (
        "nonmalignant"
    )

    cells.loc[
        malignant_mask,
        "clone_plot_label",
    ] = (
        cells.loc[
            malignant_mask,
            "subclone_id",
        ].astype(str)
    )

    #
    # ------------------------------------------------------------
    # Save canonical spatial table.
    # ------------------------------------------------------------
    #

    preferred_columns = [
        "matrix_row_index",
        "cell_id",
        "x",
        "y",
        "deme_id",
        "deme_x",
        "deme_y",
        "compartment",
        "is_malignant",
        "malignancy_label",
        "cell_type",
        "cell_type_color",
        "subclone_id",
        "clone_plot_label",
        "subclone_color",
        "driver_genotype_id",
        "source_demon_subclone_id",
        "source_demon_driver_genotype_id",
        "nonmalignant_spatial_score",
    ]

    output_columns = [
        column
        for column in preferred_columns
        if column in cells.columns
    ]

    extra_columns = [
        column
        for column in cells.columns
        if column not in output_columns
    ]

    cells[
        output_columns
        + extra_columns
    ].to_csv(
        output_cells,
        sep="\t",
        index=False,
    )

    #
    # ------------------------------------------------------------
    # Cell-type palette.
    # ------------------------------------------------------------
    #

    cell_type_counts = (
        cells["cell_type"]
        .value_counts()
        .to_dict()
    )

    cell_type_palette = pd.DataFrame(
        [
            {
                "cell_type": cell_type,
                "color": color,
                "n_cells": int(
                    cell_type_counts.get(
                        cell_type,
                        0,
                    )
                ),
            }
            for cell_type, color
            in CELL_TYPE_COLORS.items()
        ]
    )

    cell_type_palette.to_csv(
        cell_type_palette_file,
        sep="\t",
        index=False,
    )

    #
    # ------------------------------------------------------------
    # Clone palette.
    # ------------------------------------------------------------
    #

    clone_counts = (
        cells.loc[
            malignant_mask,
            "subclone_id",
        ]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    clone_palette = pd.DataFrame(
        [
            {
                "subclone_id":
                    clone_id,

                "color":
                    clone_colors[
                        clone_id
                    ],

                "n_malignant_cells":
                    int(
                        clone_counts.get(
                            clone_id,
                            0,
                        )
                    ),
            }
            for clone_id
            in malignant_clones
        ]
    )

    clone_palette.to_csv(
        clone_palette_file,
        sep="\t",
        index=False,
    )

    #
    # ------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------
    #

    summary = {
        "stage":
            "spatial_cell_table",

        "n_cells":
            int(len(cells)),

        "n_malignant_cells":
            int(
                malignant_mask.sum()
            ),

        "n_nonmalignant_cells":
            int(
                (~malignant_mask).sum()
            ),

        "n_cell_types":
            int(
                cells[
                    "cell_type"
                ].nunique()
            ),

        "n_malignant_subclones":
            int(
                len(
                    malignant_clones
                )
            ),

        "coordinate_columns": {
            "continuous":
                [
                    "x",
                    "y",
                ],

            "deme":
                [
                    "deme_x",
                    "deme_y",
                ],
        },

        "cell_type_counts": {
            str(key):
                int(value)
            for key, value
            in cells[
                "cell_type"
            ]
            .value_counts()
            .items()
        },

        "coordinate_ranges": {
            "x_min":
                float(
                    cells["x"].min()
                ),

            "x_max":
                float(
                    cells["x"].max()
                ),

            "y_min":
                float(
                    cells["y"].min()
                ),

            "y_max":
                float(
                    cells["y"].max()
                ),
        },

        "palette_policy": {
            "cell_type":
                "fixed_named_palette",

            "subclone":
                (
                    "deterministic_SHA256_"
                    "derived_RGB"
                ),

            "nonmalignant_clone_color":
                NONMALIGNANT_CLONE_COLOR,
        },

        "latent_expression_alignment":
            "exact_cell_id_order_match_passed",

        "outputs": {
            "cells_spatial":
                "spatial/cells_spatial.tsv",

            "cell_type_palette":
                "spatial/cell_type_palette.tsv",

            "subclone_palette":
                "spatial/subclone_palette.tsv",
        },
    }

    with summary_file.open(
        "w",
        encoding="utf-8",
    ) as handle:

        yaml.safe_dump(
            summary,
            handle,
            sort_keys=False,
        )

    print(
        "Cells:",
        f"{len(cells):,}",
    )

    print(
        "Malignant:",
        f"{malignant_mask.sum():,}",
    )

    print(
        "Nonmalignant:",
        f"{(~malignant_mask).sum():,}",
    )

    print(
        "Malignant subclones:",
        len(malignant_clones),
    )

    print()
    print("Cell types:")

    print(
        cells[
            "cell_type"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Output:",
        output_cells,
    )

    print(
        "Cell-type palette:",
        cell_type_palette_file,
    )

    print(
        "Clone palette:",
        clone_palette_file,
    )

    print(
        "Summary:",
        summary_file,
    )


if __name__ == "__main__":
    main()
