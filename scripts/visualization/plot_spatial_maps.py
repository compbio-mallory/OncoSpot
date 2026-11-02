#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


POINT_SIZE = 6
FIGSIZE = (9, 9)
DPI = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def setup_axis(ax):
    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel(
        "Spatial x",
        fontsize=14,
    )

    ax.set_ylabel(
        "Spatial y",
        fontsize=14,
    )

    ax.tick_params(
        axis="both",
        labelsize=11,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_cell_types(
    cells: pd.DataFrame,
    palette: pd.DataFrame,
    output: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    #
    # Plot malignant first so smaller TME populations
    # remain visible on top.
    #
    order = [
        "malignant",
        "T_cell",
        "myeloid_macrophage",
        "fibroblast",
        "B_plasma",
        "endothelial",
        "normal_epithelial",
    ]

    palette_map = dict(
        zip(
            palette["cell_type"],
            palette["color"],
        )
    )

    for cell_type in order:

        x = cells.loc[
            cells["cell_type"]
            == cell_type
        ]

        if x.empty:
            continue

        ax.scatter(
            x["x"],
            x["y"],
            s=POINT_SIZE,
            c=palette_map[cell_type],
            linewidths=0,
            alpha=0.9,
            rasterized=True,
            label=cell_type,
        )

    setup_axis(ax)

    ax.set_title(
        "Spatial Distribution of Cell Types",
        fontsize=17,
        pad=12,
    )

    handles = []

    for cell_type in order:

        if cell_type not in palette_map:
            continue

        n = int(
            (
                cells["cell_type"]
                == cell_type
            ).sum()
        )

        if n == 0:
            continue

        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor=(
                    palette_map[
                        cell_type
                    ]
                ),
                markeredgecolor="none",
                label=(
                    f"{cell_type} "
                    f"(n={n:,})"
                ),
            )
        )

    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(
            1.02,
            0.5,
        ),
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_malignant_vs_nonmalignant(
    cells: pd.DataFrame,
    output: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    colors = {
        "malignant":
            "#B2182B",

        "nonmalignant":
            "#BDBDBD",
    }

    #
    # Nonmalignant first, tumor cells on top.
    #
    order = [
        "nonmalignant",
        "malignant",
    ]

    for label in order:

        x = cells.loc[
            cells[
                "malignancy_label"
            ]
            == label
        ]

        ax.scatter(
            x["x"],
            x["y"],
            s=POINT_SIZE,
            c=colors[label],
            linewidths=0,
            alpha=0.9,
            rasterized=True,
        )

    setup_axis(ax)

    ax.set_title(
        "Malignant vs Nonmalignant Cells",
        fontsize=17,
        pad=12,
    )

    handles = []

    for label in [
        "malignant",
        "nonmalignant",
    ]:

        n = int(
            (
                cells[
                    "malignancy_label"
                ]
                == label
            ).sum()
        )

        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor=(
                    colors[label]
                ),
                markeredgecolor="none",
                label=(
                    f"{label} "
                    f"(n={n:,})"
                ),
            )
        )

    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(
            1.02,
            0.5,
        ),
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_malignant_subclones(
    cells: pd.DataFrame,
    clone_palette: pd.DataFrame,
    output: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    nonmalignant = cells.loc[
        ~cells[
            "is_malignant"
        ].astype(bool)
    ]

    malignant = cells.loc[
        cells[
            "is_malignant"
        ].astype(bool)
    ]

    #
    # TME background.
    #
    ax.scatter(
        nonmalignant["x"],
        nonmalignant["y"],
        s=POINT_SIZE,
        c="#D9D9D9",
        linewidths=0,
        alpha=0.45,
        rasterized=True,
    )

    clone_palette = (
        clone_palette
        .sort_values(
            "n_malignant_cells",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    #
    # Plot smallest clones first so large clones do not
    # completely hide them.
    #
    plotting_order = (
        clone_palette
        .sort_values(
            "n_malignant_cells",
            ascending=True,
        )
    )

    for _, row in (
        plotting_order.iterrows()
    ):

        clone_id = str(
            row["subclone_id"]
        )

        color = str(
            row["color"]
        )

        x = malignant.loc[
            malignant[
                "subclone_id"
            ].astype(str)
            == clone_id
        ]

        if x.empty:
            continue

        ax.scatter(
            x["x"],
            x["y"],
            s=POINT_SIZE,
            c=color,
            linewidths=0,
            alpha=0.9,
            rasterized=True,
        )

    setup_axis(ax)

    ax.set_title(
        "Spatial Distribution of Malignant Subclones",
        fontsize=17,
        pad=12,
    )

    #
    # 64 clone labels would make the figure unreadable.
    # Show the 15 most abundant clones in the legend.
    #
    top = clone_palette.head(
        15
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=str(
                row["color"]
            ),
            markeredgecolor="none",
            label=(
                f"{row['subclone_id']} "
                f"(n={int(row['n_malignant_cells']):,})"
            ),
        )
        for _, row
        in top.iterrows()
    ]

    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor="#D9D9D9",
            markeredgecolor="none",
            label="nonmalignant",
        )
    )

    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(
            1.02,
            0.5,
        ),
        frameon=False,
        fontsize=9,
        title=(
            "Top 15 clones\n"
            "(15 largest listed)"
        ),
        title_fontsize=10,
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:

    args = parse_args()

    run_dir = (
        args.run_dir
        .resolve()
    )

    spatial_dir = (
        run_dir
        / "spatial"
    )

    plot_dir = (
        spatial_dir
        / "plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cells = pd.read_csv(
        spatial_dir
        / "cells_spatial.tsv",
        sep="\t",
        low_memory=False,
    )

    cell_type_palette = (
        pd.read_csv(
            spatial_dir
            / "cell_type_palette.tsv",
            sep="\t",
        )
    )

    clone_palette = (
        pd.read_csv(
            spatial_dir
            / "subclone_palette.tsv",
            sep="\t",
        )
    )

    print()
    print("=" * 80)
    print("GENERATING SPATIAL MAPS")
    print("=" * 80)

    plot_cell_types(
        cells,
        cell_type_palette,
        plot_dir
        / "cell_types.png",
    )

    print(
        "Created: cell_types.png"
    )

    plot_malignant_subclones(
        cells,
        clone_palette,
        plot_dir
        / "malignant_subclones.png",
    )

    print(
        "Created: malignant_subclones.png"
    )

    plot_malignant_vs_nonmalignant(
        cells,
        plot_dir
        / "malignant_vs_nonmalignant.png",
    )

    print(
        "Created: malignant_vs_nonmalignant.png"
    )

    print()
    print(
        "Output directory:",
        plot_dir,
    )


if __name__ == "__main__":
    main()
