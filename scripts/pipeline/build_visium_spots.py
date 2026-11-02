#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import ConvexHull, cKDTree


RADIUS_TO_SPACING = 27.5 / 100.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Construct a classic-Visium-like staggered "
            "capture-spot lattice over latent simulator cells."
        )
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        help="Completed simulator run directory.",
    )

    parser.add_argument(
        "--target-median-cells",
        type=float,
        default=5.0,
        help=(
            "Target median number of latent cells "
            "per occupied capture spot."
        ),
    )

    parser.add_argument(
        "--n-scale-candidates",
        type=int,
        default=41,
        help="Number of lattice scales tested.",
    )

    parser.add_argument(
        "--min-scale-factor",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--max-scale-factor",
        type=float,
        default=2.25,
    )

    return parser.parse_args()


def tissue_area(coords: np.ndarray) -> float:
    """
    Estimate tissue area in simulator-coordinate units.
    """
    if len(coords) < 3:
        raise ValueError(
            "At least three cells are required."
        )

    try:
        hull = ConvexHull(coords)
        area = float(hull.volume)
    except Exception:
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)

        area = float(
            (xmax - xmin)
            * (ymax - ymin)
        )

    if not np.isfinite(area) or area <= 0:
        raise ValueError(
            f"Invalid tissue area: {area}"
        )

    return area


def make_lattice(
    coords: np.ndarray,
    spacing: float,
):
    """
    Hexagonally staggered spot-center lattice.

    Nearest-neighbor center spacing = spacing.
    Effective spot radius = 0.275 * spacing.
    """

    radius = (
        RADIUS_TO_SPACING
        * spacing
    )

    vertical_spacing = (
        math.sqrt(3.0)
        / 2.0
        * spacing
    )

    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)

    x_start = xmin - radius - spacing
    x_stop = xmax + radius + spacing

    y_start = ymin - radius - vertical_spacing
    y_stop = ymax + radius + vertical_spacing

    centers = []
    rows = []
    cols = []

    row_index = 0
    y = y_start

    while y <= y_stop:
        shift = (
            0.5 * spacing
            if row_index % 2 == 1
            else 0.0
        )

        x = x_start + shift
        col_index = 0

        while x <= x_stop:
            centers.append(
                (x, y)
            )

            rows.append(
                row_index
            )

            cols.append(
                col_index
            )

            x += spacing
            col_index += 1

        y += vertical_spacing
        row_index += 1

    return (
        np.asarray(
            centers,
            dtype=float,
        ),
        np.asarray(
            rows,
            dtype=int,
        ),
        np.asarray(
            cols,
            dtype=int,
        ),
        radius,
    )


def evaluate_spacing(
    tree: cKDTree,
    coords: np.ndarray,
    spacing: float,
):
    centers, rows, cols, radius = (
        make_lattice(
            coords,
            spacing,
        )
    )

    memberships = (
        tree.query_ball_point(
            centers,
            r=radius,
        )
    )

    counts = np.fromiter(
        (
            len(x)
            for x in memberships
        ),
        dtype=int,
        count=len(memberships),
    )

    occupied = counts > 0

    occupied_counts = (
        counts[occupied]
    )

    if len(occupied_counts) == 0:
        return {
            "spacing": spacing,
            "radius": radius,
            "n_lattice_spots": len(counts),
            "n_occupied_spots": 0,
            "median_cells": np.nan,
            "mean_cells": np.nan,
            "q25_cells": np.nan,
            "q75_cells": np.nan,
            "max_cells": 0,
            "captured_cell_assignments": 0,
        }

    return {
        "spacing": float(spacing),
        "radius": float(radius),
        "n_lattice_spots": int(len(counts)),
        "n_occupied_spots": int(
            occupied.sum()
        ),
        "median_cells": float(
            np.median(
                occupied_counts
            )
        ),
        "mean_cells": float(
            np.mean(
                occupied_counts
            )
        ),
        "q25_cells": float(
            np.quantile(
                occupied_counts,
                0.25,
            )
        ),
        "q75_cells": float(
            np.quantile(
                occupied_counts,
                0.75,
            )
        ),
        "max_cells": int(
            occupied_counts.max()
        ),
        "captured_cell_assignments": int(
            occupied_counts.sum()
        ),
    }


def choose_spacing(
    coords: np.ndarray,
    target_median: float,
    n_candidates: int,
    min_factor: float,
    max_factor: float,
):
    area = tissue_area(
        coords
    )

    density = (
        len(coords)
        / area
    )

    # Initial approximation:
    #
    # target ~= density * pi * radius^2
    # radius = 0.275 * spacing
    #
    # solve for spacing.
    initial_spacing = math.sqrt(
        target_median
        / (
            density
            * math.pi
            * RADIUS_TO_SPACING**2
        )
    )

    factors = np.geomspace(
        min_factor,
        max_factor,
        n_candidates,
    )

    spacings = (
        initial_spacing
        * factors
    )

    tree = cKDTree(
        coords
    )

    records = []

    for spacing in spacings:
        record = evaluate_spacing(
            tree=tree,
            coords=coords,
            spacing=float(spacing),
        )

        records.append(
            record
        )

    search = pd.DataFrame(
        records
    )

    valid = search.loc[
        search[
            "median_cells"
        ].notna()
    ].copy()

    if valid.empty:
        raise RuntimeError(
            "No occupied Visium-like spots "
            "were produced during scale search."
        )

    valid["target_error"] = (
        valid["median_cells"]
        - target_median
    ).abs()

    valid["initial_scale_error"] = (
        np.log(
            valid["spacing"]
            / initial_spacing
        )
    ).abs()

    best = (
        valid.sort_values(
            [
                "target_error",
                "initial_scale_error",
            ]
        )
        .iloc[0]
    )

    return (
        float(best["spacing"]),
        float(initial_spacing),
        area,
        density,
        search,
    )


def dominant_label(series):
    clean = (
        series
        .dropna()
        .astype(str)
    )

    clean = clean.loc[
        ~clean.isin(
            [
                "",
                "nan",
                "None",
            ]
        )
    ]

    if clean.empty:
        return ""

    return (
        clean
        .value_counts()
        .index[0]
    )


def main():
    args = parse_args()

    run_dir = Path(
        args.run_dir
    ).resolve()

    spatial_path = (
        run_dir
        / "spatial"
        / "cells_spatial.tsv"
    )

    if not spatial_path.exists():
        raise FileNotFoundError(
            spatial_path
        )

    output_dir = (
        run_dir
        / "observation"
        / "visium"
    )

    plot_dir = (
        output_dir
        / "plots"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading:")
    print(spatial_path)

    cells = pd.read_csv(
        spatial_path,
        sep="\t",
    )

    required_columns = {
        "matrix_row_index",
        "cell_id",
        "x",
        "y",
        "is_malignant",
        "cell_type",
        "subclone_id",
    }

    missing = (
        required_columns
        - set(cells.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    cells = (
        cells.sort_values(
            "matrix_row_index"
        )
        .reset_index(drop=True)
    )

    expected_rows = np.arange(
        len(cells)
    )

    if not np.array_equal(
        cells[
            "matrix_row_index"
        ].to_numpy(),
        expected_rows,
    ):
        raise ValueError(
            "matrix_row_index is not "
            "canonical 0..N-1 order."
        )

    coords = cells[
        ["x", "y"]
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        coords
    ).all():
        raise ValueError(
            "Non-finite coordinates detected."
        )

    # --------------------------------------------------------------
    # Search simulator-coordinate lattice scale.
    # --------------------------------------------------------------

    (
        selected_spacing,
        initial_spacing,
        area,
        density,
        search,
    ) = choose_spacing(
        coords=coords,
        target_median=(
            args.target_median_cells
        ),
        n_candidates=(
            args.n_scale_candidates
        ),
        min_factor=(
            args.min_scale_factor
        ),
        max_factor=(
            args.max_scale_factor
        ),
    )

    search.to_csv(
        output_dir
        / "scale_search.tsv",
        sep="\t",
        index=False,
    )

    print()
    print(
        "Initial spacing estimate:",
        initial_spacing,
    )

    print(
        "Selected spacing:",
        selected_spacing,
    )

    # --------------------------------------------------------------
    # Build final lattice.
    # --------------------------------------------------------------

    (
        centers,
        array_rows,
        array_cols,
        radius,
    ) = make_lattice(
        coords,
        selected_spacing,
    )

    tree = cKDTree(
        coords
    )

    memberships = (
        tree.query_ball_point(
            centers,
            r=radius,
        )
    )

    counts = np.fromiter(
        (
            len(x)
            for x in memberships
        ),
        dtype=int,
        count=len(memberships),
    )

    occupied_indices = np.flatnonzero(
        counts > 0
    )

    # Because 2R / spacing = 0.55,
    # capture disks cannot overlap.
    # Therefore no cell should belong
    # to more than one spot.

    assignment_rows = []
    spot_rows = []

    old_to_new = {}

    for new_index, lattice_index in enumerate(
        occupied_indices
    ):
        spot_id = (
            f"spot_{new_index:06d}"
        )

        old_to_new[
            lattice_index
        ] = spot_id

        members = memberships[
            lattice_index
        ]

        spot_rows.append(
            {
                "spot_index": new_index,
                "spot_id": spot_id,
                "array_row": int(
                    array_rows[
                        lattice_index
                    ]
                ),
                "array_col": int(
                    array_cols[
                        lattice_index
                    ]
                ),
                "x": float(
                    centers[
                        lattice_index,
                        0,
                    ]
                ),
                "y": float(
                    centers[
                        lattice_index,
                        1,
                    ]
                ),
                "capture_radius": float(
                    radius
                ),
                "center_spacing": float(
                    selected_spacing
                ),
                "n_cells": int(
                    len(members)
                ),
            }
        )

        for cell_index in members:
            assignment_rows.append(
                {
                    "spot_index": new_index,
                    "spot_id": spot_id,
                    "matrix_row_index": int(
                        cells.iloc[
                            cell_index
                        ][
                            "matrix_row_index"
                        ]
                    ),
                    "cell_id": (
                        cells.iloc[
                            cell_index
                        ][
                            "cell_id"
                        ]
                    ),
                }
            )

    spots = pd.DataFrame(
        spot_rows
    )

    assignments = pd.DataFrame(
        assignment_rows
    )

    if assignments.empty:
        raise RuntimeError(
            "No cells were captured "
            "by the final lattice."
        )

    duplicate_cells = (
        assignments[
            "matrix_row_index"
        ]
        .duplicated()
        .sum()
    )

    if duplicate_cells != 0:
        raise RuntimeError(
            "A cell was assigned to "
            "multiple Visium spots. "
            f"Duplicate assignments: "
            f"{duplicate_cells}"
        )

    spots.to_csv(
        output_dir
        / "spots.tsv",
        sep="\t",
        index=False,
    )

    assignments.to_csv(
        output_dir
        / "cell_to_spot.tsv",
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------------

    joined = assignments.merge(
        cells,
        on=[
            "matrix_row_index",
            "cell_id",
        ],
        how="left",
        validate="one_to_one",
    )

    ground_truth_rows = []

    cell_type_rows = []
    subclone_rows = []

    for spot_id, group in joined.groupby(
        "spot_id",
        sort=False,
    ):
        n_cells = len(group)

        malignant_mask = (
            group[
                "is_malignant"
            ].astype(bool)
        )

        malignant = group.loc[
            malignant_mask
        ]

        n_malignant = int(
            malignant_mask.sum()
        )

        n_nonmalignant = int(
            n_cells
            - n_malignant
        )

        # Cell-type composition
        cell_type_counts = (
            group[
                "cell_type"
            ]
            .fillna("unknown")
            .astype(str)
            .value_counts()
        )

        for label, count in (
            cell_type_counts.items()
        ):
            cell_type_rows.append(
                {
                    "spot_id": spot_id,
                    "cell_type": label,
                    "n_cells": int(
                        count
                    ),
                    "fraction": float(
                        count
                        / n_cells
                    ),
                }
            )

        # Malignant subclone composition
        clone_counts = (
            malignant[
                "subclone_id"
            ]
            .dropna()
            .astype(str)
        )

        clone_counts = (
            clone_counts[
                ~clone_counts.isin(
                    [
                        "",
                        "nan",
                        "None",
                    ]
                )
            ]
            .value_counts()
        )

        for clone_id, count in (
            clone_counts.items()
        ):
            subclone_rows.append(
                {
                    "spot_id": spot_id,
                    "subclone_id": clone_id,
                    "n_malignant_cells": int(
                        count
                    ),
                    "fraction_of_malignant": (
                        float(
                            count
                            / n_malignant
                        )
                        if n_malignant > 0
                        else np.nan
                    ),
                    "fraction_of_all_cells": float(
                        count
                        / n_cells
                    ),
                }
            )

        if len(clone_counts):
            dominant_clone = (
                clone_counts.index[0]
            )

            dominant_clone_fraction = (
                float(
                    clone_counts.iloc[0]
                    / n_malignant
                )
            )
        else:
            dominant_clone = ""
            dominant_clone_fraction = (
                np.nan
            )

        ground_truth_rows.append(
            {
                "spot_id": spot_id,
                "n_cells": int(
                    n_cells
                ),
                "n_malignant": (
                    n_malignant
                ),
                "n_nonmalignant": (
                    n_nonmalignant
                ),
                "malignant_fraction": float(
                    n_malignant
                    / n_cells
                ),
                "n_cell_types": int(
                    len(
                        cell_type_counts
                    )
                ),
                "dominant_cell_type": (
                    cell_type_counts.index[
                        0
                    ]
                ),
                "dominant_cell_type_fraction": float(
                    cell_type_counts.iloc[
                        0
                    ]
                    / n_cells
                ),
                "n_malignant_subclones": int(
                    len(
                        clone_counts
                    )
                ),
                "dominant_malignant_subclone": (
                    dominant_clone
                ),
                "dominant_malignant_subclone_fraction": (
                    dominant_clone_fraction
                ),
            }
        )

    ground_truth = pd.DataFrame(
        ground_truth_rows
    )

    ground_truth.to_csv(
        output_dir
        / "spot_ground_truth.tsv",
        sep="\t",
        index=False,
    )

    pd.DataFrame(
        cell_type_rows
    ).to_csv(
        output_dir
        / "spot_cell_type_composition.tsv",
        sep="\t",
        index=False,
    )

    pd.DataFrame(
        subclone_rows
    ).to_csv(
        output_dir
        / "spot_subclone_composition.tsv",
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    occupied_counts = (
        spots[
            "n_cells"
        ].to_numpy()
    )

    captured_cells = int(
        len(assignments)
    )

    summary = {
        "stage": "visium_spot_geometry",
        "n_latent_cells": int(
            len(cells)
        ),
        "n_captured_cells": (
            captured_cells
        ),
        "captured_cell_fraction": float(
            captured_cells
            / len(cells)
        ),
        "n_occupied_spots": int(
            len(spots)
        ),
        "target_median_cells_per_spot": float(
            args.target_median_cells
        ),
        "observed_median_cells_per_spot": float(
            np.median(
                occupied_counts
            )
        ),
        "observed_mean_cells_per_spot": float(
            np.mean(
                occupied_counts
            )
        ),
        "observed_q25_cells_per_spot": float(
            np.quantile(
                occupied_counts,
                0.25,
            )
        ),
        "observed_q75_cells_per_spot": float(
            np.quantile(
                occupied_counts,
                0.75,
            )
        ),
        "max_cells_per_spot": int(
            occupied_counts.max()
        ),
        "simulator_tissue_area": float(
            area
        ),
        "simulator_cell_density": float(
            density
        ),
        "initial_spacing_estimate": float(
            initial_spacing
        ),
        "selected_center_spacing": float(
            selected_spacing
        ),
        "selected_capture_radius": float(
            radius
        ),
        "radius_to_spacing_ratio": float(
            RADIUS_TO_SPACING
        ),
        "physical_reference": {
            "classic_visium_spot_diameter_um": 55.0,
            "classic_visium_center_spacing_um": 100.0,
        },
        "outputs": {
            "spots": (
                "observation/visium/"
                "spots.tsv"
            ),
            "cell_to_spot": (
                "observation/visium/"
                "cell_to_spot.tsv"
            ),
            "ground_truth": (
                "observation/visium/"
                "spot_ground_truth.tsv"
            ),
            "cell_type_composition": (
                "observation/visium/"
                "spot_cell_type_composition.tsv"
            ),
            "subclone_composition": (
                "observation/visium/"
                "spot_subclone_composition.tsv"
            ),
            "scale_search": (
                "observation/visium/"
                "scale_search.tsv"
            ),
        },
    }

    (
        output_dir
        / "geometry_summary.yaml"
    ).write_text(
        yaml.safe_dump(
            summary,
            sort_keys=False,
        )
    )

    # --------------------------------------------------------------
    # Validation plots
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=2,
        alpha=0.30,
        label="latent cells",
    )

    circles = [
        Circle(
            (
                row.x,
                row.y,
            ),
            radius=row.capture_radius,
        )
        for row in spots.itertuples()
    ]

    collection = PatchCollection(
        circles,
        facecolor="none",
        linewidth=0.4,
        alpha=0.6,
    )

    ax.add_collection(
        collection
    )

    ax.scatter(
        spots["x"],
        spots["y"],
        s=5,
        label="occupied spot centers",
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_xlabel(
        "Simulator x"
    )

    ax.set_ylabel(
        "Simulator y"
    )

    ax.set_title(
        "Visium-like capture geometry"
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    fig.savefig(
        plot_dir
        / "spot_geometry.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    bins = np.arange(
        0.5,
        occupied_counts.max()
        + 1.5,
        1.0,
    )

    ax.hist(
        occupied_counts,
        bins=bins,
        edgecolor="black",
    )

    ax.axvline(
        np.median(
            occupied_counts
        ),
        linestyle="--",
        linewidth=1.5,
        label=(
            "median = "
            f"{np.median(occupied_counts):.1f}"
        ),
    )

    ax.set_xlabel(
        "Latent cells per occupied spot"
    )

    ax.set_ylabel(
        "Number of spots"
    )

    ax.set_title(
        "Visium-like cells-per-spot distribution"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        plot_dir
        / "cells_per_spot.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Scale-search diagnostic
    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    valid_search = search.loc[
        search[
            "median_cells"
        ].notna()
    ]

    ax.plot(
        valid_search[
            "spacing"
        ],
        valid_search[
            "median_cells"
        ],
        marker="o",
        markersize=3,
    )

    ax.axhline(
        args.target_median_cells,
        linestyle="--",
        label=(
            "target median = "
            f"{args.target_median_cells:g}"
        ),
    )

    ax.axvline(
        selected_spacing,
        linestyle=":",
        label=(
            "selected spacing = "
            f"{selected_spacing:.3f}"
        ),
    )

    ax.set_xlabel(
        "Spot center spacing "
        "(simulator units)"
    )

    ax.set_ylabel(
        "Median cells per "
        "occupied spot"
    )

    ax.set_title(
        "Visium-like lattice scale search"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        plot_dir
        / "scale_search.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(
        "Visium-like geometry complete."
    )

    print()
    print(
        "Occupied spots:",
        len(spots),
    )

    print(
        "Captured cells:",
        captured_cells,
        "/",
        len(cells),
        "(",
        f"{captured_cells / len(cells):.3f}",
        ")",
    )

    print(
        "Median cells/spot:",
        np.median(
            occupied_counts
        ),
    )

    print(
        "Mean cells/spot:",
        np.mean(
            occupied_counts
        ),
    )

    print(
        "Q25 cells/spot:",
        np.quantile(
            occupied_counts,
            0.25,
        ),
    )

    print(
        "Q75 cells/spot:",
        np.quantile(
            occupied_counts,
            0.75,
        ),
    )

    print()
    print(
        "Outputs:"
    )

    print(
        output_dir
    )

    print()
    print("PASS")


if __name__ == "__main__":
    main()
