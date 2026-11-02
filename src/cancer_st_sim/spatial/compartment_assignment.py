from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter


class CompartmentAssignmentError(RuntimeError):
    """Raised when malignant/non-malignant assignment fails."""


@dataclass
class CompartmentAssignmentResult:
    cells: pd.DataFrame
    summary: dict[str, Any]


# ---------------------------------------------------------------------
# Accepted HGSOC v1 spatial-refinement defaults.
#
# DEMON remains the biological/spatial scaffold.
# These values only refine the final geometry and compartment layout.
# ---------------------------------------------------------------------

DEFAULT_WARP_SIGMA = 1.60
DEFAULT_WARP_STRENGTH = 0.45
DEFAULT_BOUNDARY_JITTER_SD = 0.10

DEFAULT_FIELD_SIGMA = 0.95
DEFAULT_BOUNDARY_WEIGHT = 0.30
DEFAULT_RANDOM_FIELD_WEIGHT = 0.70
DEFAULT_CELL_NOISE_SD = 0.07


def _load_yaml(
    path: Path,
) -> dict[str, Any]:

    if not path.exists():
        raise CompartmentAssignmentError(
            f"Missing compartment preset: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = yaml.safe_load(
            handle
        )

    if not isinstance(
        data,
        dict,
    ):
        raise CompartmentAssignmentError(
            f"Expected YAML mapping in {path}"
        )

    return data


def _minmax(
    values: np.ndarray,
) -> np.ndarray:

    x = np.asarray(
        values,
        dtype=float,
    )

    minimum = float(
        np.min(x)
    )

    maximum = float(
        np.max(x)
    )

    if maximum <= minimum:
        return np.zeros_like(
            x,
            dtype=float,
        )

    return (
        x - minimum
    ) / (
        maximum - minimum
    )


def _resolve_parameter(
    block: dict[str, Any],
    name: str,
    default: float,
) -> float:

    value = block.get(
        name,
        default,
    )

    try:
        value = float(
            value
        )

    except Exception as exc:
        raise CompartmentAssignmentError(
            f"Invalid spatial parameter "
            f"{name}={value!r}"
        ) from exc

    if not np.isfinite(
        value
    ):
        raise CompartmentAssignmentError(
            f"Spatial parameter {name} "
            "must be finite."
        )

    return value


def _build_occupancy_grid(
    cells: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    required = {
        "deme_x",
        "deme_y",
    }

    missing = (
        required
        - set(cells.columns)
    )

    if missing:
        raise CompartmentAssignmentError(
            "Correlated spatial realization requires "
            "DEMON deme coordinates: "
            + ", ".join(
                sorted(missing)
            )
        )

    deme_x = (
        cells[
            "deme_x"
        ]
        .astype(int)
        .to_numpy()
    )

    deme_y = (
        cells[
            "deme_y"
        ]
        .astype(int)
        .to_numpy()
    )

    min_x = int(
        np.min(deme_x)
    )

    max_x = int(
        np.max(deme_x)
    )

    min_y = int(
        np.min(deme_y)
    )

    max_y = int(
        np.max(deme_y)
    )

    grid_x = np.arange(
        min_x,
        max_x + 1,
        dtype=float,
    )

    grid_y = np.arange(
        min_y,
        max_y + 1,
        dtype=float,
    )

    occupied = np.zeros(
        (
            len(grid_y),
            len(grid_x),
        ),
        dtype=bool,
    )

    gx = (
        deme_x
        - min_x
    )

    gy = (
        deme_y
        - min_y
    )

    occupied[
        gy,
        gx,
    ] = True

    return (
        grid_x,
        grid_y,
        occupied,
    )


def _build_boundary_grid(
    occupied: np.ndarray,
) -> np.ndarray:
    """
    Return a 0..1 peripheral score over the DEMON tissue.

    Higher value = closer to the tissue boundary.

    Padding ensures that the outside of the represented
    DEMON grid is treated as non-tissue.
    """

    padded = np.pad(
        occupied,
        pad_width=1,
        mode="constant",
        constant_values=False,
    )

    distance_padded = (
        distance_transform_edt(
            padded
        )
    )

    inside_distance = (
        distance_padded[
            1:-1,
            1:-1,
        ]
    )

    boundary = np.zeros_like(
        inside_distance,
        dtype=float,
    )

    occupied_values = (
        inside_distance[
            occupied
        ]
    )

    if len(
        occupied_values
    ) == 0:
        raise CompartmentAssignmentError(
            "DEMON occupancy grid contains no cells."
        )

    max_inside = float(
        np.max(
            occupied_values
        )
    )

    if max_inside <= 1.0:

        boundary[
            occupied
        ] = 1.0

    else:

        boundary[
            occupied
        ] = (
            1.0
            -
            (
                occupied_values
                - 1.0
            )
            /
            (
                max_inside
                - 1.0
            )
        )

    return np.clip(
        boundary,
        0.0,
        1.0,
    )


def _make_interpolator(
    field: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> RegularGridInterpolator:

    return RegularGridInterpolator(
        (
            grid_y,
            grid_x,
        ),
        field,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )


def _standardize_field(
    field: np.ndarray,
    occupied: np.ndarray,
) -> np.ndarray:

    output = np.asarray(
        field,
        dtype=float,
    ).copy()

    values = output[
        occupied
    ]

    mean = float(
        np.mean(values)
    )

    sd = float(
        np.std(values)
    )

    output -= mean

    if sd > 0.0:
        output /= sd

    return output


def _build_spatial_realization(
    cells: pd.DataFrame,
    rng: np.random.Generator,
    preset: dict[str, Any],
) -> dict[str, np.ndarray | float]:
    """
    Build the accepted DEMON-preserving spatial refinement.

    The DEMON scaffold determines:
      - original positions
      - occupied demes
      - clone identities
      - coarse tissue morphology

    The refinement adds:
      - smooth low-frequency coordinate deformation
      - an irregular correlated nonmalignant field
      - modest preference for the DEMON tissue boundary
      - small cell-level roughness
    """

    geometry = preset.get(
        "geometry_warp",
        {},
    )

    compartment = preset.get(
        "correlated_field",
        {},
    )

    if not isinstance(
        geometry,
        dict,
    ):
        raise CompartmentAssignmentError(
            "geometry_warp must be a mapping."
        )

    if not isinstance(
        compartment,
        dict,
    ):
        raise CompartmentAssignmentError(
            "correlated_field must be a mapping."
        )

    warp_sigma = _resolve_parameter(
        geometry,
        "sigma",
        DEFAULT_WARP_SIGMA,
    )

    warp_strength = _resolve_parameter(
        geometry,
        "strength",
        DEFAULT_WARP_STRENGTH,
    )

    boundary_jitter_sd = _resolve_parameter(
        geometry,
        "boundary_jitter_sd",
        DEFAULT_BOUNDARY_JITTER_SD,
    )

    field_sigma = _resolve_parameter(
        compartment,
        "sigma",
        DEFAULT_FIELD_SIGMA,
    )

    boundary_weight = _resolve_parameter(
        compartment,
        "boundary_weight",
        DEFAULT_BOUNDARY_WEIGHT,
    )

    random_field_weight = _resolve_parameter(
        compartment,
        "random_field_weight",
        DEFAULT_RANDOM_FIELD_WEIGHT,
    )

    cell_noise_sd = _resolve_parameter(
        compartment,
        "cell_noise_sd",
        DEFAULT_CELL_NOISE_SD,
    )

    if warp_sigma <= 0:
        raise CompartmentAssignmentError(
            "geometry_warp.sigma must be > 0."
        )

    if warp_strength < 0:
        raise CompartmentAssignmentError(
            "geometry_warp.strength must be >= 0."
        )

    if boundary_jitter_sd < 0:
        raise CompartmentAssignmentError(
            "geometry_warp.boundary_jitter_sd "
            "must be >= 0."
        )

    if field_sigma <= 0:
        raise CompartmentAssignmentError(
            "correlated_field.sigma must be > 0."
        )

    if boundary_weight < 0:
        raise CompartmentAssignmentError(
            "correlated_field.boundary_weight "
            "must be >= 0."
        )

    if random_field_weight < 0:
        raise CompartmentAssignmentError(
            "correlated_field.random_field_weight "
            "must be >= 0."
        )

    if cell_noise_sd < 0:
        raise CompartmentAssignmentError(
            "correlated_field.cell_noise_sd "
            "must be >= 0."
        )

    weight_sum = (
        boundary_weight
        + random_field_weight
    )

    if weight_sum <= 0:
        raise CompartmentAssignmentError(
            "At least one of boundary_weight or "
            "random_field_weight must be > 0."
        )

    #
    # Normalize weights defensively.
    #
    boundary_weight /= weight_sum
    random_field_weight /= weight_sum

    (
        grid_x,
        grid_y,
        occupied,
    ) = _build_occupancy_grid(
        cells
    )

    boundary_grid = (
        _build_boundary_grid(
            occupied
        )
    )

    #
    # -------------------------------------------------------------
    # Smooth correlated compartment field.
    # -------------------------------------------------------------
    #

    raw_field = rng.normal(
        size=occupied.shape
    )

    irregular_grid = gaussian_filter(
        raw_field,
        sigma=field_sigma,
        mode="reflect",
    )

    irregular_grid = _minmax(
        irregular_grid
    )

    #
    # -------------------------------------------------------------
    # Smooth coordinate deformation fields.
    # -------------------------------------------------------------
    #

    warp_x_grid = gaussian_filter(
        rng.normal(
            size=occupied.shape
        ),
        sigma=warp_sigma,
        mode="reflect",
    )

    warp_y_grid = gaussian_filter(
        rng.normal(
            size=occupied.shape
        ),
        sigma=warp_sigma,
        mode="reflect",
    )

    warp_x_grid = _standardize_field(
        warp_x_grid,
        occupied,
    )

    warp_y_grid = _standardize_field(
        warp_y_grid,
        occupied,
    )

    boundary_interp = _make_interpolator(
        boundary_grid,
        grid_x,
        grid_y,
    )

    irregular_interp = _make_interpolator(
        irregular_grid,
        grid_x,
        grid_y,
    )

    warp_x_interp = _make_interpolator(
        warp_x_grid,
        grid_x,
        grid_y,
    )

    warp_y_interp = _make_interpolator(
        warp_y_grid,
        grid_x,
        grid_y,
    )

    x_original = (
        cells[
            "x"
        ]
        .to_numpy(
            dtype=float
        )
    )

    y_original = (
        cells[
            "y"
        ]
        .to_numpy(
            dtype=float
        )
    )

    points = np.column_stack(
        [
            y_original,
            x_original,
        ]
    )

    boundary_score = np.asarray(
        boundary_interp(
            points
        ),
        dtype=float,
    )

    boundary_score = np.clip(
        boundary_score,
        0.0,
        1.0,
    )

    irregular_score = np.asarray(
        irregular_interp(
            points
        ),
        dtype=float,
    )

    irregular_score = _minmax(
        irregular_score
    )

    warp_dx = np.asarray(
        warp_x_interp(
            points
        ),
        dtype=float,
    )

    warp_dy = np.asarray(
        warp_y_interp(
            points
        ),
        dtype=float,
    )

    #
    # Nonmalignant spatial propensity.
    #
    nonmalignant_score = (
        boundary_weight
        * boundary_score

        +

        random_field_weight
        * irregular_score

        +

        rng.normal(
            loc=0.0,
            scale=cell_noise_sd,
            size=len(cells),
        )
    )

    #
    # Coordinate deformation.
    #
    x_refined = (
        x_original
        + warp_strength
        * warp_dx
    )

    y_refined = (
        y_original
        + warp_strength
        * warp_dy
    )

    #
    # Add only small additional roughness near the DEMON
    # tissue boundary.
    #
    edge_strength = (
        boundary_score ** 2
    )

    x_refined += (
        rng.normal(
            loc=0.0,
            scale=boundary_jitter_sd,
            size=len(cells),
        )
        * edge_strength
    )

    y_refined += (
        rng.normal(
            loc=0.0,
            scale=boundary_jitter_sd,
            size=len(cells),
        )
        * edge_strength
    )

    return {
        "x_original":
            x_original,

        "y_original":
            y_original,

        "x_refined":
            x_refined,

        "y_refined":
            y_refined,

        "boundary_score":
            boundary_score,

        "irregular_score":
            irregular_score,

        "nonmalignant_score":
            nonmalignant_score,

        "warp_sigma":
            float(
                warp_sigma
            ),

        "warp_strength":
            float(
                warp_strength
            ),

        "boundary_jitter_sd":
            float(
                boundary_jitter_sd
            ),

        "field_sigma":
            float(
                field_sigma
            ),

        "boundary_weight":
            float(
                boundary_weight
            ),

        "random_field_weight":
            float(
                random_field_weight
            ),

        "cell_noise_sd":
            float(
                cell_noise_sd
            ),
    }


def assign_tissue_compartments(
    cells: pd.DataFrame,
    tumor_fraction: float,
    seed: int,
    preset: dict[str, Any],
) -> CompartmentAssignmentResult:

    required = {
        "cell_id",
        "deme_id",
        "deme_x",
        "deme_y",
        "x",
        "y",
        "subclone_id",
        "driver_genotype_id",
    }

    missing = (
        required
        - set(cells.columns)
    )

    if missing:
        raise CompartmentAssignmentError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    if not 0.0 < tumor_fraction < 1.0:
        raise CompartmentAssignmentError(
            "tumor_fraction must be between 0 and 1."
        )

    cells = cells.copy()

    n_total = int(
        len(cells)
    )

    n_malignant = int(
        round(
            n_total
            * tumor_fraction
        )
    )

    n_nonmalignant = (
        n_total
        - n_malignant
    )

    #
    # -------------------------------------------------------------
    # IMPORTANT:
    #
    # Use the master simulation seed directly here so that this
    # implementation reproduces the accepted preview realization
    # when supplied the same cell scaffold and seed.
    # -------------------------------------------------------------
    #

    rng = np.random.default_rng(
        int(seed)
    )

    realization = (
        _build_spatial_realization(
            cells=cells,
            rng=rng,
            preset=preset,
        )
    )

    nonmalignant_score = np.asarray(
        realization[
            "nonmalignant_score"
        ],
        dtype=float,
    )

    if (
        ~np.isfinite(
            nonmalignant_score
        )
    ).any():
        raise CompartmentAssignmentError(
            "Non-finite nonmalignant spatial scores."
        )

    #
    # Exact abundance enforcement.
    #
    # Highest-scoring cells become nonmalignant.
    #
    ranking = np.argsort(
        nonmalignant_score,
        kind="mergesort",
    )

    if n_nonmalignant > 0:

        nonmalignant_indices = ranking[
            -n_nonmalignant:
        ]

    else:

        nonmalignant_indices = np.array(
            [],
            dtype=int,
        )

    malignant_mask = np.ones(
        n_total,
        dtype=bool,
    )

    malignant_mask[
        nonmalignant_indices
    ] = False

    #
    # -------------------------------------------------------------
    # Preserve DEMON geometry + tumor lineage provenance.
    # -------------------------------------------------------------
    #

    if (
        "source_demon_deme_x"
        in cells.columns
        and
        "source_demon_deme_y"
        in cells.columns
    ):
        cells[
            "source_demon_x"
        ] = cells[
            "source_demon_deme_x"
        ].to_numpy(
            dtype=float
        )

        cells[
            "source_demon_y"
        ] = cells[
            "source_demon_deme_y"
        ].to_numpy(
            dtype=float
        )

    else:
        cells[
            "source_demon_x"
        ] = np.asarray(
            realization[
                "x_original"
            ],
            dtype=float,
        )

        cells[
            "source_demon_y"
        ] = np.asarray(
            realization[
                "y_original"
            ],
            dtype=float,
        )

    #
    # x/y become the refined canonical spatial coordinates.
    #
    cells[
        "x"
    ] = np.asarray(
        realization[
            "x_refined"
        ],
        dtype=float,
    )

    cells[
        "y"
    ] = np.asarray(
        realization[
            "y_refined"
        ],
        dtype=float,
    )

    cells[
        "source_demon_subclone_id"
    ] = cells[
        "subclone_id"
    ]

    cells[
        "source_demon_driver_genotype_id"
    ] = (
        cells[
            "driver_genotype_id"
        ]
        .astype(
            "Int64"
        )
    )

    #
    # -------------------------------------------------------------
    # Biological compartment.
    # -------------------------------------------------------------
    #

    cells[
        "compartment"
    ] = np.where(
        malignant_mask,
        "malignant",
        "non_malignant",
    )

    cells[
        "is_malignant"
    ] = malignant_mask

    cells[
        "driver_genotype_id"
    ] = (
        cells[
            "driver_genotype_id"
        ]
        .astype(
            "Int64"
        )
    )

    #
    # Nonmalignant cells do NOT retain a biological tumor clone.
    # Original DEMON lineage remains in source_demon_* columns.
    #
    cells.loc[
        ~cells[
            "is_malignant"
        ],
        "subclone_id",
    ] = pd.NA

    cells.loc[
        ~cells[
            "is_malignant"
        ],
        "driver_genotype_id",
    ] = pd.NA

    #
    # Spatial diagnostics retained in the canonical table.
    #
    cells[
        "nonmalignant_spatial_score"
    ] = nonmalignant_score

    cells[
        "boundary_spatial_score"
    ] = np.asarray(
        realization[
            "boundary_score"
        ],
        dtype=float,
    )

    cells[
        "correlated_spatial_score"
    ] = np.asarray(
        realization[
            "irregular_score"
        ],
        dtype=float,
    )

    cells[
        "spatial_geometry_refined"
    ] = True

    #
    # -------------------------------------------------------------
    # Count validation.
    # -------------------------------------------------------------
    #

    actual_malignant = int(
        cells[
            "is_malignant"
        ].sum()
    )

    actual_nonmalignant = (
        n_total
        - actual_malignant
    )

    if (
        actual_malignant
        != n_malignant
        or
        actual_nonmalignant
        != n_nonmalignant
    ):

        raise CompartmentAssignmentError(
            "Exact malignant/non-malignant "
            "cell counts were not preserved."
        )

    #
    # -------------------------------------------------------------
    # Deme-level diagnostics.
    # -------------------------------------------------------------
    #

    deme_summary = (
        cells
        .groupby(
            "deme_id",
            as_index=False,
        )
        .agg(
            n_cells=(
                "cell_id",
                "size",
            ),

            n_malignant=(
                "is_malignant",
                "sum",
            ),
        )
    )

    deme_summary[
        "n_nonmalignant"
    ] = (
        deme_summary[
            "n_cells"
        ]
        - deme_summary[
            "n_malignant"
        ]
    )

    mixed = (
        (
            deme_summary[
                "n_malignant"
            ]
            > 0
        )
        &
        (
            deme_summary[
                "n_nonmalignant"
            ]
            > 0
        )
    )

    #
    # Coordinate displacement audit.
    #
    displacement = np.sqrt(
        (
            cells["x"].to_numpy(
                dtype=float
            )
            -
            cells["source_demon_x"].to_numpy(
                dtype=float
            )
        ) ** 2

        +

        (
            cells["y"].to_numpy(
                dtype=float
            )
            -
            cells["source_demon_y"].to_numpy(
                dtype=float
            )
        ) ** 2
    )

    is_continuous_scaffold = (
        "spatial_tile_x" in cells.columns
        and "spatial_tile_y" in cells.columns
    )

    summary = {
        "source":
            (
                "continuous_cell_scaffold"
                if is_continuous_scaffold
                else "DEMON_cell_scaffold"
            ),

        "mode":
            str(
                preset.get(
                    "mode",
                    "unknown",
                )
            ),

        "demon_backbone_preserved":
            (
                False
                if is_continuous_scaffold
                else True
            ),

        "n_total_cells":
            n_total,

        "requested_tumor_fraction":
            float(
                tumor_fraction
            ),

        "target_malignant_cells":
            n_malignant,

        "target_nonmalignant_cells":
            n_nonmalignant,

        "actual_malignant_cells":
            actual_malignant,

        "actual_nonmalignant_cells":
            actual_nonmalignant,

        "actual_tumor_fraction":
            float(
                actual_malignant
                / n_total
            ),

        "geometry_refinement": {
            "method":
                "smooth_correlated_coordinate_warp",

            "warp_sigma":
                float(
                    realization[
                        "warp_sigma"
                    ]
                ),

            "warp_strength":
                float(
                    realization[
                        "warp_strength"
                    ]
                ),

            "boundary_jitter_sd":
                float(
                    realization[
                        "boundary_jitter_sd"
                    ]
                ),
        },

        "source_coordinate_displacement_audit": {
            "reference":
                "original_demon_coordinates",

            "mean_distance":
                float(
                    np.mean(
                        displacement
                    )
                ),

            "median_distance":
                float(
                    np.median(
                        displacement
                    )
                ),

            "q95_distance":
                float(
                    np.quantile(
                        displacement,
                        0.95,
                    )
                ),

            "max_distance":
                float(
                    np.max(
                        displacement
                    )
                ),
        },

        "compartment_realization": {
            "method":
                "correlated_spatial_field",

            "field_sigma":
                float(
                    realization[
                        "field_sigma"
                    ]
                ),

            "boundary_weight":
                float(
                    realization[
                        "boundary_weight"
                    ]
                ),

            "random_field_weight":
                float(
                    realization[
                        "random_field_weight"
                    ]
                ),

            "cell_noise_sd":
                float(
                    realization[
                        "cell_noise_sd"
                    ]
                ),

            "exact_count_enforcement":
                True,
        },

        "master_seed":
            int(
                seed
            ),

        "rng_scheme":
            "direct_master_seed_v1",

        "exact_cell_count_conserved":
            True,

        "n_demes_with_nonmalignant_cells":
            int(
                (
                    deme_summary[
                        "n_nonmalignant"
                    ]
                    > 0
                ).sum()
            ),

        "n_demes_fully_malignant":
            int(
                (
                    deme_summary[
                        "n_nonmalignant"
                    ]
                    == 0
                ).sum()
            ),

        "n_demes_mixed":
            int(
                mixed.sum()
            ),

        "malignant_cells_retain_demon_clone":
            True,

        "nonmalignant_cells_retain_demon_clone":
            False,

        "source_demon_identity_preserved_for_provenance":
            True,

        "source_demon_coordinates_preserved_for_provenance":
            True,

        "specific_tme_cell_types_assigned":
            False,
    }

    return CompartmentAssignmentResult(
        cells=cells,
        summary=summary,
    )


def write_tissue_compartments(
    cells_path: Path,
    output_dir: Path,
    tumor_fraction: float,
    seed: int,
    preset_path: Path,
) -> CompartmentAssignmentResult:

    cells = pd.read_csv(
        cells_path,
        sep="\t",
    )

    preset = _load_yaml(
        preset_path
    )

    result = assign_tissue_compartments(
        cells=cells,
        tumor_fraction=tumor_fraction,
        seed=seed,
        preset=preset,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.cells.to_csv(
        output_dir
        / "cells_with_compartment.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )

    with (
        output_dir
        / "compartment_summary.yaml"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        yaml.safe_dump(
            result.summary,
            handle,
            sort_keys=False,
        )

    return result
