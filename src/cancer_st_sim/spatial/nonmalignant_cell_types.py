from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree


class NonmalignantCellTypeError(RuntimeError):
    pass


@dataclass
class NonmalignantCellTypeResult:
    cells: pd.DataFrame
    summary: dict[str, Any]


OLD_PATCH_RNG_STAGE = 3001
NEIGHBORHOOD_RNG_STAGE = 3101


def _load_yaml(
    path: Path,
) -> dict[str, Any]:

    if not path.exists():
        raise NonmalignantCellTypeError(
            f"Missing cell-type preset: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = yaml.safe_load(handle)

    if not isinstance(
        data,
        dict,
    ):
        raise NonmalignantCellTypeError(
            f"Expected YAML mapping in {path}"
        )

    return data


def _target_counts(
    n_cells: int,
    cell_type_cfg: dict[str, Any],
) -> dict[str, int]:

    names = list(
        cell_type_cfg
    )

    fractions = np.asarray(
        [
            float(
                cell_type_cfg[
                    name
                ][
                    "fraction"
                ]
            )
            for name in names
        ],
        dtype=float,
    )

    if np.any(
        fractions < 0
    ):
        raise NonmalignantCellTypeError(
            "Cell-type fractions cannot "
            "be negative."
        )

    if not np.isclose(
        fractions.sum(),
        1.0,
        atol=1e-8,
    ):
        raise NonmalignantCellTypeError(
            "Cell-type fractions must sum "
            "to 1.0; found "
            f"{fractions.sum():.8f}."
        )

    raw = (
        fractions
        * n_cells
    )

    counts = np.floor(
        raw
    ).astype(int)

    remainder = (
        n_cells
        - int(
            counts.sum()
        )
    )

    if remainder > 0:

        fractional_parts = (
            raw
            - counts
        )

        order = np.argsort(
            -fractional_parts
        )

        for index in order[
            :remainder
        ]:
            counts[
                index
            ] += 1

    return {
        name: int(count)
        for name, count in zip(
            names,
            counts,
        )
    }


# ======================================================================
# Legacy type-specific-patch model
# ======================================================================


def _spatial_score(
    x: np.ndarray,
    y: np.ndarray,
    seed_indices: np.ndarray,
    scale: float,
) -> np.ndarray:

    if scale <= 0:
        raise NonmalignantCellTypeError(
            "spatial.scale must be > 0."
        )

    score = np.zeros(
        len(x),
        dtype=float,
    )

    denominator = (
        2.0
        * scale
        * scale
    )

    for index in seed_indices:

        d2 = (
            (x - x[index]) ** 2
            +
            (y - y[index]) ** 2
        )

        local = np.exp(
            -d2
            / denominator
        )

        score = np.maximum(
            score,
            local,
        )

    return score


def _assign_type_specific_patches(
    cells: pd.DataFrame,
    type_cfg: dict[str, Any],
    spatial_cfg: dict[str, Any],
    seed: int,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:

    cells = cells.copy()

    normal_mask = (
        cells[
            "compartment"
        ]
        == "non_malignant"
    )

    normal_indices = np.flatnonzero(
        normal_mask.to_numpy()
    )

    n_normal = int(
        len(
            normal_indices
        )
    )

    scale = float(
        spatial_cfg.get(
            "scale",
            1.25,
        )
    )

    targets = _target_counts(
        n_cells=n_normal,
        cell_type_cfg=type_cfg,
    )

    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                OLD_PATCH_RNG_STAGE,
            ]
        )
    )

    normal_cells = (
        cells
        .loc[
            normal_mask
        ]
        .copy()
        .reset_index()
        .rename(
            columns={
                "index":
                    "_original_index"
            }
        )
    )

    x = normal_cells[
        "x"
    ].to_numpy(
        dtype=float
    )

    y = normal_cells[
        "y"
    ].to_numpy(
        dtype=float
    )

    score_fields: dict[
        str,
        np.ndarray,
    ] = {}

    for cell_type, cfg in (
        type_cfg.items()
    ):

        n_seeds = int(
            cfg.get(
                "n_spatial_seeds",
                1,
            )
        )

        n_seeds = max(
            1,
            min(
                n_seeds,
                n_normal,
            ),
        )

        seed_indices = rng.choice(
            n_normal,
            size=n_seeds,
            replace=False,
        )

        score = _spatial_score(
            x=x,
            y=y,
            seed_indices=(
                seed_indices
            ),
            scale=scale,
        )

        score += rng.uniform(
            0.0,
            1.0e-10,
            size=n_normal,
        )

        score_fields[
            cell_type
        ] = score

    assigned = np.full(
        n_normal,
        "",
        dtype=object,
    )

    available = np.ones(
        n_normal,
        dtype=bool,
    )

    assignment_order = sorted(
        targets,
        key=lambda name:
            targets[name],
        reverse=True,
    )

    for position, cell_type in (
        enumerate(
            assignment_order
        )
    ):

        target = targets[
            cell_type
        ]

        if target == 0:
            continue

        available_indices = (
            np.flatnonzero(
                available
            )
        )

        is_last = (
            position
            == len(
                assignment_order
            ) - 1
        )

        if is_last:

            chosen = (
                available_indices
            )

            if len(
                chosen
            ) != target:
                raise NonmalignantCellTypeError(
                    "Final target count does "
                    "not match remaining cells."
                )

        else:

            scores = (
                score_fields[
                    cell_type
                ][
                    available_indices
                ]
            )

            ranking = np.argsort(
                scores
            )

            chosen = (
                available_indices[
                    ranking[
                        -target:
                    ]
                ]
            )

        assigned[
            chosen
        ] = cell_type

        available[
            chosen
        ] = False

    normal_cells[
        "cell_type"
    ] = assigned

    original_indices = (
        normal_cells[
            "_original_index"
        ].to_numpy(
            dtype=int
        )
    )

    cells[
        "cell_type"
    ] = "malignant"

    cells.loc[
        original_indices,
        "cell_type",
    ] = assigned

    return (
        cells,
        {
            "spatial_mode":
                "type_specific_patches",

            "spatial_scale":
                scale,

            "rng_stage":
                OLD_PATCH_RNG_STAGE,

            "targets":
                targets,
        },
    )


# ======================================================================
# Neighborhood-growth model
# ======================================================================


def _build_knn(
    xy: np.ndarray,
    k_neighbors: int,
) -> np.ndarray:

    n_cells = len(
        xy
    )

    if n_cells <= 1:
        raise NonmalignantCellTypeError(
            "At least two non-malignant "
            "cells are required."
        )

    k_neighbors = min(
        int(
            k_neighbors
        ),
        n_cells - 1,
    )

    if k_neighbors < 1:
        raise NonmalignantCellTypeError(
            "spatial.k_neighbors must "
            "be >= 1."
        )

    tree = cKDTree(
        xy
    )

    _, neighbor_indices = (
        tree.query(
            xy,
            k=k_neighbors + 1,
        )
    )

    return np.asarray(
        neighbor_indices[
            :,
            1:,
        ],
        dtype=int,
    )


def _tumor_interface_scores(
    cells: pd.DataFrame,
    normal_indices: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
]:

    malignant = cells.loc[
        cells[
            "is_malignant"
        ].astype(bool)
    ]

    if len(
        malignant
    ) == 0:
        raise NonmalignantCellTypeError(
            "Cannot compute tumor interface "
            "without malignant cells."
        )

    normal = cells.iloc[
        normal_indices
    ]

    malignant_xy = malignant[
        [
            "x",
            "y",
        ]
    ].to_numpy(
        dtype=float
    )

    normal_xy = normal[
        [
            "x",
            "y",
        ]
    ].to_numpy(
        dtype=float
    )

    tree = cKDTree(
        malignant_xy
    )

    distances, _ = tree.query(
        normal_xy,
        k=1,
    )

    distances = np.asarray(
        distances,
        dtype=float,
    )

    sigma = float(
        np.median(
            distances
        )
    )

    if sigma <= 0:

        positive = distances[
            distances > 0
        ]

        if len(
            positive
        ) == 0:
            raise NonmalignantCellTypeError(
                "Could not determine positive "
                "tumor-interface scale."
            )

        sigma = float(
            np.median(
                positive
            )
        )

    interface_score = np.exp(
        -(
            distances ** 2
        )
        /
        (
            2.0
            * sigma ** 2
        )
    )

    return (
        distances,
        interface_score,
        sigma,
    )


def _choose_frontier_cell(
    assigned: np.ndarray,
    frontier: set[int],
    interface_score: np.ndarray,
    rng: np.random.Generator,
) -> int:

    if frontier:

        candidates = np.asarray(
            sorted(
                frontier
            ),
            dtype=int,
        )

    else:

        candidates = np.flatnonzero(
            assigned == ""
        )

    if len(
        candidates
    ) == 0:
        raise NonmalignantCellTypeError(
            "No unassigned cells remain."
        )

    #
    # Slight preference for advancing along
    # tumor-interface regions, while every
    # candidate retains non-zero probability.
    #
    weights = (
        interface_score[
            candidates
        ]
        + 0.05
    )

    weights = (
        weights
        / weights.sum()
    )

    return int(
        rng.choice(
            candidates,
            p=weights,
        )
    )


def _assign_neighborhood_growth_v1(
    cells: pd.DataFrame,
    type_cfg: dict[str, Any],
    spatial_cfg: dict[str, Any],
    seed: int,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:

    cells = cells.copy()

    normal_mask = (
        cells[
            "compartment"
        ]
        == "non_malignant"
    )

    normal_indices = np.flatnonzero(
        normal_mask.to_numpy()
    )

    n_normal = int(
        len(
            normal_indices
        )
    )

    if n_normal == 0:
        raise NonmalignantCellTypeError(
            "No non-malignant cells found."
        )

    if (
        "boundary_spatial_score"
        not in cells.columns
    ):
        raise NonmalignantCellTypeError(
            "neighborhood_growth_v1 requires "
            "boundary_spatial_score."
        )

    k_neighbors = int(
        spatial_cfg.get(
            "k_neighbors",
            12,
        )
    )

    top_fraction = float(
        spatial_cfg.get(
            "initial_seed_top_interface_fraction",
            0.20,
        )
    )

    if not (
        0.0
        < top_fraction
        <= 1.0
    ):
        raise NonmalignantCellTypeError(
            "initial_seed_top_interface_fraction "
            "must be in (0, 1]."
        )

    targets = _target_counts(
        n_cells=n_normal,
        cell_type_cfg=type_cfg,
    )

    remaining_counts = dict(
        targets
    )

    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                NEIGHBORHOOD_RNG_STAGE,
            ]
        )
    )

    normal_cells = (
        cells
        .iloc[
            normal_indices
        ]
        .copy()
        .reset_index()
        .rename(
            columns={
                "index":
                    "_original_index"
            }
        )
    )

    xy = normal_cells[
        [
            "x",
            "y",
        ]
    ].to_numpy(
        dtype=float
    )

    knn = _build_knn(
        xy,
        k_neighbors,
    )

    (
        nearest_tumor_distance,
        interface_score,
        interface_sigma,
    ) = _tumor_interface_scores(
        cells=cells,
        normal_indices=(
            normal_indices
        ),
    )

    boundary_score = (
        normal_cells[
            "boundary_spatial_score"
        ].to_numpy(
            dtype=float
        )
    )

    boundary_score = np.clip(
        boundary_score,
        0.0,
        1.0,
    )

    assigned = np.full(
        n_normal,
        "",
        dtype=object,
    )

    frontier: set[int] = set()

    n_initial_candidates = max(
        1,
        int(
            round(
                n_normal
                * top_fraction
            )
        ),
    )

    initial_candidates = (
        np.argsort(
            interface_score
        )[
            -n_initial_candidates:
        ]
    )

    current = int(
        rng.choice(
            initial_candidates
        )
    )

    type_names = list(
        type_cfg
    )

    for step in range(
        n_normal
    ):

        if assigned[
            current
        ] != "":

            current = (
                _choose_frontier_cell(
                    assigned=assigned,
                    frontier=frontier,
                    interface_score=(
                        interface_score
                    ),
                    rng=rng,
                )
            )

        valid_types = [
            name
            for name in type_names
            if remaining_counts[
                name
            ] > 0
        ]

        if not valid_types:
            raise NonmalignantCellTypeError(
                "No cell types retain capacity."
            )

        if len(
            valid_types
        ) == 1:

            chosen_type = (
                valid_types[
                    0
                ]
            )

        else:

            neighbors = knn[
                current
            ]

            assigned_neighbors = [
                assigned[
                    neighbor
                ]
                for neighbor
                in neighbors
                if assigned[
                    neighbor
                ] != ""
            ]

            n_assigned_neighbors = len(
                assigned_neighbors
            )

            total_remaining = sum(
                remaining_counts[
                    name
                ]
                for name
                in valid_types
            )

            logits = []

            for cell_type in (
                valid_types
            ):

                cfg = type_cfg[
                    cell_type
                ]

                if (
                    n_assigned_neighbors
                    > 0
                ):

                    same_fraction = (
                        sum(
                            label
                            == cell_type
                            for label
                            in assigned_neighbors
                        )
                        /
                        n_assigned_neighbors
                    )

                else:

                    same_fraction = 0.0

                neighbor_weight = float(
                    cfg.get(
                        "neighbor_weight",
                        0.0,
                    )
                )

                boundary_weight = float(
                    cfg.get(
                        "boundary_weight",
                        0.0,
                    )
                )

                interface_weight = float(
                    cfg.get(
                        "tumor_interface_weight",
                        0.0,
                    )
                )

                remaining_fraction = (
                    remaining_counts[
                        cell_type
                    ]
                    /
                    total_remaining
                )

                spatial_logit = (
                    neighbor_weight
                    * same_fraction

                    +

                    boundary_weight
                    * boundary_score[
                        current
                    ]

                    +

                    interface_weight
                    * interface_score[
                        current
                    ]
                )

                logit = (
                    np.log(
                        remaining_fraction
                    )
                    +
                    spatial_logit
                )

                logits.append(
                    logit
                )

            logits = np.asarray(
                logits,
                dtype=float,
            )

            logits -= np.max(
                logits
            )

            probabilities = np.exp(
                logits
            )

            probabilities /= (
                probabilities.sum()
            )

            chosen_type = str(
                rng.choice(
                    valid_types,
                    p=probabilities,
                )
            )

        assigned[
            current
        ] = chosen_type

        remaining_counts[
            chosen_type
        ] -= 1

        for neighbor in knn[
            current
        ]:

            neighbor = int(
                neighbor
            )

            if assigned[
                neighbor
            ] == "":

                frontier.add(
                    neighbor
                )

        frontier.discard(
            current
        )

        if step < (
            n_normal - 1
        ):

            current = (
                _choose_frontier_cell(
                    assigned=assigned,
                    frontier=frontier,
                    interface_score=(
                        interface_score
                    ),
                    rng=rng,
                )
            )

    if np.any(
        assigned == ""
    ):
        raise NonmalignantCellTypeError(
            "Some non-malignant cells "
            "remain unassigned."
        )

    if any(
        value != 0
        for value in (
            remaining_counts.values()
        )
    ):
        raise NonmalignantCellTypeError(
            "Exact target counts were "
            "not consumed."
        )

    normal_cells[
        "cell_type"
    ] = assigned

    normal_cells[
        "nearest_malignant_distance"
    ] = nearest_tumor_distance

    normal_cells[
        "tumor_interface_score"
    ] = interface_score

    original_indices = (
        normal_cells[
            "_original_index"
        ].to_numpy(
            dtype=int
        )
    )

    cells[
        "cell_type"
    ] = "malignant"

    cells.loc[
        original_indices,
        "cell_type",
    ] = assigned

    #
    # Save diagnostics on the full cell table.
    #
    cells[
        "nearest_malignant_distance"
    ] = np.nan

    cells[
        "tumor_interface_score"
    ] = np.nan

    cells.loc[
        original_indices,
        "nearest_malignant_distance",
    ] = nearest_tumor_distance

    cells.loc[
        original_indices,
        "tumor_interface_score",
    ] = interface_score

    return (
        cells,
        {
            "spatial_mode":
                "neighborhood_growth_v1",

            "k_neighbors":
                int(
                    k_neighbors
                ),

            "initial_seed_top_interface_fraction":
                float(
                    top_fraction
                ),

            "interface_sigma":
                float(
                    interface_sigma
                ),

            "rng_stage":
                NEIGHBORHOOD_RNG_STAGE,

            "targets":
                targets,
        },
    )


# ======================================================================
# Public assignment API
# ======================================================================


def assign_nonmalignant_cell_types(
    cells: pd.DataFrame,
    preset: dict[str, Any],
    seed: int,
) -> NonmalignantCellTypeResult:

    required = {
        "cell_id",
        "x",
        "y",
        "compartment",
        "is_malignant",
    }

    missing = (
        required
        - set(
            cells.columns
        )
    )

    if missing:
        raise NonmalignantCellTypeError(
            "Missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    cells = cells.copy()

    normal_mask = (
        cells[
            "compartment"
        ]
        == "non_malignant"
    )

    n_normal = int(
        normal_mask.sum()
    )

    if n_normal == 0:
        raise NonmalignantCellTypeError(
            "No non-malignant cells found."
        )

    type_cfg = preset.get(
        "cell_types"
    )

    if not isinstance(
        type_cfg,
        dict,
    ):
        raise NonmalignantCellTypeError(
            "Preset must define cell_types."
        )

    spatial_cfg = preset.get(
        "spatial",
        {},
    )

    if not isinstance(
        spatial_cfg,
        dict,
    ):
        raise NonmalignantCellTypeError(
            "spatial must be a mapping."
        )

    mode = str(
        spatial_cfg.get(
            "mode",
            "type_specific_patches",
        )
    )

    if mode == (
        "type_specific_patches"
    ):

        cells, model_info = (
            _assign_type_specific_patches(
                cells=cells,
                type_cfg=type_cfg,
                spatial_cfg=(
                    spatial_cfg
                ),
                seed=seed,
            )
        )

    elif mode == (
        "neighborhood_growth_v1"
    ):

        cells, model_info = (
            _assign_neighborhood_growth_v1(
                cells=cells,
                type_cfg=type_cfg,
                spatial_cfg=(
                    spatial_cfg
                ),
                seed=seed,
            )
        )

    else:

        raise NonmalignantCellTypeError(
            "Unsupported spatial mode: "
            f"{mode}"
        )

    targets = model_info[
        "targets"
    ]

    observed_counts = (
        cells.loc[
            ~cells[
                "is_malignant"
            ].astype(bool)
        ][
            "cell_type"
        ]
        .value_counts()
        .to_dict()
    )

    for cell_type, target in (
        targets.items()
    ):

        observed = int(
            observed_counts.get(
                cell_type,
                0,
            )
        )

        if observed != target:
            raise NonmalignantCellTypeError(
                f"{cell_type}: expected "
                f"{target}, observed "
                f"{observed}."
            )

    n_malignant = int(
        cells[
            "is_malignant"
        ].sum()
    )

    summary: dict[
        str,
        Any,
    ] = {
        "source":
            "cells_with_compartment",

        "preset":
            preset.get(
                "preset_name"
            ),

        "preset_version":
            preset.get(
                "preset_version"
            ),

        "status":
            preset.get(
                "status"
            ),

        "n_total_cells":
            int(
                len(
                    cells
                )
            ),

        "n_malignant_cells":
            n_malignant,

        "n_nonmalignant_cells":
            n_normal,

        "cell_type_counts": {
            name:
                int(
                    observed_counts[
                        name
                    ]
                )
            for name
            in targets
        },

        "cell_type_fractions_within_nonmalignant": {
            name:
                float(
                    observed_counts[
                        name
                    ]
                    / n_normal
                )
            for name
            in targets
        },

        "spatial_mode":
            model_info[
                "spatial_mode"
            ],

        "master_seed":
            int(
                seed
            ),

        "rng_stage":
            int(
                model_info[
                    "rng_stage"
                ]
            ),

        "exact_counts_conserved":
            True,

        "fraction_calibration_status":
            preset.get(
                "provenance",
                {},
            ).get(
                "calibration_status"
            ),

        "malignant_cell_type_label":
            "malignant",
    }

    if (
        model_info[
            "spatial_mode"
        ]
        == "type_specific_patches"
    ):

        summary[
            "spatial_scale"
        ] = float(
            model_info[
                "spatial_scale"
            ]
        )

    else:

        summary[
            "k_neighbors"
        ] = int(
            model_info[
                "k_neighbors"
            ]
        )

        summary[
            "initial_seed_top_interface_fraction"
        ] = float(
            model_info[
                "initial_seed_top_interface_fraction"
            ]
        )

        summary[
            "tumor_interface_sigma"
        ] = float(
            model_info[
                "interface_sigma"
            ]
        )

    return NonmalignantCellTypeResult(
        cells=cells,
        summary=summary,
    )


def write_nonmalignant_cell_types(
    cells_path: Path,
    output_dir: Path,
    preset_path: Path,
    seed: int,
) -> NonmalignantCellTypeResult:

    cells = pd.read_csv(
        cells_path,
        sep="\t",
    )

    preset = _load_yaml(
        preset_path
    )

    result = assign_nonmalignant_cell_types(
        cells=cells,
        preset=preset,
        seed=seed,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.cells.to_csv(
        output_dir
        / "cells_with_cell_types.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )

    with (
        output_dir
        / "cell_type_summary.yaml"
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
