from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog


class ContinuousScaffoldError(RuntimeError):
    pass


def _transport_plan(
    source_xy: np.ndarray,
    source_counts: np.ndarray,
    target_xy: np.ndarray,
    target_counts: np.ndarray,
) -> np.ndarray:
    """
    Minimum-distance transportation plan.

    Rows = original DEMON demes.
    Columns = filled spatial tiles.

    Exact source-deme population totals and exact target-tile
    capacities are preserved.
    """
    n_source = len(source_counts)
    n_target = len(target_counts)

    dx = (
        source_xy[:, None, 0]
        - target_xy[None, :, 0]
    )
    dy = (
        source_xy[:, None, 1]
        - target_xy[None, :, 1]
    )

    cost = (
        dx * dx
        + dy * dy
    ).reshape(-1)

    n_variables = n_source * n_target

    A_eq = np.zeros(
        (
            n_source + n_target,
            n_variables,
        ),
        dtype=float,
    )

    # Source-deme population constraints.
    for i in range(n_source):
        start = i * n_target
        stop = start + n_target
        A_eq[i, start:stop] = 1.0

    # Target-tile capacity constraints.
    for j in range(n_target):
        A_eq[
            n_source + j,
            j::n_target,
        ] = 1.0

    b_eq = np.concatenate(
        [
            source_counts.astype(float),
            target_counts.astype(float),
        ]
    )

    result = linprog(
        c=cost,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )

    if not result.success:
        raise ContinuousScaffoldError(
            "Continuous-scaffold transportation "
            f"optimization failed: {result.message}"
        )

    raw = result.x.reshape(
        n_source,
        n_target,
    )

    plan = np.rint(raw).astype(int)

    if not np.array_equal(
        plan.sum(axis=1),
        source_counts,
    ):
        raise ContinuousScaffoldError(
            "Transportation plan does not preserve "
            "source DEMON populations."
        )

    if not np.array_equal(
        plan.sum(axis=0),
        target_counts,
    ):
        raise ContinuousScaffoldError(
            "Transportation plan does not preserve "
            "target-tile capacities."
        )

    return plan


def build_continuous_rectangle_scaffold(
    deme_clone_composition: pd.DataFrame,
    seed: int,
    settings: dict[str, Any] | None = None,
):
    """
    Redistribute DEMON-derived biological cell records over a filled
    rectangular tissue footprint.

    Biological lineage counts are preserved exactly. Geometry is
    changed; DEMON evolution is not.
    """
    settings = settings or {}

    required = {
        "deme_id",
        "x",
        "y",
        "subclone_id",
        "driver_genotype_id",
        "n_cells",
    }

    missing = required - set(
        deme_clone_composition.columns
    )

    if missing:
        raise ContinuousScaffoldError(
            "Missing required scaffold columns: "
            + ", ".join(sorted(missing))
        )

    composition = (
        deme_clone_composition
        .copy()
        .sort_values(
            [
                "deme_id",
                "driver_genotype_id",
            ]
        )
        .reset_index(drop=True)
    )

    if (composition["n_cells"] < 0).any():
        raise ContinuousScaffoldError(
            "Negative cell counts found."
        )

    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                1001,
                20261023,
            ]
        )
    )

    jitter = float(
        settings.get(
            "coordinate_jitter",
            0.48,
        )
    )

    if not 0.0 <= jitter < 0.5:
        raise ContinuousScaffoldError(
            "coordinate_jitter must satisfy "
            "0 <= jitter < 0.5."
        )

    # ------------------------------------------------------------
    # Original DEMON demes and their exact populations.
    # ------------------------------------------------------------

    deme_meta = (
        composition
        .groupby(
            "deme_id",
            as_index=False,
        )
        .agg(
            x=("x", "first"),
            y=("y", "first"),
            n_cells=("n_cells", "sum"),
        )
        .sort_values("deme_id")
        .reset_index(drop=True)
    )

    source_ids = (
        deme_meta["deme_id"]
        .to_numpy()
    )

    source_xy = (
        deme_meta[
            ["x", "y"]
        ]
        .to_numpy(dtype=float)
    )

    source_counts = (
        deme_meta["n_cells"]
        .to_numpy(dtype=int)
    )

    n_total = int(
        source_counts.sum()
    )

    # ------------------------------------------------------------
    # Filled rectangular spatial footprint.
    # ------------------------------------------------------------

    xmin = int(
        np.floor(
            deme_meta["x"].min()
        )
    )
    xmax = int(
        np.ceil(
            deme_meta["x"].max()
        )
    )

    ymin = int(
        np.floor(
            deme_meta["y"].min()
        )
    )
    ymax = int(
        np.ceil(
            deme_meta["y"].max()
        )
    )

    tiles = np.array(
        [
            (x, y)
            for y in range(ymin, ymax + 1)
            for x in range(xmin, xmax + 1)
        ],
        dtype=int,
    )

    n_tiles = len(tiles)

    if n_tiles == 0:
        raise ContinuousScaffoldError(
            "No spatial tiles generated."
        )

    if n_total < n_tiles:
        raise ContinuousScaffoldError(
            "Fewer cells than spatial tiles; cannot "
            "guarantee complete footprint occupancy."
        )

    # Nearly uniform cell density across the complete rectangle.
    base = n_total // n_tiles
    remainder = n_total % n_tiles

    target_counts = np.full(
        n_tiles,
        base,
        dtype=int,
    )

    if remainder:
        # Deterministic seeded distribution of residual cells.
        chosen = rng.permutation(
            n_tiles
        )[:remainder]

        target_counts[chosen] += 1

    # ------------------------------------------------------------
    # Distance-minimizing transfer from DEMON demes to filled
    # spatial tiles while preserving each original deme population.
    # ------------------------------------------------------------

    plan = _transport_plan(
        source_xy=source_xy,
        source_counts=source_counts,
        target_xy=tiles.astype(float),
        target_counts=target_counts,
    )

    records = []
    next_cell = 0

    original_occupied = {
        (
            int(row.x),
            int(row.y),
        )
        for row in deme_meta.itertuples(
            index=False
        )
    }

    # ------------------------------------------------------------
    # Expand exact clone composition within each source deme.
    # ------------------------------------------------------------

    for source_index, source_id in enumerate(
        source_ids
    ):
        source_rows = (
            composition[
                composition["deme_id"]
                == source_id
            ]
            .copy()
        )

        biological_records = []

        for row in source_rows.itertuples(
            index=False
        ):
            n = int(row.n_cells)

            biological_records.extend(
                [
                    (
                        row.subclone_id,
                        int(
                            row.driver_genotype_id
                        ),
                    )
                ]
                * n
            )

        if len(biological_records) != int(
            source_counts[source_index]
        ):
            raise ContinuousScaffoldError(
                "Source-deme population expansion "
                "failed."
            )

        order = rng.permutation(
            len(biological_records)
        )

        biological_records = [
            biological_records[i]
            for i in order
        ]

        cursor = 0

        source_x = float(
            source_xy[source_index, 0]
        )
        source_y = float(
            source_xy[source_index, 1]
        )

        for target_index in range(n_tiles):
            n_here = int(
                plan[
                    source_index,
                    target_index,
                ]
            )

            if n_here == 0:
                continue

            target_x = int(
                tiles[target_index, 0]
            )
            target_y = int(
                tiles[target_index, 1]
            )

            chunk = biological_records[
                cursor:
                cursor + n_here
            ]

            cursor += n_here

            x = (
                target_x
                + rng.uniform(
                    -jitter,
                    jitter,
                    size=n_here,
                )
            )

            y = (
                target_y
                + rng.uniform(
                    -jitter,
                    jitter,
                    size=n_here,
                )
            )

            cell_numbers = np.arange(
                next_cell,
                next_cell + n_here,
                dtype=np.int64,
            )

            block = pd.DataFrame(
                {
                    "cell_id": [
                        f"cell_{value:07d}"
                        for value in cell_numbers
                    ],

                    # Keep original DEMON identity.
                    "deme_id": int(source_id),

                    # Final spatial tile used by the compartment
                    # model to construct occupancy/boundaries.
                    "deme_x": target_x,
                    "deme_y": target_y,

                    "x": x,
                    "y": y,

                    "subclone_id": [
                        value[0]
                        for value in chunk
                    ],

                    "driver_genotype_id": [
                        value[1]
                        for value in chunk
                    ],

                    # Explicit provenance.
                    "source_demon_deme_id":
                        int(source_id),

                    "source_demon_deme_x":
                        source_x,

                    "source_demon_deme_y":
                        source_y,

                    "spatial_tile_x":
                        target_x,

                    "spatial_tile_y":
                        target_y,
                }
            )

            records.append(block)

            next_cell += n_here

        if cursor != len(
            biological_records
        ):
            raise ContinuousScaffoldError(
                "Not all source-deme cells were "
                "transported."
            )

    cells = pd.concat(
        records,
        ignore_index=True,
    )

    if len(cells) != n_total:
        raise ContinuousScaffoldError(
            "Population conservation failed: "
            f"{len(cells)} != {n_total}"
        )

    # ------------------------------------------------------------
    # Exact original DEMON deme × clone conservation.
    # ------------------------------------------------------------

    observed = (
        cells
        .groupby(
            [
                "source_demon_deme_id",
                "driver_genotype_id",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "observed_n_cells"
            }
        )
    )

    expected = (
        composition[
            [
                "deme_id",
                "driver_genotype_id",
                "n_cells",
            ]
        ]
        .rename(
            columns={
                "deme_id":
                    "source_demon_deme_id",
                "n_cells":
                    "expected_n_cells",
            }
        )
    )

    check = (
        expected
        .merge(
            observed,
            on=[
                "source_demon_deme_id",
                "driver_genotype_id",
            ],
            how="outer",
        )
        .fillna(0)
    )

    clone_conserved = bool(
        (
            check["expected_n_cells"].astype(int)
            ==
            check["observed_n_cells"].astype(int)
        ).all()
    )

    if not clone_conserved:
        raise ContinuousScaffoldError(
            "Original DEMON deme × clone composition "
            "was not conserved."
        )

    tile_counts = (
        cells
        .groupby(
            [
                "spatial_tile_x",
                "spatial_tile_y",
            ]
        )
        .size()
    )

    all_tiles_occupied = (
        len(tile_counts) == n_tiles
        and int(tile_counts.min()) > 0
    )

    if not all_tiles_occupied:
        raise ContinuousScaffoldError(
            "One or more target spatial tiles are empty."
        )

    within_tile = (
        (
            cells["x"]
            >= cells["spatial_tile_x"] - 0.5
        )
        &
        (
            cells["x"]
            < cells["spatial_tile_x"] + 0.5
        )
        &
        (
            cells["y"]
            >= cells["spatial_tile_y"] - 0.5
        )
        &
        (
            cells["y"]
            < cells["spatial_tile_y"] + 0.5
        )
    )

    coordinates_valid = bool(
        within_tile.all()
    )

    if not coordinates_valid:
        raise ContinuousScaffoldError(
            "One or more cells lie outside their "
            "target spatial tile."
        )

    n_original_occupied = len(
        original_occupied
    )

    summary = {
        "source":
            "canonical_demon_scaffold",

        "n_cells":
            int(len(cells)),

        "n_demes":
            int(len(source_ids)),

        "n_extant_subclones":
            int(
                cells[
                    "driver_genotype_id"
                ].nunique()
            ),

        "master_seed":
            int(seed),

        "spatial_rng_stage":
            1001,

        "coordinate_model":
            "continuous_rectangle_v1",

        "population_conserved":
            True,

        "deme_subclone_composition_conserved":
            clone_conserved,

        "all_cells_within_source_deme":
            False,

        "all_cells_within_target_tile":
            coordinates_valid,

        "all_target_tiles_occupied":
            all_tiles_occupied,

        "n_original_demon_demes":
            int(n_original_occupied),

        "n_spatial_tiles":
            int(n_tiles),

        "n_filled_previous_empty_tiles":
            int(
                n_tiles
                - n_original_occupied
            ),

        "rectangle_xmin":
            int(xmin),

        "rectangle_xmax":
            int(xmax),

        "rectangle_ymin":
            int(ymin),

        "rectangle_ymax":
            int(ymax),

        "min_cells_per_tile":
            int(tile_counts.min()),

        "max_cells_per_tile":
            int(tile_counts.max()),

        "coordinate_jitter":
            float(jitter),

        "biological_identity_assigned":
            False,

        "note":
            (
                "DEMON lineage counts are preserved exactly, "
                "but the cell coordinates are redistributed "
                "over a completely occupied rectangular "
                "tissue footprint."
            ),
    }

    return cells, summary
