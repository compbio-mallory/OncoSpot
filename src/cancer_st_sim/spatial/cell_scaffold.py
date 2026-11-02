from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .continuous_scaffold import (
    build_continuous_rectangle_scaffold,
)


class CellScaffoldError(RuntimeError):
    """Raised when the cell-level scaffold cannot be built safely."""


@dataclass
class CellScaffoldResult:
    cells: pd.DataFrame
    summary: dict[str, Any]


def build_cell_scaffold(
    deme_clone_composition: pd.DataFrame,
    seed: int,
    spatial_preset: dict[str, Any] | None = None,
) -> CellScaffoldResult:

    spatial_preset = spatial_preset or {}

    scaffold_cfg = spatial_preset.get(
        "cell_scaffold",
        {},
    )

    if not isinstance(scaffold_cfg, dict):
        raise CellScaffoldError(
            "cell_scaffold preset must be a mapping."
        )

    scaffold_mode = scaffold_cfg.get(
        "mode",
        "demon_deme_v1",
    )

    if scaffold_mode == "continuous_rectangle_v1":
        cells, summary = (
            build_continuous_rectangle_scaffold(
                deme_clone_composition=
                    deme_clone_composition,
                seed=seed,
                settings=scaffold_cfg,
            )
        )

        return CellScaffoldResult(
            cells=cells,
            summary=summary,
        )

    if scaffold_mode not in {
        "demon_deme_v1",
        "uniform_within_demon_deme",
    }:
        raise CellScaffoldError(
            "Unknown cell scaffold mode: "
            f"{scaffold_mode}"
        )
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
        raise CellScaffoldError(
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

    if (
        composition["n_cells"] < 0
    ).any():
        raise CellScaffoldError(
            "Negative cell counts found in "
            "deme_clone_composition."
        )

    # Separate deterministic random stream for the spatial-cell stage.
    seed_sequence = np.random.SeedSequence(
        [
            int(seed),
            1001,
        ]
    )

    rng = np.random.default_rng(
        seed_sequence
    )

    records: list[pd.DataFrame] = []

    next_cell = 0

    for row in composition.itertuples(
        index=False
    ):
        n = int(row.n_cells)

        if n == 0:
            continue

        # DEMON deme centers are integer lattice coordinates.
        # Generate cells uniformly inside the corresponding
        # one-unit square centered on that DEMON coordinate.
        x = (
            float(row.x)
            + rng.uniform(
                -0.5,
                0.5,
                size=n,
            )
        )

        y = (
            float(row.y)
            + rng.uniform(
                -0.5,
                0.5,
                size=n,
            )
        )

        cell_numbers = np.arange(
            next_cell,
            next_cell + n,
            dtype=np.int64,
        )

        block = pd.DataFrame(
            {
                "cell_id": [
                    f"cell_{value:07d}"
                    for value in cell_numbers
                ],
                "deme_id":
                    int(row.deme_id),
                "deme_x":
                    int(row.x),
                "deme_y":
                    int(row.y),
                "x":
                    x,
                "y":
                    y,
                "subclone_id":
                    row.subclone_id,
                "driver_genotype_id":
                    int(
                        row.driver_genotype_id
                    ),
            }
        )

        records.append(
            block
        )

        next_cell += n

    if not records:
        raise CellScaffoldError(
            "No cells were generated."
        )

    cells = pd.concat(
        records,
        ignore_index=True,
    )

    expected_total = int(
        composition[
            "n_cells"
        ].sum()
    )

    actual_total = int(
        cells.shape[0]
    )

    if actual_total != expected_total:
        raise CellScaffoldError(
            "Cell expansion failed population conservation: "
            f"expected {expected_total}, "
            f"generated {actual_total}."
        )

    # ------------------------------------------------------------
    # Verify exact deme × clone conservation.
    # ------------------------------------------------------------

    observed = (
        cells
        .groupby(
            [
                "deme_id",
                "driver_genotype_id",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "observed_n_cells",
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
                "n_cells":
                    "expected_n_cells",
            }
        )
    )

    check = expected.merge(
        observed,
        on=[
            "deme_id",
            "driver_genotype_id",
        ],
        how="outer",
    ).fillna(0)

    check[
        "expected_n_cells"
    ] = (
        check[
            "expected_n_cells"
        ].astype(int)
    )

    check[
        "observed_n_cells"
    ] = (
        check[
            "observed_n_cells"
        ].astype(int)
    )

    composition_conserved = bool(
        (
            check[
                "expected_n_cells"
            ]
            == check[
                "observed_n_cells"
            ]
        ).all()
    )

    if not composition_conserved:
        raise CellScaffoldError(
            "Deme × subclone composition was not conserved."
        )

    # ------------------------------------------------------------
    # Coordinate checks.
    # ------------------------------------------------------------

    within_x = (
        (cells["x"] >= cells["deme_x"] - 0.5)
        &
        (cells["x"] < cells["deme_x"] + 0.5)
    )

    within_y = (
        (cells["y"] >= cells["deme_y"] - 0.5)
        &
        (cells["y"] < cells["deme_y"] + 0.5)
    )

    coordinates_valid = bool(
        (within_x & within_y).all()
    )

    if not coordinates_valid:
        raise CellScaffoldError(
            "One or more cells were generated outside "
            "their DEMON deme."
        )

    summary = {
        "source":
            "canonical_demon_scaffold",

        "n_cells":
            actual_total,

        "n_demes":
            int(
                cells["deme_id"].nunique()
            ),

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
            "uniform_within_demon_deme",

        "deme_width":
            1.0,

        "population_conserved":
            actual_total
            == expected_total,

        "deme_subclone_composition_conserved":
            composition_conserved,

        "all_cells_within_source_deme":
            coordinates_valid,

        "biological_identity_assigned":
            False,

        "note":
            (
                "Every row currently represents a DEMON-derived "
                "spatial scaffold cell. Malignant versus normal/TME "
                "identity has not yet been assigned."
            ),
    }

    return CellScaffoldResult(
        cells=cells,
        summary=summary,
    )


def write_cell_scaffold(
    scaffold_dir: Path,
    output_dir: Path,
    seed: int,
    spatial_preset_path: Path | None = None,
) -> CellScaffoldResult:
    composition_path = (
        scaffold_dir
        / "deme_clone_composition.tsv"
    )

    if not composition_path.exists():
        raise CellScaffoldError(
            f"Missing canonical scaffold file: "
            f"{composition_path}"
        )

    composition = pd.read_csv(
        composition_path,
        sep="\t",
    )

    spatial_preset: dict[str, Any] = {}

    if spatial_preset_path is not None:
        if not spatial_preset_path.exists():
            raise CellScaffoldError(
                "Missing spatial preset: "
                f"{spatial_preset_path}"
            )

        with spatial_preset_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            spatial_preset = (
                yaml.safe_load(handle)
                or {}
            )

    result = build_cell_scaffold(
        deme_clone_composition=composition,
        seed=seed,
        spatial_preset=spatial_preset,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.cells.to_csv(
        output_dir
        / "cells.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )

    with (
        output_dir
        / "cell_scaffold_summary.yaml"
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
