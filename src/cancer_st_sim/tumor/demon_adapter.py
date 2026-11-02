from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


class DemonAdapterError(RuntimeError):
    """Raised when DEMON output cannot be converted safely."""


@dataclass
class DemonAdapterResult:
    clone_tree: pd.DataFrame
    clone_sizes: pd.DataFrame
    deme_coordinates: pd.DataFrame
    deme_clone_composition: pd.DataFrame
    summary: dict[str, Any]


def _read_whitespace_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DemonAdapterError(
            f"Required DEMON file does not exist: {path}"
        )

    if path.stat().st_size == 0:
        raise DemonAdapterError(
            f"Required DEMON file is empty: {path}"
        )

    return pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
    )


def _final_generation(
    df: pd.DataFrame,
    column: str = "Generation",
) -> float:
    if column not in df.columns:
        raise DemonAdapterError(
            f"Missing required column {column!r}."
        )

    return float(df[column].max())


def _final_rows(
    df: pd.DataFrame,
    generation: float,
    column: str = "Generation",
) -> pd.DataFrame:
    # DEMON writes repeated floating-point generation values.
    # Selecting the exact maximum is safe because all rows from the
    # terminal snapshot share the same stored value.
    out = df.loc[
        df[column] == generation
    ].copy()

    if out.empty:
        raise DemonAdapterError(
            f"No rows found for final generation {generation}."
        )

    return out


def _subclone_name(driver_id: int) -> str:
    return f"clone_{int(driver_id)}"


def _read_effective_seed(
    parameters_path: Path,
) -> int | None:
    if not parameters_path.exists():
        return None

    try:
        parameters = pd.read_csv(
            parameters_path,
            sep=r"\s+",
            engine="python",
        )
    except Exception:
        return None

    if (
        parameters.empty
        or "seed" not in parameters.columns
    ):
        return None

    return int(parameters.iloc[-1]["seed"])


def build_demon_scaffold(
    raw_dir: Path,
    requested_seed: int | None = None,
) -> DemonAdapterResult:
    raw_dir = raw_dir.resolve()

    clones = _read_whitespace_table(
        raw_dir / "clones.dat"
    )

    demes = _read_whitespace_table(
        raw_dir / "demes.dat"
    )

    driver_properties = _read_whitespace_table(
        raw_dir / "output_driver_genotype_properties.dat"
    )

    # ------------------------------------------------------------
    # Select final DEMON state.
    # ------------------------------------------------------------

    clone_generation = _final_generation(
        clones
    )

    deme_generation = _final_generation(
        demes
    )

    final_clones = _final_rows(
        clones,
        clone_generation,
    )

    final_demes = _final_rows(
        demes,
        deme_generation,
    )

    # Only positive tumor populations belong to our scaffold.
    final_clones = final_clones.loc[
        final_clones["Population"] > 0
    ].copy()

    occupied_demes = final_demes.loc[
        final_demes["Population"] > 0
    ].copy()

    # ------------------------------------------------------------
    # Canonical clone tree.
    # ------------------------------------------------------------

    required_driver_columns = {
        "Population",
        "Parent",
        "DriverIdentity",
        "DriverMutations",
        "OriginTime",
        "Descendants",
    }

    missing = (
        required_driver_columns
        - set(driver_properties.columns)
    )

    if missing:
        raise DemonAdapterError(
            "Driver genotype table is missing columns: "
            + ", ".join(sorted(missing))
        )

    tree = driver_properties[
        [
            "DriverIdentity",
            "Parent",
            "Population",
            "DriverMutations",
            "OriginTime",
            "Descendants",
        ]
    ].copy()

    tree = tree.rename(
        columns={
            "DriverIdentity":
                "driver_genotype_id",
            "Parent":
                "driver_parent_id",
            "Population":
                "reported_final_population",
            "DriverMutations":
                "n_driver_mutations",
            "OriginTime":
                "origin_generation",
            "Descendants":
                "descendant_population",
        }
    )

    tree["driver_genotype_id"] = (
        tree["driver_genotype_id"].astype(int)
    )

    tree["driver_parent_id"] = (
        tree["driver_parent_id"].astype(int)
    )

    tree["subclone_id"] = (
        tree["driver_genotype_id"]
        .map(_subclone_name)
    )

    tree["parent_subclone_id"] = (
        tree["driver_parent_id"]
        .map(_subclone_name)
    )

    # Root is its own DEMON parent. Represent its biological parent
    # explicitly as NA in the canonical simulator table.
    root_mask = (
        tree["driver_genotype_id"]
        == tree["driver_parent_id"]
    )

    tree.loc[
        root_mask,
        "parent_subclone_id",
    ] = pd.NA

    # Final population should be derived directly from final clones.dat,
    # not trusted from a secondary output.
    final_clone_sizes = (
        final_clones
        .groupby(
            "DriverGenotype",
            as_index=False,
        )["Population"]
        .sum()
        .rename(
            columns={
                "DriverGenotype":
                    "driver_genotype_id",
                "Population":
                    "final_population",
            }
        )
    )

    final_clone_sizes[
        "driver_genotype_id"
    ] = (
        final_clone_sizes[
            "driver_genotype_id"
        ].astype(int)
    )

    tree = tree.merge(
        final_clone_sizes,
        on="driver_genotype_id",
        how="left",
    )

    tree["final_population"] = (
        tree["final_population"]
        .fillna(0)
        .astype(int)
    )

    tree = tree[
        [
            "subclone_id",
            "parent_subclone_id",
            "driver_genotype_id",
            "driver_parent_id",
            "final_population",
            "n_driver_mutations",
            "origin_generation",
            "descendant_population",
            "reported_final_population",
        ]
    ].sort_values(
        [
            "n_driver_mutations",
            "origin_generation",
            "driver_genotype_id",
        ]
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Canonical clone sizes.
    # ------------------------------------------------------------

    clone_sizes = (
        final_clone_sizes.copy()
    )

    clone_sizes["subclone_id"] = (
        clone_sizes[
            "driver_genotype_id"
        ].map(_subclone_name)
    )

    clone_sizes = clone_sizes[
        [
            "subclone_id",
            "driver_genotype_id",
            "final_population",
        ]
    ].sort_values(
        "final_population",
        ascending=False,
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Canonical deme × clone composition.
    #
    # clones.dat already provides the DEMON deme ID and coordinates.
    # Aggregate away passenger-genotype subdivisions so the canonical
    # biological clone is the driver genotype.
    # ------------------------------------------------------------

    required_clone_columns = {
        "Deme",
        "X",
        "Y",
        "DriverGenotype",
        "Population",
    }

    missing = (
        required_clone_columns
        - set(final_clones.columns)
    )

    if missing:
        raise DemonAdapterError(
            "Final clones table is missing columns: "
            + ", ".join(sorted(missing))
        )

    composition = (
        final_clones
        .groupby(
            [
                "Deme",
                "X",
                "Y",
                "DriverGenotype",
            ],
            as_index=False,
        )["Population"]
        .sum()
    )

    composition = composition.rename(
        columns={
            "Deme":
                "deme_id",
            "X":
                "x",
            "Y":
                "y",
            "DriverGenotype":
                "driver_genotype_id",
            "Population":
                "n_cells",
        }
    )

    composition["deme_id"] = (
        composition["deme_id"].astype(int)
    )

    composition[
        "driver_genotype_id"
    ] = (
        composition[
            "driver_genotype_id"
        ].astype(int)
    )

    composition["n_cells"] = (
        composition["n_cells"].astype(int)
    )

    composition["subclone_id"] = (
        composition[
            "driver_genotype_id"
        ].map(_subclone_name)
    )

    parent_lookup = dict(
        zip(
            tree["driver_genotype_id"],
            tree["driver_parent_id"],
        )
    )

    composition["driver_parent_id"] = (
        composition[
            "driver_genotype_id"
        ].map(parent_lookup)
    )

    composition["parent_subclone_id"] = (
        composition[
            "driver_parent_id"
        ].map(
            lambda value:
                _subclone_name(int(value))
                if pd.notna(value)
                else pd.NA
        )
    )

    root_ids = set(
        tree.loc[
            tree["parent_subclone_id"].isna(),
            "driver_genotype_id",
        ]
    )

    composition.loc[
        composition[
            "driver_genotype_id"
        ].isin(root_ids),
        "parent_subclone_id",
    ] = pd.NA

    composition = composition[
        [
            "deme_id",
            "x",
            "y",
            "subclone_id",
            "parent_subclone_id",
            "driver_genotype_id",
            "driver_parent_id",
            "n_cells",
        ]
    ].sort_values(
        [
            "deme_id",
            "driver_genotype_id",
        ]
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Canonical occupied-deme coordinates.
    #
    # We intentionally omit DEMON-only normal demes. The active
    # simulator later creates its own biological normal/TME cells.
    # ------------------------------------------------------------

    deme_from_composition = (
        composition
        .groupby(
            [
                "deme_id",
                "x",
                "y",
            ],
            as_index=False,
        )["n_cells"]
        .sum()
        .rename(
            columns={
                "n_cells":
                    "tumor_cells",
            }
        )
    )

    final_deme_lookup = (
        final_demes[
            [
                "X",
                "Y",
                "Population",
                "NormalCells",
            ]
        ]
        .rename(
            columns={
                "X": "x",
                "Y": "y",
                "Population":
                    "reported_tumor_cells",
                "NormalCells":
                    "demon_normal_cells",
            }
        )
    )

    deme_coordinates = (
        deme_from_composition
        .merge(
            final_deme_lookup,
            on=["x", "y"],
            how="left",
            validate="one_to_one",
        )
    )

    deme_coordinates[
        "reported_tumor_cells"
    ] = (
        deme_coordinates[
            "reported_tumor_cells"
        ].fillna(0).astype(int)
    )

    deme_coordinates[
        "demon_normal_cells"
    ] = (
        deme_coordinates[
            "demon_normal_cells"
        ].fillna(0).astype(int)
    )

    deme_coordinates = (
        deme_coordinates[
            [
                "deme_id",
                "x",
                "y",
                "tumor_cells",
                "demon_normal_cells",
            ]
        ]
        .sort_values("deme_id")
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # Conservation checks.
    # ------------------------------------------------------------

    tumor_cells_from_clones = int(
        final_clones["Population"].sum()
    )

    tumor_cells_from_composition = int(
        composition["n_cells"].sum()
    )

    tumor_cells_from_demes = int(
        occupied_demes["Population"].sum()
    )

    clone_population_from_sizes = int(
        clone_sizes[
            "final_population"
        ].sum()
    )

    global_population_conserved = (
        tumor_cells_from_clones
        == tumor_cells_from_composition
        == tumor_cells_from_demes
        == clone_population_from_sizes
    )

    comp_by_clone = (
        composition
        .groupby(
            "driver_genotype_id"
        )["n_cells"]
        .sum()
        .sort_index()
    )

    size_by_clone = (
        clone_sizes
        .set_index(
            "driver_genotype_id"
        )["final_population"]
        .sort_index()
    )

    subclone_population_conserved = (
        comp_by_clone.equals(
            size_by_clone
        )
    )

    comp_by_deme = (
        composition
        .groupby(
            "deme_id"
        )["n_cells"]
        .sum()
        .sort_index()
    )

    coords_by_deme = (
        deme_coordinates
        .set_index(
            "deme_id"
        )["tumor_cells"]
        .sort_index()
    )

    deme_population_conserved = (
        comp_by_deme.equals(
            coords_by_deme
        )
    )

    if not global_population_conserved:
        raise DemonAdapterError(
            "Global DEMON population was not conserved "
            "during scaffold conversion."
        )

    if not subclone_population_conserved:
        raise DemonAdapterError(
            "Subclone populations were not conserved "
            "during scaffold conversion."
        )

    if not deme_population_conserved:
        raise DemonAdapterError(
            "Deme populations were not conserved "
            "during scaffold conversion."
        )

    effective_seed = _read_effective_seed(
        raw_dir / "parameters.dat"
    )

    n_trials = None

    error_log_path = (
        raw_dir / "error_log.dat"
    )

    if error_log_path.exists():
        text = error_log_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        import re

        match = re.search(
            r"Ran\s+(\d+)\s+trials",
            text,
        )

        if match:
            n_trials = int(
                match.group(1)
            )

    summary: dict[str, Any] = {
        "source": "DEMON",
        "subclone_definition": "driver_genotype",
        "raw_directory": str(raw_dir),

        "final_clone_generation":
            clone_generation,

        "final_deme_generation":
            deme_generation,

        "tumor_cells":
            tumor_cells_from_clones,

        "occupied_tumor_demes":
            int(
                deme_coordinates.shape[0]
            ),

        "extant_driver_subclones":
            int(
                (
                    clone_sizes[
                        "final_population"
                    ] > 0
                ).sum()
            ),

        "driver_nodes_in_tree":
            int(tree.shape[0]),

        "requested_seed":
            requested_seed,

        "effective_demon_seed":
            effective_seed,

        "demon_trials":
            n_trials,

        "global_population_conserved":
            bool(
                global_population_conserved
            ),

        "subclone_population_conserved":
            bool(
                subclone_population_conserved
            ),

        "deme_population_conserved":
            bool(
                deme_population_conserved
            ),

        "demon_normal_cells_used_as_final_tme":
            False,

        "note":
            (
                "DEMON normal cells participate in DEMON's "
                "internal invasion/competition dynamics only. "
                "They are not exported as biological normal/TME "
                "cells. The active simulator will later relabel "
                "a subset of the DEMON scaffold cells as "
                "normal/TME cells using its own spatial model."
            ),
    }

    return DemonAdapterResult(
        clone_tree=tree,
        clone_sizes=clone_sizes,
        deme_coordinates=deme_coordinates,
        deme_clone_composition=composition,
        summary=summary,
    )


def write_demon_scaffold(
    raw_dir: Path,
    output_dir: Path,
    requested_seed: int | None = None,
) -> DemonAdapterResult:
    result = build_demon_scaffold(
        raw_dir=raw_dir,
        requested_seed=requested_seed,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.clone_tree.to_csv(
        output_dir / "clone_tree.tsv",
        sep="\t",
        index=False,
    )

    result.clone_sizes.to_csv(
        output_dir / "clone_sizes.tsv",
        sep="\t",
        index=False,
    )

    result.deme_coordinates.to_csv(
        output_dir / "deme_coordinates.tsv",
        sep="\t",
        index=False,
    )

    result.deme_clone_composition.to_csv(
        output_dir / "deme_clone_composition.tsv",
        sep="\t",
        index=False,
    )

    with (
        output_dir
        / "scaffold_summary.yaml"
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
