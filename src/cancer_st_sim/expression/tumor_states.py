from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


STATE_COLUMNS = [
    "G2M",
    "S_PHASE",
    "HYPOXIA",
    "IFN_BROAD",
    "IFN_INFLAMMATORY",
    "EMT",
]


class TumorStateError(
    RuntimeError
):
    pass


@dataclass
class TumorStateResult:
    malignant_states: pd.DataFrame
    summary: dict[str, Any]


def assign_empirical_tumor_states(
    cells: pd.DataFrame,
    reference: pd.DataFrame,
    seed: int,
    rng_stage: int = 4001,
    state_columns: list[str] | None = None,
) -> TumorStateResult:
    state_columns = (
        list(state_columns)
        if state_columns is not None
        else list(STATE_COLUMNS)
    )
    if not state_columns or len(set(state_columns)) != len(state_columns):
        raise TumorStateError('Invalid tumor-state names')

    required_cells = {
        "cell_id",
        "is_malignant",
    }

    missing = (
        required_cells
        - set(cells.columns)
    )

    if missing:
        raise TumorStateError(
            "Cell table missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    required_reference = {
        "reference_cell_id",
        *state_columns,
    }

    missing = (
        required_reference
        - set(reference.columns)
    )

    if missing:
        raise TumorStateError(
            "State reference missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    if reference.empty:
        raise TumorStateError(
            "State reference is empty."
        )

    if (
        reference[
            state_columns
        ]
        .isna()
        .any()
        .any()
    ):
        raise TumorStateError(
            "State reference contains missing values."
        )

    malignant_mask = (
        cells[
            "is_malignant"
        ]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
            ]
        )
    )

    malignant = (
        cells.loc[
            malignant_mask,
            [
                "cell_id",
            ],
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if malignant.empty:
        raise TumorStateError(
            "No malignant cells found."
        )

    #
    # Dedicated deterministic RNG stream for this biological stage.
    #
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                int(rng_stage),
            ]
        )
    )

    sampled_indices = rng.integers(
        low=0,
        high=len(reference),
        size=len(malignant),
    )

    sampled = (
        reference.iloc[
            sampled_indices
        ]
        .reset_index(
            drop=True
        )
    )

    result = malignant.copy()

    result[
        "state_reference_cell_id"
    ] = sampled[
        "reference_cell_id"
    ].to_numpy()

    result[
        "state_reference_row_index"
    ] = sampled_indices

    for state in state_columns:
        result[
            state
        ] = sampled[
            state
        ].to_numpy(
            dtype=float
        )

    #
    # Validation summaries.
    #
    reference_summary = (
        reference[
            state_columns
        ]
        .agg(
            [
                "mean",
                "std",
                "min",
                "median",
                "max",
            ]
        )
    )

    simulated_summary = (
        result[
            state_columns
        ]
        .agg(
            [
                "mean",
                "std",
                "min",
                "median",
                "max",
            ]
        )
    )

    max_abs_mean_difference = float(
        (
            simulated_summary.loc[
                "mean"
            ]
            - reference_summary.loc[
                "mean"
            ]
        )
        .abs()
        .max()
    )

    max_abs_sd_difference = float(
        (
            simulated_summary.loc[
                "std"
            ]
            - reference_summary.loc[
                "std"
            ]
        )
        .abs()
        .max()
    )

    ref_corr = (
        reference[
            state_columns
        ]
        .corr()
    )

    sim_corr = (
        result[
            state_columns
        ]
        .corr()
    )

    max_abs_correlation_difference = float(
        (
            sim_corr
            - ref_corr
        )
        .abs()
        .to_numpy()
        .max()
    )

    unique_reference_cells_used = int(
        result[
            "state_reference_cell_id"
        ].nunique()
    )

    summary = {
        "source":
            ("SA1053_empirical_state_reference" if state_columns == STATE_COLUMNS else "cancer_specific_empirical_state_reference"),

        "sampling_model":
            "row_sampling_with_replacement",

        "clone_conditioned":
            False,

        "lineage_inheritance":
            False,

        "n_reference_cells":
            int(
                len(reference)
            ),

        "n_simulated_malignant_cells":
            int(
                len(result)
            ),

        "n_unique_reference_cells_used":
            unique_reference_cells_used,

        "fraction_reference_cells_used":
            float(
                unique_reference_cells_used
                / len(reference)
            ),

        "master_seed":
            int(seed),

        "rng_stage":
            int(rng_stage),

        "states":
            state_columns,

        "validation": {
            "max_abs_mean_difference":
                max_abs_mean_difference,

            "max_abs_sd_difference":
                max_abs_sd_difference,

            "max_abs_correlation_difference":
                max_abs_correlation_difference,
        },

        "reference_state_summary":
            {
                state: {
                    "mean":
                        float(
                            reference[
                                state
                            ].mean()
                        ),

                    "sd":
                        float(
                            reference[
                                state
                            ].std()
                        ),

                    "median":
                        float(
                            reference[
                                state
                            ].median()
                        ),
                }
                for state
                in state_columns
            },

        "simulated_state_summary":
            {
                state: {
                    "mean":
                        float(
                            result[
                                state
                            ].mean()
                        ),

                    "sd":
                        float(
                            result[
                                state
                            ].std()
                        ),

                    "median":
                        float(
                            result[
                                state
                            ].median()
                        ),
                }
                for state
                in state_columns
            },

        "note":
            (
                "Each malignant simulated cell independently samples "
                "one complete empirical reference-state vector. "
                "Sampling complete rows preserves the empirical joint "
                "state distribution without clone-conditioned or "
                "lineage-inherited state models."
            ),
    }

    return TumorStateResult(
        malignant_states=result,
        summary=summary,
    )


def write_empirical_tumor_states(
    cells_path: Path,
    reference_path: Path,
    output_dir: Path,
    seed: int,
    rng_stage: int = 4001,
    state_columns: list[str] | None = None,
) -> TumorStateResult:

    cells = pd.read_csv(
        cells_path,
        sep="\t",
    )

    reference = pd.read_csv(
        reference_path,
        sep="\t",
    )

    result = assign_empirical_tumor_states(
        cells=cells,
        reference=reference,
        seed=seed,
        rng_stage=rng_stage,
        state_columns=state_columns,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.malignant_states.to_csv(
        output_dir
        / "malignant_cell_states.tsv",
        sep="\t",
        index=False,
    )

    with (
        output_dir
        / "tumor_state_summary.yaml"
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
