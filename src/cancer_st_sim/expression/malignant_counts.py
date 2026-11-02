from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import sparse


STATES = [
    "G2M",
    "S_PHASE",
    "HYPOXIA",
    "IFN_BROAD",
    "IFN_INFLAMMATORY",
    "EMT",
]


class MalignantCountError(RuntimeError):
    pass


@dataclass
class MalignantCountResult:
    summary: dict[str, Any]


def _malignant_mask(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1"])
        .to_numpy()
    )


def _load_library_pool(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, sep="\t")

    candidates = [
        "library_size",
        "rna_library_size",
        "total_counts",
    ]

    column = next(
        (x for x in candidates if x in frame.columns),
        None,
    )

    if column is None:
        raise MalignantCountError(
            "Could not identify library-size column."
        )

    values = pd.to_numeric(
        frame[column],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    values = values[
        np.isfinite(values)
        & (values > 0)
    ]

    if len(values) == 0:
        raise MalignantCountError(
            "No positive empirical library sizes."
        )

    return values


def _draw_nb2(
    rng: np.random.Generator,
    mu: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    if np.any(mu < 0) or not np.isfinite(mu).all():
        raise MalignantCountError(
            "Invalid NB2 mean encountered."
        )

    if np.any(phi < 0) or not np.isfinite(phi).all():
        raise MalignantCountError(
            "Invalid NB2 dispersion encountered."
        )

    counts = np.zeros(
        mu.shape,
        dtype=np.int64,
    )

    poisson_genes = phi <= 1e-12

    if poisson_genes.any():
        counts[:, poisson_genes] = rng.poisson(
            mu[:, poisson_genes]
        )

    nb_genes = ~poisson_genes

    if nb_genes.any():
        phi_nb = phi[nb_genes]

        size = (
            1.0
            / phi_nb
        )

        mu_nb = mu[:, nb_genes]

        probability = (
            size[np.newaxis, :]
            /
            (
                size[np.newaxis, :]
                + mu_nb
            )
        )

        counts[:, nb_genes] = (
            rng.negative_binomial(
                n=size[np.newaxis, :],
                p=probability,
            )
        )

    return counts


def generate_malignant_counts(
    *,
    cells_path: Path,
    states_path: Path,
    clone_gene_cna_path: Path,
    parameter_path: Path,
    library_size_path: Path,
    output_dir: Path,
    seed: int,
    chunk_size: int = 250,
    preflight_only: bool = False,
    preflight_cells: int = 200,
    library_rng_stage: int = 5001,
    count_rng_stage: int = 5002,
    state_names: list[str] | None = None,
) -> MalignantCountResult:
    state_names = (
        list(state_names)
        if state_names is not None
        else list(STATES)
    )
    if not state_names or len(set(state_names)) != len(state_names):
        raise MalignantCountError('Invalid tumor-state names')

    if chunk_size <= 0:
        raise MalignantCountError(
            "chunk_size must be positive."
        )

    cells = pd.read_csv(
        cells_path,
        sep="\t",
        low_memory=False,
    )

    states = pd.read_csv(
        states_path,
        sep="\t",
        low_memory=False,
    )

    parameters = pd.read_csv(
        parameter_path,
        sep="\t",
        low_memory=False,
    )

    required_cells = {
        "cell_id",
        "is_malignant",
        "subclone_id",
    }

    missing = required_cells - set(cells.columns)

    if missing:
        raise MalignantCountError(
            f"Cell table missing: {sorted(missing)}"
        )

    required_states = {
        "cell_id",
        *state_names,
    }

    missing = required_states - set(states.columns)

    if missing:
        raise MalignantCountError(
            f"State table missing: {sorted(missing)}"
        )

    required_parameters = {
        "gene_id",
        "gene_symbol",
        "alpha",
        "beta",
        "dispersion",
        "state_effect_lower",
        "state_effect_upper",
        *[
            f"gamma_{state}"
            for state in state_names
        ],
    }

    missing = (
        required_parameters
        - set(parameters.columns)
    )

    if missing:
        raise MalignantCountError(
            f"Parameter table missing: {sorted(missing)}"
        )

    if parameters["gene_id"].duplicated().any():
        raise MalignantCountError(
            "Duplicate runtime gene IDs."
        )

    malignant = (
        cells.loc[
            _malignant_mask(
                cells["is_malignant"]
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    if malignant["cell_id"].duplicated().any():
        raise MalignantCountError(
            "Duplicate malignant cell IDs."
        )

    malignant = malignant.merge(
        states[
            [
                "cell_id",
                *state_names,
            ]
        ],
        on="cell_id",
        how="left",
        validate="one_to_one",
    )

    if malignant[state_names].isna().any().any():
        raise MalignantCountError(
            "Missing malignant state scores."
        )

    gene_ids = (
        parameters["gene_id"]
        .astype(str)
        .tolist()
    )

    runtime_gene_set = set(gene_ids)

    #
    # Load only CNA rows needed by the runtime expression universe.
    #
    cna_parts = []

    for chunk in pd.read_csv(
        clone_gene_cna_path,
        sep="\t",
        usecols=[
            "subclone_id",
            "gene_id",
            "total_cn",
        ],
        chunksize=500_000,
    ):
        keep = chunk[
            chunk["gene_id"]
            .astype(str)
            .isin(runtime_gene_set)
        ]

        if not keep.empty:
            cna_parts.append(keep)

    if not cna_parts:
        raise MalignantCountError(
            "No runtime genes found in CNA table."
        )

    cna = pd.concat(
        cna_parts,
        ignore_index=True,
    )

    if cna.duplicated(
        ["subclone_id", "gene_id"]
    ).any():
        raise MalignantCountError(
            "Duplicate clone-gene CNA rows."
        )

    cna_matrix = (
        cna.pivot(
            index="subclone_id",
            columns="gene_id",
            values="total_cn",
        )
        .reindex(columns=gene_ids)
    )

    required_clones = set(
        malignant["subclone_id"]
        .astype(str)
    )

    missing_clones = (
        required_clones
        - set(
            cna_matrix.index.astype(str)
        )
    )

    if missing_clones:
        raise MalignantCountError(
            "Missing CNA for malignant clones: "
            f"{sorted(missing_clones)[:10]}"
        )

    if cna_matrix.isna().any().any():
        raise MalignantCountError(
            "Missing clone-gene CNA values."
        )

    cna_matrix.index = (
        cna_matrix.index.astype(str)
    )

    alpha = parameters[
        "alpha"
    ].to_numpy(dtype=np.float64)

    beta = parameters[
        "beta"
    ].to_numpy(dtype=np.float64)

    dispersion = parameters[
        "dispersion"
    ].to_numpy(dtype=np.float64)

    gamma = parameters[
        [
            f"gamma_{state}"
            for state in state_names
        ]
    ].to_numpy(dtype=np.float64)

    state_effect_lower = parameters[
        "state_effect_lower"
    ].to_numpy(dtype=np.float64)

    state_effect_upper = parameters[
        "state_effect_upper"
    ].to_numpy(dtype=np.float64)

    if np.any(
        state_effect_lower
        > state_effect_upper
    ):
        raise MalignantCountError(
            "Invalid state-effect bounds."
        )

    if not np.isfinite(
        np.concatenate(
            [
                alpha,
                beta,
                dispersion,
                gamma.ravel(),
                state_effect_lower,
                state_effect_upper,
            ]
        )
    ).all():
        raise MalignantCountError(
            "Non-finite expression parameter."
        )

    if np.any(beta <= 0):
        raise MalignantCountError(
            "Runtime beta must be positive."
        )

    if np.any(dispersion < 0):
        raise MalignantCountError(
            "Dispersion must be non-negative."
        )

    library_pool = _load_library_pool(
        library_size_path
    )

    library_rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                int(library_rng_stage),
            ]
        )
    )

    latent_library_size = (
        library_rng.choice(
            library_pool,
            size=len(malignant),
            replace=True,
        )
        .astype(np.float64)
    )

    malignant[
        "latent_library_size"
    ] = latent_library_size

    if preflight_only:
        n_cells = min(
            int(preflight_cells),
            len(malignant),
        )
    else:
        n_cells = len(malignant)

    work_cells = (
        malignant.iloc[:n_cells]
        .copy()
        .reset_index(drop=True)
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunk_dir = (
        output_dir
        / "count_chunks"
    )

    if not preflight_only:
        chunk_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    count_rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                int(count_rng_stage),
            ]
        )
    )

    chunk_records = []

    total_counts = 0
    total_nonzero = 0
    max_mu = 0.0
    sum_expected_counts = 0.0
    cn0_gene_cell_pairs = 0
    cn0_nonzero_counts = 0

    state_effect_values = 0
    state_effect_clipped_low = 0
    state_effect_clipped_high = 0
    max_raw_state_effect = -np.inf
    min_raw_state_effect = np.inf
    max_runtime_state_effect = -np.inf
    min_runtime_state_effect = np.inf

    cell_count_totals = np.zeros(
        n_cells,
        dtype=np.int64,
    )

    for start in range(
        0,
        n_cells,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            n_cells,
        )

        chunk = work_cells.iloc[
            start:end
        ]

        clone_ids = (
            chunk[
                "subclone_id"
            ]
            .astype(str)
            .tolist()
        )

        cn = (
            cna_matrix.loc[
                clone_ids,
                gene_ids,
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        z = chunk[
            state_names
        ].to_numpy(
            dtype=np.float64
        )

        lib = chunk[
            "latent_library_size"
        ].to_numpy(
            dtype=np.float64
        )

        raw_state_effect = (
            z
            @ gamma.T
        )

        low_mask = (
            raw_state_effect
            < state_effect_lower[
                np.newaxis,
                :
            ]
        )

        high_mask = (
            raw_state_effect
            > state_effect_upper[
                np.newaxis,
                :
            ]
        )

        state_effect_clipped_low += int(
            low_mask.sum()
        )

        state_effect_clipped_high += int(
            high_mask.sum()
        )

        state_effect_values += int(
            raw_state_effect.size
        )

        max_raw_state_effect = max(
            max_raw_state_effect,
            float(
                raw_state_effect.max()
            ),
        )

        min_raw_state_effect = min(
            min_raw_state_effect,
            float(
                raw_state_effect.min()
            ),
        )

        state_effect = np.clip(
            raw_state_effect,
            state_effect_lower[
                np.newaxis,
                :
            ],
            state_effect_upper[
                np.newaxis,
                :
            ],
        )

        max_runtime_state_effect = max(
            max_runtime_state_effect,
            float(
                state_effect.max()
            ),
        )

        min_runtime_state_effect = min(
            min_runtime_state_effect,
            float(
                state_effect.min()
            ),
        )

        positive_cn = (
            cn > 0
        )

        log_mu = (
            np.log(
                lib
            )[:, np.newaxis]
            + alpha[np.newaxis, :]
            + state_effect
        )

        #
        # CNA term is only evaluated for positive CN.
        #
        cna_effect = np.zeros_like(
            cn,
            dtype=np.float64,
        )

        cna_effect[
            positive_cn
        ] = (
            np.broadcast_to(
                beta,
                cn.shape,
            )[
                positive_cn
            ]
            * np.log(
                cn[
                    positive_cn
                ]
                / 2.0
            )
        )

        log_mu += cna_effect

        #
        # Avoid evaluating exp() for hard-zero CN entries.
        #
        mu = np.zeros_like(
            log_mu,
            dtype=np.float64,
        )

        mu[
            positive_cn
        ] = np.exp(
            log_mu[
                positive_cn
            ]
        )

        if not np.isfinite(mu).all():
            raise MalignantCountError(
                f"Non-finite mu in cells {start}:{end}."
            )

        counts = _draw_nb2(
            rng=count_rng,
            mu=mu,
            phi=dispersion,
        )

        zero_mask = ~positive_cn

        cn0_gene_cell_pairs += int(
            zero_mask.sum()
        )

        cn0_nonzero_counts += int(
            counts[
                zero_mask
            ].sum()
        )

        if counts[
            zero_mask
        ].any():
            raise MalignantCountError(
                "CN=0 generated non-zero counts."
            )

        chunk_cell_totals = (
            counts.sum(axis=1)
        )

        cell_count_totals[
            start:end
        ] = chunk_cell_totals

        chunk_total = int(
            counts.sum()
        )

        chunk_nonzero = int(
            np.count_nonzero(
                counts
            )
        )

        total_counts += chunk_total
        total_nonzero += chunk_nonzero

        sum_expected_counts += float(
            mu.sum()
        )

        max_mu = max(
            max_mu,
            float(
                mu.max()
            ),
        )

        if not preflight_only:
            matrix = sparse.csr_matrix(
                counts
            )

            filename = (
                f"counts_{start:07d}_{end - 1:07d}.npz"
            )

            sparse.save_npz(
                chunk_dir
                / filename,
                matrix,
                compressed=True,
            )

            chunk_records.append(
                {
                    "chunk_index":
                        len(chunk_records),

                    "cell_start":
                        start,

                    "cell_end_exclusive":
                        end,

                    "n_cells":
                        end - start,

                    "filename":
                        filename,

                    "total_counts":
                        chunk_total,

                    "nonzero_entries":
                        chunk_nonzero,
                }
            )

    work_cells[
        "simulated_total_counts"
    ] = cell_count_totals

    summary = {
        "model":
            "FTSEC_CNA_state_NB2",

        "n_cells":
            int(n_cells),

        "n_genes":
            int(len(gene_ids)),

        "preflight_only":
            bool(preflight_only),

        "library_size_source":
            (
                "SA1053 empirical RNA library "
                "sizes sampled with replacement"
            ),

        "library_size_median":
            float(
                np.median(
                    work_cells[
                        "latent_library_size"
                    ]
                )
            ),

        "library_size_q05":
            float(
                np.quantile(
                    work_cells[
                        "latent_library_size"
                    ],
                    0.05,
                )
            ),

        "library_size_q95":
            float(
                np.quantile(
                    work_cells[
                        "latent_library_size"
                    ],
                    0.95,
                )
            ),

        "simulated_count_median":
            float(
                np.median(
                    cell_count_totals
                )
            ),

        "simulated_count_q05":
            float(
                np.quantile(
                    cell_count_totals,
                    0.05,
                )
            ),

        "simulated_count_q95":
            float(
                np.quantile(
                    cell_count_totals,
                    0.95,
                )
            ),

        "mean_expected_counts_per_cell":
            float(
                sum_expected_counts
                / n_cells
            ),

        "total_counts":
            int(total_counts),

        "total_nonzero_entries":
            int(total_nonzero),

        "max_mu":
            float(max_mu),

        "cn0_gene_cell_pairs":
            int(cn0_gene_cell_pairs),

        "cn0_nonzero_counts":
            int(cn0_nonzero_counts),

        "cn0_rule_passed":
            bool(
                cn0_nonzero_counts
                == 0
            ),

        "state_effect_stabilization": {
            "method":
                "gene_specific_reference_quantile_winsorization",

            "lower_quantile":
                0.005,

            "upper_quantile":
                0.995,

            "n_state_effect_values":
                int(
                    state_effect_values
                ),

            "n_clipped_low":
                int(
                    state_effect_clipped_low
                ),

            "n_clipped_high":
                int(
                    state_effect_clipped_high
                ),

            "n_clipped_total":
                int(
                    state_effect_clipped_low
                    + state_effect_clipped_high
                ),

            "fraction_clipped":
                float(
                    (
                        state_effect_clipped_low
                        + state_effect_clipped_high
                    )
                    / state_effect_values
                    if state_effect_values
                    else 0.0
                ),

            "raw_state_effect_min":
                float(
                    min_raw_state_effect
                ),

            "raw_state_effect_max":
                float(
                    max_raw_state_effect
                ),

            "runtime_state_effect_min":
                float(
                    min_runtime_state_effect
                ),

            "runtime_state_effect_max":
                float(
                    max_runtime_state_effect
                ),
        },

        "nb2_variance":
            "mu + phi * mu^2",

        "reference_copy_number":
            2,

        "rng": {
            "master_seed":
                int(seed),

            "library_stage":
                int(library_rng_stage),

            "count_stage":
                int(count_rng_stage),
        },
    }

    if preflight_only:
        with (
            output_dir
            / "preflight_summary.yaml"
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:
            yaml.safe_dump(
                summary,
                handle,
                sort_keys=False,
            )

    else:
        parameters[
            [
                "gene_id",
                "gene_symbol",
                "baseline_fraction",
                "alpha",
                "beta",
                "dispersion",
            ]
        ].to_csv(
            output_dir
            / "genes.tsv",
            sep="\t",
            index=False,
        )

        work_cells.to_csv(
            output_dir
            / "cells.tsv",
            sep="\t",
            index=False,
        )

        pd.DataFrame(
            chunk_records
        ).to_csv(
            output_dir
            / "count_chunk_manifest.tsv",
            sep="\t",
            index=False,
        )

        with (
            output_dir
            / "summary.yaml"
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:
            yaml.safe_dump(
                summary,
                handle,
                sort_keys=False,
            )

    return MalignantCountResult(
        summary=summary
    )
