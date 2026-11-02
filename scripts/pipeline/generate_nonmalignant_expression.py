#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_FILE = (
    PROJECT_ROOT
    / "parameters/HGSOC/nonmalignant/expression/"
    "tme_expression_baselines.tsv"
)

LIBRARY_FILE = (
    PROJECT_ROOT
    / "parameters/HGSOC/nonmalignant/expression/"
    "library_sizes_by_cell_type.tsv"
)

DISPERSION_FILE = (
    PROJECT_ROOT
    / "parameters/HGSOC/nonmalignant/expression/"
    "tme_dispersion.yaml"
)


RUNTIME_TO_PARAMETER_TYPE = {
    "T_cell": "T_NK",
    "myeloid_macrophage": "macrophage_myeloid",
    "fibroblast": "fibroblast",
    "B_plasma": "B_plasma",
    "endothelial": "endothelial",
    "normal_epithelial": "normal_epithelial",
}


LIBRARY_STAGE = 6001
COUNT_STAGE = 6002

DEFAULT_CHUNK_SIZE = 250


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)

    return {
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def sample_nb2(
    rng: np.random.Generator,
    mu: np.ndarray,
    phi: float,
) -> np.ndarray:

    mu = np.asarray(
        mu,
        dtype=np.float64,
    )

    if phi <= 1e-12:
        return rng.poisson(
            mu
        ).astype(np.int64)

    n = 1.0 / phi

    p = (
        n
        / (
            n + mu
        )
    )

    return rng.negative_binomial(
        n=n,
        p=p,
    ).astype(np.int64)


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )

    parser.add_argument(
        "--preflight-cells",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )

    args = parser.parse_args()

    run_dir = args.run_dir.resolve()

    cells_file = (
        run_dir
        / "cells/cells_with_cell_types.tsv"
    )

    output_dir = (
        run_dir
        / "expression/nonmalignant"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunk_dir = (
        output_dir
        / "count_chunks"
    )

    if not args.preflight_only:
        chunk_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    #
    # ------------------------------------------------------------
    # Load runtime cells.
    # ------------------------------------------------------------
    #

    cells = pd.read_csv(
        cells_file,
        sep="\t",
    )

    required_cell_columns = {
        "cell_id",
        "is_malignant",
        "cell_type",
    }

    missing = (
        required_cell_columns
        - set(cells.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing cell columns: "
            + ", ".join(sorted(missing))
        )

    cells = cells.loc[
        ~cells["is_malignant"].astype(bool)
    ].copy()

    # Validate against the resolved run configuration rather than
    # assuming a fixed nonmalignant population size.
    import yaml

    resolved_path = (
        run_dir
        / "resolved_config.yaml"
    )

    if not resolved_path.exists():
        raise RuntimeError(
            f"Missing resolved configuration: {resolved_path}"
        )

    with resolved_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        resolved_config = yaml.safe_load(
            handle
        )


    # Cancer-specific reference paths, with HGSOC defaults.
    normal_config = (
        resolved_config.get("expression", {}).get("normal", {})
    )

    def reference_path(key, default):
        value = normal_config.get(key)
        path = PROJECT_ROOT / value if value else default
        if not path.is_file():
            raise RuntimeError(f"Missing expression reference: {path}")
        return path

    baseline_file = reference_path("baseline_file", BASELINE_FILE)
    library_file = reference_path("library_size_file", LIBRARY_FILE)
    dispersion_file = reference_path("dispersion_file", DISPERSION_FILE)

    expected_nonmalignant = int(
        resolved_config[
            "tissue"
        ][
            "n_nonmalignant"
        ]
    )

    master_seed = int(
        resolved_config[
            "simulation"
        ][
            "seed"
        ]
    )

    if len(cells) != expected_nonmalignant:
        raise RuntimeError(
            "Nonmalignant cell-count mismatch: "
            f"resolved configuration expects "
            f"{expected_nonmalignant:,}, "
            f"but cells_with_cell_types.tsv contains "
            f"{len(cells):,} nonmalignant cells."
        )

    unknown_types = (
        set(cells["cell_type"].unique())
        - set(RUNTIME_TO_PARAMETER_TYPE)
    )

    if unknown_types:
        raise RuntimeError(
            "Unmapped runtime cell types: "
            + ", ".join(sorted(unknown_types))
        )

    cells[
        "parameter_cell_type"
    ] = (
        cells["cell_type"]
        .map(RUNTIME_TO_PARAMETER_TYPE)
    )

    if args.preflight_only:

        n = min(
            args.preflight_cells,
            len(cells),
        )

        #
        # Preserve deterministic original order.
        #
        cells = (
            cells.iloc[:n]
            .copy()
        )

    #
    # ------------------------------------------------------------
    # Load expression baselines.
    # ------------------------------------------------------------
    #

    baseline = pd.read_csv(
        baseline_file,
        sep="\t",
    )

    gene_table = (
        baseline[
            [
                "gene_id",
                "gene_symbol",
            ]
        ]
        .drop_duplicates(
            subset=["gene_id"]
        )
        .reset_index(drop=True)
    )

    gene_ids = (
        gene_table[
            "gene_id"
        ]
        .astype(str)
        .to_numpy()
    )

    gene_symbols = (
        gene_table[
            "gene_symbol"
        ]
        .astype(str)
        .to_numpy()
    )

    n_genes = len(gene_ids)

    expected_rows = (
        n_genes
        * len(
            RUNTIME_TO_PARAMETER_TYPE.values()
        )
    )

    if len(baseline) != expected_rows:
        raise RuntimeError(
            f"Unexpected baseline table size: "
            f"{len(baseline):,}; "
            f"expected {expected_rows:,}"
        )

    baseline_vectors = {}

    for parameter_type in sorted(
        set(
            RUNTIME_TO_PARAMETER_TYPE.values()
        )
    ):

        x = (
            baseline.loc[
                baseline[
                    "broad_cell_type"
                ] == parameter_type
            ]
            .set_index("gene_id")
            .loc[
                gene_ids
            ]
        )

        p = (
            x[
                "baseline_fraction"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        if not np.isclose(
            p.sum(),
            1.0,
            atol=1e-8,
        ):
            raise RuntimeError(
                f"{parameter_type} baseline "
                "does not sum to 1."
            )

        if (
            p < 0
        ).any():
            raise RuntimeError(
                f"Negative baseline fraction "
                f"for {parameter_type}."
            )

        baseline_vectors[
            parameter_type
        ] = p

    #
    # ------------------------------------------------------------
    # Load empirical library-size reference.
    # ------------------------------------------------------------
    #

    library_reference = pd.read_csv(
        library_file,
        sep="\t",
    )

    library_vectors = {}

    for parameter_type in sorted(
        set(
            RUNTIME_TO_PARAMETER_TYPE.values()
        )
    ):

        x = (
            library_reference.loc[
                library_reference[
                    "broad_cell_type"
                ] == parameter_type,
                "library_size",
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        if len(x) == 0:
            raise RuntimeError(
                f"No library sizes for "
                f"{parameter_type}."
            )

        if (
            x <= 0
        ).any():
            raise RuntimeError(
                f"Non-positive library size "
                f"for {parameter_type}."
            )

        library_vectors[
            parameter_type
        ] = x

    #
    # ------------------------------------------------------------
    # Load dispersion.
    # ------------------------------------------------------------
    #

    with dispersion_file.open(
        "r",
        encoding="utf-8",
    ) as handle:

        dispersion_yaml = yaml.safe_load(
            handle
        )

    dispersion = {}

    for parameter_type in sorted(
        set(
            RUNTIME_TO_PARAMETER_TYPE.values()
        )
    ):

        phi = float(
            dispersion_yaml[
                "parameters"
            ][
                parameter_type
            ][
                "dispersion"
            ]
        )

        if (
            not np.isfinite(phi)
            or phi < 0
        ):
            raise RuntimeError(
                f"Invalid dispersion "
                f"for {parameter_type}: {phi}"
            )

        dispersion[
            parameter_type
        ] = phi

    #
    # ------------------------------------------------------------
    # Deterministic RNG.
    # ------------------------------------------------------------
    #

    library_rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                master_seed,
                LIBRARY_STAGE,
            ]
        )
    )

    count_rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                master_seed,
                COUNT_STAGE,
            ]
        )
    )

    #
    # ------------------------------------------------------------
    # Sample empirical library sizes within cell type.
    # ------------------------------------------------------------
    #

    sampled_libraries = np.zeros(
        len(cells),
        dtype=np.int64,
    )

    sampled_reference_ids = np.empty(
        len(cells),
        dtype=object,
    )

    for parameter_type in sorted(
        set(
            cells[
                "parameter_cell_type"
            ]
        )
    ):

        idx = np.flatnonzero(
            cells[
                "parameter_cell_type"
            ].to_numpy()
            == parameter_type
        )

        ref = (
            library_reference.loc[
                library_reference[
                    "broad_cell_type"
                ] == parameter_type
            ]
            .reset_index(drop=True)
        )

        draws = library_rng.integers(
            low=0,
            high=len(ref),
            size=len(idx),
        )

        sampled_libraries[
            idx
        ] = (
            ref.loc[
                draws,
                "library_size",
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        sampled_reference_ids[
            idx
        ] = (
            ref.loc[
                draws,
                "reference_cell_id",
            ]
            .astype(str)
            .to_numpy()
        )

    cells[
        "library_size"
    ] = sampled_libraries

    cells[
        "library_reference_cell_id"
    ] = sampled_reference_ids

    #
    # ------------------------------------------------------------
    # Generate counts.
    # ------------------------------------------------------------
    #

    total_counts_per_cell = np.zeros(
        len(cells),
        dtype=np.int64,
    )

    max_gene_count_per_cell = np.zeros(
        len(cells),
        dtype=np.int64,
    )

    max_mu_global = 0.0
    max_count_global = 0

    total_nonzero_entries = 0
    total_counts_global = 0

    type_stats = {}

    chunk_manifest = []

    n_chunks = int(
        np.ceil(
            len(cells)
            / args.chunk_size
        )
    )

    for chunk_index in range(
        n_chunks
    ):

        start = (
            chunk_index
            * args.chunk_size
        )

        end = min(
            start
            + args.chunk_size,
            len(cells),
        )

        chunk_cells = (
            cells.iloc[
                start:end
            ]
        )

        rows = []

        for local_i, (_, row) in enumerate(
            chunk_cells.iterrows()
        ):

            parameter_type = (
                row[
                    "parameter_cell_type"
                ]
            )

            L = float(
                row[
                    "library_size"
                ]
            )

            p = (
                baseline_vectors[
                    parameter_type
                ]
            )

            phi = (
                dispersion[
                    parameter_type
                ]
            )

            mu = (
                L
                * p
            )

            counts = sample_nb2(
                count_rng,
                mu,
                phi,
            )

            max_mu_global = max(
                max_mu_global,
                float(
                    np.max(mu)
                ),
            )

            max_count_global = max(
                max_count_global,
                int(
                    np.max(counts)
                ),
            )

            cell_total = int(
                counts.sum()
            )

            cell_max = int(
                counts.max()
            )

            global_i = (
                start
                + local_i
            )

            total_counts_per_cell[
                global_i
            ] = cell_total

            max_gene_count_per_cell[
                global_i
            ] = cell_max

            total_counts_global += (
                cell_total
            )

            total_nonzero_entries += int(
                np.count_nonzero(
                    counts
                )
            )

            rows.append(
                sp.csr_matrix(
                    counts.reshape(
                        1,
                        -1,
                    )
                )
            )

        chunk_matrix = sp.vstack(
            rows,
            format="csr",
        )

        if not args.preflight_only:

            chunk_name = (
                f"counts_{chunk_index:04d}.npz"
            )

            chunk_path = (
                chunk_dir
                / chunk_name
            )

            sp.save_npz(
                chunk_path,
                chunk_matrix,
            )

            chunk_manifest.append(
                {
                    "chunk_index":
                        chunk_index,

                    "filename":
                        chunk_name,

                    "start_row":
                        start,

                    "end_row_exclusive":
                        end,

                    "n_cells":
                        end - start,

                    "nnz":
                        int(
                            chunk_matrix.nnz
                        ),
                }
            )

    #
    # ------------------------------------------------------------
    # Diagnostics.
    # ------------------------------------------------------------
    #

    cells[
        "simulated_total_counts"
    ] = total_counts_per_cell

    cells[
        "count_library_ratio"
    ] = (
        total_counts_per_cell
        / sampled_libraries
    )

    cells[
        "max_gene_count"
    ] = (
        max_gene_count_per_cell
    )

    summary_by_type = {}

    for runtime_type, parameter_type in (
        RUNTIME_TO_PARAMETER_TYPE.items()
    ):

        mask = (
            cells[
                "cell_type"
            ]
            .eq(runtime_type)
            .to_numpy()
        )

        if not mask.any():
            continue

        L = (
            sampled_libraries[
                mask
            ]
        )

        totals = (
            total_counts_per_cell[
                mask
            ]
        )

        ratios = (
            totals
            / L
        )

        max_gene = (
            max_gene_count_per_cell[
                mask
            ]
        )

        summary_by_type[
            runtime_type
        ] = {
            "parameter_cell_type":
                parameter_type,

            "n_cells":
                int(mask.sum()),

            "dispersion":
                float(
                    dispersion[
                        parameter_type
                    ]
                ),

            "library_size":
                summarize(L),

            "simulated_total_counts":
                summarize(totals),

            "count_library_ratio":
                summarize(ratios),

            "max_gene_count":
                summarize(max_gene),
        }

    summary = {
        "model":
            "cell_type_baseline_NB2",

        "n_cells":
            int(len(cells)),

        "n_genes":
            int(n_genes),

        "preflight_only":
            bool(
                args.preflight_only
            ),

        "baseline_source":
            (
                "GSE184880 equal-donor "
                "cell-type pseudobulk"
            ),

        "library_size_source":
            (
                "GSE184880 empirical "
                "within-cell-type sampling"
            ),

        "variance":
            "Var(Y)=mu+phi_t*mu^2",

        "runtime_cell_type_mapping":
            RUNTIME_TO_PARAMETER_TYPE,

        "total_counts":
            int(
                total_counts_global
            ),

        "total_nonzero_entries":
            int(
                total_nonzero_entries
            ),

        "max_mu":
            float(
                max_mu_global
            ),

        "max_gene_count":
            int(
                max_count_global
            ),

        "overall_library_size":
            summarize(
                sampled_libraries
            ),

        "overall_simulated_total_counts":
            summarize(
                total_counts_per_cell
            ),

        "overall_count_library_ratio":
            summarize(
                cells[
                    "count_library_ratio"
                ].to_numpy()
            ),

        "by_cell_type":
            summary_by_type,

        "rng": {
            "master_seed":
                master_seed,

            "library_stage":
                LIBRARY_STAGE,

            "count_stage":
                COUNT_STAGE,
        },
    }

    #
    # ------------------------------------------------------------
    # Outputs.
    # ------------------------------------------------------------
    #

    if args.preflight_only:

        summary_path = (
            output_dir
            / "preflight_summary.yaml"
        )

    else:

        summary_path = (
            output_dir
            / "summary.yaml"
        )

        cells[
            [
                "cell_id",
                "cell_type",
                "parameter_cell_type",
                "library_size",
                "library_reference_cell_id",
                "simulated_total_counts",
                "count_library_ratio",
                "max_gene_count",
            ]
        ].to_csv(
            output_dir
            / "cells.tsv",
            sep="\t",
            index=False,
        )

        pd.DataFrame(
            {
                "gene_index":
                    np.arange(
                        n_genes
                    ),

                "gene_id":
                    gene_ids,

                "gene_symbol":
                    gene_symbols,
            }
        ).to_csv(
            output_dir
            / "genes.tsv",
            sep="\t",
            index=False,
        )

        pd.DataFrame(
            chunk_manifest
        ).to_csv(
            output_dir
            / "count_chunk_manifest.tsv",
            sep="\t",
            index=False,
        )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        yaml.safe_dump(
            summary,
            handle,
            sort_keys=False,
        )

    print()
    print("=" * 80)

    if args.preflight_only:
        print(
            "NONMALIGNANT EXPRESSION PREFLIGHT COMPLETE"
        )
    else:
        print(
            "NONMALIGNANT EXPRESSION GENERATION COMPLETE"
        )

    print("=" * 80)

    print(
        "Cells:",
        f"{len(cells):,}",
    )

    print(
        "Genes:",
        f"{n_genes:,}",
    )

    print(
        "Total counts:",
        f"{total_counts_global:,}",
    )

    print(
        "NNZ:",
        f"{total_nonzero_entries:,}",
    )

    print(
        "Max mu:",
        max_mu_global,
    )

    print(
        "Max observed gene count:",
        max_count_global,
    )

    print()

    print(
        "Overall count/library ratio:"
    )

    print(
        summary[
            "overall_count_library_ratio"
        ]
    )

    print()

    print(
        "Summary:",
        summary_path,
    )

    print()


if __name__ == "__main__":
    main()
