#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.io import mmwrite


VISIUM_RNG_STREAM = 7001


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate latent cell-level expression into "
            "Visium-like spots and generate depth-adjusted "
            "spot-level UMI counts."
        )
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        help="Completed simulator run directory.",
    )

    parser.add_argument(
        "--depth-reference-dir",
        required=True,
        help=(
            "Directory containing depth_histogram.tsv "
            "and metadata.yaml."
        ),
    )

    parser.add_argument(
        "--depth-multiplier",
        type=float,
        default=1.0,
        help="Visium UMI-depth multiplier.",
    )

    return parser.parse_args()


def load_master_seed(run_dir: Path) -> int:
    config_path = run_dir / "resolved_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    config = yaml.safe_load(
        config_path.read_text()
    )

    return int(
        config["simulation"]["seed"]
    )


def sample_spot_depths(
    histogram: pd.DataFrame,
    n_spots: int,
    depth_multiplier: float,
    rng: np.random.Generator,
):
    required = {
        "bin_index",
        "lower_umi",
        "upper_umi",
        "probability",
    }

    missing = required - set(histogram.columns)

    if missing:
        raise ValueError(
            "Depth histogram missing columns: "
            + ", ".join(sorted(missing))
        )

    probs = histogram[
        "probability"
    ].to_numpy(dtype=float)

    if np.any(probs < 0):
        raise ValueError(
            "Negative depth probabilities detected."
        )

    probs = probs / probs.sum()

    selected_rows = rng.choice(
        len(histogram),
        size=n_spots,
        replace=True,
        p=probs,
    )

    baseline = np.empty(
        n_spots,
        dtype=np.int64,
    )

    sampled_bins = np.empty(
        n_spots,
        dtype=np.int64,
    )

    for i, row_index in enumerate(selected_rows):
        row = histogram.iloc[row_index]

        lower = int(row["lower_umi"])
        upper = int(row["upper_umi"])

        if upper < lower:
            raise ValueError(
                f"Invalid UMI bin: {lower} > {upper}"
            )

        sampled_bins[i] = int(
            row["bin_index"]
        )

        # integers() uses an exclusive upper bound.
        baseline[i] = rng.integers(
            lower,
            upper + 1,
        )

    final_depth = np.rint(
        baseline.astype(float)
        * depth_multiplier
    ).astype(np.int64)

    final_depth = np.maximum(
        final_depth,
        1,
    )

    return sampled_bins, baseline, final_depth


def build_spot_cell_membership(
    assignments: pd.DataFrame,
    n_spots: int,
    n_cells: int,
):
    spot_index = assignments[
        "spot_index"
    ].to_numpy(dtype=np.int64)

    cell_index = assignments[
        "matrix_row_index"
    ].to_numpy(dtype=np.int64)

    data = np.ones(
        len(assignments),
        dtype=np.int8,
    )

    membership = sparse.csr_matrix(
        (
            data,
            (
                spot_index,
                cell_index,
            ),
        ),
        shape=(
            n_spots,
            n_cells,
        ),
    )

    return membership


def aggregate_latent_expression(
    latent_dir: Path,
    membership: sparse.csr_matrix,
    manifest: pd.DataFrame,
    n_spots: int,
    n_genes: int,
):
    """
    Aggregate cell-level sparse chunks into a spot x gene matrix.

    The latent matrix is stored as row chunks. Rather than reconstruct
    the full 15000 x 24640 cell matrix, each chunk is aggregated
    independently using only the corresponding columns of the
    spot-to-cell membership matrix.
    """

    aggregate = sparse.csr_matrix(
        (
            n_spots,
            n_genes,
        ),
        dtype=np.int64,
    )

    chunk_dir = (
        latent_dir
        / "count_chunks"
    )

    total_input_counts = 0

    for row in manifest.itertuples(
        index=False
    ):
        filename = str(row.filename)

        start = int(row.start_row)
        end = int(row.end_row_exclusive)

        chunk_path = (
            chunk_dir
            / filename
        )

        if not chunk_path.exists():
            raise FileNotFoundError(
                chunk_path
            )

        latent_chunk = sparse.load_npz(
            chunk_path
        ).tocsr()

        expected_shape = (
            end - start,
            n_genes,
        )

        if latent_chunk.shape != expected_shape:
            raise ValueError(
                f"{filename}: expected shape "
                f"{expected_shape}, observed "
                f"{latent_chunk.shape}"
            )

        total_input_counts += int(
            latent_chunk.sum()
        )

        local_membership = (
            membership[
                :,
                start:end,
            ]
        )

        if local_membership.nnz == 0:
            continue

        spot_chunk = (
            local_membership
            @ latent_chunk
        ).tocsr()

        aggregate = (
            aggregate
            + spot_chunk
        ).tocsr()

    aggregate.eliminate_zeros()

    return aggregate, total_input_counts


def multinomial_downsample_rows(
    aggregate: sparse.csr_matrix,
    target_depths: np.ndarray,
    rng: np.random.Generator,
):
    """
    For each spot v:

        theta_gv = Z_gv / sum_g Z_gv

        X_.v ~ Multinomial(U_v, theta_.v)

    Only genes with nonzero latent aggregate are passed to the
    multinomial sampler.
    """

    aggregate = aggregate.tocsr()

    out_data = []
    out_indices = []
    out_indptr = [0]

    latent_totals = np.asarray(
        aggregate.sum(axis=1)
    ).ravel().astype(np.int64)

    observed_genes = np.zeros(
        aggregate.shape[0],
        dtype=np.int64,
    )

    for spot_index in range(
        aggregate.shape[0]
    ):
        start = aggregate.indptr[
            spot_index
        ]

        end = aggregate.indptr[
            spot_index + 1
        ]

        indices = aggregate.indices[
            start:end
        ]

        values = aggregate.data[
            start:end
        ].astype(float)

        total = float(values.sum())

        if total <= 0:
            raise RuntimeError(
                "Spot "
                f"{spot_index} has zero latent "
                "transcript count."
            )

        probabilities = (
            values
            / total
        )

        sampled = rng.multinomial(
            int(
                target_depths[
                    spot_index
                ]
            ),
            probabilities,
        )

        keep = sampled > 0

        kept_indices = indices[
            keep
        ]

        kept_values = sampled[
            keep
        ].astype(np.int64)

        observed_genes[
            spot_index
        ] = len(
            kept_values
        )

        out_indices.extend(
            kept_indices.tolist()
        )

        out_data.extend(
            kept_values.tolist()
        )

        out_indptr.append(
            len(out_data)
        )

    observed = sparse.csr_matrix(
        (
            np.asarray(
                out_data,
                dtype=np.int64,
            ),
            np.asarray(
                out_indices,
                dtype=np.int32,
            ),
            np.asarray(
                out_indptr,
                dtype=np.int64,
            ),
        ),
        shape=aggregate.shape,
    )

    return (
        observed,
        latent_totals,
        observed_genes,
    )


def write_10x_outputs(
    output_dir: Path,
    matrix_spot_gene: sparse.csr_matrix,
    genes: pd.DataFrame,
    spots: pd.DataFrame,
):
    tenx_dir = (
        output_dir
        / "filtered_feature_bc_matrix"
    )

    tenx_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 10x convention is features x barcodes.
    tenx_matrix = (
        matrix_spot_gene.T
        .tocoo()
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        mtx_tmp = (
            tmp / "matrix.mtx"
        )

        mmwrite(
            mtx_tmp,
            tenx_matrix,
            field="integer",
        )

        mtx_gz = (
            tenx_dir
            / "matrix.mtx.gz"
        )

        with open(
            mtx_tmp,
            "rb",
        ) as src, gzip.open(
            mtx_gz,
            "wb",
        ) as dst:
            shutil.copyfileobj(
                src,
                dst,
            )

    feature_path = (
        tenx_dir
        / "features.tsv.gz"
    )

    gene_ids = genes[
        "gene_id"
    ].astype(str)

    gene_symbols = genes[
        "gene_symbol"
    ].astype(str)

    feature_table = pd.DataFrame(
        {
            0: gene_ids,
            1: gene_symbols,
            2: "Gene Expression",
        }
    )

    feature_table.to_csv(
        feature_path,
        sep="\t",
        header=False,
        index=False,
        compression="gzip",
    )

    barcode_path = (
        tenx_dir
        / "barcodes.tsv.gz"
    )

    spots[
        ["spot_id"]
    ].to_csv(
        barcode_path,
        sep="\t",
        header=False,
        index=False,
        compression="gzip",
    )


def main():
    args = parse_args()

    if args.depth_multiplier <= 0:
        raise ValueError(
            "--depth-multiplier must be > 0."
        )

    run_dir = Path(
        args.run_dir
    ).resolve()

    depth_reference_dir = Path(
        args.depth_reference_dir
    ).resolve()

    latent_dir = (
        run_dir
        / "expression"
        / "latent"
    )

    observation_dir = (
        run_dir
        / "observation"
        / "visium"
    )

    expression_dir = (
        observation_dir
        / "expression"
    )

    plot_dir = (
        observation_dir
        / "plots"
    )

    expression_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cells_path = (
        latent_dir
        / "cells.tsv"
    )

    genes_path = (
        latent_dir
        / "genes.tsv"
    )

    manifest_path = (
        latent_dir
        / "count_chunk_manifest.tsv"
    )

    spots_path = (
        observation_dir
        / "spots.tsv"
    )

    assignments_path = (
        observation_dir
        / "cell_to_spot.tsv"
    )

    histogram_path = (
        depth_reference_dir
        / "depth_histogram.tsv"
    )

    reference_values_path = (
        depth_reference_dir
        / "reference_spot_umis.tsv.gz"
    )

    metadata_path = (
        depth_reference_dir
        / "metadata.yaml"
    )

    required_paths = [
        cells_path,
        genes_path,
        manifest_path,
        spots_path,
        assignments_path,
        histogram_path,
        metadata_path,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    print(
        "Loading Visium observation inputs..."
    )

    cells = pd.read_csv(
        cells_path,
        sep="\t",
    )

    genes = pd.read_csv(
        genes_path,
        sep="\t",
    )

    manifest = pd.read_csv(
        manifest_path,
        sep="\t",
    )

    spots = pd.read_csv(
        spots_path,
        sep="\t",
    )

    assignments = pd.read_csv(
        assignments_path,
        sep="\t",
    )

    histogram = pd.read_csv(
        histogram_path,
        sep="\t",
    )

    reference_metadata = yaml.safe_load(
        metadata_path.read_text()
    )

    n_cells = len(cells)
    n_genes = len(genes)
    n_spots = len(spots)

    if n_spots == 0:
        raise RuntimeError(
            "No Visium-like spots available."
        )

    if not np.array_equal(
        spots[
            "spot_index"
        ].to_numpy(),
        np.arange(n_spots),
    ):
        raise ValueError(
            "spot_index must be canonical 0..V-1."
        )

    if not np.array_equal(
        cells[
            "matrix_row_index"
        ].to_numpy(),
        np.arange(n_cells),
    ):
        raise ValueError(
            "Latent cells are not in canonical "
            "matrix row order."
        )

    if not np.array_equal(
        genes[
            "gene_index"
        ].to_numpy(),
        np.arange(n_genes),
    ):
        raise ValueError(
            "Genes are not in canonical "
            "gene-index order."
        )

    if assignments[
        "matrix_row_index"
    ].duplicated().any():
        raise ValueError(
            "A latent cell belongs to more than "
            "one Visium-like spot."
        )

    master_seed = load_master_seed(
        run_dir
    )

    seed_sequence = np.random.SeedSequence(
        [
            master_seed,
            VISIUM_RNG_STREAM,
        ]
    )

    rng = np.random.default_rng(
        seed_sequence
    )

    print()
    print(
        "Building sparse spot-to-cell membership..."
    )

    membership = (
        build_spot_cell_membership(
            assignments=assignments,
            n_spots=n_spots,
            n_cells=n_cells,
        )
    )

    print(
        "Captured cell assignments:",
        membership.nnz,
    )

    print()
    print(
        "Aggregating latent expression..."
    )

    (
        aggregate,
        total_latent_counts,
    ) = aggregate_latent_expression(
        latent_dir=latent_dir,
        membership=membership,
        manifest=manifest,
        n_spots=n_spots,
        n_genes=n_genes,
    )

    print(
        "Aggregated matrix shape:",
        aggregate.shape,
    )

    print(
        "Aggregated nnz:",
        aggregate.nnz,
    )

    print(
        "Aggregated latent counts:",
        int(aggregate.sum()),
    )

    print()
    print(
        "Sampling Visium-like spot depths..."
    )

    (
        sampled_bins,
        baseline_depths,
        final_depths,
    ) = sample_spot_depths(
        histogram=histogram,
        n_spots=n_spots,
        depth_multiplier=(
            args.depth_multiplier
        ),
        rng=rng,
    )

    print()
    print(
        "Generating multinomial "
        "spot-level UMI counts..."
    )

    (
        observed,
        latent_spot_totals,
        observed_genes,
    ) = multinomial_downsample_rows(
        aggregate=aggregate,
        target_depths=final_depths,
        rng=rng,
    )

    observed_totals = np.asarray(
        observed.sum(axis=1)
    ).ravel().astype(np.int64)

    if not np.array_equal(
        observed_totals,
        final_depths,
    ):
        raise RuntimeError(
            "Observed spot totals do not equal "
            "their sampled target UMI depths."
        )

    # --------------------------------------------------------------
    # Save sparse matrices
    # --------------------------------------------------------------

    sparse.save_npz(
        expression_dir
        / "latent_spot_aggregate.npz",
        aggregate,
        compressed=True,
    )

    sparse.save_npz(
        expression_dir
        / "counts.npz",
        observed,
        compressed=True,
    )

    # --------------------------------------------------------------
    # Spot-level observation metadata
    # --------------------------------------------------------------

    depth_table = spots[
        [
            "spot_index",
            "spot_id",
            "array_row",
            "array_col",
            "x",
            "y",
            "n_cells",
        ]
    ].copy()

    depth_table[
        "reference_bin"
    ] = sampled_bins

    depth_table[
        "baseline_target_umis"
    ] = baseline_depths

    depth_table[
        "depth_multiplier"
    ] = float(
        args.depth_multiplier
    )

    depth_table[
        "target_umis"
    ] = final_depths

    depth_table[
        "latent_aggregate_counts"
    ] = latent_spot_totals

    depth_table[
        "observed_umis"
    ] = observed_totals

    depth_table[
        "genes_detected"
    ] = observed_genes

    depth_table.to_csv(
        expression_dir
        / "spot_observation.tsv",
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------------
    # 10x-like expression outputs
    # --------------------------------------------------------------

    print()
    print(
        "Writing 10x-compatible count files..."
    )

    write_10x_outputs(
        output_dir=observation_dir,
        matrix_spot_gene=observed,
        genes=genes,
        spots=spots,
    )

    # Retain simulator-space coordinates explicitly.
    spots[
        [
            "spot_id",
            "array_row",
            "array_col",
            "x",
            "y",
        ]
    ].to_csv(
        observation_dir
        / "spatial_coordinates.tsv",
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------------
    # Validation plots
    # --------------------------------------------------------------

    if reference_values_path.exists():
        reference_values = pd.read_csv(
            reference_values_path,
            sep="\t",
        )[
            "total_umis"
        ].to_numpy(
            dtype=np.int64
        )

        reference_values = (
            reference_values
            * args.depth_multiplier
        )
    else:
        reference_values = None

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    if reference_values is not None:
        ax.hist(
            reference_values,
            bins=40,
            density=True,
            alpha=0.45,
            label="reference SP4",
        )

    ax.hist(
        observed_totals,
        bins=40,
        density=True,
        alpha=0.55,
        label="simulated Visium-like",
    )

    ax.set_xlabel(
        "Total UMIs per spot"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        "Visium-like spot-depth distribution"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        plot_dir
        / "reference_vs_simulated_umi_depth.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.hist(
        observed_genes,
        bins=35,
        edgecolor="black",
    )

    ax.set_xlabel(
        "Genes detected per spot"
    )

    ax.set_ylabel(
        "Number of spots"
    )

    ax.set_title(
        "Simulated Visium-like genes detected"
    )

    fig.tight_layout()

    fig.savefig(
        plot_dir
        / "genes_detected_per_spot.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    summary = {
        "stage": "visium_expression_observation",
        "model": (
            "cell_aggregation_empirical_depth_multinomial"
        ),
        "rng": {
            "master_seed": int(
                master_seed
            ),
            "stream_id": int(
                VISIUM_RNG_STREAM
            ),
        },
        "n_latent_cells": int(
            n_cells
        ),
        "n_genes": int(
            n_genes
        ),
        "n_spots": int(
            n_spots
        ),
        "n_captured_cells": int(
            membership.nnz
        ),
        "reference_dataset": (
            reference_metadata.get(
                "reference_dataset"
            )
        ),
        "reference_sample": (
            reference_metadata.get(
                "reference_sample"
            )
        ),
        "reference_median_umis": float(
            reference_metadata.get(
                "median_umis"
            )
        ),
        "depth_multiplier": float(
            args.depth_multiplier
        ),
        "simulated_mean_umis": float(
            observed_totals.mean()
        ),
        "simulated_median_umis": float(
            np.median(
                observed_totals
            )
        ),
        "simulated_q25_umis": float(
            np.quantile(
                observed_totals,
                0.25,
            )
        ),
        "simulated_q75_umis": float(
            np.quantile(
                observed_totals,
                0.75,
            )
        ),
        "simulated_q95_umis": float(
            np.quantile(
                observed_totals,
                0.95,
            )
        ),
        "median_genes_detected": float(
            np.median(
                observed_genes
            )
        ),
        "mean_genes_detected": float(
            observed_genes.mean()
        ),
        "total_observed_umis": int(
            observed.sum()
        ),
        "total_latent_counts_all_cells": int(
            total_latent_counts
        ),
        "total_latent_counts_captured_cells": int(
            aggregate.sum()
        ),
        "output_shape": [
            int(n_spots),
            int(n_genes),
        ],
        "outputs": {
            "counts_sparse": (
                "observation/visium/"
                "expression/counts.npz"
            ),
            "latent_spot_aggregate": (
                "observation/visium/"
                "expression/"
                "latent_spot_aggregate.npz"
            ),
            "spot_observation": (
                "observation/visium/"
                "expression/"
                "spot_observation.tsv"
            ),
            "matrix_market": (
                "observation/visium/"
                "filtered_feature_bc_matrix/"
                "matrix.mtx.gz"
            ),
            "features": (
                "observation/visium/"
                "filtered_feature_bc_matrix/"
                "features.tsv.gz"
            ),
            "barcodes": (
                "observation/visium/"
                "filtered_feature_bc_matrix/"
                "barcodes.tsv.gz"
            ),
            "coordinates": (
                "observation/visium/"
                "spatial_coordinates.tsv"
            ),
        },
    }

    (
        expression_dir
        / "summary.yaml"
    ).write_text(
        yaml.safe_dump(
            summary,
            sort_keys=False,
        )
    )

    print()
    print(
        "================================================="
    )

    print(
        "VISIUM-LIKE EXPRESSION COMPLETE"
    )

    print(
        "================================================="
    )

    print(
        "Spots:",
        n_spots,
    )

    print(
        "Genes:",
        n_genes,
    )

    print(
        "Median UMI depth:",
        np.median(
            observed_totals
        ),
    )

    print(
        "Mean UMI depth:",
        observed_totals.mean(),
    )

    print(
        "Median genes detected:",
        np.median(
            observed_genes
        ),
    )

    print(
        "Total observed UMIs:",
        int(
            observed.sum()
        ),
    )

    print()
    print(
        "Output:"
    )

    print(
        observation_dir
    )

    print()
    print("FINAL CHECK: PASS")


if __name__ == "__main__":
    main()
