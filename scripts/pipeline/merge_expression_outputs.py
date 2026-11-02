#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml


DEFAULT_CHUNK_SIZE = 250


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge malignant and nonmalignant latent expression "
            "into one sparse 50k-cell expression object."
        )
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )

    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
    )


def normalize_manifest(
    manifest: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """
    Normalize existing malignant/nonmalignant manifests.

    Current generators use:
        start_row
        end_row_exclusive

    Older/proposed variants may use:
        cell_start
        cell_end_exclusive
    """

    x = manifest.copy()

    aliases = {
        "cell_start": "start_row",
        "cell_end_exclusive": "end_row_exclusive",
    }

    for old, new in aliases.items():
        if new not in x.columns and old in x.columns:
            x[new] = x[old]

    required = {
        "chunk_index",
        "filename",
        "start_row",
        "end_row_exclusive",
        "n_cells",
    }

    missing = required - set(x.columns)

    if missing:
        raise RuntimeError(
            f"{source_name} manifest missing columns: "
            + ", ".join(sorted(missing))
        )

    x = (
        x.sort_values(
            "chunk_index"
        )
        .reset_index(drop=True)
    )

    #
    # Validate contiguous row coverage.
    #
    expected_start = 0

    for _, row in x.iterrows():

        start = int(
            row["start_row"]
        )

        end = int(
            row["end_row_exclusive"]
        )

        n = int(
            row["n_cells"]
        )

        if start != expected_start:
            raise RuntimeError(
                f"{source_name} manifest is not contiguous: "
                f"expected start {expected_start}, found {start}."
            )

        if end - start != n:
            raise RuntimeError(
                f"{source_name} manifest row-count mismatch "
                f"for chunk {row['chunk_index']}."
            )

        expected_start = end

    return x


def load_source(
    run_dir: Path,
    source_name: str,
) -> dict:
    base = (
        run_dir
        / "expression"
        / source_name
    )

    cells = read_table(
        base / "cells.tsv"
    )

    genes = read_table(
        base / "genes.tsv"
    )

    manifest = normalize_manifest(
        read_table(
            base / "count_chunk_manifest.tsv"
        ),
        source_name,
    )

    chunk_dir = (
        base / "count_chunks"
    )

    if "cell_id" not in cells.columns:
        raise RuntimeError(
            f"{source_name} cells.tsv lacks cell_id."
        )

    if "gene_id" not in genes.columns:
        raise RuntimeError(
            f"{source_name} genes.tsv lacks gene_id."
        )

    if cells["cell_id"].duplicated().any():
        raise RuntimeError(
            f"{source_name} cells contain duplicate cell IDs."
        )

    if genes["gene_id"].duplicated().any():
        raise RuntimeError(
            f"{source_name} genes contain duplicate gene IDs."
        )

    if int(
        manifest["n_cells"].sum()
    ) != len(cells):

        raise RuntimeError(
            f"{source_name}: manifest covers "
            f"{int(manifest['n_cells'].sum())} cells "
            f"but cells.tsv contains {len(cells)}."
        )

    for filename in manifest["filename"]:

        path = (
            chunk_dir
            / str(filename)
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing chunk: {path}"
            )

    cells = cells.copy()

    cells[
        "source_row_index"
    ] = np.arange(
        len(cells),
        dtype=np.int64,
    )

    cells[
        "source_compartment"
    ] = source_name

    return {
        "name": source_name,
        "base": base,
        "cells": cells,
        "genes": genes,
        "manifest": manifest,
        "chunk_dir": chunk_dir,
    }


def build_union_genes(
    malignant_genes: pd.DataFrame,
    nonmalignant_genes: pd.DataFrame,
) -> pd.DataFrame:

    #
    # Preserve malignant gene order first.
    # Add nonmalignant-only genes afterward.
    #
    malignant = malignant_genes.copy()

    nonmalignant = (
        nonmalignant_genes.copy()
    )

    mal_ids = set(
        malignant[
            "gene_id"
        ].astype(str)
    )

    extras = nonmalignant.loc[
        ~nonmalignant[
            "gene_id"
        ].astype(str).isin(
            mal_ids
        )
    ].copy()

    columns = [
        "gene_id",
        "gene_symbol",
    ]

    for frame in [
        malignant,
        extras,
    ]:
        for column in columns:
            if column not in frame.columns:
                frame[column] = pd.NA

    union = pd.concat(
        [
            malignant[columns],
            extras[columns],
        ],
        ignore_index=True,
    )

    if union[
        "gene_id"
    ].duplicated().any():

        raise RuntimeError(
            "Union gene table contains duplicate gene IDs."
        )

    union.insert(
        0,
        "gene_index",
        np.arange(
            len(union),
            dtype=np.int64,
        ),
    )

    return union


def build_gene_projection(
    source_genes: pd.DataFrame,
    union_genes: pd.DataFrame,
) -> np.ndarray:

    target = pd.Series(
        union_genes[
            "gene_index"
        ].to_numpy(
            dtype=np.int64
        ),
        index=union_genes[
            "gene_id"
        ].astype(str),
    )

    source_ids = (
        source_genes[
            "gene_id"
        ]
        .astype(str)
        .to_numpy()
    )

    projection = (
        target
        .loc[
            source_ids
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    return projection


def project_matrix(
    matrix: sp.csr_matrix,
    projection: np.ndarray,
    n_union_genes: int,
) -> sp.csr_matrix:

    matrix = matrix.tocoo()

    new_columns = (
        projection[
            matrix.col
        ]
    )

    projected = sp.csr_matrix(
        (
            matrix.data,
            (
                matrix.row,
                new_columns,
            ),
        ),
        shape=(
            matrix.shape[0],
            n_union_genes,
        ),
    )

    return projected


def find_chunk_for_row(
    manifest: pd.DataFrame,
    source_row: int,
) -> pd.Series:

    matches = manifest.loc[
        (
            manifest[
                "start_row"
            ] <= source_row
        )
        &
        (
            manifest[
                "end_row_exclusive"
            ] > source_row
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Could not uniquely locate source row "
            f"{source_row} in manifest."
        )

    return matches.iloc[0]


def load_source_rows(
    source: dict,
    source_rows: np.ndarray,
    projection: np.ndarray,
    n_union_genes: int,
) -> sp.csr_matrix:

    if len(source_rows) == 0:
        return sp.csr_matrix(
            (
                0,
                n_union_genes,
            ),
            dtype=np.int64,
        )

    #
    # Group requested rows by source chunk.
    #
    chunk_requests = {}

    for output_position, source_row in enumerate(
        source_rows
    ):

        record = find_chunk_for_row(
            source[
                "manifest"
            ],
            int(source_row),
        )

        chunk_index = int(
            record[
                "chunk_index"
            ]
        )

        chunk_requests.setdefault(
            chunk_index,
            [],
        ).append(
            (
                output_position,
                int(source_row),
                record,
            )
        )

    result_rows = [
        None
        for _ in range(
            len(source_rows)
        )
    ]

    for chunk_index, requests in (
        chunk_requests.items()
    ):

        record = requests[
            0
        ][
            2
        ]

        chunk_path = (
            source[
                "chunk_dir"
            ]
            / str(
                record[
                    "filename"
                ]
            )
        )

        matrix = sp.load_npz(
            chunk_path
        ).tocsr()

        start = int(
            record[
                "start_row"
            ]
        )

        local_rows = np.asarray(
            [
                source_row
                - start
                for _, source_row, _
                in requests
            ],
            dtype=np.int64,
        )

        selected = (
            matrix[
                local_rows,
                :
            ]
            .tocsr()
        )

        selected = project_matrix(
            selected,
            projection,
            n_union_genes,
        )

        for local_i, (
            output_position,
            _,
            _,
        ) in enumerate(
            requests
        ):

            result_rows[
                output_position
            ] = selected[
                local_i,
                :
            ]

    if any(
        row is None
        for row in result_rows
    ):
        raise RuntimeError(
            "Failed to load all requested source rows."
        )

    return sp.vstack(
        result_rows,
        format="csr",
    )


def main() -> None:

    args = parse_args()

    run_dir = (
        args.run_dir
        .resolve()
    )

    if args.chunk_size <= 0:
        raise RuntimeError(
            "chunk-size must be positive."
        )

    output_dir = (
        run_dir
        / "expression"
        / "latent"
    )

    chunk_dir = (
        output_dir
        / "count_chunks"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunk_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("MERGING LATENT EXPRESSION")
    print("=" * 80)

    malignant = load_source(
        run_dir,
        "malignant",
    )

    nonmalignant = load_source(
        run_dir,
        "nonmalignant",
    )

    #
    # ------------------------------------------------------------
    # Canonical 50k cell order.
    # ------------------------------------------------------------
    #

    canonical_path = (
        run_dir
        / "cells"
        / "cells_with_cell_types.tsv"
    )

    canonical = read_table(
        canonical_path
    )

    if "cell_id" not in canonical.columns:
        raise RuntimeError(
            "Canonical cells file lacks cell_id."
        )

    if canonical[
        "cell_id"
    ].duplicated().any():

        raise RuntimeError(
            "Canonical cells file contains duplicate cell IDs."
        )

    #
    # Source lookup.
    #
    mal_lookup = (
        malignant[
            "cells"
        ]
        .set_index(
            "cell_id"
        )
    )

    non_lookup = (
        nonmalignant[
            "cells"
        ]
        .set_index(
            "cell_id"
        )
    )

    source_compartment = []
    source_row = []

    for cell_id in canonical[
        "cell_id"
    ].astype(str):

        in_mal = (
            cell_id
            in mal_lookup.index
        )

        in_non = (
            cell_id
            in non_lookup.index
        )

        if in_mal and in_non:
            raise RuntimeError(
                f"Cell appears in both branches: "
                f"{cell_id}"
            )

        if not in_mal and not in_non:
            raise RuntimeError(
                f"Canonical cell absent from both "
                f"expression branches: {cell_id}"
            )

        if in_mal:

            source_compartment.append(
                "malignant"
            )

            source_row.append(
                int(
                    mal_lookup.loc[
                        cell_id,
                        "source_row_index",
                    ]
                )
            )

        else:

            source_compartment.append(
                "nonmalignant"
            )

            source_row.append(
                int(
                    non_lookup.loc[
                        cell_id,
                        "source_row_index",
                    ]
                )
            )

    cells = canonical.copy()

    cells.insert(
        0,
        "matrix_row_index",
        np.arange(
            len(cells),
            dtype=np.int64,
        ),
    )

    cells[
        "expression_source"
    ] = source_compartment

    cells[
        "expression_source_row"
    ] = np.asarray(
        source_row,
        dtype=np.int64,
    )

    #
    # ------------------------------------------------------------
    # Gene union.
    # ------------------------------------------------------------
    #

    genes = build_union_genes(
        malignant[
            "genes"
        ],
        nonmalignant[
            "genes"
        ],
    )

    n_genes = len(
        genes
    )

    mal_projection = (
        build_gene_projection(
            malignant[
                "genes"
            ],
            genes,
        )
    )

    non_projection = (
        build_gene_projection(
            nonmalignant[
                "genes"
            ],
            genes,
        )
    )

    malignant_gene_ids = set(
        malignant[
            "genes"
        ][
            "gene_id"
        ].astype(str)
    )

    nonmalignant_gene_ids = set(
        nonmalignant[
            "genes"
        ][
            "gene_id"
        ].astype(str)
    )

    n_shared_genes = len(
        malignant_gene_ids
        &
        nonmalignant_gene_ids
    )

    n_malignant_only = len(
        malignant_gene_ids
        -
        nonmalignant_gene_ids
    )

    n_nonmalignant_only = len(
        nonmalignant_gene_ids
        -
        malignant_gene_ids
    )

    print(
        "Cells:",
        f"{len(cells):,}",
    )

    print(
        "Malignant genes:",
        f"{len(malignant['genes']):,}",
    )

    print(
        "Nonmalignant genes:",
        f"{len(nonmalignant['genes']):,}",
    )

    print(
        "Shared genes:",
        f"{n_shared_genes:,}",
    )

    print(
        "Malignant-only genes:",
        f"{n_malignant_only:,}",
    )

    print(
        "Nonmalignant-only genes:",
        f"{n_nonmalignant_only:,}",
    )

    print(
        "Union genes:",
        f"{n_genes:,}",
    )

    #
    # ------------------------------------------------------------
    # Write metadata before counts.
    # ------------------------------------------------------------
    #

    cells.to_csv(
        output_dir
        / "cells.tsv",
        sep="\t",
        index=False,
    )

    genes.to_csv(
        output_dir
        / "genes.tsv",
        sep="\t",
        index=False,
    )

    #
    # ------------------------------------------------------------
    # Generate merged sparse chunks.
    # ------------------------------------------------------------
    #

    manifest_rows = []

    n_cells = len(
        cells
    )

    n_chunks = int(
        np.ceil(
            n_cells
            / args.chunk_size
        )
    )

    print()
    print(
        "Writing",
        n_chunks,
        "merged chunks...",
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
            n_cells,
        )

        block = (
            cells.iloc[
                start:end
            ]
            .copy()
        )

        result_rows = [
            None
            for _ in range(
                len(block)
            )
        ]

        #
        # Malignant cells in this output block.
        #
        mal_mask = (
            block[
                "expression_source"
            ]
            .eq(
                "malignant"
            )
            .to_numpy()
        )

        if mal_mask.any():

            positions = np.flatnonzero(
                mal_mask
            )

            rows = (
                block.loc[
                    mal_mask,
                    "expression_source_row",
                ]
                .to_numpy(
                    dtype=np.int64
                )
            )

            matrix = load_source_rows(
                malignant,
                rows,
                mal_projection,
                n_genes,
            )

            for i, position in enumerate(
                positions
            ):
                result_rows[
                    position
                ] = matrix[
                    i,
                    :
                ]

        #
        # Nonmalignant cells in this output block.
        #
        non_mask = (
            block[
                "expression_source"
            ]
            .eq(
                "nonmalignant"
            )
            .to_numpy()
        )

        if non_mask.any():

            positions = np.flatnonzero(
                non_mask
            )

            rows = (
                block.loc[
                    non_mask,
                    "expression_source_row",
                ]
                .to_numpy(
                    dtype=np.int64
                )
            )

            matrix = load_source_rows(
                nonmalignant,
                rows,
                non_projection,
                n_genes,
            )

            for i, position in enumerate(
                positions
            ):
                result_rows[
                    position
                ] = matrix[
                    i,
                    :
                ]

        if any(
            x is None
            for x in result_rows
        ):
            raise RuntimeError(
                f"Chunk {chunk_index}: "
                "some output rows were not populated."
            )

        merged = sp.vstack(
            result_rows,
            format="csr",
        )

        filename = (
            f"counts_{chunk_index:04d}.npz"
        )

        sp.save_npz(
            chunk_dir
            / filename,
            merged,
        )

        manifest_rows.append(
            {
                "chunk_index":
                    chunk_index,

                "filename":
                    filename,

                "start_row":
                    start,

                "end_row_exclusive":
                    end,

                "n_cells":
                    end - start,

                "n_genes":
                    n_genes,

                "nnz":
                    int(
                        merged.nnz
                    ),

                "total_counts":
                    int(
                        merged.sum()
                    ),
            }
        )

        if (
            chunk_index == 0
            or
            (chunk_index + 1) % 20 == 0
            or
            chunk_index + 1 == n_chunks
        ):
            print(
                f"  chunk "
                f"{chunk_index + 1:>3}/"
                f"{n_chunks}"
            )

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest.to_csv(
        output_dir
        / "count_chunk_manifest.tsv",
        sep="\t",
        index=False,
    )

    #
    # ------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------
    #

    source_counts = (
        cells[
            "expression_source"
        ]
        .value_counts()
        .to_dict()
    )

    summary = {
        "stage":
            "latent_expression_merge",

        "gene_policy":
            "union",

        "canonical_cell_order":
            (
                "cells/cells_with_cell_types.tsv"
            ),

        "n_cells":
            int(n_cells),

        "n_malignant_cells":
            int(
                source_counts.get(
                    "malignant",
                    0,
                )
            ),

        "n_nonmalignant_cells":
            int(
                source_counts.get(
                    "nonmalignant",
                    0,
                )
            ),

        "malignant_input_genes":
            int(
                len(
                    malignant[
                        "genes"
                    ]
                )
            ),

        "nonmalignant_input_genes":
            int(
                len(
                    nonmalignant[
                        "genes"
                    ]
                )
            ),

        "shared_genes":
            int(
                n_shared_genes
            ),

        "malignant_only_genes":
            int(
                n_malignant_only
            ),

        "nonmalignant_only_genes":
            int(
                n_nonmalignant_only
            ),

        "union_genes":
            int(
                n_genes
            ),

        "missing_branch_gene_policy":
            (
                "zero_count_for_genes_not_simulated_"
                "in_that_expression_branch"
            ),

        "chunk_size":
            int(
                args.chunk_size
            ),

        "n_chunks":
            int(
                n_chunks
            ),

        "total_counts":
            int(
                manifest[
                    "total_counts"
                ].sum()
            ),

        "total_nonzero_entries":
            int(
                manifest[
                    "nnz"
                ].sum()
            ),

        "outputs": {
            "cells":
                (
                    "expression/latent/"
                    "cells.tsv"
                ),

            "genes":
                (
                    "expression/latent/"
                    "genes.tsv"
                ),

            "manifest":
                (
                    "expression/latent/"
                    "count_chunk_manifest.tsv"
                ),

            "chunks":
                (
                    "expression/latent/"
                    "count_chunks"
                ),
        },
    }

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

    print()
    print("=" * 80)
    print("LATENT EXPRESSION MERGE COMPLETE")
    print("=" * 80)

    print(
        "Cells:",
        f"{n_cells:,}",
    )

    print(
        "  malignant:",
        f"{summary['n_malignant_cells']:,}",
    )

    print(
        "  nonmalignant:",
        f"{summary['n_nonmalignant_cells']:,}",
    )

    print(
        "Union genes:",
        f"{n_genes:,}",
    )

    print(
        "Chunks:",
        n_chunks,
    )

    print(
        "Total counts:",
        f"{summary['total_counts']:,}",
    )

    print()

    print(
        "Output:",
        output_dir,
    )


if __name__ == "__main__":
    main()
