from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


class CNAsimOutputAdapterError(RuntimeError):
    pass


@dataclass
class CNAsimOutputAdapterResult:
    clone_bin_cna: pd.DataFrame
    clone_cna_summary: pd.DataFrame
    cell_clone_mapping: pd.DataFrame
    summary: dict[str, Any]


def _read_table(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise CNAsimOutputAdapterError(
            f"Missing required file: {path}"
        )

    if path.stat().st_size == 0:
        raise CNAsimOutputAdapterError(
            f"Required file is empty: {path}"
        )

    return pd.read_csv(
        path,
        sep="\t",
    )


def _parse_cn_state(
    value: Any,
) -> tuple[int, int]:
    text = str(value).strip()

    parts = [
        part.strip()
        for part in text.split(",")
    ]

    if len(parts) != 2:
        raise CNAsimOutputAdapterError(
            f"Invalid CN state {value!r}; "
            "expected two allele copy numbers."
        )

    try:
        a = int(parts[0])
        b = int(parts[1])
    except ValueError as exc:
        raise CNAsimOutputAdapterError(
            f"Non-integer CN state: {value!r}"
        ) from exc

    if a < 0 or b < 0:
        raise CNAsimOutputAdapterError(
            f"Negative copy number found: {value!r}"
        )

    return a, b


def _prepare_profile_table(
    frame: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    required = {
        "CELL",
        "chrom",
        "start",
        "end",
        "CN states",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise CNAsimOutputAdapterError(
            f"{source_name} missing columns: "
            + ", ".join(sorted(missing))
        )

    frame = frame.copy()

    parsed = frame[
        "CN states"
    ].map(
        _parse_cn_state
    )

    frame[
        "allele_a_cn"
    ] = [
        value[0]
        for value in parsed
    ]

    frame[
        "allele_b_cn"
    ] = [
        value[1]
        for value in parsed
    ]

    frame[
        "total_cn"
    ] = (
        frame["allele_a_cn"]
        + frame["allele_b_cn"]
    )

    frame[
        "start"
    ] = pd.to_numeric(
        frame["start"],
        errors="raise",
    ).astype(np.int64)

    frame[
        "end"
    ] = pd.to_numeric(
        frame["end"],
        errors="raise",
    ).astype(np.int64)

    return frame


def build_canonical_cnasim_output(
    node_mapping: pd.DataFrame,
    leaf_profiles: pd.DataFrame,
    ancestral_profiles: pd.DataFrame,
    cells: pd.DataFrame,
) -> CNAsimOutputAdapterResult:
    required_mapping = {
        "subclone_id",
        "cnasim_node_id",
        "cnasim_tree_role",
        "biological_role",
        "malignant_cells",
    }

    missing = (
        required_mapping
        - set(node_mapping.columns)
    )

    if missing:
        raise CNAsimOutputAdapterError(
            "Node mapping missing columns: "
            + ", ".join(sorted(missing))
        )

    required_cells = {
        "cell_id",
        "is_malignant",
        "source_demon_subclone_id",
    }

    missing = (
        required_cells
        - set(cells.columns)
    )

    if missing:
        raise CNAsimOutputAdapterError(
            "Cell table missing columns: "
            + ", ".join(sorted(missing))
        )

    leaf_profiles = _prepare_profile_table(
        leaf_profiles,
        "profiles.tsv",
    )

    ancestral_profiles = _prepare_profile_table(
        ancestral_profiles,
        "ancestral_profiles.tsv",
    )

    # ------------------------------------------------------------
    # Validate which CNAsim nodes exist in each profile file.
    # ------------------------------------------------------------

    leaf_nodes = set(
        leaf_profiles[
            "CELL"
        ].astype(str)
    )

    ancestral_nodes = set(
        ancestral_profiles[
            "CELL"
        ].astype(str)
    )

    expected_leaf_nodes = set(
        node_mapping.loc[
            node_mapping[
                "cnasim_tree_role"
            ] == "leaf",
            "cnasim_node_id",
        ].astype(str)
    )

    expected_ancestral_nodes = set(
        node_mapping.loc[
            node_mapping[
                "cnasim_tree_role"
            ].isin(
                [
                    "root",
                    "internal",
                ]
            ),
            "cnasim_node_id",
        ].astype(str)
    )

    if leaf_nodes != expected_leaf_nodes:
        raise CNAsimOutputAdapterError(
            "Leaf CNAsim profile nodes do not exactly "
            "match bridge leaf nodes."
        )

    if ancestral_nodes != expected_ancestral_nodes:
        raise CNAsimOutputAdapterError(
            "Ancestral CNAsim profile nodes do not exactly "
            "match bridge root/internal nodes."
        )

    # ------------------------------------------------------------
    # Validate common genome-bin structure.
    # ------------------------------------------------------------

    combined = pd.concat(
        [
            leaf_profiles,
            ancestral_profiles,
        ],
        ignore_index=True,
    )

    bin_counts = (
        combined
        .groupby("CELL")
        .size()
    )

    if bin_counts.nunique() != 1:
        raise CNAsimOutputAdapterError(
            "CNAsim nodes do not all contain the same "
            "number of genomic bins."
        )

    n_bins = int(
        bin_counts.iloc[0]
    )

    reference_node = str(
        combined[
            "CELL"
        ].iloc[0]
    )

    reference_bins = (
        combined.loc[
            combined[
                "CELL"
            ].astype(str)
            == reference_node,
            [
                "chrom",
                "start",
                "end",
            ],
        ]
        .reset_index(drop=True)
    )

    for node_id, group in (
        combined.groupby(
            "CELL",
            sort=False,
        )
    ):
        bins = (
            group[
                [
                    "chrom",
                    "start",
                    "end",
                ]
            ]
            .reset_index(drop=True)
        )

        if not bins.equals(
            reference_bins
        ):
            raise CNAsimOutputAdapterError(
                f"Genome-bin coordinates differ for "
                f"CNAsim node {node_id!r}."
            )

    # ------------------------------------------------------------
    # Biological clone -> canonical CNA bins.
    # ------------------------------------------------------------

    mapping_lookup = (
        node_mapping[
            [
                "subclone_id",
                "cnasim_node_id",
                "cnasim_tree_role",
                "biological_role",
                "malignant_cells",
            ]
        ]
        .copy()
    )

    profile_blocks: list[
        pd.DataFrame
    ] = []

    for row in mapping_lookup.itertuples(
        index=False
    ):
        node_id = str(
            row.cnasim_node_id
        )

        if row.cnasim_tree_role == "leaf":
            source = leaf_profiles
        else:
            source = ancestral_profiles

        block = source.loc[
            source[
                "CELL"
            ].astype(str)
            == node_id,
            [
                "chrom",
                "start",
                "end",
                "allele_a_cn",
                "allele_b_cn",
                "total_cn",
            ],
        ].copy()

        if len(block) != n_bins:
            raise CNAsimOutputAdapterError(
                f"Expected {n_bins} bins for "
                f"{node_id}, found {len(block)}."
            )

        block.insert(
            0,
            "subclone_id",
            row.subclone_id,
        )

        block.insert(
            1,
            "cnasim_node_id",
            node_id,
        )

        block.insert(
            2,
            "cnasim_tree_role",
            row.cnasim_tree_role,
        )

        block.insert(
            3,
            "biological_role",
            row.biological_role,
        )

        block.insert(
            4,
            "malignant_cells",
            int(
                row.malignant_cells
            ),
        )

        profile_blocks.append(
            block
        )

    clone_bin_cna = pd.concat(
        profile_blocks,
        ignore_index=True,
    )

    expected_rows = (
        len(node_mapping)
        * n_bins
    )

    if len(clone_bin_cna) != expected_rows:
        raise CNAsimOutputAdapterError(
            "Canonical clone-bin CNA table has "
            "unexpected number of rows."
        )

    # ------------------------------------------------------------
    # Clone-level CNA burden summaries.
    # Diploid reference total CN = 2.
    # ------------------------------------------------------------

    clone_cna_summary = (
        clone_bin_cna
        .groupby(
            [
                "subclone_id",
                "cnasim_node_id",
                "cnasim_tree_role",
                "biological_role",
                "malignant_cells",
            ],
            as_index=False,
        )
        .agg(
            n_bins=(
                "total_cn",
                "size",
            ),
            altered_bins=(
                "total_cn",
                lambda x:
                    int(
                        (x != 2).sum()
                    ),
            ),
            deletion_bins=(
                "total_cn",
                lambda x:
                    int(
                        (x < 2).sum()
                    ),
            ),
            amplification_bins=(
                "total_cn",
                lambda x:
                    int(
                        (x > 2).sum()
                    ),
            ),
            homozygous_deletion_bins=(
                "total_cn",
                lambda x:
                    int(
                        (x == 0).sum()
                    ),
            ),
            mean_total_cn=(
                "total_cn",
                "mean",
            ),
            max_total_cn=(
                "total_cn",
                "max",
            ),
        )
    )

    clone_cna_summary[
        "altered_fraction"
    ] = (
        clone_cna_summary[
            "altered_bins"
        ]
        / clone_cna_summary[
            "n_bins"
        ]
    )

    clone_cna_summary[
        "deletion_fraction"
    ] = (
        clone_cna_summary[
            "deletion_bins"
        ]
        / clone_cna_summary[
            "n_bins"
        ]
    )

    clone_cna_summary[
        "amplification_fraction"
    ] = (
        clone_cna_summary[
            "amplification_bins"
        ]
        / clone_cna_summary[
            "n_bins"
        ]
    )

    clone_cna_summary[
        "homozygous_deletion_fraction"
    ] = (
        clone_cna_summary[
            "homozygous_deletion_bins"
        ]
        / clone_cna_summary[
            "n_bins"
        ]
    )

    clone_cna_summary = (
        clone_cna_summary
        .sort_values(
            "malignant_cells",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # Malignant cell -> biological clone -> CNAsim node.
    #
    # No huge cell x genome matrix is generated here.
    # ------------------------------------------------------------

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

    malignant_cells = (
        cells.loc[
            malignant_mask,
            [
                "cell_id",
                "source_demon_subclone_id",
            ],
        ]
        .rename(
            columns={
                "source_demon_subclone_id":
                    "subclone_id",
            }
        )
        .copy()
    )

    cell_clone_mapping = (
        malignant_cells
        .merge(
            node_mapping[
                [
                    "subclone_id",
                    "cnasim_node_id",
                ]
            ],
            on="subclone_id",
            how="left",
            validate="many_to_one",
        )
    )

    if (
        cell_clone_mapping[
            "cnasim_node_id"
        ].isna().any()
    ):
        raise CNAsimOutputAdapterError(
            "One or more malignant cells could not "
            "be mapped to a CNAsim profile."
        )

    # ------------------------------------------------------------
    # Population-weighted CNA burden.
    #
    # This computes what a randomly selected malignant cell sees
    # without materializing 40,000 x 2,877 rows.
    # ------------------------------------------------------------

    observed_clone_summary = (
        clone_cna_summary.loc[
            clone_cna_summary[
                "malignant_cells"
            ] > 0
        ].copy()
    )

    malignant_total = int(
        observed_clone_summary[
            "malignant_cells"
        ].sum()
    )

    if malignant_total != len(
        cell_clone_mapping
    ):
        raise CNAsimOutputAdapterError(
            "Malignant population is not conserved "
            "through CNA profile mapping."
        )

    weights = (
        observed_clone_summary[
            "malignant_cells"
        ].to_numpy(
            dtype=float
        )
        / malignant_total
    )

    weighted_altered_fraction = float(
        np.sum(
            weights
            * observed_clone_summary[
                "altered_fraction"
            ].to_numpy(
                dtype=float
            )
        )
    )

    weighted_deletion_fraction = float(
        np.sum(
            weights
            * observed_clone_summary[
                "deletion_fraction"
            ].to_numpy(
                dtype=float
            )
        )
    )

    weighted_amplification_fraction = float(
        np.sum(
            weights
            * observed_clone_summary[
                "amplification_fraction"
            ].to_numpy(
                dtype=float
            )
        )
    )

    weighted_homdel_fraction = float(
        np.sum(
            weights
            * observed_clone_summary[
                "homozygous_deletion_fraction"
            ].to_numpy(
                dtype=float
            )
        )
    )

    weighted_mean_total_cn = float(
        np.sum(
            weights
            * observed_clone_summary[
                "mean_total_cn"
            ].to_numpy(
                dtype=float
            )
        )
    )

    total_cn_counts = (
        clone_bin_cna.loc[
            clone_bin_cna[
                "malignant_cells"
            ] > 0
        ]
        .groupby(
            "total_cn"
        )[
            "malignant_cells"
        ]
        .sum()
        .sort_index()
    )

    weighted_total_cn_distribution = {
        int(cn):
            float(
                weighted_count
                / (
                    malignant_total
                    * n_bins
                )
            )
        for cn, weighted_count
        in total_cn_counts.items()
    }

    summary = {
        "source":
            "CNAsim",

        "n_cnasim_nodes":
            int(
                len(
                    node_mapping
                )
            ),

        "n_observed_malignant_clones":
            int(
                (
                    node_mapping[
                        "malignant_cells"
                    ]
                    > 0
                ).sum()
            ),

        "n_ancestor_only_nodes":
            int(
                (
                    node_mapping[
                        "malignant_cells"
                    ]
                    == 0
                ).sum()
            ),

        "n_genome_bins_per_node":
            n_bins,

        "canonical_clone_bin_rows":
            int(
                len(
                    clone_bin_cna
                )
            ),

        "malignant_cells":
            malignant_total,

        "malignant_cells_with_cna_profile":
            int(
                len(
                    cell_clone_mapping
                )
            ),

        "all_malignant_cells_mapped":
            bool(
                len(
                    cell_clone_mapping
                )
                == malignant_total
            ),

        "reference_total_cn":
            2,

        "population_weighted_cna_burden":
            {
                "altered_bin_fraction":
                    weighted_altered_fraction,

                "deletion_bin_fraction":
                    weighted_deletion_fraction,

                "amplification_bin_fraction":
                    weighted_amplification_fraction,

                "homozygous_deletion_bin_fraction":
                    weighted_homdel_fraction,

                "mean_total_cn":
                    weighted_mean_total_cn,
            },

        "population_weighted_total_cn_distribution":
            weighted_total_cn_distribution,

        "note":
            (
                "CNA profiles are stored once per biological "
                "malignant lineage node. Malignant cells reference "
                "their biological subclone rather than duplicating "
                "the full genome-wide CNA profile per cell."
            ),
    }

    return CNAsimOutputAdapterResult(
        clone_bin_cna=clone_bin_cna,
        clone_cna_summary=clone_cna_summary,
        cell_clone_mapping=cell_clone_mapping,
        summary=summary,
    )


def write_canonical_cnasim_output(
    cnasim_dir: Path,
    node_mapping_path: Path,
    cells_path: Path,
    output_dir: Path,
) -> CNAsimOutputAdapterResult:
    node_mapping = _read_table(
        node_mapping_path
    )

    leaf_profiles = _read_table(
        cnasim_dir
        / "profiles.tsv"
    )

    ancestral_profiles = _read_table(
        cnasim_dir
        / "ancestral_profiles.tsv"
    )

    cells = _read_table(
        cells_path
    )

    result = build_canonical_cnasim_output(
        node_mapping=node_mapping,
        leaf_profiles=leaf_profiles,
        ancestral_profiles=ancestral_profiles,
        cells=cells,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.clone_bin_cna.to_csv(
        output_dir
        / "clone_bin_cna.tsv",
        sep="\t",
        index=False,
    )

    result.clone_cna_summary.to_csv(
        output_dir
        / "clone_cna_summary.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )

    result.cell_clone_mapping.to_csv(
        output_dir
        / "cell_clone_cna_mapping.tsv",
        sep="\t",
        index=False,
    )

    with (
        output_dir
        / "cna_summary.yaml"
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
