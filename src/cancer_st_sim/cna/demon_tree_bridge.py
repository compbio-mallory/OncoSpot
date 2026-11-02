from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


class CNAsimTreeBridgeError(RuntimeError):
    """Raised when the malignant DEMON -> CNAsim bridge is invalid."""


@dataclass
class CNAsimTreeBridgeResult:
    newick: str
    node_mapping: pd.DataFrame
    malignant_clone_sizes: pd.DataFrame
    summary: dict[str, Any]


def _normalize_parent(value: Any) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()

    if text in {
        "",
        "nan",
        "None",
        "NA",
        "NaN",
    }:
        return None

    return text


def _load_clone_tree(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise CNAsimTreeBridgeError(
            f"Missing clone tree: {path}"
        )

    frame = pd.read_csv(
        path,
        sep="\t",
        dtype={
            "subclone_id": str,
        },
    )

    required = {
        "subclone_id",
        "parent_subclone_id",
        "origin_generation",
        "final_population",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise CNAsimTreeBridgeError(
            "Clone tree missing required columns: "
            + ", ".join(sorted(missing))
        )

    frame = frame.copy()

    frame["subclone_id"] = (
        frame["subclone_id"]
        .astype(str)
        .str.strip()
    )

    frame["parent_subclone_id"] = (
        frame["parent_subclone_id"]
        .map(_normalize_parent)
    )

    if frame["subclone_id"].duplicated().any():
        raise CNAsimTreeBridgeError(
            "Duplicate subclone IDs found in clone tree."
        )

    roots = frame[
        frame["parent_subclone_id"].isna()
    ]

    if len(roots) != 1:
        raise CNAsimTreeBridgeError(
            "Clone tree must contain exactly one root; "
            f"found {len(roots)}."
        )

    clone_ids = set(
        frame["subclone_id"]
    )

    for row in frame.itertuples(
        index=False
    ):
        parent = row.parent_subclone_id

        if parent is None:
            continue

        if parent not in clone_ids:
            raise CNAsimTreeBridgeError(
                f"Unknown parent {parent!r} "
                f"for {row.subclone_id!r}."
            )

    return frame


def _load_malignant_clone_sizes(
    cells_path: Path,
) -> pd.DataFrame:
    if not cells_path.exists():
        raise CNAsimTreeBridgeError(
            f"Missing cell table: {cells_path}"
        )

    cells = pd.read_csv(
        cells_path,
        sep="\t",
    )

    required = {
        "cell_id",
        "is_malignant",
        "source_demon_subclone_id",
    }

    missing = (
        required
        - set(cells.columns)
    )

    if missing:
        raise CNAsimTreeBridgeError(
            "Cell table missing required columns: "
            + ", ".join(sorted(missing))
        )

    malignant_mask = (
        cells["is_malignant"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
            ]
        )
    )

    malignant = cells.loc[
        malignant_mask
    ].copy()

    if malignant.empty:
        raise CNAsimTreeBridgeError(
            "No malignant cells found."
        )

    if (
        malignant[
            "source_demon_subclone_id"
        ].isna().any()
    ):
        raise CNAsimTreeBridgeError(
            "One or more malignant cells lack "
            "source DEMON subclone identity."
        )

    sizes = (
        malignant[
            "source_demon_subclone_id"
        ]
        .astype(str)
        .value_counts()
        .rename_axis(
            "subclone_id"
        )
        .reset_index(
            name="malignant_cells"
        )
    )

    sizes["malignant_cells"] = (
        sizes["malignant_cells"]
        .astype(int)
    )

    return sizes


def _required_malignant_lineage(
    clone_tree: pd.DataFrame,
    observed_subclones: set[str],
) -> set[str]:
    parent_lookup = dict(
        zip(
            clone_tree[
                "subclone_id"
            ],
            clone_tree[
                "parent_subclone_id"
            ],
        )
    )

    missing = (
        observed_subclones
        - set(parent_lookup)
    )

    if missing:
        raise CNAsimTreeBridgeError(
            "Malignant cells reference clones absent "
            "from canonical clone tree: "
            + ", ".join(
                sorted(missing)
            )
        )

    required: set[str] = set()

    for clone_id in observed_subclones:
        current: str | None = clone_id
        visited: set[str] = set()

        while current is not None:
            if current in visited:
                raise CNAsimTreeBridgeError(
                    "Cycle detected while tracing "
                    f"ancestry from {clone_id}."
                )

            visited.add(current)
            required.add(current)

            current = parent_lookup[
                current
            ]

    return required


def _validate_pruned_tree(
    frame: pd.DataFrame,
) -> str:
    roots = frame[
        frame[
            "parent_subclone_id"
        ].isna()
    ]

    if len(roots) != 1:
        raise CNAsimTreeBridgeError(
            "Pruned malignant lineage must contain "
            "exactly one root."
        )

    root_id = str(
        roots.iloc[0][
            "subclone_id"
        ]
    )

    ids = set(
        frame["subclone_id"]
    )

    parent_lookup = dict(
        zip(
            frame["subclone_id"],
            frame["parent_subclone_id"],
        )
    )

    for clone_id in ids:
        visited: set[str] = set()
        current: str | None = clone_id

        while current is not None:
            if current in visited:
                raise CNAsimTreeBridgeError(
                    "Cycle detected in malignant lineage."
                )

            visited.add(current)

            current = parent_lookup.get(
                current
            )

        if root_id not in visited:
            raise CNAsimTreeBridgeError(
                f"Node {clone_id} is disconnected "
                "from malignant lineage root."
            )

    return root_id


def _children_map(
    frame: pd.DataFrame,
) -> dict[str, list[str]]:
    children = {
        clone_id: []
        for clone_id in frame[
            "subclone_id"
        ]
    }

    generation_lookup = dict(
        zip(
            frame[
                "subclone_id"
            ],
            frame[
                "origin_generation"
            ],
        )
    )

    for row in frame.itertuples(
        index=False
    ):
        parent = row.parent_subclone_id

        if parent is None:
            continue

        children[
            parent
        ].append(
            str(
                row.subclone_id
            )
        )

    # Deterministic order.
    #
    # This order is used BOTH in the Newick and in the breadth-first
    # CNAsim naming reproduction below.
    for parent in children:
        children[parent].sort(
            key=lambda clone_id: (
                float(
                    generation_lookup[
                        clone_id
                    ]
                ),
                clone_id,
            )
        )

    return children


def _tree_depths(
    root_id: str,
    children: dict[str, list[str]],
) -> dict[str, int]:
    depths = {
        root_id: 0
    }

    queue = deque(
        [root_id]
    )

    while queue:
        parent = queue.popleft()

        for child in children[
            parent
        ]:
            depths[
                child
            ] = (
                depths[parent]
                + 1
            )

            queue.append(
                child
            )

    return depths


def _build_newick(
    node_id: str,
    root_id: str,
    children: dict[str, list[str]],
) -> str:
    """
    Use unit branch lengths.

    This matches the archived bridge default when
    --use-generation-lengths was not requested.
    """
    child_ids = children[
        node_id
    ]

    if child_ids:
        node_text = (
            "("
            + ",".join(
                _build_newick(
                    node_id=child,
                    root_id=root_id,
                    children=children,
                )
                for child
                in child_ids
            )
            + ")"
            + node_id
        )
    else:
        node_text = node_id

    if node_id != root_id:
        node_text += ":1"

    return node_text


def _build_node_mapping(
    frame: pd.DataFrame,
    root_id: str,
    children: dict[str, list[str]],
    malignant_sizes: dict[str, int],
) -> pd.DataFrame:
    lookup = frame.set_index(
        "subclone_id"
    )

    depths = _tree_depths(
        root_id=root_id,
        children=children,
    )

    leaf_counter = 1
    internal_counter = 1

    queue = deque(
        [root_id]
    )

    records: list[
        dict[str, Any]
    ] = []

    while queue:
        clone_id = queue.popleft()

        child_ids = children[
            clone_id
        ]

        if clone_id == root_id:
            cnasim_node_id = (
                "founder"
            )
            cnasim_tree_role = (
                "root"
            )

        elif child_ids:
            cnasim_node_id = (
                f"ancestor{internal_counter}"
            )

            internal_counter += 1

            cnasim_tree_role = (
                "internal"
            )

        else:
            cnasim_node_id = (
                f"cell{leaf_counter}"
            )

            leaf_counter += 1

            cnasim_tree_role = (
                "leaf"
            )

        malignant_cells = int(
            malignant_sizes.get(
                clone_id,
                0,
            )
        )

        biological_role = (
            "observed"
            if malignant_cells > 0
            else "ancestor_only"
        )

        row = lookup.loc[
            clone_id
        ]

        records.append(
            {
                "subclone_id":
                    clone_id,

                "parent_subclone_id":
                    row[
                        "parent_subclone_id"
                    ],

                "cnasim_node_id":
                    cnasim_node_id,

                "cnasim_tree_role":
                    cnasim_tree_role,

                "biological_role":
                    biological_role,

                "malignant_cells":
                    malignant_cells,

                "original_demon_final_population":
                    int(
                        row[
                            "final_population"
                        ]
                    ),

                "origin_generation":
                    float(
                        row[
                            "origin_generation"
                        ]
                    ),

                "tree_depth":
                    int(
                        depths[
                            clone_id
                        ]
                    ),

                "n_children":
                    int(
                        len(
                            child_ids
                        )
                    ),

                "branch_length":
                    (
                        0.0
                        if clone_id
                        == root_id
                        else 1.0
                    ),
            }
        )

        queue.extend(
            child_ids
        )

    mapping = pd.DataFrame(
        records
    )

    if mapping[
        "subclone_id"
    ].duplicated().any():
        raise CNAsimTreeBridgeError(
            "Duplicate biological subclone IDs "
            "in node mapping."
        )

    if mapping[
        "cnasim_node_id"
    ].duplicated().any():
        raise CNAsimTreeBridgeError(
            "Duplicate CNAsim node IDs "
            "in node mapping."
        )

    return mapping


def build_malignant_cnasim_tree(
    clone_tree: pd.DataFrame,
    malignant_clone_sizes: pd.DataFrame,
) -> CNAsimTreeBridgeResult:
    observed = set(
        malignant_clone_sizes[
            "subclone_id"
        ].astype(str)
    )

    required = (
        _required_malignant_lineage(
            clone_tree=clone_tree,
            observed_subclones=observed,
        )
    )

    pruned = clone_tree.loc[
        clone_tree[
            "subclone_id"
        ].isin(
            required
        )
    ].copy()

    # All retained non-root parents must remain in the pruned tree
    # because ancestors were explicitly added above.
    root_id = _validate_pruned_tree(
        pruned
    )

    children = _children_map(
        pruned
    )

    newick = (
        _build_newick(
            node_id=root_id,
            root_id=root_id,
            children=children,
        )
        + ";\n"
    )

    malignant_sizes_lookup = dict(
        zip(
            malignant_clone_sizes[
                "subclone_id"
            ],
            malignant_clone_sizes[
                "malignant_cells"
            ],
        )
    )

    mapping = _build_node_mapping(
        frame=pruned,
        root_id=root_id,
        children=children,
        malignant_sizes=(
            malignant_sizes_lookup
        ),
    )

    # ------------------------------------------------------------
    # Conservation / coverage checks
    # ------------------------------------------------------------

    malignant_total = int(
        malignant_clone_sizes[
            "malignant_cells"
        ].sum()
    )

    mapping_total = int(
        mapping[
            "malignant_cells"
        ].sum()
    )

    observed_mapping = set(
        mapping.loc[
            mapping[
                "malignant_cells"
            ] > 0,
            "subclone_id",
        ]
    )

    all_observed_mapped = (
        observed_mapping
        == observed
    )

    malignant_population_conserved = (
        mapping_total
        == malignant_total
    )

    if not all_observed_mapped:
        raise CNAsimTreeBridgeError(
            "Not all observed malignant clones "
            "were mapped to CNAsim nodes."
        )

    if not malignant_population_conserved:
        raise CNAsimTreeBridgeError(
            "Malignant population was not conserved "
            "during CNAsim tree construction."
        )

    clone_sizes = (
        malignant_clone_sizes
        .merge(
            clone_tree[
                [
                    "subclone_id",
                    "final_population",
                ]
            ],
            on="subclone_id",
            how="left",
            validate="one_to_one",
        )
        .rename(
            columns={
                "final_population":
                    "original_demon_final_population"
            }
        )
        .sort_values(
            "malignant_cells",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    clone_sizes[
        "retained_fraction_after_compartment_assignment"
    ] = (
        clone_sizes[
            "malignant_cells"
        ]
        / clone_sizes[
            "original_demon_final_population"
        ]
    )

    n_original_nodes = int(
        len(
            clone_tree
        )
    )

    n_observed = int(
        len(
            malignant_clone_sizes
        )
    )

    n_retained = int(
        len(
            mapping
        )
    )

    n_ancestor_only = int(
        (
            mapping[
                "biological_role"
            ]
            == "ancestor_only"
        ).sum()
    )

    n_pruned = (
        n_original_nodes
        - n_retained
    )

    summary = {
        "source":
            "canonical_demon_tree_plus_malignant_cells",

        "root_subclone_id":
            root_id,

        "branch_length_mode":
            "unit",

        "original_demon_tree_nodes":
            n_original_nodes,

        "observed_malignant_subclones":
            n_observed,

        "retained_cnasim_tree_nodes":
            n_retained,

        "ancestor_only_nodes":
            n_ancestor_only,

        "pruned_irrelevant_nodes":
            int(
                n_pruned
            ),

        "malignant_cells":
            malignant_total,

        "mapped_malignant_cells":
            mapping_total,

        "all_observed_malignant_clones_mapped":
            bool(
                all_observed_mapped
            ),

        "malignant_population_conserved":
            bool(
                malignant_population_conserved
            ),

        "unique_biological_node_ids":
            bool(
                not mapping[
                    "subclone_id"
                ].duplicated().any()
            ),

        "unique_cnasim_node_ids":
            bool(
                not mapping[
                    "cnasim_node_id"
                ].duplicated().any()
            ),

        "cnasim_naming":
            {
                "root":
                    "founder",

                "internal":
                    "ancestorN",

                "leaf":
                    "cellN",

                "traversal":
                    "breadth_first",
            },

        "note":
            (
                "The CNAsim tree contains every DEMON subclone "
                "with at least one malignant cell plus any ancestral "
                "nodes required to connect those clones to the root. "
                "Branches irrelevant to the final malignant tissue "
                "are removed. Internal biological clones may still "
                "contain malignant cells and therefore use CNAsim "
                "ancestral-node CNA profiles."
            ),
    }

    return CNAsimTreeBridgeResult(
        newick=newick,
        node_mapping=mapping,
        malignant_clone_sizes=clone_sizes,
        summary=summary,
    )


def write_malignant_cnasim_tree(
    clone_tree_path: Path,
    cells_path: Path,
    output_dir: Path,
) -> CNAsimTreeBridgeResult:
    clone_tree = _load_clone_tree(
        clone_tree_path
    )

    malignant_clone_sizes = (
        _load_malignant_clone_sizes(
            cells_path
        )
    )

    result = build_malignant_cnasim_tree(
        clone_tree=clone_tree,
        malignant_clone_sizes=(
            malignant_clone_sizes
        ),
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_dir
        / "malignant_tree.nwk"
    ).write_text(
        result.newick,
        encoding="utf-8",
    )

    result.node_mapping.to_csv(
        output_dir
        / "node_mapping.tsv",
        sep="\t",
        index=False,
    )

    result.malignant_clone_sizes.to_csv(
        output_dir
        / "malignant_clone_sizes.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )

    with (
        output_dir
        / "bridge_summary.yaml"
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
