#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import math
import re
import shutil
from pathlib import Path

import pandas as pd
import yaml


CLONE_RE = re.compile(r"^clone_(\d+)$")


def clone_number(value: str) -> int:
    m = CLONE_RE.match(str(value))
    if m is None:
        raise ValueError(
            f"Unexpected clone label {value!r}; expected clone_<integer>."
        )
    return int(m.group(1))


class NewickNode:
    def __init__(self, name: str | None = None):
        self.name = name
        self.children: list["NewickNode"] = []


def parse_newick(text: str) -> NewickNode:
    """
    Minimal Newick parser sufficient for the CNVoxel/CNAsim bridge tree.
    Branch lengths are ignored because 'nearest' is topological distance.
    """

    s = text.strip()
    pos = 0
    internal_counter = 0

    def skip_ws():
        nonlocal pos
        while pos < len(s) and s[pos].isspace():
            pos += 1

    def read_token() -> str:
        nonlocal pos
        skip_ws()

        start = pos

        while (
            pos < len(s)
            and s[pos] not in "(),:;"
            and not s[pos].isspace()
        ):
            pos += 1

        return s[start:pos]

    def skip_branch_length():
        nonlocal pos
        skip_ws()

        if pos < len(s) and s[pos] == ":":
            pos += 1
            skip_ws()

            while (
                pos < len(s)
                and s[pos] not in ",);"
            ):
                pos += 1

    def parse_subtree() -> NewickNode:
        nonlocal pos, internal_counter

        skip_ws()

        if pos >= len(s):
            raise ValueError("Unexpected end of Newick string.")

        if s[pos] == "(":
            pos += 1

            children = []

            while True:
                children.append(parse_subtree())
                skip_ws()

                if pos >= len(s):
                    raise ValueError("Unterminated Newick subtree.")

                if s[pos] == ",":
                    pos += 1
                    continue

                if s[pos] == ")":
                    pos += 1
                    break

                raise ValueError(
                    f"Unexpected character {s[pos]!r} at position {pos}."
                )

            name = read_token()

            if not name:
                name = f"__internal_{internal_counter}"
                internal_counter += 1

            node = NewickNode(name)
            node.children = children

            skip_branch_length()
            return node

        name = read_token()

        if not name:
            raise ValueError(
                f"Missing leaf name at position {pos}."
            )

        node = NewickNode(name)
        skip_branch_length()
        return node

    root = parse_subtree()
    skip_ws()

    if pos < len(s) and s[pos] == ";":
        pos += 1

    skip_ws()

    if pos != len(s):
        raise ValueError(
            f"Unexpected trailing Newick content: {s[pos:]!r}"
        )

    return root


def build_graph(root: NewickNode):
    adjacency: dict[str, set[str]] = collections.defaultdict(set)
    parent: dict[str, str | None] = {root.name: None}
    names: set[str] = set()

    stack = [root]

    while stack:
        node = stack.pop()
        names.add(node.name)

        for child in node.children:
            adjacency[node.name].add(child.name)
            adjacency[child.name].add(node.name)

            parent[child.name] = node.name
            stack.append(child)

    return adjacency, parent, names


def shortest_distance(
    adjacency: dict[str, set[str]],
    source: str,
    target: str,
) -> int:
    if source == target:
        return 0

    queue = collections.deque([(source, 0)])
    seen = {source}

    while queue:
        node, distance = queue.popleft()

        for nxt in adjacency[node]:
            if nxt in seen:
                continue

            if nxt == target:
                return distance + 1

            seen.add(nxt)
            queue.append((nxt, distance + 1))

    raise ValueError(
        f"No phylogenetic path between {source} and {target}."
    )


def is_ancestor(
    candidate: str,
    node: str,
    parent: dict[str, str | None],
) -> bool:
    current = node

    while current is not None:
        if current == candidate:
            return True
        current = parent.get(current)

    return False


def identify_mapping_columns(
    mapping: pd.DataFrame,
    clone_ids: set[str],
    tree_names: set[str],
) -> tuple[str, str]:
    clone_scores = {}

    for col in mapping.columns:
        vals = mapping[col].dropna().astype(str)
        clone_scores[col] = int(
            vals.isin(clone_ids).sum()
        )

    clone_col = max(
        clone_scores,
        key=clone_scores.get,
    )

    if clone_scores[clone_col] == 0:
        raise ValueError(
            "Could not identify clone column in node_mapping.tsv. "
            f"Columns: {list(mapping.columns)}"
        )

    node_scores = {}

    for col in mapping.columns:
        if col == clone_col:
            continue

        vals = mapping[col].dropna().astype(str)
        node_scores[col] = int(
            vals.isin(tree_names).sum()
        )

    if not node_scores:
        raise ValueError(
            "Could not identify tree-node column in node_mapping.tsv."
        )

    node_col = max(
        node_scores,
        key=node_scores.get,
    )

    if node_scores[node_col] == 0:
        raise ValueError(
            "No node_mapping.tsv column matched Newick node names. "
            f"Tree names={sorted(tree_names)}; "
            f"columns={list(mapping.columns)}"
        )

    return clone_col, node_col


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--min-fraction",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    run = args.run_dir.resolve()

    resolved_path = run / "resolved_config.yaml"
    typed_path = run / "cells/cells_with_cell_types.tsv"
    compartment_path = run / "cells/cells_with_compartment.tsv"

    bridge = run / "cna/bridge"
    tree_path = bridge / "malignant_tree.nwk"
    mapping_path = bridge / "node_mapping.tsv"

    for path in [
        resolved_path,
        typed_path,
        tree_path,
        mapping_path,
    ]:
        if not path.exists():
            raise SystemExit(
                f"Required input does not exist: {path}"
            )

    cfg = yaml.safe_load(
        resolved_path.read_text()
    )

    merge_cfg = (
        cfg
        .get("malignant", {})
        .get("clone_merging", {})
    )

    enabled = bool(
        merge_cfg.get("enabled", False)
    )

    if not enabled:
        print("Rare-clone merging disabled; no changes made.")
        return

    fraction = (
        float(args.min_fraction)
        if args.min_fraction is not None
        else float(
            merge_cfg.get(
                "min_fraction",
                0.01,
            )
        )
    )

    if not 0 < fraction < 1:
        raise SystemExit(
            "min_fraction must be between 0 and 1."
        )

    cells = pd.read_csv(
        typed_path,
        sep="\t",
    )

    required = {
        "cell_id",
        "is_malignant",
        "subclone_id",
        "driver_genotype_id",
    }

    missing = required - set(cells.columns)

    if missing:
        raise SystemExit(
            "Missing required cell columns: "
            + ", ".join(sorted(missing))
        )

    malignant = cells.loc[
        cells["is_malignant"].astype(bool)
    ].copy()

    if malignant.empty:
        raise SystemExit("No malignant cells found.")

    if malignant["subclone_id"].isna().any():
        raise SystemExit(
            "Malignant cells contain missing subclone IDs."
        )

    counts = (
        malignant["subclone_id"]
        .astype(str)
        .value_counts()
        .sort_values(ascending=False)
    )

    n_malignant = int(len(malignant))

    threshold = fraction * n_malignant

    retained = set(
        counts[
            counts >= threshold
        ].index.astype(str)
    )

    rare = set(
        counts[
            counts < threshold
        ].index.astype(str)
    )

    if not retained:
        raise SystemExit(
            "No clone passes the retention threshold."
        )

    # Validate the current clone -> driver genotype convention before
    # modifying anything.
    malignant_check = malignant[
        ["subclone_id", "driver_genotype_id"]
    ].drop_duplicates()

    for row in malignant_check.itertuples(index=False):
        clone = str(row.subclone_id)
        expected_driver = clone_number(clone)

        if int(row.driver_genotype_id) != expected_driver:
            raise SystemExit(
                "Clone/driver identity invariant failed: "
                f"{clone} has driver_genotype_id="
                f"{row.driver_genotype_id}, expected {expected_driver}."
            )

    root = parse_newick(
        tree_path.read_text()
    )

    adjacency, parent, tree_names = (
        build_graph(root)
    )

    # ---------------------------------------------------------
    # The bridge Newick deliberately uses biological DEMON
    # subclone IDs (clone_0, clone_7, ...) as node labels.
    #
    # CNAsim-specific names such as founder/cell1/ancestor1 are
    # recorded separately in node_mapping.tsv and must NOT be
    # used to compute biological phylogenetic distance.
    # ---------------------------------------------------------

    biological_clones = set(
        counts.index.astype(str)
    )

    missing_tree_nodes = (
        biological_clones
        - tree_names
    )

    if missing_tree_nodes:
        raise SystemExit(
            "Malignant biological clone(s) are absent from "
            "the pre-merge Newick tree: "
            + ", ".join(
                sorted(
                    missing_tree_nodes,
                    key=clone_number,
                )
            )
        )

    clone_to_node = {
        clone: clone
        for clone in biological_clones
    }

    merge_rows = []
    merge_map: dict[str, str] = {}

    for rare_clone in sorted(
        rare,
        key=clone_number,
    ):
        rare_node = clone_to_node[rare_clone]

        candidates = []

        for target in retained:
            target_node = clone_to_node[target]

            distance = shortest_distance(
                adjacency,
                rare_node,
                target_node,
            )

            ancestor_preference = (
                0
                if is_ancestor(
                    target_node,
                    rare_node,
                    parent,
                )
                else 1
            )

            candidates.append(
                (
                    distance,
                    ancestor_preference,
                    -int(counts[target]),
                    clone_number(target),
                    target,
                )
            )

        candidates.sort()

        (
            distance,
            ancestor_preference,
            _,
            _,
            target,
        ) = candidates[0]

        merge_map[rare_clone] = target

        merge_rows.append(
            {
                "source_subclone_id": rare_clone,
                "source_n_malignant_cells":
                    int(counts[rare_clone]),
                "target_subclone_id": target,
                "target_n_malignant_cells_before_merge":
                    int(counts[target]),
                "phylogenetic_edge_distance":
                    int(distance),
                "target_is_ancestor":
                    bool(
                        ancestor_preference == 0
                    ),
            }
        )

    out_dir = run / "cells/clone_merge"
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    merge_table = pd.DataFrame(
        merge_rows
    )

    merge_table.to_csv(
        out_dir / "clone_merge_map.tsv",
        sep="\t",
        index=False,
    )

    counts.rename_axis(
        "subclone_id"
    ).rename(
        "n_malignant_cells"
    ).reset_index().to_csv(
        out_dir / "clone_counts_before.tsv",
        sep="\t",
        index=False,
    )

    files_to_update = [
        typed_path,
    ]

    if compartment_path.exists():
        files_to_update.append(
            compartment_path
        )

    for path in files_to_update:
        df = pd.read_csv(
            path,
            sep="\t",
        )

        backup = path.with_name(
            path.stem
            + "_pre_clone_merge"
            + path.suffix
        )

        if backup.exists():
            raise SystemExit(
                f"Backup already exists; refusing to overwrite: {backup}"
            )

        shutil.copy2(
            path,
            backup,
        )

        if "premerge_subclone_id" not in df.columns:
            df["premerge_subclone_id"] = (
                df["subclone_id"]
            )

        if "premerge_driver_genotype_id" not in df.columns:
            df["premerge_driver_genotype_id"] = (
                df["driver_genotype_id"]
            )

        malignant_mask = (
            df["is_malignant"].astype(bool)
        )

        original_clone = (
            df.loc[
                malignant_mask,
                "subclone_id",
            ]
            .astype(str)
        )

        final_clone = original_clone.map(
            lambda x: merge_map.get(x, x)
        )

        df.loc[
            malignant_mask,
            "subclone_id",
        ] = final_clone.values

        df.loc[
            malignant_mask,
            "driver_genotype_id",
        ] = [
            clone_number(x)
            for x in final_clone
        ]

        df["driver_genotype_id"] = (
            pd.to_numeric(
                df["driver_genotype_id"],
                errors="coerce",
            )
            .astype("Int64")
        )

        df.to_csv(
            path,
            sep="\t",
            index=False,
        )

    updated = pd.read_csv(
        typed_path,
        sep="\t",
    )

    updated_malignant = updated.loc[
        updated["is_malignant"].astype(bool)
    ]

    after_counts = (
        updated_malignant[
            "subclone_id"
        ]
        .astype(str)
        .value_counts()
        .sort_values(ascending=False)
    )

    after_counts.rename_axis(
        "subclone_id"
    ).rename(
        "n_malignant_cells"
    ).reset_index().to_csv(
        out_dir / "clone_counts_after.tsv",
        sep="\t",
        index=False,
    )

    if int(after_counts.sum()) != n_malignant:
        raise SystemExit(
            "Malignant population was not conserved."
        )

    if (
        after_counts
        < threshold
    ).any():
        raise SystemExit(
            "Rare malignant clones remain after merging."
        )

    summary = {
        "stage":
            "rare_malignant_subclone_merge",
        "enabled":
            True,
        "criterion":
            "n_clone < min_fraction * n_malignant",
        "min_fraction":
            float(fraction),
        "n_malignant_cells":
            n_malignant,
        "absolute_threshold":
            float(threshold),
        "n_subclones_before":
            int(len(counts)),
        "n_subclones_retained_before_merge":
            int(len(retained)),
        "n_rare_subclones_merged":
            int(len(rare)),
        "n_subclones_after":
            int(len(after_counts)),
        "malignant_population_conserved":
            bool(
                int(after_counts.sum())
                == n_malignant
            ),
        "all_final_subclones_meet_threshold":
            bool(
                (
                    after_counts
                    >= threshold
                ).all()
            ),
        "distance_metric":
            "unweighted_shortest_path_on_premerge_malignant_phylogeny",
        "tie_breaking":
            [
                "prefer retained ancestor",
                "prefer larger malignant-cell abundance",
                "prefer smaller numeric clone ID",
            ],
        "provenance":
            (
                "source_demon_subclone_id remains unchanged; "
                "premerge_subclone_id stores the malignant clone label "
                "immediately before benchmark clone collapsing; "
                "subclone_id is the final benchmark clone label."
            ),
    }

    with (
        out_dir / "clone_merge_summary.yaml"
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
    print("RARE MALIGNANT SUBCLONE MERGE COMPLETE")
    print("=" * 80)
    print(f"Malignant cells: {n_malignant:,}")
    print(
        f"Threshold: < {fraction:.4f} "
        f"of malignant cells "
        f"(< {threshold:.2f} cells)"
    )
    print(f"Subclones before: {len(counts)}")
    print(f"Rare clones merged: {len(rare)}")
    print(f"Subclones after: {len(after_counts)}")
    print()

    if merge_rows:
        print(merge_table.to_string(index=False))
        print()

    print("Final malignant clone sizes:")
    print(after_counts.to_string())
    print()
    print("Outputs:", out_dir)


if __name__ == "__main__":
    main()
