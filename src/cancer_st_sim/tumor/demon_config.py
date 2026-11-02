from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class DemonConfigError(ValueError):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise DemonConfigError(
            f"Expected YAML mapping in {path}"
        )

    return data


def _default(section: dict[str, Any], key: str) -> Any:
    value = section[key]

    if isinstance(value, dict) and "default" in value:
        return value["default"]

    return value


def build_demon_config(
    resolved_config: dict[str, Any],
    demon_preset: dict[str, Any],
) -> str:

    simulation = resolved_config["simulation"]

    max_pop = int(
        simulation["n_cells"]
    )

    seed = int(
        simulation["seed"]
    )

    spatial = demon_preset["spatial_structure"]
    dispersal = demon_preset["dispersal"]
    mutation = demon_preset["mutation_rates"]
    fitness = demon_preset["fitness_effects"]
    runtime = demon_preset["runtime"]
    outputs = demon_preset["outputs"]

    log2_k = int(
        _default(
            spatial,
            "log2_deme_carrying_capacity",
        )
    )

    mu_driver_birth = float(
        _default(
            mutation,
            "mu_driver_birth",
        )
    )

    mu_passenger = float(
        _default(
            mutation,
            "mu_passenger",
        )
    )

    mu_driver_migration = float(
        _default(
            mutation,
            "mu_driver_migration",
        )
    )

    normal_birth_rate = float(
        _default(
            fitness,
            "normal_birth_rate",
        )
    )

    baseline_death_rate = float(
        _default(
            fitness,
            "baseline_death_rate",
        )
    )

    s_driver_birth = float(
        _default(
            fitness,
            "s_driver_birth",
        )
    )

    s_passenger = float(
        _default(
            fitness,
            "s_passenger",
        )
    )

    s_driver_migration = float(
        _default(
            fitness,
            "s_driver_migration",
        )
    )

    max_relative_birth_rate = float(
        _default(
            fitness,
            "max_relative_birth_rate",
        )
    )

    max_relative_migration_rate = float(
        _default(
            fitness,
            "max_relative_migration_rate",
        )
    )

    config = f"""parameters
{{
    spatial_structure
    {{
        log2_deme_carrying_capacity {log2_k}
    }}

    dispersal
    {{
        migration_type {dispersal["migration_type"]}
        init_migration_rate {dispersal["init_migration_rate"]}
        migration_edge_only {dispersal["migration_edge_only"]}
        migration_rate_scales_with_K {dispersal["migration_rate_scales_with_K"]}
    }}

    mutation_rates
    {{
        mu_driver_birth {mu_driver_birth}
        mu_passenger {mu_passenger}
        mu_driver_migration {mu_driver_migration}
        passenger_pop_threshold {mutation["passenger_pop_threshold"]}
    }}

    fitness_effects
    {{
        normal_birth_rate {normal_birth_rate}
        baseline_death_rate {baseline_death_rate}
        s_driver_birth {s_driver_birth}
        s_passenger {s_passenger}
        s_driver_migration {s_driver_migration}
        max_relative_birth_rate {max_relative_birth_rate}
        max_relative_migration_rate {max_relative_migration_rate}
    }}

    non_biological_parameters
    {{
        init_pop {runtime["init_pop"]}
        matrix_max -1

        max_pop {max_pop}
        max_time {runtime["max_time"]}
        max_generations {runtime["max_generations"]}

        seed {seed}

        write_grid {outputs["write_grid"]}
        write_clones_file {outputs["write_clones_file"]}
        write_demes_file {outputs["write_demes_file"]}
        record_matrix {outputs["record_matrix"]}
        write_phylo {outputs["write_phylo"]}
        calculate_total_diversity {outputs["calculate_total_diversity"]}

        biopsy_size_per_sample {outputs["biopsy_size_per_sample"]}
    }}
}}
"""

    return config
