from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a simulator configuration is invalid."""


SUPPORTED_TECHNOLOGIES = {
    "visium",
    "xenium",
}


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(
            f"Configuration file does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigError(
            f"Top-level YAML content must be a mapping: {path}"
        )

    return data


def deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """
    Recursively merge override into base.

    Mapping values are merged recursively.
    Scalars/lists in override replace values in base.
    """
    result = copy.deepcopy(base)

    for key, value in override.items():

        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(
                result[key],
                value,
            )
        else:
            result[key] = copy.deepcopy(value)

    return result


def resolve_numeric_parameter(
    name: str,
    user_value: Any,
    cancer_spec: dict[str, Any],
    global_spec: dict[str, Any],
    warnings: list[str],
) -> float | int:

    default = cancer_spec.get(
        "default",
        global_spec.get("default"),
    )

    if user_value is None:
        value = default
    else:
        value = user_value

    if not isinstance(value, (int, float)):
        raise ConfigError(
            f"{name} must be numeric; received {value!r}"
        )

    hard_min = global_spec.get("hard_min")
    hard_max = global_spec.get("hard_max")

    if hard_min is not None and value < hard_min:
        raise ConfigError(
            f"{name}={value} is below the hard minimum "
            f"{hard_min}."
        )

    if hard_max is not None and value > hard_max:
        raise ConfigError(
            f"{name}={value} is above the hard maximum "
            f"{hard_max}."
        )

    recommended_min = cancer_spec.get(
        "recommended_min",
        global_spec.get("recommended_min"),
    )

    recommended_max = cancer_spec.get(
        "recommended_max",
        global_spec.get("recommended_max"),
    )

    if (
        recommended_min is not None
        and value < recommended_min
    ):
        warnings.append(
            f"{name}={value} is below the recommended "
            f"range minimum {recommended_min}."
        )

    if (
        recommended_max is not None
        and value > recommended_max
    ):
        warnings.append(
            f"{name}={value} is above the recommended "
            f"range maximum {recommended_max}."
        )

    return value


def resolve_config(
    project_root: Path,
    user_config_path: Path,
) -> tuple[dict[str, Any], list[str]]:

    warnings: list[str] = []

    global_path = (
        project_root
        / "configs/defaults/global.yaml"
    )

    global_config = load_yaml(global_path)
    user_config = load_yaml(user_config_path)

    cancer_type = user_config.get("cancer_type")

    if not cancer_type:
        raise ConfigError(
            "User configuration must define cancer_type."
        )

    cancer_type = str(cancer_type).upper()

    cancer_path = (
        project_root
        / "configs/cancers"
        / f"{cancer_type}.yaml"
    )

    cancer_config = load_yaml(cancer_path)

    # ---------------------------------------------------------
    # Resolve simulation size.
    # ---------------------------------------------------------

    user_simulation = user_config.get(
        "simulation",
        {},
    )

    if not isinstance(user_simulation, dict):
        raise ConfigError(
            "simulation must be a YAML mapping."
        )

    global_n_cells = (
        global_config
        .get("simulation", {})
        .get("n_cells", {})
    )

    cancer_n_cells = (
        cancer_config
        .get("simulation", {})
        .get("n_cells", {})
    )

    n_cells = resolve_numeric_parameter(
        name="simulation.n_cells",
        user_value=user_simulation.get("n_cells"),
        cancer_spec=cancer_n_cells,
        global_spec=global_n_cells,
        warnings=warnings,
    )

    n_cells = int(n_cells)

    # ---------------------------------------------------------
    # Resolve seed.
    # ---------------------------------------------------------

    seed = user_simulation.get(
        "seed",
        global_config
        .get("simulation", {})
        .get("seed", 1),
    )

    if not isinstance(seed, int):
        raise ConfigError(
            "simulation.seed must be an integer."
        )

    # ---------------------------------------------------------
    # Resolve technology.
    # ---------------------------------------------------------

    technology = user_simulation.get(
        "technology",
        cancer_config
        .get("simulation", {})
        .get("technology", {})
        .get(
            "default",
            global_config
            .get("simulation", {})
            .get("technology", {})
            .get("default", "visium"),
        ),
    )

    technology = str(
        technology
    ).lower()

    if technology not in SUPPORTED_TECHNOLOGIES:
        raise ConfigError(
            "simulation.technology must be one of: "
            + ", ".join(
                sorted(
                    SUPPORTED_TECHNOLOGIES
                )
            )
        )

    # ---------------------------------------------------------
    # Resolve tumor fraction.
    # ---------------------------------------------------------

    user_tissue = user_config.get(
        "tissue",
        {},
    )

    if not isinstance(user_tissue, dict):
        raise ConfigError(
            "tissue must be a YAML mapping."
        )

    global_tumor_fraction = (
        global_config
        .get("tissue", {})
        .get("tumor_fraction", {})
    )

    cancer_tumor_fraction = (
        cancer_config
        .get("tissue", {})
        .get("tumor_fraction", {})
    )

    tumor_fraction = resolve_numeric_parameter(
        name="tissue.tumor_fraction",
        user_value=user_tissue.get(
            "tumor_fraction"
        ),
        cancer_spec=cancer_tumor_fraction,
        global_spec=global_tumor_fraction,
        warnings=warnings,
    )

    tumor_fraction = float(
        tumor_fraction
    )

    n_malignant = int(
        round(
            n_cells
            * tumor_fraction
        )
    )

    n_nonmalignant = (
        n_cells
        - n_malignant
    )

    # ---------------------------------------------------------
    # Resolve high-level user customization.
    # ---------------------------------------------------------

    customize = user_config.get(
        "customize",
        {},
    )

    if not isinstance(customize, dict):
        raise ConfigError(
            "customize must be a YAML mapping."
        )

    cna_burden = customize.get(
        "cna_burden",
        cancer_config
        .get("cnasim", {})
        .get("user_controls", {})
        .get("cna_burden", {})
        .get("default", "default"),
    )

    allowed_cna_burden = (
        cancer_config
        .get("cnasim", {})
        .get("user_controls", {})
        .get("cna_burden", {})
        .get(
            "allowed",
            [
                "low",
                "default",
                "high",
            ],
        )
    )

    if cna_burden not in allowed_cna_burden:
        raise ConfigError(
            "customize.cna_burden must be one of: "
            + ", ".join(
                allowed_cna_burden
            )
        )

    # ---------------------------------------------------------
    # Assemble resolved configuration.
    # ---------------------------------------------------------

    resolved = {
        "schema_version": "1.0",

        "cancer_type":
            cancer_type,

        "preset_metadata":
            copy.deepcopy(
                cancer_config.get(
                    "metadata",
                    {},
                )
            ),

        "simulation": {
            "seed":
                seed,

            "n_cells":
                n_cells,

            "technology":
                technology,
        },

        "tissue": {
            "tumor_fraction":
                tumor_fraction,

            "n_malignant":
                n_malignant,

            "n_nonmalignant":
                n_nonmalignant,
        },

        "malignant":
            deep_merge(
                cancer_config.get(
                    "malignant",
                    {},
                ),
                user_config.get(
                    "malignant",
                    {},
                ),
            ),

        "demon":
            copy.deepcopy(
                cancer_config.get(
                    "demon",
                    {},
                )
            ),

        "cnasim":
            copy.deepcopy(
                cancer_config.get(
                    "cnasim",
                    {},
                )
            ),

        "tme":
            deep_merge(
                cancer_config.get(
                    "tme",
                    {},
                ),
                user_config.get(
                    "tme",
                    {},
                ),
            ),

        "expression":
            deep_merge(
                cancer_config.get(
                    "expression",
                    {},
                ),
                user_config.get(
                    "expression",
                    {},
                ),
            ),

        "tumor_states":
            deep_merge(
                cancer_config.get(
                    "tumor_states",
                    {},
                ),
                user_config.get(
                    "tumor_states",
                    {},
                ),
            ),

        "observation":
            deep_merge(
                cancer_config.get(
                    "observation",
                    {},
                ),
                user_config.get(
                    "observation",
                    {},
                ),
            ),

        "user_controls": {
            "cna_burden":
                cna_burden,
        },
    }

    # ---------------------------------------------------------
    # Technology-specific checks.
    # ---------------------------------------------------------

    if technology == "xenium":

        xenium_status = (
            resolved
            .get("observation", {})
            .get("xenium", {})
            .get(
                "status",
                "pending_implementation",
            )
        )

        warnings.append(
            "Xenium configuration is accepted by the "
            f"v1 config layer, but observation status is "
            f"{xenium_status!r}."
        )

    # ---------------------------------------------------------
    # DEMON mode warning.
    # ---------------------------------------------------------

    demon_mode = (
        resolved
        .get("demon", {})
        .get("mode")
    )

    if demon_mode in {
        None,
        "",
        "pending",
    }:
        warnings.append(
            f"{cancer_type} DEMON architecture is currently "
            "pending preset assignment. This configuration "
            "can be resolved, but the DEMON stage should not "
            "run until the cancer-specific mode is finalized."
        )

    resolved["validation"] = {
        "warnings":
            warnings,

        "n_warnings":
            len(warnings),
    }

    return resolved, warnings
