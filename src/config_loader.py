"""
Configuration Loader Module.

Loads and validates YAML configuration files for the customer churn pipeline.
"""

import os
from typing import Any, Dict
import yaml


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str, default="config/config.yaml"
        Path to the YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the configuration file is empty or invalid.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not config:
        raise ValueError(f"Configuration file is empty: {config_path}")

    return config
