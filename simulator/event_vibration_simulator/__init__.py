"""Standalone synthetic vibration experiment generator."""

from .config import SCENARIO_TYPES, deep_merge, default_parameters, load_parameter_file
from .experiment import generate_experiment

__all__ = [
    "SCENARIO_TYPES",
    "deep_merge",
    "default_parameters",
    "generate_experiment",
    "load_parameter_file",
]

__version__ = "1.0.0"
