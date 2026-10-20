"""CFB Power forecasting package."""

from .exceptions import (
    CFBPowerError,
    ConfigurationError,
    DataMissingError,
    ExportError,
    ModelNotFittedError,
    ValidationError,
)

MODEL_VERSION = "v1.0.0-baseline-ridge"

__all__ = [
    "CFBPowerError",
    "ConfigurationError",
    "DataMissingError",
    "ExportError",
    "MODEL_VERSION",
    "ModelNotFittedError",
    "ValidationError",
]
