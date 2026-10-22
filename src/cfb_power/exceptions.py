"""Custom exception hierarchy for CFB-Power."""

from __future__ import annotations


class CFBPowerError(Exception):
    """Base exception for all CFB-Power errors."""


class DataMissingError(CFBPowerError, FileNotFoundError):
    """Raised when expected data files (CSVs, parquet, datasets) are not found."""


class ModelNotFittedError(CFBPowerError, RuntimeError):
    """Raised when predictions or exports are attempted before models are trained."""


class ConfigurationError(CFBPowerError, ValueError):
    """Raised when pipeline configuration, paths, or CLI options are invalid."""


class ValidationError(CFBPowerError, ValueError):
    """Raised when input datasets fail schema or integrity validation."""


class ExportError(CFBPowerError, RuntimeError):
    """Raised when an error occurs during workbook or dataset export."""
