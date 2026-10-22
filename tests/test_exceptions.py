"""Tests for custom exception hierarchy in cfb_power.exceptions."""

import pytest

from cfb_power.exceptions import (
    CFBPowerError,
    ConfigurationError,
    DataMissingError,
    ExportError,
    ModelNotFittedError,
    ValidationError,
)


def test_exception_inheritance():
    assert issubclass(DataMissingError, (CFBPowerError, FileNotFoundError))
    assert issubclass(ModelNotFittedError, (CFBPowerError, RuntimeError))
    assert issubclass(ConfigurationError, (CFBPowerError, ValueError))
    assert issubclass(ValidationError, (CFBPowerError, ValueError))
    assert issubclass(ExportError, (CFBPowerError, RuntimeError))


def test_catching_base_exception():
    with pytest.raises(CFBPowerError):
        raise DataMissingError("data missing")

    with pytest.raises(CFBPowerError):
        raise ModelNotFittedError("model not fitted")

    with pytest.raises(CFBPowerError):
        raise ConfigurationError("bad config")

    with pytest.raises(CFBPowerError):
        raise ValidationError("invalid schema")

    with pytest.raises(CFBPowerError):
        raise ExportError("export failed")
