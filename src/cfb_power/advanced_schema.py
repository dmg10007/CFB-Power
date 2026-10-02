"""Schema contract for normalized CFBD advanced box score data."""

from __future__ import annotations

KEY_COLUMNS = ("game_id", "team")

KNOWN_CATEGORIES = (
    "ppa",
    "cumulativePpa",
    "successRates",
    "explosiveness",
    "rushing",
    "havoc",
    "scoringOpportunities",
    "fieldPosition",
)


class AdvancedSchemaError(ValueError):
    """Raised when an advanced box payload does not match the expected shape."""
