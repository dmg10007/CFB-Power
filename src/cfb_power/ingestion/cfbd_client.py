"""Minimal CollegeFootballData API client (standard library only)."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any

BASE_URL = "https://api.collegefootballdata.com"


class CfbdClient:
    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
        self.api_key = api_key or os.environ.get("CFBD_API_KEY")
        if not self.api_key:
            raise ValueError("CFBD API key missing: pass api_key or set CFBD_API_KEY")
        self.timeout = timeout

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = urllib.parse.urlencode(params or {})
        url = f"{BASE_URL}{path}" + (f"?{query}" if query else "")
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def get_advanced_box(self, game_id: int) -> Any:
        return self.get_json("/game/box/advanced", {"gameId": game_id})
