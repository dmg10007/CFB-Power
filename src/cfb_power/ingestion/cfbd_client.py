"""Minimal CollegeFootballData API client (standard library only)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.collegefootballdata.com"
SAFE_ENDPOINTS = frozenset({"/games", "/games/teams", "/game/box/advanced", "/teams"})


class CfbdHttpError(RuntimeError):
    """Controlled HTTP diagnostics, without response bodies, URLs, or headers."""

    def __init__(self, status_code: int, endpoint: str, detail: str | None = None) -> None:
        self.status_code = (
            status_code if type(status_code) is int and 100 <= status_code <= 599 else 0
        )
        self.endpoint = endpoint if endpoint in SAFE_ENDPOINTS else "[redacted endpoint]"
        self.detail = detail
        msg = f"CFBD request failed: HTTP {self.status_code}; endpoint {self.endpoint}"
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


class CfbdClient:
    def __init__(
        self, api_key: str | None = None, timeout: float = 30.0, max_retries: int = 5
    ) -> None:
        self.api_key = api_key or os.environ.get("CFBD_API_KEY")
        if not self.api_key:
            raise ValueError("CFBD API key missing: pass api_key or set CFBD_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries

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
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    call_remaining = (
                        exc.headers.get("X-CallLimit-Remaining") if exc.headers else None
                    )
                    is_monthly_quota = call_remaining == "0"
                    if not is_monthly_quota:
                        try:
                            body = exc.read().decode("utf-8", errors="replace")
                            if "quota exceeded" in body.lower():
                                is_monthly_quota = True
                        except (OSError, UnicodeDecodeError):
                            pass

                    if is_monthly_quota:
                        raise CfbdHttpError(
                            exc.code,
                            path,
                            detail="Monthly call quota exceeded on your CFBD account",
                        ) from None

                    if attempt < self.max_retries - 1:
                        retry_header = exc.headers.get("Retry-After") if exc.headers else None
                        if retry_header and retry_header.isdigit():
                            wait_sec = float(retry_header)
                        else:
                            wait_sec = min(30.0, (2.0**attempt) * 1.5)
                        time.sleep(wait_sec)
                        continue
                raise CfbdHttpError(exc.code, path) from None

    def get_advanced_box(self, game_id: int) -> Any:
        return self.get_json("/game/box/advanced", {"id": game_id})
