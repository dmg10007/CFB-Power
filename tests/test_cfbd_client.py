"""Offline regression tests for the CFBD request contract."""

import json
from unittest.mock import MagicMock, patch

from cfb_power.ingestion.cfbd_client import CfbdClient

GAME_ID = 401856766


def test_advanced_box_uses_id_parameter_and_returns_response():
    client = CfbdClient(api_key="test-key")
    response = {"teams": {}}
    with patch.object(client, "get_json", return_value=response) as get_json:
        assert client.get_advanced_box(GAME_ID) is response
    get_json.assert_called_once_with("/game/box/advanced", {"id": GAME_ID})


def test_advanced_box_request_url_headers_and_timeout():
    response_payload = {"teams": {}}
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps(response_payload).encode("utf-8")
    client = CfbdClient(api_key="test-key", timeout=7.0)
    with patch("cfb_power.ingestion.cfbd_client.urllib.request.urlopen", return_value=response) as urlopen:
        assert client.get_advanced_box(GAME_ID) == response_payload
    urlopen.assert_called_once()
    request = urlopen.call_args.args[0]
    assert request.full_url == f"https://api.collegefootballdata.com/game/box/advanced?id={GAME_ID}"
    assert request.get_header("Authorization") == "Bearer test-key"
    assert request.get_header("Accept") == "application/json"
    assert urlopen.call_args.kwargs == {"timeout": 7.0}
