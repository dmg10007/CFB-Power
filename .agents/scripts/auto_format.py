#!/usr/bin/env python3
"""Antigravity Lifecycle Hook: Auto-format and lint Python files after modification."""

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    # Always respond with a valid JSON object
    output = {}

    try:
        raw_input = sys.stdin.read()
        if raw_input.strip():
            payload = json.loads(raw_input)
            tool_call = payload.get("toolCall", {})
            args = tool_call.get("args", {})
            file_path = args.get("Path")

            if file_path and file_path.endswith(".py"):
                path_obj = Path(file_path)
                if path_obj.is_file():
                    # Format and fix with ruff if .venv exists
                    venv_ruff = Path(__file__).resolve().parent.parent.parent / ".venv" / "bin" / "ruff"
                    ruff_cmd = str(venv_ruff) if venv_ruff.is_file() else "ruff"
                    subprocess.run(
                        [ruff_cmd, "format", str(path_obj)],
                        capture_output=True,
                        check=False,
                    )
                    subprocess.run(
                        [ruff_cmd, "check", "--fix", str(path_obj)],
                        capture_output=True,
                        check=False,
                    )
    except Exception:
        # Hooks should be fail-safe and not disrupt the agent loop
        pass

    # PostToolUse expects an empty JSON object
    sys.stdout.write(json.dumps(output) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
