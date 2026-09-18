"""Run detect-secrets scan against codebase and verify no secrets are present."""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout

import detect_secrets.main


def main() -> int:
    args = [
        "scan",
        "--exclude-files",
        r"\.env.*",
        "--exclude-files",
        r"uv\.lock",
        "--exclude-files",
        r"data/.*",
        "--exclude-files",
        r"chroma/.*",
        "src",
        "tests",
        "main.py",
        "pyproject.toml",
    ]
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        try:
            detect_secrets.main.main(args)
        except SystemExit:
            pass

    output = buffer.getvalue()
    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        print(
            f"Failed to parse detect-secrets output: {exc}\n{output}",
            file=sys.stderr,
        )
        return 1

    results = data.get("results", {})
    if results:
        print(
            f"Secrets detected in codebase:\n{json.dumps(results, indent=2)}",
            file=sys.stderr,
        )
        return 1

    print("No secrets detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
