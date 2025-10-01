#!/usr/bin/env python
"""Simple smoke test for the running FastAPI server.

Run the script while `uvicorn app.main:app` is active. It pings `/health`
and then exercises `/email/analyze` with a sample payload, printing the
response bodies so you can eyeball the results quickly from the terminal.
"""

import json
import sys
from typing import Any

import requests


BASE_URL = "http://127.0.0.1:8000"


def pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def main() -> None:
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        health.raise_for_status()
    except requests.RequestException as exc:
        print("✗ Health check failed:", exc)
        sys.exit(1)

    print("✓ /health response:")
    print(pretty(health.json()))

    payload = {
        "message_id": "smoke-001",
        "headers_raw": "",
        "subject": "Weekly update",
        "body_html": None,
        "body_text": "Hello team, please review the attached report.",
    }

    try:
        analyze = requests.post(
            f"{BASE_URL}/email/analyze",
            json=payload,
            timeout=30,
        )
        analyze.raise_for_status()
    except requests.RequestException as exc:
        print("✗ Analyze endpoint failed:", exc)
        if exc.response is not None:
            print("Response body:")
            print(exc.response.text)
        sys.exit(1)

    print("✓ /email/analyze response:")
    print(pretty(analyze.json()))


if __name__ == "__main__":
    main()
