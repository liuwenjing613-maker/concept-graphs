#!/usr/bin/env python3
"""Serial, single-key health check through the official OpenAI Python SDK."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


def check(env_name: str, image_path: Path | None) -> dict:
    key = os.environ.get(env_name)
    if not key:
        return {"status": "MISSING", "api_key_env": env_name, "elapsed_seconds": 0.0}
    content = [{"type": "text", "text": "Reply exactly OK."}]
    image_bytes = 0
    if image_path is not None:
        raw = image_path.read_bytes()
        if not raw.startswith(b"\xff\xd8\xff"):
            raise ValueError(f"image must be a rendered JPEG: {image_path}")
        image_bytes = len(raw)
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii"), "detail": "high"}})
    payload = {
        "model": "gpt-5.6-terra",
        "reasoning_effort": "high",
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": 100,
    }
    started = time.perf_counter()
    transport = httpx.Client(timeout=120, trust_env=False)
    client = OpenAI(api_key=key, base_url="https://api.codelink.chat/v1", timeout=120, max_retries=0, http_client=transport)
    try:
        response = client.chat.completions.create(**payload)
        answer = str(response.choices[0].message.content).strip()
        return {
            "status": "PASS" if answer == "OK" else "UNEXPECTED_CONTENT",
            "http_status": 200,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "request_body_utf8_bytes": len(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
            "image_bytes": image_bytes,
            "request_id": getattr(response, "_request_id", None),
        }
    except APIStatusError as exc:
        return {
            "status": f"HTTP_{exc.status_code}",
            "http_status": exc.status_code,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "request_body_utf8_bytes": len(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
            "image_bytes": image_bytes,
            "request_id": getattr(exc, "request_id", None) or exc.response.headers.get("x-request-id"),
            "error_body": exc.response.text,
        }
    except (APIConnectionError, APITimeoutError) as exc:
        return {
            "status": type(exc).__name__,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "request_body_utf8_bytes": len(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
            "image_bytes": image_bytes,
            "error": str(exc),
        }
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key-env", default="GATE_API_KEY")
    parser.add_argument("--image", type=Path, default=None, help="Rendered JPEG evidence; omit only for a text-only diagnosis")
    args = parser.parse_args()
    print(json.dumps({
        "model": "gpt-5.6-terra",
        "reasoning_effort": "high",
        "credentials_redacted": True,
        "client": "openai-python/httpx trust_env=false",
        "result": check(args.api_key_env, args.image),
    }, indent=2))


if __name__ == "__main__":
    main()
