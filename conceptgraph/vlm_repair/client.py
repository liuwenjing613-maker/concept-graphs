from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .evidence import ImageEvidence


class VLMRequestError(RuntimeError):
    """A redacted error from an OpenAI-compatible VLM endpoint."""


@dataclass(frozen=True)
class VLMResponse:
    text: str
    model: str
    response_id: str | None
    usage: dict[str, Any]
    elapsed_seconds: float


def _data_uri(image: ImageEvidence) -> str:
    mime = mimetypes.guess_type(image.path.name)[0] or "image/png"
    payload = base64.b64encode(image.path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


class OpenAICompatibleChatVLM:
    """Minimal Chat Completions client that never serializes its API key."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str = "VLM_API_KEY",
        timeout_seconds: float = 240.0,
        max_retries: int = 3,
        max_completion_tokens: int = 900,
    ) -> None:
        key = os.environ.get(api_key_env)
        if not key:
            raise VLMRequestError(f"API key environment variable is not set: {api_key_env}")
        self._api_key = key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_completion_tokens = max_completion_tokens

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[ImageEvidence],
    ) -> VLMResponse:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _data_uri(image), "detail": image.detail},
                }
            )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "max_completion_tokens": self.max_completion_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
            "store": False,
        }
        return self._post(body)

    def _post(self, body: dict[str, Any]) -> VLMResponse:
        url = self.base_url + "/chat/completions"
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                url,
                data=payload,
                method="POST",
                headers={
                    "Authorization": "Bearer " + self._api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "ali-my-vlm-repair/1.0",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    decoded = json.loads(response.read())
                choices = decoded.get("choices") or []
                text = (
                    choices[0].get("message", {}).get("content") if choices else None
                )
                if not isinstance(text, str) or not text.strip():
                    raise VLMRequestError("VLM returned no message content")
                return VLMResponse(
                    text=text,
                    model=str(decoded.get("model") or self.model),
                    response_id=decoded.get("id"),
                    usage=decoded.get("usage") or {},
                    elapsed_seconds=time.monotonic() - started,
                )
            except urllib.error.HTTPError as exc:
                safe_body = exc.read(2000).decode("utf-8", "replace")
                last_error = VLMRequestError(f"HTTP {exc.code}: {safe_body}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = VLMRequestError(f"transport/response error: {type(exc).__name__}: {exc}")
            if attempt < self.max_retries:
                time.sleep(min(2**attempt, 8))
        raise last_error or VLMRequestError("VLM request failed")
