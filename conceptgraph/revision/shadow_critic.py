"""VLM critic for comparing executed shadow outcomes, never mutating the map."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .auto_constraints import forbidden_inference_paths


@dataclass(frozen=True)
class ShadowCriticEvidence:
    incident_uid: str
    prompt: str
    image_paths: tuple[Path, ...]
    image_manifest: tuple[dict[str, Any], ...]
    allowed_state_ids: tuple[str, ...]
    system_prompt: str = (
        "You compare executed counterfactual 3D scene-graph outcomes using only "
        "the supplied held-out views. Judge physical-instance grouping, look for "
        "counterevidence, and return DEFER when views are insufficient. You do not "
        "modify the map and you do not know the proposal rationale or human answer."
    )

    def __post_init__(self) -> None:
        if len(set(self.allowed_state_ids)) != len(self.allowed_state_ids):
            raise ValueError("state IDs must be unique")
        if len(self.allowed_state_ids) < 2:
            raise ValueError("at least two distinct shadow states are required")
        if len(self.image_paths) != len(self.image_manifest):
            raise ValueError("critic image paths and manifest must align")
        payload = {
            "incident_uid": self.incident_uid,
            "prompt": self.prompt,
            "image_manifest": self.image_manifest,
            "allowed_state_ids": self.allowed_state_ids,
        }
        forbidden = forbidden_inference_paths(payload)
        if forbidden:
            raise ValueError("oracle-like critic evidence: " + ", ".join(forbidden))


def _text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, Iterable):
        return [str(value)]
    return [str(item) for item in value]


def parse_shadow_critic_response(
    text: str,
    *,
    allowed_state_ids: Iterable[str],
    allowed_evidence_ids: Iterable[str],
) -> dict[str, Any]:
    value = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", value, flags=re.DOTALL)
    if fenced:
        value = fenced.group(1)
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("critic response must be one JSON object")
    forbidden = forbidden_inference_paths(decoded)
    if forbidden:
        raise ValueError("oracle-like critic response: " + ", ".join(forbidden))
    states = {str(item).upper() for item in allowed_state_ids}
    preferred = str(decoded.get("preferred_state") or "").upper()
    if preferred not in states | {"DEFER"}:
        raise ValueError(f"unknown preferred_state: {preferred}")
    confidence_raw = decoded.get("confidence", 0.0)
    confidence_parse_status = "NUMERIC"
    if isinstance(confidence_raw, str):
        qualitative = {
            "very low": 0.1,
            "low": 0.25,
            "moderately low": 0.4,
            "medium low": 0.4,
            "moderate": 0.5,
            "medium": 0.5,
            "moderately high": 0.65,
            "medium high": 0.65,
            "high": 0.75,
            "very high": 0.9,
        }
        normalized = confidence_raw.strip().lower().replace("_", " ")
        confidence = qualitative.get(normalized, 0.5)
        confidence_parse_status = (
            "QUALITATIVE_NORMALIZED"
            if normalized in qualitative
            else "UNRECOGNIZED_DIAGNOSTIC_DEFAULT"
        )
    else:
        confidence = float(confidence_raw)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("critic confidence must be in [0, 1]")
    allowed_evidence = {str(item) for item in allowed_evidence_ids}
    cited = tuple(
        dict.fromkeys(str(item) for item in decoded.get("cited_evidence_ids") or ())
    )
    if not cited or not set(cited).issubset(allowed_evidence):
        raise ValueError("critic cited evidence is empty or outside the frozen split")
    return {
        "preferred_state": preferred,
        "confidence_diagnostic_only": confidence,
        "confidence_raw_diagnostic": confidence_raw,
        "confidence_parse_status": confidence_parse_status,
        "reason": str(decoded.get("reason") or ""),
        "counterevidence": _text_list(decoded.get("counterevidence")),
        "needed_evidence": _text_list(decoded.get("needed_evidence")),
        "cited_evidence_ids": list(cited),
        "confidence_is_not_calibrated_probability": True,
    }


class OpenAICompatibleShadowCriticClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 300.0,
    ) -> None:
        if not api_key:
            raise ValueError("empty API key")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = float(timeout_seconds)

    @staticmethod
    def _data_uri(path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def complete(self, evidence: ShadowCriticEvidence) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": evidence.prompt}]
        for path in evidence.image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._data_uri(path), "detail": "high"},
                }
            )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": evidence.system_prompt},
                {"role": "user", "content": content},
            ],
            "max_completion_tokens": 1000,
            "response_format": {"type": "json_object"},
            "stream": False,
            "store": False,
        }
        started = time.monotonic()
        decoded = None
        last_error: Exception | None = None
        for attempt in range(3):
            request = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=json.dumps(body, separators=(",", ":")).encode(),
                method="POST",
                headers={
                    "Authorization": "Bearer " + self._api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "ali-my-shadow-critic/0.1",
                },
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    decoded = json.loads(response.read())
                break
            except urllib.error.HTTPError as exc:
                safe = exc.read(1000).decode("utf-8", "replace")
                last_error = RuntimeError(f"VLM HTTP {exc.code}: {safe}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = RuntimeError(
                    f"VLM transport error: {type(exc).__name__}: {exc}"
                )
            if attempt < 2:
                time.sleep(2**attempt)
        if decoded is None:
            raise last_error or RuntimeError("VLM critic request failed")
        choices = decoded.get("choices") or []
        response_text = (
            choices[0].get("message", {}).get("content") if choices else None
        )
        if not isinstance(response_text, str):
            raise RuntimeError("VLM critic returned no text content")
        parsed = parse_shadow_critic_response(
            response_text,
            allowed_state_ids=evidence.allowed_state_ids,
            allowed_evidence_ids=(
                str(item["evidence_id"]) for item in evidence.image_manifest
            ),
        )
        return {
            "incident_uid": evidence.incident_uid,
            "critic": parsed,
            "model": str(decoded.get("model") or self.model),
            "response_id": decoded.get("id"),
            "usage": decoded.get("usage") or {},
            "elapsed_seconds": time.monotonic() - started,
        }


def run_parallel_shadow_critics(
    *,
    jobs: list[tuple[str, ShadowCriticEvidence]],
    api_keys: list[str],
    base_url: str,
    model: str,
) -> list[dict[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if len(jobs) != len(api_keys):
        raise ValueError("one in-memory API key is required per critic job")

    def run(index: int) -> dict[str, Any]:
        case_uid, evidence = jobs[index]
        response = OpenAICompatibleShadowCriticClient(
            api_key=api_keys[index], base_url=base_url, model=model
        ).complete(evidence)
        response["case_uid"] = case_uid
        response["credential_slot"] = index
        return response

    results = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {executor.submit(run, index): index for index in range(len(jobs))}
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: int(item["credential_slot"]))
