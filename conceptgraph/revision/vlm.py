from __future__ import annotations

import base64
import gzip
import hashlib
import json
import mimetypes
import pickle
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .cases import canonical_obs_key
from .index import ProvenanceIndex


ALLOWED_ACTIONS = {
    "SAME_INSTANCE",
    "SEPARATE_MEMBER_GROUPS",
    "MOVE_OBSERVATION",
    "RELABEL",
    "RESTORE_OBSERVATION_GEOMETRY",
    "PARTITION_OBSERVATION",
    "DEFER",
}


def _single_json(text: str) -> dict[str, Any]:
    value = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", value, flags=re.DOTALL)
    if fenced:
        value = fenced.group(1)
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("VLM response must be one JSON object")
    action = str(decoded.get("action", "")).upper()
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"unsupported VLM constraint action: {action}")
    confidence = float(decoded.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    decoded["action"] = action
    decoded["confidence"] = confidence
    return decoded


def _resolve_ref(provenance: ProvenanceIndex, ref: Mapping[str, Any]) -> Path:
    path = Path(str(ref["path"]))
    if not path.is_absolute():
        path = provenance.experiment_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_crop(provenance: ProvenanceIndex, obs_uid: str):
    from PIL import Image

    row = provenance.get_observation(obs_uid)
    ref = row.get("crop_ref")
    if not ref:
        raise FileNotFoundError(f"no crop reference for {obs_uid}")
    path = _resolve_ref(provenance, ref)
    if path.name.endswith(".pkl.gz"):
        with gzip.open(path, "rb") as handle:
            value = pickle.load(handle)
        index = ref.get("index")
        if index is not None:
            value = value[int(index)]
    else:
        value = Image.open(path)
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    array = np.asarray(value)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _representatives(members: Iterable[str], limit: int = 2) -> list[str]:
    values = list(dict.fromkeys(str(item) for item in members))
    if len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, limit, dtype=int)
    return [values[int(index)] for index in indices]


@dataclass(frozen=True)
class VLMIncidentEvidence:
    incident_uid: str
    prompt: str
    image_paths: tuple[Path, ...]
    image_manifest: tuple[dict[str, Any], ...]
    system_prompt: str = (
        "You generate conservative, machine-executable repair constraints for "
        "a 3D scene graph. Never use semantic class alone as proof of physical "
        "identity. Prefer DEFER over an unsupported mutation."
    )


class VLMIncidentBuilder:
    """Build inference evidence without final membership or oracle labels."""

    def __init__(self, provenance: ProvenanceIndex) -> None:
        self.provenance = provenance

    def _version_members(self, version_uid: str | None) -> tuple[str, ...]:
        if not version_uid or version_uid not in self.provenance.object_versions:
            return ()
        return self.provenance.get_member_observations(version_uid)

    def build(
        self, case: Mapping[str, Any], output_dir: str | Path
    ) -> VLMIncidentEvidence:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        association = self.provenance.get_event(
            str(case["anchor_association_event_uid"])
        )
        anchor_obs = str(association["obs_uid"])
        candidate_versions = list(
            association.get("candidate_object_version_uids") or ()
        )
        objects_before = list(association.get("object_uids_before") or ())
        version_by_object = {
            str(object_uid): str(candidate_versions[index])
            for index, object_uid in enumerate(objects_before)
            if index < len(candidate_versions)
        }

        contexts: list[tuple[str, list[str]]] = [("ANCHOR", [anchor_obs])]
        current_uid = case.get("target_object_uid")
        current_version = case.get(
            "target_object_version_uid"
        ) or version_by_object.get(str(current_uid))
        if current_version:
            contexts.append(
                (
                    "CURRENT_ENTITY_CONTEXT",
                    _representatives(self._version_members(str(current_version))),
                )
            )
        observed_current_decision = case.get("observed_current_decision")
        if observed_current_decision is None:
            corruption_plan = case.get("corruption_plan") or {}
            if corruption_plan.get("corruption_type") == "FORCE_CREATE":
                observed_current_decision = "CREATE"
            elif str(association.get("decision", "")).upper() == "CREATE_OBJECT":
                observed_current_decision = "CREATE"
            else:
                observed_current_decision = "ASSOCIATE"
        observed_current_decision = str(observed_current_decision).upper()
        if observed_current_decision not in {"CREATE", "ASSOCIATE"}:
            raise ValueError("observed_current_decision must be CREATE or ASSOCIATE")
        for rank, candidate in enumerate(association.get("top_candidates") or (), 1):
            object_uid = str(candidate.get("object_uid", ""))
            version_uid = version_by_object.get(object_uid)
            members = _representatives(self._version_members(version_uid))
            if members:
                contexts.append((f"CANDIDATE_{rank}_CONTEXT", members))
            if rank >= 2:
                break

        image_paths: list[Path] = []
        manifest: list[dict[str, Any]] = []
        seen = set()
        for alias, members in contexts:
            for obs_uid in members:
                if obs_uid in seen:
                    continue
                seen.add(obs_uid)
                image_id = f"I{len(image_paths) + 1:02d}"
                path = output / f"{image_id}_{canonical_obs_key(obs_uid)}.jpg"
                _load_crop(self.provenance, obs_uid).save(path, quality=92)
                row = self.provenance.get_observation(obs_uid)
                image_paths.append(path)
                manifest.append(
                    {
                        "image_id": image_id,
                        "context_alias": alias,
                        "obs_key": canonical_obs_key(obs_uid),
                        "class_name": row.get("class_name"),
                        "confidence": row.get("confidence"),
                    }
                )
        incident_uid = (
            "incident_" + hashlib.sha256(anchor_obs.encode()).hexdigest()[:12]
        )
        prompt_payload = {
            "incident_uid": incident_uid,
            "anchor_obs_key": canonical_obs_key(anchor_obs),
            "anchor_class_name": self.provenance.get_observation(anchor_obs).get(
                "class_name"
            ),
            "observed_current_decision": observed_current_decision,
            "candidate_scores": [
                {
                    "alias": f"CANDIDATE_{rank}_CONTEXT",
                    "spatial": item.get("spatial_score"),
                    "visual": item.get("visual_score"),
                    "aggregate": item.get("aggregate_score"),
                }
                for rank, item in enumerate(association.get("top_candidates") or (), 1)
            ][:2],
            "images": manifest,
        }
        prompt = (
            "Review this 3D-mapping identity incident using only the supplied crops and raw "
            "association evidence. No ground-truth membership, failure label, or oracle answer is "
            "provided. Decide the safest typed structural constraint. SAME_INSTANCE means the "
            "anchor and one named context are the same physical instance. SEPARATE_MEMBER_GROUPS "
            "means named contexts must remain distinct. MOVE_OBSERVATION moves only the anchor "
            "from CURRENT_ENTITY_CONTEXT to one candidate. Use DEFER whenever crops are ambiguous. "
            "Return exactly one JSON object with keys: action, confidence, entities, groups, "
            "obs_key, from_alias, to_alias, evidence_image_ids, reason. Do not invent observation "
            "keys or aliases.\n\nINCIDENT:\n"
            + json.dumps(prompt_payload, indent=2, sort_keys=True)
        )
        return VLMIncidentEvidence(
            incident_uid=incident_uid,
            prompt=prompt,
            image_paths=tuple(image_paths),
            image_manifest=tuple(manifest),
        )


class OpenAICompatibleConstraintClient:
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
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _data_uri(path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode(
            "ascii"
        )

    def complete(self, evidence: VLMIncidentEvidence) -> dict[str, Any]:
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
                {
                    "role": "system",
                    "content": evidence.system_prompt,
                },
                {"role": "user", "content": content},
            ],
            "max_completion_tokens": 1200,
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
                    "User-Agent": "ali-my-revision-kernel/0.1",
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
            raise last_error or RuntimeError("VLM request failed")
        choices = decoded.get("choices") or []
        text = choices[0].get("message", {}).get("content") if choices else None
        if not isinstance(text, str):
            raise RuntimeError("VLM returned no text content")
        constraint = _single_json(text)
        return {
            "incident_uid": evidence.incident_uid,
            "constraint": constraint,
            "model": str(decoded.get("model") or self.model),
            "response_id": decoded.get("id"),
            "usage": decoded.get("usage") or {},
            "elapsed_seconds": time.monotonic() - started,
        }


def run_parallel_votes(
    *,
    jobs: list[tuple[str, VLMIncidentEvidence]],
    api_keys: list[str],
    base_url: str,
    model: str,
) -> list[dict[str, Any]]:
    if len(jobs) != len(api_keys):
        raise ValueError("one in-memory API key is required per parallel vote")

    def run(index: int) -> dict[str, Any]:
        case_uid, evidence = jobs[index]
        response = OpenAICompatibleConstraintClient(
            api_key=api_keys[index], base_url=base_url, model=model
        ).complete(evidence)
        response["case_uid"] = case_uid
        response["vote_index"] = index
        return response

    results = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {executor.submit(run, index): index for index in range(len(jobs))}
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: int(item["vote_index"]))


def aggregate_votes(votes: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for vote in votes:
        grouped.setdefault(str(vote["case_uid"]), []).append(vote)
    result = {}
    for case_uid, rows in grouped.items():
        counts = Counter(str(row["constraint"]["action"]) for row in rows)
        action = counts.most_common(1)[0][0]
        supporting = [row for row in rows if row["constraint"]["action"] == action]
        result[case_uid] = {
            "action": action,
            "vote_counts": dict(counts),
            "mean_supporting_confidence": sum(
                float(row["constraint"]["confidence"]) for row in supporting
            )
            / len(supporting),
            "vote_count": len(rows),
        }
    return result


def normalize_incident_constraint(
    constraint: Mapping[str, Any], *, observed_current_decision: str
) -> dict[str, Any]:
    """Compile observation-to-context identity statements into executable types."""
    normalized = dict(constraint)
    normalized["raw_action"] = str(constraint["action"])
    entities = [str(item) for item in constraint.get("entities") or ()]
    candidate_aliases = sorted(
        item
        for item in entities
        if item.startswith("CANDIDATE_") and item.endswith("_CONTEXT")
    )
    if (
        str(constraint["action"]) == "SAME_INSTANCE"
        and "ANCHOR" in entities
        and candidate_aliases
        and observed_current_decision == "ASSOCIATE"
    ):
        normalized.update(
            {
                "action": "MOVE_OBSERVATION",
                "obs_key": constraint.get("obs_key"),
                "from_alias": "CURRENT_ENTITY_CONTEXT",
                "to_alias": candidate_aliases[0],
                "normalization": (
                    "ANCHOR same-as alternate candidate while currently associated "
                    "compiles to MOVE_OBSERVATION"
                ),
            }
        )
    else:
        normalized["normalization"] = "identity"
    return normalized
