from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from .client import OpenAICompatibleChatVLM, VLMRequestError
from .evidence import EndpointEvidence, EvidenceError, sha256_file
from .policy import (
    assess_execution,
    validate_against_evidence,
    validate_audit_against_evidence,
)
from .prompts import (
    audit_system_prompt,
    diagnosis_user_prompt,
    diagnosis_system_prompt,
    verification_system_prompt,
    verification_user_prompt,
)
from .schemas import (
    Diagnosis,
    EvidenceAudit,
    SchemaError,
    Verification,
    extract_json_object,
)


ParsedT = TypeVar("ParsedT")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_worklist(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        # Only this allowlist can influence inference. Human annotation fields, even if
        # later added to the worklist, are deliberately discarded here.
        projected = {
            key: row.get(key)
            for key in (
                "scene_id",
                "case_uid",
                "incident_uid",
                "representative_finding_uid",
                "case_dir",
            )
        }
        if not projected.get("case_dir"):
            raise ValueError(f"worklist line {line_number} has no case_dir")
        rows.append(projected)
    return rows


def _result_path(output_root: Path, evidence: EndpointEvidence) -> Path:
    safe_case = evidence.case_uid.replace("/", "_")
    return output_root / "cases" / evidence.scene_id / f"{safe_case}.json"


def _selected_asset_records(images: Iterable[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": image.path.name,
            "role": image.role,
            "sha256": sha256_file(image.path),
            "detail": image.detail,
        }
        for image in images
    ]


def _response_record(response: Any) -> dict[str, Any]:
    return {
        "model": response.model,
        "response_id": response.response_id,
        "usage": response.usage,
        "elapsed_seconds": round(response.elapsed_seconds, 6),
        "raw_text": response.text,
    }


def _complete_validated(
    *,
    client: OpenAICompatibleChatVLM,
    system_prompt: str,
    user_prompt: str,
    images: list[Any],
    parse_and_validate: Callable[[str], ParsedT],
) -> tuple[ParsedT, list[Any]]:
    """Call once for judgment and at most once more for schema-only correction."""

    first = client.complete(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=images,
    )
    attempts = [first]
    try:
        return parse_and_validate(first.text), attempts
    except SchemaError as exc:
        correction_system = (
            system_prompt
            + "\n\nFORMAT CORRECTION: Your previous answer failed the machine safety schema. "
            "Re-inspect the same images and return a complete corrected JSON object only. "
            "Preserve the evidence-grounded judgment unless the stated schema conflict makes "
            "that judgment internally inconsistent."
        )
        correction_user = (
            user_prompt
            + "\n\nPrevious invalid JSON:\n"
            + first.text
            + "\n\nValidation error:\n"
            + f"{type(exc).__name__}: {exc}"
        )
        corrected = client.complete(
            system_prompt=correction_system,
            user_prompt=correction_user,
            images=images,
        )
        attempts.append(corrected)
        return parse_and_validate(corrected.text), attempts


def run_case(
    *,
    case_dir: Path,
    audit_client: OpenAICompatibleChatVLM,
    decision_client: OpenAICompatibleChatVLM,
    verification_client: OpenAICompatibleChatVLM,
    max_images: int,
    verify_mutations: bool,
) -> dict[str, Any]:
    evidence = EndpointEvidence.load(case_dir)
    images = evidence.select_images(max_images=max_images)
    summary = evidence.summary_text(images)
    audit_prompt = audit_system_prompt()
    def parse_audit(text: str) -> EvidenceAudit:
        parsed = EvidenceAudit.from_mapping(extract_json_object(text))
        validate_audit_against_evidence(evidence, parsed)
        return parsed

    audit, audit_responses = _complete_validated(
        client=audit_client,
        system_prompt=audit_prompt,
        user_prompt=summary,
        images=images,
        parse_and_validate=parse_audit,
    )

    system_prompt = diagnosis_system_prompt()
    decision_user = diagnosis_user_prompt(summary, audit)
    prompt_fingerprint = evidence.fingerprint(
        audit_prompt + "\n" + summary + "\n" + system_prompt + "\n" + decision_user,
        images,
    )
    def parse_diagnosis(text: str) -> Diagnosis:
        parsed = Diagnosis.from_mapping(extract_json_object(text))
        validate_against_evidence(evidence, parsed)
        return parsed

    diagnosis, decision_responses = _complete_validated(
        client=decision_client,
        system_prompt=system_prompt,
        user_prompt=decision_user,
        images=images,
        parse_and_validate=parse_diagnosis,
    )

    verification: Verification | None = None
    verification_response = None
    if verify_mutations and diagnosis.repair.action not in {"KEEP", "ABSTAIN"}:
        verification, verification_responses = _complete_validated(
            client=verification_client,
            system_prompt=verification_system_prompt(),
            user_prompt=verification_user_prompt(summary, audit, diagnosis),
            images=images,
            parse_and_validate=lambda text: Verification.from_mapping(
                extract_json_object(text)
            ),
        )
        verification_response = verification_responses[-1]

    execution = assess_execution(evidence, diagnosis, verification)
    audit_record = {
        **_response_record(audit_responses[-1]),
        "parsed": audit.as_dict(),
        "format_retry_count": len(audit_responses) - 1,
        "attempts": [_response_record(response) for response in audit_responses],
    }
    response_record = {
        **_response_record(decision_responses[-1]),
        "format_retry_count": len(decision_responses) - 1,
        "attempts": [_response_record(response) for response in decision_responses],
    }
    verifier_record = None
    if verification_response is not None:
        verifier_record = {
            **_response_record(verification_response),
            "parsed": verification.as_dict() if verification else None,
            "format_retry_count": len(verification_responses) - 1,
            "attempts": [
                _response_record(response) for response in verification_responses
            ],
        }
    return {
        "schema_version": "1.1.0",
        "method": "ali-my-VLM-only-repair-v1",
        "created_at": _utc_now(),
        "scene_id": evidence.scene_id,
        "case_uid": evidence.case_uid,
        "case_dir": str(evidence.case_dir),
        "target_alias": evidence.target_alias,
        "target_object_uid": evidence.target_uid,
        "labels_used_for_inference": False,
        "prompt_fingerprint": prompt_fingerprint,
        "selected_assets": _selected_asset_records(images),
        "audit": audit.as_dict(),
        "diagnosis": diagnosis.as_dict(),
        "audit_pass": audit_record,
        "decision_pass": response_record,
        "verification": verifier_record,
        "execution": execution.as_dict(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run label-blind VLM diagnosis and conservative repair planning."
    )
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worklist", type=Path)
    parser.add_argument("--base-url", default=os.environ.get("VLM_API_BASE_URL", "https://api.pinaic.com/v1"))
    parser.add_argument(
        "--model",
        help="Optional single-model override for all roles.",
    )
    parser.add_argument("--audit-model")
    parser.add_argument("--decision-model")
    parser.add_argument("--verifier-model")
    parser.add_argument("--api-key-env", default="VLM_API_KEY")
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--case-uid", action="append", default=[])
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--request-delay", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-completion-tokens", type=int, default=1800)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validation_root = args.validation_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    worklist = (args.worklist or validation_root / "labels" / "r1_worklist.jsonl").resolve()
    if not worklist.is_file():
        raise SystemExit(f"worklist not found: {worklist}")
    if output_root == validation_root or validation_root in output_root.parents:
        raise SystemExit("output-root must be outside the frozen validation root")
    output_root.mkdir(parents=True, exist_ok=True)

    rows = _load_worklist(worklist)
    requested_scenes = set(args.scene)
    requested_cases = set(args.case_uid)
    if requested_scenes:
        rows = [row for row in rows if row.get("scene_id") in requested_scenes]
    if requested_cases:
        rows = [
            row
            for row in rows
            if row.get("case_uid") in requested_cases or row.get("incident_uid") in requested_cases
        ]
    if args.limit is not None:
        rows = rows[: args.limit]

    audit_model = args.audit_model or args.model or "gpt-5.6-terra"
    decision_model = args.decision_model or args.model or "gpt-5.6-sol"
    verifier_model = args.verifier_model or args.model or "gpt-5.5"
    clients: dict[str, OpenAICompatibleChatVLM] = {}

    def client_for(model: str) -> OpenAICompatibleChatVLM:
        if model not in clients:
            clients[model] = OpenAICompatibleChatVLM(
                base_url=args.base_url,
                model=model,
                api_key_env=args.api_key_env,
                timeout_seconds=args.timeout,
                max_retries=args.max_retries,
                max_completion_tokens=args.max_completion_tokens,
            )
        return clients[model]

    audit_client = client_for(audit_model)
    decision_client = client_for(decision_model)
    verification_client = client_for(verifier_model)
    started = _utc_now()
    outcomes: Counter[str] = Counter()
    result_paths: list[str] = []
    errors: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        case_dir = Path(str(row["case_dir"]))
        try:
            evidence = EndpointEvidence.load(case_dir)
            result_path = _result_path(output_root, evidence)
            if result_path.exists() and not args.overwrite:
                outcomes["SKIPPED_EXISTING"] += 1
                result_paths.append(str(result_path))
                continue
            result = run_case(
                case_dir=case_dir,
                audit_client=audit_client,
                decision_client=decision_client,
                verification_client=verification_client,
                max_images=args.max_images,
                verify_mutations=not args.no_verify,
            )
            _json_dump(result_path, result)
            result_paths.append(str(result_path))
            outcomes[result["execution"]["status"]] += 1
            print(
                f"[{index}/{len(rows)}] {result['scene_id']} {result['case_uid']} "
                f"{result['diagnosis']['final_state']} "
                f"{result['diagnosis']['error_type']} "
                f"{result['execution']['status']}",
                flush=True,
            )
        except (EvidenceError, SchemaError, VLMRequestError, OSError, ValueError) as exc:
            outcomes["ERROR"] += 1
            safe_error = f"{type(exc).__name__}: {exc}"
            errors.append({"case_dir": str(case_dir), "error": safe_error})
            print(f"[{index}/{len(rows)}] ERROR {case_dir}: {safe_error}", file=sys.stderr, flush=True)
        if args.request_delay > 0 and index < len(rows):
            time.sleep(args.request_delay)

    manifest = {
        "schema_version": "1.1.0",
        "method": "ali-my-VLM-only-repair-v1",
        "started_at": started,
        "completed_at": _utc_now(),
        "validation_root": str(validation_root),
        "worklist": str(worklist),
        "worklist_sha256": sha256_file(worklist),
        "output_root": str(output_root),
        "model": decision_model,
        "models": {
            "audit": audit_model,
            "decision": decision_model,
            "verification": verifier_model,
        },
        "base_url": args.base_url,
        "api_key_env_name": args.api_key_env,
        "api_key_persisted": False,
        "labels_used_for_inference": False,
        "audit_prompt_sha256": hashlib.sha256(
            audit_system_prompt().encode("utf-8")
        ).hexdigest(),
        "diagnosis_prompt_sha256": hashlib.sha256(
            diagnosis_system_prompt().encode("utf-8")
        ).hexdigest(),
        "verification_prompt_sha256": hashlib.sha256(
            verification_system_prompt().encode("utf-8")
        ).hexdigest(),
        "selected_case_count": len(rows),
        "outcomes": dict(sorted(outcomes.items())),
        "result_paths": result_paths,
        "errors": errors,
    }
    _json_dump(output_root / "run_manifest.json", manifest)
    print(json.dumps(manifest["outcomes"], sort_keys=True), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
