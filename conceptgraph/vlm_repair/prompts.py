from __future__ import annotations

import json
from pathlib import Path

from .schemas import Diagnosis, EvidenceAudit


PROMPT_DIR = Path(__file__).with_name("prompts")


def load_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"missing prompt: {path}")
    return path.read_text(encoding="utf-8").strip()


def audit_system_prompt() -> str:
    return load_prompt("audit_v1.txt")


def diagnosis_system_prompt() -> str:
    return load_prompt("diagnose_v1.txt")


def verification_system_prompt() -> str:
    return load_prompt("verify_v1.txt")


def diagnosis_user_prompt(summary: str, audit: EvidenceAudit) -> str:
    return (
        summary
        + "\n\nFirst-stage forced audit (a fallible structured observation, not a final answer):\n"
        + json.dumps(audit.as_dict(), ensure_ascii=False, sort_keys=True)
        + "\n\nRe-inspect the images, challenge the audit where needed, and make the terminal decision."
    )


def verification_user_prompt(
    summary: str, audit: EvidenceAudit, proposal: Diagnosis
) -> str:
    return (
        summary
        + "\n\nFirst-stage forced audit to challenge independently:\n"
        + json.dumps(audit.as_dict(), ensure_ascii=False, sort_keys=True)
        + "\n\nTerminal proposal to verify independently:\n"
        + json.dumps(proposal.as_dict(), ensure_ascii=False, sort_keys=True)
    )
