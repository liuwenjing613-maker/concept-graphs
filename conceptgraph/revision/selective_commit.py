"""Single-threshold calibrated selective commit for revision candidates."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .auto_constraints import forbidden_inference_paths
from .candidate_verifier import CandidateEvidenceScore


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _uid(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return prefix + digest[:20]


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


@dataclass(frozen=True)
class CalibrationArtifact:
    calibration_uid: str
    capability: str
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    commit_threshold: float
    ready_for_automatic_commit: bool
    fit_case_count: int
    fit_positive_count: int
    fit_negative_count: int
    target_harm_rate: float
    source_hashes: dict[str, str]
    primary_statistic: str
    evidence_policy_uid: str
    evidence_schema_declared: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CalibrationArtifact":
        forbidden = forbidden_inference_paths(value)
        if forbidden:
            raise ValueError("oracle-like calibration fields: " + ", ".join(forbidden))
        feature_names = tuple(str(item) for item in value.get("feature_names") or ())
        coefficients = tuple(float(item) for item in value.get("coefficients") or ())
        if not feature_names or len(feature_names) != len(coefficients):
            raise ValueError("calibration features and coefficients must align")
        allowed = {"primary_advantage", "vlm_pairwise_preference"}
        if set(feature_names) - allowed:
            raise ValueError("unsupported calibration feature")
        intercept = float(value.get("intercept"))
        commit_threshold = float(value.get("commit_threshold"))
        target_harm_rate = float(value.get("target_harm_rate"))
        numeric = (*coefficients, intercept, commit_threshold, target_harm_rate)
        if not all(math.isfinite(item) for item in numeric):
            raise ValueError("calibration values must be finite")
        if not 0.0 < commit_threshold < 1.0:
            raise ValueError("commit_threshold must be in (0, 1)")
        if not 0.0 <= target_harm_rate < 1.0:
            raise ValueError("target_harm_rate must be in [0, 1)")
        source_hashes = {
            str(key): str(digest)
            for key, digest in (value.get("source_hashes") or {}).items()
        }
        schema_declared = (
            "primary_statistic" in value and "evidence_policy_uid" in value
        )
        payload = {
            "capability": str(value.get("capability") or "").upper(),
            "feature_names": list(feature_names),
            "coefficients": list(coefficients),
            "intercept": intercept,
            "commit_threshold": commit_threshold,
            "ready_for_automatic_commit": bool(value.get("ready_for_automatic_commit")),
            "fit_case_count": int(value.get("fit_case_count", 0)),
            "fit_positive_count": int(value.get("fit_positive_count", 0)),
            "fit_negative_count": int(value.get("fit_negative_count", 0)),
            "target_harm_rate": target_harm_rate,
            "source_hashes": source_hashes,
        }
        if schema_declared:
            payload["primary_statistic"] = str(value["primary_statistic"])
            payload["evidence_policy_uid"] = str(value["evidence_policy_uid"])
        expected_uid = _uid("calibration_", payload)
        supplied_uid = str(value.get("calibration_uid") or "")
        if supplied_uid and supplied_uid != expected_uid:
            raise ValueError("calibration_uid does not match canonical parameters")
        return cls(
            calibration_uid=expected_uid,
            capability=payload["capability"],
            feature_names=feature_names,
            coefficients=coefficients,
            intercept=intercept,
            commit_threshold=commit_threshold,
            ready_for_automatic_commit=payload["ready_for_automatic_commit"],
            fit_case_count=payload["fit_case_count"],
            fit_positive_count=payload["fit_positive_count"],
            fit_negative_count=payload["fit_negative_count"],
            target_harm_rate=target_harm_rate,
            source_hashes=source_hashes,
            primary_statistic=str(
                value.get("primary_statistic")
                or "HELD_OUT_APPLIED_ASSIGNMENT_LOG_LIKELIHOOD"
            ),
            evidence_policy_uid=str(
                value.get("evidence_policy_uid")
                or "LEGACY_ANY_ASSIGNMENT_EVIDENCE_POLICY"
            ),
            evidence_schema_declared=schema_declared,
        )

    def compatibility_reasons(self, score: CandidateEvidenceScore) -> tuple[str, ...]:
        reasons = []
        if not self.evidence_schema_declared:
            if (
                score.primary_statistic
                == "COUNTERFACTUAL_MULTI_VIEW_INSTANCE_CONSISTENCY"
            ):
                reasons.append("calibration_evidence_schema_undeclared")
            return tuple(reasons)
        if score.primary_statistic != self.primary_statistic:
            reasons.append("calibration_primary_statistic_mismatch")
        if score.evidence_policy_uid != self.evidence_policy_uid:
            reasons.append("calibration_evidence_policy_mismatch")
        return tuple(reasons)

    def predict(self, score: CandidateEvidenceScore) -> float:
        if score.capability != self.capability:
            raise ValueError("candidate capability does not match calibration")
        features = score.feature_vector(self.feature_names)
        logit = self.intercept + sum(
            coefficient * feature
            for coefficient, feature in zip(self.coefficients, features)
        )
        return _sigmoid(logit)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["feature_names"] = list(self.feature_names)
        value["coefficients"] = list(self.coefficients)
        return value


@dataclass(frozen=True)
class SelectiveCandidate:
    score: CandidateEvidenceScore
    candidate_constraint: dict[str, Any]


def decide_selective_commit(
    *,
    incident_uid: str,
    candidates: Iterable[SelectiveCandidate],
    calibration: CalibrationArtifact,
) -> dict[str, Any]:
    """Commit the highest calibrated beneficial repair or abstain.

    `valid` is the only per-candidate hard gate. The sole noisy semantic threshold
    is `calibration.commit_threshold`; exact probability ties abstain because the
    argmax is not unique.
    """

    rows = []
    for candidate in candidates:
        compatibility_reasons = calibration.compatibility_reasons(candidate.score)
        schema_compatible = not compatibility_reasons
        valid = bool(candidate.score.valid and schema_compatible)
        probability = calibration.predict(candidate.score) if valid else 0.0
        rows.append(
            {
                "candidate_uid": candidate.score.candidate_uid,
                "score_uid": candidate.score.score_uid,
                "runtime_valid": candidate.score.valid,
                "valid": valid,
                "schema_compatible": schema_compatible,
                "schema_defer_reasons": list(compatibility_reasons),
                "candidate_primary_statistic": (candidate.score.primary_statistic),
                "candidate_evidence_policy_uid": (candidate.score.evidence_policy_uid),
                "benefit_probability": probability,
                "candidate_constraint": dict(candidate.candidate_constraint),
            }
        )
    reasons = []
    if not calibration.ready_for_automatic_commit:
        reasons.append("calibration_not_ready_for_automatic_commit")
    valid_rows = [row for row in rows if row["valid"]]
    if not valid_rows:
        runtime_valid_rows = [row for row in rows if row["runtime_valid"]]
        if runtime_valid_rows:
            mismatch_reasons = sorted(
                {
                    reason
                    for row in runtime_valid_rows
                    for reason in row["schema_defer_reasons"]
                }
            )
            reasons.extend(mismatch_reasons)
            reasons.append("no_calibration_compatible_candidate")
        else:
            reasons.append("no_runtime_valid_candidate")
        best = None
    else:
        best_probability = max(row["benefit_probability"] for row in valid_rows)
        winners = [
            row for row in valid_rows if row["benefit_probability"] == best_probability
        ]
        best = winners[0] if len(winners) == 1 else None
        if len(winners) != 1:
            reasons.append("non_unique_best_candidate")
    if best is not None and best["benefit_probability"] < calibration.commit_threshold:
        reasons.append("calibrated_confidence_below_commit_threshold")
    commit = bool(best is not None and not reasons)
    result = {
        "schema_version": "1.0.0",
        "incident_uid": str(incident_uid),
        "decision": "COMMIT" if commit else "DEFER",
        "semantic_commit_threshold_count": 1,
        "commit_threshold": calibration.commit_threshold,
        "calibration_uid": calibration.calibration_uid,
        "calibration_primary_statistic": calibration.primary_statistic,
        "calibration_evidence_policy_uid": calibration.evidence_policy_uid,
        "calibration_evidence_schema_declared": calibration.evidence_schema_declared,
        "selected_candidate_uid": best["candidate_uid"] if commit else None,
        "selected_benefit_probability": (
            best["benefit_probability"] if best is not None else None
        ),
        "commit_constraint": best["candidate_constraint"] if commit else None,
        "defer_reasons": reasons,
        "candidate_results": rows,
    }
    forbidden = forbidden_inference_paths(result)
    if forbidden:
        raise ValueError(
            "oracle-like selective decision fields: " + ", ".join(forbidden)
        )
    result["decision_uid"] = _uid("selective_decision_", result)
    return result
