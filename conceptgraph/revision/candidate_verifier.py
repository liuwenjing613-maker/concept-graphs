"""Continuous candidate evidence scores for selective revision.

Runtime validity remains a separate hard contract. This module deliberately
emits continuous statistics and diagnostics, never a stack of semantic booleans.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .auto_constraints import forbidden_inference_paths
from .counterfactual_projection import (
    CMVIC_STATISTIC_NAME,
    CMVICResult,
)
from .evidence_split import EvidenceSplitManifest


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _uid(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return prefix + digest[:20]


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


@dataclass(frozen=True)
class CandidateEvidenceScore:
    incident_uid: str
    candidate_uid: str
    capability: str
    primary_statistic: str
    primary_score: float
    score_advantage_over_noop: float
    vlm_pairwise_preference: float | None
    valid: bool
    verification_observation_count: int
    evidence_policy_uid: str
    diagnostics: dict[str, Any]
    score_uid: str

    @classmethod
    def build(
        cls,
        *,
        incident_uid: str,
        candidate_uid: str,
        capability: str,
        primary_statistic: str,
        primary_score: float,
        noop_primary_score: float,
        valid: bool,
        verification_observation_count: int,
        diagnostics: Mapping[str, Any],
        vlm_pairwise_preference: float | None = None,
        evidence_policy_uid: str | None = None,
    ) -> "CandidateEvidenceScore":
        primary = float(primary_score)
        noop = float(noop_primary_score)
        if not math.isfinite(primary) or not math.isfinite(noop):
            raise ValueError("candidate and NO-OP scores must be finite")
        if vlm_pairwise_preference is not None:
            vlm_pairwise_preference = float(vlm_pairwise_preference)
            if not -1.0 <= vlm_pairwise_preference <= 1.0:
                raise ValueError("vlm_pairwise_preference must be in [-1, 1]")
        payload = {
            "incident_uid": str(incident_uid),
            "candidate_uid": str(candidate_uid),
            "capability": str(capability).upper(),
            "primary_statistic": str(primary_statistic),
            "primary_score": primary,
            "score_advantage_over_noop": primary - noop,
            "vlm_pairwise_preference": vlm_pairwise_preference,
            "valid": bool(valid),
            "verification_observation_count": int(verification_observation_count),
            "evidence_policy_uid": str(
                evidence_policy_uid or "LEGACY_UNDECLARED_EVIDENCE_POLICY"
            ),
            "diagnostics": dict(diagnostics),
        }
        forbidden = forbidden_inference_paths(payload)
        if forbidden:
            raise ValueError("oracle-like verifier fields: " + ", ".join(forbidden))
        return cls(**payload, score_uid=_uid("candidate_score_", payload))

    def feature_vector(self, feature_names: Sequence[str]) -> tuple[float, ...]:
        values = {
            "primary_advantage": self.score_advantage_over_noop,
            "vlm_pairwise_preference": self.vlm_pairwise_preference,
        }
        result = []
        for name in feature_names:
            value = values.get(str(name))
            if value is None:
                raise ValueError(f"missing calibrated feature: {name}")
            result.append(float(value))
        return tuple(result)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawPrimaryScore:
    score: float
    observation_count: int
    diagnostics: dict[str, Any]


class HeldOutAssignmentLikelihood:
    """Score actual shadow assignments on frozen future observations.

    Mapper likelihood is retained as a diagnostic primary statistic, not
    assumed to be independent truth. A native error can remain more self-likely
    than its correct repair; calibration or critic evidence must expose that.
    """

    statistic_name = "HELD_OUT_APPLIED_ASSIGNMENT_LOG_LIKELIHOOD"

    def __init__(self, temperature: float = 0.1) -> None:
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        self.temperature = float(temperature)

    def score(
        self,
        *,
        state: Mapping[str, Any],
        split: EvidenceSplitManifest,
    ) -> RawPrimaryScore:
        wanted = set(split.verification_obs_uids)
        decisions = {
            str(row.get("obs_uid")): row
            for row in state.get("decision_trace") or ()
            if str(row.get("obs_uid")) in wanted
        }
        log_probabilities: list[float] = []
        score_minus_threshold: list[float] = []
        natural_agreement = 0
        overridden = 0
        missing = []
        applied_outside_top_candidates = []
        for obs_uid in split.verification_obs_uids:
            decision = decisions.get(obs_uid)
            if decision is None:
                missing.append(obs_uid)
                continue
            candidates = [
                row
                for row in decision.get("natural_candidates") or ()
                if row.get("score") is not None
            ]
            threshold = (decision.get("threshold_semantics") or {}).get("sim_threshold")
            if threshold is None:
                missing.append(obs_uid)
                continue
            threshold = float(threshold)
            applied = decision.get("applied_match")
            if applied is None:
                selected_score = threshold
            else:
                selected_score = next(
                    (
                        float(row["score"])
                        for row in candidates
                        if int(row.get("index")) == int(applied)
                    ),
                    None,
                )
                if selected_score is None:
                    applied_outside_top_candidates.append(obs_uid)
                    continue
            alternatives = [float(row["score"]) for row in candidates] + [threshold]
            scaled = [value / self.temperature for value in alternatives]
            selected_scaled = selected_score / self.temperature
            log_probabilities.append(selected_scaled - _logsumexp(scaled))
            score_minus_threshold.append(selected_score - threshold)
            natural_agreement += int(decision.get("natural_match") == applied)
            overridden += int(bool(decision.get("constraint_overrode_natural")))
        if not log_probabilities:
            raise ValueError("no scoreable held-out assignment observations")
        count = len(log_probabilities)
        diagnostics = {
            "temperature": self.temperature,
            "requested_observation_count": len(wanted),
            "scored_observation_count": count,
            "missing_observation_uids": sorted(missing),
            "applied_outside_top_candidates": sorted(applied_outside_top_candidates),
            "mean_applied_score_minus_threshold": sum(score_minus_threshold) / count,
            "natural_agreement_rate": natural_agreement / count,
            "constraint_override_rate": overridden / count,
            "semantic_threshold_count": 0,
        }
        return RawPrimaryScore(
            score=sum(log_probabilities) / count,
            observation_count=count,
            diagnostics=diagnostics,
        )


class CandidateVerifier:
    """Build one continuous candidate score relative to explicit NO-OP."""

    def __init__(self, identity_scorer: HeldOutAssignmentLikelihood | None = None):
        self.identity_scorer = identity_scorer or HeldOutAssignmentLikelihood()

    def score_identity(
        self,
        *,
        incident_uid: str,
        candidate_uid: str,
        candidate_state: Mapping[str, Any],
        noop_state: Mapping[str, Any],
        split: EvidenceSplitManifest,
        runtime_valid: bool,
        vlm_pairwise_preference: float | None = None,
        primary_scorer: str = "ASSIGNMENT_LIKELIHOOD",
        candidate_cmvic: CMVICResult | None = None,
        noop_cmvic: CMVICResult | None = None,
    ) -> CandidateEvidenceScore:
        if not split.verification_available:
            raise ValueError("independent verification evidence is unavailable")
        assignment_noop = self.identity_scorer.score(state=noop_state, split=split)
        assignment_candidate = self.identity_scorer.score(
            state=candidate_state, split=split
        )
        scorer = str(primary_scorer).strip().upper()
        if scorer in {"ASSIGNMENT", "ASSIGNMENT_LIKELIHOOD"}:
            primary_statistic = self.identity_scorer.statistic_name
            primary_score = assignment_candidate.score
            noop_primary_score = assignment_noop.score
            observation_count = assignment_candidate.observation_count
            evidence_policy_uid = split.manifest_uid
            primary_diagnostics = {
                "noop": assignment_noop.diagnostics,
                "candidate": assignment_candidate.diagnostics,
            }
        elif scorer == "CMVIC":
            if candidate_cmvic is None or noop_cmvic is None:
                raise ValueError("CMVIC primary scoring requires both state results")
            if candidate_cmvic.evidence_policy_uid != noop_cmvic.evidence_policy_uid:
                raise ValueError("candidate and NO-OP CMVIC evidence policies differ")
            primary_statistic = CMVIC_STATISTIC_NAME
            primary_score = candidate_cmvic.score
            noop_primary_score = noop_cmvic.score
            observation_count = len(candidate_cmvic.frame_results)
            evidence_policy_uid = candidate_cmvic.evidence_policy_uid
            primary_diagnostics = {
                "counterfactual_observable": candidate_cmvic.observable,
                "projected_difference_pixel_count": (
                    candidate_cmvic.projected_difference_pixel_count
                ),
                "noop_score_uid": noop_cmvic.score_uid,
                "candidate_score_uid": candidate_cmvic.score_uid,
                "frame_count": observation_count,
            }
        else:
            raise ValueError(f"unsupported identity primary scorer: {primary_scorer}")
        diagnostics = {
            **primary_diagnostics,
            "evidence_split_uid": split.manifest_uid,
            "evidence_sequestered": True,
            "mapper_likelihood_is_not_independent_truth": True,
            "diagnostic_assignment_likelihood": {
                "noop_score": assignment_noop.score,
                "candidate_score": assignment_candidate.score,
                "advantage": (assignment_candidate.score - assignment_noop.score),
                "noop": assignment_noop.diagnostics,
                "candidate": assignment_candidate.diagnostics,
            },
        }
        return CandidateEvidenceScore.build(
            incident_uid=incident_uid,
            candidate_uid=candidate_uid,
            capability="IDENTITY",
            primary_statistic=primary_statistic,
            primary_score=primary_score,
            noop_primary_score=noop_primary_score,
            valid=runtime_valid,
            verification_observation_count=observation_count,
            diagnostics=diagnostics,
            vlm_pairwise_preference=vlm_pairwise_preference,
            evidence_policy_uid=evidence_policy_uid,
        )


def critic_preference_value(
    *, preferred_state: str, candidate_state: str, noop_state: str
) -> float:
    preferred = str(preferred_state).strip().upper()
    candidate = str(candidate_state).strip().upper()
    noop = str(noop_state).strip().upper()
    if preferred == "DEFER":
        return 0.0
    if preferred == candidate:
        return 1.0
    if preferred == noop:
        return -1.0
    raise ValueError(f"critic preferred unknown state: {preferred_state}")
