#!/usr/bin/env python3
"""Run plans, structured Bond/Task results and storage contracts.

Store dimensions and categories separately as defined by the method; do not calculate cross-dimension totals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Capture identity at module load, not by rereading disk immediately before a run.
LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class StrictModel(BaseModel):
    """Shared base for canonical structures; undeclared output fields are forbidden."""

    model_config = ConfigDict(extra="forbid")


# Bordin (1979) defines Bond as the relational component; Horvath and Greenberg (1989)
# provide the WAI foundation. The dimensions below organise observable evidence rather than reuse WAI scores.
BondDimension: TypeAlias = Literal[
    "Accurate Empathy",
    "Therapist accepts/validates Patient",
    "Grounded content uptake",
    "Client confirmation/disconfirmation",
    "Client-perceived acceptance",
    "Mutual trust and respect",
]

# The Task core also follows Bordin (1979) and Horvath and Greenberg (1989).
# These are project Contributors/Indicators, not original WAI subscale items.
TaskDimension: TypeAlias = Literal[
    "Collaborative activity negotiation",
    "Activity and responsibility clarification",
    "Activity–client relevance linkage",
    "Activity-related responsiveness",
    "Activity agreement/disagreement",
    "Client engagement in the activity",
    "Client-expressed activity relevance",
    "Client-expressed activity efficacy",
]

BOND_DIMENSIONS = (
    "Accurate Empathy",
    "Therapist accepts/validates Patient",
    "Grounded content uptake",
    "Client confirmation/disconfirmation",
    "Client-perceived acceptance",
    "Mutual trust and respect",
)

TASK_DIMENSIONS = (
    "Collaborative activity negotiation",
    "Activity and responsibility clarification",
    "Activity–client relevance linkage",
    "Activity-related responsiveness",
    "Activity agreement/disagreement",
    "Client engagement in the activity",
    "Client-expressed activity relevance",
    "Client-expressed activity efficacy",
)

# The original observation units determine output levels: Truax accurate empathy and 3RS WT2/WT3/WT5
# first accumulate turn evidence rather than receive a final rating at every turn.
SUMMARY_LEVEL_DIMENSIONS = (
    "Accurate Empathy",
    "Therapist accepts/validates Patient",
    "Mutual trust and respect",
    "Client engagement in the activity",
)

COUNSELLOR_ELIGIBLE_DIMENSIONS = (
    "Accurate Empathy",
    "Therapist accepts/validates Patient",
    "Grounded content uptake",
    "Mutual trust and respect",
    "Collaborative activity negotiation",
    "Activity and responsibility clarification",
    "Activity–client relevance linkage",
    "Activity-related responsiveness",
    "Activity agreement/disagreement",
)

CLIENT_ELIGIBLE_DIMENSIONS = (
    "Client confirmation/disconfirmation",
    "Client-perceived acceptance",
    "Mutual trust and respect",
    "Activity agreement/disagreement",
    "Client engagement in the activity",
    "Client-expressed activity relevance",
    "Client-expressed activity efficacy",
)


class EvidenceSpan(StrictModel):
    """A verifiable evidence span containing only its source turn ID and exact quote."""

    turn_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class ActivityReference(StrictModel):
    """An Activity ID reused consistently within a sample, with a label favouring the source wording."""

    activity_id: str = Field(pattern=r"^ACTIVITY_[0-9]{2,}$")
    activity_label: str = Field(min_length=1)


class ActivityIdentificationLLMOutput(StrictModel):
    """Activity identification returns only newly registered specific therapeutic work in the current turn."""

    new_activities: list[ActivityReference]


class ActivityIdentificationStoredRecord(StrictModel):
    """Activity identification record with sample and current-turn identity bound by the pipeline."""

    sample_id: str = Field(min_length=1)
    current_turn_id: str = Field(min_length=1)
    new_activities: list[ActivityReference]


class SampleSelectionItem(StrictModel):
    """An explicitly selected dialogue or contiguous excerpt in a sample plan."""

    # Do not coerce booleans, strings or floats into turn indices.
    model_config = ConfigDict(extra="forbid", strict=True)

    source_file: str = Field(min_length=1)
    dialogue_id: str = Field(min_length=1)
    start_turn_index: int | None = Field(ge=0)
    end_turn_index: int | None = Field(ge=0)

    @model_validator(mode="after")
    def validate_range(self):
        """Ensure that an optional contiguous excerpt does not end before it starts."""

        if (
            self.start_turn_index is not None
            and self.end_turn_index is not None
            and self.start_turn_index > self.end_turn_index
        ):
            raise ValueError("start_turn_index must not exceed end_turn_index")
        return self


class SamplePlan(StrictModel):
    """A sample plan explicitly identifying selected dialogues and ranges."""

    model_config = ConfigDict(extra="forbid", strict=True)

    sample_version: str = Field(min_length=1)
    items: list[SampleSelectionItem] = Field(min_length=1)


class BondSummaryEvidenceFinding(StrictModel):
    """Current-turn evidence for the three sample-level Bond dimensions."""

    dimension: Literal[
        "Accurate Empathy",
        "Therapist accepts/validates Patient",
        "Mutual trust and respect",
    ]
    activity_reference: None
    finding_status: Literal["evidence_only"]
    rating_label: None
    evidence: list[EvidenceSpan] = Field(min_length=1)
    reason: str = Field(min_length=1)


# Runtime output models restrict dimensions to the current speaker. These subclasses inherit
# the same fields and status logic without redefining ratings.
class CounsellorBondSummaryEvidenceFinding(BondSummaryEvidenceFinding):
    """Evidence available from a counsellor turn for the three summary-level Bond dimensions."""

    dimension: Literal[
        "Accurate Empathy",
        "Therapist accepts/validates Patient",
        "Mutual trust and respect",
    ]


class ClientBondSummaryEvidenceFinding(BondSummaryEvidenceFinding):
    """Dyadic Mutual trust and respect evidence available from a client turn."""

    dimension: Literal["Mutual trust and respect"]


# Turn-level Activity engagement accumulates evidence for summary only.
class ClientEngagementEvidenceFinding(StrictModel):
    """Client-turn WT5 evidence for a specific Activity; ratings are assigned at summary level (Eubanks & Muran, 2022)."""

    dimension: Literal["Client engagement in the activity"]
    activity_reference: ActivityReference
    finding_status: Literal["evidence_only"]
    rating_label: None
    evidence: list[EvidenceSpan] = Field(min_length=1)
    reason: str = Field(min_length=1)


class GroundedRatedFinding(StrictModel):
    """Project turn-level categories for Grounded content uptake.

    The no/mechanical/substantive distinction follows Hughes et al. (2025). The dimension name
    and three textual anchors are project adaptations and do not inherit the source study's validity.
    """

    dimension: Literal["Grounded content uptake"]
    activity_reference: None
    finding_status: Literal["rated"]
    rating_label: Literal[
        "Not grounded / not taken up",
        "Mechanical or surface uptake",
        "Substantive grounded uptake",
    ]
    evidence: list[EvidenceSpan] = Field(min_length=1)
    reason: str = Field(min_length=1)


class ConfirmationRatedFinding(StrictModel):
    """Turn-level interaction events for Client confirmation/disconfirmation.

    Confirmation/disconfirmation sequences follow Elliott et al. (2023). The three
    textual categories are project codes for the available dialogue evidence.
    """

    dimension: Literal["Client confirmation/disconfirmation"]
    activity_reference: None
    finding_status: Literal["rated"]
    rating_label: Literal[
        "Clear confirmation",
        "Partial confirmation",
        "Clear disconfirmation",
    ]
    evidence: list[EvidenceSpan] = Field(min_length=1)
    reason: str = Field(min_length=1)


class PerceivedAcceptanceRatedFinding(StrictModel):
    """Events in which the client explicitly expresses perceived acceptance.

    The theoretical role of client perception follows Rogers (1957/1992) and Barrett-Lennard
    (1962). The three textual categories are project adaptations, not Relationship Inventory scores.
    """

    dimension: Literal["Client-perceived acceptance"]
    activity_reference: None
    finding_status: Literal["rated"]
    rating_label: Literal[
        "Clear perceived acceptance",
        "Mixed/conditional perceived acceptance",
        "Clear perceived non-acceptance",
    ]
    evidence: list[EvidenceSpan] = Field(min_length=1)
    reason: str = Field(min_length=1)


class NegotiationRatedFinding(StrictModel):
    """Counsellor-side negotiation outcomes for a specific Activity.

    Activity agreement and collaboration boundaries follow Bordin (1979), Horvath and Greenberg
    (1989), and Rohde, Feeny and Robins (2005). The three categories are project codes.
    """

    dimension: Literal["Collaborative activity negotiation"]
    activity_reference: ActivityReference
    finding_status: Literal["rated"]
    rating_label: Literal[
        "Counsellor-led without meaningful negotiation",
        "Limited or procedural consultation",
        "Collaborative activity negotiation",
    ]
    evidence: list[EvidenceSpan] = Field(min_length=1)
    reason: str = Field(min_length=1)


class ClarificationRatedFinding(StrictModel):
    """A project-synthesised result for clarity of an Activity and applicable responsibilities.

    This dimension combines Activity and responsibility sources (Bordin, 1979;
    Horvath & Greenberg, 1989; Rohde et al., 2005; Ryum & Kazantzis, 2024).
    Its name and categories are project syntheses, not items from an original scale.
    """

    dimension: Literal["Activity and responsibility clarification"]
    activity_reference: ActivityReference
    finding_status: Literal["rated"]
    rating_label: Literal[
        "Unclear activity or applicable responsibilities",
        "Partially specified activity or responsibilities",
        "Clear activity and applicable responsibilities",
    ]
    evidence: list[EvidenceSpan] = Field(min_length=1)
    reason: str = Field(min_length=1)


class RelevanceLinkageRatedFinding(StrictModel):
    """Whether the counsellor links an Activity to the current client material.

    Relevance is part of the Task core in Bordin (1979) and Horvath and Greenberg (1989).
    The three categories of counsellor-text linkage are project codes.
    """

    dimension: Literal["Activity–client relevance linkage"]
    activity_reference: ActivityReference
    finding_status: Literal["rated"]
    rating_label: Literal[
        "No client-specific linkage",
        "General rationale or broad linkage",
        "Explicit client-specific linkage",
    ]
    evidence: list[EvidenceSpan] = Field(min_length=1)
    reason: str = Field(min_length=1)


class ResponsivenessRatedFinding(StrictModel):
    """Whether the counsellor uses new client feedback about an Activity in subsequent work.

    The ballistic/responsive distinction follows Stiles, Honos-Webb and Surko (1998).
    The three Activity-specific categories are project adaptations bounded by Ryum and Kazantzis (2024).
    """

    dimension: Literal["Activity-related responsiveness"]
    activity_reference: ActivityReference
    finding_status: Literal["rated"]
    rating_label: Literal[
        "Ballistic activity continuation",
        "Feedback acknowledged but not used",
        "Activity-responsive use of feedback",
    ]
    evidence: list[EvidenceSpan] = Field(min_length=1)
    reason: str = Field(min_length=1)


class ActivityAgreementRatedFinding(StrictModel):
    """Events of agreement or disagreement between both parties about a specific Activity or its steps.

    Agreement is an explicit core term in Bordin (1979) and Horvath and Greenberg (1989).
    Clear agreement, partial agreement and clear disagreement are dialogue-text codes.
    """

    dimension: Literal["Activity agreement/disagreement"]
    activity_reference: ActivityReference
    finding_status: Literal["rated"]
    rating_label: Literal[
        "Clear activity disagreement",
        "Partial or conditional activity agreement",
        "Clear activity agreement",
    ]
    evidence: list[EvidenceSpan] = Field(min_length=1)
    reason: str = Field(min_length=1)


class ActivityRelevanceRatedFinding(StrictModel):
    """Events in which the client explicitly expresses the relevance of a specific Activity.

    Relevance follows Horvath and Greenberg (1989). The three event categories are
    project text codes, not WAI scores.
    """

    dimension: Literal["Client-expressed activity relevance"]
    activity_reference: ActivityReference
    finding_status: Literal["rated"]
    rating_label: Literal[
        "Clear client-perceived irrelevance",
        "Mixed or conditional client-perceived relevance",
        "Clear client-perceived relevance",
    ]
    evidence: list[EvidenceSpan] = Field(min_length=1)
    reason: str = Field(min_length=1)


class ActivityEfficacyRatedFinding(StrictModel):
    """Client judgements explicitly expressing the helpfulness of a specific Activity.

    Efficacy/helpfulness follows Horvath and Greenberg (1989). The three event
    categories are project text codes recording client views, not objective clinical effects.
    """

    dimension: Literal["Client-expressed activity efficacy"]
    activity_reference: ActivityReference
    finding_status: Literal["rated"]
    rating_label: Literal[
        "Clear client-perceived inefficacy",
        "Mixed or uncertain client-perceived efficacy",
        "Clear client-perceived efficacy",
    ]
    evidence: list[EvidenceSpan] = Field(min_length=1)
    reason: str = Field(min_length=1)


TurnFinding: TypeAlias = Union[
    GroundedRatedFinding,
    ConfirmationRatedFinding,
    PerceivedAcceptanceRatedFinding,
    NegotiationRatedFinding,
    ClarificationRatedFinding,
    RelevanceLinkageRatedFinding,
    ResponsivenessRatedFinding,
    ActivityAgreementRatedFinding,
    ActivityRelevanceRatedFinding,
    ActivityEfficacyRatedFinding,
    BondSummaryEvidenceFinding,
    ClientEngagementEvidenceFinding,
]

CounsellorTurnFinding: TypeAlias = Union[
    GroundedRatedFinding,
    NegotiationRatedFinding,
    ClarificationRatedFinding,
    RelevanceLinkageRatedFinding,
    ResponsivenessRatedFinding,
    ActivityAgreementRatedFinding,
    CounsellorBondSummaryEvidenceFinding,
]

ClientTurnFinding: TypeAlias = Union[
    ConfirmationRatedFinding,
    PerceivedAcceptanceRatedFinding,
    ActivityAgreementRatedFinding,
    ActivityRelevanceRatedFinding,
    ActivityEfficacyRatedFinding,
    ClientBondSummaryEvidenceFinding,
    ClientEngagementEvidenceFinding,
]

CounsellorBondOnlyFinding: TypeAlias = Union[
    GroundedRatedFinding,
    CounsellorBondSummaryEvidenceFinding,
]

ClientBondOnlyFinding: TypeAlias = Union[
    ConfirmationRatedFinding,
    PerceivedAcceptanceRatedFinding,
    ClientBondSummaryEvidenceFinding,
]


class CounsellorBondOnlyTurnLLMOutput(StrictModel):
    """Bond-only findings for a counsellor turn when no Activity has been registered."""

    findings: list[CounsellorBondOnlyFinding]


class ClientBondOnlyTurnLLMOutput(StrictModel):
    """Bond-only findings for a client turn when no Activity has been registered."""

    findings: list[ClientBondOnlyFinding]


class CounsellorTurnLLMOutput(StrictModel):
    """Speaker-specific structured output for the current counsellor turn."""

    findings: list[CounsellorTurnFinding]


class ClientTurnLLMOutput(StrictModel):
    """Speaker-specific structured output for the current client turn."""

    findings: list[ClientTurnFinding]


class TurnLLMOutput(StrictModel):
    """The permitted model-output envelope for one current turn."""

    findings: list[TurnFinding]


class TurnStoredRecord(StrictModel):
    """Canonical turn record with blinded sample and current-turn identity bound by the pipeline."""

    sample_id: str = Field(min_length=1)
    current_turn_id: str = Field(min_length=1)
    findings: list[TurnFinding]


class AccurateEmpathySummary(StrictModel):
    """Representative Accurate Empathy anchors for the complete sample (abridged adaptation of Truax, 1967)."""

    dimension: Literal["Accurate Empathy"]
    activity_reference: None
    summary_status: Literal["rated"]
    rating_label: Literal["AE-1", "AE-3", "AE-5", "AE-7", "AE-9"]
    evidence: list[EvidenceSpan] = Field(min_length=1)
    counterevidence: list[EvidenceSpan]
    reason: str = Field(min_length=1)


class AcceptanceValidationSummary(StrictModel):
    """WT3 acceptance/validation stance for the complete sample (Eubanks & Muran, 2022)."""

    dimension: Literal["Therapist accepts/validates Patient"]
    activity_reference: None
    summary_status: Literal["rated"]
    rating_label: Literal["Not salient", "Somewhat salient", "Very salient"]
    evidence: list[EvidenceSpan] = Field(min_length=1)
    counterevidence: list[EvidenceSpan]
    reason: str = Field(min_length=1)


class MutualTrustRespectSummary(StrictModel):
    """Dyadic WT2 mutual trust/respect result for the complete sample (Eubanks & Muran, 2022)."""

    dimension: Literal["Mutual trust and respect"]
    activity_reference: None
    summary_status: Literal["rated"]
    rating_label: Literal["Not salient", "Somewhat salient", "Very salient"]
    evidence: list[EvidenceSpan] = Field(min_length=1)
    counterevidence: list[EvidenceSpan]
    reason: str = Field(min_length=1)


class ClientEngagementSummary(StrictModel):
    """Text-based WT5 engagement result for a specific Activity (adapted from Eubanks & Muran, 2022)."""

    dimension: Literal["Client engagement in the activity"]
    activity_reference: ActivityReference
    summary_status: Literal["rated"]
    rating_label: Literal["Not salient", "Somewhat salient", "Very salient"]
    evidence: list[EvidenceSpan] = Field(min_length=1)
    counterevidence: list[EvidenceSpan]
    reason: str = Field(min_length=1)


class BondInsufficientSummary(StrictModel):
    """An evaluable opportunity exists for a Bond summary dimension, but valid evidence remains insufficient."""

    dimension: Literal[
        "Accurate Empathy", "Therapist accepts/validates Patient", "Mutual trust and respect"
    ]
    activity_reference: None
    summary_status: Literal["insufficient_evidence"]
    rating_label: None
    evidence: list[EvidenceSpan]
    counterevidence: list[EvidenceSpan]
    reason: str = Field(min_length=1)


class BondNoOpportunitySummary(StrictModel):
    """No evaluable opportunity exists for a Bond summary dimension in the complete sample."""

    dimension: Literal[
        "Accurate Empathy", "Therapist accepts/validates Patient", "Mutual trust and respect"
    ]
    activity_reference: None
    summary_status: Literal["no_evaluable_opportunity"]
    rating_label: None
    evidence: list[EvidenceSpan] = Field(max_length=0)
    counterevidence: list[EvidenceSpan] = Field(max_length=0)
    reason: str = Field(min_length=1)


class ActivityInsufficientSummary(StrictModel):
    """An engagement opportunity exists for the Activity, but valid evidence remains insufficient."""

    dimension: Literal["Client engagement in the activity"]
    activity_reference: ActivityReference
    summary_status: Literal["insufficient_evidence"]
    rating_label: None
    evidence: list[EvidenceSpan]
    counterevidence: list[EvidenceSpan]
    reason: str = Field(min_length=1)


class ActivityNoOpportunitySummary(StrictModel):
    """No evaluable engagement opportunity exists for the specific Activity."""

    dimension: Literal["Client engagement in the activity"]
    activity_reference: ActivityReference
    summary_status: Literal["no_evaluable_opportunity"]
    rating_label: None
    evidence: list[EvidenceSpan] = Field(max_length=0)
    counterevidence: list[EvidenceSpan] = Field(max_length=0)
    reason: str = Field(min_length=1)


SummaryResult: TypeAlias = Union[
    AccurateEmpathySummary,
    AcceptanceValidationSummary,
    MutualTrustRespectSummary,
    ClientEngagementSummary,
    BondInsufficientSummary,
    BondNoOpportunitySummary,
    ActivityInsufficientSummary,
    ActivityNoOpportunitySummary,
]


class SummaryLLMOutput(StrictModel):
    """The permitted summary-output envelope after the complete input sample."""

    results: list[SummaryResult]


class SummaryStoredRecord(StrictModel):
    """Canonical summary record with blinded sample identity bound by the pipeline."""

    sample_id: str = Field(min_length=1)
    results: list[SummaryResult]


def export_json_schemas(output_dir: Path) -> None:
    """Deterministically export Activity, turn and summary output schemas from Pydantic models."""

    output_dir.mkdir(parents=True, exist_ok=True)
    schemas = {
        "activity_identification.schema.json": (
            ActivityIdentificationLLMOutput.model_json_schema()
        ),
        "evaluation_turn_counsellor.schema.json": (
            CounsellorTurnLLMOutput.model_json_schema()
        ),
        "evaluation_turn_client.schema.json": (
            ClientTurnLLMOutput.model_json_schema()
        ),
        "evaluation_turn_counsellor_no_activity.schema.json": (
            CounsellorBondOnlyTurnLLMOutput.model_json_schema()
        ),
        "evaluation_turn_client_no_activity.schema.json": (
            ClientBondOnlyTurnLLMOutput.model_json_schema()
        ),
        "evaluation_summary.schema.json": SummaryLLMOutput.model_json_schema(),
    }
    for filename, schema in schemas.items():
        path = output_dir / filename
        path.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    raise SystemExit("Use alliance-eval export-schema --output-dir DIR to select an export destination.")
