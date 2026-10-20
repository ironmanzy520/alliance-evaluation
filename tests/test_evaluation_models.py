"""Structural constraint tests for Pydantic evaluation models."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alliance_eval.evaluation_models import (
    BOND_DIMENSIONS,
    TASK_DIMENSIONS,
    SUMMARY_LEVEL_DIMENSIONS,
    ClientBondOnlyTurnLLMOutput,
    ClientTurnLLMOutput,
    CounsellorBondOnlyTurnLLMOutput,
    CounsellorTurnLLMOutput,
    SampleSelectionItem,
    SummaryLLMOutput,
    ActivityIdentificationLLMOutput,
    TurnLLMOutput,
)


@pytest.mark.parametrize("payload", [
    {"new_tasks": []},
    {"new_activities": [{"activity_id": "TASK_01", "activity_label": "Write a note"}]},
    {"new_activities": [{"task_id": "TASK_01", "task_label": "Write a note"}]},
])
def test_activity_model_rejects_legacy_contract_without_auto_repair(payload):
    with pytest.raises(ValidationError):
        ActivityIdentificationLLMOutput.model_validate(payload)


def test_activity_identification_schema_is_minimal() -> None:
    """Activity identification stores new references without dimension-rating fields."""

    output = ActivityIdentificationLLMOutput.model_validate(
        {"new_activities": [{"activity_id": "ACTIVITY_01", "activity_label": "write the thoughts down"}]}
    )
    assert output.new_activities[0].activity_id == "ACTIVITY_01"


def test_no_activity_runtime_schemas_reject_activity_dimensions() -> None:
    """Runtime schemas for an empty registry cannot produce Activity findings."""

    activity_finding = {
        "findings": [
            {
                "dimension": "Activity agreement/disagreement",
                "activity_reference": {
                    "activity_id": "ACTIVITY_01",
                    "activity_label": "write the thoughts down",
                },
                "finding_status": "evidence_only",
                "rating_label": None,
                "evidence": [{"turn_id": "T01", "quote": "okay"}],
                "reason": "Activity evidence.",
            }
        ]
    }
    with pytest.raises(ValidationError):
        CounsellorBondOnlyTurnLLMOutput.model_validate(activity_finding)
    with pytest.raises(ValidationError):
        ClientBondOnlyTurnLLMOutput.model_validate(activity_finding)


def test_speaker_specific_turn_schemas_reject_other_speaker_dimensions() -> None:
    """Each runtime schema rejects dimensions exclusive to the other speaker."""

    client_confirmation = {
        "findings": [
            {
                "dimension": "Client confirmation/disconfirmation",
                "activity_reference": None,
                "finding_status": "rated",
                "rating_label": "Clear confirmation",
                "evidence": [{"turn_id": "T01", "quote": "That is right."}],
                "reason": "The client confirms the prior understanding.",
            }
        ]
    }
    counsellor_grounded = {
        "findings": [
            {
                "dimension": "Grounded content uptake",
                "activity_reference": None,
                "finding_status": "rated",
                "rating_label": "Substantive grounded uptake",
                "evidence": [{"turn_id": "T02", "quote": "You feel anxious."}],
                "reason": "The counsellor uses the preceding client contribution.",
            }
        ]
    }

    ClientTurnLLMOutput.model_validate(client_confirmation)
    CounsellorTurnLLMOutput.model_validate(counsellor_grounded)
    with pytest.raises(ValidationError):
        CounsellorTurnLLMOutput.model_validate(client_confirmation)
    with pytest.raises(ValidationError):
        ClientTurnLLMOutput.model_validate(counsellor_grounded)


def test_turn_dimension_rejects_wrong_label() -> None:
    """Grounded content uptake rejects labels from other dimensions or invented labels."""

    raw = {
        "findings": [
            {
                "dimension": "Grounded content uptake",
                "activity_reference": None,
                "finding_status": "rated",
                "rating_label": "Clear confirmation",
                "evidence": [{"turn_id": "T01", "quote": "I hear you"}],
                "reason": "Wrong cross-dimension label.",
            }
        ]
    }
    with pytest.raises(ValidationError):
        TurnLLMOutput.model_validate(raw)


def test_turn_rejects_extra_field() -> None:
    """Model output cannot add confidence or undeclared fields."""

    raw = {"findings": [], "confidence": "high"}
    with pytest.raises(ValidationError):
        TurnLLMOutput.model_validate(raw)


def test_summary_level_dimension_cannot_be_rated_at_turn_level() -> None:
    """Accurate Empathy accumulates turn evidence without assigning an early AE rating."""

    raw = {
        "findings": [
            {
                "dimension": "Accurate Empathy",
                "activity_reference": None,
                "finding_status": "rated",
                "rating_label": "AE-5",
                "evidence": [{"turn_id": "T01", "quote": "I hear you"}],
                "reason": "Premature summary-level score.",
            }
        ]
    }
    with pytest.raises(ValidationError):
        TurnLLMOutput.model_validate(raw)


def test_turn_level_bond_dimension_cannot_use_evidence_only() -> None:
    """Turn-level Bond dimensions must be rated, not deferred as evidence_only."""

    raw = {
        "findings": [
            {
                "dimension": "Grounded content uptake",
                "activity_reference": None,
                "finding_status": "evidence_only",
                "rating_label": None,
                "evidence": [{"turn_id": "T01", "quote": "I hear you."}],
                "reason": "The process has started.",
            }
        ]
    }
    with pytest.raises(ValidationError):
        TurnLLMOutput.model_validate(raw)


def test_turn_output_rejects_insufficient_evidence() -> None:
    """Turns do not output insufficiency statuses; summary does not explain missing turn targets."""

    raw = {
        "findings": [
            {
                "dimension": "Grounded content uptake",
                "activity_reference": None,
                "finding_status": "insufficient_evidence",
                "rating_label": None,
                "evidence": [{"turn_id": "T01", "quote": "I hear you."}],
                "reason": "The evidence is ambiguous.",
            }
        ]
    }
    with pytest.raises(ValidationError):
        TurnLLMOutput.model_validate(raw)


@pytest.mark.parametrize("dimension", [
    d for d in (*BOND_DIMENSIONS, *TASK_DIMENSIONS) if d not in SUMMARY_LEVEL_DIMENSIONS
])
@pytest.mark.parametrize("status", ["no_evaluable_opportunity", "insufficient_evidence"])
def test_all_turn_dimensions_are_rejected_by_summary(dimension, status) -> None:
    """Summary excludes explanatory judgements for turn-level dimensions."""

    result = {
        "dimension": dimension,
        "activity_reference": ({"activity_id": "ACTIVITY_01", "activity_label": "the work"}
                           if dimension in TASK_DIMENSIONS else None),
        "summary_status": status, "rating_label": None,
        "evidence": [], "counterevidence": [], "reason": "Invalid turn target.",
    }
    with pytest.raises(ValidationError):
        SummaryLLMOutput.model_validate({"results": [result]})


@pytest.mark.parametrize("dimension", SUMMARY_LEVEL_DIMENSIONS)
@pytest.mark.parametrize("status", ["rated", "insufficient_evidence", "no_evaluable_opportunity"])
def test_four_summary_dimensions_retain_original_statuses(dimension, status) -> None:
    """The four summary dimensions permit their ratings and missing statuses; context validation checks evidence pools."""

    result = {
        "dimension": dimension,
        "activity_reference": ({"activity_id": "ACTIVITY_01", "activity_label": "the work"}
                           if dimension in TASK_DIMENSIONS else None),
        "summary_status": status,
        "rating_label": ("AE-3" if dimension == "Accurate Empathy" else "Somewhat salient")
                        if status == "rated" else None,
        "evidence": [{"turn_id": "T01", "quote": "Evidence."}] if status != "no_evaluable_opportunity" else [],
        "counterevidence": [], "reason": "Offline structural fixture.",
    }
    SummaryLLMOutput.model_validate({"results": [result]})


def test_no_opportunity_requires_empty_evidence() -> None:
    """No evaluable opportunity must not include textual evidence."""

    raw = {
        "results": [
            {
                "dimension": "Accurate Empathy",
                "activity_reference": None,
                "summary_status": "no_evaluable_opportunity",
                "rating_label": None,
                "evidence": [{"turn_id": "T01", "quote": "text"}],
                "counterevidence": [],
                "reason": "Contradictory structure.",
            }
        ]
    }
    with pytest.raises(ValidationError):
        SummaryLLMOutput.model_validate(raw)


def test_acceptance_validation_uses_source_description_labels() -> None:
    """WT3 summary accepts descriptive manual labels, not project-number labels."""

    base_result = {
        "dimension": "Therapist accepts/validates Patient",
        "activity_reference": None,
        "summary_status": "rated",
        "evidence": [{"turn_id": "T02", "quote": "That makes sense."}],
        "counterevidence": [],
        "reason": "The stance is observable.",
    }
    valid = {**base_result, "rating_label": "Somewhat salient"}
    SummaryLLMOutput.model_validate({"results": [valid]})

    obsolete = {**base_result, "rating_label": "WT3-3"}
    with pytest.raises(ValidationError):
        SummaryLLMOutput.model_validate({"results": [obsolete]})


def test_mutual_trust_respect_uses_source_description_labels() -> None:
    """WT2 summary accepts descriptive manual labels, not project-number labels."""

    base_result = {
        "dimension": "Mutual trust and respect",
        "activity_reference": None,
        "summary_status": "rated",
        "evidence": [{"turn_id": "T03", "quote": "I trust your judgement."}],
        "counterevidence": [],
        "reason": "Reciprocal evidence is visible.",
    }
    valid = {**base_result, "rating_label": "Somewhat salient"}
    SummaryLLMOutput.model_validate({"results": [valid]})

    obsolete = {**base_result, "rating_label": "WT2-3"}
    with pytest.raises(ValidationError):
        SummaryLLMOutput.model_validate({"results": [obsolete]})


def test_client_engagement_uses_source_description_labels() -> None:
    """WT5 Activity summary accepts descriptive manual labels, not project-number labels."""

    base_result = {
        "dimension": "Client engagement in the activity",
        "activity_reference": {"activity_id": "ACTIVITY_01", "activity_label": "thought record"},
        "summary_status": "rated",
        "evidence": [{"turn_id": "T04", "quote": "I can try writing it down."}],
        "counterevidence": [],
        "reason": "Engagement with the identified Activity is observable.",
    }
    valid = {**base_result, "rating_label": "Somewhat salient"}
    SummaryLLMOutput.model_validate({"results": [valid]})

    obsolete = {**base_result, "rating_label": "WT5-3"}
    with pytest.raises(ValidationError):
        SummaryLLMOutput.model_validate({"results": [obsolete]})


def test_sample_range_must_be_ordered() -> None:
    """A sample plan must not specify a reversed turn range."""

    with pytest.raises(ValidationError):
        SampleSelectionItem(
            source_file="data.jsonl",
            dialogue_id="D01",
            start_turn_index=8,
            end_turn_index=4,
        )
