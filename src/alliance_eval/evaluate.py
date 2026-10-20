#!/usr/bin/env python3
"""Model interfaces, Activity/turn/summary inputs, and structural and exact-evidence validation.

Use only permitted dialogue context and retain unmodified model responses. See framework/ for definitions."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from alliance_eval.evaluation_models import (
    BOND_DIMENSIONS,
    CLIENT_ELIGIBLE_DIMENSIONS,
    COUNSELLOR_ELIGIBLE_DIMENSIONS,
    SUMMARY_LEVEL_DIMENSIONS,
    TASK_DIMENSIONS,
    ClientBondOnlyTurnLLMOutput,
    ClientTurnLLMOutput,
    CounsellorBondOnlyTurnLLMOutput,
    CounsellorTurnLLMOutput,
    SummaryLLMOutput,
    ActivityIdentificationLLMOutput,
    TurnLLMOutput,
    TurnStoredRecord,
)


# Retain the loaded source identity; changed source requires a process restart.
LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
MISTRAL_MODEL_ID = "mistral-small-2603"
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


OutputModel = TypeVar("OutputModel", bound=BaseModel)


@dataclass
class JudgeCall:
    """Parsed output, raw text and available token usage from one judge call."""

    parsed: BaseModel
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: str | None = None


class JudgeOutputError(ValueError):
    """Retain failed responses and known usage for inspection."""

    def __init__(
        self, message: str, raw_response: str, usage: dict[str, int],
        *, schema_validation_failed: bool = False,
    ) -> None:
        """Preserve failure details without converting failures into ratings."""

        super().__init__(message)
        self.raw_response = raw_response
        self.usage = usage
        # Only completed responses with structural errors may undergo group validation; never salvage truncated or refused output.
        self.schema_validation_failed = schema_validation_failed


def parse_judge_response(
    content: str, usage: dict[str, int], response_model: type[OutputModel]
) -> JudgeCall:
    """Validate JSON structure and retain response text and token usage on failure."""

    try:
        parsed = response_model.model_validate_json(content)
    except ValidationError as error:
        raise JudgeOutputError(
            str(error), content, usage, schema_validation_failed=True
        ) from error
    return JudgeCall(parsed=parsed, usage=usage, raw_response=content)


def validate_output_groups(
    content: str,
    response_model: type[OutputModel],
    validator: Callable[[OutputModel, str], None],
    active_cores: tuple[str, ...],
) -> tuple[OutputModel | None, list[dict[str, str]]]:
    """Validate entire Bond/Task groups without changing order, fields or validation rules.

    First validate the JSON envelope and dimension membership; reject ambiguous grouping.
    A failed finding or result rejects its entire group; do not fill evidence gaps or repair entries.
    """

    payload = json.loads(content)
    field_name = "findings" if "findings" in response_model.model_fields else "results"
    if not isinstance(payload, dict) or not isinstance(payload.get(field_name), list):
        raise ValueError(f"The response must be a JSON object containing a {field_name} array")
    # Use the original model to enforce envelope constraints, including extra-field rejection.
    response_model.model_validate({**payload, field_name: []})
    groups: dict[str, list[dict[str, Any]]] = {"Bond": [], "Task": []}
    item_cores = []
    for item in payload[field_name]:
        dimension = item.get("dimension") if isinstance(item, dict) else None
        if dimension in BOND_DIMENSIONS:
            core = "Bond"
        elif dimension in TASK_DIMENSIONS:
            core = "Task"
        else:
            raise ValueError(f"Cannot assign result dimension to Bond/Task: {dimension!r}")
        groups[core].append(item)
        item_cores.append(core)

    accepted_cores = set()
    failures = []
    for core, items in groups.items():
        try:
            parsed = response_model.model_validate({field_name: items})
            validator(parsed, core)
        except ValueError as error:
            failures.append({
                "core": core, "error_type": type(error).__name__, "message": str(error),
            })
        else:
            accepted_cores.add(core)

    # An empty Task group without Activity targets is not a successful turn or summary.
    if not accepted_cores.intersection(active_cores):
        return None, failures
    accepted = [
        item for item, core in zip(payload[field_name], item_cores)
        if core in accepted_cores
    ]
    return response_model.model_validate({field_name: accepted}), failures


class JudgeBackend(Protocol):
    """Minimal generation interface for judge backends."""

    model_id: str

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[OutputModel],
    ) -> JudgeCall:
        """Return structured output for the prompts and Pydantic response model."""

        ...


class MockJudge:
    """Deterministic response queue for tests, not a research judge."""

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        """Store ordered test responses; an empty queue returns an empty output envelope."""

        self.model_id = "mock"
        self.responses = list(responses or [])

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[OutputModel],
    ) -> JudgeCall:
        """Validate the next test response with the same Pydantic model used for real backends."""

        if self.responses:
            raw = self.responses.pop(0)
        elif response_model in {
            TurnLLMOutput,
            CounsellorBondOnlyTurnLLMOutput,
            ClientBondOnlyTurnLLMOutput,
            CounsellorTurnLLMOutput,
            ClientTurnLLMOutput,
        }:
            raw = {"findings": []}
        elif response_model is ActivityIdentificationLLMOutput:
            raw = {"new_activities": []}
        else:
            raw = {"results": []}
        return parse_judge_response(json.dumps(raw), {}, response_model)


class DeepSeekJudge:
    """DeepSeek JSON Output interface with local structural and evidence validation.

    JSON mode does not enforce a strict JSON Schema. The generated schema accompanies the input
    to describe output structure, not new scoring rules. Credentials remain in HTTP authentication only.
    """

    def __init__(
        self,
        model_id: str = "deepseek-flash",
        api_key: str | None = None,
        parameters: dict[str, Any] | None = None,
        timeout_seconds: float = 600.0,
    ) -> None:
        """Read a supplied or environment key; omit unsupported seed and num_ctx parameters."""

        key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not key.strip():
            raise ValueError("Enter a DeepSeek API key in the app or set DEEPSEEK_API_KEY.")
        self.model_id = model_id
        self.base_url = "https://api.deepseek.com"
        self.dialect = "deepseek"
        supplied = dict(parameters or {})
        if supplied.keys() - {"temperature", "max_tokens", "thinking", "reasoning_effort"}:
            raise ValueError("DeepSeek parameters support only temperature, max_tokens, thinking and reasoning_effort.")
        if supplied.get("thinking", {"type": "disabled"}) not in (
            {"type": "enabled"}, {"type": "disabled"},
        ):
            raise ValueError("DeepSeek thinking accepts only type enabled or disabled.")
        if supplied.get("reasoning_effort", "high") not in ("low", "high", "max"):
            raise ValueError("DeepSeek reasoning_effort accepts only low, high or max.")
        self.parameters = {
            "temperature": 0.0,
            # Use one default output limit across stages; explicit parameters take precedence.
            "max_tokens": 32768,
            "thinking": {"type": "disabled"},
            **supplied,
        }
        # Omit temperature when thinking is enabled, since the service ignores it.
        # Source: https://api-docs.deepseek.com/guides/thinking_mode/
        if self.parameters["thinking"]["type"] == "enabled":
            self.parameters.pop("temperature", None)
            self.parameters.setdefault("reasoning_effort", "high")
        else:
            self.parameters.pop("reasoning_effort", None)
        self.client = httpx.Client(
            timeout=timeout_seconds, headers={"Authorization": f"Bearer {key}"}
        )

    def close(self) -> None:
        """Close the HTTP client owned by this backend."""

        self.client.close()

    def generate(
        self, system_prompt: str, user_prompt: str,
        response_model: type[OutputModel],
    ) -> JudgeCall:
        """Send the prompts and output schema, parse JSON and retain actual usage."""

        output_contract = (
            "\n\nReturn a JSON object matching this JSON Schema:\n"
            + json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        )
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt + output_contract},
                ],
                "response_format": {"type": "json_object"},
                "stream": False,
                **self.parameters,
            },
        )
        response.raise_for_status()
        body = response.json()
        api_usage = body.get("usage", {})
        usage = {
            "input_tokens": int(api_usage.get("prompt_tokens", 0)),
            "output_tokens": int(api_usage.get("completion_tokens", 0)),
        }
        for key in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
            if key in api_usage:
                usage[key] = int(api_usage[key])
        content = body["choices"][0]["message"].get("content") or ""
        finish_reason = body["choices"][0].get("finish_reason")
        if finish_reason in {"length", "content_filter"}:
            raise JudgeOutputError(f"Incomplete response: {finish_reason}", content, usage)
        return parse_judge_response(content, usage, response_model)


class OpenAIJudge:
    """OpenAI JSON mode interface with local structural and evidence validation.

    Send the Pydantic output schema with the input rather than enabling server-side strict schema mode.
    """

    def __init__(
        self, model_id: str = "gpt-5.6-luna", api_key: str | None = None,
        parameters: dict[str, Any] | None = None, timeout_seconds: float = 600.0,
    ) -> None:
        """Read credentials and reasoning effort; default to none and omit local seed or num_ctx."""

        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key.strip():
            raise ValueError("Enter an OpenAI API key in the app or set OPENAI_API_KEY.")
        supplied = dict(parameters or {})
        if supplied.keys() - {"temperature", "max_completion_tokens", "reasoning_effort"}:
            raise ValueError("OpenAI parameters support only temperature, max_completion_tokens and reasoning_effort.")
        if supplied.get("reasoning_effort", "none") not in (
            "none", "low", "medium", "high", "xhigh", "max",
        ):
            raise ValueError("OpenAI reasoning_effort accepts only none, low, medium, high, xhigh or max.")
        self.model_id = model_id
        self.base_url = "https://api.openai.com/v1"
        self.dialect = "openai"
        self.parameters = {
            "temperature": 0.0,
            "max_completion_tokens": 32768,
            "reasoning_effort": "none", **supplied,
            "response_format": {"type": "json_object"}, "store": False,
        }
        # In reasoning mode, omit the inactive temperature setting from requests and manifests.
        if self.parameters["reasoning_effort"] != "none":
            self.parameters.pop("temperature", None)
        self.client = httpx.Client(
            timeout=timeout_seconds, headers={"Authorization": f"Bearer {key}"}
        )

    def close(self) -> None:
        """Close the HTTP client owned by this backend."""

        self.client.close()

    def generate(
        self, system_prompt: str, user_prompt: str,
        response_model: type[OutputModel],
    ) -> JudgeCall:
        """Use the shared input contract; retain final JSON and usage breakdowns without double-counting tokens."""

        output_contract = (
            "\n\nReturn a JSON object matching this JSON Schema:\n"
            + json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        )
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt + output_contract},
                ],
                "stream": False, **self.parameters,
            },
        )
        response.raise_for_status()
        body = response.json()
        api_usage = body.get("usage", {})
        usage = {
            "input_tokens": int(api_usage.get("prompt_tokens", 0)),
            "output_tokens": int(api_usage.get("completion_tokens", 0)),
        }
        # Cached and reasoning tokens are already included in totals; retain them only as breakdowns.
        for section, source, target in (
            ("prompt_tokens_details", "cached_tokens", "prompt_cache_hit_tokens"),
            ("completion_tokens_details", "reasoning_tokens", "reasoning_tokens"),
        ):
            details = api_usage.get(section) or {}
            if source in details:
                usage[target] = int(details[source])
        choice = body["choices"][0]
        content = choice["message"].get("content") or ""
        if choice["message"].get("refusal"):
            raise JudgeOutputError("OpenAI refusal", choice["message"]["refusal"], usage)
        if choice.get("finish_reason") in {"length", "content_filter"}:
            raise JudgeOutputError(
                f"Incomplete response: {choice['finish_reason']}", content, usage
            )
        return parse_judge_response(content, usage, response_model)


class LocalHTTPJudge:
    """Call user-configured models through Ollama, Mistral or an OpenAI-compatible interface.

    Ollama receives the JSON Schema through `format`. Compatible services receive it
    through `response_format.json_schema`, or as message text in explicit JSON object mode.
    Local Pydantic and evidence validation remain necessary after either mode.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str,
        dialect: str,
        parameters: dict[str, Any] | None = None,
        timeout_seconds: float = 600.0,
        api_key: str | None = None,
        response_format: str = "json_schema",
    ) -> None:
        """Store the service protocol, generation parameters and per-request timeout."""

        if dialect not in {"ollama", "openai_compatible", "mistral"}:
            raise ValueError("dialect must be ollama, openai_compatible or mistral")
        if response_format not in {"json_schema", "json_object"}:
            raise ValueError("response_format must be json_schema or json_object")
        if dialect == "ollama" and response_format != "json_schema":
            raise ValueError("Ollama uses a format schema; response_format applies only to compatible APIs")
        if dialect == "mistral" and response_format != "json_schema":
            raise ValueError("Mistral requires json_schema; output formats are not downgraded automatically")
        url = httpx.URL(base_url)
        if url.scheme not in {"http", "https"} or not url.host:
            raise ValueError("The model URL must be a complete HTTP(S) service address")
        if url.userinfo or url.query or url.fragment:
            raise ValueError("The model URL must not contain credentials, query parameters or a fragment; use an API key for authentication")
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.dialect = dialect
        self.response_format = response_format
        self.parameters = dict(parameters or {})
        reserved = {"model", "messages", "response_format", "stream", "api_key", "headers", "authorization", "base_url"}
        if reserved.intersection(key.lower() for key in self.parameters):
            raise ValueError("Model parameters must not override request structure or contain authentication information; use the dedicated settings")
        if self.dialect == "mistral":
            api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
            if not api_key.strip():
                raise ValueError("Enter a Mistral API key in the app or set MISTRAL_API_KEY.")
            if self.parameters.keys() - {"temperature", "max_tokens", "reasoning_effort", "random_seed"}:
                raise ValueError("Mistral parameters support only temperature, max_tokens, reasoning_effort and random_seed.")
            effort = self.parameters.get("reasoning_effort", "none")
            if effort not in ("none", "high"):
                raise ValueError("Mistral Small 4 reasoning_effort accepts only none or high.")
            self.parameters = {
                "temperature": 0.0, "max_tokens": 32768,
                "reasoning_effort": "none", "random_seed": 666, **self.parameters,
            }
        # Disable Ollama thinking by default; evaluation consumes the final structured response.
        # Record defaults in parameters so the manifest captures the actual settings.
        if self.dialect == "ollama":
            self.parameters.setdefault("seed", 666)
            self.parameters.setdefault("think", False)
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        """Close the HTTP client owned by this backend."""

        self.client.close()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[OutputModel],
    ) -> JudgeCall:
        """Send the prompts and schema through the selected service protocol."""

        if self.dialect == "ollama":
            return self._generate_ollama(system_prompt, user_prompt, response_model)
        return self._generate_openai_compatible(
            system_prompt, user_prompt, response_model
        )

    def _generate_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[OutputModel],
    ) -> JudgeCall:
        """Call Ollama /api/chat with the Pydantic JSON Schema and validate the response."""

        endpoint = f"{self.base_url}/api/chat"
        options = {
            key: value for key, value in self.parameters.items() if key != "think"
        }
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": self.parameters["think"],
            "format": response_model.model_json_schema(),
            "options": options,
        }
        response = self.client.post(endpoint, json=payload, headers=self._headers)
        response.raise_for_status()
        body = response.json()
        usage = {
            "input_tokens": int(body.get("prompt_eval_count", 0)),
            "output_tokens": int(body.get("eval_count", 0)),
        }
        content = body["message"]["content"]
        if body.get("done") is not True or body.get("done_reason") != "stop":
            raw_content = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            raise JudgeOutputError(
                f"Incomplete response: done={body.get('done')}, done_reason={body.get('done_reason')}",
                raw_content, usage,
            )
        return parse_judge_response(content, usage, response_model)

    def _generate_openai_compatible(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[OutputModel],
    ) -> JudgeCall:
        """Call the /v1/chat/completions structured-output interface."""

        base = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        endpoint = f"{base}/v1/chat/completions"
        if self.response_format == "json_object":
            user_prompt += (
                "\n\nReturn a JSON object matching this JSON Schema:\n"
                + json.dumps(response_model.model_json_schema(), ensure_ascii=False)
            )
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            **self.parameters,
        }
        payload["response_format"] = {"type": self.response_format}
        if self.response_format == "json_schema":
            payload["response_format"]["json_schema"] = {
                "name": response_model.__name__,
                "strict": True,
                "schema": response_model.model_json_schema(),
            }
        response = self.client.post(endpoint, json=payload, headers=self._headers)
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        content = choice["message"]["content"]
        api_usage = body.get("usage", {})
        usage = {
            "input_tokens": int(api_usage.get("prompt_tokens", 0)),
            "output_tokens": int(api_usage.get("completion_tokens", 0)),
        }
        raw_content = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        if choice.get("finish_reason") != "stop":
            raise JudgeOutputError(f"Incomplete response: {choice.get('finish_reason')}", raw_content, usage)
        if self.dialect == "mistral":
            # For Mistral reasoning blocks, parse only final text as evaluation output.
            # https://docs.mistral.ai/studio/conversations/reasoning
            if isinstance(content, list):
                if any(not isinstance(chunk, dict) or chunk.get("type") not in {"thinking", "text"}
                       or (chunk["type"] == "text" and not isinstance(chunk.get("text"), str))
                       for chunk in content):
                    raise JudgeOutputError("Unsupported Mistral content chunks", raw_content, usage)
                content = "".join(chunk["text"] for chunk in content if chunk["type"] == "text")
            elif not isinstance(content, str):
                raise JudgeOutputError("Missing Mistral text response", raw_content, usage)
        return parse_judge_response(content, usage, response_model)


def load_prompt(path: Path) -> str:
    """Load the prompt resource rather than embedding the rubric in Python constants."""

    return path.read_text(encoding="utf-8")


def _available_activity_ids(activity_registry: dict[str, str], count: int = 3) -> list[str]:
    """Provide consecutive new Activity IDs after those already in the registry."""

    used_numbers = [int(activity_id.split("_")[1]) for activity_id in activity_registry]
    start = max(used_numbers, default=0) + 1
    return [f"ACTIVITY_{number:02d}" for number in range(start, start + count)]


def build_activity_identification_user_prompt(
    dialogue: dict[str, Any],
    current_position: int,
    activity_registry: dict[str, str],
) -> str:
    """Build blinded input for identifying new Activities in the current turn."""

    current_turn = dialogue["turns"][current_position]
    visible = dialogue["turns"][: current_position + 1]
    payload = {
        "visible_turns": [
            {
                "turn_id": turn["turn_id"],
                "role": turn["role"],
                "text": turn["text"],
            }
            for turn in visible
        ],
        "current_turn_id": current_turn["turn_id"],
        "current_role": current_turn["role"],
        "activity_registry": [
            {"activity_id": activity_id, "activity_label": label}
            for activity_id, label in activity_registry.items()
        ],
        "available_new_activity_ids": _available_activity_ids(activity_registry),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _eligible_dimensions(role: str, activity_registry: dict[str, str]) -> tuple[str, ...]:
    """Return speaker-eligible dimensions, excluding Task dimensions without a registered Activity."""

    speaker_dimensions = (
        COUNSELLOR_ELIGIBLE_DIMENSIONS
        if role == "counsellor"
        else CLIENT_ELIGIBLE_DIMENSIONS
    )
    if activity_registry:
        return speaker_dimensions
    # Turn annotation reads the registry established by the separate Activity identification step.
    return tuple(
        dimension
        for dimension in speaker_dimensions
        if dimension not in TASK_DIMENSIONS
    )


def build_turn_user_prompt(
    dialogue: dict[str, Any],
    current_position: int,
    activity_registry: dict[str, str],
) -> str:
    """Build blinded input containing the current and preceding turns and Activity registry."""

    current_turn = dialogue["turns"][current_position]
    visible = dialogue["turns"][: current_position + 1]
    role = current_turn["role"]
    eligible = _eligible_dimensions(role, activity_registry)
    payload = {
        "visible_turns": [
            {
                "turn_id": turn["turn_id"],
                "role": turn["role"],
                "text": turn["text"],
            }
            for turn in visible
        ],
        "current_turn_id": current_turn["turn_id"],
        "current_role": role,
        "eligible_dimensions": list(eligible),
        "activity_registry": [
            {"activity_id": activity_id, "activity_label": label}
            for activity_id, label in activity_registry.items()
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_summary_user_prompt(
    dialogue: dict[str, Any],
    turn_records: list[TurnStoredRecord],
    activity_registry: dict[str, str],
    failed_annotations: list[dict[str, Any]] | None = None,
) -> str:
    """Build summary input for the completed sample without dataset or condition metadata.

    List only three Bond summary targets and engagement for each registered Activity.
    Do not request unresolved turn targets or explanations of absent turn findings.
    Each target receives existing evidence, reasons and annotation turns for that
    dimension and Activity, without extraction, merging or prior judgements of evidence direction.
    """

    required_summary_targets = _build_required_summary_targets(activity_registry)

    annotations_by_key: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    for record in turn_records:
        for finding in record.findings:
            key = _result_key(finding.dimension, finding.activity_reference)
            annotations_by_key.setdefault(key, []).append({
                "current_turn_id": record.current_turn_id,
                "evidence": [span.model_dump() for span in finding.evidence],
                "reason": finding.reason,
            })

    # Group existing records by target while preserving their order and individual spans.
    for targets in required_summary_targets.values():
        for target in targets:
            activity = target["activity_reference"]
            key = (target["dimension"], activity["activity_id"] if activity else None)
            target["validated_turn_annotations"] = annotations_by_key.get(key, [])

    payload = {
        "completed_sample": [
            {
                "turn_id": turn["turn_id"],
                "role": turn["role"],
                "text": turn["text"],
            }
            for turn in dialogue["turns"]
        ],
        "required_summary_targets": required_summary_targets,
    }
    if failed_annotations:
        payload["failed_annotations"] = failed_annotations
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_required_summary_targets(
    activity_registry: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    """Build the three Bond targets and an engagement target for each registered Activity."""

    targets: list[dict[str, Any]] = []
    for dimension in SUMMARY_LEVEL_DIMENSIONS:
        if dimension in BOND_DIMENSIONS:
            targets.append(
                {"dimension": dimension, "activity_reference": None}
            )

    for activity_id, activity_label in activity_registry.items():
        activity_reference = {"activity_id": activity_id, "activity_label": activity_label}
        for dimension in SUMMARY_LEVEL_DIMENSIONS:
            if dimension in TASK_DIMENSIONS:
                targets.append(
                    {
                        "dimension": dimension,
                        "activity_reference": activity_reference,
                    }
                )
    return {"summary_level_ratings": targets}


def _validate_spans(
    spans: list[Any],
    turn_map: dict[str, dict[str, Any]],
    visible_turn_ids: set[str],
) -> None:
    """Check that evidence refers to visible turns and quotes exact substrings.

    Exact quotes provide locatable dialogue evidence as discussed by Li et al. (2024).
    Do not use fuzzy matching to repair paraphrased model quotes.
    """

    for span in spans:
        if span.turn_id not in visible_turn_ids:
            raise ValueError(f"Evidence refers to an unavailable turn: {span.turn_id}")
        if span.quote not in turn_map[span.turn_id]["text"]:
            raise ValueError(
                f"Evidence is not an exact quote from {span.turn_id}: {span.quote!r}"
            )


def validate_activity_identification_output(
    output: ActivityIdentificationLLMOutput,
    dialogue: dict[str, Any],
    current_position: int,
    activity_registry: dict[str, str],
) -> dict[str, str]:
    """Validate uniqueness, availability and continuity of new Activity IDs.

    The pipeline records source turns; label meaning still requires source-text review.
    """

    available_ids = set(_available_activity_ids(activity_registry))
    new_activities: dict[str, str] = {}
    for reference in output.new_activities:
        activity_id = reference.activity_id
        activity_label = reference.activity_label
        if activity_id in activity_registry:
            raise ValueError(f"Activity identification must not register an existing Activity again: {activity_id}")
        if activity_id not in available_ids:
            raise ValueError(f"New Activity ID is unavailable for this turn: {activity_id}")
        if activity_id in new_activities:
            raise ValueError(f"Duplicate Activity ID in the same turn: {activity_id}")
        new_activities[activity_id] = activity_label

    if new_activities:
        ordered = sorted(new_activities, key=lambda value: int(value.split("_")[1]))
        expected = _available_activity_ids(activity_registry, len(ordered))
        if ordered != expected:
            raise ValueError("New Activity IDs must be consecutive, starting with the next available ID")
    return new_activities


def validate_turn_output(
    output: TurnLLMOutput,
    dialogue: dict[str, Any],
    current_position: int,
    activity_registry: dict[str, str],
) -> None:
    """Validate speaker eligibility, current-turn evidence, visible references and registry consistency."""

    current_turn = dialogue["turns"][current_position]
    visible = dialogue["turns"][: current_position + 1]
    turn_map = {turn["turn_id"]: turn for turn in visible}
    visible_ids = set(turn_map)
    eligible = set(_eligible_dimensions(current_turn["role"], activity_registry))

    for finding in output.findings:
        if finding.dimension not in eligible:
            raise ValueError(
                f"{current_turn['role']} turn cannot output dimension: {finding.dimension}"
            )
        _validate_spans(finding.evidence, turn_map, visible_ids)
        if not any(
            span.turn_id == current_turn["turn_id"] for span in finding.evidence
        ):
            raise ValueError("Each turn finding must cite an exact quote from the current turn")

        reference = finding.activity_reference
        if reference is None:
            continue
        activity_id = reference.activity_id
        activity_label = reference.activity_label
        if activity_id not in activity_registry:
            raise ValueError(f"Turn judge refers to an unregistered Activity: {activity_id}")
        if activity_registry[activity_id] != activity_label:
            raise ValueError(f"Activity label does not match the registry: {activity_id}")


def _result_key(dimension: str, activity_reference: Any) -> tuple[str, str | None]:
    """Build a comparable summary key for a Bond or Activity result."""

    activity_id = activity_reference.activity_id if activity_reference is not None else None
    return dimension, activity_id


def validate_summary_output(
    output: SummaryLLMOutput,
    dialogue: dict[str, Any],
    turn_records: list[TurnStoredRecord],
    activity_registry: dict[str, str],
    core: str | None = None,
) -> None:
    """Validate summary evidence, Activity references, uniqueness and target coverage.

    Do not rate turn-level dimensions or infer reasons for absent ratings.
    Check each activity_reference separately against the specific work boundary
    described by Bordin (1979), without substituting one Activity result for another.
    """

    turn_map = {turn["turn_id"]: turn for turn in dialogue["turns"]}
    all_turn_ids = set(turn_map)

    required_targets = _build_required_summary_targets(activity_registry)
    expected_keys = {
        (
            target["dimension"],
            (
                target["activity_reference"]["activity_id"]
                if target["activity_reference"] is not None
                else None
            ),
        )
        for group in required_targets.values()
        for target in group
        if core is None or ("Bond" if target["dimension"] in BOND_DIMENSIONS else "Task") == core
    }
    validated_spans_by_key: dict[
        tuple[str, str | None], set[tuple[str, str]]
    ] = {}
    for record in turn_records:
        for finding in record.findings:
            key = _result_key(finding.dimension, finding.activity_reference)
            validated_spans_by_key.setdefault(key, set()).update(
                (span.turn_id, span.quote) for span in finding.evidence
            )

    actual_keys: set[tuple[str, str | None]] = set()
    for result in output.results:
        key = _result_key(result.dimension, result.activity_reference)
        if key in actual_keys:
            raise ValueError(f"Duplicate summary result: {key}")
        if key not in expected_keys:
            raise ValueError(f"Summary includes an unrequested target: {key}")
        actual_keys.add(key)

        if result.activity_reference is not None:
            activity_id = result.activity_reference.activity_id
            if activity_id not in activity_registry:
                raise ValueError(f"Summary refers to an unknown Activity: {activity_id}")
            if activity_registry[activity_id] != result.activity_reference.activity_label:
                raise ValueError(f"Summary Activity label does not match the registry: {activity_id}")

        _validate_spans(result.evidence, turn_map, all_turn_ids)
        _validate_spans(result.counterevidence, turn_map, all_turn_ids)
        allowed_spans = validated_spans_by_key.get(key, set())
        # Empty-evidence exceptions depend on the upstream evidence pool, not JSON Schema alone.
        if result.summary_status == "insufficient_evidence" and not result.evidence:
            if allowed_spans:
                raise ValueError("Insufficient evidence permits empty evidence only when the summary target has no existing evidence")
        for span in (*result.evidence, *result.counterevidence):
            # Allow contiguous shortening, but not extension, stitching, paraphrase or cross-target borrowing.
            if not any(
                span.turn_id == turn_id and span.quote in quote
                for turn_id, quote in allowed_spans
            ):
                raise ValueError(
                    "Summary evidence must come from a validated turn span for the same dimension and Activity: "
                    f"{key}, {span.turn_id}, {span.quote!r}"
                )

    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(f"Incomplete summary coverage, missing={missing}, extra={extra}")
