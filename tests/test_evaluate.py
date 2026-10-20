"""Regression tests for judge HTTP request and response boundaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alliance_eval.evaluate import (
    LocalHTTPJudge,
    MISTRAL_BASE_URL,
    MISTRAL_MODEL_ID,
    DeepSeekJudge,
    OpenAIJudge,
    MockJudge,
    JudgeOutputError,
    build_summary_user_prompt,
    build_activity_identification_user_prompt,
    build_turn_user_prompt,
    validate_activity_identification_output,
)
from alliance_eval.evaluation_models import CounsellorTurnLLMOutput, ActivityIdentificationLLMOutput, SummaryLLMOutput, TASK_DIMENSIONS
from alliance_eval.pipeline import run_turn_annotation


@pytest.mark.parametrize("backend,kwargs", [
    (DeepSeekJudge, {"api_key": "test-only"}),
    (OpenAIJudge, {"api_key": "test-only"}),
    (LocalHTTPJudge, {"model_id": "test-model", "base_url": "http://127.0.0.1:11434", "dialect": "ollama"}),
    (LocalHTTPJudge, {"model_id": "test-model", "base_url": "https://model.example/v1", "dialect": "openai_compatible"}),
    (LocalHTTPJudge, {"model_id": MISTRAL_MODEL_ID, "base_url": MISTRAL_BASE_URL, "dialect": "mistral", "api_key": "test-only"}),
])
def test_http_backend_close_releases_its_client_without_requests(backend, kwargs) -> None:
    judge = backend(**kwargs)
    assert not judge.client.is_closed
    judge.close()
    assert judge.client.is_closed


@pytest.mark.parametrize("backend,kwargs", [
    (DeepSeekJudge, {"api_key": "test-only"}),
    (OpenAIJudge, {"api_key": "test-only"}),
    (LocalHTTPJudge, {"model_id": "test-model", "base_url": "http://127.0.0.1:11434", "dialect": "ollama"}),
    (LocalHTTPJudge, {"model_id": "test-model", "base_url": "https://model.example/v1", "dialect": "openai_compatible"}),
    (LocalHTTPJudge, {"model_id": MISTRAL_MODEL_ID, "base_url": MISTRAL_BASE_URL, "dialect": "mistral", "api_key": "test-only"}),
])
@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ConnectTimeout])
def test_pipeline_retries_identical_request_for_each_http_backend(monkeypatch, backend, kwargs, error_type):
    from alliance_eval import pipeline

    waits, requests, failures = [], [], []
    monkeypatch.setattr(pipeline.time, "sleep", waits.append)

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            raise error_type("Offline connection failure", request=request)
        payload = json.loads(request.content)
        output = {"new_activities": []} if payload["messages"][0]["content"] == "activity" else {"findings": []}
        content = json.dumps(output)
        if kwargs.get("dialect") == "ollama":
            return httpx.Response(200, json={
                "message": {"content": content}, "prompt_eval_count": 7, "eval_count": 2,
                "done": True, "done_reason": "stop",
            })
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
        })

    judge = backend(**kwargs)
    judge.client.close()
    judge.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        records, registry, usage = run_turn_annotation(
            {"turns": [{"turn_id": "T0", "role": "counsellor", "text": "How are you?"}]},
            "S1", judge, "activity", "turn", error_callback=failures.append,
        )
    finally:
        judge.close()
    assert len(requests) == 3 and waits == [1.0]
    assert requests[0].url == requests[1].url
    assert requests[0].content == requests[1].content
    assert registry == {} and len(records) == 1
    assert usage["requests"] == 3 and usage["input_tokens"] == 14 and usage["output_tokens"] == 4
    assert len(failures) == 1 and failures[0]["retry_scheduled"] is True


@pytest.mark.parametrize("second_label", ["Write down thoughts", "Practise breathing"])
def test_duplicate_activity_ids_fail_without_silent_registry_merge(second_label) -> None:
    dialogue = {"turns": [{
        "turn_id": "T01", "role": "counsellor", "text": "We could write the thoughts down.",
    }]}
    activity_output = {"new_activities": [
        {"activity_id": "ACTIVITY_01", "activity_label": "Write down thoughts"},
        {"activity_id": "ACTIVITY_01", "activity_label": second_label},
    ]}
    errors = []
    saved_activities = []
    records, registry, usage = run_turn_annotation(
        dialogue, "S0001", MockJudge([activity_output, {"findings": []}]),
        "activity prompt", "turn prompt", errors=errors,
        activity_record_callback=saved_activities.append,
    )

    assert registry == {}
    assert saved_activities == []
    assert len(records) == 1
    assert usage["requests"] == 2
    assert len(errors) == 1
    assert errors[0]["stage"] == "activity_identification"
    assert errors[0]["affected_cores"] == ["Task"]
    assert "Duplicate Activity ID in the same turn: ACTIVITY_01" in errors[0]["message"]
    assert json.loads(errors[0]["raw_response"]) == activity_output


def test_distinct_activity_ids_are_not_deduplicated_by_label() -> None:
    output = ActivityIdentificationLLMOutput.model_validate({"new_activities": [
        {"activity_id": "ACTIVITY_01", "activity_label": "Write down thoughts"},
        {"activity_id": "ACTIVITY_02", "activity_label": "Write down thoughts"},
    ]})
    assert validate_activity_identification_output(output, {"turns": []}, 0, {}) == {
        "ACTIVITY_01": "Write down thoughts", "ACTIVITY_02": "Write down thoughts",
    }


@pytest.mark.parametrize("response_format", ["json_schema", "json_object"])
def test_compatible_api_auth_and_explicit_output_format(response_format) -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://model.example/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-only-key"
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"findings":[]}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
        })

    judge = LocalHTTPJudge(
        model_id="user-model", base_url="https://model.example/v1",
        dialect="openai_compatible", api_key="test-only-key",
        response_format=response_format, parameters={"temperature": 0.1},
    )
    judge.client.close()
    judge.client = httpx.Client(transport=httpx.MockTransport(handler))
    result = judge.generate("system", "user", CounsellorTurnLLMOutput)
    judge.client.close()
    assert result.parsed.findings == []
    assert result.usage == {"input_tokens": 7, "output_tokens": 2}
    assert captured["response_format"]["type"] == response_format
    assert ("json_schema" in captured["response_format"]) == (response_format == "json_schema")
    assert captured["temperature"] == 0.1
    assert "seed" not in captured
    assert "test-only-key" not in json.dumps(captured)
    assert "test-only-key" not in json.dumps(judge.parameters)
    assert captured["messages"][0] == {"role": "system", "content": "system"}
    if response_format == "json_schema":
        assert captured["messages"][1]["content"] == "user"
        assert captured["response_format"]["json_schema"]["schema"] == CounsellorTurnLLMOutput.model_json_schema()
    else:
        prefix, schema = captured["messages"][1]["content"].split("\n\nReturn a JSON object matching this JSON Schema:\n")
        assert prefix == "user"
        assert json.loads(schema) == CounsellorTurnLLMOutput.model_json_schema()


@pytest.mark.parametrize("response_model,output", [
    (ActivityIdentificationLLMOutput, {"new_activities": []}),
    (CounsellorTurnLLMOutput, {"findings": []}),
    (SummaryLLMOutput, {"results": []}),
])
def test_compatible_json_mode_supplies_the_selected_schema(response_model, output):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(output)}, "finish_reason": "stop"}],
        })

    judge = LocalHTTPJudge("test-model", "https://model.example/v1", "openai_compatible",
                           response_format="json_object")
    judge.client.close()
    judge.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        result = judge.generate("unchanged system", "unchanged input", response_model)
    finally:
        judge.close()
    assert result.parsed.model_dump() == output
    assert len(requests) == 1
    assert requests[0]["messages"][0]["content"] == "unchanged system"
    prefix, schema = requests[0]["messages"][1]["content"].split(
        "\n\nReturn a JSON object matching this JSON Schema:\n"
    )
    assert prefix == "unchanged input"
    assert json.loads(schema) == response_model.model_json_schema()


@pytest.mark.parametrize("dialect,response_format", [
    ("ollama", "json_schema"), ("openai_compatible", "json_schema"),
    ("openai_compatible", "json_object"),
])
@pytest.mark.parametrize("reason", ["length", "content_filter", "tool_calls", "error", None])
def test_local_http_incomplete_outputs_preserve_content_and_usage(dialect, response_format, reason):
    requests = []
    content = '{"new_activities":[]}'

    def handler(request):
        requests.append(request)
        if dialect == "ollama":
            body = {"message": {"content": content}, "done": True,
                    "prompt_eval_count": 5, "eval_count": 12}
            if reason is not None:
                body["done_reason"] = reason
        else:
            choice = {"message": {"content": content}}
            if reason is not None:
                choice["finish_reason"] = reason
            body = {"choices": [choice], "usage": {"prompt_tokens": 5, "completion_tokens": 12}}
        return httpx.Response(200, json=body)

    judge = LocalHTTPJudge("test-model", "http://127.0.0.1:1234", dialect,
                           response_format=response_format)
    judge.client.close()
    judge.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(JudgeOutputError, match="Incomplete response") as caught:
            judge.generate("system", "input", ActivityIdentificationLLMOutput)
    finally:
        judge.close()
    assert len(requests) == 1
    assert caught.value.raw_response == content
    assert caught.value.usage == {"input_tokens": 5, "output_tokens": 12}
    assert not caught.value.schema_validation_failed


@pytest.mark.parametrize("done", [False, None])
def test_ollama_requires_completed_response_even_with_stop_reason(done):
    body = {"message": {"content": '{"findings":[]}'}, "done_reason": "stop"}
    if done is not None:
        body["done"] = done
    judge = LocalHTTPJudge("test-model", "http://127.0.0.1:1234", "ollama")
    judge.client.close()
    judge.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)))
    try:
        with pytest.raises(JudgeOutputError, match="Incomplete response"):
            judge.generate("system", "input", CounsellorTurnLLMOutput)
    finally:
        judge.close()


@pytest.mark.parametrize("dialect", ["ollama", "openai_compatible"])
def test_pipeline_does_not_retry_or_salvage_truncated_local_http_output(dialect):
    requests, errors = [], []

    def handler(request):
        requests.append(request)
        content = '{"new_activities":[]}' if len(requests) == 1 else '{"findings":[]}'
        if dialect == "ollama":
            body = {"message": {"content": content}, "done": True, "done_reason": "length"}
        else:
            body = {"choices": [{"message": {"content": content}, "finish_reason": "length"}]}
        return httpx.Response(200, json=body)

    judge = LocalHTTPJudge("test-model", "http://127.0.0.1:1234", dialect)
    judge.client.close()
    judge.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        records, registry, usage = run_turn_annotation(
            {"turns": [{"turn_id": "T0", "role": "counsellor", "text": "How are you?"}]},
            "S1", judge, "activity", "turn", errors=errors,
        )
    finally:
        judge.close()
    assert len(requests) == 2 and usage["requests"] == 2
    assert records == [] and registry == {}
    assert len(errors) == 2
    assert [json.loads(error["raw_response"]) for error in errors] == [
        {"new_activities": []}, {"findings": []},
    ]


@pytest.mark.parametrize("parameters", [
    {"model": "different"}, {"messages": []}, {"stream": True},
    {"response_format": {}}, {"api_key": "secret"}, {"Authorization": "secret"},
])
def test_compatible_parameters_cannot_override_contract_or_store_credentials(parameters) -> None:
    with pytest.raises(ValueError, match="authentication information"):
        LocalHTTPJudge(model_id="user-model", base_url="https://model.example",
                       dialect="openai_compatible", parameters=parameters)


@pytest.mark.parametrize("base_url", [
    "https://user:secret@model.example", "https://model.example?key=secret",
    "https://model.example/#secret", "model.example",
])
def test_model_url_does_not_embed_credentials_in_manifest(base_url) -> None:
    with pytest.raises(ValueError):
        LocalHTTPJudge(model_id="user-model", base_url=base_url, dialect="openai_compatible")


def test_compatible_json_mode_still_rejects_invalid_schema_without_retry() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"findings":"invalid"}'}, "finish_reason": "stop"}],
        })

    judge = LocalHTTPJudge(model_id="user-model", base_url="http://127.0.0.1:1234",
                           dialect="openai_compatible", response_format="json_object")
    judge.client.close()
    judge.client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(JudgeOutputError) as caught:
        judge.generate("system", "user", CounsellorTurnLLMOutput)
    judge.client.close()
    assert len(calls) == 1
    assert caught.value.schema_validation_failed


@pytest.mark.parametrize("effort", ["none", "high"])
def test_mistral_uses_fixed_model_schema_and_only_supported_parameters(monkeypatch, effort):
    requests = []
    final_text = '{"findings":[]}'
    content = final_text if effort == "none" else [
        {"type": "thinking", "thinking": [{"type": "text", "text": "Not an evaluation output."}]},
        {"type": "text", "text": final_text},
    ]

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        })

    monkeypatch.setenv("MISTRAL_API_KEY", "fictional-mistral-key")
    judge = LocalHTTPJudge(model_id=MISTRAL_MODEL_ID, base_url=MISTRAL_BASE_URL,
                           dialect="mistral", parameters={"reasoning_effort": effort})
    judge.client.close()
    judge.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        result = judge.generate("unchanged system", "unchanged input", CounsellorTurnLLMOutput)
    finally:
        judge.close()
    assert len(requests) == 1
    request = requests[0]
    payload = json.loads(request.content)
    assert str(request.url) == "https://api.mistral.ai/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer fictional-mistral-key"
    assert payload["model"] == "mistral-small-2603"
    assert payload["messages"] == [{"role": "system", "content": "unchanged system"},
                                   {"role": "user", "content": "unchanged input"}]
    assert judge.parameters == {
        "temperature": 0.0, "max_tokens": 32768,
        "reasoning_effort": effort, "random_seed": 666,
    }
    assert all(payload[key] == value for key, value in judge.parameters.items())
    assert payload["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "CounsellorTurnLLMOutput", "strict": True,
        "schema": CounsellorTurnLLMOutput.model_json_schema(),
    }}
    assert not {"think", "thinking", "seed", "num_ctx", "max_completion_tokens"} & payload.keys()
    assert "fictional-mistral-key" not in json.dumps(payload)
    assert result.parsed.findings == [] and result.raw_response == final_text
    assert result.usage == {"input_tokens": 12, "output_tokens": 4}


@pytest.mark.parametrize("content,finish_reason", [
    ('{"findings":[]}', "length"), ('{"findings":[]}', "model_length"),
    ('{"findings":[]}', "error"), ('{"findings":[]}', "tool_calls"),
    (None, "stop"), ([{"type": "tool_call", "text": "{}"}], "stop"),
    ([{"type": "text", "text": None}], "stop"),
    ('{"findings":"invalid"}', "stop"),
])
def test_mistral_failed_outputs_preserve_raw_content_usage_without_retry(content, finish_reason):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        })

    judge = LocalHTTPJudge(model_id=MISTRAL_MODEL_ID, base_url=MISTRAL_BASE_URL,
                           dialect="mistral", api_key="test-only")
    judge.client.close()
    judge.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(JudgeOutputError) as caught:
            judge.generate("system", "user", CounsellorTurnLLMOutput)
    finally:
        judge.close()
    assert len(requests) == 1
    assert caught.value.raw_response == (content if isinstance(content, str) else json.dumps(content))
    assert caught.value.usage == {"input_tokens": 3, "output_tokens": 2}
    assert caught.value.schema_validation_failed == (content == '{"findings":"invalid"}')


@pytest.mark.parametrize("parameters", [
    {"reasoning_effort": "low"}, {"reasoning_effort": "max"}, {"think": False},
    {"seed": 666}, {"num_ctx": 16384}, {"headers": {}},
])
def test_mistral_rejects_other_provider_parameters(parameters):
    with pytest.raises(ValueError):
        LocalHTTPJudge(model_id=MISTRAL_MODEL_ID, base_url=MISTRAL_BASE_URL,
                       dialect="mistral", api_key="test-only", parameters=parameters)


def test_mistral_requires_credentials_and_schema_mode():
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        LocalHTTPJudge(model_id=MISTRAL_MODEL_ID, base_url=MISTRAL_BASE_URL, dialect="mistral")
    with pytest.raises(ValueError, match="json_schema"):
        LocalHTTPJudge(model_id=MISTRAL_MODEL_ID, base_url=MISTRAL_BASE_URL,
                       dialect="mistral", api_key="test-only", response_format="json_object")


def test_turn_input_excludes_prior_model_annotations() -> None:
    """The turn judge receives source context and the registry, not previous model judgements."""

    dialogue = {
        "turns": [
            {"turn_id": "T00", "role": "client", "text": "I feel anxious."},
            {"turn_id": "T01", "role": "counsellor", "text": "You feel anxious."},
        ]
    }
    payload = json.loads(build_turn_user_prompt(dialogue, 1, {}))

    assert "prior_annotations" not in payload
    assert payload["visible_turns"] == dialogue["turns"]
    assert payload["activity_registry"] == []
    assert "available_new_activity_ids" not in payload
    assert set(payload["eligible_dimensions"]).isdisjoint(TASK_DIMENSIONS)


def test_activity_identification_input_owns_new_activity_ids() -> None:
    """Only the Activity identification step receives candidate new Activity IDs."""

    dialogue = {
        "turns": [
            {"turn_id": "T00", "role": "client", "text": "I feel anxious."},
            {
                "turn_id": "T01",
                "role": "counsellor",
                "text": "We could write the thoughts down.",
            },
        ]
    }
    payload = json.loads(build_activity_identification_user_prompt(dialogue, 1, {}))

    assert payload["available_new_activity_ids"][0] == "ACTIVITY_01"
    assert payload["activity_registry"] == []


def test_ollama_thinking_is_disabled_at_top_level() -> None:
    """Ollama defaults to disabled thinking while retaining other generation options."""

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Capture the mock request and return minimal valid turn output."""

        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {"content": '{"findings":[]}'},
                "prompt_eval_count": 10,
                "eval_count": 3,
                "done": True,
                "done_reason": "stop",
            },
        )

    judge = LocalHTTPJudge(
        model_id="test-model",
        base_url="http://127.0.0.1:11434",
        dialect="ollama",
        parameters={"temperature": 0, "num_ctx": 16384},
    )
    judge.client = httpx.Client(transport=httpx.MockTransport(handler))
    result = judge.generate("system", "user", CounsellorTurnLLMOutput)

    assert result.parsed.findings == []
    assert judge.parameters["think"] is False
    assert judge.parameters["seed"] == 666
    assert captured["think"] is False
    assert "think" not in captured["options"]
    assert captured["options"]["temperature"] == 0
    assert captured["options"]["seed"] == 666


@pytest.mark.parametrize("thinking,effort", [
    (None, None), ("enabled", None), ("disabled", None),
    ("enabled", "low"), ("enabled", "high"), ("enabled", "max"),
])
def test_deepseek_request_uses_json_mode_and_no_local_parameters(thinking, effort) -> None:
    """Check the interface contract using a fictional key and mocked HTTP."""

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Check authentication and return valid JSON with cached-token counts."""

        assert request.headers["Authorization"] == "Bearer test-only"
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"findings":[]}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4,
                      "prompt_cache_hit_tokens": 8, "prompt_cache_miss_tokens": 4},
        })

    parameters = {} if thinking is None else {"thinking": {"type": thinking}}
    if effort is not None:
        parameters["reasoning_effort"] = effort
    judge = DeepSeekJudge(api_key="test-only", parameters=parameters)
    headers = dict(judge.client.headers)
    judge.client.close()
    judge.client = httpx.Client(transport=httpx.MockTransport(handler), headers=headers)
    result = judge.generate("unchanged system", "original input", CounsellorTurnLLMOutput)
    assert captured["model"] == "deepseek-flash"
    assert captured["thinking"] == {"type": thinking or "disabled"}
    expected = {
        "max_tokens": 32768,
        "thinking": {"type": thinking or "disabled"},
    }
    if thinking == "enabled":
        expected["reasoning_effort"] = effort or "high"
        assert "temperature" not in captured
    else:
        expected["temperature"] = 0.0
        assert "reasoning_effort" not in captured
    assert judge.parameters == expected
    for key, value in expected.items():
        assert captured[key] == value
    assert captured["max_tokens"] == expected["max_tokens"]
    assert captured["response_format"] == {"type": "json_object"}
    assert not {"seed", "num_ctx", "think"} & captured.keys()
    assert captured["messages"][0]["content"] == "unchanged system"
    assert captured["messages"][1]["content"].startswith("original input\n\nReturn a JSON")
    assert result.usage["input_tokens"] == 12
    assert result.usage["prompt_cache_hit_tokens"] == 8
    assert "test-only" not in json.dumps(judge.parameters)
    judge.client.close()


@pytest.mark.parametrize("thinking", ["enabled", "disabled"])
def test_deepseek_explicit_output_limit_is_preserved(thinking) -> None:
    """Explicit output limits override defaults in every mode."""

    judge = DeepSeekJudge(api_key="test-only", parameters={
        "thinking": {"type": thinking}, "max_tokens": 16384,
    })
    assert judge.parameters["max_tokens"] == 16384
    judge.client.close()


def test_deepseek_thinking_rejects_invalid_option() -> None:
    """Reject misspelt thinking values and unsupported local boolean parameters."""

    with pytest.raises(ValueError, match="thinking accepts only"):
        DeepSeekJudge(api_key="test-only", parameters={"thinking": True})
    with pytest.raises(ValueError, match="reasoning_effort accepts only"):
        DeepSeekJudge(api_key="test-only", parameters={"reasoning_effort": "medium"})


def test_openai_request_matches_deepseek_input_contract() -> None:
    """Both providers receive the same input and schema; only provider parameters and authentication differ."""

    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Capture the mock HTTP request and return a valid empty Activity result."""

        assert request.headers["Authorization"] == "Bearer test-only"
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"new_activities":[]}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4,
                      "prompt_tokens_details": {"cached_tokens": 8},
                      "completion_tokens_details": {"reasoning_tokens": 0}},
        })

    for judge_class in (DeepSeekJudge, OpenAIJudge):
        judge = judge_class(api_key="test-only")
        headers = dict(judge.client.headers)
        judge.client.close()
        judge.client = httpx.Client(transport=httpx.MockTransport(handler), headers=headers)
        result = judge.generate("same system", "same input", ActivityIdentificationLLMOutput)
        if judge_class is OpenAIJudge:
            assert result.usage == {"input_tokens": 12, "output_tokens": 4,
                                    "prompt_cache_hit_tokens": 8, "reasoning_tokens": 0}
            assert "test-only" not in json.dumps(judge.parameters)
            assert judge.parameters["reasoning_effort"] == "none"
        judge.client.close()
    assert captured[0]["messages"] == captured[1]["messages"]
    assert captured[0]["response_format"] == captured[1]["response_format"]
    assert captured[1]["model"] == "gpt-5.6-luna"
    assert captured[1]["temperature"] == 0
    assert captured[1]["max_completion_tokens"] == 32768
    assert captured[1]["store"] is False
    assert not {"seed", "num_ctx", "thinking", "max_tokens"} & captured[1].keys()


@pytest.mark.parametrize("content,finish,refusal", [
    ('{"findings":', "stop", None),
    ('{"findings":[]}', "length", None),
    ("", "content_filter", None),
    (None, "stop", "Cannot comply"),
])
def test_openai_failure_retains_original_output_and_usage(content, finish, refusal) -> None:
    """Invalid JSON, truncation, filtering and refusals must fail while retaining usage."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a controlled API failure without contacting external services."""

        return httpx.Response(200, json={
            "choices": [{"message": {"content": content, "refusal": refusal},
                         "finish_reason": finish}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        })

    judge = OpenAIJudge(api_key="test-only")
    judge.client.close()
    judge.client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(JudgeOutputError) as caught:
        judge.generate("system", "input", CounsellorTurnLLMOutput)
    assert caught.value.raw_response == (refusal or content or "")
    assert caught.value.usage == {"input_tokens": 5, "output_tokens": 10}
    judge.client.close()


def test_openai_missing_key_and_unapproved_parameters(monkeypatch) -> None:
    """Reject missing keys and unsupported reasoning settings before sending a request."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        OpenAIJudge()
    for parameters in ({"seed": 666}, {"reasoning_effort": "minimal"}):
        with pytest.raises(ValueError):
            OpenAIJudge(api_key="test-only", parameters=parameters)


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh", "max"])
@pytest.mark.parametrize("explicit_limit", [None, 12345])
def test_openai_reasoning_effort_is_sent_without_hidden_temperature(effort, explicit_limit) -> None:
    """Request parameters match the manifest; explicit limits override the 32768 default."""

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Capture the mock request and return valid empty Activity identification output."""

        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"new_activities":[]}'}, "finish_reason": "stop"}],
        })

    parameters = {"reasoning_effort": effort, "temperature": 0.2}
    if explicit_limit is not None:
        parameters["max_completion_tokens"] = explicit_limit
    judge = OpenAIJudge(api_key="test-only", parameters=parameters)
    judge.client.close()
    judge.client = httpx.Client(transport=httpx.MockTransport(handler))
    judge.generate("system", "input", ActivityIdentificationLLMOutput)
    assert captured["reasoning_effort"] == effort
    assert captured["max_completion_tokens"] == (
        explicit_limit if explicit_limit is not None else 32768
    )
    assert ("temperature" in captured) == (effort == "none")
    assert ("temperature" in judge.parameters) == (effort == "none")
    if effort == "none":
        assert captured["temperature"] == 0.2
    assert all(captured[key] == value for key, value in judge.parameters.items())
    judge.client.close()


def test_deepseek_invalid_json_preserves_response_and_usage() -> None:
    """JSON parsing failures retain original output and token usage."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Simulate a truncated model response."""

        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"findings":'}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        })

    judge = DeepSeekJudge(api_key="test-only")
    judge.client.close()
    judge.client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(JudgeOutputError) as caught:
        judge.generate("system", "input", CounsellorTurnLLMOutput)
    assert caught.value.raw_response == '{"findings":'
    assert caught.value.usage == {"input_tokens": 100, "output_tokens": 10}
    judge.client.close()


def test_diagnostic_summary_excludes_unresolved_turn_statuses() -> None:
    """List unresolved calls explicitly rather than treating them as no evaluable opportunity."""

    failures = [{"current_turn_id": "T0", "stage": "turn_annotation"}]
    payload = json.loads(build_summary_user_prompt({"turns": []}, [], {}, failures))
    assert payload["failed_annotations"] == failures
    assert set(payload["required_summary_targets"]) == {"summary_level_ratings"}
    assert len(payload["required_summary_targets"]["summary_level_ratings"]) == 3


def test_deepseek_truncated_output_is_not_accepted_even_if_json_parses() -> None:
    """A token-limited response must not count as complete merely because its JSON is valid."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Simulate output explicitly marked length by the service."""

        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"findings":[]}'},
                         "finish_reason": "length"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 8192},
        })

    judge = DeepSeekJudge(api_key="test-only")
    judge.client.close()
    judge.client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(JudgeOutputError, match="length") as caught:
        judge.generate("system", "input", CounsellorTurnLLMOutput)
    assert caught.value.usage["output_tokens"] == 8192
    judge.client.close()
