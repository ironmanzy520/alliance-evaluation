# Model configuration and execution

Try the complete workflow without a model or key by starting the App from the [quick start](../README.md) and clicking `Run preset demo`. Real evaluation requires a running model service and a prepared sample plan. This project does not install model services, download models or verify permission to transmit data.

## App configuration

Run `alliance-eval app --workspace .`, choose an existing `Sample plan`, then set `Provider`, `Model ID` and the relevant address/key. Once input checks pass, clicking `Run All` sends the selected dialogues to that service and may incur charges. Confirm data-processing permission and costs first. Opening the App, viewing runs or changing settings does not call models. Manually entered keys remain in the current session and are not written to runs.

The four built-in entries below appear by default. Custom connections appear under their registered names in the same `Provider` selector only after being added as described in the next section.

| Provider | Interface and configuration |
|:---|:---|
| `ollama` | A running Ollama service, for example `http://127.0.0.1:11434`; use an actually available model ID; requests use `/api/chat` and a `format` schema |
| `openai` / `deepseek` | Direct interfaces using `OPENAI_API_KEY` / `DEEPSEEK_API_KEY`, or the App password field; model availability depends on the account |
| `mistral` | Official `https://api.mistral.ai/v1/chat/completions`, using `MISTRAL_API_KEY` or the App password field, with `json_schema` output |

The App prefills `deepseek-flash`, `gpt-5.6-luna` and Mistral Small 4's fixed ID `mistral-small-2603`, avoiding Mistral's `latest` alias, which may change as the service is updated. These are the project's reference models, directly matched to its existing interfaces and parameter configurations. Their defaults reduce initial setup work. Ollama's model ID is supplied by the user.

DeepSeek currently maps `deepseek-flash` to DeepSeek-V4.1-Flash. The older `deepseek-v4-flash` is a compatibility alias and no longer denotes the previous V4 Flash model. Check the [official model information](https://api-docs.deepseek.com/quick_start/pricing/) for the mapping: `deepseek-flash` is not a dated snapshot, so research runs must record the run date and then-current version description. The App and backend explicitly default to `thinking.type=disabled`, `temperature=0` and `max_tokens=32768`, avoiding the service's default of enabling thinking. DeepSeek, OpenAI and Mistral use a default 32,768 output-token limit in identification, turn evaluation and summary, regardless of the Thinking toggle. OpenAI uses `max_completion_tokens`. Explicit limits take precedence; truncation fails without an automatic limit increase or retry.

Model names remain editable, but changing a name does not adapt Thinking, reasoning effort, Temperature or output-format capabilities. The existing DeepSeek/OpenAI/Mistral controls are configured for the reference models above. Check parameters and output formats for the selected model and adapt the interface where necessary. Identical names or “compatible” endpoints need not offer identical capabilities. Defaults do not promise account availability or judgement quality, nor do they limit the project to these three models.

Mistral defaults to `reasoning_effort="none"`, `temperature=0`, `max_tokens=32768` and `random_seed=666`. Thinking offers `high` for the reference model without changing the default output limit. The backend parses final text blocks only; truncation, abnormal termination and unsupported blocks remain failures, and reasoning text is not rating output. Parameters are recorded in the manifest; a fixed seed is not research evidence of repeatability. See [Mistral Small 4](https://docs.mistral.ai/models/mistral-small-4-0-26-03), [reasoning](https://docs.mistral.ai/studio/conversations/reasoning) and the [API](https://docs.mistral.ai/api). The official 256K context limit is not a program-configured `num_ctx` setting; capacity must account for complete prompts, registry, summary evidence and output allowance, not turn count alone.

### Add a custom Provider

Create optional `providers.json` at the root selected by App `--workspace`:

```json
{
  "My model": {
    "dialect": "openai_compatible",
    "base_url": "https://model.example/v1",
    "model_id": "your-model-id",
    "api_key_env": "MY_MODEL_API_KEY",
    "response_format": "json_schema",
    "parameters": {}
  }
}
```

Replace the placeholder URL and model, then refresh to select `My model`. Multiple distinct connections are allowed; built-in names cannot be overridden. Without this file, no extra configuration is needed. Invalid configuration displays an error and disables `Run All`, without silently switching connections.

- `dialect` is a protocol, not a brand. Supported values are `openai_compatible` (default, `/v1/chat/completions`, with or without trailing `/v1` in the base URL) and `ollama` (`/api/chat`). No scoring-code change is needed.
- `model_id` may be omitted and entered later in the App. The address is read-only there; edit the file and refresh to change the address, parameters or default model.
- `api_key_env` names an environment variable; omit it for unauthenticated services. Supply its value through that variable or the App password field. A declared but unavailable key prevents execution. Environment variables must reach the App process, potentially requiring a restart. Changes to a connection name or configuration invalidate manually entered credentials.
- `response_format` defaults to `json_schema`. Explicitly use `json_object` only when the compatible service supports JSON objects but not schema mode. Ollama uses only `json_schema`. There is no automatic downgrade or output-error retry.
- `parameters` defaults to an empty object. Supply only parameters supported by the target model. Custom connections do not receive generic Thinking/Temperature controls or built-in reference-model defaults. Ollama still defaults to `seed=666` and `think=false`, which can be explicitly overridden.

`providers.json` is local configuration excluded from Git and installation packages. Do not store keys, authentication headers or credential-bearing URLs in it. Registration does not prove compatibility or authorise data transmission. Other protocols require a backend adapter, not an invented `dialect` value.

## CLI examples

First perform model-free checks:

```bash
alliance-eval check-runtime
alliance-eval validate-input --sample-file data/plans/sample_plan.json
```

The following commands actually transmit selected dialogues and require authorisation. Replace `<model-id>` with a confirmed ID. If a different source root is required, add `--data-root` to both validation and execution:

```bash
alliance-eval all --sample-file data/plans/sample_plan.json \
  --dialect ollama --base-url http://127.0.0.1:11434 --model '<model-id>' \
  --parameters '{"temperature":0,"num_ctx":16384,"seed":666}' \
  --allow-model-calls
```

For a compatible API, securely set `ALLIANCE_API_KEY` locally; unauthenticated local services may omit it. Pass the variable name, not the key:

```bash
alliance-eval all --sample-file data/plans/sample_plan.json \
  --dialect openai_compatible --base-url https://model.example/v1 \
  --model '<model-id>' --response-format json_schema \
  --api-key-env ALLIANCE_API_KEY --allow-model-calls
```

`model.example` is a placeholder. `json_schema` requests strict schema output. `json_object` requests valid JSON and also supplies the same generated response schema in the user message; JSON mode itself does not enforce that structure. Both modes retain identical runtime structural and evidence validation, without automatic downgrade or output-error retries. Ollama uses its own `format` schema, not `json_object`.

`--parameters` accepts supported service parameters, not addresses, keys, headers or request structure. Ollama's `num_ctx` is not a generic API parameter. Its default `seed=666` and `think=false` are recorded in the manifest. `--timeout-seconds` sets each request timeout; `--run-root` and `--run-id` choose a new output location without overwriting existing directories. See `alliance-eval all --help` for all options.

For official Mistral, securely set `MISTRAL_API_KEY` and, after authorisation, run:

```bash
alliance-eval all --sample-file data/plans/sample_plan.json \
  --dialect mistral --model mistral-small-2603 \
  --parameters '{"reasoning_effort":"none","temperature":0,"max_tokens":32768,"random_seed":666}' \
  --allow-model-calls
```

This entry does not require `--base-url` and uses `MISTRAL_API_KEY`, not the generic `--api-key-env`. It requests `json_schema` without a JSON-object fallback and uses the same pipeline validation.

## Other protocols

Adapt the existing minimal [JudgeBackend](../src/alliance_eval/evaluate.py) protocol: supply `model_id` and implement `generate(system_prompt, user_prompt, response_model) -> JudgeCall`, returning validated structure, raw response and available usage. Reuse `parse_judge_response` where appropriate. Backends needing cleanup can expose a no-argument `close()`; CLI and App call it on completion or failure. Preserve scoring schemas, evidence and failure records.

Keep `run_all` as the orchestrator. To expose a new protocol in the App, also extend `load_custom_providers` validation and `_run_with_progress` backend dispatch in [app.py](../src/alliance_eval/app.py), then register the connection. Verify selection, requests, authentication, accepted responses and failure retention with original fictional data before authorised integration. Adding a backend does not broaden model or data permissions. Keys must not enter URLs, parameters, log descriptions or results.

## Results and failures

Ollama responses must have `done: true` and `done_reason: "stop"`. Compatible Chat Completions responses must have `finish_reason: "stop"`. Missing or abnormal completion metadata, including length truncation, remains a failed call even when the returned text happens to parse as valid JSON. Raw content and available usage are retained; rejected responses are not salvaged as successful results.

Only `ConnectError` and `ConnectTimeout` (connection-establishment failures) receive at most two additional attempts, after one and two seconds, with unchanged inputs, parameters and output contracts. Exhaustion records a final failure and continues the existing workflow. Read/write timeouts, HTTP status errors and invalid model output are not retried. Fatal authentication, balance and parameter errors retain their termination behaviour. There is no automatic resume.

Each attempt increments `usage.requests`; each failed attempt increments `counts.failed_calls` and retains an error record. Recovered output is used normally, without treating intermediate connection failures as final missing results. The manifest records retry policy; see [run artefacts](../runs/README.md).

Inspect the manifest and `validation_errors.jsonl`, reporting Bond and Task completeness separately. Connection failure, structural error, mismatched evidence, valid empty findings and insufficient evidence are distinct. Do not repair returns or rerun automatically to make status appear successful. See [result review](review.md).
