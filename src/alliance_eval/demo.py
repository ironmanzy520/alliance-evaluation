"""Run an explicitly scripted illustration through the ordinary evaluation pipeline.

The bundled dialogues and preset responses are original fictional teaching material.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from alliance_eval.evaluate import JudgeCall, OutputModel, parse_judge_response
from alliance_eval.pipeline import ProgressCallback, run_all
from alliance_eval.resources import resource_directory


class DemoJudge:
    """Consume the bundled dataset's ordered, labelled preset responses."""

    model_id = "scripted-illustration"
    dialect = "preset_demo"

    def __init__(
        self, dialogues: list[dict[str, Any]],
        responses: dict[str, list[dict[str, Any]]],
    ) -> None:
        dialogue_ids = [dialogue["dialogue_id"] for dialogue in dialogues]
        if not dialogue_ids or len(set(dialogue_ids)) != len(dialogue_ids) or set(dialogue_ids) != set(responses):
            raise ValueError("Demonstration dialogues and response groups must match exactly.")
        self._calls: list[tuple[dict[str, Any], list[dict[str, str]]]] = []
        for dialogue in dialogues:
            turns = [
                {key: turn[key] for key in ("turn_id", "role", "text")}
                for turn in dialogue["turns"]
            ]
            entries = responses[dialogue["dialogue_id"]]
            if len(entries) != 2 * len(turns) + 1:
                raise ValueError("The demonstration requires two presets per turn and one summary.")
            for index, entry in enumerate(entries):
                visible = turns if index == len(entries) - 1 else turns[:index // 2 + 1]
                self._calls.append((entry, visible))
        self._position = 0
        self.parameters = {
            "purpose": "scripted illustration; not model output or an expert gold standard",
            "network_calls": False,
        }

    def generate(
        self, system_prompt: str, user_prompt: str,
        response_model: type[OutputModel],
    ) -> JudgeCall:
        """Check call order and exact visible text; never invent missing responses."""

        if self._position >= len(self._calls):
            raise RuntimeError("The demonstration response queue is exhausted.")
        entry, expected_text = self._calls[self._position]
        payload = json.loads(user_prompt)
        if (
            response_model.__name__ != entry["response_model"]
            or payload.get("current_turn_id") != entry["current_turn_id"]
        ):
            raise RuntimeError(f"Demonstration call order mismatch at call {self._position + 1}.")
        if entry["current_turn_id"] is None:
            received_text = payload.get("completed_sample")
        else:
            received_text = payload.get("visible_turns")
        if received_text != expected_text:
            raise RuntimeError("Preset demonstration cannot annotate other or modified dialogue text.")
        self._position += 1
        return parse_judge_response(json.dumps(entry["output"], ensure_ascii=False), {}, response_model)

    def ensure_consumed(self) -> None:
        """A complete illustration cannot silently omit trailing responses."""

        if self._position != len(self._calls):
            raise RuntimeError("The demonstration did not consume every preset response.")


def run_demo(
    workspace: Path, run_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Create persistent, non-overwriting input copies and run without a model."""

    if run_id is None:
        run_id = f"{datetime.now():%Y%m%d_%H%M%S}_demo_{uuid4().hex[:8]}"
    if run_id in {".", ".."} or not run_id.strip() or any(char in run_id for char in "/\\"):
        raise ValueError("run_id must be a single non-empty directory name.")
    workspace = workspace.expanduser().resolve()
    run_root = workspace / "runs"
    input_dir = workspace / "data" / "demo" / run_id
    if (run_root / run_id).exists() or (run_root / run_id).is_symlink():
        raise FileExistsError(f"Run already exists: {run_root / run_id}")
    source_dir = resource_directory("examples") / "demo"
    inputs = {
        name: (source_dir / name).read_bytes()
        for name in ("dialogue.jsonl", "sample_plan.json", "responses.json")
    }
    dialogue_content = inputs["dialogue.jsonl"]
    response_content = inputs["responses.json"]
    dialogues = [json.loads(line) for line in dialogue_content.decode("utf-8").splitlines() if line.strip()]
    backend = DemoJudge(dialogues, json.loads(response_content))
    backend.parameters["responses_sha256"] = hashlib.sha256(response_content).hexdigest()
    input_dir.mkdir(parents=True, exist_ok=False)
    for name, content in inputs.items():
        (input_dir / name).write_bytes(content)
    run_dir = run_all(
        sample_plan_path=input_dir / "sample_plan.json", backend=backend,
        run_root=run_root, run_id=run_id, progress_callback=progress_callback,
    )
    backend.ensure_consumed()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise RuntimeError(f"Demonstration validation failed; inspect preserved results in {run_dir}.")
    return run_dir
