"""Command-line entry points for input preparation, offline demonstration, evaluation and the app."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from alliance_eval import __version__
from alliance_eval.evaluate import DeepSeekJudge, LocalHTTPJudge, OpenAIJudge, MISTRAL_BASE_URL, MISTRAL_MODEL_ID
from alliance_eval.evaluation_models import SamplePlan, export_json_schemas
from alliance_eval.input_data import create_sample_plan, export_input_schema, parse_json_value, prepare_data
from alliance_eval.pipeline import check_runtime_version, load_selected_dialogues, run_all
from alliance_eval.resources import code_identity


def build_parser() -> argparse.ArgumentParser:
    """Resolve paths against the working directory rather than the source installation."""

    parser = argparse.ArgumentParser(description="Bond/Task dialogue evaluation")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check-runtime", help="Check runtime code and schema identity without model calls")

    demo = commands.add_parser("demo", help="Demonstrate the pipeline with fictional dialogue and preset responses; no model calls")
    demo.add_argument("--workspace", type=Path, default=Path.cwd())
    demo.add_argument("--run-id", help="New demonstration run name; existing directories are not overwritten")

    prepare = commands.add_parser("prepare-data", help="Preview and convert local dialogue using explicit field and role mappings")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--format", choices=["csv", "tsv", "json", "jsonl", "xlsx", "txt", "md"])
    prepare.add_argument("--sheet", help="XLSX sheet name; required when the workbook has multiple sheets")
    prepare.add_argument("--dialogue-id", help="Required for TXT/MD: confirm that the whole file is one dialogue and specify its ID")
    prepare.add_argument("--dataset", help="Output basename; defaults to the input stem and does not select conversion rules")
    prepare.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    prepare.add_argument("--columns", default="{}", help="JSON mapping from canonical fields to source fields")
    prepare.add_argument("--role-map", help="JSON mapping from source roles to counsellor/client/other; unnecessary for canonical roles")
    prepare.add_argument("--dry-run", action="store_true", help="Validate and preview all records without creating files")

    plan = commands.add_parser("create-plan", help="Select complete dialogues from converted input without model calls")
    plan.add_argument("--input", type=Path, required=True)
    plan.add_argument("--output", type=Path, default=Path("data/plans/sample_plan.json"))
    selection = plan.add_mutually_exclusive_group(required=True)
    selection.add_argument("--dialogue-id", action="append", help="Dialogue ID to select; repeat for multiple dialogues")
    selection.add_argument("--all-dialogues", action="store_true", help="Explicitly select all dialogues; does not run evaluation")

    export = commands.add_parser("export-schema", help="Export the input schema and six output schemas to a new directory")
    export.add_argument("--output-dir", type=Path, required=True)

    validate = commands.add_parser("validate-input", help="Validate the plan and selected dialogues without model calls")
    validate.add_argument("--sample-file", type=Path, required=True)
    validate.add_argument("--data-root", type=Path, help="Override the base for relative source paths; defaults to the plan directory")

    run = commands.add_parser("all", help="Run Activity identification, turn annotation, summary and Core mapping")
    run.add_argument("--sample-file", type=Path, required=True)
    run.add_argument("--data-root", type=Path, help="Explicit base directory for relative source paths in the plan")
    run.add_argument("--run-root", type=Path, default=Path("runs"))
    run.add_argument("--run-id")
    run.add_argument("--model", help="Model ID; defaults to the reference model for the selected provider")
    run.add_argument("--base-url", help="Service URL for ollama/openai_compatible")
    run.add_argument("--dialect", choices=["deepseek", "openai", "mistral", "ollama", "openai_compatible"], default="deepseek")
    run.add_argument("--parameters", default="{}", help="Model parameters as JSON")
    run.add_argument("--timeout-seconds", type=float, default=600.0)
    run.add_argument("--api-key-env", default="ALLIANCE_API_KEY", help="Environment variable containing the Ollama/compatible API key; do not pass the key itself")
    run.add_argument("--response-format", choices=["json_schema", "json_object"], default="json_schema", help="Compatible API output mode; no automatic downgrade or output-error retries")
    run.add_argument("--allow-model-calls", action="store_true", help="Confirm data-processing permission and authorise model calls and their cost")

    app = commands.add_parser("app", help="Start, inspect, stop or restart the local app without model calls")
    app.add_argument("action", nargs="?", choices=["start", "status", "stop", "restart"], default="start")
    app.add_argument("--workspace", type=Path, default=Path.cwd())
    app.add_argument("--port", type=int, default=8501)
    app.add_argument("--force", action="store_true", help="For stop/restart only: force termination of the identified app after normal shutdown times out")
    return parser


def _json_mapping(text: str, name: str) -> dict:
    value = parse_json_value(text)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _execute(args: argparse.Namespace) -> int:
    if args.command == "demo":
        from alliance_eval.demo import run_demo

        print(run_demo(args.workspace, run_id=args.run_id))
        return 0

    if args.command == "check-runtime":
        print(json.dumps({**code_identity(), "runtime": check_runtime_version()}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "prepare-data":
        result = prepare_data(
            args.input, args.output_dir, input_format=args.format,
            columns=_json_mapping(args.columns, "--columns"),
            role_map=_json_mapping(args.role_map, "--role-map") if args.role_map is not None else None,
            dry_run=args.dry_run,
            dataset=args.dataset, sheet=args.sheet, dialogue_id=args.dialogue_id,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "create-plan":
        result = create_sample_plan(args.input, args.output, dialogue_ids=args.dialogue_id, select_all=args.all_dialogues)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "export-schema":
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        export_json_schemas(output_dir)
        export_input_schema(output_dir / "dialogue.schema.json")
        print(output_dir)
        return 0

    if args.command == "validate-input":
        sample_file = args.sample_file.resolve()
        plan = SamplePlan.model_validate(parse_json_value(sample_file.read_text(encoding="utf-8")))
        data_root = args.data_root.resolve() if args.data_root else sample_file.parent
        dialogues = load_selected_dialogues(plan, data_root)
        print(json.dumps({
            "sample_version": plan.sample_version, "samples": len(dialogues),
            "turns": sum(len(dialogue["turns"]) for dialogue in dialogues),
            "data_root": str(data_root),
        }, ensure_ascii=False, indent=2))
        return 0

    if args.command == "app":
        if not 1 <= args.port <= 65535:
            raise ValueError("--port must be between 1 and 65535")
        workspace = args.workspace.resolve()
        if not workspace.is_dir():
            raise ValueError("--workspace must be an existing local directory")
        if args.force and args.action not in {"stop", "restart"}:
            raise ValueError("--force is only valid with app stop or app restart")
        try:
            from alliance_eval.app_service import manage_app
        except ImportError as error:
            raise ValueError('App management requires optional dependencies: python -m pip install ".[app]"') from error
        return manage_app(args.action, workspace, args.port, force=args.force)

    # Require model-call authorisation before reading credentials or constructing a backend.
    if not args.allow_model_calls:
        raise ValueError("Model calls are not authorised. Run validate-input first, then pass --allow-model-calls after confirming data permissions and cost.")
    parameters = _json_mapping(args.parameters, "--parameters")
    if args.dialect in {"deepseek", "openai"}:
        judge_class = DeepSeekJudge if args.dialect == "deepseek" else OpenAIJudge
        model_id = args.model or ("deepseek-flash" if args.dialect == "deepseek" else "gpt-5.6-luna")
        backend = judge_class(model_id=model_id, parameters=parameters, timeout_seconds=args.timeout_seconds)
    elif args.dialect == "mistral":
        backend = LocalHTTPJudge(
            model_id=args.model or MISTRAL_MODEL_ID, base_url=MISTRAL_BASE_URL,
            dialect="mistral", parameters=parameters, timeout_seconds=args.timeout_seconds,
            response_format=args.response_format,
        )
    else:
        if not args.base_url or not args.model:
            raise ValueError("ollama/openai_compatible requires --base-url and --model")
        backend = LocalHTTPJudge(
            model_id=args.model, base_url=args.base_url, dialect=args.dialect,
            parameters=parameters, timeout_seconds=args.timeout_seconds,
            api_key=os.environ.get(args.api_key_env), response_format=args.response_format,
        )
    try:
        run_dir = run_all(
            sample_plan_path=args.sample_file, backend=backend,
            run_root=args.run_root, run_id=args.run_id, data_root=args.data_root,
        )
    finally:
        close = getattr(backend, "close", None)
        if close is not None:
            close()
    print(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return 0 if manifest["status"] == "completed" else 1


def main() -> int:
    """Report errors with a non-zero exit code; do not repair input or rerun the pipeline."""

    parser = build_parser()
    args = parser.parse_args()
    try:
        return _execute(args)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
