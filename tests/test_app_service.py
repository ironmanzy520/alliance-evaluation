"""Test service control with fake processes; never signal real processes."""

from pathlib import Path
from unittest.mock import Mock

import pytest

psutil = pytest.importorskip("psutil")

from alliance_eval import app_service, cli


@pytest.fixture(autouse=True)
def isolate_processes(monkeypatch):
    monkeypatch.setattr(psutil, "process_iter", Mock(return_value=[]))
    monkeypatch.setattr(psutil, "Process", Mock(return_value=Mock(username=lambda: "tester")))
    monkeypatch.setattr(psutil, "Popen", Mock(side_effect=AssertionError("Unexpected process launch")))


def fake_process(workspace: Path, port: int = 8501):
    process = Mock(pid=123)
    process.cmdline.return_value = app_service._command(port)
    process.info = {"cmdline": process.cmdline.return_value}
    process.username.return_value = "tester"
    process.environ.return_value = {"ALLIANCE_EVAL_WORKSPACE": str(workspace)}
    process.is_running.return_value = True
    process.wait.return_value = 0
    return process


def register_process(monkeypatch, process):
    monkeypatch.setattr(psutil, "process_iter", Mock(return_value=[process]))
    monkeypatch.setattr(psutil, "Process", Mock(side_effect=lambda pid=None: (
        Mock(username=lambda: "tester") if pid is None else process
    )))


def test_find_existing_cli_process_without_pid_file(tmp_path, monkeypatch):
    process = fake_process(tmp_path)
    register_process(monkeypatch, process)
    assert app_service._find_server(tmp_path, 8501) is process
    assert list(tmp_path.iterdir()) == []
    process.terminate.assert_not_called()


@pytest.mark.parametrize("mismatch", ["port", "script", "workspace", "user", "gone", "missing_workspace", "changed_command"])
def test_identity_mismatch_never_selected(tmp_path, monkeypatch, mismatch):
    process = fake_process(tmp_path)
    if mismatch == "port":
        process.cmdline.return_value = app_service._command(8502)
        process.info["cmdline"] = process.cmdline.return_value
    elif mismatch == "script":
        process.cmdline.return_value[4] = "/another/app.py"
    elif mismatch == "workspace":
        process.environ.return_value["ALLIANCE_EVAL_WORKSPACE"] = str(tmp_path / "other")
    elif mismatch == "user":
        process.username.return_value = "other"
    elif mismatch == "gone":
        process.is_running.return_value = False
    elif mismatch == "missing_workspace":
        process.environ.return_value = {}
    else:
        process.cmdline.return_value = ["python", "another.py"]
    register_process(monkeypatch, process)
    assert app_service._find_server(tmp_path, 8501) is None
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


def test_inaccessible_workspace_is_not_guessed(tmp_path, monkeypatch):
    process = fake_process(tmp_path)
    process.environ.side_effect = psutil.AccessDenied(123)
    register_process(monkeypatch, process)
    with pytest.raises(RuntimeError, match="Cannot verify"):
        app_service._find_server(tmp_path, 8501)
    process.terminate.assert_not_called()


def test_disappeared_candidate_is_ignored(tmp_path, monkeypatch):
    process = fake_process(tmp_path)
    process.cmdline.side_effect = psutil.NoSuchProcess(123)
    register_process(monkeypatch, process)
    assert app_service._find_server(tmp_path, 8501) is None


def test_ambiguous_candidates_are_not_stopped(tmp_path, monkeypatch):
    process = fake_process(tmp_path)
    register_process(monkeypatch, process)
    monkeypatch.setattr(psutil, "process_iter", Mock(return_value=[process, process]))
    with pytest.raises(RuntimeError, match="Multiple"):
        app_service._find_server(tmp_path, 8501)
    process.terminate.assert_not_called()


def test_normal_stop_waits_without_kill(tmp_path):
    process = fake_process(tmp_path)
    app_service._stop_process(process, force=False)
    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=10)
    process.kill.assert_not_called()


@pytest.mark.parametrize("force", [False, True])
def test_force_only_after_stop_timeout(tmp_path, force):
    process = fake_process(tmp_path)
    process.wait.side_effect = [psutil.TimeoutExpired(10, pid=123), 0]
    if force:
        app_service._stop_process(process, force=True)
        process.kill.assert_called_once_with()
        assert process.wait.call_count == 2
    else:
        with pytest.raises(RuntimeError, match="--force"):
            app_service._stop_process(process, force=False)
        process.kill.assert_not_called()


def test_force_still_reports_failure_to_exit(tmp_path):
    process = fake_process(tmp_path)
    process.wait.side_effect = psutil.TimeoutExpired(10, pid=123)
    with pytest.raises(RuntimeError, match="still running"):
        app_service._stop_process(process, force=True)


def test_pid_disappearance_or_reuse_is_not_killed(tmp_path):
    process = fake_process(tmp_path)
    process.terminate.side_effect = psutil.NoSuchProcess(123)
    app_service._stop_process(process, force=True)
    process.kill.assert_not_called()


@pytest.mark.parametrize("action", ["start", "status", "stop", "restart"])
def test_unknown_occupied_port_is_not_stopped_or_restarted(tmp_path, monkeypatch, action):
    monkeypatch.setattr(app_service, "_find_server", Mock(return_value=None))
    probe = Mock(side_effect=RuntimeError("Port is occupied"))
    monkeypatch.setattr(app_service, "_ensure_port_available", probe)
    with pytest.raises(RuntimeError, match="occupied"):
        app_service.manage_app(action, tmp_path, 8501, force=True)
    psutil.Popen.assert_not_called()


def test_status_and_repeated_stop_are_safe(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(app_service, "_ensure_port_available", Mock())
    for action in ("status", "stop", "stop"):
        assert app_service.manage_app(action, tmp_path, 8501) == 0
    assert "App is not running" in capsys.readouterr().out
    psutil.Popen.assert_not_called()


def test_system_process_inspection_denied_is_explained(tmp_path, monkeypatch):
    monkeypatch.setattr(psutil, "process_iter", Mock(side_effect=PermissionError("denied")))
    with pytest.raises(RuntimeError, match="terminal has permission"):
        app_service.manage_app("stop", tmp_path, 8501, force=True)
    psutil.Popen.assert_not_called()


def test_running_status_and_duplicate_start(tmp_path, monkeypatch, capsys):
    process = fake_process(tmp_path)
    register_process(monkeypatch, process)
    assert app_service.manage_app("status", tmp_path, 8501) == 0
    assert "PID 123" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="already running on port"):
        app_service.manage_app("start", tmp_path, 8501)
    process.terminate.assert_not_called()
    psutil.Popen.assert_not_called()


def test_restart_orders_stop_port_check_start(tmp_path, monkeypatch):
    process = fake_process(tmp_path)
    events = []
    monkeypatch.setattr(app_service, "_find_server", lambda *args: process)
    monkeypatch.setattr(app_service, "_stop_process", lambda *args, **kwargs: events.append("stop"))
    monkeypatch.setattr(app_service, "_ensure_port_available", lambda *args: events.append("port"))
    monkeypatch.setattr(app_service, "_start_server", lambda *args: events.append("start") or 0)
    assert app_service.manage_app("restart", tmp_path, 8501) == 0
    assert events == ["stop", "port", "start"]


def test_failed_stop_prevents_restart(tmp_path, monkeypatch):
    process = fake_process(tmp_path)
    register_process(monkeypatch, process)
    process.wait.side_effect = psutil.TimeoutExpired(10, pid=123)
    with pytest.raises(RuntimeError, match="did not exit"):
        app_service.manage_app("restart", tmp_path, 8501)
    psutil.Popen.assert_not_called()


def test_ctrl_c_waits_for_owned_child_cleanup(tmp_path, monkeypatch):
    process = fake_process(tmp_path)
    process.wait.side_effect = [KeyboardInterrupt(), 0]
    launch = Mock(return_value=process)
    monkeypatch.setattr(psutil, "Popen", launch)
    monkeypatch.setattr(app_service, "_ensure_port_available", Mock())
    assert app_service._start_server(tmp_path, 8502) == 0
    assert launch.call_args.args[0] == app_service._command(8502)
    assert launch.call_args.kwargs["env"]["ALLIANCE_EVAL_WORKSPACE"] == str(tmp_path)
    process.terminate.assert_called_once_with()
    assert process.wait.call_count == 2
    process.kill.assert_not_called()


def test_port_probe_distinguishes_occupancy_from_other_errors(monkeypatch):
    probe = Mock()
    context = Mock(__enter__=Mock(return_value=probe), __exit__=Mock(return_value=False))
    monkeypatch.setattr(app_service.socket, "socket", Mock(return_value=context))
    probe.bind.side_effect = OSError(app_service.errno.EADDRINUSE, "occupied")
    with pytest.raises(RuntimeError, match="no other process was stopped"):
        app_service._ensure_port_available(8501)
    probe.bind.side_effect = OSError(app_service.errno.EACCES, "denied")
    with pytest.raises(OSError, match="denied"):
        app_service._ensure_port_available(8501)


@pytest.mark.parametrize("action", [None, "start", "status", "stop", "restart"])
def test_cli_default_and_explicit_actions(tmp_path, monkeypatch, action):
    command = ["app"] + ([action] if action else []) + ["--workspace", str(tmp_path), "--port", "8502"]
    args = cli.build_parser().parse_args(command)
    manager = Mock(return_value=0)
    monkeypatch.setattr(app_service, "manage_app", manager)
    assert cli._execute(args) == 0
    manager.assert_called_once_with(action or "start", tmp_path, 8502, force=False)


@pytest.mark.parametrize("arguments,message", [
    (["app", "--port", "0"], "--port"),
    (["app", "--workspace", "/nonexistent-app-workspace"], "--workspace"),
    (["app", "status", "--force"], "--force"),
    (["app", "start", "--force"], "--force"),
])
def test_cli_rejects_invalid_control_arguments(arguments, message):
    with pytest.raises(ValueError, match=message):
        cli._execute(cli.build_parser().parse_args(arguments))
