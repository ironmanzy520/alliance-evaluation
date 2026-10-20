"""Manage CLI-started local apps for the same installation, user, workspace and port."""

from __future__ import annotations

import errno
import os
import socket
import sys
from pathlib import Path

import psutil

from alliance_eval.resources import PACKAGE_DIR

STOP_TIMEOUT_SECONDS = 10


def _command(port: int) -> list[str]:
    return [
        sys.executable, "-m", "streamlit", "run", str(PACKAGE_DIR / "app.py"),
        "--server.address", "127.0.0.1", "--server.port", str(port),
        "--server.headless", "true", "--browser.gatherUsageStats", "false",
    ]


def _find_server(workspace: Path, port: int) -> psutil.Process | None:
    """Identify the app from its process, not from a port number or reusable PID file."""

    expected = _command(port)[1:]
    username = psutil.Process().username()
    matches = []
    for candidate in psutil.process_iter(["cmdline"]):
        if (candidate.info["cmdline"] or [])[1:] != expected:
            continue
        try:
            # Refresh cached process objects and retain PID and creation time for identity checks.
            process = psutil.Process(candidate.pid)
            if process.cmdline()[1:] != expected or process.username() != username:
                continue
            location = process.environ().get("ALLIANCE_EVAL_WORKSPACE")
            if location and Path(location).resolve() == workspace and process.is_running():
                matches.append(process)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as error:
            raise RuntimeError("Cannot verify the app workspace; no process was stopped. Use the same user and installation that started the app.") from error
    if len(matches) > 1:
        raise RuntimeError("Multiple app processes match this workspace and port; no process was stopped.")
    return matches[0] if matches else None


def _ensure_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if os.name != "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                raise RuntimeError(
                    f"Port {port} is occupied, but its process cannot be identified as this workspace's app; no other process was stopped. "
                    "Check the original --workspace, --port and installation, or choose another port."
                ) from error
            raise


def _stop_process(process: psutil.Process, *, force: bool) -> None:
    """Request exit first; explicit force permits termination of the same process after timeout."""

    try:
        process.terminate()
        try:
            process.wait(timeout=STOP_TIMEOUT_SECONDS)
        except psutil.TimeoutExpired as error:
            if not force:
                raise RuntimeError(
                    f"App process {process.pid} did not exit within {STOP_TIMEOUT_SECONDS} seconds. "
                    "After confirming that the current work may be interrupted, run app stop --force with the same workspace and port."
                ) from error
            process.kill()
            try:
                process.wait(timeout=STOP_TIMEOUT_SECONDS)
            except psutil.TimeoutExpired as error:
                raise RuntimeError(f"App process {process.pid} is still running after forced termination; no new service was started.") from error
    except psutil.NoSuchProcess:
        return


def _start_server(workspace: Path, port: int) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError as error:
        raise ValueError('The app requires optional dependencies: python -m pip install ".[app]"') from error
    _ensure_port_available(port)
    process = psutil.Popen(
        _command(port), env=dict(os.environ, ALLIANCE_EVAL_WORKSPACE=str(workspace)),
        start_new_session=os.name != "nt",
    )
    print(f"Starting app (PID {process.pid}): http://127.0.0.1:{port}", flush=True)
    print("Press Ctrl+C to stop, or use app stop/restart in another terminal with the same workspace and port.", flush=True)
    try:
        return process.wait()
    except KeyboardInterrupt:
        print("\nStopping app...", flush=True)
        _stop_process(process, force=False)
        print("App stopped.", flush=True)
        return 0


def manage_app(action: str, workspace: Path, port: int, *, force: bool = False) -> int:
    try:
        process = _find_server(workspace, port)
        if action == "status":
            if process is not None:
                print(f"App is running (PID {process.pid}): http://127.0.0.1:{port}")
                print(f"Workspace: {workspace}")
            else:
                _ensure_port_available(port)
                print(f"App is not running (port {port} is available).")
            return 0
        if process is not None and action == "start":
            raise RuntimeError(f"App is already running on port {port} (PID {process.pid}); open the page or use app restart/stop.")
        if action in {"stop", "restart"}:
            if process is not None:
                print(f"Stopping app (PID {process.pid})...", flush=True)
                _stop_process(process, force=force)
            _ensure_port_available(port)
            print("App stopped." if process is not None else "App is not running; nothing to stop.", flush=True)
            if action == "stop":
                return 0
        return _start_server(workspace, port)
    except (psutil.Error, PermissionError) as error:
        raise RuntimeError(
            "Cannot inspect or control the app process. Use the same user and installation that started it, "
            "and ensure the terminal has permission to inspect and control that process."
        ) from error
