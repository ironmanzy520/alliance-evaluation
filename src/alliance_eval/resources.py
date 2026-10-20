"""Locate read-only source or package resources and record the executing code identity."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from alliance_eval import __version__


PACKAGE_DIR = Path(__file__).resolve().parent
LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def source_root() -> Path | None:
    """Identify only the source checkout containing this package, not an unrelated working directory."""

    root = PACKAGE_DIR.parent.parent
    if PACKAGE_DIR.parent.name == "src" and (root / "pyproject.toml").is_file():
        return root
    return None


def resource_directory(name: str) -> Path:
    """Use source resources for editable installs and packaged resources for regular installs."""

    if name not in {"prompts", "schemas", "examples"}:
        raise ValueError(f"Unknown resource directory: {name}")
    root = source_root()
    directory = root / name if root is not None else PACKAGE_DIR / "resources" / name
    if not directory.is_dir():
        raise FileNotFoundError(f"Package resource directory missing: {directory}")
    return directory


def code_identity() -> dict[str, str | bool | None]:
    """Installed wheels need no Git metadata; source checkouts report their own commit and dirty state."""

    root = source_root()
    identity: dict[str, str | bool | None] = {
        "package_version": __version__, "git_commit": None, "git_dirty": None,
    }
    if root is None:
        return identity
    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=root,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if Path(git_root).resolve() != root:
            return identity
        identity["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        identity["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=root, text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        # An extracted or installed package may have no Git metadata; do not borrow the working directory's commit.
        pass
    return identity
