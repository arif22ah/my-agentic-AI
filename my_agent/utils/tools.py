"""Sandboxed quality gate used by the tester agent: pytest first, then ruff (bugs only).

Backends:
  * docker (default): fresh locked-down container per run (no network, memory/CPU/PID caps,
    all capabilities dropped, unprivileged user). Files are streamed in with put_archive, so
    it also works when this app itself runs in a container (sibling-container pattern).
  * local: plain subprocess. Dev convenience only, NOT isolated.
A semaphore bounds how many sandboxes run at once so many API runs can't exhaust the host.
"""
from __future__ import annotations

import importlib.util
import io
import logging
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from my_agent.config import get_settings

log = logging.getLogger(__name__)

SOLUTION_FILE = "solution.py"
TEST_FILE = "test_solution.py"
LINT_MARKER = "--- ruff ---"
_PYTEST = ["python", "-m", "pytest", "-x", "-q", "--tb=short", "-p", "no:cacheprovider",
           "--timeout=30", TEST_FILE]
_RUFF = ["ruff", "check", "--no-cache", "--select", "E9,F,B", "--output-format", "concise",
         SOLUTION_FILE]

_slots: threading.BoundedSemaphore | None = None
_slots_lock = threading.Lock()


def _slot() -> threading.BoundedSemaphore:
    global _slots
    with _slots_lock:
        if _slots is None:
            _slots = threading.BoundedSemaphore(get_settings().max_parallel_sandboxes)
    return _slots


@dataclass
class SandboxResult:
    passed: bool
    output: str
    timed_out: bool = False
    stage: str = "tests"  # "tests" | "lint"


def _tail(text: str, limit: int) -> str:
    """Keep the END of the log: pytest/ruff put the useful summary last."""
    return text if len(text) <= limit else "...[truncated]...\n" + text[-limit:]


def run_in_sandbox(code: str, tests: str) -> SandboxResult:
    """Run `tests` (then lint) against `code`; return pass/fail plus the truncated log."""
    s = get_settings()
    files = {SOLUTION_FILE: code, TEST_FILE: tests}
    with _slot():
        if s.sandbox_mode == "docker":
            try:
                result = _run_docker(files)
            except Exception as exc:
                log.warning("Docker sandbox unavailable (%s); falling back to local subprocess.", exc)
                result = _run_local(files)
        else:
            result = _run_local(files)
    result.output = _tail(result.output, s.max_log_chars)
    return result


# --------------------------------------------------------------------------- local
def _run_local(files: dict[str, str]) -> SandboxResult:
    s = get_settings()
    with tempfile.TemporaryDirectory() as tmp:
        for name, content in files.items():
            Path(tmp, name).write_text(content, encoding="utf-8")

        # Preserve Windows & POSIX environment variables required by Python/WinSock/Tempdir
        env = {"PYTHONDONTWRITEBYTECODE": "1"}
        for key in (
            "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
            "TEMP", "TMP", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
            "APPDATA", "LOCALAPPDATA", "HOME", "LANG", "LC_ALL",
        ):
            val = os.environ.get(key)
            if val is not None:
                env[key] = val

        # Use --timeout=30 only if pytest-timeout is installed in the current environment
        pytest_args = list(_PYTEST[1:])
        if not importlib.util.find_spec("pytest_timeout"):
            pytest_args = [arg for arg in pytest_args if not arg.startswith("--timeout")]

        def run(cmd: list[str]) -> subprocess.CompletedProcess:
            return subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True,
                                  timeout=s.sandbox_timeout_s)

        try:
            proc = run([sys.executable, *pytest_args])
            out = proc.stdout + proc.stderr
            if proc.returncode != 0:
                return SandboxResult(False, out)
            if s.lint and importlib.util.find_spec("ruff"):
                lint = run([sys.executable, "-m", *_RUFF])
                if lint.returncode != 0:
                    return SandboxResult(False, f"{out}\n{LINT_MARKER}\n{lint.stdout}{lint.stderr}",
                                         stage="lint")
            return SandboxResult(True, out)
        except subprocess.TimeoutExpired:
            return SandboxResult(False, f"TIMEOUT after {s.sandbox_timeout_s}s (infinite loop?)",
                                 True)


# --------------------------------------------------------------------------- docker
def _tar(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _shell_command(lint: bool) -> str:
    cmd = shlex.join(_PYTEST) + "; rc=$?; "
    if lint:
        cmd += (f"if [ $rc -eq 0 ]; then echo '{LINT_MARKER}'; {shlex.join(_RUFF)}; rc=$?; fi; ")
    return cmd + "exit $rc"


def _run_docker(files: dict[str, str]) -> SandboxResult:
    import docker  # lazy: local mode doesn't need a daemon

    s = get_settings()
    container = docker.from_env().containers.create(
        s.sandbox_image,
        command=["sh", "-c", _shell_command(s.lint)],
        working_dir="/work",
        network_disabled=True,
        mem_limit=s.sandbox_mem,
        nano_cpus=int(s.sandbox_cpus * 1e9),
        pids_limit=256,
        cap_drop=["ALL"],
        security_opt=["no-new-privileges"],
        user="65534:65534",  # nobody
        environment={"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"},
        tmpfs={"/tmp": "size=64m"},
    )
    timed_out, exit_code = False, 1
    try:
        container.put_archive("/work", _tar(files))
        container.start()
        try:
            exit_code = container.wait(timeout=s.sandbox_timeout_s)["StatusCode"]
        except Exception:  # noqa: BLE001  (requests timeout / connection error)
            timed_out = True
            container.kill()
        logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
    finally:
        container.remove(force=True)

    if timed_out:
        logs = f"TIMEOUT after {s.sandbox_timeout_s}s (infinite loop?)\n{logs}"
    stage = "lint" if (LINT_MARKER in logs and exit_code != 0) else "tests"
    return SandboxResult(exit_code == 0 and not timed_out, logs, timed_out, stage)
