"""Locate and drive the vk_gaussian_splatting executable headless.

Invocation contract (mirrors benchmark.py / docs/getting-started.md): the app
is launched with ``--headless 1 --benchmark 1 --sequencefile <cfg>``; in that
mode headlessFrameCount = UINT32_MAX and the process exits when the sequencer
finishes (src/main.cpp:229-232). Scenes come from ``--inputProject <.vkgs>``
or repeated ``--inputFile <.ply>`` (+ ``--loadDefaultScene 0`` so the bundled
demo scene does not load underneath).

Log parsing ports the ParameterSequence/Timer regexes of benchmark.py
(parse_benchmark, benchmark.py:115-172). Failure LOGW markers worth failing
on: 'Camera preset index %d is out of range' and 'Failed to load'
(src/gaussian_splatting_ui.cpp:129,141).
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_EXE_NAMES = ("vk_gaussian_splatting", "vk_gaussian_splatting.exe", "vk_gaussian_splatting_app")


def _repo_root() -> Path:
    # runner.py lives in <repo>/python/vkgs/
    return Path(__file__).resolve().parents[2]


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(str(path), os.X_OK)


def find_executable(explicit: Optional[str] = None) -> Path:
    """Locate the renderer executable.

    Search order: ``explicit`` argument -> $VKGS_BIN -> repo
    ``_bin/{Release,Debug}/`` (the CMake output dir, docs/getting-started.md)
    -> newest match under repo ``build*/`` trees.
    """
    if explicit is not None:
        path = Path(explicit).expanduser()
        if _is_executable(path):
            return path.resolve()
        raise FileNotFoundError(f"explicit executable not found or not executable: {explicit}")

    env = os.environ.get("VKGS_BIN")
    if env:
        path = Path(env).expanduser()
        if _is_executable(path):
            return path.resolve()
        raise FileNotFoundError(f"$VKGS_BIN does not point to an executable: {env}")

    root = _repo_root()
    for config in ("Release", "Debug"):
        for name in _EXE_NAMES:
            path = root / "_bin" / config / name
            if _is_executable(path):
                return path.resolve()

    candidates: List[Path] = []
    for pattern in (str(root / "build*" / "**" / name) for name in _EXE_NAMES):
        candidates += [Path(p) for p in glob.glob(pattern, recursive=True)]
    candidates = [c for c in candidates if _is_executable(c)]
    if candidates:
        return max(candidates, key=lambda c: c.stat().st_mtime).resolve()

    raise FileNotFoundError(
        "vk_gaussian_splatting executable not found. Build it first (from the repo root:\n"
        "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel\n"
        "which outputs to _bin/Release/), or set $VKGS_BIN / pass executable= explicitly."
    )


# --------------------------------------------------------------- log parsing

# Ported from benchmark.py:117-118 (parse_benchmark).
_SEQUENCE_RE = re.compile(r'ParameterSequence\s+(\d+)\s+"([^"]+)"\s*=')
_TIMER_RE = re.compile(r'Timer\s+"([^"]+)"\s*;\s*GPU;\s*avg\s+(\d+);.*?CPU;\s*avg\s+(\d+);')

# Lines collected into RunResult.warnings. nvutils LOGW/LOGE print the bare
# message, so also match the known warning texts emitted by the app.
_WARNING_RE = re.compile(
    r"LOGW|LOGE|WARNING|\bERROR\b"
    r"|Camera preset index"
    r"|Failed to load"
    r"|out of range"
    r"|No buffers available",
    re.IGNORECASE,
)

# Substrings that turn a completed run into a hard failure.
FATAL_LOG_MARKERS = ("Camera preset index", "Failed to load", "ERROR")


@dataclass
class SequenceInfo:
    """One 'ParameterSequence N "name" =' section of the log."""

    id: int
    name: str
    # timer stage -> {"gpu_ms": float, "cpu_ms": float} (avg values, ms)
    timers: Dict[str, Dict[str, float]] = field(default_factory=dict)


def parse_log(log_text: str) -> Tuple[List[SequenceInfo], List[str]]:
    """Parse sequences (with per-stage timers) and warning lines from a log."""
    sequences: List[SequenceInfo] = []
    parts = _SEQUENCE_RE.split(log_text)[1:]
    for i in range(0, len(parts) - 2, 3):
        seq = SequenceInfo(id=int(parts[i]), name=parts[i + 1].strip())
        for match in _TIMER_RE.finditer(parts[i + 2]):
            seq.timers[match.group(1).strip()] = {
                "gpu_ms": float(match.group(2)) / 1000.0,
                "cpu_ms": float(match.group(3)) / 1000.0,
            }
        sequences.append(seq)

    warnings = [line.rstrip() for line in log_text.splitlines() if _WARNING_RE.search(line)]
    return sequences, warnings


@dataclass
class RunResult:
    returncode: int
    log_path: str
    log_text: str
    sequences: List[SequenceInfo]
    warnings: List[str]
    duration_s: float
    output_files: List[str]


class RunError(RuntimeError):
    """Raised when a headless run fails; the message embeds the log tail."""


def _log_tail(log_text: str, lines: int = 20) -> str:
    tail = log_text.splitlines()[-lines:]
    return "\n".join(tail)


class HeadlessRunner:
    """Run the renderer headless over a .cfg sequence file."""

    def __init__(self, executable: Optional[str] = None):
        self.executable = find_executable(executable)

    def run(
        self,
        cfg: str,
        project: Optional[str] = None,
        input_files: Sequence[str] = (),
        size: Tuple[int, int] = (1920, 1080),
        gpu: Optional[int] = None,
        timeout: float = 1800,
        extra_args: Sequence[str] = (),
        log_path: Optional[str] = None,
        expected_outputs: Sequence[str] = (),
    ) -> RunResult:
        """Execute one headless benchmark run.

        Raises :class:`RunError` on nonzero exit, timeout, missing
        ``expected_outputs`` or fatal log markers. stdout+stderr are merged
        into ``log_path`` (default: <cfg>.log next to the cfg file).
        """
        cfg_path = os.path.abspath(cfg)
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"sequence file not found: {cfg}")
        if log_path is None:
            log_path = os.path.splitext(cfg_path)[0] + ".log"
        log_path = os.path.abspath(log_path)

        command = [
            str(self.executable),
            "--size", str(int(size[0])), str(int(size[1])),
            "--benchmark", "1",
            "--headless", "1",
            "--sequencefile", cfg_path,
        ]
        if project is not None:
            command += ["--inputProject", os.path.abspath(project)]
        for input_file in input_files:
            command += ["--inputFile", os.path.abspath(input_file)]
        if project is not None or input_files:
            command += ["--loadDefaultScene", "0"]
        if gpu is not None:
            command += ["--forcegpu", str(int(gpu))]
        command += list(extra_args)

        start = time.monotonic()
        try:
            with open(log_path, "w", encoding="utf-8") as log_file:
                completed = subprocess.run(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.executable.parent),
                    shell=False,
                    timeout=timeout,
                )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            raise RunError(
                f"renderer timed out after {timeout}s (log: {log_path});\n"
                "check that the .cfg terminates (headless+benchmark exits only "
                "when the sequencer finishes)"
            ) from None
        duration = time.monotonic() - start

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                log_text = f.read()
        except OSError:
            log_text = ""

        sequences, warnings = parse_log(log_text)

        if returncode != 0:
            raise RunError(
                f"renderer exited with code {returncode} (log: {log_path})\n{_log_tail(log_text)}"
            )

        fatal = [marker for marker in FATAL_LOG_MARKERS if marker in log_text]
        if fatal:
            offending = [line for line in log_text.splitlines() if any(m in line for m in fatal)]
            raise RunError(
                f"fatal marker(s) {fatal} in renderer log (log: {log_path}):\n"
                + "\n".join(offending[:10])
            )

        missing = [os.path.abspath(p) for p in expected_outputs if not os.path.isfile(os.path.abspath(p))]
        if missing:
            raise RunError(
                "renderer completed but expected outputs are missing:\n  "
                + "\n  ".join(missing)
                + f"\n(log: {log_path})\n{_log_tail(log_text)}"
            )

        return RunResult(
            returncode=returncode,
            log_path=log_path,
            log_text=log_text,
            sequences=sequences,
            warnings=warnings,
            duration_s=duration,
            output_files=[os.path.abspath(p) for p in expected_outputs],
        )
