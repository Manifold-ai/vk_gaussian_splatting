import os
import stat
import textwrap

import pytest

from vkgs.runner import HeadlessRunner, RunError, find_executable, parse_log

# Realistic log snippet: 'ParameterSequence N "name" =' sections with Timer
# lines in the profiler report format matched by benchmark.py:117-118.
SYNTHETIC_LOG = textwrap.dedent(
    """\
    Loading PLY file /data/flowers_1.ply
    ParameterSequence 0 "Load scene and settle" =
     --sequenceframes 8 --sequenceaverages 4 --sequenceresetframes 0
    Timer "Frame"; GPU; avg 1234; min 1100; max 1400; CPU; avg 2345; min 2200; max 2500;
    Timer "GPU Sort"; GPU; avg 500; min 450; max 560; CPU; avg 100; min 90; max 120;
    ParameterSequence 1 "cam0" =
     --sequenceframes 8 --activateCameraPreset 0
    Timer "Frame"; GPU; avg 4321; min 4000; max 4600; CPU; avg 5432; min 5100; max 5800;
    ParameterSequence 2 "cam0 save" =
     --saveImageBuffer -1
    """
)


def make_fake_exe(tmp_path, body="exit 0", name="fake_vkgs"):
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def make_cfg(tmp_path):
    cfg = tmp_path / "run.cfg"
    cfg.write_text('SEQUENCE "Load scene and settle"\n--sequenceframes 1\n')
    return str(cfg)


# ------------------------------------------------------------ find_executable


def test_find_executable_explicit(tmp_path):
    exe = make_fake_exe(tmp_path)
    assert find_executable(str(exe)) == exe.resolve()


def test_find_executable_explicit_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="explicit"):
        find_executable(str(tmp_path / "nope"))


def test_find_executable_explicit_not_executable(tmp_path):
    plain = tmp_path / "not_exe"
    plain.write_text("data")
    with pytest.raises(FileNotFoundError):
        find_executable(str(plain))


def test_find_executable_env_override(tmp_path, monkeypatch):
    exe = make_fake_exe(tmp_path)
    monkeypatch.setenv("VKGS_BIN", str(exe))
    assert find_executable() == exe.resolve()


def test_find_executable_env_invalid(tmp_path, monkeypatch):
    monkeypatch.setenv("VKGS_BIN", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="VKGS_BIN"):
        find_executable()


def test_find_executable_default_search(monkeypatch):
    """Without overrides either a built binary is found or the error carries
    a build hint."""
    monkeypatch.delenv("VKGS_BIN", raising=False)
    try:
        exe = find_executable()
    except FileNotFoundError as err:
        assert "cmake" in str(err)
        assert "VKGS_BIN" in str(err)
    else:
        assert exe.is_file()


# ------------------------------------------------------------------ parse_log


def test_parse_log_sequences_and_timers():
    sequences, warnings = parse_log(SYNTHETIC_LOG)
    assert [(s.id, s.name) for s in sequences] == [
        (0, "Load scene and settle"),
        (1, "cam0"),
        (2, "cam0 save"),
    ]
    load = sequences[0]
    assert load.timers["Frame"] == {"gpu_ms": 1.234, "cpu_ms": 2.345}
    assert load.timers["GPU Sort"] == {"gpu_ms": 0.5, "cpu_ms": 0.1}
    assert sequences[1].timers["Frame"]["gpu_ms"] == pytest.approx(4.321)
    assert sequences[2].timers == {}
    assert warnings == []


def test_parse_log_warnings():
    log = SYNTHETIC_LOG + "colorBufferFormat 5 is out of range [0, 2]\n"
    _, warnings = parse_log(log)
    assert warnings == ["colorBufferFormat 5 is out of range [0, 2]"]


def test_parse_log_empty():
    sequences, warnings = parse_log("")
    assert sequences == [] and warnings == []


# ------------------------------------------------------------------- run()


def test_run_success_parses_and_logs_args(tmp_path):
    out_file = tmp_path / "cam0_main.png"
    log_body = SYNTHETIC_LOG.replace('"', '\\"')
    exe = make_fake_exe(
        tmp_path,
        body=f'echo "ARGS: $@"\nprintf "%s" "{log_body}"\ntouch "{out_file}"\n',
    )
    runner = HeadlessRunner(str(exe))
    result = runner.run(
        make_cfg(tmp_path),
        project=str(tmp_path / "scene.vkgs"),
        size=(640, 480),
        gpu=1,
        expected_outputs=[str(out_file)],
        log_path=str(tmp_path / "run.log"),
    )

    assert result.returncode == 0
    assert result.output_files == [str(out_file)]
    assert len(result.sequences) == 3
    assert result.sequences[1].name == "cam0"
    assert result.duration_s >= 0

    # Invocation contract: exact flags forwarded to the app
    args_line = next(l for l in result.log_text.splitlines() if l.startswith("ARGS:"))
    assert "--size 640 480" in args_line
    assert "--benchmark 1" in args_line
    assert "--headless 1" in args_line
    assert "--sequencefile" in args_line
    assert "--inputProject" in args_line
    assert "--loadDefaultScene 0" in args_line
    assert "--forcegpu 1" in args_line


def test_run_input_files(tmp_path):
    exe = make_fake_exe(tmp_path, body='echo "ARGS: $@"')
    runner = HeadlessRunner(str(exe))
    result = runner.run(
        make_cfg(tmp_path),
        input_files=[str(tmp_path / "a.ply"), str(tmp_path / "b.ply")],
        log_path=str(tmp_path / "run.log"),
    )
    args_line = result.log_text.splitlines()[0]
    assert args_line.count("--inputFile") == 2
    assert "--loadDefaultScene 0" in args_line
    assert "--inputProject" not in args_line


def test_run_missing_expected_output(tmp_path):
    exe = make_fake_exe(tmp_path, body='echo "all good"')
    runner = HeadlessRunner(str(exe))
    with pytest.raises(RunError, match="missing"):
        runner.run(
            make_cfg(tmp_path),
            expected_outputs=[str(tmp_path / "never_written.png")],
            log_path=str(tmp_path / "run.log"),
        )


def test_run_fatal_log_marker(tmp_path):
    exe = make_fake_exe(tmp_path, body="echo 'Camera preset index 5 is out of range [0, 2)'")
    runner = HeadlessRunner(str(exe))
    with pytest.raises(RunError, match="Camera preset index"):
        runner.run(make_cfg(tmp_path), log_path=str(tmp_path / "run.log"))


def test_run_failed_to_load_marker(tmp_path):
    exe = make_fake_exe(tmp_path, body="echo 'Failed to load camera presets'")
    runner = HeadlessRunner(str(exe))
    with pytest.raises(RunError, match="Failed to load"):
        runner.run(make_cfg(tmp_path), log_path=str(tmp_path / "run.log"))


def test_run_nonzero_exit(tmp_path):
    exe = make_fake_exe(tmp_path, body='echo "boom"\nexit 3')
    runner = HeadlessRunner(str(exe))
    with pytest.raises(RunError, match="code 3"):
        runner.run(make_cfg(tmp_path), log_path=str(tmp_path / "run.log"))


def test_run_timeout(tmp_path):
    exe = make_fake_exe(tmp_path, body="sleep 30")
    runner = HeadlessRunner(str(exe))
    with pytest.raises(RunError, match="timed out"):
        runner.run(make_cfg(tmp_path), timeout=0.5, log_path=str(tmp_path / "run.log"))


def test_run_missing_cfg(tmp_path):
    exe = make_fake_exe(tmp_path)
    runner = HeadlessRunner(str(exe))
    with pytest.raises(FileNotFoundError):
        runner.run(str(tmp_path / "absent.cfg"))
