from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py


@dataclass(frozen=True)
class V2ELauncher:
    prefix: list[str]
    source: str


def locate_v2e(config: Mapping[str, Any], project_root: Path) -> V2ELauncher | None:
    settings = config["v2e"]
    python_executable = str(settings.get("python_executable") or sys.executable)

    explicit_command = str(settings.get("command") or os.environ.get("V2E_COMMAND", "")).strip()
    if explicit_command:
        resolved = shutil.which(explicit_command)
        if resolved:
            return V2ELauncher([resolved], "configured_command")
        command_path = _resolve_candidate(explicit_command, project_root)
        if command_path and command_path.is_file():
            return V2ELauncher([str(command_path)], "configured_command_path")

    explicit_script = str(settings.get("script_path") or os.environ.get("V2E_SCRIPT", "")).strip()
    if explicit_script:
        script = _resolve_candidate(explicit_script, project_root)
        if script and script.is_dir():
            script = script / "v2e.py"
        if script and script.is_file():
            return V2ELauncher([python_executable, str(script)], "configured_script")

    for command_name in ("v2e", "v2e.py"):
        executable = shutil.which(command_name)
        if executable:
            return V2ELauncher([executable], "path_command")

    for candidate in (
        project_root / "third_party" / "v2e" / "v2e.py",
        project_root / "v2e" / "v2e.py",
        project_root.parent / "v2e" / "v2e.py",
    ):
        if candidate.is_file():
            return V2ELauncher([python_executable, str(candidate)], "nearby_script")

    try:
        spec = importlib.util.find_spec("v2e")
    except (ImportError, AttributeError, ValueError):
        spec = None
    if spec is not None and spec.origin and Path(spec.origin).is_file():
        return V2ELauncher([python_executable, str(Path(spec.origin))], "installed_module_script")
    return None


def convert_video_with_v2e(
    experiment_dir: Path,
    config: Mapping[str, Any],
    project_root: Path,
    progress=print,
) -> dict[str, Any]:
    """Run only V2E. Missing V2E is a documented skip, never an emulated fallback."""
    settings = config["v2e"]
    output_dir = experiment_dir / "v2e_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    launcher = locate_v2e(config, project_root)
    arguments = _v2e_arguments(experiment_dir, config, output_dir)
    _write_later_scripts(experiment_dir, launcher, arguments)

    if not bool(settings["enabled"]):
        result = _status(
            status="disabled",
            reason="V2E conversion was disabled in the experiment parameters.",
            launcher=launcher,
            command=None,
            event_count=None,
        )
        return _write_status(experiment_dir, result)

    if launcher is None:
        result = _status(
            status="skipped_not_installed",
            reason=(
                "V2E was not found. All non-event experiment artifacts were generated; "
                "events.h5 was intentionally not created."
            ),
            launcher=None,
            command=None,
            event_count=None,
        )
        progress("V2E not found; keeping the completed experiment without events.h5.")
        return _write_status(experiment_dir, result)

    command = launcher.prefix + arguments
    (output_dir / "v2e_command.json").write_text(
        json.dumps(command, indent=2), encoding="utf-8"
    )
    progress(f"Running V2E ({launcher.source}) ...")
    timeout = int(settings["timeout_sec"])
    try:
        completed = subprocess.run(
            command,
            cwd=str(experiment_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout if timeout > 0 else None,
            env=_v2e_environment(launcher),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _as_text(exc.stdout)
        stderr = _as_text(exc.stderr)
        (output_dir / "v2e_stdout.txt").write_text(stdout, encoding="utf-8", errors="ignore")
        (output_dir / "v2e_stderr.txt").write_text(stderr, encoding="utf-8", errors="ignore")
        result = _status(
            status="failed",
            reason=f"V2E exceeded timeout_sec={timeout}.",
            launcher=launcher,
            command=command,
            event_count=None,
        )
        return _write_status(experiment_dir, result)
    except OSError as exc:
        result = _status(
            status="failed",
            reason=f"Could not start V2E: {type(exc).__name__}: {exc}",
            launcher=launcher,
            command=command,
            event_count=None,
        )
        return _write_status(experiment_dir, result)

    (output_dir / "v2e_stdout.txt").write_text(
        completed.stdout, encoding="utf-8", errors="ignore"
    )
    (output_dir / "v2e_stderr.txt").write_text(
        completed.stderr, encoding="utf-8", errors="ignore"
    )
    if completed.returncode != 0:
        tail = completed.stderr[-1500:].replace("\n", " ").strip()
        result = _status(
            status="failed",
            reason=f"V2E returned code {completed.returncode}. stderr tail: {tail}",
            launcher=launcher,
            command=command,
            event_count=None,
        )
        progress(f"V2E failed with return code {completed.returncode}; see v2e_output logs.")
        return _write_status(experiment_dir, result)

    source_h5 = output_dir / "events.h5"
    if not source_h5.is_file() or source_h5.stat().st_size == 0:
        result = _status(
            status="failed",
            reason="V2E completed but did not create a non-empty events.h5.",
            launcher=launcher,
            command=command,
            event_count=None,
        )
        return _write_status(experiment_dir, result)
    try:
        event_count = count_events(source_h5)
    except Exception as exc:
        result = _status(
            status="failed",
            reason=f"V2E HDF5 validation failed: {type(exc).__name__}: {exc}",
            launcher=launcher,
            command=command,
            event_count=None,
        )
        return _write_status(experiment_dir, result)
    if event_count <= 0:
        result = _status(
            status="failed",
            reason="V2E produced an HDF5 file with zero readable events.",
            launcher=launcher,
            command=command,
            event_count=0,
        )
        return _write_status(experiment_dir, result)

    events_path = experiment_dir / "events.h5"
    shutil.copy2(source_h5, events_path)
    duration = float(config["derived"]["duration_sec"])
    result = _status(
        status="completed",
        reason="",
        launcher=launcher,
        command=command,
        event_count=event_count,
    )
    result.update(
        {
            "used_v2e": True,
            "events_path": str(events_path),
            "event_rate_hz": float(event_count / duration),
            "v2e_output_path": str(source_h5),
            "event_file_layout": describe_event_file(source_h5),
        }
    )
    progress(f"V2E completed: {event_count:,} events.")
    return _write_status(experiment_dir, result)


def count_events(path: Path) -> int:
    with h5py.File(path, "r") as h5:
        if "events" in h5:
            node = h5["events"]
            if isinstance(node, h5py.Dataset):
                return int(node.shape[0])
            for key in ("t", "t_us", "ts", "x"):
                if key in node and isinstance(node[key], h5py.Dataset):
                    return int(node[key].shape[0])
        for key in ("t", "t_us", "ts", "x"):
            if key in h5 and isinstance(h5[key], h5py.Dataset):
                return int(h5[key].shape[0])
    raise ValueError("No supported event dataset was found in the HDF5 file.")


def describe_event_file(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        if "events" in h5 and isinstance(h5["events"], h5py.Dataset):
            dataset = h5["events"]
            columns = ["t_us", "x", "y", "p"] if dataset.ndim == 2 and dataset.shape[1] == 4 else None
            return {
                "hdf5_dataset": "events",
                "shape": [int(value) for value in dataset.shape],
                "dtype": str(dataset.dtype),
                "columns": columns,
                "timestamp_unit": "microseconds" if columns else "inspect_source_file",
                "polarity": "1=ON, 0=OFF" if columns else "inspect_source_file",
            }
        return {
            "hdf5_keys": list(h5.keys()),
            "layout": "separate_arrays_or_custom",
            "timestamp_unit": "inspect_source_file",
        }


def _v2e_arguments(
    experiment_dir: Path,
    config: Mapping[str, Any],
    output_dir: Path,
) -> list[str]:
    settings = config["v2e"]
    sampling = config["sampling"]
    arguments = [
        "-i",
        str(experiment_dir / "video.mp4"),
        "--input_frame_rate",
        f"{float(sampling['video_fps']):.12g}",
    ]
    if bool(settings["disable_slomo"]):
        arguments.append("--disable_slomo")
    arguments.extend(
        [
            "--no_preview",
            "--skip_video_output",
            "--overwrite",
            "--output_folder",
            str(output_dir),
            "--output_width",
            str(int(sampling["width_px"])),
            "--output_height",
            str(int(sampling["height_px"])),
            "--pos_thres",
            f"{float(settings['pos_thres']):.12g}",
            "--neg_thres",
            f"{float(settings['neg_thres']):.12g}",
            "--sigma_thres",
            f"{float(settings['sigma_thres']):.12g}",
            "--refractory_period",
            f"{float(settings['refractory_period_sec']):.12g}",
            "--shot_noise_rate_hz",
            f"{float(settings['shot_noise_rate_hz']):.12g}",
            "--cutoff_hz",
            f"{float(settings['cutoff_hz']):.12g}",
            "--dvs_emulator_seed",
            str(int(settings["resolved_dvs_emulator_seed"])),
            "--dvs_h5",
            "events.h5",
        ]
    )
    return arguments


def _write_later_scripts(
    experiment_dir: Path,
    launcher: V2ELauncher | None,
    arguments: Sequence[str],
) -> None:
    if launcher is not None:
        command = launcher.prefix + list(arguments)
        powershell = "& " + " ".join(_quote_powershell(value) for value in command)
        shell = " ".join(shlex.quote(value) for value in command)
    else:
        powershell_arguments = " ".join(_quote_powershell(value) for value in arguments)
        powershell = (
            "if (-not $env:V2E_SCRIPT) { throw 'Set V2E_SCRIPT to the full path of v2e.py.' }\n"
            f"& python $env:V2E_SCRIPT {powershell_arguments}"
        )
        shell_arguments = " ".join(shlex.quote(value) for value in arguments)
        shell = (
            ": \"${V2E_SCRIPT:?Set V2E_SCRIPT to the full path of v2e.py}\"\n"
            f"python \"$V2E_SCRIPT\" {shell_arguments}"
        )
    (experiment_dir / "run_v2e_later.ps1").write_text(
        powershell + "\n", encoding="utf-8"
    )
    (experiment_dir / "run_v2e_later.sh").write_text(shell + "\n", encoding="utf-8")


def _status(
    status: str,
    reason: str,
    launcher: V2ELauncher | None,
    command: Sequence[str] | None,
    event_count: int | None,
) -> dict[str, Any]:
    return {
        "status": status,
        "used_v2e": False,
        "v2e_available": launcher is not None,
        "launcher_source": launcher.source if launcher else None,
        "command": list(command) if command is not None else None,
        "events_path": None,
        "event_count": event_count,
        "event_rate_hz": None,
        "reason": reason,
        "fallback_used": False,
    }


def _write_status(experiment_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    (experiment_dir / "event_conversion_status.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _resolve_candidate(value: str, project_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _v2e_environment(launcher: V2ELauncher) -> dict[str, str]:
    environment = dict(os.environ)
    if len(launcher.prefix) >= 2 and launcher.prefix[1].lower().endswith(".py"):
        script_parent = str(Path(launcher.prefix[1]).resolve().parent)
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            script_parent + (os.pathsep + existing if existing else "")
        )
    return environment


def _quote_powershell(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return value
