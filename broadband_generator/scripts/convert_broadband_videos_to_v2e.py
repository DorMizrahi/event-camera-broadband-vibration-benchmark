from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
RUNTIME_DEPS = WORKSPACE_ROOT / "synthetic_event_vibration_dataset" / ".runtime_deps"
V2E_SCRIPT = RUNTIME_DEPS / "v2e.py"
PYTHON_EXE = Path(sys.executable)

if RUNTIME_DEPS.exists():
    sys.path.insert(0, str(RUNTIME_DEPS))

import h5py  # noqa: E402
import pandas as pd  # noqa: E402


def build_command(exp_dir: Path, metadata: Dict[str, object]) -> List[str]:
    resolution = metadata.get("resolution", [240, 180])
    video_fps = float(metadata.get("video_fps", 40000.0))
    v2e_dir = exp_dir / "v2e_output"
    input_path = exp_dir / "video.mp4"
    return [
        str(PYTHON_EXE),
        str(V2E_SCRIPT),
        "-i",
        str(input_path),
        "--input_frame_rate",
        f"{video_fps:.12g}",
        "--disable_slomo",
        "--no_preview",
        "--skip_video_output",
        "--overwrite",
        "--output_folder",
        str(v2e_dir),
        "--output_width",
        str(int(resolution[0])),
        "--output_height",
        str(int(resolution[1])),
        "--pos_thres",
        "0.2",
        "--neg_thres",
        "0.2",
        "--sigma_thres",
        "0.03",
        "--refractory_period",
        "0.0001",
        "--shot_noise_rate_hz",
        "0.0",
        "--cutoff_hz",
        "0.0",
        "--dvs_h5",
        "events.h5",
    ]


def convert_one(exp_dir: Path, force: bool = False) -> Dict[str, object]:
    metadata_path = exp_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    event_method = str(metadata.get("event_conversion_method", ""))
    if event_method == "v2e" and (exp_dir / "events.h5").exists() and not force:
        count = count_events(exp_dir / "events.h5")
        return {"experiment": exp_dir.name, "status": "skipped", "total_events": count, "failure_reason": ""}

    if not V2E_SCRIPT.exists():
        return {"experiment": exp_dir.name, "status": "failed", "total_events": 0, "failure_reason": f"missing {V2E_SCRIPT}"}

    v2e_dir = exp_dir / "v2e_output"
    v2e_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(exp_dir, metadata)
    (exp_dir / "run_v2e_later.ps1").write_text(" ".join(quote_ps(part) for part in command) + "\n", encoding="utf-8")

    completed = subprocess.run(
        command,
        cwd=str(exp_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=v2e_environment(),
    )
    (v2e_dir / "v2e_stdout.txt").write_text(completed.stdout, encoding="utf-8", errors="ignore")
    (v2e_dir / "v2e_stderr.txt").write_text(completed.stderr, encoding="utf-8", errors="ignore")
    if completed.returncode != 0:
        reason = f"v2e_return_code={completed.returncode}; stderr_tail={completed.stderr[-1000:]}"
        (exp_dir / "v2e_failed.txt").write_text(reason, encoding="utf-8", errors="ignore")
        return {"experiment": exp_dir.name, "status": "failed", "total_events": 0, "failure_reason": reason}

    source_h5 = v2e_dir / "events.h5"
    if not source_h5.exists() or source_h5.stat().st_size == 0:
        reason = "v2e completed but events.h5 was not created"
        (exp_dir / "v2e_failed.txt").write_text(reason, encoding="utf-8")
        return {"experiment": exp_dir.name, "status": "failed", "total_events": 0, "failure_reason": reason}

    shutil.copy2(source_h5, exp_dir / "events.h5")
    count = count_events(exp_dir / "events.h5")
    duration = float(metadata.get("duration_sec", 0.2))
    metadata["event_conversion_method"] = "v2e"
    metadata["used_v2e"] = True
    metadata["v2e_command"] = command
    metadata["total_events"] = int(count)
    metadata["event_rate_hz"] = float(count / max(duration, 1e-12))
    metadata["v2e_output_dir"] = str(v2e_dir)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    failed_path = exp_dir / "v2e_failed.txt"
    if failed_path.exists():
        failed_path.unlink()
    return {"experiment": exp_dir.name, "status": "ok", "total_events": count, "failure_reason": ""}


def count_events(path: Path) -> int:
    with h5py.File(path, "r") as h5:
        if "events" not in h5:
            return 0
        return int(h5["events"].shape[0])


def update_summary(results: List[Dict[str, object]]) -> pd.DataFrame:
    summary_path = PROJECT_ROOT / "dataset_summary.csv"
    summary = pd.read_csv(summary_path)
    summary["failure_reason"] = summary["failure_reason"].fillna("").astype(object)
    summary["status"] = summary["status"].fillna("").astype(object)
    by_exp = {r["experiment"]: r for r in results}
    for idx, row in summary.iterrows():
        exp = row["experiment"]
        if exp not in by_exp:
            continue
        result = by_exp[exp]
        if result["status"] in {"ok", "skipped"}:
            total_events = int(result["total_events"])
            summary.at[idx, "total_events"] = total_events
            summary.at[idx, "event_rate_hz"] = total_events / max(float(row["duration_sec"]), 1e-12)
            summary.at[idx, "status"] = "ok"
            summary.at[idx, "failure_reason"] = ""
        else:
            summary.at[idx, "status"] = "failed"
            summary.at[idx, "failure_reason"] = result["failure_reason"]
    summary.to_csv(summary_path, index=False)
    return summary


def write_zip() -> Path:
    zip_path = PROJECT_ROOT.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in PROJECT_ROOT.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(PROJECT_ROOT.parent))
    return zip_path


def v2e_environment() -> Dict[str, str]:
    env = dict(**__import__("os").environ)
    existing = env.get("PYTHONPATH", "")
    paths = [str(RUNTIME_DEPS)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = __import__("os").pathsep.join(paths)
    return env


def quote_ps(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="Optional experiment ids to convert.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    experiments_dir = PROJECT_ROOT / "experiments"
    exp_dirs = sorted(p for p in experiments_dir.glob("experiment_*") if p.is_dir())
    if args.only:
        selected = set(args.only)
        exp_dirs = [p for p in exp_dirs if p.name in selected]

    results = []
    for exp_dir in exp_dirs:
        print(f"RUN V2E {exp_dir.name}", flush=True)
        result = convert_one(exp_dir, force=args.force)
        results.append(result)
        print(json.dumps(result), flush=True)

    summary = update_summary(results)
    ok = summary[summary["status"] == "ok"]
    failed = summary[summary["status"] != "ok"]
    zip_path = None if args.no_zip else write_zip()
    print(f"successful_experiments={len(ok)}")
    print(f"failed_experiments={len(failed)}")
    print(f"total_events={int(ok['total_events'].sum()) if len(ok) else 0}")
    if zip_path is not None:
        print(f"zip_path={zip_path}")
        print(f"zip_size_bytes={zip_path.stat().st_size}")


if __name__ == "__main__":
    main()
