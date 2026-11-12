"""Rebuild an independent, fully derived evidence run in a new directory."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset",type=Path)
    p.add_argument("destination",type=Path)
    args=p.parse_args()
    destination=args.destination.resolve()
    dataset=args.dataset.resolve()
    if destination.exists() and any(destination.iterdir()):
        p.error("Use an empty destination to keep existing results intact.")
    root=Path(__file__).resolve().parents[1]
    source=root/"github_release" if (root/"github_release").is_dir() else root
    results=destination/"results"
    results.mkdir(parents=True,exist_ok=True)
    run=destination/"fresh"
    def execute(*arguments):
        subprocess.run([sys.executable,*map(str,arguments)],check=True)
    execute(source/"baseline/reproduce.py",source/"estimator",dataset,run)
    execute(source/"baseline/summarize.py",dataset,run,results)
    for old,new in [("experiments.csv","experiment_level_summary.csv"),("summary.json","baseline_summary.json"),
                    ("holm_family.csv","statistical_reassessment.csv")]:
        (results/old).rename(results/new)
    execute(root/"scripts/run_all_analyses.py",dataset,run,results)
    execute(root/"scripts/generate_figures.py",results,run,destination/"figures")
    execute(root/"tests/verify.py",dataset,run,destination)

if __name__=="__main__":
    main()
