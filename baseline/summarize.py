"""Summarize a fresh run using the original wrapper's pooled and motion-family test groups."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

p = argparse.ArgumentParser()
for name in ["dataset", "run", "out"]:
    p.add_argument(name, type=Path)
a = p.parse_args()
rows, pairs = [], []
for folder in sorted(a.run.glob("experiment_*")):
    m = json.loads((a.dataset / "experiments" / folder.name / "metadata.json").read_text())
    s = json.loads((folder / "analysis_summary.json").read_text())
    assert not s["calibration_applied"] and not s["ground_truth_used_for_estimation"]
    row = dict(experiment=folder.name, family=m["motion_type"], axes=m["active_axes"],
               amplitude=m["target_peak_displacement_px"], **s["chirp_metrics"])
    if s["peak_metrics"]:
        row.update(expected=len(s["expected_frequencies_hz"]), reported=len(s["dominant_frequencies_hz"]), **s["peak_metrics"])
    g = pd.read_csv(folder / "ground_truth_error_metrics.csv")
    for (axis, quantity), group in g.groupby(["axis","quantity"]):
        tab = group.set_index("method")
        f, t = tab.loc["frequency_domain"], tab.loc["time_domain"]
        pairs.append(dict(experiment=folder.name, family=m["motion_type"], axis=axis,
                          quantity=quantity, diff=f.rmse-t.rmse))
        if axis == "x" and quantity in ("velocity","acceleration"):
            for suffix, value in [("fd_rmse",f.rmse), ("td_rmse",t.rmse),
                                  ("fd_nrmse",f.nrmse_by_reference_rms), ("td_nrmse",t.nrmse_by_reference_rms),
                                  ("reduction_pct",100*(f.rmse-t.rmse)/f.rmse)]:
                row[quantity+"_"+suffix] = value
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(a.out/"experiments.csv", index=False)
discrete = df[df.expected > 0]
assert len(df)==30 and len(discrete)==26
counts = discrete[["expected","reported","matched_peak_count"]].sum()
chirps = df[df.chirp_ridge_rmse_hz.notna()]
assert len(chirps)==4
summary = dict(counts=counts.to_dict(), macro_precision=discrete.peak_precision.mean(),
               macro_recall=discrete.peak_recall.mean(), micro_precision=counts.matched_peak_count/counts.reported,
               micro_recall=counts.matched_peak_count/counts.expected,
               pooled_mae=np.average(discrete.mean_matched_frequency_error_hz,weights=discrete.matched_peak_count),
               mean_experiment_mae=discrete.mean_matched_frequency_error_hz.mean(),
               median_experiment_mae=discrete.mean_matched_frequency_error_hz.median(),
               chirp_rmse=chirps.chirp_ridge_rmse_hz.median(),
               chirp_within25=chirps.chirp_ridge_within_tolerance_fraction.median())
pairs = pd.DataFrame(pairs)
tests = []
for cols in [["axis","quantity"],["axis","quantity","family"]]:
    for keys, group in pairs.groupby(cols):
        if len(group)<2:
            continue
        d = group["diff"].to_numpy()
        test = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        tests.append(dict(zip(cols,keys), n=len(d), W=test.statistic, p=test.pvalue))
tests = pd.DataFrame(tests)
ordered = tests.p.dropna().sort_values()
tests.loc[ordered.index,"holm"] = np.minimum(1,(ordered*np.arange(len(ordered),0,-1)).cummax())
tests.to_csv(a.out/"holm_family.csv", index=False)
for q in ["velocity","acceleration"]:
    diff = pairs[(pairs.axis=="x") & (pairs.quantity==q)]["diff"].to_numpy()
    ranks = rankdata(np.abs(diff[diff!=0]))
    reductions = df[q+"_reduction_pct"].to_numpy()
    rng = np.random.default_rng(2027)
    medians = np.median(rng.choice(reductions, size=(100000,len(df)),replace=True),axis=1)
    test = tests[(tests.axis=="x") & (tests.quantity==q) & tests.family.isna()].iloc[0]
    summary[q] = dict(fd=df[q+"_fd_nrmse"].median(), td=df[q+"_td_nrmse"].median(),
        wins=int((diff>0).sum()), reduction=np.median(reductions), ci=np.percentile(medians,[2.5,97.5]).tolist(),
        W=test.W, raw_p=test.p, holm_p=test.holm, rank_biserial=np.sum(np.sign(diff[diff!=0])*ranks)/ranks.sum())
summary["td_acceleration_nrmse_by_displacement"] = df.groupby("amplitude").acceleration_td_nrmse.median().to_dict()
(a.out/"summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
