"""Independent assertions on data, numerical methods and deliverable coverage."""
import argparse
import hashlib
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import wilcoxon

def verify(dataset,baseline,root):
    results=root/"results"
    table=lambda name:pd.read_csv(results/(name+".csv"),float_precision="round_trip")
    for line in (dataset/"checksums.sha256").read_text().splitlines():
        digest,filename=line.split(None,1)
        assert hashlib.sha256((dataset/filename).read_bytes()).hexdigest()==digest
    cent=table("centroid_accuracy");bounds=table("boundary_sensitivity")
    gaps=table("missing_bin_analysis");matches=table("frequency_error_analysis")
    assert len(cent)==39*3 and len(bounds)==39*2*7 and len(gaps)==39*2
    assert len(matches)==237 and len(table("candidate_audit"))==861
    for folder in sorted((dataset/"experiments").glob("experiment_*")):
        case=folder.name
        b=pd.read_csv(baseline/case/"binned_event_signals.csv",float_precision="round_trip")
        k=pd.read_csv(baseline/case/"displacement_velocity_acceleration.csv",float_precision="round_trip")
        cfg=json.loads((baseline/case/"analysis_summary.json").read_text())
        r=cfg["roi"]
        with h5py.File(folder/"events.h5") as f:events=f["events"][:].astype(np.int64)
        e=events[(events[:,1]>=r["x_min"])&(events[:,1]<r["x_max"])&(events[:,2]>=r["y_min"])&(events[:,2]<r["y_max"])]
        # Match floating-point arithmetic of the archived bin width.
        ids=np.floor(((e[:,0]-e[0,0])*1e-6)/(50*1e-6)).astype(int)
        observed=e[:,3]>0
        counts=np.bincount(ids[observed],minlength=len(b))
        assert np.array_equal(counts,b.positive_count), case
        for axis in cfg["active_axes"]:
            col={"x":1,"y":2}[axis]
            for idx in np.flatnonzero(counts)[::137]:
                direct=e[(ids==idx)&observed,col].mean()
                assert np.isclose(direct,b[axis+"_on_centroid"].iloc[idx])
            valid=counts>0
            y=k[axis+"_displacement_gt"].to_numpy()
            v=b[axis+"_on_centroid"].to_numpy()-{"x":120,"y":90}[axis]
            mask=(cent.experiment==case)&(cent.axis==axis)&(cent.stage=="observed")
            assert np.isclose(cent.loc[mask,"nrmse"].item(),np.linalg.norm((v-y)[valid])/np.linalg.norm(y[valid]))
            for q in ["velocity","acceleration"]:
                for rec in bounds[(bounds.axis==axis)&(bounds.experiment==case)&(bounds.quantity==q)].itertuples():
                    trim=int(np.ceil(len(k)*rec.trim_pct/100))
                    interior=(np.arange(len(k))>=trim)&(np.arange(len(k))<len(k)-trim)
                    sel=~interior if rec.region=="boundary" else interior
                    residual=(k[axis+"_"+q+"_time_domain"]-k[axis+"_"+q+"_gt"])[sel]
                    assert np.isclose(np.linalg.norm(residual)/np.sqrt(sel.sum()),rec.td_rmse)
    t=np.arange(8000)/20000
    sg=table("savitzky_golay_analysis");dr=table("differentiation_response")
    for freq in [50,500,1000,2000]:
        w=2*np.pi*freq
        x=np.sin(w*t)
        g1=np.gradient(x,1/20000,edge_order=2);g2=np.gradient(g1,1/20000,edge_order=2)
        sl=slice(6,-6);r=dr.loc[dr.frequency_hz==freq].iloc[0]
        assert np.allclose(g1[sl],w*np.cos(w*t[sl])*r.velocity_gain_ratio,atol=1e-6)
        assert np.allclose(g2[sl],-w*w*x[sl]*r.acceleration_gain_ratio,atol=.0001)
        assert np.allclose(savgol_filter(x,11,3)[sl],x[sl]*sg.loc[sg.frequency_hz==freq,"gain"].item())
    pk=table("peak_count_sensitivity")
    ten=pk[(pk.family=="all")&(pk.K=="10")].iloc[0]
    assert (ten.matched,ten.reference,ten.reported)==(237,861,258)
    assert np.isclose(matches.abs_error_hz.mean(),ten.pooled_MAE)
    assert np.isclose(matches.groupby("experiment").abs_error_hz.mean().mean(),ten.macro_MAE)
    assert (pk.matched+pk.fn==pk.reference).all() and (pk.matched+pk.fp==pk.reported).all()
    assert (pk.micro_R<=pk.cap_micro+1e-12).all()
    baseline_table=table("experiment_level_summary")
    for q in ["velocity","acceleration"]:
        d=baseline_table[q+"_fd_rmse"]-baseline_table[q+"_td_rmse"]
        w=wilcoxon(d)
        assert w.statistic==1 and (d>0).sum()==29
        assert np.isclose(w.pvalue*35,1.30385160446167e-7,rtol=1e-10,atol=0)
    for file in results.glob("*.csv"):
        data=pd.read_csv(file)
        assert not data.empty and not np.isinf(data.select_dtypes("number")).any().any(),file.name
    from quality_checks import check
    check(root,baseline)
    print("PASS: release manifest, 30 event streams, ROI counts, sampled centroid arithmetic, paired errors, operator responses, matching totals, Holm reference and finite numeric outputs.")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ["dataset","baseline","root"]:p.add_argument(name,type=Path)
    a=p.parse_args();verify(a.dataset,a.baseline,a.root)


