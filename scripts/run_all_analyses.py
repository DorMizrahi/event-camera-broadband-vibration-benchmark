"""Run evidence diagnostics on the unmodified baseline outputs."""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy import signal, stats

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"github_release" if (ROOT/"github_release").is_dir() else ROOT
sys.path.insert(0,str(SOURCE/"estimator"))
from event_vibration_standalone import AnalysisConfig
from event_vibration_standalone.core import KinematicsEstimator, SpectralEstimator
from event_vibration_standalone.evaluation import match_peaks

FS = 20000
rms = lambda x: np.sqrt(np.mean(np.square(x)))
def compare(x,y):
    return dict(rmse=rms(x-y), nrmse=rms(x-y)/rms(y), correlation=np.corrcoef(x,y)[0,1],
                rms_ratio=rms(x)/rms(y), rms_amplitude_error_px=rms(x)-rms(y), mean_error_px=np.mean(x-y))

def analyze(dataset, baseline, output):
    output.mkdir(parents=True,exist_ok=True)
    tables=defaultdict(list)
    def add(table,**kwargs):
        tables[table].append({**labels,**kwargs})
    for directory in sorted((dataset/"experiments").glob("experiment_*")):
        name=directory.name
        meta=json.loads((directory/"metadata.json").read_text())
        folder=baseline/name
        read=lambda filename: pd.read_csv(folder/(filename+".csv"),float_precision="round_trip")
        b=read("binned_event_signals")
        kin=read("displacement_velocity_acceleration")
        time=b.t_sec.to_numpy()
        n=len(time)
        assert np.allclose(np.diff(time),1/FS) and np.array_equal(time,kin.t_sec)
        axes=meta["active_axes"]
        labels=dict(experiment=name,family=meta["motion_type"],amplitude=meta["target_peak_displacement_px"],
                    axes=axes,contrast=round(meta["marker_contrast"],8))
        frequencies=np.array(meta["active_frequencies_hz"])
        chirp=labels["family"]=="chirp"
        config=AnalysisConfig(active_axes=axes)
        spec=SpectralEstimator(config)
        recon=KinematicsEstimator(config)
        add("experiment_factor_allocation",**{k:json.dumps(v) if isinstance(v,list) else v for k,v in meta.items()})
        stages={"baseline":{},"no_sg":{},"hold":{},"off":{},"combined":{},"difference":{}}
        for axis in axes:
            gt=kin[axis+"_displacement_gt"].to_numpy()
            origin={"x":120,"y":90}[axis]
            for pol,count in [("on","positive_count"),("off","negative_count")]:
                missing=b[count].to_numpy()==0
                edges=np.diff(np.r_[False,missing,False].astype(int))
                starts,stops=np.where(edges==1)[0],np.where(edges==-1)[0]
                for start,stop in zip(starts,stops):
                    add("missing_runs",axis=axis,polarity=pol,start_bin=start,start_s=time[start],length=stop-start)
                add("missing_bin_analysis",axis=axis,polarity=pol,bins=n,observed=(~missing).sum(),
                    missing=missing.sum(),percent=100*missing.mean(),max_run_us=50*max(stops-starts,default=0),
                    **{f"quarter_{i+1}_missing":part.sum() for i,part in enumerate(np.array_split(missing,4))})
            for rep,col,mask in [("on","_on_centroid",b.positive_count>0),("off","_off_centroid",b.negative_count>0),
                    ("combined","_centroid",b.event_count>0),("difference","_polarity_difference",(b.positive_count>0)&(b.negative_count>0))]:
                col=axis+col
                values=b[col].to_numpy()
                assert np.isfinite(values).all()
                relative=values-(0 if rep=="difference" else origin)
                disp=recon.frequency_domain(values).displacement
                if rep=="on":
                    assert np.allclose(disp,kin[axis+"_displacement_frequency_domain"],atol=1e-11)
                for stage,series,valid in [("observed",relative,mask),("interpolated",relative,np.ones(n,bool)),
                                              ("reconstruction",disp,np.ones(n,bool))]:
                    metrics=compare(series[valid],gt[valid])
                    add("polarity_comparison",axis=axis,representation=rep,stage=stage,valid_samples=sum(valid),
                        coverage=mask.mean(),**metrics)
                    if rep=="on":
                        add("centroid_accuracy",axis=axis,stage=stage,valid_samples=sum(valid),**metrics)
                f,coh=signal.coherence(gt,relative,fs=FS,nperseg=512,noverlap=256)
                _,weight=signal.welch(gt,fs=FS,nperseg=512,noverlap=256)
                band=(f>=50)&(f<=2000)
                add("coherence",axis=axis,representation=rep,gt_power_weighted_msc=np.average(coh[band],weights=weight[band]))
                # Hann projection at the reference line or along the reference Chirp.
                for stage,series in [("interpolated",relative),("reconstruction",disp)]:
                    if chirp:
                        centers=read("chirp_ridge").time_sec
                        start,end=frequencies
                        rate=np.log(end/start)/meta["duration_sec"]
                        log="log" in meta["spectral_envelope"]
                        phase=start*np.expm1(rate*time)/rate if log else start*time+(end-start)*time**2/(2*meta["duration_sec"])
                        tasks=[(start*np.exp(rate*t) if log else start+(end-start)*t/meta["duration_sec"],
                                t,np.abs(time-t)<0.025,phase,np.nan) for t in centers]
                    else:
                        tasks=[(f0,np.nan,slice(None),f0*time,min(abs(frequencies[frequencies!=f0]-f0),default=np.inf)) for f0 in frequencies]
                    for freq,center,segment,cycles,spacing in tasks:
                        reference=gt[segment]
                        basis=np.hanning(len(reference))*np.exp(-2j*np.pi*cycles[segment])
                        numerator=np.dot(series[segment]-np.mean(series[segment]),basis)
                        denominator=np.dot(reference-np.mean(reference),basis)
                        reliable=abs(denominator)>1e-8 and (chirp or spacing>=2*FS/n)
                        ratio=numerator/denominator if abs(denominator)>1e-8 else complex(np.nan,np.nan)
                        add("component_transfer",axis=axis,representation=rep,stage=stage,frequency_hz=freq,center_s=center,
                            spacing_hz=spacing,resolved=reliable,gain=abs(ratio),phase_deg=np.angle(ratio,deg=True))
                key="baseline" if rep=="on" else rep
                stages[key][axis]=b[col].to_numpy()
                if rep=="on":
                    held=b[col].where(mask).ffill().bfill().to_numpy()
                    stages["hold"][axis]=held
                    stages["no_sg"][axis]=values
                    for method,triplet in [("frequency_domain",recon.frequency_domain(held)),("time_domain",recon.time_domain(held))]:
                        for q in ["displacement","velocity","acceleration"]:
                            ref=kin[axis+"_"+q+"_gt"].to_numpy()
                            original=kin[axis+"_"+q+"_"+method].to_numpy()
                            add("interpolation_kinematics",axis=axis,method=method,quantity=q,linear_nrmse=rms(original-ref)/rms(ref),
                                hold_nrmse=rms(getattr(triplet,q)-ref)/rms(ref))
            for q in ["velocity","acceleration"]:
                ref=kin[axis+"_"+q+"_gt"].to_numpy()
                fd=kin[axis+"_"+q+"_frequency_domain"].to_numpy()
                td=kin[axis+"_"+q+"_time_domain"].to_numpy()
                for pct in [0,.01,.025,.05]:
                    trim=int(np.ceil(n*pct))
                    interior=(np.arange(n)>=trim)&(np.arange(n)<n-trim)
                    for region,mask in ([("full",interior)] if not trim else [("interior",interior),("boundary",~interior)]):
                        ef,et=rms((fd-ref)[mask]),rms((td-ref)[mask])
                        add("boundary_sensitivity",axis=axis,quantity=q,trim_pct=100*pct,region=region,samples=mask.sum(),
                            fd_rmse=ef,td_rmse=et,fd_nrmse=ef/rms(ref[mask]),td_nrmse=et/rms(ref[mask]),reduction_pct=100*(ef-et)/ef)
                f=np.fft.rfftfreq(n,1/FS)
                high=(f>=1000)&(f<=2000)
                power=lambda x: abs(np.fft.rfft(x))**2
                add("differentiation_empirical",axis=axis,quantity=q,
                    reference_high_power_fraction=power(ref)[high].sum()/power(ref)[1:].sum(),
                    td_fd_high_power_ratio=power(td)[high].sum()/power(fd)[high].sum(),
                    td_fd_high_error_power_ratio=power(td-ref)[high].sum()/power(fd-ref)[high].sum(),
                    reduction_pct=100*(1-rms(td-ref)/rms(fd-ref)))
        for variant,signals in stages.items():
            estimator=SpectralEstimator(replace(config,enable_centroid_smoothing=variant!="no_sg"))
            names={a:a+"_on_centroid" for a in axes}
            frame=pd.DataFrame({names[a]:v for a,v in signals.items()})
            spectra={col:estimator.full_periodogram(estimator.prepare_signal(frame,col)) for col in names.values()}
            spectrum,_,_=estimator.primary_motion_periodogram(spectra,frame)
            if chirp:
                ridge,_=estimator.chirp_ridge(frame,names)
                original=read("chirp_ridge")
                assert np.allclose(original.time_sec,ridge.time_sec)
                err=ridge.estimated_ridge_hz-original.expected_ridge_hz
                if variant=="baseline": assert np.allclose(original.estimated_ridge_hz,ridge.estimated_ridge_hz)
                add("spectral_variants",variant=variant,ridge_rmse=rms(err),within25=(abs(err)<=25).mean())
                continue
            allpeaks=SpectralEstimator(replace(estimator.config,top_k_peaks=len(spectrum))).extract_peaks(spectrum,"diagnostic")
            assert allpeaks.prominence.notna().all(), "Inspect fallback candidate rule"
            if variant=="baseline":
                _,all_matches=match_peaks(frequencies,allpeaks,replace(config,top_k_peaks=len(allpeaks)))
                step=FS/n
                for row in all_matches.itertuples():
                    frequency=row.expected_frequency_hz
                    dist=np.abs(allpeaks.frequency_hz-frequency)
                    add("candidate_audit",frequency_hz=frequency,candidates_within10=(dist<=10).sum(),
                        all_candidates_matched=row.matched,actual_grid_hz=step,nearest_grid_distance_hz=abs(frequency-round(frequency/step)*step),
                        nominal_grid_distance_hz=abs(frequency-round(frequency/5)*5),
                        nearest_spacing_hz=min(abs(frequencies[frequencies!=frequency]-frequency),default=np.inf))
            for label in ([5,10,20,30,50,100,"reference_count"] if variant=="baseline" else [10]):
                k=len(frequencies) if label=="reference_count" else int(label)
                peaks=allpeaks.head(k)
                _,matches=match_peaks(frequencies,peaks,replace(config,top_k_peaks=k))
                matched=matches[matches.matched]
                m=len(matched)
                metrics=dict(reference=len(frequencies),reported=len(peaks),matched=m,fp=len(peaks)-m,fn=len(frequencies)-m,
                    precision=m/len(peaks),recall=m/len(frequencies),mae=matched.absolute_error_hz.mean(),
                    abs_sum=matched.absolute_error_hz.sum(),cap_recall=min(1,k/len(frequencies)),candidates=len(allpeaks))
                if variant=="baseline": add("peak_count_experiments",K=label,**metrics)
                if label==10:
                    add("spectral_variants",variant=variant,**metrics)
                    if variant=="baseline":
                        assert np.allclose(peaks.frequency_hz,read("dominant_frequencies").frequency_hz)
                        grid=FS/n
                        for row in matched.itertuples():
                            f0=row.expected_frequency_hz
                            peak=peaks.loc[peaks.frequency_hz==row.predicted_frequency_hz].iloc[0]
                            gap=min(abs(frequencies[frequencies!=f0]-f0),default=np.inf)
                            offset=f0-round(f0/grid)*grid
                            add("frequency_error_analysis",reference_hz=f0,refined_hz=peak.frequency_hz,bin_hz=peak.discrete_frequency_hz,
                                signed_error_hz=peak.frequency_hz-f0,abs_error_hz=abs(peak.frequency_hz-f0),
                                bin_error_hz=peak.discrete_frequency_hz-f0,grid_hz=grid,grid_offset_hz=offset,
                                grid_aligned=abs(offset)<1e-6,nearest_spacing_hz=gap)
    assert len(tables["experiment_factor_allocation"])==30
    for name,records in tables.items():
        pd.DataFrame(records).to_csv(output/(name+".csv"),index=False)
    from summaries import summarize
    summarize(output)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ["dataset","baseline","results"]: parser.add_argument(key,type=Path)
    args=parser.parse_args()
    analyze(args.dataset,args.baseline,args.results)


