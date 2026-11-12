"""Render deterministic representative cases and cohort diagnostics from saved data."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import spectrogram, savgol_filter
import json

def generate(results,baseline,figures):
    read=lambda name:pd.read_csv(results/(name+".csv"))
    table=read("experiment_level_summary")
    def nearest(frame,metric):
        return frame.loc[(frame[metric]-frame[metric].median()).abs().sort_values(kind="stable").index[0],"experiment"]
    sparse=nearest(table[table.family=="sparse_multi_sine"],"mean_matched_frequency_error_hz")
    dense=nearest(table[table.family=="dense_multi_sine"],"peak_recall")
    small=nearest(table[table.amplitude==2.5],"acceleration_td_nrmse")
    chirp=nearest(table[table.family=="chirp"],"chirp_ridge_rmse_hz")
    cases={"sparse":sparse,"dense":dense,"small":small,"chirp":chirp}
    (results/"representative_cases.json").write_text(json.dumps(cases,indent=2))
    load=lambda case,name:pd.read_csv(baseline/case/(name+".csv"))
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    def panels(rows=1,cols=1):
        return plt.subplots(rows,cols,figsize=(6.8,2.65*rows),layout="constrained",squeeze=False)
    def save(fig,path):
        target=figures/path
        target.parent.mkdir(exist_ok=True,parents=True)
        for ext in [".png",".pdf"]:fig.savefig(target.with_suffix(ext),dpi=220)
        plt.close(fig)
    fig,ax=panels(2)
    b=load(small,"binned_event_signals")
    k=load(small,"displacement_velocity_acceleration")
    for a,(ylabel,series) in zip(ax[:,0],[("Displacement (px)",[(k.x_displacement_gt,"Reference"),(b.x_on_centroid-120,"ON centroid"),(k.x_displacement_frequency_domain,"Reconstructed")]),
        ("Acceleration (10⁶ px/s²)",[(k.x_acceleration_gt/1e6,"Reference"),(k.x_acceleration_frequency_domain/1e6,"Fourier"),(k.x_acceleration_time_domain/1e6,"Gradient")])]):
        for y,label in series:a.plot(1000*k.t_sec,y,label=label,lw=1)
        a.set(xlim=(80,90),xlabel="Time (ms)",ylabel=ylabel,title=small+"  •  x axis");a.legend(ncol=3,fontsize=8)
    save(fig,"kinematics/time_histories")
    fig,ax=panels(2)
    for a,case in zip(ax[:,0],[sparse,dense]):
        spec=load(case,"primary_vector_periodogram")
        peaks=load(case,"dominant_frequencies")
        truth=json.loads((baseline/case/"analysis_summary.json").read_text())["expected_frequencies_hz"]
        power=spec.periodogram_psd/spec.periodogram_psd.max()
        a.plot(spec.frequency_hz,10*np.log10(np.maximum(power,1e-12)),label="Primary spectrum",lw=.8)
        a.scatter(truth,np.full(len(truth),-53),marker="|",label="Reference components")
        a.scatter(peaks.frequency_hz,10*np.log10(peaks.power/spec.periodogram_psd.max()),marker="v",s=15,label="Reported peaks")
        a.set(xlim=(45,2005),ylim=(-56,5),xlabel="Frequency (Hz)",ylabel="Relative power (dB)",title=case);a.legend(ncol=3,fontsize=8)
    save(fig,"frequency/representative_spectra")
    fig,ax=panels()
    b=load(chirp,"binned_event_signals")
    s=json.loads((baseline/chirp/"analysis_summary.json").read_text())
    powers=[]
    for col in s["primary_signals"].values():
        f,t,p=spectrogram(savgol_filter(b[col]-b[col].mean(),11,3),20000,window="hann",nperseg=1000,noverlap=750,nfft=4096)
        powers.append(p)
    power=sum(powers);band=(f>=50)&(f<=2000)
    a=ax[0,0];im=a.pcolormesh(t*1000,f[band],10*np.log10(np.maximum(power[band]/power[band].max(),1e-8)),vmin=-40,vmax=0,shading="auto")
    fig.colorbar(im,ax=a,label="Relative power (dB)")
    ridge=load(chirp,"chirp_ridge")
    a.plot(t*1000,ridge.expected_ridge_hz,color="white",label="Reference")
    a.plot(t*1000,ridge.estimated_ridge_hz,"o",ms=3,color="orange",label="Estimated ridge")
    a.set(xlabel="Window center (ms)",ylabel="Frequency (Hz)",title=chirp);a.legend()
    save(fig,"chirp/ridge")
    fig,ax=panels(2)
    d=read("component_transfer").query("axis=='x' and resolved and family!='chirp' and stage=='interpolated' and representation=='on'")
    for a,col,ylabel in [(ax[0,0],"gain","ON/reference amplitude ratio"),(ax[1,0],"phase_deg","ON − reference phase (°)")]:
        a.scatter(d.frequency_hz,d[col],s=5,alpha=.35)
        a.axhline(1 if col=="gain" else 0,color="black",lw=.8);a.set(xlabel="Reference frequency (Hz)",ylabel=ylabel)
    save(fig,"centroid/amplitude_phase")
    fig,ax=panels(2)
    dif=read("differentiation_response");sg=read("savitzky_golay_analysis")
    for col,l in [("velocity_gain_ratio","First gradient"),("acceleration_gain_ratio","Two gradients")]:
        ax[0,0].plot(dif.frequency_hz,dif[col],label=l)
    ax[0,0].axhline(1,color="black",ls="--",label="Fourier");ax[0,0].set(ylabel="Gain / ideal gain",xlabel="Frequency (Hz)");ax[0,0].legend()
    ax[1,0].plot(sg.frequency_hz,sg.dB,label="SG(11, 3)");ax[1,0].set(ylabel="Centered-filter gain (dB)",xlabel="Frequency (Hz)");ax[1,0].legend()
    save(fig,"differentiation/operator_responses")
    fig,ax=panels(2);a=ax[0,0]
    peak=read("peak_count_sensitivity").query("family=='all' and K!='reference_count'");kk=peak.K.astype(int)
    for col,label in [("micro_P","Precision"),("micro_R","Recall"),("cap_micro","Cap ceiling")]:a.plot(kk,peak[col],"o-",label=label)
    a.set(xlabel="Maximum reported peaks K",ylabel="Pooled fraction");a.legend()
    err=read("frequency_error_analysis")
    ax[1,0].hist(err.signed_error_hz,bins=np.arange(-4,4.26,.25));ax[1,0].set(xlabel="Refined − reference frequency (Hz)",ylabel="Matched components")
    save(fig,"frequency/cap_and_bias")
    fig,ax=panels(2)
    for a,q in zip(ax[:,0],["velocity","acceleration"]):
        for _,row in table.iterrows():
            a.plot([0,1],[row[q+"_fd_nrmse"],row[q+"_td_nrmse"]],color="#667d96",alpha=.6,lw=.9)
        a.axhline(1,color="black",ls="--",lw=.8);a.set(xticks=[0,1],xticklabels=["Fourier","Gradient"],ylabel=q.capitalize()+" NRMSE",title="Paired x-axis errors across 30 experiments")
    save(fig,"kinematics/absolute_errors")
    fig,ax=panels(2);boundary=read("boundary_summary").query("quantity=='acceleration'")
    for region,g in boundary.groupby("region"):
        ax[0,0].plot(g.trim_pct,g.median_pct,"o-",label=region)
    ax[0,0].legend();ax[0,0].set(xlabel="Excluded per end (%)",ylabel="Median RMSE reduction (%)")
    missing=read("missing_bin_analysis").query("axis=='x' and polarity=='on'")
    case=missing.loc[missing.percent.idxmax(),"experiment"];b=load(case,"binned_event_signals")
    for j,col in enumerate(["positive_count","negative_count"]):
        times=b.t_sec[b[col]==0];ax[1,0].scatter(times*1000,np.full(len(times),j),marker="|")
    ax[1,0].set(yticks=[0,1],yticklabels=["ON","OFF"],ylim=(-.5,1.5),xlabel="Bin center (ms)",title=case+"  •  missing bins (highest ON missing fraction)")
    save(fig,"diagnostics/boundaries_missing")
    fig,ax=panels()
    k=load(sparse,"displacement_velocity_acceleration");b=load(sparse,"binned_event_signals")
    for v,l in [(k.x_displacement_gt,"Reference"),(b.x_on_centroid,"Interpolated ON"),(k.x_displacement_frequency_domain,"Reconstructed")]:
        amp=2*abs(np.fft.rfft((v-v.mean())*np.hanning(len(v))))/np.hanning(len(v)).sum()
        ax[0,0].semilogy(np.fft.rfftfreq(len(v),1/20000),np.maximum(amp,1e-8),label=l,lw=.8)
    ax[0,0].set(xlim=(50,2000),ylim=(1e-4,20),xlabel="Frequency (Hz)",ylabel="Hann amplitude (px)",title=sparse+"  •  x axis");ax[0,0].legend()
    save(fig,"centroid/displacement_spectra")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ["results","baseline","figures"]:parser.add_argument(key,type=Path)
    a=parser.parse_args()
    generate(a.results,a.baseline,a.figures)
