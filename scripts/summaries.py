from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_coeffs

def summarize(out):
    def read(name):
        return pd.read_csv(out/(name+".csv"))
    def save(name, df):
        df.to_csv(out/(name+".csv"),index=False)
    peaks=read("peak_count_experiments")
    rows=[]
    for family,frame in [("all",peaks),*peaks.groupby("family")]:
        for k,g in frame.groupby("K",sort=False):
            rows.append(dict(family=family,K=k,n=len(g),macro_P=g.precision.mean(),macro_R=g.recall.mean(),
                micro_P=g.matched.sum()/g.reported.sum(),micro_R=g.matched.sum()/g.reference.sum(),
                macro_MAE=g.mae.mean(),pooled_MAE=g.abs_sum.sum()/g.matched.sum(),cap_macro=g.cap_recall.mean(),
                cap_micro=np.minimum(g.reference,g.reference if k=="reference_count" else int(k)).sum()/g.reference.sum(),
                **g[["matched","reference","reported","fp","fn"]].sum().to_dict()))
    save("peak_count_sensitivity",pd.DataFrame(rows))
    f=np.arange(50,2001,1)
    w=2*np.pi*f
    gain=np.sinc(2*f/20000)
    coeff=savgol_coeffs(11,3)
    sg=np.exp(-1j*w[:,None]*np.arange(-5,6)/20000)@coeff
    save("differentiation_response",pd.DataFrame(dict(frequency_hz=f,fd_velocity=w,td_velocity=w*gain,
            fd_acceleration=w**2,td_acceleration=w**2*gain**2,velocity_gain_ratio=gain,acceleration_gain_ratio=gain**2)))
    save("savitzky_golay_analysis",pd.DataFrame(dict(frequency_hz=f,gain=abs(sg),dB=20*np.log10(abs(sg)),phase_deg=np.angle(sg,deg=True))))
    save("sg_coefficients",pd.DataFrame(dict(offset=np.arange(-5,6),coefficient=coeff)))
    boundary=read("boundary_sensitivity").query("axis=='x'")
    rows=[]
    for keys,g in boundary.groupby(["quantity","trim_pct","region"]):
        d=g.reduction_pct.to_numpy()
        rng=np.random.default_rng(2027)
        ci=np.percentile(np.median(rng.choice(d,(10000,len(d))),axis=1),[2.5,97.5])
        rows.append(dict(quantity=keys[0],trim_pct=keys[1],region=keys[2],n=len(d),fd_nrmse=g.fd_nrmse.median(),
              td_nrmse=g.td_nrmse.median(),wins=(d>0).sum(),median_pct=np.median(d),ci_low=ci[0],ci_high=ci[1]))
    save("boundary_summary",pd.DataFrame(rows))
    for table,keys,cols in [
        ("centroid_accuracy",["stage"],["nrmse","correlation","mean_error_px","rms_ratio"]),
        ("missing_bin_analysis",["polarity"],["percent","max_run_us"]),
        ("polarity_comparison",["representation","stage"],["nrmse","rms_ratio","correlation"]),
        ("experiment_level_summary",[],["acceleration_td_nrmse","velocity_td_nrmse"])]:
        t=read(table)
        if "axis" in t: t=t[t.axis=="x"]
        rows=[]
        for dimension in ["all","family","amplitude","axes"]:
            for level,g in ([("all",t)] if dimension=="all" else t.groupby(dimension)):
                for categories,z in (g.groupby(keys) if keys else [((),g)]):
                    if not isinstance(categories,tuple): categories=(categories,)
                    rows.append(dict(stratum=dimension,level=level,n=len(z),**dict(zip(keys,categories)),**z[cols].median().to_dict()))
        save(table+"_strata",pd.DataFrame(rows))
    errors=read("frequency_error_analysis")
    errors["spacing_band"]=pd.cut(errors.nearest_spacing_hz,[0,5,10,20,50,np.inf])
    errors["frequency_band"]=pd.cut(errors.reference_hz,[0,500,1000,1500,2000])
    rows=[]
    for col in ["all","grid_aligned","spacing_band","frequency_band"]:
        for group,g in ([("all",errors)] if col=="all" else errors.groupby(col,observed=True)):
            rows.append(dict(stratum=col,level=str(group),matches=len(g),experiments=g.experiment.nunique(),
                signed_mean=g.signed_error_hz.mean(),signed_median=g.signed_error_hz.median(),sd=g.signed_error_hz.std(),
                abs_mean=g.abs_error_hz.mean(),abs_median=g.abs_error_hz.median(),p90=g.abs_error_hz.quantile(.9),
                p95=g.abs_error_hz.quantile(.95),max_abs=g.abs_error_hz.max(),bin_mae=g.bin_error_hz.abs().mean()))
    save("frequency_error_summary",pd.DataFrame(rows))
    factors=read("experiment_factor_allocation")
    save("factor_crosstab",factors.groupby(["family","amplitude","axes","contrast"]).size().reset_index(name="cases"))
    design=pd.get_dummies(factors[["family","amplitude","axes","contrast"]].astype(str),drop_first=True,dtype=float)
    X=np.column_stack([np.ones(len(design)),design])
    save("design_rank",pd.DataFrame([dict(n=len(X),columns=X.shape[1],rank=np.linalg.matrix_rank(X),occupied_cells=len(factors.groupby(["family","amplitude","axes","contrast"])),possible_cells=np.prod(factors[["family","amplitude","axes","contrast"]].nunique()))]))
    variants=read("spectral_variants")
    rows=[]
    for k,g in variants.groupby("variant"):
        d=g.dropna(subset=["reference"])
        rows.append(dict(variant=k,macro_P=d.precision.mean(),macro_R=d.recall.mean(),macro_MAE=d.mae.mean(),
                         matched=d.matched.sum(),reported=d.reported.sum(),median_ridge_rmse=g.ridge_rmse.median(),
                         median_within25=g.within25.median()))
    save("spectral_variant_summary",pd.DataFrame(rows))
    transfer=read("component_transfer").query("resolved and axis=='x'").copy()
    transfer["band"]=pd.cut(transfer.frequency_hz,[0,500,1000,1500,2000])
    transfer["signal_class"]=np.where(transfer.family=="chirp","chirp","stationary")
    groups=["signal_class","representation","stage","band"]
    def circular(x):
        return np.angle(np.exp(1j*np.deg2rad(x)).mean(),deg=True)
    summary=transfer.groupby(groups,observed=True).agg(n=("gain","size"),experiments=("experiment","nunique"),
        median_gain=("gain","median"),circular_phase_deg=("phase_deg",circular),
        phase_concentration=("phase_deg",lambda x:abs(np.exp(1j*np.deg2rad(x)).mean()))).reset_index()
    save("transfer_summary",summary)
