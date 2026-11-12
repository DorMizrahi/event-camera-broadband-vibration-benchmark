"""Independent checks of diagnostic variants, handling of gaps and representative figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def check(root,baseline):
    def read(name):return pd.read_csv(root/"results"/(name+".csv"),float_precision="round_trip")
    polarity=read("polarity_comparison")
    spectra=read("spectral_variants")
    transfer=read("component_transfer")
    gaps=read("missing_bin_analysis")
    runs=read("missing_runs")
    factor=read("experiment_factor_allocation")
    candidates=read("candidate_audit")
    assert spectra.groupby("variant").size().eq(30).all()
    assert len(factor)==30 and factor.random_seed.nunique()==30
    assert len(factor[["family","amplitude","axes","contrast"]].drop_duplicates())==21
    assert len(candidates)==861 and candidates.all_candidates_matched.sum()==758
    assert (candidates.nearest_grid_distance_hz<=1e-6).sum()==15
    for case in factor.itertuples():
        folder=baseline/case.experiment
        b=pd.read_csv(folder/"binned_event_signals.csv",float_precision="round_trip")
        k=pd.read_csv(folder/"displacement_velocity_acceleration.csv",float_precision="round_trip")
        for axis in case.axes:
            for label,col in [("on","positive_count"),("off","negative_count")]:
                g=gaps[(gaps.experiment==case.experiment)&(gaps.axis==axis)&(gaps.polarity==label)].iloc[0]
                missing=(b[col]==0).to_numpy()
                assert g.missing==missing.sum()
                z=runs[(runs.experiment==case.experiment)&(runs.axis==axis)&(runs.polarity==label)]
                reconstructed=np.zeros(len(b),bool)
                for gap in z.itertuples():
                    reconstructed[gap.start_bin:gap.start_bin+gap.length]=True
                assert np.array_equal(missing,reconstructed)
            both=(b.positive_count>0)&(b.negative_count>0)
            cplus=b[axis+"_on_centroid"];cminus=b[axis+"_off_centroid"]
            assert np.allclose(b.loc[both,axis+"_polarity_difference"],(cplus-cminus)[both])
            combined=(cplus*b.positive_count+cminus*b.negative_count)/b.event_count.replace(0,np.nan)
            ok=b.event_count>0
            assert np.allclose(b.loc[ok,axis+"_centroid"],combined[ok])
            for rep,col in [("on","_on_centroid"),("off","_off_centroid"),("combined","_centroid"),("difference","_polarity_difference")]:
                values=b[axis+col].to_numpy()
                vals=values-(0 if rep=="difference" else {"x":120,"y":90}[axis])
                y=k[axis+"_displacement_gt"].to_numpy()
                row=polarity[(polarity.experiment==case.experiment)&(polarity.axis==axis)&(polarity.representation==rep)&(polarity.stage=="interpolated")].iloc[0]
                assert np.isclose(row.rms_ratio,np.linalg.norm(vals)/np.linalg.norm(y))
            if case.family!="chirp":
                diag=transfer[(transfer.experiment==case.experiment)&(transfer.axis==axis)&(transfer.representation=="on")&(transfer.stage=="interpolated")].iloc[0]
                theta=2*np.pi*diag.frequency_hz*b.t_sec.to_numpy()
                h=np.hanning(len(b))
                def phasor(x):
                    centered=(x-x.mean())*h
                    return complex(np.sum(centered*np.cos(theta)),-np.sum(centered*np.sin(theta)))
                gain=phasor(cplus.to_numpy())/phasor(y)
                assert np.isclose(abs(gain),diag.gain)
                assert np.isclose(np.angle(gain,deg=True),diag.phase_deg)
        if case.family=="chirp":
            original=pd.read_csv(folder/"chirp_ridge.csv")
            diagnostic=spectra[(spectra.experiment==case.experiment)&(spectra.variant=="baseline")].iloc[0]
            assert np.isclose(diagnostic.ridge_rmse,np.linalg.norm(original.error_hz)/np.sqrt(len(original)))
    selection=json.loads((root/"results/representative_cases.json").read_text())
    assert set(selection.values()).issubset(factor.experiment)
    pngs=list((root/"figures").rglob("*.png"))
    assert len(pngs)==9
    for fig in pngs:
        assert fig.read_bytes().startswith(b"\x89PNG") and fig.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    print("PASS: independent polarity masks, weighted centroids, RMS ratios, complex projections, Chirp errors, candidate audit, factor allocation and nine figure pairs.")
