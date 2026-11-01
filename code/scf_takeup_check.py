"""Untargeted check (27 Oct 2026): does trust take-up across the top of the SCF wealth distribution
separate the calibrated theta = 1.38 economy from the theta = 1 economy?

SCF 2022 summary extract (data/raw/SCFP2022.csv). TRUSTS = trusts and managed investment accounts the
household reports owning; discretionary dynasty trusts held for beneficiaries are largely missed. "Has a trust"
= TRUSTS >= $100k (the model's minimum trust). Groups by NETWORTH, averaged over the five implicates.
Uncertainty: household bootstrap (200 draws; ignores the SCF's survey design, so a rough guide).
Model counterparts are read from the GE results files.
Run from code/:  python scf_takeup_check.py
"""
import json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(os.path.join(ROOT, "data", "raw", "SCFP2022.csv"), usecols=["Y1", "YY1", "WGT", "TRUSTS", "NETWORTH"])
d["imp"] = d.Y1 % 10


def ratios(g):
    out = []
    for _, gi in g.groupby("imp"):
        gi = gi.sort_values("NETWORTH", ascending=False)
        w = (gi.WGT / gi.WGT.sum()).values; c1 = w.cumsum(); c0 = c1 - w
        grp = lambda lo, hi: np.clip(np.minimum(c1, hi) - np.maximum(c0, lo), 0, None)
        p01, p99, p1 = grp(0, .001), grp(.001, .01), grp(0, .01)
        has = gi.TRUSTS.values >= 1e5; T = gi.TRUSTS.values; N = gi.NETWORTH.values
        take = (p01 * has).sum() / p01.sum() / ((p99 * has).sum() / p99.sum())        # take-up, top 0.1% / 99-99.9
        asset = ((p01 * T).sum() / (p01 * N).sum()) / ((p1 * T).sum() / (p1 * N).sum())  # trust share, top 0.1% / top 1%
        out.append((take, asset))
    return np.mean(out, axis=0)


pt = ratios(d)
rng = np.random.default_rng(0); hh = d.YY1.unique(); idx = d.groupby("YY1").indices
B = np.array([ratios(d.iloc[np.concatenate([idx[k] for k in rng.choice(hh, len(hh))])]) for _ in range(200)])
print(f"SCF 2022   take-up ratio {pt[0]:.2f} (sd {B[:,0].std():.2f}, 90% {np.percentile(B[:,0],5):.2f}-{np.percentile(B[:,0],95):.2f})"
      f"   asset ratio {pt[1]:.2f} (sd {B[:,1].std():.2f}, 90% {np.percentile(B[:,1],5):.2f}-{np.percentile(B[:,1],95):.2f})")

RES = os.path.join(ROOT, "results")
for name, f in [("v2c (theta 1.38)", "layerC5_v2c_base.json"), ("theta = 1", "layerC5_sweep_v2c_theta1_base.json"),
                ("v2b (theta 0.94)", "layerC5_v2b_base.json"), ("s15", "layerC5_sweep_v2c_s15_base.json"),
                ("s27", "layerC5_sweep_v2c_s27_base.json")]:
    m = json.load(open(os.path.join(RES, f)))["ge_moments"]
    t01, t1 = m["takeup_top0.1"], m["takeup_top1"]
    t99 = (t1 - 0.1 * t01) / 0.9                     # 99-99.9 from the top 1% and top 0.1% (by total wealth)
    print(f"{name:18s} take-up ratio {t01/t99:.2f}   asset ratio "
          f"{m['trust_share_of_top0.1_wealth']/m['trust_share_of_top1_wealth']:.2f}")
