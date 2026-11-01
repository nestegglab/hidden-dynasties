"""DFA sensitivity grid (mock-panel check, pre-registered 27 Oct 2026).

The DFA-consistent effect of trusts depends on two measurement choices in `dfa_top_shares`:
  c        the share of trust assets the SCF records and attributes to the holding family
           (main results: c = 0.0231 / tau, about 0.26 at the v2c trust share), and
  theta    the share of the unrecorded trust wealth that the DFA hands to the top 1% of personal
           wealth (main results: 0.509, the top 1%'s share of equities and mutual funds).
This script solves the v2c base economy once at its general-equilibrium interest rate, saves the
household distribution, and recomputes DFA-measured top shares over a grid of (c, theta).
The economy without trusts has no trust assets, so its measured top shares do not depend on (c, theta);
the measured effect is DFA(base; c, theta) - top share(off).

Writes results/dfa_grid_v2c.json and results/dfa_grid_v2c_dist.npz (weights, a, d for reanalysis).
Restart-safe: if the .npz exists the solve is skipped.
Run from code/:  python dfa_grid.py
"""
import os, sys, json, time
from dataclasses import replace
import numpy as np
import calibrate_C as cc
import run_c5 as r5
from layerC import EconomyC, dfa_top_shares
from equilibrium import ky_for

RES = cc.RES
# calibration: "python dfa_grid.py" = v2c (as run on 27 Oct); "python dfa_grid.py cal=v2c_free3" = any other
key = next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("cal=")), "v2c")
if key not in r5.CAL:                          # read its parameter file, as run_c5 does
    pf = os.path.join(RES, f"layerC4_{key}_params.json")
    r5.CAL[key] = dict(x=np.array([json.load(open(pf))["params"][k] for k in cc.NAMES]), tag=f"{key}_")
cal = r5.CAL[key]
cc.use_v2b()                                   # same setup run_c5 uses for the v2c calibrations (nd = 55)
bp, ef, su = cc.base_params()

base = json.load(open(os.path.join(RES, f"layerC5_{cal['tag']}base.json")))
off = json.load(open(os.path.join(RES, f"layerC5_{cal['tag']}off.json")))
r_star = base["r_star"]
npz = os.path.join(RES, f"dfa_grid_{key}_dist.npz")

if not os.path.exists(npz):
    t = time.time()
    p0 = replace(cc.to_params(cal["x"], bp), **cal.get("extra", {}))
    p = replace(p0, KY=ky_for(p0, r_star))
    e = EconomyC(p, ef, su).solve().distribution(tol=1e-6, wtol=cc.WTOL, maxgen=120)
    m = e.moments()
    W = e.dist * e.cohort[:, None]
    w = W.sum(axis=0) / W.sum()
    np.savez(npz, w=w, a=e.fa, d=e.fd, top1=m["top1"], top01=m["top0.1"], secs=time.time() - t)
    print(f"solved base at r*={r_star:.4%} in {time.time()-t:.0f}s top1={m['top1']:.4f} "
          f"(json: {base['ge_moments']['top1']:.4f})", flush=True)

z = np.load(npz); w, a, d = z["w"], z["a"], z["d"]
A = (w * a).sum(); Dt = (w * d).sum(); tau = Dt / (A + Dt)
off_top1 = off["ge_moments"]["top1"]; off_top01 = off["ge_moments"]["top0.1"]
cs = [0.0, 0.05, 0.10, 0.1122, 0.15, 0.20, 0.26, 0.30, 0.40, 0.50, 0.75, 1.0]
ths = [0.40, 0.45, 0.509, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
c_main = min(1.0, 0.0231 / tau)
if c_main not in cs:
    cs = sorted(cs + [round(c_main, 4)])
rows = []
for c in cs:
    for th in ths:
        s = dfa_top_shares(w, a, d, theta=th, scf_share=c * tau)
        rows.append({"c": c, "theta": th, "top1": s[0.01], "top01": s[0.001],
                     "effect1": s[0.01] - off_top1, "effect01": s[0.001] - off_top01})
true1 = float(z["top1"]); true_eff = true1 - off_top1
out = {"r_star": r_star, "tau": tau, "c_main": c_main, "true_top1": true1, "off_top1": off_top1,
       "true_effect1": true_eff, "grid": rows}
json.dump(out, open(os.path.join(RES, f"dfa_grid_{key}.json"), "w"), indent=1, default=float)
print(f"tau={tau:.4f} c_main={c_main:.4f} true effect top1={100*true_eff:+.2f} pp")
print("measured top-1% effect (pp); rows c, columns theta_DFA; share invisible = 1 - measured/true")
print("c      " + " ".join(f"{t:6.3f}" for t in ths))
for c in cs:
    print(f"{c:6.4f} " + " ".join(f"{100*r['effect1']:+6.2f}" for r in rows if r["c"] == c))
