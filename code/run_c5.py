"""
C5: main experiments on the C4 v2 calibration, each with the interest rate re-solved so that the
capital market clears (general equilibrium). Partial-equilibrium results (r held at the calibrated
6%) come for free from each experiment's first evaluation.

Experiments (overrides of the calibrated ParamsC):
  base      as calibrated: tier-1 dynasty trusts (binding, tax-shielded) + ordinary lifetime gifts
  off       no dynasty trusts (trusts=False); lifetime gifts still available
  lock0     tax shield kept, lock-in removed: heirs can reach all principal (lock=0)
  noshield  lock-in kept, tax shield removed: trust assets enter the taxable estate (shield=False)

Restart-safe: every evaluation is appended to results/layerC5_<name>_evals.jsonl and reused; a
finished experiment (results/layerC5_<name>.json) is skipped.

Usage: python run_c5.py [cal=v2|v2b|v2c|v2c_free3|...] [names...]   (default: off lock0 noshield base)
"""
# ==================================================================================================
# FILE BANNER: run_c5.py -- the four main experiments, with r re-solved (paper Section 3.6)
# ==================================================================================================
# WHAT THIS FILE DOES
#   It takes a calibrated parameter vector (from the CAL dictionary; the final one is "v2c"), and for
#   each experiment (base, off, lock0, noshield) changes the relevant switch, then searches for the
#   annual interest rate r* at which the capital market clears (secant method with a bisection
#   safeguard, as in equilibrium.solve_r). Each experiment's results are saved to a JSON file:
#   r*, the moments at r* (general equilibrium, "GE") and at r = 6% (partial equilibrium, "PE").
#   The paper's effect of trusts is, for example, top1 in base minus top1 in off.
#
# WHERE IT SITS
#   Layer A (life cycle) -> Layer B (fortunes) -> Layer C (trusts) -> calibrate_C.py (parameters) ->
#   THIS DRIVER. run_sweep.py and run_gridcheck.py reuse this file's run() for their own base/off
#   experiments, by replacing the module variable `cal` before calling it.
#
# HOW IT IS LAUNCHED (author's PC, from a .bat launcher)
#   `py run_c5.py cal=v2c base off noshield lock0`
#   main() reads the arguments as follows:
#     - if the FIRST argument starts with "cal=", the text after "=" picks the calibration from CAL
#       (v2, v2b or v2c); for v2b and v2c it also switches the trust grid to nd = 55. Without it, the
#       old "v2" calibration (nd = 40) is used. (The docstring's "cal=v2|v2b" predates v2c.)
#     - every remaining argument is an experiment name (keys of EXPERIMENTS), run in the order given;
#       with none, the default order is off, lock0, noshield, base.
#
# FILES (results/ folder; <tag> is "" for v2, "v2b_" for v2b, "v2c_" for v2c)
#   Reads:  data/eflife.dat, data/surlife.dat, results/layerB_v2_params_bestof2.json (via
#           calibrate_C.base_params); results/layerC5_<tag><name>_evals.jsonl if present (cache);
#           results/layerC5_<tag><name>.json if present (then that experiment is skipped).
#   Writes: results/layerC5_<tag><name>_evals.jsonl (appended: one JSON line per evaluation),
#           results/layerC5_<tag><name>.json (the experiment's result, written atomically),
#           results/layerC5_<tag>log.txt (appended progress log).
#   e.g. for the final runs: results/layerC5_v2c_base.json, ..._off.json, ..._noshield.json,
#   ..._lock0.json.
#
# HOW LONG IT RUNS
#   One evaluation with trusts takes 35-60 minutes; without trusts (the "off" experiment) about
#   2 minutes. Each experiment needs 1 to 11 evaluations (usually a handful), so base, lock0 and
#   noshield take a few hours each and off a few minutes to tens of minutes.
#
# RESTART SAFETY
#   Every finished evaluation is appended to its cache file at once; a restarted run reads the cache
#   and continues the secant search from where it stopped, and skips experiments whose final JSON
#   already loads. CAUTION: the cache is identified only by calibration tag and experiment name, not
#   by the parameter values, so if CAL or EXPERIMENTS were edited, old cache files must be deleted
#   by hand or they would be reused.
#
# READING ORDER
#   CAL, EXPERIMENTS, solver settings -> scalars -> run (with its inner evaluate) -> main.
#
# PAPER: Section 3.6 (experiments: base, off, noshield, lock0, each with r re-solved); Section 3.3
#   (capital-market clearing).
#
# WORDS YOU'LL SEE
#   GE / PE     general equilibrium (r re-solved) / partial equilibrium (r held at 6%).
#   excess      (K_supply - K_b) / K_c - 1; > 0 means too much saving at this r, so r must fall.
#   r* (r_star) the rate where |excess| is smallest (the clearing rate).
#   tag         prefix of the result files for a calibration ("v2c_" for the final one).
#   cache       results/layerC5_<tag><name>_evals.jsonl, one JSON line per evaluation.
# ==================================================================================================

# PYTHON NOTE: gc = garbage collector (gc.collect() frees memory now); json reads/writes .json files.
import sys, os, json, time, gc
import numpy as np
# PYTHON NOTE: replace(obj, field=value) copies a @dataclass parameter object with fields changed.
from dataclasses import replace
# Make the modules next to this file importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_C as cc
from layerC import EconomyC
from equilibrium import ky_for                # r -> corporate K/Y (the model's price input)

# ==================================================================================================
# CALIBRATIONS (CAL), EXPERIMENTS AND SOLVER SETTINGS
# ==================================================================================================

RES = cc.RES                                  # results/ folder
X_V2 = cc.STARTS["V2"]                      # C4 v2 best point (evaluation 42)
# calibration to run on: "v2" (default) or "v2b" (C4 v2b best point, evaluation 51, nd = 55)
# CAL: one entry per calibration. Each entry is a dict with
#   x    the 7 calibrated parameters, order calibrate_C.NAMES: beta_annual, phi1, chi, fric_mu,
#        fric_nu, trust_weight (theta), gift_phi;
#   tag  prefix of the result files; desc  a description saved in each result file.
# Optional key "extra" (set by run_sweep.py) = further parameter overrides.
# "v2c" is the FINAL calibration (C4 v2c, evaluation 33: e.g. annual beta 0.9470, theta 1.382,
# friction mean 0.075). Evaluation numbers count calls of calibrate_C.Objective.
CAL = {"v2": dict(x=X_V2, tag="", desc="C4 v2, evaluation 42"),
       "v2b": dict(x=np.array([0.9454397485946558, -1.8738656265147329, 0.20514187576522447, 0.07986465007338328,
                               0.03203010498655948, 0.9377722776835895, -1.9272035892980588]),
                   tag="v2b_", desc="C4 v2b, evaluation 51 (nd = 55)"),
       "v2c": dict(x=np.array([0.9470258472085058, -1.5494726762187743, 0.1493978828816961, 0.07496201209382677,
                               0.02647145719235874, 1.3819218010771774, -1.7931406611023284]),
                   tag="v2c_", desc="C4 v2c (final), evaluation 33 (nd = 55, DFA-consistent top shares)")}
# Later calibrations are read from their calibrate_C output files, so their parameters are not
# copied here by hand. "v2c_theta1" is the full recalibration with theta fixed at 1 (pre-registered
# 27 Oct 2026). Any other key, e.g. cal=v2c_free3 (the MAIN calibration in the paper, theta 1.04),
# is read from results/layerC4_<key>_params.json in main() below. In the paper, "v2c" (theta 1.38)
# is the "alternative calibration".
_TH1 = os.path.join(RES, "layerC4_v2c_theta1_params.json")
if os.path.exists(_TH1):
    CAL["v2c_theta1"] = dict(x=np.array([json.load(open(_TH1))["params"][k] for k in cc.NAMES]),
                             tag="v2c_theta1_", desc="C4 v2c with theta = 1 (full recalibration)")
# The calibration in use; main() (or run_sweep.py / run_gridcheck.py) replaces it.
cal = CAL["v2"]
# The four experiments: parameter overrides applied on top of the calibrated ParamsC.
EXPERIMENTS = {
    "base":     {},                         # as calibrated: tier-1 trusts, lock = 1, shield on
    "off":      {"trusts": False},          # no dynasty trusts (lifetime gifts still available)
    "lock0":    {"lock": 0.0},              # heirs can draw down all principal (lock-in removed)
    "noshield": {"shield": False},          # trust assets enter the taxable estate
    # Gift-exemption bound (pre-registered 26 Oct 2026): gifts above the $300k allowance get no
    # exemption at all; both economies re-solved (paper Section 4.5).
    "base_giftnoex": {"gift_exempt": False},
    "off_giftnoex":  {"trusts": False, "gift_exempt": False},
    # Checks requested by the mock review panel (pre-registered 27 Oct 2026):
    "noshield_fix":  {"shield": False},                         # noshield rerun after the exemption
                                                                # double-count fix in layerC
    "lock0_partial": {"lock": 0.0, "payout_fracs": (0.1, 0.25, 0.5)},   # lock0 with partial withdrawals
    "base_capE":     {"fund_cap_E": True},                      # funding capped at the exemption
}
# Secant search settings (UNITS: annual rates).
#   R0 = 6%: first guess (the calibrated rate). R1_STEP = 0.5 pp: second guess is R0 -/+ 0.005.
#   FTOL = 0.002: stop when |excess| < 0.2%. XTOL = 2e-4: stop when the rate moves < 0.02 pp.
#   MAXIT = 10: at most 10 evaluations after the first (11 in all).
R0, R1_STEP = 0.06, 0.005
FTOL, XTOL, MAXIT = 2e-3, 2e-4, 10


# ==================================================================================================
# HELPERS
# ==================================================================================================

def scalars(m):
    """Make a moments dict JSON-friendly: single numbers become Python floats, arrays (e.g. mean
    wealth by age) become lists of floats; anything that cannot be converted is dropped.
    Input: dict name -> number or array. Output: a new dict. Also used by run_sweep.py."""
    out = {}
    for k, v in m.items():
        if np.ndim(v) == 0:                     # np.ndim = number of array dimensions; 0 = a single number
            out[k] = float(v)
        else:
            try:
                out[k] = list(map(float, v))
            except Exception:                    # e.g. nested or non-numeric values: skip them
                pass
    return out


# ==================================================================================================
# ONE EXPERIMENT: secant search on r, restart-safe
# ==================================================================================================

def run(name, bp, ef, su, log):
    """Run experiment `name` (a key of EXPERIMENTS) on the calibration in the module variable `cal`,
    re-solving the annual interest rate r so that the capital market clears.

    Inputs: name; bp, ef, su from calibrate_C.base_params(); log -- open text file for progress.
    Output: nothing returned; writes results/layerC5_<tag><name>.json with
      r_star, excess_at_r_star, converged (|excess| < FTOL), ge_moments (at r_star), pe_moments
      (at r = 6%), history (r, excess, generations, seconds of every evaluation), plus labels.
    NUMERICS: the same secant + bisection-safeguard search as equilibrium.solve_r, but every
      evaluation is cached on disk so an interrupted run resumes.
    APPROXIMATION: the warm-start distribution D is not saved in the cache, so after a restart the
      next evaluation starts from scratch; results agree within the convergence tolerances.
    PAPER: Section 3.6.
    """
    # Final result file; if it exists and is valid JSON the experiment is already done.
    final = os.path.join(RES, f"layerC5_{cal['tag']}{name}.json")
    if os.path.exists(final):
        try:
            json.load(open(final))
            print(f"# {name}: already done", file=log, flush=True)
            return
        except Exception:                        # incomplete file from an interrupted save: rebuild it
            pass
    # Cache of earlier evaluations of this experiment: a list of dicts, one per line of the file.
    # PYTHON NOTE: [json.loads(l) for l in open(cache)] is a list comprehension reading each line.
    cache = os.path.join(RES, f"layerC5_{cal['tag']}{name}_evals.jsonl")
    hist = [json.loads(l) for l in open(cache)] if os.path.exists(cache) else []
    # Parameters of this experiment: calibrated ParamsC, then the calibration's extra overrides (if
    # any, from run_sweep.py), then the experiment's overrides.
    p0 = replace(cc.to_params(cal["x"], bp), **cal.get("extra", {}), **EXPERIMENTS[name])
    D = None                                     # warm-start distribution, updated after each solve

    def evaluate(r):
        """Solve the model at annual rate r, append the result to the cache and the log, and
        return the row {"r", "excess", "gens", "secs", "m" (moments)}."""
        # PYTHON NOTE: `nonlocal D` lets this inner function update the variable D of run().
        nonlocal D
        t = time.time()
        p = replace(p0, KY=ky_for(p0, r))        # same parameters, corporate K/Y for this r
        # Solve; stop the generation fixed point at mass change < 1e-6 AND wealth-weighted change
        # < cc.WTOL (1e-4), at most 120 generations, warm-started from D.
        e = EconomyC(p, ef, su).solve().distribution(tol=1e-6, wtol=cc.WTOL, maxgen=120, D0=D)
        m = e.moments(); D = e.parent_D.copy(); gens = e.gens_to_converge
        del e; gc.collect()                      # free the large economy object before the next solve
        f = float(m["K_supply-Kb"] / m["Kc_demand"] - 1.0)       # excess(r)
        row = {"r": float(r), "excess": f, "gens": int(gens), "secs": float(time.time() - t), "m": scalars(m)}
        with open(cache, "a") as fh:             # append to the cache at once (restart safety)
            fh.write(json.dumps(row) + "\n")
        print(f"# {name}: r={r:.4%} excess={f:+.4f} gens={gens} {row['secs']:.0f}s "
              f"top1={m['top1']:.4f} top1_liq={m['top1_liquid_only']:.4f} top0.1={m['top0.1']:.4f}", file=log, flush=True)
        return row

    # Step 1: the first evaluation at r = 6% (skipped if the cache already has rows).
    if not hist:
        hist.append(evaluate(R0))
    # Step 2: if that does not clear and there is no second point yet, try 6% -/+ 0.5 pp
    # (lower r if excess supply, higher r if excess demand).
    if abs(hist[-1]["excess"]) >= FTOL and len(hist) < 2:
        f0 = hist[0]["excess"]
        hist.append(evaluate(R0 - R1_STEP if f0 > 0 else R0 + R1_STEP))
    # Step 3: secant steps through the last two points until the market clears, the rate stops
    # moving, or 11 evaluations in total have been made.
    while abs(hist[-1]["excess"]) >= FTOL and len(hist) < MAXIT + 1:
        (ra, fa), (rb, fb) = (hist[-2]["r"], hist[-2]["excess"]), (hist[-1]["r"], hist[-1]["excess"])
        if abs(rb - ra) < XTOL:
            break
        # Rates seen with positive and with negative excess (the bracket ends).
        pos = [h["r"] for h in hist if h["excess"] > 0]; neg = [h["r"] for h in hist if h["excess"] < 0]
        rn = rb - fb * (rb - ra) / (fb - fa)     # NUMERICS: secant step
        if pos and neg:                          # excess supply (>0) means r is too high
            hi, lo = min(pos), max(neg)
            if not (min(lo, hi) < rn < max(lo, hi)):
                rn = 0.5 * (lo + hi)             # safeguard: bisect inside the bracket
        hist.append(evaluate(float(np.clip(rn, 0.005, 0.12))))   # keep r in [0.5%, 12%]
    # The GE answer: the evaluation with the smallest |excess|.
    # PYTHON NOTE: min(list, key=f) returns the item for which f(item) is smallest; `lambda h: ...` is a
    # one-line unnamed function.
    best = min(hist, key=lambda h: abs(h["excess"]))
    # The PE answer: the evaluation at exactly r = 6% (else the first one).
    pe = next((h for h in hist if abs(h["r"] - R0) < 1e-12), hist[0])
    out = {"name": name, "overrides": EXPERIMENTS[name], "calibration": cal["desc"],
           "r_star": float(best["r"]), "excess_at_r_star": float(best["excess"]), "converged": bool(abs(best["excess"]) < FTOL),
           "ge_moments": best["m"], "pe_moments": pe["m"], "history": [{k: h[k] for k in ("r", "excess", "gens", "secs")} for h in hist]}
    # Write to a temporary file first, then rename it over the final name: the final file is never
    # left half-written if the program is interrupted.
    tmp = final + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, final)                       # write atomically
    print(f"# {name}: DONE r*={best['r']:.4%} excess={best['excess']:+.4f}", file=log, flush=True)


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():
    """Command line: py run_c5.py [cal=KEY] [experiment names...]   (see the file banner).
    e.g. `py run_c5.py cal=v2c base off noshield lock0`."""
    # PYTHON NOTE: `global cal` lets main() replace the module-level variable cal, which run() reads.
    global cal
    args = sys.argv[1:]                           # the command-line words after the script name
    if args and args[0].startswith("cal="):
        key = args.pop(0).split("=")[1]           # remove "cal=KEY" from the list and keep KEY
        if key not in CAL:                        # any other calibration: read its parameter file
            pf = os.path.join(RES, f"layerC4_{key}_params.json")
            CAL[key] = dict(x=np.array([json.load(open(pf))["params"][k] for k in cc.NAMES]),
                            tag=f"{key}_", desc=f"C4 {key} (from {os.path.basename(pf)})")
        cal = CAL[key]
        if key in ("v2b", "v2c", "v2c_theta1") or key.startswith("v2c_"):
            cc.use_v2b()                          # nd = 55 (targets are not used here)
    names = args or ["off", "lock0", "noshield", "base"]    # `a or b`: b when the list a is empty
    bp, ef, su = cc.base_params()
    log = open(os.path.join(RES, f"layerC5_{cal['tag']}log.txt"), "a")
    print(f"# start {time.ctime()} {names}", file=log, flush=True)
    for n in names:
        run(n, bp, ef, su, log)
    print(f"# all done {time.ctime()}", file=log, flush=True)


# Run main() only when launched directly; run_sweep.py and run_gridcheck.py import this file instead.
if __name__ == "__main__":
    main()
