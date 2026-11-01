"""
Trust-share sweep on the C4 v2b calibration (pre-registered in DECISIONS.md, 18 Oct).

For each point, the founding-friction mean (fric_mu) and the discount factor (beta) are re-solved
so that (i) the trust share of top-1% wealth hits the point's target and (ii) the capital market
clears at r = 6% (K_gap = 0). All other parameters stay at the v2b calibration. Then the
economy with trusts (base) is compared with the economy without trusts (off, r re-solved),
using the same machinery as run_c5.py.

Points:
  s10, s15, s27   trust share of top-1% wealth = 10%, 15%, 27%  (v2b itself is s = 21.3%)
  steep           s = 21.3%, with the founding friction falling with wealth
                  (fric_slope = 0.5: hurdle ~ (W/$30M)^-0.5), to concentrate trust wealth at the top

Restart-safe: evaluations are cached in results/sweep_<name>_cal.jsonl; experiments use
run_c5's caches (results/layerC5_sweep_<name>_*).

Usage: python run_sweep.py [cal=v2b|v2c] [names...]   (default: s15 s10 s27 steep)

theta1, theta1_t2: theta check; ret05, ret10: trust-return sensitivity (both pre-registered in DECISIONS.md, 25 Oct 2026).
"""
# ==================================================================================================
# FILE BANNER: run_sweep.py -- trust-share sweep, theta checks and trust-return checks
# ==================================================================================================
# WHAT THIS FILE DOES
#   Each "point" is a variant of the calibration. For each point it (1) RE-CALIBRATES two parameters,
#   the founding-friction mean fric_mu (mu_f) and the annual discount factor beta, so that the trust
#   share of top-1% wealth equals the point's target AND the capital market clears at r = 6%
#   (a 2-equation, 2-unknown problem, solved by Broyden's method); then (2) runs the "base" (trusts)
#   and "off" (no trusts) experiments of run_c5.py with r re-solved, and logs the effect of trusts on
#   the top 1% and top 0.1% wealth shares.
#   Although the docstring says "v2b", the calibration actually used is the one named by cal= on the
#   command line (v2c for the final results); the variable X_V2B then holds the v2c vector.
#
# THE POINTS (dictionary POINTS)
#   s10, s15, s27   trust-share sweep: trust share of top-1% wealth = 10%, 15%, 27%.
#   steep           DISCARDED variant: s = 21.27% (the v2b value) with the founding friction falling
#                   with wealth (fric_slope = 0.5). Still in the default list of main(), but the
#                   .bat launchers name their points explicitly and leave it out; not reported.
#   theta1          pre-registered theta check: theta (trust_weight) fixed at 1, trust share held at
#                   the v2c value 20.58%, fric_mu allowed down to -0.05.
#   theta1_t2       same, with a restricted tier 2 switched on (trust-feeding planning tools: no tax
#                   advantage, no valuation discount, small legal/admin costs). Note: tier 2 is OFF in
#                   all other reported results; this check is the exception.
#   ret05, ret10    pre-registered trust-return checks: trust assets earn an extra 0.5% / 1.0% a year
#                   (trust_prem), trust share held at the v2c value.
#
# WHERE IT SITS
#   Layer A -> Layer B -> Layer C -> calibrate_C.py (parameters and helpers) -> run_c5.py (experiment
#   machinery) -> THIS DRIVER, which reuses run_c5.run for its base/off experiments.
#
# HOW IT IS LAUNCHED (author's PC, from .bat launchers)
#   `py run_sweep.py cal=v2c s15 s10 s27`        the sweep
#   `py run_sweep.py cal=v2c theta1_t2 theta1`   the theta checks
#   `py run_sweep.py cal=v2c ret05 ret10`        the trust-return checks
#   main(): if the FIRST argument starts with "cal=", the text after "=" (v2b or v2c) picks the
#   calibration from run_c5.CAL; default v2b. All remaining arguments are point names (keys of
#   POINTS), run in order; with none, the default is s15 s10 s27 steep. The trust grid is always
#   nd = 55 (cc.use_v2b()).
#
# FILES (results/ folder; <pre> is "" for v2b and "v2c_" for v2c; <name> is the point)
#   Reads:  data/eflife.dat, data/surlife.dat, results/layerB_v2_params_bestof2.json (via
#           calibrate_C.base_params), and the caches below if they exist.
#   Writes: results/sweep_<pre><name>_cal.jsonl          the re-calibration evaluations (cache)
#           results/sweep_<pre>log.txt                    progress log (appended), incl. "RESULT" lines
#           results/layerC5_sweep_<pre><name>_base_evals.jsonl, ..._base.json   base experiment
#           results/layerC5_sweep_<pre><name>_off_evals.jsonl,  ..._off.json    off experiment
#   (the last four through run_c5.run; the base cache is first seeded with the calibration's r = 6%
#   evaluation, so base often needs no new evaluation at all).
#
# HOW LONG IT RUNS
#   One evaluation with trusts takes 35-60 minutes; without trusts about 2 minutes. Per point: up to
#   8 re-calibration evaluations (10 for theta1, theta1_t2, ret05, ret10), then base (usually 0 to a
#   few more) and off (a few short ones): roughly 3 to 10 hours per point.
#
# READING ORDER
#   POINTS, tolerances, J0, S0 -> point_x -> calibrate_point (inner evaluate, then the Broyden loop)
#   -> main.
#
# PAPER: Section 3.6 (trust-share sweep, theta check, trust-return sensitivity); Section 3.4 (the
#   re-calibrated parameters mu_f and beta).
#
# WORDS YOU'LL SEE
#   s1          the trust share of top-1% wealth: trust assets d / total wealth (a + d), top 1% by a + d.
#   K_gap       capital-market excess at r = 6%: (K_supply - K_b) / K_c - 1.
#   z           the two unknowns being solved for: z = (fric_mu, beta).
#   f           the two equations' misses: f = (s1 - target, K_gap); the goal is f = (0, 0).
#   J           the 2x2 Jacobian: how f responds to z. Row 1 = s1, row 2 = K_gap; column 1 = fric_mu,
#               column 2 = beta.
#   Broyden     a Newton-like method that never recomputes J by finite differences (too expensive);
#               it starts from a guess J0 and corrects J after each step using the observed change.
#   xset        parameters among the 7 calibrated ones that a point fixes at a new value.
#   extra       further ParamsC overrides for a point (tier 2 settings, trust_prem).
# ==================================================================================================

# PYTHON NOTE: gc = garbage collector; json reads/writes .json and .jsonl (one JSON object per line).
import sys, os, json, time, gc
import numpy as np
# PYTHON NOTE: replace(obj, field=value) copies a @dataclass parameter object with fields changed.
from dataclasses import replace
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_C as cc
import run_c5 as r5
from layerC import EconomyC

# ==================================================================================================
# SETTINGS: the calibration in use, the points, tolerances, starting Jacobian
# ==================================================================================================

RES = cc.RES
# The 7-parameter calibration vector the points start from (order calibrate_C.NAMES). Despite the
# name, main() replaces it with the vector of the calibration chosen by cal= (v2c in final runs).
X_V2B = r5.CAL["v2b"]["x"].copy()
CALKEY = "v2b"                               # set by "cal=" on the command line
I_BETA, I_MU = 0, 3                          # positions in cc.NAMES
# Each point: target = trust share of top-1% wealth to hit; slope = fric_slope (0 = flat friction);
# optional: xset (fixed calibrated parameters), mu_lo (lower bound for fric_mu, default 0.0),
# maxev (max re-calibration evaluations, default MAXEV = 8), extra (other ParamsC overrides).
POINTS = {
    "s15":   dict(target=0.15, slope=0.0),
    "s10":   dict(target=0.10, slope=0.0),
    "s27":   dict(target=0.27, slope=0.0),
    # DISCARDED variant (not in the final results): friction mean falls with wealth W as
    # (W / $30M)^-0.5, at the v2b trust share 21.27%.
    "steep": dict(target=0.2127, slope=0.5),
    # theta check (pre-registered 25 Oct): founder's trust weight fixed at 1, trust share held at the v2c value
    "theta1":    dict(target=0.2058, slope=0.0, xset={"trust_weight": 1.0}, mu_lo=-0.05, maxev=10),
    # ... and with the trust-feeding planning tools on (tier 2 restricted to freeze transfers into the trust:
    # no tax-drag advantage, no valuation discount, legal/admin costs instead of a family office)
    # In parameters: tier 2 on; its tax rates equal the tier-1 ones (tau_a2 = 18% = tau_a,
    # tau_ng2 = 24% = tau_ng); valuation discount 0; set-up $100k; running cost $25k a year minimum and
    # 0% of assets (instead of the family-office defaults $500k, $1M, 0.5%).
    "theta1_t2": dict(target=0.2058, slope=0.0, xset={"trust_weight": 1.0}, mu_lo=-0.05, maxev=10,
                      extra=dict(tier2=True, tau_a2=0.18, tau_ng2=0.24, disc_v=0.0,
                                 setup2_usd=100e3, fo_min_usd=25e3, fo_rate=0.0)),
    # trust-return sensitivity (pre-registered 25 Oct): trust assets earn an extra 0.5% / 1% a year over
    # the family's own return; trust share held at the v2c value
    "ret05": dict(target=0.2058, slope=0.0, maxev=10, extra=dict(trust_prem=0.005)),
    "ret10": dict(target=0.2058, slope=0.0, maxev=10, extra=dict(trust_prem=0.010)),
}
# Stopping tolerances of the re-calibration: |s1 - target| < 0.5 percentage points (STOL) and
# |K_gap| < 0.2% (KTOL); MAXEV = default maximum number of evaluations per point.
STOL, KTOL, MAXEV = 0.005, 0.002, 8
# initial Jacobian of (s1, K_gap) w.r.t. (fric_mu, beta), from the v2b calibration evaluations 42/43/46
# Reading it: raising fric_mu by 0.01 lowers s1 by ~0.059 and K_gap by ~0.031; raising annual beta
# by 0.001 raises s1 by ~0.008 and K_gap by ~0.034.
# APPROXIMATION: the same v2b-based J0 is used for v2c points; Broyden updates correct it as it goes.
J0 = np.array([[-5.85, 7.89], [-3.12, 33.6]])
S0 = {"v2b": 0.2127, "v2c": 0.2060}          # trust share of top-1% wealth at each calibration
# NOTE: S0["v2c"] = 0.2060 while the theta/ret points target "the v2c value" as 0.2058 (a 0.02 pp
# difference, far inside STOL).


# ==================================================================================================
# THE PARAMETER VECTOR OF A POINT
# ==================================================================================================

def point_x(spec, mu, beta):
    """The 7-parameter vector (order calibrate_C.NAMES) for a point: the calibration vector with
    fric_mu = mu and beta_annual = beta, and any parameters listed in spec["xset"] overwritten
    (e.g. trust_weight = 1 for the theta checks). Returns a new array."""
    x = X_V2B.copy(); x[I_MU] = mu; x[I_BETA] = beta
    # PYTHON NOTE: spec.get("xset", {}) returns the "xset" entry, or an empty dict if there is none.
    for k, v in spec.get("xset", {}).items():
        x[cc.NAMES.index(k)] = v                 # NAMES.index(k) = position of parameter k
    return x


# ==================================================================================================
# RE-CALIBRATION OF (fric_mu, beta) FOR ONE POINT: Broyden's method, restart-safe
# ==================================================================================================

def calibrate_point(name, spec, bp, ef, su, log):
    """Find (fric_mu, beta) so that the trust share of top-1% wealth s1 = spec["target"] and the
    capital market clears at r = 6% (K_gap = 0).

    Inputs: name (key of POINTS), spec (its dict), bp/ef/su from calibrate_C.base_params(), log file.
    Output: the best evaluation row, a dict {"mu", "beta", "s1", "K_gap", "gens", "secs", "m"}, where
      "best" minimises |s1 - target| / STOL + |K_gap| / KTOL over ALL rows in the cache (so a point
      that did not fully converge still returns its closest evaluation).
    NUMERICS: Broyden's method in 2 dimensions, starting from the linear-model guess z0 + J0^-1 *
      (target - S0, 0) (flat-friction points only), with steps capped at +/-0.04 in fric_mu and
      +/-0.004 in annual beta, and fric_mu kept in [mu_lo, 0.5], beta in [0.90, 0.99].
    Restart safety: every evaluation is appended to results/sweep_<pre><name>_cal.jsonl; on a restart
      the same path of (mu, beta) guesses is recomputed from the replayed rows, so finished
      evaluations are not repeated (the warm start D is not restored, as in run_c5).
    """
    pre = "" if CALKEY == "v2b" else f"{CALKEY}_"
    cache = os.path.join(RES, f"sweep_{pre}{name}_cal.jsonl")
    rows = [json.loads(l) for l in open(cache)] if os.path.exists(cache) else []
    D = None                                     # warm-start distribution

    def evaluate(mu, beta):
        """Return the row for (mu, beta): from the cache if it was already computed, else solve the
        model at r = 6% (KY unchanged), append to the cache and log, and return the new row."""
        nonlocal D
        for rw in rows:                          # replay
            if abs(rw["mu"] - mu) < 1e-10 and abs(rw["beta"] - beta) < 1e-10:
                return rw
        x = point_x(spec, mu, beta)
        # Full ParamsC: calibrated values at x, the point's friction slope and its extra overrides.
        p = replace(cc.to_params(x, bp), fric_slope=spec["slope"], **spec.get("extra", {}))
        t = time.time()
        # Same convergence settings as run_c5 (mass < 1e-6, wealth-weighted < 1e-4, <= 120 generations).
        e = EconomyC(p, ef, su).solve().distribution(tol=1e-6, wtol=cc.WTOL, maxgen=120, D0=D)
        m = e.moments(); D = e.parent_D.copy(); gens = int(e.gens_to_converge)
        del e; gc.collect()                      # free memory before the next solve
        kg = float(m["K_supply-Kb"] / m["Kc_demand"] - 1.0)      # K_gap at r = 6%
        rw = {"mu": float(mu), "beta": float(beta), "s1": float(m["trust_share_of_top1_wealth"]), "K_gap": kg,
              "gens": gens, "secs": float(time.time() - t), "m": r5.scalars(m)}
        with open(cache, "a") as fh:
            fh.write(json.dumps(rw) + "\n")
        rows.append(rw)
        print(f"# {name}: mu={mu:.5f} beta={beta:.5f} s1={rw['s1']:.4f} s0.1={m['trust_share_of_top0.1_wealth']:.4f} "
              f"K_gap={kg:+.4f} top1={m['top1']:.4f} top1_liq={m['top1_liquid_only']:.4f} {rw['secs']:.0f}s", file=log, flush=True)
        return rw

    tgt = spec["target"]
    z = np.array([X_V2B[I_MU], X_V2B[I_BETA]])   # start at the calibration's (fric_mu, beta)
    J = J0.copy()
    if spec["slope"] == 0.0:                     # first guess from the linear model
        # Solve J dz = (change in s1 wanted, 0 change in K_gap) for dz and move there.
        # (Skipped for "steep", whose different friction shape makes J0 inapplicable.)
        z = z + np.linalg.solve(J, np.array([tgt - S0[CALKEY], 0.0]))
    hist = []                                    # (z, f) of every step, for the Broyden update
    for it in range(spec.get("maxev", MAXEV)):
        # Keep the guess inside bounds: fric_mu in [mu_lo, 0.5], annual beta in [0.90, 0.99].
        z[0] = float(np.clip(z[0], spec.get("mu_lo", 0.0), 0.5)); z[1] = float(np.clip(z[1], 0.90, 0.99))
        rw = evaluate(z[0], z[1])
        f = np.array([rw["s1"] - tgt, rw["K_gap"]])          # the two misses
        hist.append((z.copy(), f))
        if abs(f[0]) < STOL and abs(f[1]) < KTOL:            # both within tolerance: done
            break
        if len(hist) >= 2:                       # Broyden update
            # NUMERICS: "good" Broyden: change J as little as possible so that it explains the last
            # step exactly: J_new dz = df. Formula: J += (df - J dz) dz' / (dz' dz).
            # PYTHON NOTE: `@` is matrix multiplication; np.outer(u, v) is the matrix u v'.
            dz = hist[-1][0] - hist[-2][0]; df = hist[-1][1] - hist[-2][1]
            if np.dot(dz, dz) > 0:
                J = J + np.outer(df - J @ dz, dz) / np.dot(dz, dz)
        # Newton step with the current J: solve J step = -f.
        step = -np.linalg.solve(J, f)
        # Cap the step: at most 0.04 in fric_mu and 0.004 in annual beta per iteration.
        step[0] = np.clip(step[0], -0.04, 0.04); step[1] = np.clip(step[1], -0.004, 0.004)
        z = z + step
    # Best row across the whole cache, scoring each miss relative to its tolerance.
    best = min(rows, key=lambda w: abs(w["s1"] - tgt) / STOL + abs(w["K_gap"]) / KTOL)
    print(f"# {name}: calibrated mu={best['mu']:.5f} beta={best['beta']:.5f} s1={best['s1']:.4f} K_gap={best['K_gap']:+.4f}",
          file=log, flush=True)
    return best


# ==================================================================================================
# MAIN: for each point, re-calibrate, then run base and off with r re-solved
# ==================================================================================================

def main():
    """Command line: py run_sweep.py [cal=KEY] [point names...]   (see the file banner).
    e.g. `py run_sweep.py cal=v2c s15 s10 s27`."""
    # PYTHON NOTE: `global` lets main() replace these module-level variables, which the functions
    # above read.
    global X_V2B, CALKEY
    args = sys.argv[1:]
    if args and args[0].startswith("cal="):
        CALKEY = args.pop(0).split("=")[1]
        X_V2B = r5.CAL[CALKEY]["x"].copy()       # the chosen calibration's 7 parameters
    names = args or ["s15", "s10", "s27", "steep"]
    cc.use_v2b()                                 # nd = 55
    bp, ef, su = cc.base_params()
    pre = "" if CALKEY == "v2b" else f"{CALKEY}_"
    log = open(os.path.join(RES, f"sweep_{pre}log.txt"), "a")
    print(f"# start {time.ctime()} {names}", file=log, flush=True)
    for name in names:
        spec = POINTS[name]
        # Step 1: re-calibrate (fric_mu, beta) for this point.
        best = calibrate_point(name, spec, bp, ef, su, log)
        x = point_x(spec, best["mu"], best["beta"])
        # Step 2: point run_c5 at this point's parameters (its own file tag, friction slope and extra
        # overrides), so run_c5.run writes results/layerC5_sweep_<pre><name>_{base,off}.json.
        # NOTE: desc says "v2b sweep point" even when cal=v2c (a label only; the numbers use v2c).
        r5.cal = dict(x=x, tag=f"sweep_{pre}{name}_", extra={"fric_slope": spec["slope"], **spec.get("extra", {})},
                      desc=f"v2b sweep point {name}: s1 target {spec['target']}, fric_slope {spec['slope']}")
        # seed the base experiment with the calibration evaluation at r = 6% (no recomputation)
        # (the calibration's best row is exactly the base economy at r = 6%; if its |K_gap| is already
        # below run_c5.FTOL = 0.002, the base experiment finishes with no new evaluation)
        bcache = os.path.join(RES, f"layerC5_sweep_{pre}{name}_base_evals.jsonl")
        if not os.path.exists(bcache):
            with open(bcache, "w") as fh:
                fh.write(json.dumps({"r": 0.06, "excess": best["K_gap"], "gens": best["gens"], "secs": best["secs"],
                                     "m": best["m"]}) + "\n")
        # Step 3: base (trusts) and off (no trusts), each with r re-solved (run_c5's secant search).
        for exp in ("base", "off"):
            r5.run(exp, bp, ef, su, log)
        # Step 4: read both results and log the effect of trusts (base minus off, in percentage points)
        # on top 1% and top 0.1% wealth shares (total wealth a + d), plus the DFA-consistent top 1%,
        # the off economy's clearing rate, and tier-2 take-up (non-zero only for theta1_t2).
        b = json.load(open(os.path.join(RES, f"layerC5_sweep_{pre}{name}_base.json")))
        o = json.load(open(os.path.join(RES, f"layerC5_sweep_{pre}{name}_off.json")))
        bm, om = b["ge_moments"], o["ge_moments"]
        print(f"# {name}: RESULT s1={bm['trust_share_of_top1_wealth']:.4f} s0.1={bm['trust_share_of_top0.1_wealth']:.4f} "
              f"top1 {bm['top1']:.4f}->{om['top1']:.4f} (effect {100*(bm['top1']-om['top1']):+.2f} pp) "
              f"top0.1 {bm['top0.1']:.4f}->{om['top0.1']:.4f} (effect {100*(bm['top0.1']-om['top0.1']):+.2f} pp) "
              f"r_off={o['r_star']:.4%} top1_dfa {bm.get('top1_dfa', float('nan')):.4f}->{om.get('top1_dfa', float('nan')):.4f} "
              f"tier2_top1={bm.get('tier2_top1', 0.0):.3f} tier2_top0.1={bm.get('tier2_top0.1', 0.0):.3f}", file=log, flush=True)
    print(f"# all done {time.ctime()}", file=log, flush=True)


if __name__ == "__main__":
    main()
