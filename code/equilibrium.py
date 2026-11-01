"""
General-equilibrium re-solve for Layers B/C experiments.

The corporate sector sets r = alpha / KY - delta and the wage. During calibration KY is fixed at 3
(r = 6%) and beta is chosen so that the capital market clears. In experiments (switching
wealth-preservation technology off or on, policy changes), all preference and technology
parameters are held fixed and r is re-solved so that household wealth net of private-business
capital equals corporate capital demand:

    excess(r) = (K_supply(r) - K_b(r)) / K_c(r) - 1 = 0

Wages, earnings, pensions and the government budget (g is the residual) follow from r.
Dollar thresholds stay fixed in model units (mean_wealth_units is held at its calibrated value),
so the exemption does not move with r.

Each evaluation warm-starts the distribution from the previous one, which cuts the number of
generation iterations sharply. The root is found by the secant method on r, safeguarded by a
bracket once one is found.
"""
# ==================================================================================================
# FILE BANNER: equilibrium.py -- the link between r and K/Y, and a stand-alone r solver
# ==================================================================================================
# WHAT THIS FILE DOES
#   It holds two small tools for general equilibrium (paper Section 3.3):
#     ky_for(p, r)  the corporate capital-output ratio K/Y that goes with an annual interest rate r
#                   (the firm's first-order condition r + delta = alpha / (K/Y), solved for K/Y);
#     solve_r(...)  a secant search for the r at which the capital market clears (excess(r) = 0).
#   The model takes K/Y as its input (field KY), not r; ky_for is how a driver sets "r = 5.5%".
#
# HOW IT IS USED IN THE FINAL RESULTS
#   run_c5.py imports ONLY ky_for from here. It has its own copy of the secant search (run_c5.run),
#   which adds restart-safe caching and the wealth-weighted stopping rule. solve_r, evaluate and gc
#   are not called by any driver in this folder (run_c5, run_sweep, run_gridcheck, calibrate_C);
#   solve_r appears to be an earlier version of the same search, kept for reference.
#
# WHERE IT SITS
#   Layer A (life cycle) -> Layer B (fortunes; owns the price block _prices) -> Layer C (trusts) ->
#   this helper -> drivers (run_c5.py and, through it, run_sweep.py / run_gridcheck.py).
#
# HOW IT IS LAUNCHED / FILES / RUN TIME
#   It is not launched on its own (no __main__ block); it is imported. It reads and writes no files.
#   If solve_r were called: each of its evaluations is a full model solve (35-60 minutes with trusts,
#   about 2 minutes without), up to 1 + maxit = 11 evaluations.
#
# READING ORDER
#   ky_for -> evaluate -> solve_r.
#
# PAPER: Section 3.3 (firms, equilibrium); Section 3.6 (experiments with r re-solved).
#
# WORDS YOU'LL SEE
#   r          ANNUAL net interest rate (0.06 = 6% a year).
#   KY         corporate capital / corporate output (3 at r = 6%, with alpha = 0.36, delta = 0.06).
#   excess     (K_supply - K_b) / K_c - 1: household wealth not tied up in private business, relative
#              to what corporations want to borrow, minus 1. Positive = too much saving at this r,
#              so the clearing r is LOWER. Negative = too little, the clearing r is HIGHER.
#   secant     a root-finding rule: draw a straight line through the last two (r, excess) points and
#              jump to where it crosses zero.
#   bracket    two rates lo < hi with excess(lo) < 0 < excess(hi): the root must lie between them.
#   warm start D0: the parent distribution from the previous evaluation, used as the starting point.
# ==================================================================================================

# PYTHON NOTE: gc is Python's "garbage collector"; gc.collect() frees memory of deleted objects now.
import gc
import time
# PYTHON NOTE: replace(obj, field=value) copies a @dataclass object with some fields changed.
from dataclasses import replace
import numpy as np

from layerC import EconomyC


# ==================================================================================================
# PRICE LINK: interest rate -> capital-output ratio
# ==================================================================================================

def ky_for(p, r):
    """Corporate capital-output ratio K/Y implied by the annual interest rate r.

    The Cobb-Douglas firm rents capital until its marginal product equals the user cost:
    alpha * Y / K = r + delta, so K/Y = alpha / (r + delta).
    Inputs: p -- a parameter object with alpha (capital share) and delta (annual depreciation);
            r -- ANNUAL interest rate, as a share.
    Output: K/Y (a plain number; with alpha = 0.36, delta = 0.06, r = 0.06 it is 3.0).
    PAPER: Section 3.3 (firms).
    """
    return p.alpha / (r + p.delta)


# ==================================================================================================
# ONE EVALUATION AND THE SECANT SEARCH (not used by the final drivers; see the banner)
# ==================================================================================================

def evaluate(p, ef, su, r, D0=None, tol=1e-8, maxgen=80):
    """Solve the full model at interest rate r and report the capital-market excess.

    Inputs: p -- ParamsC; ef, su -- earnings-efficiency and survival tables by age; r -- annual rate;
            D0 -- warm-start distribution or None; tol -- mass-change tolerance for the generation
            fixed point; maxgen -- maximum number of generations.
    Output: (economy, moments dict, excess). Note it does NOT pass the wealth-weighted tolerance
            (wtol) that run_c5.py uses, so it stops on the mass test alone (with a tighter tol).
    """
    q = replace(p, KY=ky_for(p, r))                           # same parameters, K/Y set for this r
    e = EconomyC(q, ef, su).solve().distribution(tol=tol, maxgen=maxgen, D0=D0)
    m = e.moments()
    return e, m, m["K_supply-Kb"] / m["Kc_demand"] - 1.0      # excess(r)


def solve_r(p, ef, su, r0=0.06, r1=None, xtol=2e-4, ftol=2e-3, maxit=10, D0=None, tol=1e-8, verbose=True):
    """Return (economy or None, moments, r, history). Only one economy is kept in memory; the
    economy is returned when the last evaluation converged. r0: first guess; r1: second guess
    (default r0 -/+ 0.005 depending on the sign of the first excess).

    Plain-English summary: find the annual interest rate r at which excess(r) = 0, by the secant
    method with a bisection safeguard.
      Stop when |excess| < ftol (0.2%) or two successive guesses are within xtol (0.02 percentage
      points) of each other. Give up after maxit further evaluations and return the best point seen
      (economy = None in that case, because that economy was already deleted).
      history: a list of (r, excess, generations needed) for every evaluation, in order.
      UNITS: r and all tolerances on r are annual shares (0.005 = half a percentage point).
    NUMERICS: secant step r_new = r_b - f_b (r_b - r_a) / (f_b - f_a). Once rates with both signs of
      excess have been seen, a step that would leave that bracket is replaced by the bracket midpoint
      (bisection), and every guess is clipped to [0.5%, 12%].
    PAPER: Section 3.3 (market clearing), 3.6 (experiments).
    """
    hist = []
    t0 = time.time()
    # First evaluation at r0 (default 6%, the calibrated rate).
    e0, m0, f0 = evaluate(p, ef, su, r0, D0=D0, tol=tol)
    hist.append((r0, f0, e0.gens_to_converge))
    if verbose:
        print(f"  r={r0:.4%} excess={f0:+.4f} gens={e0.gens_to_converge} t={time.time()-t0:.0f}s", flush=True)
    if abs(f0) < ftol:                                      # already clears: done
        return e0, m0, r0, hist
    D = e0.parent_D.copy()                                  # keep its distribution as the next warm start
    del e0; gc.collect()                                    # one economy in memory at a time (~2-4 GB)
    if r1 is None:
        r1 = r0 - 0.005 if f0 > 0 else r0 + 0.005          # excess supply -> lower r
    best = (abs(f0), m0, r0)                                # (|excess|, moments, r) of the best point so far
    lo = hi = None                                          # bracket: f(lo) < 0 < f(hi) in r
    # Start the bracket with r0 on the side its sign says (the other end is still unknown).
    (lo, hi) = (None, r0) if f0 > 0 else (r0, None)
    ra, fa = r0, f0                                         # previous point (r_a, f_a)
    rb = r1                                                 # next rate to evaluate (r_b)
    for it in range(maxit):
        eb, mb, fb = evaluate(p, ef, su, rb, D0=D, tol=tol)
        hist.append((rb, fb, eb.gens_to_converge))
        if verbose:
            print(f"  r={rb:.4%} excess={fb:+.4f} gens={eb.gens_to_converge} t={time.time()-t0:.0f}s", flush=True)
        if abs(fb) < best[0]:
            best = (abs(fb), mb, rb)
        # Tighten the bracket: a positive excess is an upper end (hi), a negative one a lower end (lo).
        if fb > 0:
            hi = rb if hi is None else min(hi, rb)
        else:
            lo = rb if lo is None else max(lo, rb)
        # Converged: excess small enough, or the rate has stopped moving.
        if abs(fb) < ftol or abs(rb - ra) < xtol:
            return eb, mb, rb, hist
        D = eb.parent_D.copy()                              # warm start for the next evaluation
        del eb; gc.collect()
        rn = rb - fb * (rb - ra) / (fb - fa)                # secant step
        # PYTHON NOTE: `a < x < b` is a chained comparison, true when x lies strictly between a and b.
        if lo is not None and hi is not None and not (min(lo, hi) < rn < max(lo, hi)):
            rn = 0.5 * (lo + hi)                            # safeguard: bisect inside the bracket
        rn = float(np.clip(rn, 0.005, 0.12))                # keep r between 0.5% and 12% a year
        ra, fa, rb = rb, fb, rn                             # shift: the newest point becomes "previous"
    # Out of iterations: return the best point's moments and rate, without an economy.
    _, m, r = best
    return None, m, r, hist
