"""Layer B v2 calibration: entrepreneur premium capped at 1% (evidence: Moskowitz & Vissing-Jorgensen
2002; Kartashova 2014), superstar owner earnings in the high-growth state (Smith, Zidar & Zwick 2019),
and the top 0.1% wealth share and top 1% earnings share added as targets."""
# ==================================================================================================
# FILE BANNER: calibrate_B2.py -- calibration of Layer B (the "fortunes" layer, no trusts yet)
# ==================================================================================================
# WHAT THIS FILE DOES
#   It picks 10 Layer B parameters (patience, bequest and charity strength, and the entrepreneurial
#   return process) so that the Layer B model, which has no trusts, matches 10 data targets
#   (top wealth shares, entrepreneurs, private-business capital, estates, charity, 75+ wealth,
#   top earnings). It does this by weighted least squares: scipy's least_squares moves the
#   parameters until the sum of squared "standardised misses" (model - data) / se is as small as
#   it can make it.
#
# WHERE IT SITS IN THE MODEL
#   Layer A (life cycle: earnings, survival, savings) -> Layer B (adds entrepreneurial returns,
#   bequests, charity, estate tax: this file calibrates it) -> Layer C (adds dynasty trusts and
#   lifetime gifts; calibrated in calibrate_C.py). Driver scripts (run_c5.py, run_sweep.py, ...)
#   sit on top.
#
# WHAT THE FINAL MODEL USES FROM THIS FILE
#   ONLY `to_params` and `NAMES`. calibrate_C.py imports this file as `c2` and calls
#   `c2.to_params(...)` with the values stored in results/layerB_v2_params_bestof2.json, read in the
#   order given by `c2.NAMES`. That is how the Layer B return process (premia, risk, transitions,
#   superstar earnings) enters the final model. The `TARGETS`, `Objective` and `__main__` parts
#   below are the record of how those values were produced; they are not run by the final model.
#
# HOW IT IS LAUNCHED (author's PC)
#   `py calibrate_B2.py` -- main takes NO command-line arguments (sys.argv is never read).
#
# FILES
#   Reads:  data/eflife.dat (age-efficiency profile of earnings) and data/surlife.dat (survival
#           probabilities by age). It looks for them in <project>/data, else <project>/eflife.
#   Writes: results/layerB2_calibration_log.txt (appended: one line per model evaluation, then
#           "DONE ..."), and results/layerB_v2_params.json (the best point found).
#   NOTE: this script writes layerB_v2_params.json, but calibrate_C.py reads
#   layerB_v2_params_bestof2.json. That file is not written by any code here; the name suggests it
#   is the better of two runs of this calibration, saved by hand (unsure -- not visible in code).
#
# HOW LONG IT RUNS
#   One Layer B evaluation (no trusts) takes on the order of 2 minutes. The optimiser is allowed
#   up to 500 of its own evaluations (max_nfev=500), and scipy does not count the extra evaluations
#   it spends on finite-difference Jacobians (10 per Jacobian here), so a full run is many hours
#   and can be days.
#
# READING ORDER
#   PREM_L, TARGETS -> NAMES / LB / UB -> to_params -> Objective -> the __main__ block.
#
# PAPER: Section 3.1 (entrepreneurial returns, superstar earnings) and Section 3.4 (calibration;
#   the Layer B values feed the joint calibration).
#
# WORDS YOU'LL SEE
#   residual      (model moment - data target) / standard error; a miss measured in standard errors.
#   ss            sum of squared residuals; the number the optimiser tries to make small.
#   x             the vector of the 10 free parameters, in the order of NAMES.
#   warm start    starting the stationary-distribution iteration from an earlier solution (D0)
#                 instead of from scratch, to save time.
#   K_gap         capital-market excess: (household wealth - private-business capital) /
#                 corporate capital demand - 1. Zero means the market clears at r = 6%.
#   N / L / H     return states: normal, entrepreneur, high-growth entrepreneur (z = 0, 1, 2).
# ==================================================================================================

# PYTHON NOTE: `import a, b, c` loads several modules at once. `numpy` (as `np`) is the array library.
import sys, os, time, json, numpy as np
# PYTHON NOTE: `replace(obj, field=value)` makes a COPY of a @dataclass object with some fields
# changed; the original is untouched. Parameters are stored in such dataclasses (ParamsB).
from dataclasses import replace
# NUMERICS: least_squares minimises a sum of squared residuals. It starts at x0, estimates how each
# residual responds to each parameter (by nudging one parameter at a time = "finite differences"),
# and takes steps inside a "trust region" while respecting lower/upper bounds.
from scipy.optimize import least_squares
# Put this file's own folder at the front of Python's search path, so `layerB` is found next to it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# PYTHON NOTE: `from layerB import *` copies every public name of layerB.py into this file
# (ParamsB, EconomyB, and also whatever layerB itself imported, such as np).
from layerB import *

# ==================================================================================================
# CALIBRATION SET-UP: fixed premium, targets, free parameters and their bounds
# ==================================================================================================

# The annual expected return premium of an ordinary entrepreneur (state L) over r, FIXED at 1%
# (not estimated), following the private-equity-premium evidence cited in the docstring above.
# UNITS: annual, as a share (0.01 = 1 percentage point a year).
PREM_L = 0.01
# The calibration targets. PYTHON NOTE: a dict maps a name to a value; here each value is a tuple
# (data value, standard error used as the weight, source). A miss of one standard error costs 1
# in the sum of squares, so a SMALL standard error means the target is matched more tightly.
TARGETS = {
    "K_gap":                    (0.0,    0.01,  "capital market clears at r = 6%"),  # market clearing
    "wealth_75plus/mean":       (1.527,  0.05,  "SCF 2022"),     # mean wealth of 75+ / overall mean
    "charity/taxable_estates":  (0.13,   0.02,  "IRS SOI filing year 2021"),  # charity share of taxable estates
    "top0.1":                   (0.150,  0.01,  "Fed DFA Q2 2026 (FRED WFRBSTP1300)"),  # top 0.1% wealth share
    "top1":                     (0.325,  0.01,  "Fed DFA Q2 2026"),   # top 1% wealth share
    "top10":                    (0.689,  0.01,  "Fed DFA Q2 2026"),   # top 10% wealth share
    "share_entrepreneurs":      (0.20,   0.01,  "SCF 2022"),     # population share in states L or H
    "Kb/K":                     (0.305,  0.02,  "SCF 2022"),     # private-business capital / all wealth
    # Share of deaths that owe estate tax (0.14%), matched in LOGS because it is a tiny number:
    # a standard error of 0.15 in logs is roughly a +/-15% relative miss.
    "log_share_deaths_taxable": (np.log(0.0014), 0.15, "Tax Policy Center"),
    "top1_earnings_share":      (0.18,   0.01,  "SCF 2022 (Kuhn & Rios-Rull)"),  # top 1% earnings share
}
# The 10 free parameters, in the order they appear in the vector x:
#   beta_annual  annual discount factor (patience)
#   phi1         bequest-motive strength (negative; see the bequest formula in the paper)
#   chi          charity strength
#   premH        annual return premium of the high-growth state H (L's premium is PREM_L)
#   sigL, sigH   annual standard deviation of the log business return in L and in H
#   p_NL         probability per 5-year period of moving from N (normal) to L (entrepreneur)
#   p_LH         probability per 5-year period of moving from L to H
#   iota         probability that a child of an L/H parent starts life in L
#   multH        "superstar" earnings multiplier: earnings of an owner in state H are multH times
#                ordinary earnings (Smith, Zidar & Zwick 2019)
NAMES = ["beta_annual", "phi1", "chi", "premH", "sigL", "sigH", "p_NL", "p_LH", "iota", "multH"]
# Lower and upper bounds for each parameter, in the same order as NAMES (the optimiser stays inside).
LB = np.array([0.85, -80.0, 0.0, 0.0, 0.03, 0.10, 0.005, 0.005, 0.0, 1.0])
UB = np.array([0.99, -0.3, 60.0, 0.15, 0.80, 1.20, 0.30, 0.60, 0.95, 60.0])


# ==================================================================================================
# FROM PARAMETER VECTOR TO MODEL PARAMETERS  (this function IS used by the final model)
# ==================================================================================================

def to_params(x, base):
    """Turn the 10-number vector x (order: NAMES) into a full Layer B parameter set.

    Inputs:  x    -- array of 10 numbers, in the order of NAMES.
             base -- a ParamsB object holding every other parameter (taxes, life cycle, grids...).
    Output:  a NEW ParamsB, equal to `base` except for the fields set here.

    Used in the final model: calibrate_C.base_params() calls this with the values from
    results/layerB_v2_params_bestof2.json, so this function defines how the stored Layer B
    numbers map into model parameters.

    Mapping (state order everywhere is (N, L, H) = (normal, entrepreneur, high-growth)):
      prem      = (0, PREM_L, premH)     annual expected premium over r; N earns no premium.
      sig       = (0, sigL, sigH)        annual sd of the log business return; N has no extra risk.
      iota0     = p_NL                   a child of an N parent starts in L with the same probability
                                         as an adult N moves to L in one period (a modelling choice
                                         tied to p_NL, not a separate free parameter).
      earn_mult = (1, 1, multH)          only H owners get the superstar earnings multiplier.
    PAPER: Section 3.1 (entrepreneurial returns).
    """
    # PYTHON NOTE: "tuple unpacking": the 10 entries of x are given 10 names in one line.
    b, phi1, chi, premH, sL, sH, pNL, pLH, iota, multH = x
    # Copy `base`, overwriting only the calibrated fields (see the mapping in the docstring).
    return replace(base, beta_annual=b, phi1=phi1, chi=chi, prem=(0.0, PREM_L, premH), sig=(0.0, sL, sH),
                   p_NL=pNL, p_LH=pLH, iota=iota, iota0=pNL, earn_mult=(1.0, 1.0, multH))


# ==================================================================================================
# THE OBJECTIVE: one model evaluation -> vector of standardised misses
# (historical: used only when this file is run on its own)
# ==================================================================================================

class Objective:
    """The function least_squares calls. Each call solves Layer B at one parameter vector and
    returns the residuals, one per target: (model - data) / standard error.

    It also keeps a running record: how many evaluations were done (n), the best point so far
    (best = (sum of squares, x, moments)), and a warm-start distribution (D).
    PYTHON NOTE: a class bundles data (attributes such as self.n) with functions (methods).
    Defining `__call__` lets an Objective object be called like a function: obj(x).
    """

    def __init__(self, base, ef, su, log):
        """Store the inputs.
        base -- ParamsB with all non-calibrated parameters; ef -- earnings-efficiency profile by
        age (from eflife.dat); su -- survival probabilities by age (from surlife.dat); log -- an
        open text file where one line per evaluation is written."""
        self.base, self.ef, self.su, self.log = base, ef, su, log
        # D: warm-start distribution (None = start from scratch); n: evaluation counter;
        # best: (lowest sum of squares so far, its x, its moments). np.inf = "infinitely bad".
        self.D = None; self.n = 0; self.best = (np.inf, None, None)

    def __call__(self, x):
        """Evaluate the model at x (10 numbers, order NAMES). Returns an array of 10 residuals."""
        t = time.time()                                   # start time, to log how long this takes
        p = to_params(x, self.base)                       # full parameter set at this x
        # Solve households' problems (.solve()), then iterate to the stationary distribution
        # (.distribution) with a tight mass tolerance of 1e-11, warm-started from self.D if set.
        # PYTHON NOTE: methods can be chained: A().solve().distribution() works because each
        # method returns the economy object itself.
        e = EconomyB(p, self.ef, self.su).solve().distribution(tol=1e-11, D0=self.D)
        m = e.moments()                                   # dict: moment name -> model value
        # Keep the FIRST evaluation's distribution as the warm start for all later evaluations
        # (it is set once and never updated afterwards).
        if self.D is None:
            self.D = e.parent_D
        # Capital-market gap: household wealth net of private-business capital, relative to the
        # corporate sector's capital demand, minus 1. Zero = market clears at the assumed r = 6%.
        m["K_gap"] = m["K_supply-Kb"] / m["Kc_demand"] - 1
        # Log of the share of deaths owing estate tax; floored at 1e-8 so the log is defined at 0.
        m["log_share_deaths_taxable"] = np.log(max(m["share_deaths_taxable"], 1e-8))
        # Residuals in the order of TARGETS: (model - data) / standard error.
        # PYTHON NOTE: `[... for k, (v, s, _) in TARGETS.items()]` is a list comprehension: it loops
        # over the dict, unpacking each value tuple into v (data), s (se) and _ (source, ignored).
        r = np.array([(m[k] - v) / s for k, (v, s, _) in TARGETS.items()])
        # If anything came out as NaN or infinite, return a huge miss so the optimiser backs away.
        if not np.all(np.isfinite(r)):
            r = np.full(len(TARGETS), 1e3)
        # Count the evaluation and compute its sum of squared residuals.
        self.n += 1; ss = float((r ** 2).sum())
        # Remember the best point seen so far (x.copy() so later changes to x do not alter it).
        if ss < self.best[0]:
            self.best = (ss, x.copy(), m)
        # One log line: evaluation number, seconds, ss, the parameters (rounded) and key moments.
        print(f"[{self.n}] {time.time()-t:.1f}s ss={ss:.3f} x={np.round(x,4).tolist()} "
              + " ".join(f"{k}={m[k]:.4f}" for k in ["top0.1","top1","top10","share_entrepreneurs","Kb/K",
                         "share_deaths_taxable","charity/taxable_estates","wealth_75plus/mean","top1_earnings_share","K_gap",
                         "parent_child_wealth_elasticity"]), file=self.log, flush=True)
        return r


# ==================================================================================================
# MAIN: run the Layer B calibration (only when this file is launched directly)
# ==================================================================================================

# PYTHON NOTE: this block runs only when the file is launched directly (`py calibrate_B2.py`),
# NOT when another file imports it (as calibrate_C.py does). So importing it never starts a run.
if __name__ == "__main__":
    # Project root = the folder above code/ (results/ and data/ live there).
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Data folder: the first of <root>/data or <root>/eflife that contains eflife.dat.
    # PYTHON NOTE: next(... for ... if ...) returns the first item of a generator that passes the test.
    DATA = next(os.path.join(ROOT, d) for d in ["data", "eflife"] if os.path.exists(os.path.join(ROOT, d, "eflife.dat")))
    # ef: earnings efficiency by age; su: survival probabilities by age (plain-text number tables).
    ef, su = np.loadtxt(os.path.join(DATA, "eflife.dat")), np.loadtxt(os.path.join(DATA, "surlife.dat"))
    # Corporate capital demand per person at the default parameters (r = 6%), in model units.
    Kc = EconomyB(ParamsB(), ef, su).Kc
    # UNITS: dollars per model unit are set from model mean wealth. If the market clears and
    # private-business capital is 30.5% of wealth (the Kb/K target), then corporate capital is the
    # other 69.5%, so mean wealth = Kc / (1 - 0.305). That value is fixed for the whole run.
    base = ParamsB(mean_wealth_units=Kc / (1 - TARGETS["Kb/K"][0]))
    # Starting guess, order NAMES: beta 0.952, phi1 -2.6, chi 0.21, premH 9%, sigL 0.10, sigH 0.70,
    # p_NL 0.06, p_LH 0.05, iota 0.75, multH 15.
    x0 = np.array([0.952, -2.6, 0.21, 0.09, 0.10, 0.70, 0.06, 0.05, 0.75, 15.0])
    # Open the log in "append" mode ("a"): restarts add to the end instead of erasing it.
    log = open(os.path.join(ROOT, "results", "layerB2_calibration_log.txt"), "a")
    obj = Objective(base, ef, su, log)
    # NUMERICS: run the bounded least-squares search.
    #   x_scale=(UB-LB)/10  tells the optimiser the natural size of each parameter (a tenth of its
    #                       allowed range), so steps are comparable across very different units.
    #   diff_step=0.01      finite-difference nudge = 1% of each parameter's value.
    #   max_nfev=500        at most 500 objective evaluations (excluding Jacobian nudges).
    # The return value is ignored: the best point is taken from obj.best, which tracks every
    # evaluation (including ones made for Jacobians).
    least_squares(obj, x0, bounds=(LB, UB), x_scale=(UB - LB) / 10, diff_step=0.01, max_nfev=500)
    ss, xb, mb = obj.best                               # best sum of squares, its x, its moments
    # Save the best point as {name: value}, plus the fixed L premium and the sum of squares.
    # PYTHON NOTE: dict(zip(NAMES, values)) pairs names with values; map(float, xb) converts each
    # NumPy number to an ordinary Python float so json can write it.
    out = dict(zip(NAMES, map(float, xb))); out["premL_fixed"] = PREM_L; out["sum_sq"] = ss
    json.dump(out, open(os.path.join(ROOT, "results", "layerB_v2_params.json"), "w"), indent=1)
    print("DONE", out, file=log, flush=True)
