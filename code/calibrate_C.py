"""
C4: joint calibration of the full model (Layers A-C, tier 1 + lifetime gifts).

Free parameters (7): beta, phi1 (bequest strength), chi (charity), founding friction mean and
spread, trust weight theta (founder's valuation of trust assets), gift warm-glow strength.
Held at Layer B v2 values: the return process (premia, risk, transitions, superstar earnings),
earnings process and all statutory/market parameters (taxes, exemption, fees, setup cost).

Targets (pre-registered in the spec; see data/targets_v2.csv):
  capital market clears at r = 6%; top 1%, 0.1%, 10% shares of PERSONAL (non-trust) wealth, which is
  the closest concept to the Fed DFA / SCF; trust share of top-1% wealth (CRS: 10-20%) and of top-0.1%
  wealth (Galle, Gamage & Lord: one-third or more); estate tax / gross estate for $50M+ estates net of
  spousal bequests (IRS 2024: 19.2%); share of deaths owing estate tax (TPC 0.14%); charity share of
  taxable estates; wealth 75+ / mean (SCF 1.53); inter vivos share of transfers (Sabelhaus 27%).
Validation (not targeted): parent-child persistence, trust take-up by wealth (SCF lower bounds),
Gini, bottom 50%, gift-tax returns.

Usage: python calibrate_C.py <label> [start=A|B] ; writes results/layerC4_<label>_*.
Runs in the fast configuration (nxi=5, ko=2, tolerance 1e-6) with warm starts.
"""
# ==================================================================================================
# FILE BANNER: calibrate_C.py -- the joint calibration of the full model (paper Section 3.4)
# ==================================================================================================
# WHAT THIS FILE DOES
#   It chooses the 7 free parameters of the full model (beta_annual, phi1, chi, fric_mu, fric_nu,
#   trust_weight = theta, gift_phi) so that the model with trusts and lifetime gifts matches 11
#   data targets as closely as possible, by weighted least squares (scipy least_squares). Every
#   other parameter is held fixed: the Layer B values come from results/layerB_v2_params_bestof2.json
#   (made by calibrate_B2.py), taxes/fees/exemption from the defaults in layerB/layerC.
#   The docstring above describes the original v2 targets; later configurations (v2b, v2c, v3)
#   change some targets -- see the configuration sections below.
#
# WHICH CONFIGURATION IS FINAL
#   v2c: the v2b set-up (corrected IRS estate-tax and charity targets, down-weighted 75+ wealth
#   target, trust grid nd = 55) with the three top-share targets matched on the DFA-consistent
#   measure of model wealth (layerC.dfa_top_shares). Its best point (evaluation 33) is stored as
#   CAL["v2c"] in run_c5.py and used by all experiments.
#   History: v2 -> v3 (longer life, medical costs; pre-registered, not the final one) -> v2b ->
#   v2c (final).
#
# WHERE IT SITS IN THE MODEL
#   Layer A (life cycle) -> Layer B (fortunes; values fixed from calibrate_B2.py) -> Layer C (trusts
#   and gifts: the layer whose parameters this file calibrates). The drivers run_c5.py, run_sweep.py
#   and run_gridcheck.py import this file for base_params(), to_params(), WTOL, RES, STARTS and the
#   use_v2b() switch; they do not run the optimiser.
#
# HOW IT IS LAUNCHED (author's PC, from a .bat launcher)
#   `py calibrate_C.py v2c start=V2B`
#   main() reads two positional arguments:
#     1st  the LABEL (default "run"). It names the output files, and its beginning picks the set-up:
#          starts with "v3" -> use_v3(); "v2b" -> use_v2b(); "v2c" -> use_v2c(); anything else
#          (e.g. "v2") -> the original v2 targets with nd = 40.
#     2nd  the starting point, written key=VALUE. Only the text after "=" is used (so "start=V2B"
#          and "x=V2B" do the same thing); it must be a key of STARTS: A, B, V2 or V2B.
#          Default A. (The docstring's "start=A|B" predates V2 and V2B.)
#
# FILES (results/ folder; <label> is the first argument)
#   Reads:  data/eflife.dat, data/surlife.dat (or data/surlife_ext99.dat for v3),
#           results/layerB_v2_params_bestof2.json (the fixed Layer B parameters),
#           results/layerC4_<label>_evals.jsonl if it exists (replay cache from an earlier run).
#   Writes: results/layerC4_<label>_log.txt (appended: one line per evaluation),
#           results/layerC4_<label>_evals.jsonl (appended: one JSON line per new evaluation),
#           results/layerC4_<label>_params.json (best parameters, moments and targets, at the end).
#
# HOW LONG IT RUNS
#   One model evaluation with trusts takes 35-60 minutes. least_squares is capped at 90 of its own
#   evaluations (max_nfev=90), and scipy does not count the 7 extra evaluations per finite-difference
#   Jacobian, so the log's counter goes well past 90 if the cap is reached. The v2b and v2c best points
#   came at evaluations 51 and 33. A run is therefore days of computing; the replay cache lets it
#   survive restarts.
#
# READING ORDER
#   TARGETS, NAMES, LB/UB, STARTS, FAST -> the v3, v2b, v2c configuration blocks and the use_*()
#   switches -> base_params -> to_params -> DIFF_STEP, WTOL -> class Objective -> main.
#
# PAPER: Section 3.4 (calibration: 7 parameters, 11 targets); Section 3.5 (DFA-consistent top shares,
#   used by v2c); Appendix A (convergence settings).
#
# WORDS YOU'LL SEE
#   residual     (model - target) / standard error, one per target; the optimiser minimises the sum
#                of their squares (ss).
#   floor        a one-sided target (v3 only): penalised only when the model falls below it.
#   liquid_only  "personal" wealth a only, excluding trust assets d.
#   _dfa         the DFA-consistent measure (personal wealth + trust wealth attributed as the Fed
#                DFA would; paper Section 3.5).
#   replay cache a file of past evaluations; a restarted run looks results up instead of recomputing.
#   warm start   starting the generation fixed point from the last converged distribution (D0).
#   FAST         the cheaper numerical settings used during calibration.
# ==================================================================================================

# PYTHON NOTE: standard library modules: sys (command-line arguments), os (file paths), json (read and
# write .json files), time (clock).
import sys, os, json, time
import numpy as np
# PYTHON NOTE: fields(SomeDataclass) lists the declared fields of a @dataclass (name, type, default).
from dataclasses import fields
# NUMERICS: bounded nonlinear least squares (trust-region method with finite-difference Jacobian).
from scipy.optimize import least_squares
# Make sure the modules next to this file (layerB, layerC, calibrate_B2) can be imported.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layerB import ParamsB, EconomyB
from layerC import ParamsC, EconomyC
# Only to_params and NAMES are used from calibrate_B2 (to load the stored Layer B values).
import calibrate_B2 as c2

# Folder layout: ROOT = project folder (one above code/); DATA = input tables; RES = results.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, RES = os.path.join(ROOT, "data"), os.path.join(ROOT, "results")

# ==================================================================================================
# TARGETS, FREE PARAMETERS, BOUNDS, STARTING POINTS (original v2 set-up)
# ==================================================================================================

# The 11 v2 targets: name -> (data value, standard error used as the weight, source).
# A miss of one standard error adds 1 to the sum of squares; small se = tight target.
# Names are the keys of the dict returned by EconomyC.moments() (two are added in Objective).
TARGETS = {
    # Capital market clears at r = 6%: (household wealth - private-business capital) / corporate demand - 1 = 0.
    "K_gap":                         (0.0,    0.01,  "capital market clears at r = 6%"),
    # Top 1% / 0.1% / 10% shares of PERSONAL wealth a (trust assets excluded), ranked by a.
    "top1_liquid_only":              (0.325,  0.01,  "Fed DFA Q2 2026"),
    "top0.1_liquid_only":            (0.150,  0.01,  "Fed DFA Q2 2026"),
    "top10_liquid_only":             (0.689,  0.015, "Fed DFA Q2 2026"),
    # Trust assets d as a share of the total wealth (a + d) of the top 1% / top 0.1% (ranked by a + d).
    "trust_share_of_top1_wealth":    (0.15,   0.05,  "CRS R48879 (2026) via CBO: 10-20%"),
    "trust_share_of_top0.1_wealth":  (0.40,   0.07,  "Galle, Gamage & Lord (2026): one-third or more"),
    # Estate tax paid / gross estate, estates of $50M and more (the v2 value 0.192; see v3 notes).
    "estate_tax/gross_50M+":         (0.192,  0.03,  "IRS SOI 2024, $50M+, net of spousal bequests"),
    # Log of the share of deaths owing estate tax (0.14%); se 0.15 in logs ~ +/-15% relative.
    "log_share_deaths_taxable":      (np.log(0.0014), 0.15, "Tax Policy Center"),
    # Charitable bequests / gross estate, among taxable estates.
    "charity/taxable_estates":       (0.15,   0.03,  "IRS SOI 2024"),
    # Mean wealth of households aged 75+ relative to overall mean wealth.
    "wealth_75plus/mean":            (1.527,  0.05,  "SCF 2022"),
    # Lifetime gifts / (lifetime gifts + bequests): the inter vivos share of transfers.
    "inter_vivos_share":             (0.27,   0.04,  "Sabelhaus (2024), SCF 2021"),
}
# The 7 free parameters, in the order of the vector x:
#   beta_annual   annual discount factor
#   phi1          bequest-motive strength (negative)
#   chi           charity strength
#   fric_mu       mean of the founding planning friction mu_f (share of resources)
#   fric_nu       scale of that logistic friction nu_f
#   trust_weight  theta: the testator's weight on trust assets d in the bequest motive
#   gift_phi      phi_g: strength of the lifetime-gift warm glow (negative, same form as phi1)
NAMES = ["beta_annual", "phi1", "chi", "fric_mu", "fric_nu", "trust_weight", "gift_phi"]
# Bounds, same order as NAMES (fric_mu may be slightly negative: a small subsidy to founding).
LB = np.array([0.90, -40.0, 0.0, -0.05, 0.005, 0.05, -40.0])
UB = np.array([0.99, -0.3, 30.0, 0.50, 0.30, 2.0, -0.05])
# Starting points, chosen on the command line by key (start=KEY). Order NAMES.
STARTS = {
    "A": np.array([0.948, -2.8, 0.25, 0.05, 0.03, 0.6, -2.0]),     # first hand-picked start
    "B": np.array([0.945, -5.0, 0.40, 0.10, 0.05, 0.4, -1.0]),     # alternative start (stronger bequests)
    # v2 best point (evaluation 42, ss 60.2): starting point for v3
    "V2": np.array([0.9449, -2.2168, 0.2809, 0.0756, 0.0389, 0.7997, -0.8893]),
    # v2b best point (evaluation 51, ss 13.56): starting point for v2c
    "V2B": np.array([0.9454397485946558, -1.8738656265147329, 0.20514187576522447, 0.07986465007338328,
                     0.03203010498655948, 0.9377722776835895, -1.9272035892980588]),
    # theta = 1 full recalibration (pre-registered 27 Oct 2026): v2c point with theta = 1 and the beta and
    # friction mean from the theta1 sweep point (trust share 20.6%, market clears at 6%)
    "V2C_TH1": np.array([0.9467341505921474, -1.5494726762187743, 0.1493978828816961, 0.06475896783001926,
                         0.02647145719235874, 1.0, -1.7931406611023284]),
}
# Parameters held fixed (not estimated) for labels ending in these suffixes: {index in NAMES: value}.
# Index 5 is trust_weight (theta), so label "v2c_theta1" estimates the other six with theta = 1.
FIXED_BY_SUFFIX = {"_theta1": {5: 1.0}}
# theta free again, restarted from the best theta = 1 point (evaluation 31 of v2c_theta1; pre-registered
# 29 Oct 2026): where does theta settle when it can move? Labels v2c_free2 and v2c_free3 start here;
# v2c_free3 (best point theta = 1.04, sum of squares 16.39) is the MAIN calibration in the paper.
STARTS["V2C_FREE2"] = np.array([0.9477677872574138, -1.2018958031220848, 0.12564306137497921, 0.051322913753263324,
                                0.022967153582720102, 1.0, -1.7876476773702106])
# Automatic stop (labels starting with these prefixes): stop once the best sum of squares has improved by
# less than STOP_TOL over the last STOP_N evaluations (the rule applied by hand to v2c and v2c_theta1).
# 30 Oct: STOP_N raised from 8 to 16. With 7 free parameters a finite-difference Jacobian takes 7 evaluations,
# so a window of 8 held only one trial step and stopped v2c_free2 after its first (rejected) step. The rerun
# with 16 is v2c_free3 (it replayed v2c_free2's cached evaluations, so the path is the same up to that point).
STOP_PREFIXES, STOP_N, STOP_TOL = ("v2c_free",), 16, 0.1


class StopCalibration(Exception):
    """Raised by the stop rule to end least_squares early; main() then saves the best point."""
    pass
# NUMERICS: the "fast" numerical configuration used during calibration (and, via to_params, in all
# the experiments):
#   nxi=5    Gauss-Hermite nodes for the return shock xi (Layer B default is 7)
#   ko=2     number of best options stored per state for the distribution (layerC default 3)
#   nd=40    points on the trust-asset grid d (v2b/v2c raise this to 55; run_gridcheck to 70)
#   gifts    lifetime gifts switched on
# PYTHON NOTE: FAST is a dict, and dicts can be changed in place (FAST["nd"] = 55); every file that
# imported this module sees the change, because they all share the same dict object.
FAST = dict(nxi=5, ko=2, nd=40, gifts=True)

# ==================================================================================================
# CONFIGURATION v3 (pre-registered; NOT the final calibration)
# ==================================================================================================

# ---- v3 configuration (pre-registered in DECISIONS.md, 13 Oct, before any v3 run) -------------
# Life extended to age 94: model entry at 25 instead of 20 (T = 14 periods, 25-94; retirement at 65,
# TL = 8), with SSA 2010 period-life-table mortality for ages 85-89 (data/surlife_ext99.dat). Extending
# to 99 with T = 16 is not possible: the distribution tracks two overlapping generations (T <= 2*gap).
# Late-life medical costs with a means-tested floor (ParamsC.med_on).
# Target corrections, measurement only:
#  - estate tax / (gross estate - bequests to surviving spouse), taxable returns over $50M:
#    IRS SOI 2024 Table 1 gives 15.85B / (82.35B - 15.75B) = 0.238. (v2 used 0.192, which is tax /
#    gross estate including spousal bequests; the model has no spouse, so 0.238 is the comparable
#    figure; the v2 label "net of spousal bequests" was wrong.)
#  - charity / (gross - spousal bequests), all taxable returns: 0.181 (v2: 0.15).
#  - trust shares of top wealth become one-sided floors (measured trust wealth is a lower bound):
#    top 1% >= 0.10 (CRS low end), top 0.1% >= 1/3 (Galle, Gamage & Lord: "one-third or more").
# Life-cycle settings for v3 (copied into CFG by use_v3()).
V3 = dict(T=14, age0=25, TL=8, surv="surlife_ext99.dat", med_on=True)
# v3 targets: a copy of the v2 targets with four entries replaced (see the notes above).
# PYTHON NOTE: dict(TARGETS) makes a new, independent copy; changing it leaves TARGETS alone.
TARGETS_V3 = dict(TARGETS)
TARGETS_V3["estate_tax/gross_50M+"] = (0.238, 0.03, "IRS SOI 2024 Table 1, taxable $50M+, tax / (gross - spousal bequests)")
TARGETS_V3["charity/taxable_estates"] = (0.181, 0.03, "IRS SOI 2024 Table 1, taxable returns, charity / (gross - spousal bequests)")
TARGETS_V3["trust_share_of_top1_wealth"] = (0.10, 0.05, "floor: CRS R48879 low end")
TARGETS_V3["trust_share_of_top0.1_wealth"] = (1 / 3, 0.07, "floor: Galle, Gamage & Lord one-third or more")
FLOORS = set()                       # targets penalised only when the model is below them (v3)
# CFG: the life-cycle set-up in force. Default = the v2/v2b/v2c one: entry at age 20, T = 14 periods
# (ages 20-89), retirement after TL = 9 periods (age 65), survival from surlife.dat, no medical costs.
CFG = dict(T=14, age0=20, TL=9, surv="surlife.dat", med_on=False)


# ==================================================================================================
# CONFIGURATION v2b
# ==================================================================================================

# ---- v2b configuration (pre-registered 15 Oct, after the independent review, before any v2b run) ----
# v2 model structure (ages 20-89, no medical costs) with: two-sided trust-share targets at the
# outside-evidence centre (unchanged from v2: top 1% 0.15 +/- 0.05, top 0.1% 0.40 +/- 0.07); the 75+
# wealth ratio down-weighted (se 0.05 -> 0.15) because the model cannot reach it structurally and it
# pushes trust wealth up; the corrected IRS estate-tax and charity targets (as in v3); trust grid
# nd = 55 (the 7 Oct decision that v2 did not implement).
TARGETS_V2B = dict(TARGETS)
TARGETS_V2B["wealth_75plus/mean"] = (1.527, 0.15, "SCF 2022 (down-weighted: structural limitation)")
TARGETS_V2B["estate_tax/gross_50M+"] = TARGETS_V3["estate_tax/gross_50M+"]       # 0.238
TARGETS_V2B["charity/taxable_estates"] = TARGETS_V3["charity/taxable_estates"]   # 0.181


# ==================================================================================================
# CONFIGURATION v2c  (THE FINAL CALIBRATION)
# ==================================================================================================

# ---- v2c configuration (pre-registered 20 Oct, before any v2c run): the final calibration ----
# As v2b, but the three top-share targets are matched on the DFA-consistent measure of model wealth
# (layerC.dfa_top_shares: personal wealth + SCF-recorded trusts + the rest of trust wealth allocated
# by asset class, top 1% receiving 50.9%) instead of personal wealth only. Data targets unchanged
# (DFA Q2 2026: 32.5 / 15.0 / 68.9%).
# PYTHON NOTE: a "dict comprehension" {new_key: v for k, v in old.items()} builds a new dict. Here it
# copies TARGETS_V2B, renaming three keys (top1_liquid_only -> top1_dfa, top0.1_liquid_only ->
# top0.1_dfa, top10_liquid_only -> top10_dfa) and keeping their values; all other keys unchanged.
# The order of targets is preserved, so the residual vector has the same layout as in v2b.
TARGETS_V2C = {("top1_dfa" if k == "top1_liquid_only" else "top0.1_dfa" if k == "top0.1_liquid_only"
                else "top10_dfa" if k == "top10_liquid_only" else k): v for k, v in TARGETS_V2B.items()}


# ==================================================================================================
# SWITCHES BETWEEN CONFIGURATIONS (called by main() from the label, and by the drivers)
# ==================================================================================================
# PYTHON NOTE: `global TARGETS` lets a function replace the module-level variable TARGETS. Without it,
# the assignment would create a new local variable that disappears when the function ends.

def use_v2c():
    """Switch to the FINAL v2c set-up: v2c targets (DFA-consistent top shares) and trust grid nd = 55.
    The life cycle (CFG) stays at the default ages 20-89."""
    global TARGETS
    TARGETS = TARGETS_V2C
    FAST["nd"] = 55


def use_v2b():
    """Switch to the v2b set-up: v2b targets and trust grid nd = 55.
    The drivers (run_c5, run_sweep, run_gridcheck) call this even for v2c runs: they need only the
    nd = 55 grid, which v2b and v2c share; they never use the targets."""
    global TARGETS
    TARGETS = TARGETS_V2B
    FAST["nd"] = 55


def use_v3():
    """Switch to the v3 set-up: v3 targets, the two trust-share targets as one-sided floors, and
    the extended life cycle (entry at 25, retirement TL = 8, extended survival table, medical costs).
    nd stays at 40 (FAST is not changed here)."""
    global TARGETS, FLOORS
    TARGETS = TARGETS_V3
    FLOORS = {"trust_share_of_top1_wealth", "trust_share_of_top0.1_wealth"}
    CFG.update(V3)                    # overwrite the life-cycle entries of CFG with the v3 ones


# ==================================================================================================
# BUILDING PARAMETER SETS
# ==================================================================================================

def base_params():
    """Build the fixed part of the parameters: everything except the 7 free ones.

    Steps: (1) load the age profiles; (2) fix the dollar scale from corporate capital demand;
    (3) apply the stored Layer B calibration (results/layerB_v2_params_bestof2.json) through
    calibrate_B2.to_params; (4) convert the result to a plain dict and set nxi to the FAST value.

    Returns (bp, ef, su):
      bp -- dict, field name -> value, for every ParamsB field (Layer B calibrated values included);
      ef -- array, earnings efficiency by age (data/eflife.dat);
      su -- array, survival probabilities by age (CFG["surv"]: surlife.dat, or surlife_ext99.dat in v3).
    Used by this file and by every driver (run_c5, run_sweep, run_gridcheck).
    """
    # Load the two age tables (plain text files of numbers).
    ef = np.loadtxt(os.path.join(DATA, "eflife.dat")); su = np.loadtxt(os.path.join(DATA, CFG["surv"]))
    # The life-cycle settings in force (number of periods, entry age, retirement period).
    life = dict(T=CFG["T"], age0=CFG["age0"], TL=CFG["TL"])
    # Corporate capital demand per person at r = 6% (KY = 3), in model units.
    # PYTHON NOTE: **life unpacks the dict into keyword arguments: ParamsB(T=14, age0=20, TL=9).
    Kc = EconomyB(ParamsB(**life), ef, su).Kc
    # UNITS: the dollar scale. With private-business capital at 30.5% of wealth (Layer B Kb/K
    # target) and the market cleared, mean wealth = Kc / (1 - 0.305) model units; layerB sets
    # `usd` (dollars per model unit) so that this equals SCF mean net worth. It is held fixed from
    # here on (also in experiments where r changes), so dollar thresholds do not move.
    base = ParamsB(mean_wealth_units=Kc / (1 - 0.305), **life)
    # The stored Layer B calibration (a dict name -> value).
    par = json.load(open(os.path.join(RES, "layerB_v2_params_bestof2.json")))
    # Put the stored values in calibrate_B2's order and map them into a ParamsB (premia, risk,
    # transitions, superstar earnings, and also Layer B's beta, phi1, chi, which are overwritten
    # later by the free parameters in to_params).
    pB = c2.to_params(np.array([par[k] for k in c2.NAMES]), base)
    # Convert the ParamsB object to a plain dict so ParamsC can be built from it plus extra fields.
    bp = {f.name: getattr(pB, f.name) for f in fields(ParamsB)}
    bp["nxi"] = FAST["nxi"]                          # fast setting: 5 return-shock nodes
    return bp, ef, su


def to_params(x, bp):
    """Turn the 7-number vector x (order NAMES) plus the fixed dict bp into a full ParamsC.

    beta_annual, phi1 and chi are Layer B fields, so they overwrite the Layer B values in a copy
    of bp; the other four (fric_mu, fric_nu, trust_weight, gift_phi) are Layer C fields and are
    passed directly. The numerical settings come from FAST (ko, nd, gifts) and the medical-cost
    switch from CFG. Every other Layer C field keeps its default (trusts on, tier 2 off, lock = 1,
    shield on, ...). Output: a ParamsC. PAPER: Section 3.4 (the 7 calibrated parameters).
    """
    # {name: value} for the 7 free parameters, as plain floats.
    kw = dict(zip(NAMES, map(float, x)))
    # Copy bp (so the shared dict is not changed) and move the three Layer B parameters into it.
    # PYTHON NOTE: kw.pop("key") returns the value AND removes the key, so kw keeps only the 4
    # Layer C parameters.
    b = dict(bp); b["beta_annual"] = kw.pop("beta_annual"); b["phi1"] = kw.pop("phi1"); b["chi"] = kw.pop("chi")
    return ParamsC(**b, ko=FAST["ko"], nd=FAST["nd"], gifts=FAST["gifts"], med_on=CFG["med_on"], **kw)


# ==================================================================================================
# OPTIMISER SETTINGS
# ==================================================================================================

# Finite-difference steps, relative to |x| (v2). v1 used 0.02 for every parameter: for beta that is
# about 0.019 in the annual discount factor, which moved the objective by two orders of magnitude.
# NUMERICS: to estimate how each target responds to a parameter, least_squares nudges that parameter
# by DIFF_STEP x its value and re-solves the model. Order NAMES: beta 0.1%, phi1 3%, chi 5%,
# fric_mu 5%, fric_nu 5%, trust_weight 3%, gift_phi 3%.
DIFF_STEP = np.array([0.001, 0.03, 0.05, 0.05, 0.05, 0.03, 0.03])
# Wealth-weighted convergence of the generation fixed point (v2). v1 stopped on the L1 mass change
# alone (1e-6) after ~14 generations while trust wealth was still accumulating (about 0.9 per
# generation), so results depended on the starting distribution.
# NUMERICS: the distribution is iterated generation after generation until it stops changing. It now
# stops only when BOTH the mass change < 1e-6 (tol) AND the wealth-weighted change < 1e-4 (WTOL): the
# second test sees changes among the very rich, whose mass is tiny but whose wealth is large.
# Also used by run_c5.py and run_sweep.py.
WTOL = 1e-4


# ==================================================================================================
# THE OBJECTIVE (one model evaluation -> 11 residuals), with a restart-safe replay cache
# ==================================================================================================

class Objective:
    """The function least_squares calls: obj(x) solves the full model at x and returns the residual
    vector, one entry per target in TARGETS: (model - target) / se (for a floor target, only the
    shortfall counts).

    Bookkeeping it keeps:
      n      evaluation counter (replayed and failed evaluations are counted too); "evaluation 33" in
             the CAL descriptions refers to this counter;
      best   (lowest sum of squares, its x, its moments as plain floats/lists);
      D      warm-start distribution: the parent distribution of the most recent converged evaluation;
      memo   the REPLAY CACHE in memory: rounded x -> (residuals, moments or None);
      cache  path of the .jsonl file that stores every new evaluation on disk.

    Why a replay cache: an evaluation takes 35-60 minutes and runs last days. least_squares is
    deterministic, so after a restart (a crash, a reboot, a container restart) it asks for the same
    x values again in the same order; those are answered from the file in no time, and real work
    resumes where it stopped.
    APPROXIMATION: replayed evaluations do not restore the warm-start distribution D, so the first new
    evaluation after a restart starts from scratch (D = None); its result can differ from an
    uninterrupted run within the convergence tolerances.
    """

    def __init__(self, bp, ef, su, log, cache, tol=1e-6):
        """bp, ef, su: from base_params(). log: open text file for progress lines.
        cache: path of the .jsonl replay file. tol: mass-change tolerance for the distribution."""
        self.bp, self.ef, self.su, self.log, self.tol, self.cache = bp, ef, su, log, tol, cache
        self.D = None; self.n = 0; self.best = (np.inf, None, None)
        self.memo = {}                                  # replay cache: x -> (r, m); survives container restarts
        # Load earlier evaluations: one JSON object per line, {"x": [...], "r": [...], "ss": ..., "m": ...}.
        # "m" holds the moments only for lines that were the best point when written (else null/None).
        if os.path.exists(cache):
            for l in open(cache):
                if l.strip():                           # skip empty lines
                    row = json.loads(l)
                    self.memo[self.key(row["x"])] = (np.array(row["r"]), row.get("m"))

    # PYTHON NOTE: @staticmethod marks a method that does not use `self`; it is a plain helper function
    # stored inside the class, called as self.key(x) or Objective.key(x).
    @staticmethod
    def key(x):
        """Dictionary key for a parameter vector: its 7 values rounded to 10 decimals, as a tuple
        (arrays cannot be dict keys; tuples can). Rounding makes tiny float noise irrelevant."""
        return tuple(np.round(np.asarray(x, float), 10).tolist())

    def __call__(self, x):
        """One evaluation at x (7 numbers, order NAMES). Returns an array of len(TARGETS) residuals."""
        t = time.time()
        k = self.key(x)
        if k in self.memo:                              # replayed evaluation: no recomputation
            r, m = self.memo[k]
            self.n += 1; ss = float((r ** 2).sum())
            if ss < self.best[0]:
                # m may be None here if this line was not the best when it was written (see __init__).
                self.best = (ss, x.copy(), m)
            print(f"[{self.n}] cached ss={ss:.3f}", file=self.log, flush=True)
            return r
        p = to_params(x, self.bp)                       # full ParamsC at this x
        try:
            # Solve households' problems, then iterate the generation fixed point to the stationary
            # distribution: stop when mass change < tol AND wealth-weighted change < WTOL, at most
            # 120 generations, warm-started from the last converged distribution (self.D).
            e = EconomyC(p, self.ef, self.su).solve().distribution(tol=self.tol, wtol=WTOL, maxgen=120, D0=self.D)
            m = e.moments()                             # dict: moment name -> model value
            self.D = e.parent_D.copy()                 # warm start from the latest converged point
            del e                                       # free the (large, ~GB) economy object now
        except Exception as err:                       # keep the optimiser alive; report the failure
            # PYTHON NOTE: try/except catches an error instead of letting it stop the program;
            # {err!r} prints the error's technical description.
            print(f"[{self.n}] FAILED x={np.round(x,4).tolist()} {err!r}", file=self.log, flush=True)
            self.n += 1
            # A huge miss on every target pushes the optimiser away; failures are NOT cached, so a
            # restart would try this point again.
            return np.full(len(TARGETS), 1e3)
        # Two moments built here from the model output (see TARGETS).
        m["K_gap"] = m["K_supply-Kb"] / m["Kc_demand"] - 1
        m["log_share_deaths_taxable"] = np.log(max(m["share_deaths_taxable"], 1e-8))
        # Residuals in the order of TARGETS. For a FLOOR target (v3 only), min(..., 0) keeps only a
        # shortfall: being above the floor costs nothing. Otherwise the ordinary (model - v) / s.
        r = np.array([min((m[k] - v) / s, 0.0) if k in FLOORS else (m[k] - v) / s for k, (v, s, _) in TARGETS.items()])
        if not np.all(np.isfinite(r)):                  # NaN/inf -> treat as a failure
            r = np.full(len(TARGETS), 1e3)
        self.n += 1; ss = float((r ** 2).sum())
        if ss < self.best[0]:
            # Store the moments as plain Python numbers (scalars) or lists (per-age arrays) so they can
            # be written to JSON. np.ndim(v) == 0 means v is a single number.
            self.best = (ss, x.copy(), {k: (float(v) if np.ndim(v) == 0 else list(map(float, v))) for k, v in m.items()})
        # The moments printed on each log line.
        show = ["top1_dfa", "top0.1_dfa", "top10_dfa", "top1_liquid_only", "top0.1_liquid_only", "top10_liquid_only", "trust_share_of_top1_wealth",
                "trust_share_of_top0.1_wealth", "estate_tax/gross_50M+", "share_deaths_taxable", "charity/taxable_estates",
                "wealth_75plus/mean", "inter_vivos_share", "K_gap", "top1", "parent_child_wealth_elasticity",
                "rank_rank_child_35_39", "takeup_top1"]
        # Append this evaluation to the replay file. The moments are saved only if this point is the
        # new best (self.best[0] == ss), which keeps the file small but lets a restart recover the
        # best point's moments.
        with open(self.cache, "a") as f:                # survives restarts: resume from the best point
            f.write(json.dumps({"x": list(map(float, x)), "r": list(map(float, r)), "ss": ss, "m": self.best[2] if self.best[0] == ss else None}) + "\n")
        self.memo[k] = (r, None)                        # also remember it for this run (moments not kept)
        # Progress line: evaluation number, seconds taken, ss, x (rounded), selected moments.
        print(f"[{self.n}] {time.time()-t:.0f}s ss={ss:.3f} x={np.round(x,4).tolist()} "
              + " ".join(f"{k}={m[k]:.4f}" for k in show), file=self.log, flush=True)
        return r


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():
    """Run one calibration. Command line (see the file banner):  py calibrate_C.py <label> [key=START]
    e.g. `py calibrate_C.py v2c start=V2B`. Writes results/layerC4_<label>_{log.txt, evals.jsonl,
    params.json}."""
    # 1st argument: the label (default "run"). 2nd argument: anything of the form key=VALUE; only
    # VALUE is used, as a key of STARTS (default "A").
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    start = sys.argv[2].split("=")[1] if len(sys.argv) > 2 else "A"
    # The label's beginning picks the configuration. Labels such as "v2" match none of these and run
    # the original v2 set-up (v2 targets, nd = 40).
    if label.startswith("v3"):
        use_v3()
    elif label.startswith("v2b"):
        use_v2b()
    elif label.startswith("v2c"):
        use_v2c()
    bp, ef, su = base_params()
    # Progress log, appended to on every run with this label.
    log = open(os.path.join(RES, f"layerC4_{label}_log.txt"), "a")
    print(f"# start {start} {time.ctime()} targets={list(TARGETS)}", file=log, flush=True)
    cache = os.path.join(RES, f"layerC4_{label}_evals.jsonl")      # the replay cache file
    obj = Objective(bp, ef, su, log, cache)
    x0 = STARTS[start]                                 # deterministic path: a restart replays cached evaluations
    if obj.memo:
        print(f"# replaying {len(obj.memo)} cached evaluations", file=log, flush=True)
    # NUMERICS: bounded least squares.
    #   x_scale=(UB-LB)/10   natural size of each parameter = a tenth of its allowed range;
    #   diff_step=DIFF_STEP  relative finite-difference nudges (above);
    #   max_nfev=90          at most 90 evaluations, not counting Jacobian nudges (scipy's convention).
    # The return value is ignored; the best point is taken from obj.best, which tracks every call.
    if label.startswith(STOP_PREFIXES):              # wrap the objective with the automatic stop rule
        inner, best_hist = obj, []

        # PYTHON NOTE: Stopper behaves like the Objective (every attribute it lacks, such as .best or
        # .memo, is looked up on `inner` by __getattr__), but each call also records the best sum of
        # squares so far and raises StopCalibration once the stop rule is met.
        class Stopper:
            def __getattr__(self, name):
                return getattr(inner, name)

            def __call__(self, x):
                r = inner(x)
                best_hist.append(inner.best[0])
                if len(best_hist) > STOP_N and best_hist[-STOP_N - 1] - best_hist[-1] < STOP_TOL:
                    print(f"# stop rule met after {len(best_hist)} evaluations: best {best_hist[-1]:.3f}, "
                          f"{STOP_N} evaluations earlier {best_hist[-STOP_N - 1]:.3f}", file=log, flush=True)
                    raise StopCalibration
                return r
        obj = Stopper()
    fixed = next((v for k, v in FIXED_BY_SUFFIX.items() if label.endswith(k)), {})
    if fixed:                                          # estimate only the free parameters; fixed ones stay put
        x0 = x0.copy()
        for i, v in fixed.items():
            x0[i] = v
        free = np.array([i for i in range(len(NAMES)) if i not in fixed])
        print(f"# fixed {[(NAMES[i], v) for i, v in fixed.items()]}", file=log, flush=True)

        # the objective as a function of the free parameters only (fixed ones re-inserted)
        def obj_free(z):
            x = x0.copy(); x[free] = z
            return obj(x)
        least_squares(obj_free, x0[free], bounds=(LB[free], UB[free]), x_scale=(UB[free] - LB[free]) / 10,
                      diff_step=DIFF_STEP[free], max_nfev=90)
    else:
        try:
            least_squares(obj, x0, bounds=(LB, UB), x_scale=(UB - LB) / 10, diff_step=DIFF_STEP, max_nfev=90)
        except StopCalibration:
            pass                                       # stopped by the rule: fall through and save the best
    ss, xb, mb = obj.best
    # Output: best parameters by name, its sum of squares, its moments, the target values (NaN data
    # values written as null; none of the current targets is NaN), the fast settings and the count.
    out = {"params": dict(zip(NAMES, map(float, xb))), "sum_sq": ss, "moments": mb,
           "targets": {k: v[0] if not isinstance(v[0], float) or not np.isnan(v[0]) else None for k, v in TARGETS.items()},
           "fast_config": FAST, "evaluations": obj.n}
    # default=float: anything json cannot write directly (e.g. NumPy numbers) is converted with float().
    json.dump(out, open(os.path.join(RES, f"layerC4_{label}_params.json"), "w"), indent=1, default=float)
    print("DONE", out["params"], ss, file=log, flush=True)


# Run main() only when this file is launched directly, not when the drivers import it.
if __name__ == "__main__":
    main()
