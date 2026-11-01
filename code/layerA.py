"""
Layer A of the FODT v2 model (see the methodology spec, "FODT Model - Methodology Spec (v2)").

De Nardi (2004)-style OLG economy:
  * 14 five-year periods (ages 20-89), work 20-64, survival certain before 60
  * CRRA utility, warm-glow bequests (De Nardi form) + optional charitable bequests
  * AR(1) earnings (Rouwenhorst, 5 states), productivity partly inherited
  * 1-2 heirs (population growth 1%/yr), estate split equally
  * estate tax 40% above a $30M exemption (married couple), converted to model
    units with SCF 2022 mean net worth
  * general equilibrium: Cobb-Douglas firm, r and w from K/L; beta calibrated to K/Y = 3
  * stationary distribution computed DIRECTLY (Young 2010 lottery on the grid), including
    the joint parent-child distribution while the parent is alive; Monte Carlo is only a
    cross-check.

Timing / dynasty structure
  Parent is 35 years (7 periods) older than the child. When the child enters at 20 the parent
  is 55 and the grandparent (90) has always died, so every parent has already received all
  inheritances. A parent who dies at the end of period t_p leaves the estate to the child at
  the start of child period t_p - 6 (child ages 30-59). Inheritances are unanticipated.
"""
# =====================================================================================================
# READER'S GUIDE TO THIS FILE (added for the publication copy; the program itself is unchanged)
# =====================================================================================================
#
# WHAT THIS FILE DOES
#   It builds and solves the simplest version of the model: a De Nardi (2004) life-cycle economy with
#   no entrepreneurs and no trusts. People are born at 20, work until 65, may die from 60 on, and are
#   all dead by 90. They save out of after-tax earnings, care about leaving a bequest (and, optionally,
#   about giving to charity), and pay a 40% estate tax above a $30M exemption. The file finds each
#   household's saving rule (by the endogenous grid method), then the long-run ("stationary")
#   distribution of wealth across people and across parent-child pairs, and reports summary
#   statistics (wealth Gini, top shares, estate counts, tax revenue, intergenerational elasticities).
#
# WHERE IT SITS IN THE PROJECT
#   Layer A (this file: life cycle, earnings, bequests, estate tax)
#     -> Layer B (layerB.py: adds entrepreneurial returns that create large fortunes)
#       -> Layer C (layerC.py: adds dynasty trusts)
#         -> drivers on top (calibrate_C.py, equilibrium.py, run_c5.py, run_sweep.py, ...).
#
# WHICH PARTS THE FINAL (PUBLISHED) MODEL ACTUALLY USES
#   * USED by the final model: only the two helper functions `rouwenhorst` (the 5-state earnings
#     process) and `normal_bins` (how a child's earnings state is drawn from the parent's).
#     layerB.py imports exactly these two (`from layerA import rouwenhorst, normal_bins`), and
#     Layer C builds on Layer B. No other file in the published code imports from layerA.py.
#   * NOT used by the final model (earlier-stage tools, kept so the Layer A stage can be reproduced):
#     `ParamsA`, the `Economy` class (prices, bequests, EGM solve, distribution, moments),
#     `_dead_branch`, `build`, `calibrate_beta`, `monte_carlo`, `_mc_gen`, `calibrate_rho_h`,
#     `calibrate_layerA`. Layer B has its own class `EconomyB` that re-implements and extends these
#     same steps, so reading this file is still the easiest way to learn how the later layers work.
#   * The default numbers in `ParamsA` are Layer A stage values, NOT the final calibration "v2c"
#     (for example the final model uses different beta_annual and phi1; see calibrate_C.py).
#
# SUGGESTED READING ORDER
#   1. ParamsA                     - every parameter and its default value.
#   2. rouwenhorst, normal_bins    - the earnings process and its inheritance (used by the final model).
#   3. Economy.__init__            - turns raw data (age-efficiency profile, survival) into model inputs.
#   4. Economy._productivity_fixed_point, Economy._prices
#                                  - long-run earnings-state shares by age; wages, interest, units.
#   5. Economy.bequest             - estate tax, charity split, and the value of leaving wealth at death.
#   6. Economy.solve, Economy.cfun - the household saving problem (endogenous grid method).
#   7. Economy.euler_errors        - an accuracy check of that solution.
#   8. Economy.lottery, Economy.transition, Economy.distribution, _dead_branch
#                                  - the long-run distribution of wealth, including parent-child pairs.
#   9. Economy.earnings_rank_rank, Economy.wealth_elasticity, Economy.moments
#                                  - statistics compared with data.
#  10. build, calibrate_beta, calibrate_rho_h, calibrate_layerA
#                                  - Layer A calibration helpers.
#  11. monte_carlo, _mc_gen        - a simulation cross-check of the distribution.
#
# PAPER SECTIONS
#   PAPER: Section 3.1 (households: demographics, preferences, earnings) - the parts reused by the
#          final model are rouwenhorst/normal_bins. The rest of this file is the Layer A precursor of
#          Sections 3.2 (bequests, estate tax), 3.3 (prices) and Appendix A (computation), which the
#          final model implements in layerB.py / layerC*.py.
#
# WORDS YOU'LL SEE IN THIS FILE
#   period        : 5 years. Age index t = 0 means ages 20-24, ..., t = 13 means ages 85-89 (T = 14).
#   TL            : number of working periods (9: ages 20-64). Retired from t = 9 (age 65) on.
#   gap           : 7 periods = 35 years, the parent-child age difference.
#   y             : log earnings ("productivity") state, one of 5 values. exp(y) multiplies earnings.
#   P             : the 5x5 Markov matrix: P[y, y'] = probability of moving from state y to y' next period.
#   Ph            : 5x5 matrix: Ph[y_parent, y_child] = probability a newborn child starts in y_child.
#   a, a'         : wealth at the start of a period / wealth saved for next period ("a prime").
#   m             : cash on hand = R * a + after-tax income; what can be spent or saved this period.
#   R             : after-tax gross 5-year return on saving (about 1.27 at the defaults).
#   ebar          : mean 5-year gross earnings per worker. Wealth grids and bequest shifters are
#                   stated as multiples of ebar.
#   model units   : the units produced by the production function (wage per efficiency unit etc.).
#                   Dollars enter only through the estate-tax exemption E, which is converted by
#                   equating model mean wealth with SCF mean net worth ($1,063,700).
#   grid          : the finite list of wealth levels (agrid) at which the problem is solved.
#   EGM           : endogenous grid method (see Economy.solve).
#   lottery       : putting a person who lands between two grid points partly on each (Young 2010).
#   state k       : one number for a (wealth grid point i, earnings state y) pair: k = i * ny + y.
#   stationary    : the long-run distribution that reproduces itself generation after generation.
#   W'(a')        : marginal value of wealth left at death (bequest + charity), "Wp" in the code.
# =====================================================================================================

# PYTHON NOTE: `import x` makes a library available. `math` = scalar maths (sqrt, erf);
# `dataclasses` = a compact way to define a bundle of named settings; `numpy` (np) = arrays of
# numbers and fast whole-array arithmetic; `scipy.sparse` (sp) = matrices that store only their
# non-zero entries; `numba.njit` = compiles a Python function to fast machine code.
import math
from dataclasses import dataclass, replace
import numpy as np
import scipy.sparse as sp
from numba import njit


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# =====================================================================================================
# SECTION 1. PARAMETERS
# =====================================================================================================
# PYTHON NOTE: `@dataclass` above a class turns a list of `name: type = default` lines into a settings
# object. `ParamsA()` gives all defaults; `ParamsA(phi1=-12.0)` overrides one. `replace(p, phi1=-12.0)`
# (imported above) makes a COPY of an existing settings object `p` with one value changed; the
# calibration routines below use this to try many values without altering the original.
@dataclass
class ParamsA:
    """All Layer A parameters with their default values.

    UNITS: rates are ANNUAL unless the name says otherwise (beta_annual, n_pop, delta, r_annual, tau_*
    are annual or plain shares). Quantities marked "_units" are multiples of ebar (mean 5-year
    earnings per worker). Quantities marked "_usd" are dollars. Grid bounds a_lo, a_max are also
    multiples of ebar.

    NOTE: these defaults are Layer A stage values. The final model's parameters live in ParamsB /
    ParamsC (layerB.py, layerC.py) and the final calibration in calibrate_C.py.
    """
    # demographics
    tlen: int = 5                # years per model period
    T: int = 14                  # number of periods of life: ages 20-24 (t=0) ... 85-89 (t=13)
    TL: int = 9                  # working periods: ages 20-64; retired from t = 9 (age 65)
    gap: int = 7                 # parent is 35 years older (see module docstring)
    n_pop: float = 0.01          # population growth, 1% per year (sets the number of children per parent)
    # preferences
    sigma: float = 1.5           # CRRA curvature: u(c) = c^(1-sigma)/(1-sigma); also used in bequest utility
    beta_annual: float = 0.95    # calibrated to K/Y = 3
    #                              (annual discount factor; the 5-year factor is beta_annual^5)
    phi1: float = -9.5           # bequest intensity (De Nardi form; negative)
    #                              PAPER: phi1 in phi1*[(1 + b/phi2)^(1-sigma) - 1]; with sigma > 1 a
    #                              negative phi1 makes this utility increasing in the bequest b.
    phi2_units: float = 8.0      # bequest luxury shifter, in mean 5-year earnings (placeholder)
    #                              (larger phi2 -> bequests behave more like a luxury good)
    chi: float = 0.0             # charitable warm glow (>= 0); 0 until IRS SOI targets are pulled
    #                              (with chi = 0 nobody gives to charity in Layer A)
    phi2c_units: float = 50.0    # charity luxury shifter, in mean 5-year earnings (placeholder)
    # taxes / policy
    tau_a: float = 0.18          # capital income (CBO effective rate, as in the paper)
    tau_l: float = 0.301         # labor tax wedge, U.S. single worker at average wage, 2024 (OECD Taxing Wages 2026)
    tau_b: float = 0.40          # estate tax, statutory top rate
    E_usd: float = 30e6          # estate exemption, married couple (2026, $15M per person)
    scf_mean_nw_usd: float = 1_063_700.0   # SCF 2022 mean family net worth (Fed Bulletin 2023)
    #                              PYTHON NOTE: underscores in 1_063_700.0 are only for readability.
    #                              UNITS: this is the anchor that converts dollars into model units.
    pens: float = 0.40           # pension = 40% of mean earnings
    #                              (40% of ebar per 5-year period, paid to every retiree, untaxed)
    # technology
    alpha: float = 0.36          # capital share in the Cobb-Douglas production function Y = K^alpha L^(1-alpha)
    delta: float = 0.06          # annual depreciation rate of capital
    KY: float = 3.0              # target capital-output ratio (capital stock / ANNUAL output)
    r_annual: float = None       # None -> implied by K/Y: r = alpha/KY - delta
    #                              (with the defaults r = 0.36/3 - 0.06 = 0.06, i.e. 6% per year)
    # earnings
    rho_y: float = 0.85          # persistence of log earnings from one 5-year period to the next
    sig2_y: float = 0.30         # variance of the earnings shock per period
    rho_h: float = 0.40          # De Nardi (2004) reference; to be estimated by SMM
    #                              (how strongly a child's starting earnings state follows the parent's;
    #                               calibrate_rho_h below sets it from the rank-rank slope)
    sig2_h: float = 0.37         # variance of the child's starting log earnings around rho_h * parent's y
    ny: int = 5                  # number of earnings states
    # grid (units of mean 5-year earnings)
    na: int = 150                # number of wealth grid points (including a = 0)
    a_lo: float = 0.005          # smallest positive grid point (0.5% of ebar)
    a_max: float = 2000.0        # largest grid point (2000 times ebar)

    # PYTHON NOTE: a function defined inside a class is a "method"; `self` is the object it belongs to,
    # so `self.alpha` reads this object's alpha. Call it as `p.r()`.
    def r(self):
        """Annual net interest rate r.

        If r_annual is not set, r comes from the firm's first-order condition at the target K/Y:
        marginal product of capital = alpha * Y / K = alpha / KY, and r = that minus depreciation.
        PAPER: Section 3.3 (firms). UNITS: per year (e.g. 0.06 = 6%).
        """
        # PYTHON NOTE: `A if condition else B` is a one-line if/else.
        return self.alpha / self.KY - self.delta if self.r_annual is None else self.r_annual


# ---------------------------------------------------------------------------
# Exogenous processes
# ---------------------------------------------------------------------------
# =====================================================================================================
# SECTION 2. EXOGENOUS PROCESSES (earnings states and their inheritance)
#   These two functions are the ONLY parts of this file used by the final model (via layerB.py).
# =====================================================================================================
def rouwenhorst(n, rho, sig2):
    """Discretise an AR(1) process for log earnings into n states (Rouwenhorst's method).

    The continuous process is  y' = rho * y + e,  e ~ Normal(0, sig2), one step = one 5-year period.
    NUMERICS: Rouwenhorst's method replaces this with n evenly spaced values and a Markov matrix
    whose persistence and long-run variance match the continuous process exactly. It is built
    recursively: start from a 2-state chain and grow it one state at a time.

    Inputs : n (number of states, 5 in the model), rho (persistence), sig2 (shock variance).
    Outputs: (grid, P)
             grid = n log-earnings values, evenly spaced from -psi to +psi (exp(grid) multiplies earnings);
             P    = n x n matrix, one row per state today, one column per state next period;
                    P[i, j] = probability of moving from state i to state j; each row sums to 1.
    PAPER: Section 3.1 (earnings process).
    """
    p = (1 + rho) / 2                                # probability of staying in the 2-state starting chain
    # PYTHON NOTE: np.array([[...], [...]]) builds a 2-D array (a matrix) from a list of rows.
    P = np.array([[p, 1 - p], [1 - p, p]])           # the 2-state chain
    for m in range(3, n + 1):                        # grow the chain to 3, 4, ..., n states
        Z = np.zeros((m, m))                         # new m x m matrix, filled with zeros
        # PYTHON NOTE: slicing. Z[:-1, :-1] = all rows but the last, all columns but the last (top-left
        # block); Z[1:, 1:] = bottom-right block, and so on. `+=` adds into that block in place.
        # The four lines place four weighted copies of the old chain in the four corners of Z.
        Z[:-1, :-1] += p * P; Z[:-1, 1:] += (1 - p) * P
        Z[1:, :-1] += (1 - p) * P; Z[1:, 1:] += p * P
        Z[1:-1, :] /= 2                              # middle rows were counted twice: halve so rows sum to 1
        P = Z
    # psi = half-width of the grid: (long-run std. dev. of y) * sqrt(n - 1).
    # The long-run variance of the AR(1) is sig2 / (1 - rho^2).
    psi = math.sqrt(sig2 / (1 - rho ** 2)) * math.sqrt(n - 1)
    # PYTHON NOTE: `**` is "to the power of". np.linspace(a, b, n) gives n evenly spaced numbers from a to b.
    return np.linspace(-psi, psi, n), P


def normal_bins(grid, mean, sd):
    """Probabilities that a Normal(mean, sd) draw falls in the 'bin' of each grid point.

    Used to draw a newborn child's earnings state: the child's log earnings are
    Normal(rho_h * parent's y, sqrt(sig2_h)), and this function maps that onto the 5 grid points.
    Bin edges are the midpoints between neighbouring grid points; the first and last bins extend
    to minus / plus infinity, so the probabilities sum to exactly 1.

    Inputs : grid (sorted array of n points), mean, sd (standard deviation).
    Output : array of n probabilities, one per grid point.
    APPROXIMATION: a continuous normal draw is rounded to the nearest grid point.
    PAPER: Section 3.1 (partly inherited productivity).
    """
    mids = (grid[1:] + grid[:-1]) / 2                # the n-1 bin edges (midpoints between grid points)
    # PYTHON NOTE: `[f(x) for x in mids]` is a "list comprehension": it applies f to every element and
    # collects the results in a list. Here f is the normal cumulative distribution function (CDF),
    # written with math.erf: CDF(x) = 0.5 * (1 + erf((x - mean) / (sd * sqrt(2)))).
    cdf = np.array([0.5 * (1 + math.erf((x - mean) / (sd * math.sqrt(2)))) for x in mids])
    # Put 0 in front and 1 at the end, then take differences of neighbours: P(edge_{j-1} < draw < edge_j).
    # PYTHON NOTE: np.concatenate joins arrays end to end; np.diff gives x[1]-x[0], x[2]-x[1], ...
    return np.diff(np.concatenate([[0.0], cdf, [1.0]]))


# =====================================================================================================
# SECTION 3. THE LAYER A ECONOMY (earlier-stage tool; the final model uses EconomyB in layerB.py)
#   Construction (data -> model inputs), prices, bequests, household solution, distribution, moments.
# =====================================================================================================
class Economy:
    """One Layer A economy: inputs, prices, household policies, stationary distribution, moments.

    Typical use (see `build` below):  Economy(p, eflife, surlife).solve().distribution().moments()
    PYTHON NOTE: each method that returns `self` lets calls be chained like this, left to right.
    """

    def __init__(self, p: ParamsA, eflife, surlife):
        """Set up the economy from parameters and two data files.

        Inputs:
          p       : a ParamsA settings object.
          eflife  : ANNUAL age-efficiency (earnings-by-age) profile, one number per year of age
                    starting at age 20 (loaded from data/eflife.dat by the driver scripts).
          surlife : ANNUAL survival probabilities, one per year of age starting at age 20
                    (probability of being alive next year; data/surlife.dat).
        Creates (all as attributes, `self.<name>`):
          eff     : 9 numbers, one per working period: average efficiency at that age, scaled so
                    that the average over the 9 working periods is exactly 1.
          sur     : 14 numbers: sur[t] = probability of surviving from period t to t+1 (5-year);
                    set to 1 before age 60 and 0 in the last period.
          y, P    : earnings grid (5 values) and its 5x5 Markov matrix (from rouwenhorst).
          Ph      : 5x5 matrix, row = parent's state at 55, column = child's starting state.
          p2, q2  : share of parents with 2 children; probability a child has a sibling.
          mass    : probability of being alive at each age (from survival), 14 numbers.
          cohort  : relative size of each age group due to population growth, 14 numbers.
        Then computes long-run earnings-state shares by age and all prices (see the two methods called last).
        """
        self.p = p
        T, TL, L = p.T, p.TL, p.tlen                 # 14 periods, 9 working periods, 5 years per period
        # Sum the annual efficiency profile in blocks of 5 years: one total per working period.
        # PYTHON NOTE: range(0, TL * L, L) = 0, 5, 10, ..., 40; eflife[i:i + L] = the 5 entries from i on.
        eff = np.array([eflife[i:i + L].sum() for i in range(0, TL * L, L)])
        self.eff = eff * TL / eff.sum()              # rescale so the mean over working periods is 1
        # 5-year survival = product of the 5 annual survival probabilities in each block (13 blocks),
        # plus a final 0: nobody survives beyond period 13 (age 89).
        sur = np.array([np.prod(surlife[i:i + L]) for i in range(0, (T - 1) * L, L)] + [0.0])
        sur[:8] = 1.0                                # APPROXIMATION: no deaths before age 60 (periods 0-7)
        self.sur = sur
        y, P = rouwenhorst(p.ny, p.rho_y, p.sig2_y)  # 5 log-earnings states and their Markov matrix
        self.y, self.P = y, P
        # For each possible parent state yi, the probabilities of each child starting state:
        # child's log earnings ~ Normal(rho_h * yi, sqrt(sig2_h)), binned onto the grid.
        self.Ph = np.array([normal_bins(y, p.rho_h * yi, math.sqrt(p.sig2_h)) for yi in y])
        # heirs: (1+n)^(35) children per parent; m in {1,2}
        # Each parent has 1 or 2 children; the average must be 1.01^35 (about 1.417) so that each
        # generation is 1.01^35 times bigger than the previous one.
        kids = (1 + p.n_pop) ** (p.tlen * p.gap)
        self.p2 = kids - 1.0                         # share of parents with 2 children
        #                                              (average = (1 - p2)*1 + p2*2 = 1 + p2 = kids)
        self.q2 = 2 * self.p2 / kids                 # child's probability of having a sibling
        #                                              (children of 2-child parents / all children,
        #                                               about 0.59 at the defaults)
        # age weights (cohort size) and survival mass
        # mass[t] = probability a newborn is still alive at age t = sur[0]*...*sur[t-1]; mass[0] = 1.
        self.mass = np.concatenate([[1.0], np.cumprod(sur[:-1])])
        # cohort[t] = size of the age-t group relative to the newest group, from population growth:
        # people who are t periods older were born when the population was (1.01)^(5t) times smaller.
        self.cohort = (1 + p.n_pop) ** (-p.tlen * np.arange(T))
        self._productivity_fixed_point()            # long-run earnings-state shares by age (pi_age)
        self._prices()                               # wages, interest, income, grid, unit conversion

    # stationary distribution of productivity by age (newborn y depends on parent's y at age 55)
    def _productivity_fixed_point(self):
        """Long-run share of people in each earnings state at each age.

        A newborn's state depends on the parent's state at 55 (via Ph); the parent's state at 55
        depends on the parent's own starting state (via 7 steps of P). So the newborn distribution
        must reproduce itself across generations. NUMERICS: we find it by simple repetition
        ("fixed-point iteration"): guess, push one generation forward, repeat until it stops changing.

        Creates self.pi_age: 14 x 5 array, one row per age, one column per earnings state; each
        row sums to 1. Earnings states do not depend on wealth or survival, so this needs no model solve.
        """
        p = self.p; P, Ph = self.P, self.Ph
        pi0 = np.ones(p.ny) / p.ny                   # first guess: newborns spread evenly over 5 states
        for _ in range(2000):                        # at most 2000 generations
            pit = pi0.copy()
            for _t in range(p.gap):                  # age the newborns 7 periods (to age 55)
                # PYTHON NOTE: `@` is matrix multiplication. A row of shares times P gives next period's
                # shares: share'(j) = sum over i of share(i) * P[i, j].
                pit = pit @ P
            new = pit @ Ph                           # their children's starting states
            if np.abs(new - pi0).max() < 1e-14:      # stop when no share changes by more than 1e-14
                break
            pi0 = new
        # With the newborn distribution d, age it period by period to fill in every age.
        self.pi_age = np.zeros((p.T, p.ny)); d = pi0
        for t in range(p.T):
            self.pi_age[t] = d; d = d @ P

    def _prices(self):
        """Prices, aggregates, after-tax incomes, the wealth grid and the dollar conversion.

        PAPER: Section 3.3 (Cobb-Douglas firm, government). Layer A version; the final model's
        version is EconomyB._prices in layerB.py.
        Creates (UNITS in brackets):
          r_ann   annual net interest rate [per year]
          KL, w   capital per efficiency unit of labour; wage per efficiency unit [per year]
          age_w   population share of each age (14 numbers summing to 1)
          L       efficiency labour per person (all ages) [per year]
          K_dem   capital the firm wants per person [model units]
          Y       output per person [model units PER YEAR]
          ebar    mean 5-year gross earnings per worker [model units per 5 years]; the scale unit
          inc     14 x 5 array (one row per age, one column per earnings state): after-tax income
                  received during the 5-year period [model units per 5 years]
          R       after-tax gross 5-year return on saving [ratio, e.g. 1.27]
          beta    5-year discount factor
          phi2, phi2c  bequest and charity shifters [model units]
          E       estate-tax exemption [model units]
          agrid   the wealth grid, 150 values [model units]
        """
        p = self.p
        r = p.r()                                    # annual interest rate (6% at the defaults)
        self.r_ann = r
        # Firm first-order condition: r + delta = alpha * (K/L)^(alpha-1). Solve for K/L.
        self.KL = (p.alpha / (r + p.delta)) ** (1 / (1 - p.alpha))
        self.w = (1 - p.alpha) * self.KL ** p.alpha  # wage = marginal product of labour (annual)
        wts = self.cohort * self.mass                # relative number of people at each age
        self.age_w = wts / wts.sum()                 # population share by age
        effy = np.exp(self.y)                        # efficiency multiplier of each earnings state
        # Average efficiency of a person at age t: E[exp(y) at age t] * age profile; 0 when retired.
        lab_t = np.array([(self.pi_age[t] @ effy) * self.eff[t] if t < p.TL else 0.0 for t in range(p.T)])
        self.L = (self.age_w * lab_t).sum()          # efficiency labor per capita
        self.K_dem = self.KL * self.L                # capital per capita
        self.Y = self.K_dem ** p.alpha * self.L ** (1 - p.alpha)   # annual output per capita
        workers = self.age_w[:p.TL].sum()            # share of the population that is working (ages 20-64)
        # Total annual wage bill per person is w*L; divide by the worker share and multiply by 5 years.
        self.ebar = p.tlen * self.w * self.L / workers   # mean 5-year gross earnings per worker
        u = self.ebar                                # "u" = the unit in which grids/shifters are stated
        # per-period after-tax income by (t, y)
        inc = np.zeros((p.T, p.ny))
        for t in range(p.T):
            # Workers: (1 - labour tax) * 5 years * wage * age efficiency * state efficiency (5 values).
            # Retirees: the same pension, 40% of ebar, whatever their earnings state (untaxed).
            inc[t] = (1 - p.tau_l) * p.tlen * self.w * self.eff[t] * effy if t < p.TL else p.pens * u
        self.inc = inc
        # After-tax gross 5-year return: the compounded 5-year net return (1+r)^5 - 1 is taxed at tau_a.
        self.R = 1 + ((1 + r) ** p.tlen - 1) * (1 - p.tau_a)
        self.beta = p.beta_annual ** p.tlen          # 5-year discount factor
        self.phi2 = p.phi2_units * u                 # bequest shifter, converted to model units
        self.phi2c = p.phi2c_units * u               # charity shifter, converted to model units
        # UNITS: dollars -> model units. Model mean wealth (taken as the firm's capital demand K_dem)
        # stands for SCF mean net worth, so $1 = K_dem / 1,063,700 model units.
        # APPROXIMATION: uses K_dem, which equals household wealth only once beta is calibrated
        # (calibrate_beta); and treats one model person as one SCF family.
        self.E = p.E_usd / p.scf_mean_nw_usd * self.K_dem    # exemption in model units
        # Wealth grid: 0, then 149 points spaced evenly in logs from 0.005 to 2000 (times ebar).
        # NUMERICS: log spacing puts many points at low wealth (where choices bend sharply) and few
        # at very high wealth (where choices are almost straight lines).
        # PYTHON NOTE: np.geomspace(a, b, n) = n points from a to b with a constant ratio between them.
        self.agrid = np.concatenate([[0.0], np.geomspace(p.a_lo, p.a_max, p.na - 1)]) * u

    # ---- bequests: charity split, net estate to heirs, marginal value W'(a') ----
    # -------------------------------------------------------------------------------------------------
    # 3a. Bequests, charity and the estate tax
    # -------------------------------------------------------------------------------------------------
    def bequest(self, x):
        """x = end-of-life wealth (array). Returns (g, net_to_heirs, Wprime).

        What happens to the wealth x of someone who dies with savings x (model units):
          1. They may give g to charity (only if chi > 0). Charity is deducted before the estate tax.
          2. The rest, z = x - g, is the gross estate; estate tax tau_b is paid on the part above E.
             net_to_heirs = z - tau_b * max(z - E, 0)   (the TOTAL left to all children together;
             it is split between 1 or 2 children elsewhere).
          3. Wprime = W'(x), the extra utility from one more unit of wealth at death. This is the
             value the household's saving decision uses for the "die this period" case.
        The bequest-utility and charity-utility functions (PAPER: Section 3.2, bequest motive) are
             bequest: sc * phi1 * [(1 + b/phi2)^(1-sigma) - 1]
             charity: sc * chi  * [1 - (1 + g/phi2c)^(1-sigma)]
        with b = net bequest. The code only needs their slopes (marginal utilities), phid and psid.
        If chi > 0, g is chosen so the last dollar to charity and the last dollar to heirs are worth
        the same (an "interior" optimum), or g = 0 if even the first dollar to charity is worth less.

        Input  : x, any-shape array of end-of-life wealth.
        Outputs: three arrays of the same shape as x: g, net_to_heirs, Wprime.
        """
        p = self.p; s = p.sigma; E = self.E
        x = np.asarray(x, float)                     # make sure x is a float array
        # PYTHON NOTE: `def f(z): return ...` on one line defines a small helper function.
        # np.maximum(a, b) takes the larger value element by element (works on whole arrays).
        def bnet(z):  return z - p.tau_b * np.maximum(z - E, 0.0)   # net estate after estate tax
        # PYTHON NOTE: np.where(condition, A, B) picks A where the condition holds and B elsewhere.
        def bder(z):  return np.where(z > E, 1 - p.tau_b, 1.0)     # d(net)/d(gross): 0.6 above E, else 1
        # phi1 and chi are stated in units where mean 5-year earnings = 1 (De Nardi's units);
        # CRRA utility scales with units, so they are rescaled by ebar^(1-sigma).
        # (Consumption utility c^(1-sigma)/(1-sigma) gets multiplied by ebar^(1-sigma) when c is
        #  measured in model units instead of ebar units; sc gives the bequest term the same factor.)
        sc = self.ebar ** (1 - s)
        # Marginal utility of a net bequest b: derivative of sc*phi1*[(1 + b/phi2)^(1-s) - 1].
        def phid(b):  return sc * p.phi1 * (1 - s) / self.phi2 * (1 + b / self.phi2) ** (-s)
        # Marginal utility of a charitable gift g: derivative of sc*chi*[1 - (1 + g/phi2c)^(1-s)].
        def psid(g):  return sc * p.chi * (s - 1) / self.phi2c * (1 + g / self.phi2c) ** (-s)
        g = np.zeros_like(x)                         # default: no charity (array of zeros shaped like x)
        if p.chi > 0:
            # Is the first dollar to charity worth more than the last dollar to the heirs?
            interior = psid(0.0) > phid(bnet(x)) * bder(x)
            # NUMERICS: bisection. For each x, g lies between lo = 0 and hi = x. Each step tests the
            # midpoint and keeps the half where the answer must lie; 60 halvings pin it down to
            # x / 2^60, i.e. to machine precision. All x values are handled at once (whole arrays).
            lo = np.zeros_like(x); hi = x.copy()
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                # f > 0: one more dollar to charity still beats giving it to heirs -> give more.
                f = psid(mid) - phid(bnet(x - mid)) * bder(x - mid)
                lo = np.where(f > 0, mid, lo); hi = np.where(f > 0, hi, mid)
            g = np.where(interior, 0.5 * (lo + hi), 0.0)   # charity only where it is optimal
        z = x - g                                     # gross estate after charity
        # Returns: charity, net bequest to heirs (total), and the marginal value of end-of-life wealth
        # (= value of the extra net bequest times how much of a gross dollar reaches the heirs).
        return g, bnet(z), phid(bnet(z)) * bder(z)

    # ---- household problem: endogenous grid method ----
    # -------------------------------------------------------------------------------------------------
    # 3b. The household problem: endogenous grid method (EGM)
    # -------------------------------------------------------------------------------------------------
    def solve(self):
        """Solve every household's consumption / saving rule, from the last period back to the first.

        The household at age t, with cash on hand m = R*a + income, chooses consumption c and
        savings a' = m - c >= 0 (no borrowing). The Euler equation (optimal saving condition) is
            u'(c) = beta * [ sur[t] * E( R * u'(c next period) )  +  (1 - sur[t]) * W'(a') ]
        i.e. the value of a unit saved = survive and spend it later, or die and leave it as a bequest.
        PAPER: Section 3.2 (household problem), Appendix A (computation) - Layer A version.

        NUMERICS (endogenous grid method, EGM): instead of searching for the best c at each m,
        we fix a' on the grid, compute the right-hand side of the Euler equation (which depends only
        on a'), invert u'(c) = rhs to get c = rhs^(-1/sigma) directly, and then learn which cash on
        hand m = c + a' that choice belongs to. This gives pairs (m, c) on an "endogenous" grid of m,
        with no root-finding. Working backwards from age 13 to age 0 uses each age's answer for the
        age before.

        Creates (each a 14 x 150 x 5 array: age, wealth grid point, earnings state):
          m_nodes, c_nodes : the EGM pairs (cash on hand, consumption) for each age and state
          c, ap            : consumption and savings a' for someone who STARTS the period with wealth
                             agrid[i] in earnings state y (model units, per 5-year period)
        Returns self (so calls can be chained).
        """
        p = self.p; T, ny = p.T, p.ny
        a = self.agrid; R = self.R; beta = self.beta; s = p.sigma
        _, _, Wp = self.bequest(a)                   # W'(a') at every grid value of a'
        #                                              (PYTHON NOTE: `_` means "ignore this output")
        self.m_nodes = np.zeros((T, a.size, ny)); self.c_nodes = np.zeros((T, a.size, ny))
        self.ap = np.zeros((T, a.size, ny))      # a'(a, y) at grid points
        self.c = np.zeros((T, a.size, ny))
        for t in range(T - 1, -1, -1):           # t = 13, 12, ..., 0 (backwards in age)
            st = self.sur[t]                     # chance of surviving to the next period
            if t < T - 1:
                cn = np.empty((a.size, ny))      # next-period consumption, row = a' grid point, column = y'
                EV = np.zeros((a.size, ny))
                for yn in range(ny):
                    mn = R * a + self.inc[t + 1, yn]          # next period's cash on hand for each a'
                    cn[:, yn] = self.cfun(t + 1, yn, mn)      # next period's consumption (already solved)
                margu = cn ** (-s)                     # (a', y')
                #                                        marginal utility u'(c') = c'^(-sigma)
                # Expected R*u'(c') given today's y: average over y' with weights P[y, y'].
                # (margu @ P.T)[a', y] = sum over y' of margu[a', y'] * P[y, y'].
                EV = R * margu @ self.P.T              # (a', y)
            else:
                EV = np.zeros((a.size, ny))            # last period: nobody survives, only the bequest matters
            # Right-hand side of the Euler equation for each (a', y).
            # PYTHON NOTE: Wp[:, None] turns a 1-D column of 150 values into a 150 x 1 array, so that
            # adding it to the 150 x 5 array EV repeats it across the 5 columns ("broadcasting").
            # The floor 1e-300 avoids dividing by zero (e.g. with no bequest motive in the last period).
            rhs = np.maximum(beta * (st * EV + (1 - st) * Wp[:, None]), 1e-300)
            cend = rhs ** (-1 / s)               # invert u'(c) = rhs: consumption that makes a' optimal
            mend = cend + a[:, None]             # the cash on hand at which that choice is made: m = c + a'
            self.m_nodes[t], self.c_nodes[t] = mend, cend
            # Now read off the policy at the fixed wealth grid: start-of-period wealth a = agrid[i].
            for y in range(ny):
                m = R * a + self.inc[t, y]       # cash on hand of someone starting with wealth a in state y
                c = self.cfun(t, y, m)
                self.c[t, :, y] = c
                self.ap[t, :, y] = m - c         # savings carried into the next period
        return self

    def cfun(self, t, y, m):
        """Consumption at age t, earnings state y, for any cash on hand m (an array).

        NUMERICS: linear interpolation between the EGM nodes (m_nodes, c_nodes) of that age/state;
        straight-line extrapolation beyond the highest node; and if m is below the lowest node
        (which is the m at which a' = 0 is just optimal), the borrowing constraint binds: the
        household consumes everything, c = m, a' = 0.
        Input m and output c are in model units per 5-year period; same shape.
        """
        mn = self.m_nodes[t, :, y]; cn = self.c_nodes[t, :, y]
        # PYTHON NOTE: np.interp(x, xp, fp) reads the straight line through points (xp, fp) at x.
        c = np.interp(m, mn, cn)
        hi = m > mn[-1]                                 # linear extrapolation above
        if hi.any():                                    # any m beyond the last node?
            slope = (cn[-1] - cn[-2]) / (mn[-1] - mn[-2])   # slope of the last segment
            c = np.where(hi, cn[-1] + slope * (m - mn[-1]), c)
        return np.where(m <= mn[0], m, c)              # borrowing constraint: a' = 0

    # ---- Euler-equation errors at unconstrained grid points ----
    def euler_errors(self):
        """Accuracy check: how far the solved policy is from satisfying the Euler equation exactly.

        For each age t < 13, earnings state and grid point where the household saves (a' > 1e-8,
        i.e. the borrowing constraint does not bind), it recomputes the right-hand side of the
        Euler equation, converts it to the consumption it implies, and compares with the solved c:
            error = | 1 - c_implied / c_solved |.
        Output: 1-D array of log10(errors). For example -4 means a 0.01% consumption error.
        Errors arise because the solve step works on a grid and interpolates between points.
        """
        p = self.p; s = p.sigma; a = self.agrid; out = []
        for t in range(p.T - 1):
            _, _, Wp = self.bequest(self.ap[t])      # W'(a') at the chosen savings (150 x 5)
            for y in range(p.ny):
                apy = self.ap[t, :, y]               # chosen savings at each grid point
                ok = apy > 1e-8                      # True where the household is unconstrained
                if not ok.any():
                    continue                         # nothing to check in this state
                EV = 0.0
                for yn in range(p.ny):               # expectation over next period's earnings state
                    cn = self.cfun(t + 1, yn, self.R * apy + self.inc[t + 1, yn])
                    EV = EV + self.P[y, yn] * self.R * cn ** (-s)
                rhs = self.beta * (self.sur[t] * EV + (1 - self.sur[t]) * Wp[:, y])
                err = np.abs(1 - rhs ** (-1 / s) / self.c[t, :, y])
                # PYTHON NOTE: err[ok] keeps only the entries where ok is True ("boolean mask").
                out.append(err[ok])
        e = np.concatenate(out)                      # all errors in one long array
        return np.log10(np.maximum(e, 1e-16))        # floor at 1e-16 so log10 of 0 never happens

    # ---- stationary distribution (direct) ----
    # -------------------------------------------------------------------------------------------------
    # 3c. The stationary distribution (computed directly, without simulation)
    # -------------------------------------------------------------------------------------------------
    def lottery(self, x):
        """Split wealth values x between the two neighbouring grid points (Young 2010 "lottery").

        NUMERICS: someone with savings x between agrid[j] and agrid[j+1] is treated as a mass that
        goes to agrid[j] with weight wlo and to agrid[j+1] with weight 1 - wlo, where the weights
        keep the average wealth exactly x. Savings above the top grid point are put at the top.
        APPROXIMATION: wealth above a_max (2000 * ebar) is truncated to a_max.
        Input : x, array of savings (any shape).
        Output: (j, wlo) of the same shape: index of the lower grid point, weight on it (0 to 1).
        """
        a = self.agrid
        x = np.minimum(x, a[-1])                     # cap at the top grid point
        # PYTHON NOTE: np.searchsorted(a, x, side="right") gives the position where x would be
        # inserted in the sorted grid a; minus 1 = index of the last grid point <= x.
        # np.clip keeps j in 0 .. na-2 so that j+1 is always a valid grid point.
        j = np.clip(np.searchsorted(a, x, side="right") - 1, 0, a.size - 2)
        wlo = (a[j + 1] - x) / (a[j + 1] - a[j])     # 1 if x is at the lower point, 0 at the upper
        return j, np.clip(wlo, 0.0, 1.0)

    def transition(self, t):
        """Sparse transition for an individual of age t: state k = i*ny + y.

        Builds the matrix that moves the population of age t to age t+1 (ignoring death and
        inheritances, which are handled in `distribution`). There are N = 150 * 5 = 750 states:
        state k = (wealth grid point i, earnings state y), numbered k = i * ny + y.
        Entry [k_next, k_now] = probability of going from state k_now to k_next:
            (lottery weight of the savings choice ap[t, i, y] on the new grid point) * P[y, y_next].
        Columns = today's state, rows = tomorrow's state, so tomorrow's masses = M @ today's masses.
        NUMERICS: most entries are zero (each state leads to at most 2 * 5 = 10 states), so a
        "sparse" matrix that stores only the non-zero entries is used.
        Output: 750 x 750 scipy sparse matrix (CSR format).
        """
        p = self.p; na, ny = self.agrid.size, p.ny
        j, wlo = self.lottery(self.ap[t])                 # (na, ny)
        rows, cols, vals = [], [], []                     # lists of (row, column, value) entries
        for i in range(na):
            for y in range(ny):
                k = i * ny + y                            # today's state number
                for yn in range(ny):
                    pr = self.P[y, yn]                    # chance of moving to earnings state yn
                    # PYTHON NOTE: `list += [u, v]` appends u and v to the list.
                    # Two entries: to the lower grid point and to the upper one, each with state yn.
                    rows += [j[i, y] * ny + yn, (j[i, y] + 1) * ny + yn]
                    cols += [k, k]
                    vals += [wlo[i, y] * pr, (1 - wlo[i, y]) * pr]
        N = na * ny
        # Build the sparse matrix from the entry lists (repeated (row, column) pairs are added up).
        return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

    def distribution(self, tol=1e-11, maxgen=500, verbose=False):
        """Stationary distribution of wealth by age, including the joint parent-child distribution.

        IDEA. Follow ONE generation of children from birth (age 20) to age 89, keeping track of
        their parents while the parents are alive, because a child's wealth jumps when the parent
        dies and leaves an inheritance. At age 55 these children become the parents of the next
        generation. Repeat generations until the distribution of parents at 55 stops changing.

        Objects inside the loop (N = 750 states, k = wealth point i * 5 + earnings state y):
          D   : length-N, distribution of PARENTS at the start of age 55 (sums to 1).
          X   : N x N, joint mass of (child state, parent state) for children whose parent is still
                ALIVE; at child age t the parent is age t + 7.
          O   : 14 x N, mass of children at each age whose parent has already DIED ("orphaned";
                they have received their inheritance).
          child_marg : 14 x N, all children at each age (parent alive + parent dead).
        Timing: a parent who dies at the end of parent age tp leaves h[tp] (net of estate tax),
        split equally among 1 or 2 children, added to the child's savings for the next period.
        APPROXIMATION: inheritances are unanticipated (the solve step ignores them).
        NOTE: the child's own survival is not applied while the parent may be alive (t <= 6);
        this is exact because nobody dies before 60 (sur[:8] = 1 in __init__).
        PAPER: Appendix A (computation of the distribution), Layer A version.

        Inputs : tol (stop when the parent distribution changes by less than this, summed over
                 states), maxgen (maximum generations), verbose (print progress).
        Creates: dist (14 x N: mass by age, including survival), parent_D (length N), X3 (N x N joint
                 distribution at child 35-39 / parent 70-74 while the parent is alive),
                 gens_to_converge, and the helper arrays _h, _gch, _M. Returns self.
        """
        p = self.p; T, gap, ny = p.T, p.gap, p.ny
        na = self.agrid.size; N = na * ny
        M = [self.transition(t) for t in range(T)]  # one transition matrix per age
        # net estate to heirs for each parent state at each parent age
        h = np.zeros((T, N)); gch = np.zeros((T, N))  # h = net bequest, gch = charity (14 x N each)
        for t in range(T):
            g, net, _ = self.bequest(self.ap[t])     # a parent dying at the end of age t leaves ap[t]
            # PYTHON NOTE: reshape(-1) flattens a 150 x 5 array into 750 numbers, in the order
            # k = i * 5 + y (the same numbering as the states).
            h[t] = net.reshape(-1); gch[t] = g.reshape(-1)
        apc = self.ap.reshape(T, N)                  # savings by age and state, 14 x 750
        # initial guess for parents' distribution at age gap: no inheritances
        # Newborns start with zero wealth (grid point 0 = states 0..4) and earnings shares pi_age[0];
        # age them 7 periods with no inheritance (no deaths before 60, so no survival factor needed).
        D = np.zeros(N); D[0 * ny:(0 + 1) * ny] = self.pi_age[0]
        for t in range(gap):
            D = M[t] @ D
        Py = self.P
        for gen in range(maxgen):                     # one pass = one generation
            O = np.zeros((T, N))
            X = np.zeros((N, N))                       # [child state, parent state]
            # newborns: a = 0, y from parent's y via Ph
            Dm = D.reshape(na, ny)                     # parents: row = wealth point, column = earnings state
            for yc in range(ny):
                # Child state yc (zero wealth, earnings yc) paired with every parent state:
                # mass = parent mass * Ph[parent's y, yc]. Dm * Ph[:, yc][None, :] multiplies each
                # column (parent's y) by that column's probability (broadcasting).
                X[yc, :] = (Dm * self.Ph[:, yc][None, :]).reshape(-1)
            child_marg = np.zeros((T, N))
            for t in range(T - 1):                     # child age t = 0..12
                tp = t + gap                           # parent's age
                if t == 3:
                    X3 = X.copy()                      # store joint at child 35-39 / parent 70-74
                if tp <= T - 1:                        # parent may still be alive (child age 0-6)
                    # all children of this age = those with a living parent (summed over parent
                    # states: X.sum(axis=1) adds along each row) + those already orphaned
                    child_marg[t] = X.sum(axis=1) + O[t]
                    st = self.sur[tp]                  # parent's chance of surviving this period
                    # parent dies: estate to the child
                    # _dead_branch returns a 150 x 5 array (child's next wealth point, child's
                    # CURRENT earnings state); `@ Py` then moves the child's earnings state forward.
                    dead = _dead_branch(X, apc[t], h[tp], self.agrid, ny, self.q2, 1.0 - st)
                    # Next period's orphans = old orphans moved forward one period (times child
                    # survival, which is 1 here) + children whose parent just died.
                    O[t + 1] = M[t] @ O[t] * self.sur[t] + (dead @ Py).reshape(-1)
                    # parent survives: joint transition
                    # Move parents with their age-tp matrix and children with their age-t matrix:
                    # (M[tp] @ X.T) moves the parent side, `.T` flips back, then M[t] @ moves the
                    # child side. Times the parent's survival probability st.
                    # PYTHON NOTE: X.T is the transpose of X (rows and columns swapped).
                    X = st * (M[t] @ (M[tp] @ X.T).T)
                else:                                  # parent is dead for sure (would be 90+)
                    child_marg[t] = O[t]
                    O[t + 1] = M[t] @ O[t] * self.sur[t]
            child_marg[T - 1] = O[T - 1]
            # Children at age 55 (all orphaned by now) are the next generation's parents.
            Dn = O[gap].copy()
            diff = np.abs(Dn - D).sum()                # total absolute change in the parent distribution
            D = Dn
            if verbose:
                print(gen, diff)
            if diff < tol:
                break
        self.gens_to_converge = gen + 1
        self.dist = child_marg                          # (T, N): mass by age (includes survival)
        #                                                 (mass per newborn; NOT yet weighted by
        #                                                  cohort size - moments() does that)
        self.parent_D = D
        self.X3 = X3                                    # joint (child 35-39, parent 70-74) while parent alive
        self._h, self._gch, self._M = h, gch, M
        return self

    # ---- intergenerational statistics ----
    # -------------------------------------------------------------------------------------------------
    # 3d. Intergenerational statistics and summary moments
    # -------------------------------------------------------------------------------------------------
    def earnings_rank_rank(self):
        """Rank-rank slope: child's earnings at 35-39 on parent's earnings at 55-59 (exact, from the chains).

        Data counterpart: Chetty et al. (2014), slope 0.341 (see calibrate_rho_h). Ranks are
        computed from earnings states: within an age, higher y means higher earnings, so ranking
        by y is ranking by earnings. Each state gets its "mid-rank" (the average percentile of the
        people in it). The slope is a weighted regression of child rank on parent rank.
        Needs only the earnings chains, not the household solution. Output: a single number.
        """
        p = self.p
        py = self.pi_age[p.gap]                                     # parent's y at 55 (exogenous)
        # Joint probability of (parent's y at 55, child's y at age 35-39): child starts via Ph,
        # then 3 periods of P. 5 x 5 array, row = parent's state, column = child's state.
        J = py[:, None] * (self.Ph @ np.linalg.matrix_power(self.P, 3))   # joint (y_p, y_c at age 35-39)
        def ranks(m):
            # PYTHON NOTE: np.cumsum gives running totals: [m0, m0+m1, m0+m1+m2, ...].
            c = np.cumsum(m); return c - m / 2                       # mid-ranks in [0, 1]
        rp, rc = ranks(J.sum(axis=1)), ranks(J.sum(axis=0))      # parent ranks, child ranks
        mp = (J.sum(axis=1) * rp).sum(); mc = (J.sum(axis=0) * rc).sum()   # mean ranks (= 0.5)
        cov = (J * (rp[:, None] - mp) * (rc[None, :] - mc)).sum()          # covariance of ranks
        var = (J.sum(axis=1) * (rp - mp) ** 2).sum()                       # variance of parent rank
        return cov / var                                         # regression slope

    def wealth_elasticity(self, floor_units=0.01):
        """Elasticity of child wealth (35-39) w.r.t. parent wealth (70-74), positive wealth only (as in Charles & Hurst).

        Regression slope of log(child wealth) on log(parent wealth) across parent-child pairs in
        which the parent is still alive, weighted by the stationary joint distribution X3.
        "Positive wealth" here means both above floor_units * ebar (default 1% of mean 5-year
        earnings); pairs with less wealth are dropped. Wealth = start-of-period wealth on the grid.
        Output: a single number.
        """
        p = self.p; ny = p.ny; a = self.agrid
        # PYTHON NOTE: reshape(150, 5, 150, 5) splits each 750-state index back into (wealth, y);
        # .sum(axis=(1, 3)) adds over both earnings states, leaving a 150 x 150 wealth table.
        J = self.X3.reshape(a.size, ny, a.size, ny).sum(axis=(1, 3))    # (a_child, a_parent)
        ok = a > floor_units * self.ebar             # grid points above the wealth floor
        # PYTHON NOTE: np.ix_(ok, ok) selects the sub-table of rows ok and columns ok.
        J = J[np.ix_(ok, ok)]; la = np.log(a[ok])
        w = J / J.sum()                              # weights summing to 1
        # mx = mean log parent wealth (column sums); my = mean log child wealth (row sums)
        mx = (w.sum(axis=0) * la).sum(); my = (w.sum(axis=1) * la).sum()
        cov = (w * (la[:, None] - my) * (la[None, :] - mx)).sum()
        var = (w.sum(axis=0) * (la - mx) ** 2).sum()
        return cov / var

    # ---- moments ----
    def moments(self):
        """Summary statistics of the stationary economy (a Python dictionary of name -> value).

        Wealth statistics use the whole population aged 20-89, weighted by cohort size and survival.
        Wealth = start-of-period wealth on the grid (model units). Main entries:
          K_supply, K_demand, K/Y  : household wealth per person, firm capital demand, and K / annual Y
          gini, top1 ... top60     : wealth Gini and shares of wealth held by the top 1% ... 60%
          share_zero               : share of people with zero wealth
          bequest_flow/Y           : value of estates (before charity and tax) per year / annual output
          share_estates_taxable    : share of deaths whose estate after charity exceeds E
          mean_wealth_by_age       : 14 numbers, mean wealth at each age / overall mean
          mass_at_grid_top         : share of people on the top two grid points (should be ~0)
          exemption/mean_wealth    : E / K
          charity/estates          : charity / value of estates
          revenue/Y, pensions/Y, implied_g/Y : government flows (annual) / annual output
          wealth_75plus/mean       : mean wealth of people 75-89 / overall mean (Layer A target 1.53)
          parent_child_wealth_elasticity, earnings_rank_rank : intergenerational statistics
        UNITS: flows are converted to ANNUAL by dividing 5-year flows by tlen = 5.
        """
        p = self.p; T, ny = p.T, p.ny; a = self.agrid; na = a.size
        # PYTHON NOTE: np.repeat(a, 5) repeats each grid value 5 times: the wealth of each state k.
        wa = np.repeat(a, ny)
        W = (self.dist * self.cohort[:, None])         # cross-section weights
        #                                                (mass per newborn * relative cohort size)
        tot = W.sum()                                   # total population (normaliser)
        w = W.sum(axis=0) / tot                         # pooled over ages
        K = (w * wa).sum()                              # mean wealth per person = capital supply
        # wealth distribution statistics
        # Lorenz curve: sort people from poorest to richest; L = cumulative share of wealth.
        # PYTHON NOTE: np.argsort gives the order that sorts the array; xs, ws = sorted wealth, weights.
        o = np.argsort(wa)
        xs, ws = wa[o], w[o]
        cw = np.cumsum(ws); cx = np.cumsum(xs * ws)    # cumulative population share; cumulative wealth
        L = cx / cx[-1]; Lp = np.concatenate([[0], L[:-1]])   # Lorenz curve, and the same shifted by one
        # Gini = 1 - 2 * (area under the Lorenz curve), area computed with trapezoids.
        gini = 1 - np.sum(ws * (L + Lp))
        cw0 = np.concatenate([[0.0], cw]); L0 = np.concatenate([[0.0], L])
        def top(f):
            # share of wealth held by the richest fraction f = 1 - Lorenz curve at (1 - f)
            return 1.0 - np.interp(1.0 - f, cw0, L0)
        # estates (deaths at ages >= 60)
        est_val = 0.0; taxable = 0.0; deaths = 0.0; estate_tax = 0.0; charity = 0.0
        for t in range(T):
            # share of the population that is age t, in each state, and dies at the end of the period
            q = self.dist[t] * self.cohort[t] / tot * (1 - self.sur[t])
            apt = self.ap[t].reshape(-1)               # what they leave: their savings a'
            g, net, _ = self.bequest(apt)
            deaths += q.sum(); est_val += (q * apt).sum(); taxable += q[apt - g > self.E].sum()
            # estate tax paid = gross estate - charity - net to heirs
            estate_tax += (q * (apt - g - net)).sum(); charity += (q * g).sum()
        # UNITS: deaths, est_val, estate_tax, charity are per 5-year period, per person.
        # government (annual, per capita): g is the residual, no lump-sum tax on households
        retirees = self.age_w[p.TL:].sum()             # population share aged 65+
        rev_l = p.tau_l * self.w * self.L              # annual labour-tax revenue per person
        # APPROXIMATION: annual capital-tax revenue uses simple annual interest r * K, while R above
        # taxes the compounded 5-year return; the two differ slightly.
        rev_k = p.tau_a * self.r_ann * K
        rev_b = estate_tax / p.tlen                    # annual estate-tax revenue
        pens = retirees * p.pens * self.ebar / p.tlen  # annual pension spending per person
        # NOTE: "Wage" here is mean WEALTH by age (not wages): wealth-weighted mass / mass, 14 numbers.
        Wage = (self.dist * wa[None, :]).sum(axis=1) / self.dist.sum(axis=1)
        # PYTHON NOTE: `{ "name": value, ... }` is a dictionary; read an entry with m["gini"].
        return {
            "K_supply": K, "K_demand": self.K_dem, "K/Y": K / self.Y,
            "gini": gini, "top1": top(0.01), "top5": top(0.05), "top10": top(0.10),
            "top20": top(0.20), "top40": top(0.40), "top60": top(0.60),
            "share_zero": w[wa <= 1e-12].sum(),
            "bequest_flow/Y": est_val / p.tlen / self.Y,
            "share_estates_taxable": taxable / deaths,
            "mean_wealth_by_age": Wage / K,
            "mass_at_grid_top": w[wa >= a[-2]].sum(),
            "exemption/mean_wealth": self.E / K,
            "charity/estates": charity / est_val if est_val > 0 else 0.0,
            "revenue/Y": (rev_l + rev_k + rev_b) / self.Y,
            "pensions/Y": pens / self.Y,
            # government purchases = whatever revenue is left after pensions (the residual)
            "implied_g/Y": (rev_l + rev_k + rev_b - pens) / self.Y,
            # ages 75-89 are t = 11, 12, 13: population-weighted mean of their mean wealth, / K
            "wealth_75plus/mean": float((Wage[11:] * (self.dist[11:].sum(axis=1) * self.cohort[11:])).sum()
                                        / (self.dist[11:].sum(axis=1) * self.cohort[11:]).sum() / K),
            "parent_child_wealth_elasticity": self.wealth_elasticity(),
            "earnings_rank_rank": self.earnings_rank_rank(),
        }


# PYTHON NOTE: `@njit` (from Numba) compiles the function below to fast machine code the first
# time it is called; `cache=True` saves the compiled version to disk for later runs. Inside such
# functions, plain loops are fast (in ordinary Python they would be slow).
@njit(cache=True)
def _dead_branch(X, apc_t, h_tp, agrid, ny, q2, pdie):
    """Children whose parent dies: child's next wealth = own savings + inheritance/m.

    (m = number of children sharing the estate: 1 or 2.)
    Inputs:
      X     : N x N joint mass [child state, parent state] of children whose parent is alive.
      apc_t : length N, the child's own savings a' at child age t, by child state.
      h_tp  : length N, the parent's net estate (after estate tax and charity) if the parent
              dies at the end of parent age tp, by parent state (model units).
      agrid : the wealth grid; ny : number of earnings states (5).
      q2    : probability the child has a sibling (then gets half the estate).
      pdie  : probability the parent dies this period (1 - sur[tp]).
    Output: na x ny array (150 x 5): mass of newly orphaned children by NEXT period's wealth grid
            point and the child's CURRENT earnings state (the caller moves earnings forward).
    NUMERICS: the new wealth is split between the two neighbouring grid points (the lottery);
    the grid point is found by binary search (halve the search range until two points remain).
    APPROXIMATION: new wealth above the top grid point is put on the top grid point.
    """
    N = X.shape[0]; na = agrid.size
    out = np.zeros((na, ny))
    amax = agrid[-1]
    for c in range(N):                               # each child state
        yc = c % ny                                  # its earnings state (`%` = remainder: k = i*ny + y)
        ac = apc_t[c]                                # the child's own savings
        for pp in range(N):                          # each parent state
            m = X[c, pp]                             # mass of this (child, parent) pair
            if m == 0.0:
                continue                             # skip empty pairs (saves time)
            m *= pdie                                # the part whose parent dies now
            for sib in range(2):                     # sib = 0: only child; sib = 1: has one sibling
                pr = (1.0 - q2) if sib == 0 else q2
                x = ac + h_tp[pp] / (1.0 + sib)      # own savings + the estate (or half of it)
                if x >= amax:
                    out[na - 1, yc] += m * pr        # beyond the grid: put on the top point
                    continue
                # binary search for lo with agrid[lo] <= x < agrid[lo + 1]
                lo, hi = 0, na - 1
                while hi - lo > 1:
                    mid = (lo + hi) // 2             # `//` = whole-number division
                    if agrid[mid] <= x:
                        lo = mid
                    else:
                        hi = mid
                wl = (agrid[lo + 1] - x) / (agrid[lo + 1] - agrid[lo])   # lottery weight on the lower point
                out[lo, yc] += m * pr * wl
                out[lo + 1, yc] += m * pr * (1.0 - wl)
    return out


# ---------------------------------------------------------------------------
# Calibration and Monte Carlo cross-check
# ---------------------------------------------------------------------------
# =====================================================================================================
# SECTION 4. LAYER A CALIBRATION HELPERS AND MONTE CARLO CROSS-CHECK
#   (earlier-stage tools; the final model is calibrated by calibrate_B2.py and calibrate_C.py)
# =====================================================================================================
def build(p, eflife, surlife):
    """Set up, solve and compute the distribution of a Layer A economy in one call.

    Inputs as in Economy.__init__. Returns the finished Economy object (call .moments() on it).
    """
    return Economy(p, eflife, surlife).solve().distribution()


def calibrate_beta(p, eflife, surlife, lo=0.90, hi=0.999, tol=1e-4, verbose=True):
    """Bisection on beta so household wealth = firm capital demand at r implied by K/Y.

    r is fixed by the K/Y = 3 target (ParamsA.r), so the firm's capital demand K_demand is fixed;
    beta_annual is moved until households want to hold exactly that much wealth (K_supply).
    NUMERICS: bisection between lo and hi (annual beta). If households save too much (gap > 0),
    the upper bound comes down; otherwise the lower bound goes up. At most 40 halvings; stops when
    |K_supply / K_demand - 1| < tol.
    Returns (beta_annual found, the solved Economy at that beta).
    """
    for it in range(40):
        mid = 0.5 * (lo + hi)
        e = build(replace(p, beta_annual=mid), eflife, surlife)   # solve the economy at this beta
        m = e.moments()
        gap = m["K_supply"] / m["K_demand"] - 1        # excess wealth (here `gap` is NOT the age gap)
        if verbose:
            # PYTHON NOTE: f"..." is an f-string: values in {} are inserted; :.5f = 5 decimals,
            # :+.4f = 4 decimals with a sign.
            print(f"  beta={mid:.5f}  K_supply/K_demand-1={gap:+.4f}")
        if abs(gap) < tol:
            break
        if gap > 0:
            hi = mid                                   # too much saving -> lower beta
        else:
            lo = mid                                   # too little saving -> higher beta
    return mid, e


def monte_carlo(e: Economy, N=200_000, G=12, burn=6, seed=7):
    """Simulate dynasties with the same policies (linear interpolation = the lottery).

    Cross-check of `distribution`: simulate N family lines (one person per generation) for G
    generations, using the solved savings policy (interpolated linearly between grid points),
    random deaths, random earnings moves, inheritances (halved if the child has a sibling) and
    inherited earnings states. The first `burn` generations are discarded so that the starting
    guess (no inheritances) no longer matters. Pools all living people of the kept generations,
    weighted by cohort size, and computes the same wealth statistics as moments().
    NUMERICS: the lottery in `distribution` and linear interpolation of the policy here give the
    same expected next-period wealth, so the two methods should agree up to sampling noise.
    Inputs : e (a solved Economy), N (dynasties), G (generations), burn (generations discarded),
             seed (random-number seed, so results are reproducible).
    Output : dictionary with gini, top1, top5, top10, top20 and mean wealth (model units).
    """
    rng = np.random.default_rng(seed)              # random-number generator
    p = e.p; T, gap, ny = p.T, p.gap, p.ny
    # Cumulative probabilities along each row: used to draw the next state with one uniform number.
    Pc = np.cumsum(e.P, axis=1); Phc = np.cumsum(e.Ph, axis=1)
    a = e.agrid
    # initial parents: no inheritance at all, y from stationary newborn dist
    y0 = rng.choice(ny, size=N, p=e.pi_age[0])     # N starting earnings states
    inh_amt = np.zeros(N); inh_t = np.full(N, -1)  # inheritance amount and the age it arrives (-1 = never)
    samples = []
    for g in range(G):                             # one generation at a time
        # Uniform random numbers for each person and age: [.., 0] for death, [.., 1] for the
        # earnings move. (The third column, [.., 2], appears to be unused by _mc_gen.)
        U = rng.random((N, T, 3))
        A, alive, y7, td, est = _mc_gen(e.ap, a, e.sur, Pc, y0, inh_amt, inh_t, U, gap)
        if g >= burn:
            samples.append((A, alive))             # keep wealth paths of later generations
        # children
        _, net, _ = e.bequest(est)                 # net estate after estate tax, for each dynasty
        sib = rng.random(N) < e.q2                 # True if the child has a sibling
        inh_amt = net / np.where(sib, 2.0, 1.0)    # child's share of the estate
        # Parent dies at the end of parent age td; the child is td - 7 then and receives the
        # inheritance at the start of child age td - 6.
        inh_t = td - (gap - 1)
        u = rng.random(N)
        # Child's starting earnings state from the parent's state at 55 (inverse-CDF draw).
        # PYTHON NOTE: zip(a, b) pairs up elements of two lists.
        y0 = np.array([np.searchsorted(Phc[yy], uu) for yy, uu in zip(y7, u)])
        y0 = np.minimum(y0, ny - 1)                # guard against rounding pushing past the last state
    # Wealth of everyone alive in the kept generations, and their ages.
    x = np.concatenate([s[0][s[1]] for s in samples])
    # PYTHON NOTE: np.broadcast_to repeats the ages 0..13 for every person, matching A's shape.
    tt = np.concatenate([np.broadcast_to(np.arange(T), s[0].shape)[s[1]] for s in samples])
    w = e.cohort[tt]                               # weight = relative cohort size of their age
    # Same Lorenz-curve, Gini and top-share calculation as in moments().
    o = np.argsort(x); xs, ws = x[o], w[o]
    cw = np.cumsum(ws) / ws.sum(); cx = np.cumsum(xs * ws)
    L = cx / cx[-1]; Lp = np.concatenate([[0], L[:-1]])
    gini = 1 - np.sum(ws / ws.sum() * (L + Lp))
    cw0 = np.concatenate([[0.0], cw]); L0 = np.concatenate([[0.0], L])
    def top(f):
        return 1.0 - np.interp(1.0 - f, cw0, L0)
    return {"gini": gini, "top1": top(0.01), "top5": top(0.05), "top10": top(0.10),
            "top20": top(0.20), "mean": (xs * ws).sum() / ws.sum()}


@njit(cache=True)
def _mc_gen(ap, agrid, sur, Pc, y0, inh_amt, inh_t, U, gap):
    """Simulate one generation of N people from age 20 until death (compiled with Numba).

    Inputs:
      ap      : 14 x 150 x 5 savings policy; agrid : wealth grid; sur : 14 survival probabilities
      Pc      : cumulative earnings transition probabilities (5 x 5)
      y0      : N starting earnings states
      inh_amt, inh_t : each person's inheritance and the age index at which it arrives (-1 = none)
      U       : N x 14 x 3 uniform random numbers; gap : 7
    Outputs:
      A     : N x 14 wealth at the start of each age (0 after death)
      alive : N x 14 True/False, alive at the start of each age
      y7    : earnings state at age 55 (t = 7), used to draw the child's state
      td    : age index at which each person dies (at the end of that period)
      est   : wealth left at death (savings a' of the last period), model units
    """
    N = y0.size; T = ap.shape[0]; na = agrid.size
    A = np.zeros((N, T)); alive = np.zeros((N, T), np.bool_)
    y7 = np.zeros(N, np.int64); td = np.zeros(N, np.int64); est = np.zeros(N)
    for i in range(N):                               # each person
        x = 0.0; y = y0[i]                           # born with zero wealth
        for t in range(T):
            if t == inh_t[i]:
                x += inh_amt[i]                      # inheritance arrives at the start of this period
            if t == gap:
                y7[i] = y                            # record earnings state at 55
            A[i, t] = x; alive[i, t] = True
            # interpolate policy a'(x, y)
            if x >= agrid[-1]:
                xn = ap[t, na - 1, y]                # above the grid: use the policy at the top point
            else:
                lo, hi = 0, na - 1                   # binary search for agrid[lo] <= x < agrid[lo + 1]
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if agrid[mid] <= x:
                        lo = mid
                    else:
                        hi = mid
                wl = (agrid[lo + 1] - x) / (agrid[lo + 1] - agrid[lo])
                xn = wl * ap[t, lo, y] + (1 - wl) * ap[t, lo + 1, y]   # linear interpolation of a'
            if U[i, t, 0] >= sur[t]:                 # dies at the end of this period
                td[i] = t; est[i] = xn
                break
            # Draw next earnings state: the first j whose cumulative probability reaches U.
            j = 0
            while j < Pc.shape[1] - 1 and U[i, t, 1] > Pc[y, j]:
                j += 1
            y = j; x = xn                            # move to next period
    return A, alive, y7, td, est


def calibrate_rho_h(p, eflife, surlife, target=0.341):
    """rho_h so that the earnings rank-rank slope matches Chetty et al. (2014). No model solve needed.

    NUMERICS: bisection on rho_h between 0 and 0.99 (40 halvings); a higher rho_h makes children's
    earnings follow their parents' more closely and raises the rank-rank slope.
    Returns the rho_h found (a number).
    """
    lo, hi = 0.0, 0.99
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        rr = Economy(replace(p, rho_h=mid), eflife, surlife).earnings_rank_rank()
        if rr > target:
            hi = mid                                 # slope too high -> less inheritance of earnings
        else:
            lo = mid
    return mid


def calibrate_layerA(p, eflife, surlife, old_target=1.53, phi_lo=-60.0, phi_hi=-0.5, iters=10, verbose=True):
    """Outer bisection on phi1 (old-age wealth / mean), inner bisection on beta (K/Y = 3).

    Layer A calibration in three steps:
      1. rho_h from the earnings rank-rank slope (calibrate_rho_h; target 0.341).
      2. For each trial phi1 (bequest strength), re-calibrate beta so that K/Y = 3 (calibrate_beta).
      3. Bisect phi1 between phi_lo and phi_hi (10 rounds) until mean wealth of people aged 75-89,
         relative to the overall mean, equals old_target = 1.53.
    Returns the solved Economy from the last trial (its parameters are in e.p).
    """
    rho = calibrate_rho_h(p, eflife, surlife)
    p = replace(p, rho_h=rho)
    if verbose:
        print(f"rho_h = {rho:.4f} (rank-rank target 0.341)")
    lo, hi = phi_lo, phi_hi
    for it in range(iters):
        mid = 0.5 * (lo + hi)
        b, e = calibrate_beta(replace(p, phi1=mid), eflife, surlife, lo=0.85, hi=0.99, tol=5e-4, verbose=False)
        old = e.moments()["wealth_75plus/mean"]
        if verbose:
            print(f"  phi1={mid:8.3f}  beta={b:.4f}  wealth 75+/mean={old:.3f}")
        if old > old_target:      # too much old-age wealth -> weaker bequest motive (phi1 closer to 0)
            lo = mid
        else:
            hi = mid
    return e
