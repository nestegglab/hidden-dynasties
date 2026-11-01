"""
Layer C of the FODT v2 model: Layers A + B + wealth-preservation technology.

=================================================================================================
READING GUIDE (start here)
=================================================================================================
WHAT THIS FILE DOES
    Layer C is the core of the model. It takes the Layer B economy (life-cycle households with
    earnings risk, entrepreneurial returns and a bequest motive) and adds dynasty trusts: a family
    can move wealth into a trust that is outside the taxable estate and passes to the heirs at every
    generation. The file (1) solves every household's problem backwards by age (consumption, saving,
    and the discrete trust choices: found / keep / take income / add / wind up), (2) moves the whole
    population forward through those choices to find the stationary distribution of parents and
    children, and (3) computes the moments used in calibration and the experiments.

WHERE IT SITS
    layerA.py (life cycle, earnings) -> layerB.py (fortunes: entrepreneurial returns, bequests,
    the parent-child distribution) -> THIS FILE, layerC (trusts, gifts, tier-2 family office).
    The drivers sit on top: calibrate_C.py (the 7-parameter calibration, paper 3.4), run_c5.py (the
    experiments with the interest rate re-solved, paper 3.6), run_sweep.py, run_gridcheck.py.

PAPER SECTIONS IMPLEMENTED HERE
    3.2 Transfers and the household problem: bequests and charity (_beq1), the option menu and the
        choice probabilities (_menu), one period's timing including lifetime gifts (_decide), and
        inheritance (the _conv_* functions, EconomyC.conv later in this file).
    3.5 DFA measurement of top shares (dfa_top_shares, in the second half of the file).
    Appendix A, computation: EconomyC.solve, EconomyC.distribution, EconomyC._anderson.

READING ORDER
    First half of the file:
      1. ParamsC                 every parameter of Layer C (on top of ParamsB).
      2. ROLE / TIER / CATMAP    how the household category c encodes (role, tier).
      3. Value-function helpers  _u (utility), _beq1 (bequest + charity + estate tax), _lot (lottery
                                 weights onto a grid), _vext / _q / _envelope_all (reading off the
                                 EGM solution, with the upper envelope), _qd (value of an off-grid
                                 trust choice).
      4. The choice kernels      _add, _contribs, _menu (the option menu and probabilities), _decide
                                 (everything that happens in one period, for every state).
      5. Distribution kernels    _estates_fill, _push, _push_idx, _push_split, _estate_hist2 (moving
                                 population mass through choices and deaths), _shiftY, _lot_tables,
                                 _conv_trust_children, _conv_c0_children, _conv_all (inheritances).
    Second half of the file: class EconomyC (solve, distribution, moments), buildC,
    dfa_top_shares and reporting tables.

WORDS YOU'LL SEE IN THIS FILE
    model units  Money is measured in model units; `usd` (set in Layer B) = dollars per model unit,
                 chosen so that model mean wealth equals the SCF mean net worth ($1.0637M). Every
                 *_usd parameter is divided by usd before use. `ebar` = mean 5-year earnings.
    period       5 years. Flows that are "per year" in ParamsC are converted to per-period in
                 EconomyC.__init__, e.g. a 0.2% annual fee becomes ffac = (1 - 0.002)^5.
    m            cash on hand: liquid wealth after this period's return, plus income.
    d, D         trust assets at the start of the period (d, a grid point) and after this period's
                 return, tax and fees (D).
    a'  (ap)     liquid wealth carried to next period (saving).  d' (dp): trust carried forward.
    category c   0 no trust, 1 heir tier 1, 2 founder tier 1, 3 / 4 heir / founder tier 2.
    ys           combined index of the earnings state y and the return state z: ys = y * nz + z.
    xi, wxi      the return shock nodes (Gauss-Hermite) and their probability weights; index k.
    EGM          endogenous grid method: a fast way to solve the saving problem (see _q, and
                 EconomyC.solve later in this file).
    upper envelope  when the saving problem has several local optima, keep the best (see _q).
    lottery      splitting mass that lands between two grid points onto both of them (see _lot).
    logit        smooth choice probabilities from small random taste shocks (see _menu).
    money-metric scale lam  converts utility differences into "share of wealth" units (see _menu).
    friction     a random planning cost (share of wealth) that a family must overcome to found a
                 trust (or to upgrade to tier 2); logistic with mean fric_mu and scale fric_nu.
    ko           how many of the most likely options are stored per state for the distribution.
    flat state index st   all states in one long vector: first the no-trust block (na * nyz states),
                 then one block of na * nd * nyz states per trust category (see _decide).
=================================================================================================

(Original module description follows; every statement still holds.)

Stage C1: tier 1, dynasty trusts. Stage C2: tier 2, advanced management / family office, with
the two estate-tax avoidance channels (freeze transfers and valuation discounts).
(All reported results use stage C1 only: tier2 = False.)

Household categories (role, tier):
    0  no trust                         (role 0, tier 0)
    1  heir of a trust,    tier 1       (role 1: non-grantor trust, taxed at tau_ng)
    2  founder,            tier 1       (role 2: grantor trust; founder pays its income tax)
    3  heir,               tier 2 (family office)
    4  founder,            tier 2
With tier2=False only categories 0-2 exist (stage C1).

Trust assets d earn the family's return state (same shock as liquid wealth), pay a trustee fee,
and pass to heirs outside the estate at every generation (no estate or GST tax).

Choices each period (after returns and income are realised):
  * category 0: stay, or FOUND a trust (tier 1): pay the setup cost and move x into it. Funding is
    free of gift tax up to the exemption E and taxed at tau_gift per dollar above it
    (tax-exclusive, so 0.40 per dollar = 28.6% of the outlay). Founding also depends on a planning
    friction kappa (share of wealth), a logistic draw each period, so take-up rises smoothly with
    wealth (spec: "Who opts in").
  * heirs: retain the trust's income or have it paid out, or add to the trust with their own
    exemption (they then become grantors: approximation, a second trust is not tracked). Winding
    up is not allowed in the baseline (dead-hand terms, spendthrift clauses; allow_windup=True is a
    robustness case), except that trusts below d_min are terminated as uneconomic (UTC 414).
  * founders: retain or contribute more. The founder is not a beneficiary (completed gift; a
    retained interest would pull the trust into the estate, IRC 2036); founder_payout=True is a
    robustness case.
  * tier 1 -> tier 2 (UPGRADE): pay setup2 and a second planning friction (logistic, mean mu2).
    Tier 2 -> tier 1 (downgrade) is free. Heirs inherit the family's tier.

Tier 2 (family office / advanced management), relative to tier 1:
  * lower tax drag on returns: tau_a2 on liquid wealth and grantor-trust income, tau_ng2 on
    non-grantor trust income (deferral, borrowing against assets, step-up; Leiserson & Yagan 2021);
  * running cost max(fo_min, fo_rate * (a + d)) a year; no return premium (pi_D = 0 baseline);
  * FREEZE TRANSFERS (founders and heirs in tier 2): a share grat of liquid wealth is held
    through rolling GRATs / sales to grantor trusts with the family's dynasty trust as remainder:
    each period the realised return above the hurdle
    (1 + hurdle)^5 passes to the trust free of gift tax, the downside stays with the family;
  * VALUATION DISCOUNTS: when the family is in a business return state, the taxable value of
    liquid wealth at death and of gifts is (1 - disc_v) of its value (family partnerships,
    minority and marketability discounts).
Behavioral return gap lam_b (annual, on liquid wealth): full in tier 0, half in tier 1, none in
tier 2. Default 0 (to be calibrated with the rest in C4).
(Note: lam_b is not among the 7 parameters calibrated in calibrate_C.py, so it stays 0 in v2c.)

Approximations (logged in DECISIONS.md):
  * Remaining exemption of a founder = max(E - d, 0) with the current trust value (counts trust
    growth as exemption used; conservative). A founder's liquid estate gets the same remaining
    exemption at death.
  * Heirs who add to an inherited trust merge their own grantor trust into it.
  * Non-grantor trust income is taxed at tau_ng whether or not it is paid out.
  * A child who founded a trust before the parent dies merges the inherited trust into it and
    takes the higher of the two tiers.
  * Contribution sizes are a discrete menu (the remaining exemption and fractions of cash on hand);
    choices among options are smoothed by small logit shocks (scale nu_d, share of wealth).
  * Freeze transfers are automatic for tier-2 founders at share grat (the choice is the tier).

Solution: for every trust column d_j and next-period category, the liquid-saving problem is solved
by EGM with a DC-EGM upper envelope; off-grid trust choices are interpolated along lines of
constant total wealth. Distribution: factorised direct method (as Layer B) over two assets, with
Anderson acceleration of the generation fixed point.

Nesting: trusts=False reproduces Layer B; tier2=False reproduces stage C1 (both tested).
"""
# PYTHON NOTE: `import x` makes a library available. `math` = scalar maths (exp, sqrt); numpy (`np`)
# = arrays of numbers; numba's `njit` compiles a Python function to fast machine code and `prange`
# is a loop whose iterations run in parallel on several processor cores.
import math
from dataclasses import dataclass
import numpy as np
from numba import njit, prange

# Layer C builds on Layer B: ParamsB holds all the Layer A/B parameters (preferences, taxes, earnings,
# returns, grids), and EconomyB the Layer B economy, which EconomyC extends.
from layerB import ParamsB, EconomyB


# ==============================================================================================
# PARAMETERS
# ==============================================================================================
# PYTHON NOTE: `@dataclass` turns the class below into a simple record of named settings with
# default values. `ParamsC(ParamsB)` means ParamsC contains every ParamsB setting plus the new ones
# listed here. A run creates one with e.g. ParamsC(nd=55, ko=2, gifts=True, ...); any setting not
# given keeps the default written here. `name: type = value` declares one setting.
#
# FINAL CALIBRATION v2c (calibrate_C.py, paper 3.4). The values used in all reported results differ
# from the defaults below as follows:
#   calibrated (7):  beta_annual 0.9470, phi1 -1.549, chi 0.149 (these three live in ParamsB),
#                    fric_mu 0.0750, fric_nu 0.0265, trust_weight 1.38, gift_phi -1.79.
#   configuration:   nd = 55 (not 40), ko = 2 (not 3), nxi = 5 (ParamsB default 7), gifts = True,
#                    tier2 = False (so every tier-2 parameter below is unused), med_on = False.
#   The remaining Layer B parameters come from calibrate_B2.py (see calibrate_C.base_params).
# UNITS: "_usd" parameters are in dollars and are converted to model units in EconomyC.__init__
# by dividing by usd; "per year" amounts and rates are converted to 5-year periods there.
@dataclass
class ParamsC(ParamsB):
    trusts: bool = True               # False switches trusts off: reproduces Layer B (the "off" experiment)
    nd: int = 40                      # trust-asset grid: number of trust grid points d_j (v2c: 55)
                                      #   the grid is 0 plus nd - 1 points spaced evenly in logs from
                                      #   d_lo_usd up to the top of the liquid-wealth grid
    d_lo_usd: float = 250e3           # lowest positive trust grid point, dollars
    setup_usd: float = 50e3           # legal cost of founding a dynasty trust (dollars, paid once)
    fee_fixed_usd: float = 10e3       # trustee fee, fixed part (per year); x5 for a period
    fee_prop: float = 0.002           # trustee fee, share of assets (per year); ffac = (1 - 0.002)^5
    tau_gift: float = 0.40            # gift tax per dollar transferred above the exemption
    tau_ng: float = 0.24              # tax on non-grantor trust income (top rate on dividends/gains)
    fric_mu: float = 0.05             # founding friction: mean (share of wealth)  (mu_f; v2c: 0.0750)
    fric_nu: float = 0.03             # founding friction: logistic scale           (nu_f; v2c: 0.0265)
    fric_slope: float = 0.0           # robustness: founding friction mean scales as (W/wref)^(-slope)
                                      #   (0 = the friction is the same share of wealth for everyone)
    fric_wref_usd: float = 30e6       # reference wealth for fric_slope (the family exemption)
    dfa_theta: float = 0.509          # DFA-style measurement: top-1% share of corporate equities (Q2 2026)
                                      #   (used by dfa_top_shares, paper 3.5, as the 50.9% top-1% share)
    scf_trust_share: float = 0.0231   # trust assets recorded in SCF 2022, share of net worth
    nu_d: float = 0.002               # logit smoothing of the contribution/payout choice
                                      #   (scale of the taste shocks, as a share of wealth; see _menu)
    fracs: tuple = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)   # contribution menu (shares of cash on hand)
    ko: int = 3                       # options stored per state for the distribution (v2c: 2)
    d_min_usd: float = 100e3          # heirs' trusts below this are terminated (UTC 414-style)
    allow_windup: bool = False        # heirs may dissolve the trust (robustness)
    founder_payout: bool = False      # founder may receive trust income (robustness)
    # ---- tier 2 (C2) ----   (OFF in every reported result: tier2 = False)
    tier2: bool = False               # True adds categories 3 and 4 (family office)
    setup2_usd: float = 500e3         # setting up a family office / MFO relationship (dollars, once)
    fo_min_usd: float = 1e6           # minimum annual running cost (k_min), dollars per year
    fo_rate: float = 0.005            # annual running cost per dollar managed (k_FO)
    tau_a2: float = 0.10              # effective tax on returns with advanced management
    tau_ng2: float = 0.133            # non-grantor trust income tax with advanced management
    fric2_mu: float = 0.05            # upgrade friction: mean (share of wealth)
    fric2_nu: float = 0.03            # upgrade friction: logistic scale
    grat: float = 0.5                 # share of a tier-2 founder's liquid wealth run through GRATs
    hurdle: float = 0.05              # IRC 7520 rate (annual); per period Hg = (1 + hurdle)^5
    disc_v: float = 0.30              # valuation discount on business assets (tier 2)
    lam_b: float = 0.0                # behavioral return gap on unmanaged liquid wealth (annual)
    # ---- switches added after the first review (all off in the main results; see DECISIONS.md) ----
    payout_fracs: tuple = ()          # extra heir options: withdraw these shares of the drawable amount
                                      #   (partial withdrawal). () = all-or-nothing payout only (main
                                      #   results); (0.25, 0.5, 0.75) is the "lock0_partial" check.
    fund_cap_E: bool = False          # True: trust funding capped at the remaining exemption, so no
                                      #   shielded funding above it (a bound for the GST inclusion
                                      #   ratio; the "base_capE" experiment, paper Section 4.5).
    gift_exempt: bool = True          # False: lifetime gifts above gift_excl get no exemption at all
                                      #   (the opposite bound to the main model, which lets gifts use
                                      #   the full remaining exemption without using it up; "giftnoex").
    trust_prem: float = 0.0           # robustness: extra annual return on trust assets over the family's
                                      # own return (professional management, concentrated positions)
    # ---- C3: lock-in, tax-shield switch, lifetime gifts ----
    lock: float = 1.0                 # share of principal heirs cannot reach (1 = binding trust)
                                      #   (ell in the paper; lock = 0 is the "lock0" experiment)
    trust_weight: float = 1.0         # testator's weight on trust assets in the bequest motive (theta)
                                      #   (v2c: 1.38)
    shield: bool = True               # trust assets outside the taxable estate (False: 'no shield')
    gifts: bool = False               # one lifetime-gift decision when children form households
                                      #   (at parent age 55, t = gap; v2c: True)
    gift_phi: float = -2.0            # strength of the gift warm glow (same form as the bequest motive)
                                      #   (phi_g; negative like phi1; v2c: -1.79)
    gift_excl_usd: float = 300e3      # gifts per family free of gift tax at that stage (annual
                                      # exclusions for 2 parents x ~1.4 children x 5 years, plus
                                      # tuition paid directly)
    gift_fracs: tuple = (0.01, 0.03, 0.08, 0.2, 0.4)  # gift menu (shares of cash on hand)
    # The next four settings only affect how the distribution code tracks gifts: which
    # parent-child pairs are kept separate. They do not change any household decision.
    gift_pool_usd: float = 1e6        # gifts below this are pooled in the parent-child pairing
    gift_grid_factor: float = 3.0     # spacing of the gift grid above gift_pool_usd (ratio)
    gift_min_pair: float = 0.0        # large-gift bins below this share of a pair's mass are pooled
    gift_merge_types: bool = True     # large-gift pairs merged across children's initial types
    # C4 v3: late-life medical costs (De Nardi, French & Jones 2010, JPE). Mean annual out-of-pocket
    # cost at age 75 in 1998 dollars, growing exponentially with age (fitted to their reported means:
    # about $1,100 at 75, $9,200 at 95, $17,700 at 100), rising with permanent-income rank (top/bottom
    # quintile ratio about 9 at age 95). Proxy for permanent income: the earnings state y. Floor on
    # cash on hand: their estimated consumption floor ($2,663, 1998 dollars). cpi_98: 1998 -> 2022.
    # (OFF in v2c: med_on = False. It belongs to the v3 variant in calibrate_C.py.)
    med_on: bool = False
    med75_usd98: float = 1100.0       # mean annual cost at age 75, 1998 dollars
    med_growth: float = 0.106         # growth of the cost per year of age (exponential)
    med_pi_slope: float = 2.79        # slope in the permanent-income rank (cost ~ exp(slope * rank))
    cfloor_usd98: float = 2663.0      # consumption floor, 1998 dollars per year
    cpi_98: float = 1.80              # price level 2022 / 1998
    med_start_age: int = 65           # costs apply from this age


# ==============================================================================================
# HOUSEHOLD CATEGORIES
# ==============================================================================================
# Category c (0..4) -> role and tier. Role: 0 no trust, 1 heir (non-grantor trust), 2 founder
# (grantor trust). Tier: 0 none, 1 dynasty trust, 2 family office.
# PYTHON NOTE: np.array([...]) makes an array; ROLE[c] reads entry c (counting from 0).
ROLE = np.array([0, 1, 2, 1, 2])
TIER = np.array([0, 1, 1, 2, 2])
# The reverse map: CATMAP[role, tier] -> category. Row = role, column = tier. Entries that make no
# sense (e.g. role 0 with tier 1) map to 0. Example: CATMAP[2, 1] = 2 = "founder, tier 1".
CATMAP = np.array([[0, 0, 0], [0, 1, 3], [0, 2, 4]])     # CATMAP[role, tier]


# ==============================================================================================
# NUMBA KERNELS, PART 1: UTILITY, BEQUESTS, GRIDS AND READING OFF THE SAVING SOLUTION
# ==============================================================================================
# PYTHON NOTE: `@njit(cache=True)` (Numba) compiles the function below to machine code the first time
# it runs, and caches the compiled version on disk. Compiled functions are 10-100x faster for loops,
# which is why the inner loops of the model are written as plain loops over numbers. Comments and
# docstrings inside them are fine.
# ----------------------------------------------------------------------------------------------
@njit(cache=True)
def _u(c, s):
    """Utility of consumption c (model units per 5-year period): CRRA u(c) = c^(1-s) / (1-s),
    with s = sigma = 1.5 (so utility is negative and rises towards 0 as c grows).
    PAPER 3.1 (preferences)."""
    return c ** (1.0 - s) / (1.0 - s)


@njit(cache=True)
def _beq1(x, d, E, kap, phi1, sc, s, phi2, chi, phi2c, tau_b, theta, shield):
    """Charity g, net liquid bequest b, marginal value of liquid estate, value phi(b+theta*d)+psi(g).
    kap: taxable share of the liquid estate (1 - valuation discount). theta: weight the testator
    puts on trust assets relative to liquid bequests (lock-in valuation). shield=False: trust
    assets are included in the taxable estate (the 'no tax shield' experiment); b can then be
    negative, meaning part of the tax is paid out of the trust. Same arithmetic as
    EconomyB.bequest when d = 0 and kap = 1.

    WHAT IT DOES
        A person dies holding liquid estate x and a trust d. They split x between charity g and the
        rest; the rest pays estate tax and what is left, b, goes to the children. The trust passes
        to the children outside the estate (when shield is True). This function finds the best g
        and returns the utility the dying person gets from leaving (b, d, g).
    PAPER 3.2, bequest motive:
        value = phi1 * [(1 + (b + theta*d)/phi2)^(1-sigma) - 1] + chi * [1 - (1 + g/phi2c)^(1-sigma)],
        both terms multiplied by sc = ebar^(1-sigma) (rescaling, because phi2, phi2c are measured
        in units of mean 5-year earnings). phi1 < 0 and 1 - sigma < 0, so the bequest term is
        positive and rising in b. Estate tax: tau_b * max(kap * (x - g) + (trust if no shield) - E, 0).
    INPUTS (all money in model units)
        x      liquid estate (the saving a' of the person who dies)
        d      trust assets passed on (d' after this period's choice)
        E      exemption still available (E, or max(E - d, 0) for a founder: see module docstring)
        kap    taxable share of the liquid estate (1, or 1 - disc_v in tier 2 business states)
        phi1, chi       bequest and charity strengths;  phi2, phi2c  their luxury shifters
        sc, s           rescaling factor and sigma;  tau_b  estate tax rate (40%)
        theta           trust_weight;  shield   trust outside the taxable estate?
    OUTPUTS (g, b, mar, val)
        g    charity;  b  net bequest to children after estate tax (can be negative if no shield)
        mar  marginal value of one more unit of liquid estate (d val / d x), used by EGM
        val  the bequest + charity utility
    """
    dt = 0.0 if shield else d       # trust assets counted in the taxable estate (only if no shield)
    dv = theta * d                  # trust assets as valued by the testator (theta = trust_weight)

    # PYTHON NOTE: `def` inside a function defines small helper functions that can see the outer
    # function's variables (tau_b, kap, dt, E, ...).
    def bnet(z):
        # net liquid bequest after estate tax when z goes to the estate (after charity).
        # Taxable estate = kap * z (+ trust if no shield); tax = tau_b on the part above E.
        return z - tau_b * max(kap * z + dt - E, 0.0)

    def bder(z):
        # d bnet / d z: one more dollar of estate gives 1 - tau_b * kap net when above the exemption
        return (1.0 - tau_b * kap) if kap * z + dt > E else 1.0

    def phid(b):
        # marginal utility of the bequest term, d/db of sc * phi1 * [(1 + b/phi2)^(1-s) - 1]
        return sc * phi1 * (1.0 - s) / phi2 * (1.0 + b / phi2) ** (-s)

    def psid(g):
        # marginal utility of charity, d/dg of sc * chi * [1 - (1 + g/phi2c)^(1-s)]
        return sc * chi * (s - 1.0) / phi2c * (1.0 + g / phi2c) ** (-s)

    g = 0.0
    if chi > 0.0:
        # Charity is positive only if the first dollar to charity is worth more than the first
        # dollar of net bequest (marginal bequest utility times the after-tax share).
        if psid(0.0) > phid(bnet(x) + dv) * bder(x):
            # NUMERICS: bisection. The best g solves psid(g) = phid(bnet(x - g) + dv) * bder(x - g).
            # Start with the interval [0, x], halve it 60 times, each time keeping the half where
            # the sign change is. 60 halvings shrink the interval by 2^60, i.e. to machine precision.
            lo = 0.0; hi = x
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                f = psid(mid) - phid(bnet(x - mid) + dv) * bder(x - mid)
                if f > 0:
                    lo = mid        # charity still more valuable at the margin: give more
                else:
                    hi = mid        # bequest more valuable: give less
            g = 0.5 * (lo + hi)
    zz = x - g                      # liquid estate left after charity
    b = bnet(zz)                    # net liquid bequest after estate tax
    mar = phid(b + dv) * bder(zz)   # marginal value of liquid estate (envelope: charity is optimised)
    # Bequest utility. The max(..., -0.999999 * phi2) guard keeps 1 + (b + dv)/phi2 positive, which
    # only matters in the no-shield case where b can be negative.
    val = sc * phi1 * ((1.0 + max(b + dv, -0.999999 * phi2) / phi2) ** (1.0 - s) - 1.0)
    if chi > 0.0:
        val += sc * chi * (1.0 - (1.0 + g / phi2c) ** (1.0 - s))     # plus the charity warm glow
    return g, b, mar, val


@njit(cache=True)
def _lot(grid, x):
    """Lottery weights of a value x on an increasing grid.

    NUMERICS: the model keeps wealth on a fixed grid of points. When a household ends up with a value
    x that lies between two grid points grid[j] < x < grid[j+1], we do not round it; we split the
    household's population mass between the two points so that the average is exactly x:
    weight w on grid[j] and 1 - w on grid[j+1], with w = (grid[j+1] - x) / (grid[j+1] - grid[j]).
    This "lottery" keeps mean wealth exact. Used for the distribution, not for values.
    APPROXIMATION: values above the top grid point are put on the top point; values below grid[0]
    put all the mass on grid[0] (w is clipped to [0, 1]).
    INPUTS  grid: 1-D array of increasing points; x: a number.
    OUTPUT  (j, w): the lower point's index j (0 .. n-2) and the weight w on grid[j].
    """
    n = grid.size
    if x > grid[n - 1]:
        x = grid[n - 1]             # cap at the top of the grid
    # NUMERICS: binary search for j with grid[j] <= x < grid[j+1]: repeatedly halve the index
    # range [lo, hi] until the two ends are neighbours (about log2(n) steps instead of n).
    lo = 0; hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2        # PYTHON NOTE: `//` is division rounded down to a whole number
        if grid[mid] <= x:
            lo = mid
        else:
            hi = mid
    j = lo
    if x >= grid[n - 1]:
        j = n - 2                   # at the top: use the last interval (all weight goes to grid[n-1])
    w = (grid[j + 1] - x) / (grid[j + 1] - grid[j])     # weight on the lower point
    if w < 0.0:
        w = 0.0
    if w > 1.0:
        w = 1.0
    return j, w


@njit(cache=True)
def _vext(v_top, c_top, sl, dm, s):
    """Value beyond the top of the EGM grid: integrate the envelope condition dV/dm = u'(c(m))
    along the linearly extrapolated consumption function (bounded for sigma > 1).

    When cash on hand m is dm above the highest EGM point (value v_top, consumption c_top), we
    assume consumption keeps rising with slope sl: c(m) = c_top + sl * dm. Since one more unit of
    cash is worth u'(c) = c^(-s), V(m) = v_top + integral of c^(-s) dm
                                       = v_top + (c^(1-s) - c_top^(1-s)) / ((1-s) * sl).
    If the slope is (almost) zero the formula breaks down and we return v_top (APPROXIMATION).
    Units: model units; the value is in utils.
    """
    if sl <= 1e-12:
        return v_top
    c = c_top + sl * dm             # extrapolated consumption at m = m_top + dm
    return v_top + (c ** (1.0 - s) - c_top ** (1.0 - s)) / ((1.0 - s) * sl)


@njit(cache=True)
def _q(mE, cE, vE, W0, mono, m, s):
    """Value and consumption at cash on hand m for one (category, trust column, exog state).
    Monotone case: exactly EconomyB.cfun. Otherwise: upper envelope over EGM segments.

    NUMERICS, EGM (endogenous grid method) in plain words. The usual way to solve a saving problem
    is: for each cash on hand m, search for the best saving a'. EGM turns this around: fix a grid
    of SAVINGS a' (the liquid-wealth grid), and for each a' use the first-order condition
    u'(c) = beta * E[marginal value of a'] to get the consumption c that makes a' optimal; then
    the cash on hand at which this happens is m = c + a'. No search is needed. The result is a
    list of points (mE, cE, vE): "at cash on hand mE, consume cE and get value vE". Because the
    m points are found rather than chosen, they are an "endogenous grid". This function reads the
    solution at any m by interpolating between those points. (The arrays are built in
    EconomyC.solve, later in this file.)

    NUMERICS, upper envelope. With discrete choices next period (found a trust or not, etc.), the
    value of saving can have kinks and the first-order condition can have several solutions. Then
    the mE points are not increasing in a' (the list "folds back"), and at one m several segments
    give candidate answers. The correct answer is the one with the highest value: the "upper
    envelope" of the candidates (DC-EGM, Iskhakov et al. 2017).

    INPUTS  mE, cE, vE: 1-D arrays, one point per liquid-saving grid point a' (model units; utils).
            W0: continuation value when saving a' = 0 (used when the borrowing limit binds).
            mono: True if mE is increasing (no fold-back). m: cash on hand. s: sigma.
    OUTPUT  (value, consumption) at m.
    """
    n = mE.size
    if m <= mE[0]:
        # below the first EGM point the household would like to borrow but cannot: it consumes
        # everything (c = m, a' = 0) and gets u(m) plus the value of saving nothing.
        return _u(m, s) + W0, m
    if mono:
        if m > mE[n - 1]:
            # extrapolate with the last segment of non-negligible width
            kk = n - 1
            while kk > 1 and mE[kk] - mE[kk - 1] < 1e-8 * mE[kk]:
                kk -= 1             # skip segments whose two ends (almost) coincide
            sl = (cE[kk] - cE[kk - 1]) / (mE[kk] - mE[kk - 1])      # slope of consumption in m
            if sl < 0.0:
                sl = 0.0
            return _vext(vE[n - 1], cE[n - 1], sl, m - mE[n - 1], s), cE[n - 1] + sl * (m - mE[n - 1])
        # binary search for the segment [mE[lo], mE[lo+1]] that contains m
        lo = 0; hi = n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if mE[mid] <= m:
                lo = mid
            else:
                hi = mid
        if m == mE[n - 1]:
            return vE[n - 1], cE[n - 1]
        # NUMERICS: linear interpolation of value and consumption on that segment
        sl = (cE[lo + 1] - cE[lo]) / (mE[lo + 1] - mE[lo])
        sv = (vE[lo + 1] - vE[lo]) / (mE[lo + 1] - mE[lo])
        return sv * (m - mE[lo]) + vE[lo], sl * (m - mE[lo]) + cE[lo]
    # non-monotone endogenous grid: upper envelope
    # Start from the "consume everything" candidate, then check every segment of the folded list
    # that covers m and keep the one with the highest interpolated value.
    bv = _u(m, s) + W0; bc = m
    for k in range(n - 1):
        m0 = mE[k]; m1 = mE[k + 1]
        a = min(m0, m1); b = max(m0, m1)
        if m >= a and m <= b and b > a:
            t = (m - m0) / (m1 - m0)                    # position of m along the segment (0..1)
            v = vE[k] + t * (vE[k + 1] - vE[k])
            if v > bv:
                bv = v; bc = cE[k] + t * (cE[k + 1] - cE[k])
    # beyond the last point: also consider the extrapolated last segment (if it is increasing)
    if m > mE[n - 1] and mE[n - 1] > mE[n - 2] * (1.0 + 1e-8):
        sl = max((cE[n - 1] - cE[n - 2]) / (mE[n - 1] - mE[n - 2]), 0.0)
        v = _vext(vE[n - 1], cE[n - 1], sl, m - mE[n - 1], s)
        if v > bv:
            bv = v; bc = cE[n - 1] + sl * (m - mE[n - 1])
    return bv, bc


@njit(cache=True)
def _envelope_all(mE, cE, vE, W0, mono, s):
    """Replace non-monotone EGM columns by their upper envelope evaluated at the sorted
    endogenous points (DC-EGM style), so later queries are a binary search.

    Doing the full upper-envelope scan in _q for every query would be slow, so this is done once
    per age: each folded column is replaced by a clean increasing list of points (the sorted m
    values, with the best value and consumption at each). Afterwards every column is monotone
    and _q takes its fast branch.
    INPUTS (changed in place)
        mE, cE, vE: 4-D arrays, one entry per (next category, trust column d', exog state ys,
                    liquid-saving point a'); shape (ncat, nd, nyz, na).
        W0:   (ncat, nd, nyz) value of saving nothing;  mono: (ncat, nd, nyz) True/False flags.
    """
    A, B, C, n = mE.shape           # PYTHON NOTE: .shape gives the size along each dimension
    for i in range(A):
        for j in range(B):
            for k in range(C):
                if mono[i, j, k]:
                    continue        # already increasing: nothing to do
                # PYTHON NOTE: mE[i, j, k] is the last dimension (one column); .copy() makes an
                # independent copy so the original stays intact while we overwrite mE below.
                m0 = mE[i, j, k].copy(); c0 = cE[i, j, k].copy(); v0 = vE[i, j, k].copy()
                ms = np.sort(m0)    # the same m points in increasing order
                # make them strictly increasing (tiny nudges) so later divisions never hit 0/0
                for q in range(1, n):
                    if ms[q] <= ms[q - 1] * (1.0 + 1e-10):
                        ms[q] = ms[q - 1] * (1.0 + 1e-10) + 1e-300
                # at each sorted point, the upper envelope of the original (folded) column
                for q in range(n):
                    v, c = _q(m0, c0, v0, W0[i, j, k], False, ms[q], s)
                    mE[i, j, k, q] = ms[q]; cE[i, j, k, q] = c; vE[i, j, k, q] = v
                mono[i, j, k] = True


@njit(cache=True)
def _shiftY(C, rows, agrid, sib):
    """Y[r] = child slice C (na, nyz) with liquid wealth shifted by agrid[rows[r]] / (1 + sib).

    Used by EconomyC.conv for inheritances: for each possible estate size agrid[rows[r]]
    (the net liquid bequest), move the child's whole distribution up by the child's share of it.
    sib = 0: only child gets everything; sib = 1: one sibling, the child gets half.
    INPUTS  C: child mass, one row per liquid-wealth point, one column per (y, z) state.
            rows: the estate grid indices with positive mass.  agrid: liquid-wealth grid.
    OUTPUT  Y: (len(rows), na, nyz): one shifted copy of C per estate size, mass lotteried
            onto agrid.
    """
    na, nyz = C.shape
    Y = np.zeros((rows.size, na, nyz))
    # list the wealth rows of C that hold any mass (skip empty rows: speed only)
    nzi = np.empty(na, np.int64); nn = 0
    for i in range(na):
        tot = 0.0
        for q in range(nyz):
            tot += C[i, q]
        if tot > 0.0:
            nzi[nn] = i; nn += 1
    for r in range(rows.size):
        h = agrid[rows[r]] / (1.0 + sib)                # this child's share of the estate
        for ii in range(nn):
            i = nzi[ii]
            lo, wl = _lot(agrid, agrid[i] + h)          # own wealth + inheritance, onto the grid
            for q in range(nyz):
                Y[r, lo, q] += wl * C[i, q]
                Y[r, lo + 1, q] += (1.0 - wl) * C[i, q]
    return Y


@njit(cache=True)
def _qd(mE, cE, vE, W0, mono, dgrid, jcp, dp, ys, m, s):
    """Value and next liquid wealth a' when choosing trust d' (off grid) and category jcp, with
    cash on hand m left for consumption and liquid saving. Returns (v, a').

    The saving solution is only known on trust grid columns d_j. A trust choice dp usually lies
    between two columns d_j < dp < d_{j+1}. NUMERICS: we interpolate "along lines of constant total
    wealth": at the lower column the household holds less in the trust, so we give it the
    difference as extra cash (m_lo = m + (dp - d_j)); at the upper column it holds more in the
    trust, so we take the difference from cash (m_hi = m - (d_{j+1} - dp)). Total wealth m + dp is
    the same in both. Then value and saving are averaged with the lottery weights of dp.
    INPUTS  mE, cE, vE, W0, mono: the EGM solution (see _envelope_all for shapes).
            dgrid: trust grid. jcp: next-period category. dp: trust carried forward (model units).
            ys: exogenous state index. m: cash left for consumption + liquid saving.
    OUTPUT  (v, a'): value (utils) and liquid saving a' (model units).
    """
    nd = dgrid.size
    if dp <= 0.0:
        # no trust carried forward: use trust column 0 (d' = 0) directly
        v, c = _q(mE[jcp, 0, ys], cE[jcp, 0, ys], vE[jcp, 0, ys], W0[jcp, 0, ys], mono[jcp, 0, ys], m, s)
        if not (c > 1e-9 * m):          # guards (never bind in Layer B; off-grid corners only)
            c = 1e-9 * m                #   (written as `not (c > ...)` so it also catches NaN)
        if c > m:
            c = m                       # cannot consume more than cash on hand
        return v, m - c                 # saving a' = cash minus consumption
    j, wlo = _lot(dgrid, dp)            # dp lies between dgrid[j] and dgrid[j+1]; weight wlo on j
    om = 1.0 - wlo                      # weight on the upper column j + 1
    if om <= 0.0:
        # dp is exactly on column j
        v, c = _q(mE[jcp, j, ys], cE[jcp, j, ys], vE[jcp, j, ys], W0[jcp, j, ys], mono[jcp, j, ys], m, s)
        c = min(max(c, 1e-9 * m), m)
        return v, m - c
    if om >= 1.0:
        # dp is exactly on column j + 1 (or above the top of the trust grid)
        v, c = _q(mE[jcp, j + 1, ys], cE[jcp, j + 1, ys], vE[jcp, j + 1, ys], W0[jcp, j + 1, ys],
                  mono[jcp, j + 1, ys], m, s)
        c = min(max(c, 1e-9 * m), m)
        return v, m - c
    m_lo = m + (dp - dgrid[j])          # cash if the trust were d_j (extra trust money given as cash)
    m_hi = m - (dgrid[j + 1] - dp)      # cash if the trust were d_{j+1} (the extra taken from cash)
    if m_hi <= 0.0:
        m_lo = m; m_hi = m              # not enough cash to shift: plain interpolation instead
    v1, c1 = _q(mE[jcp, j, ys], cE[jcp, j, ys], vE[jcp, j, ys], W0[jcp, j, ys], mono[jcp, j, ys], m_lo, s)
    v2, c2 = _q(mE[jcp, j + 1, ys], cE[jcp, j + 1, ys], vE[jcp, j + 1, ys], W0[jcp, j + 1, ys],
                mono[jcp, j + 1, ys], m_hi, s)
    ap = wlo * (m_lo - c1) + om * (m_hi - c2)       # weighted liquid saving
    if ap > m * (1.0 - 1e-6):
        ap = m * (1.0 - 1e-6)           # keep a little consumption: saving below cash on hand
    if ap < 0.0:
        ap = 0.0                        # no borrowing
    return wlo * v1 + om * v2, ap


# ==============================================================================================
# NUMBA KERNELS, PART 2: THE OPTION MENU AND ONE PERIOD'S DECISIONS (PAPER 3.2)
# ==============================================================================================
# The household's discrete choices are stored as a list of "options". Each option n has:
#   ov[n] value (utils)            oa[n] liquid saving a'        od[n] trust carried forward d'
#   oj[n] next-period category     ot[n] taxes paid (gift tax)   oc[n] consumption
#   of[n] 1 if the option sits behind a planning friction (founding, upgrading), else 0
#   opr[n] choice probability (filled by _menu);  og[n] lifetime gift (filled by _decide)
# All money in model units, per 5-year period.
# ----------------------------------------------------------------------------------------------
@njit(cache=True)
def _add(ov, oa, od, oj, ot, oc, of, n, mE, cE, vE, W0, mono, dgrid, cp, dp, ys, me, s, tx, fr, vcap):
    """Append one option at position n of the option arrays and return n + 1.
    The option: move to category cp with trust dp, cash me left for consumption and liquid
    saving, tax tx already paid, friction flag fr (1 = behind a planning friction).
    Its value and saving come from _qd (the best consumption/saving given that choice).
    vcap: an upper bound on values (set later in this file from the bequest and charity bounds)."""
    v, ap = _qd(mE, cE, vE, W0, mono, dgrid, cp, dp, ys, me, s)
    if v > vcap:                    # values are bounded above by the bequest-utility bound; only
        v = vcap                    # reached off the grid (beyond-grid trust corners)
    # store the option; consumption = cash left minus liquid saving
    ov[n] = v; oa[n] = ap; od[n] = dp; oj[n] = cp; ot[n] = tx; oc[n] = me - ap; of[n] = fr
    return n + 1


@njit(cache=True)
def _contribs(ov, oa, od, oj, ot, oc, of, n, mE, cE, vE, W0, mono, dgrid, cp, D, ys, m, s,
              Erem, kap, tau_gift, fracs, mfloor, vcap, capE=False):
    """Contribution menu: the (discounted) remaining exemption, then fractions of cash on hand.

    Adds one option per contribution size x into the trust (which then holds D + x):
      q = 0: x = Erem / kap, exactly the remaining exemption (grossed up by the valuation discount
             kap, so the taxable value kap * x equals Erem and no gift tax is due);
      q >= 1: x = fracs[q-1] * m, a fraction of cash on hand (10%, 25%, ..., 85%).
    Gift tax: tau_gift on the taxable value above the remaining exemption, paid on top of the gift
    from the household's cash (tax-exclusive: 0.40 per dollar transferred = 28.6% of the outlay).
    Sizes that are not affordable (x >= m, or too little cash left) are skipped.
    INPUTS  cp: category after contributing; D: trust before contributing; m: cash on hand;
            Erem: remaining exemption; kap: taxable share; fracs: the menu; mfloor: minimum cash.
            capE: if True, sizes whose taxable value exceeds the remaining exemption are skipped
                  (the fund_cap_E switch; False in the main results).
    OUTPUT  the new number of options n.  PAPER 3.2 (trust options, gift tax).
    """
    nf = fracs.size
    for q in range(nf + 1):
        # PYTHON NOTE: `A if condition else B` picks A when the condition holds, otherwise B
        x = Erem / kap if q == 0 else fracs[q - 1] * m
        if x <= 0.0 or x >= m:
            continue                # PYTHON NOTE: `continue` skips to the next q
        if capE and kap * x > Erem * (1.0 + 1e-9):
            continue                # fund_cap_E: no funding above the remaining exemption
                                    # (the 1e-9 lets the exemption-sized option q = 0 through)
        tx = tau_gift * max(kap * x - Erem, 0.0)        # gift tax on the part above the exemption
        me = m - x - tx                                 # cash left after the gift and the tax
        if me <= mfloor:
            continue
        n = _add(ov, oa, od, oj, ot, oc, of, n, mE, cE, vE, W0, mono, dgrid, cp, D + x, ys, me, s, tx, 0, vcap)
    return n


@njit(cache=True)
def _menu(ov, oa, od, oj, ot, oc, of, opr, off, c, role, tier, biz, d, m, D, ys, mE, cE, vE, W0, mono, dgrid,
          E, setup1, setup2, tau_gift, kap_v, fracs, mu1, nu1, mu2, nu2, nu_d, trusts, tier2, s, windup,
          founder_payout, dmin, lock, catmap, vcap, mfloor, fslope, wref, pfr, capE):
    """Fill the option list (from index off) for one household with cash on hand m and trust value
    D, and their choice probabilities. Returns (n_end, V, mu_c, lam).

    PAPER 3.2, "Choice probabilities" (and the list of trust options).
    STEP 1, the menu, by category c:
      no trust (c = 0):  stay; or found a tier-1 trust (pay setup1, then contribute the exemption
                         or a fraction of cash). All founding options are behind the friction.
      heir with D < dmin: forced wind-up (trust paid out to the heir; becomes category 0).
      trust holders:     retain; take a payout (heirs; founders only if founder_payout); wind up
                         (heirs, only if windup); contribute more (becoming / staying a founder);
                         with tier 2 on: upgrade to tier 2 (behind a friction) or downgrade (free).
    STEP 2, probabilities (see the NUMERICS notes below): logit within the free options and within
      the friction options, then a logistic friction draw decides between the two groups.
    INPUTS (selected)
      off: first free slot in the option arrays. c, role, tier: category and its parts.
      biz: True in a business return state (for the tier-2 valuation discount).
      d: trust at the start of the period; D: trust after return, tax, fees; m: cash on hand.
      E: exemption; setup1, setup2: founding / upgrade costs; kap_v = 1 - disc_v.
      mu1, nu1: founding friction mean and scale (fric_mu, fric_nu); mu2, nu2: upgrade friction.
      nu_d: logit scale (share of wealth); lock: lock-in share; dmin: termination threshold.
      pfr: shares for partial-withdrawal options (payout_fracs; empty in the main results).
      capE: cap trust funding at the remaining exemption (fund_cap_E; False in the main results).
      All money in model units.
    OUTPUTS
      n_end: one past the last option filled; V: expected value of this menu (utils);
      mu_c: expected marginal utility of consumption over the options (for EGM, via Mb);
      lam: the money-metric scale (also reused for the gift choice in _decide).
    """
    n = off
    fmu = mu1; fnu = nu1            # the friction that applies (founding friction by default)
    if c == 0 and fslope != 0.0:
        # founding friction falling with wealth (robustness): share-of-wealth hurdle scales as
        # (W / wref)^(-fslope), W = cash on hand
        fmu = mu1 * max(m / wref, 1e-3) ** (-fslope)
    if c == 0:
        # option: stay without a trust (category 0, no trust, all cash available)
        n = _add(ov, oa, od, oj, ot, oc, of, n, mE, cE, vE, W0, mono, dgrid, 0, 0.0, ys, m, s, 0.0, 0, vcap)
        if trusts:
            # options: found a tier-1 trust (become category catmap[2, 1] = founder tier 1) with
            # cash m - setup1, the full exemption E, no valuation discount (kap = 1)
            n0 = n
            n = _contribs(ov, oa, od, oj, ot, oc, of, n, mE, cE, vE, W0, mono, dgrid, catmap[2, 1],
                          0.0, ys, m - setup1, s, E, 1.0, tau_gift, fracs, mfloor, vcap, capE)
            for q in range(n0, n):
                of[q] = 1           # all founding options are behind the planning friction
    elif role == 1 and D < dmin:
        # uneconomic small trust: terminated, assets paid to the heir (UTC 414)
        n = _add(ov, oa, od, oj, ot, oc, of, n, mE, cE, vE, W0, mono, dgrid, 0, 0.0, ys, m + D, s, 0.0, 0, vcap)
    else:
        fmu = mu2; fnu = nu2        # for trust holders the only friction option is the tier-2 upgrade
        # retain
        n = _add(ov, oa, od, oj, ot, oc, of, n, mE, cE, vE, W0, mono, dgrid, c, D, ys, m, s, 0.0, 0, vcap)
        # payout: heirs can reach income, and principal down to the lock-in share
        # lb = the lowest the trust may go. With lock = 1 and growth (D > d) that is d: the heir can
        # take this period's income D - d. With a loss (D < d) lb = D, so no payout is possible.
        # Founders (only with founder_payout) can take the growth above d.
        lb = lock * min(d, D) if role == 1 else min(d, D)
        if (role == 1 or founder_payout) and D > lb:
            n = _add(ov, oa, od, oj, ot, oc, of, n, mE, cE, vE, W0, mono, dgrid, c, lb, ys,
                     m + D - lb, s, 0.0, 0, vcap)
            # partial withdrawals (payout_fracs; none in the main results): take a share of the
            # drawable amount D - lb and leave the rest in the trust
            for fq in range(pfr.size):
                wdr = pfr[fq] * (D - lb)
                n = _add(ov, oa, od, oj, ot, oc, of, n, mE, cE, vE, W0, mono, dgrid, c, D - wdr, ys,
                         m + wdr, s, 0.0, 0, vcap)
        if role == 1 and windup:
            # wind-up (robustness only): all trust assets to the heir, back to category 0
            n = _add(ov, oa, od, oj, ot, oc, of, n, mE, cE, vE, W0, mono, dgrid, 0, 0.0, ys,
                     m + D, s, 0.0, 0, vcap)
        # remaining exemption: founders have used part of it (APPROXIMATION: max(E - d, 0) with the
        # current trust value); heirs bring their own full exemption E
        Erem = max(E - d, 0.0) if role == 2 else E
        kap = kap_v if (tier == 2 and biz) else 1.0     # valuation discount only in tier 2 business states
        # contributions: the household becomes (or stays) a founder in its current tier
        n = _contribs(ov, oa, od, oj, ot, oc, of, n, mE, cE, vE, W0, mono, dgrid, catmap[2, tier],
                      D, ys, m, s, Erem, kap, tau_gift, fracs, mfloor, vcap, capE)
        if tier2:
            if tier == 1:                              # upgrade (friction)
                if m - setup2 > mfloor:
                    n = _add(ov, oa, od, oj, ot, oc, of, n, mE, cE, vE, W0, mono, dgrid,
                             catmap[role, 2], D, ys, m - setup2, s, 0.0, 1, vcap)
            else:                                      # downgrade (free)
                n = _add(ov, oa, od, oj, ot, oc, of, n, mE, cE, vE, W0, mono, dgrid,
                         catmap[role, 1], D, ys, m, s, 0.0, 0, vcap)
    # choice probabilities: logit within the free options and within the options behind a
    # planning friction; the friction is a logistic draw (share of wealth). Money-metric scale:
    # marginal utility at the best free option times wealth.
    #
    # NUMERICS, money-metric logit in plain words. Utility numbers have no natural scale, so a
    # utility gap of 0.01 may be huge for a poor household and tiny for a rich one. We convert gaps
    # into money: near the best free option, one unit of cash is worth u'(c) utils, so a utility gap
    # dV is worth about dV / u'(c) units of cash, and dV / (u'(c) * (m + D)) as a SHARE OF TOTAL
    # WEALTH. lam = u'(c_best) * (m + D) is that conversion factor. Each option then gets a small
    # random taste shock (extreme-value, scale nu_d = 0.2% of wealth), which gives the logit formula
    #     prob(option q) = exp((V_q - V_max) / (lam * nu_d)) / sum over options of the same.
    # Options worth much less than 0.2% of wealth are almost never chosen; near-ties are shared.
    # Subtracting V_max before exp() avoids numerical overflow and does not change the result.
    qb = off
    for q in range(off + 1, n):
        if of[q] == 0 and ov[q] > ov[qb]:
            qb = q                  # the best free option (the first option is always free)
    lam = max(oc[qb] ** (-s) * (m + D), 1e-300)        # u'(c) at the best free option x total wealth
    vB = 0.0; vF = 0.0; nF = 0
    # grp 0 = free options, grp 1 = options behind the friction
    for grp in range(2):
        vmax = -1e300; cnt = 0
        for q in range(off, n):
            if of[q] == grp:
                vmax = max(vmax, ov[q]); cnt += 1
        if cnt == 0:
            continue
        tot = 0.0
        for q in range(off, n):
            if of[q] == grp:
                opr[q] = math.exp((ov[q] - vmax) / (lam * nu_d)); tot += opr[q]
        vg = 0.0
        for q in range(off, n):
            if of[q] == grp:
                # normalise to probabilities within the group, and accumulate the group's value.
                # Note: the group value is the probability-weighted average of the option values
                # (not the logit "log-sum" inclusive value).
                opr[q] /= tot; vg += opr[q] * ov[q]
        if grp == 0:
            vB = vg                 # value of the free branch
        else:
            vF = vg; nF = cnt       # value of the friction branch and its number of options
    if nF > 0:
        # NUMERICS / PAPER 3.2: the planning friction kappa is a random share of wealth with a
        # logistic distribution (mean fmu, scale fnu). The household takes the friction branch when
        # its gain in wealth-share terms, (vF - vB) / lam, exceeds kappa. The probability of that is
        # the logistic CDF:  pf = 1 / (1 + exp(-zz)),  zz = ((vF - vB) / lam - fmu) / fnu.
        zz = ((vF - vB) / lam - fmu) / fnu
        # the two algebraically equal forms avoid overflow of exp() for large |zz|
        if zz > 0:
            pf = 1.0 / (1.0 + math.exp(-zz))
        else:
            ez = math.exp(zz); pf = ez / (1.0 + ez)
        # the friction shifts the choice; the continuation value is the probability-weighted
        # value of the two branches (no option value from the friction draw)
        # APPROXIMATION: the friction is not subtracted from utility or resources; it only shifts
        # the probabilities.
        V = (1.0 - pf) * vB + pf * vF
        for q in range(off, n):
            opr[q] *= pf if of[q] == 1 else (1.0 - pf)  # within-group prob x prob of the group
    else:
        V = vB
    # expected marginal utility of consumption over all options (envelope condition for EGM)
    mu_c = 0.0
    for q in range(off, n):
        if opr[q] > 0.0:
            mu_c += opr[q] * oc[q] ** (-s)
    return n, V, mu_c, lam


# PYTHON NOTE: `parallel=True` lets Numba run the `prange` loop below on several cores at once;
# each iteration handles a different (category, liquid wealth) pair, so they do not interfere.
@njit(cache=True, parallel=True)
def _decide(mE, cE, vE, W0, mono, agrid, dgrid, inc, med, cfl, Rn, kbar, Gtc, Gpre, Gtr, wxi, nz, zof,
            E, setup1, setup2, Ffix, ffac, tauc, taungc, tau_gift, kap_v, fo_min, fo_rate, grat, Hg,
            fracs, mu1, nu1, mu2, nu2, nu_d, trusts, tier2, s, ko, ebar, windup, founder_payout, dmin,
            role_c, tier_c, catmap, vcap, lock, gift_on, gfracs, gift_phi, gift_excl, sc, phi2,
            Vb, Mb, P, AP, DP, JC, TX, FR, GF, fslope, wref, gift_ex, pfr, capE):
    """All decisions at one age for every category. Vb, Mb: (ncat, na, nd, nyz). Policies: flat
    state index (category 0 first, then one trust block per category). With gift_on, the
    household first chooses a lifetime gift g from the menu (0 and shares gfracs of cash on hand);
    its warm glow is sc*gift_phi*((1+g/phi2)^(1-s)-1); gifts above gift_excl use the remaining
    exemption, then pay tau_gift. The usual choices follow with the remaining cash.

    Called once per age t by EconomyC.solve, working backwards from the last age.
    PAPER 3.2 (the timing of one period). For every state (category c, liquid wealth a_i, trust
    d_j, exogenous state ys) and every return shock node k, in this order:
      1. returns and income: cash on hand m = after-tax return on liquid wealth + income;
         medical costs (if on) with a cash floor;
      2. trust growth: heirs' trust grows at the after-tax (tau_ng) return; founders' grows
         untaxed and the founder pays its income tax (grantor trust); fees are deducted;
      3. tier 2 only: family-office costs and the freeze (GRAT) transfer into the trust;
      4. age-55 gift stage (only when gift_on): choose a lifetime gift g to the child;
      5. the option menu (_menu) with the remaining cash;
      6. keep the ko most likely options for the distribution.
    Averaging over the shock nodes k gives the value and marginal value of the state.

    INPUTS (selected; money in model units, per 5-year period)
      mE, cE, vE, W0, mono: the EGM solution for next period's choices (see _envelope_all).
      agrid, dgrid: liquid and trust grids.  inc[ys]: income by exogenous state (this age).
      med[ys]: medical costs (0 if off);  cfl: cash-on-hand floor.
      Rn: after-tax gross market return.  kbar: business scale limit (none in Layer C).
      Gtc[c, z, k]: after-tax gross return on liquid wealth by category, return state, shock.
      Gpre[z, k], Gtr[z, k]: pre-tax gross return; pre-tax gross return on trust assets.
      wxi[k]: probabilities of the shock nodes.  nz: number of (composite) return states;
      zof[z]: return state (0 N, 1 L, 2 H) of composite state z.
      Ffix, ffac: fixed trustee fee per period; proportional fee factor (1 - fee)^5.
      tauc[c], taungc[c]: tax on returns (tau_a) / on non-grantor trust income (tau_ng) by category.
      fo_min, fo_rate, grat, Hg: tier-2 cost floor, cost rate, GRAT share, hurdle factor (1 + h)^5.
      ko: options to store.  ebar: mean 5-year earnings.  gift_on: gift stage at this age?
      gfracs: gift menu; gift_phi: gift warm-glow strength; gift_excl: tax-free gift amount.
    OUTPUTS (filled in place)
      Vb[c, i, j, ys]: expected value at the start of the period (before the return shock).
      Mb[c, i, j, ys]: expected marginal value of liquid wealth a (for next round of EGM).
      P, AP, DP, JC, TX, GF [st, k, r]: for the r-th most likely option at state st and shock k:
          probability, liquid saving a', trust d', next category, taxes paid, lifetime gift.
      FR[st, k]: the freeze (GRAT) transfer.
    Flat state index st: category 0 states first (st = i * nyz + ys, N0 = na * nyz of them), then
    one block of na * nd * nyz states for each trust category c >= 1, ordered (a_i, d_j, ys).
    Switches added after the first review (see ParamsC): gift_ex = 1.0, or 0.0 when gifts get no
    exemption (gift_exempt = False); pfr = payout_fracs; capE = fund_cap_E.
    """
    ncat = Vb.shape[0]
    na = agrid.size; nd = dgrid.size; nyz = inc.size; nx = wxi.size
    N0 = na * nyz                   # number of no-trust states
    ng = gfracs.size + 1 if gift_on else 1      # gift choices: none, plus one per gift fraction
    KM1 = 2 * fracs.size + 8 + pfr.size     # room for the options of one menu (a safe upper bound)
    KM = KM1 * ng                   # room for all menus (one menu per gift amount)
    mfloor = 1e-4 * ebar            # minimum cash on hand (0.01% of mean earnings)
    # PYTHON NOTE: `prange` = parallel loop. One iteration per (category c, liquid point i):
    # idx // na gives c and idx % na (the remainder) gives i.
    for idx in prange(ncat * na):
        c = idx // na; i = idx % na
        # scratch arrays for the options of this household (see the list above _add)
        # PYTHON NOTE: np.empty makes an array without setting its values; np.zeros sets them to 0
        ov = np.empty(KM); oa = np.empty(KM); od = np.empty(KM); oj = np.empty(KM, np.int64)
        ot = np.empty(KM); opr = np.empty(KM); oc = np.empty(KM); of = np.empty(KM, np.int64)
        og = np.zeros(KM)
        # per gift amount h: value gV, marginal utility gM, probability gP, option range gs..ge
        gV = np.empty(ng); gM = np.empty(ng); gP = np.empty(ng); gs = np.empty(ng, np.int64); ge = np.empty(ng, np.int64)
        role = role_c[c]; tier = tier_c[c]
        jmax = 1 if c == 0 else nd  # households without a trust have only the d = 0 column
        a = agrid[i]                # liquid wealth at the start of the period
        for j in range(jmax):
            d = dgrid[j]            # trust assets at the start of the period
            for ys in range(nyz):
                z = ys % nz         # return state part of ys (ys = y * nz + z)
                biz = zof[z] > 0    # entrepreneur (L or H) return state?
                if c == 0:
                    st = i * nyz + ys
                else:
                    st = N0 + (((c - 1) * na + i) * nd + j) * nyz + ys
                vacc = 0.0; macc = 0.0      # accumulators: expected value and marginal value
                for k in range(nx):         # loop over return shock nodes
                    # 1. cash on hand: market return on all liquid wealth plus the business excess
                    # return on wealth up to kbar (no limit in Layer C), plus income
                    m = Rn * a + (Gtc[c, z, k] - Rn) * min(a, kbar) + inc[ys]
                    dmda = Rn + (Gtc[c, z, k] - Rn) * (1.0 if a < kbar else 0.0)   # d m / d a
                    if med[ys] > 0.0:
                        # out-of-pocket medical costs, with a means-tested floor on cash on hand
                        # (Medicaid / SSI): below the floor, extra assets do not raise resources
                        m -= med[ys]
                        if m < cfl:
                            m = cfl; dmda = 0.0
                    D = 0.0; fr = 0.0       # trust after growth; freeze transfer
                    # 2. trust growth
                    if role == 1:
                        # heir: non-grantor trust pays tau_ng on its return (APPROXIMATION: losses are
                        # credited symmetrically), then the proportional fee ffac and the fixed fee
                        D = max(d * (1.0 + (Gtr[z, k] - 1.0) * (1.0 - taungc[c])) * ffac - Ffix, 0.0)
                    m0 = m                  # cash on hand before trust-related payments
                    if role == 2:
                        # grantor pays the trust's income tax from liquid wealth, up to half of cash
                        # on hand; beyond that the trust reimburses (reimbursement clause /
                        # grantor status toggled off)
                        D = max(d * Gtr[z, k] * ffac - Ffix, 0.0)  # trust grows untaxed, less fees
                        gt = tauc[c] * (Gtr[z, k] - 1.0) * d       # tax on the trust's income (tau_a)
                        # part paid from liquid wealth; a negative tax (a loss) is credited in full
                        gl = min(gt, 0.5 * m0) if gt > 0.0 else gt
                        m -= gl; D -= gt - gl                      # the rest is paid by the trust
                    if tier == 2:
                        # family-office running cost, shared by liquid and trust assets in proportion
                        # (liquid share capped at a quarter of cash on hand)
                        fo = max(fo_min, fo_rate * (a + d))
                        fl = min(fo * a / max(a + d, 1e-300), 0.25 * m0)
                        m -= fl; D -= fo - fl
                        if grat > 0.0:                         # freeze transfer (GRAT) into the trust
                            # the pre-tax return above the hurdle on the GRAT share of liquid wealth
                            fr = grat * a * max(Gpre[z, k] - Hg, 0.0)
                            fr = min(fr, 0.5 * m)              # at most half of cash on hand
                            m -= fr; D += fr
                    if D < 0.0:
                        m += D; D = 0.0     # costs the trust cannot cover fall on liquid wealth
                    if m < mfloor:                             # shortfall paid from the trust
                        # APPROXIMATION: if the trust cannot cover it either, m is still set to
                        # mfloor (a tiny amount, 0.01% of mean earnings, is created)
                        D = max(D - (mfloor - m), 0.0); m = mfloor
                    # lifetime gift stage (one menu per gift amount), then the usual choices
                    # (without gifts ng = 1 and this loop runs once with g = 0)
                    n = 0; ngv = 0; lam0 = 1.0
                    for gq in range(ng):
                        g = 0.0 if gq == 0 else gfracs[gq - 1] * m     # gift to the child
                        # remaining exemption (times gift_ex: 0 under the gift_exempt = False bound)
                        Erem_g = (max(E - d, 0.0) if role == 2 else E) * gift_ex
                        # gift tax: tau_gift on the gift above the tax-free amount and the exemption
                        # (no valuation discount on gifts here). Note: the exemption used by the gift
                        # is not deducted from E in the menu below or later; see the reply notes.
                        tx_g = tau_gift * max(g - gift_excl - Erem_g, 0.0) if gq > 0 else 0.0
                        mg = m - g - tx_g                              # cash left after the gift
                        if gq > 0 and mg <= mfloor:
                            continue
                        s0 = n              # this gift's options start at index s0
                        n, Vg, mug, lamg = _menu(ov, oa, od, oj, ot, oc, of, opr, n, c, role, tier, biz, d, mg, D, ys,
                                                 mE, cE, vE, W0, mono, dgrid, E, setup1, setup2, tau_gift, kap_v,
                                                 fracs, mu1, nu1, mu2, nu2, nu_d, trusts, tier2, s, windup,
                                                 founder_payout, dmin, lock, catmap, vcap, mfloor, fslope, wref,
                                                 pfr, capE)
                        for q in range(s0, n):
                            og[q] = g; ot[q] += tx_g        # record the gift and add its tax
                        if gq > 0:
                            # PAPER 3.2: gift warm glow, same form as the bequest motive with phi_g
                            Vg += sc * gift_phi * ((1.0 + g / phi2) ** (1.0 - s) - 1.0)
                        else:
                            lam0 = lamg     # money-metric scale of the no-gift menu, used below
                        gV[ngv] = Vg; gM[ngv] = mug; gs[ngv] = s0; ge[ngv] = n; ngv += 1
                    if ngv == 1:
                        V = gV[0]; mu_c = gM[0]     # one gift option only (or gifts off)
                    else:
                        # logit over the gift amounts, with the same money-metric scale as _menu
                        # (no friction on gifts)
                        vmax = -1e300
                        for h in range(ngv):
                            vmax = max(vmax, gV[h])
                        tot = 0.0
                        for h in range(ngv):
                            gP[h] = math.exp((gV[h] - vmax) / (lam0 * nu_d)); tot += gP[h]
                        V = 0.0; mu_c = 0.0
                        for h in range(ngv):
                            gP[h] /= tot; V += gP[h] * gV[h]; mu_c += gP[h] * gM[h]
                            for q in range(gs[h], ge[h]):
                                opr[q] *= gP[h]     # option prob = prob(gift) x prob(option | gift)
                    # expectation over the return shock; marginal value of a = dm/da x E[u'(c)]
                    # (the envelope condition)
                    vacc += wxi[k] * V
                    macc += wxi[k] * dmda * mu_c
                    FR[st, k] = fr
                    # keep the ko most likely options
                    # APPROXIMATION: the other options are dropped and the kept probabilities are
                    # rescaled to sum to one (v2c keeps ko = 2 per state and shock node).
                    kept = 0.0
                    for r in range(ko):
                        # find the most likely option not yet taken
                        best = -1; bp = 0.0
                        for q in range(n):
                            if opr[q] > bp:
                                bp = opr[q]; best = q
                        if best < 0:
                            # fewer than ko options with positive probability: empty slot
                            P[st, k, r] = 0.0; AP[st, k, r] = 0.0; DP[st, k, r] = 0.0
                            JC[st, k, r] = 0; TX[st, k, r] = 0.0; GF[st, k, r] = 0.0
                            continue
                        P[st, k, r] = bp; AP[st, k, r] = oa[best]; DP[st, k, r] = od[best]
                        JC[st, k, r] = oj[best]; TX[st, k, r] = ot[best]; GF[st, k, r] = og[best]
                        kept += bp
                        opr[best] = -1.0    # mark as taken
                    for r in range(ko):
                        P[st, k, r] /= kept     # rescale the kept probabilities to sum to one
                Vb[c, i, j, ys] = vacc
                Mb[c, i, j, ys] = macc


# ==============================================================================================
# NUMBA KERNELS, PART 3: MOVING THE POPULATION (DISTRIBUTION) AND INHERITANCES
# ==============================================================================================
# The population is a vector v of mass (shares of the population), one entry per flat state st
# (see _decide). These kernels move mass through the stored choices (P, AP, DP, JC), using
# lotteries (_lot) to put it on the grids, and build the estates of those who die.
# ----------------------------------------------------------------------------------------------
@njit(cache=True)
def _estates_fill(v, P, AP, DP, JC, BN, GC, done, phi1_s, zof, nz, nyz, E, kap_v, role_c, tier_c,
                  sc, s, phi2, chi, phi2c, tau_b, theta, shield):
    """Net bequest and charity for the stored options of every state with mass in v (lazy, cached).

    For each state with mass that has not been done before (flag `done`), and each stored option,
    compute what a person who dies after making that choice leaves: charity GC and net liquid
    bequest BN (after estate tax), via _beq1 with liquid estate a' = AP and trust d' = DP.
    "Lazy": only states that actually hold mass are computed, and each only once.
    INPUTS  v: mass by flat state; P, AP, DP, JC: stored options (see _decide); phi1_s[z]: bequest
            strength by composite return state; other inputs as in _beq1.
    OUTPUTS (filled in place) BN[st, k, r], GC[st, k, r]; done[st] set to True.
    PAPER 3.2 (bequests, estate tax).
    """
    Ns, nx, ko = P.shape
    for st in range(Ns):
        if v[st] <= 0.0 or done[st]:
            continue
        done[st] = True
        z = (st % nyz) % nz         # composite return state of this flat state
        for k in range(nx):
            for r in range(ko):
                if P[st, k, r] <= 0.0:
                    continue
                d = DP[st, k, r]; cp = JC[st, k, r]
                # founders: exemption used by the trust. Without the shield ("noshield") the trust is
                # in the taxable estate, so the full exemption applies (fixed after the first review)
                Ee = max(E - d, 0.0) if (role_c[cp] == 2 and shield) else E
                kap = kap_v if (tier_c[cp] == 2 and zof[z] > 0) else 1.0
                # PYTHON NOTE: `_` is a name for results we do not need
                g, b, _, _ = _beq1(AP[st, k, r], d, Ee, kap, phi1_s[z], sc, s, phi2, chi, phi2c, tau_b, theta, shield)
                BN[st, k, r] = b; GC[st, k, r] = g


@njit(cache=True, parallel=True)
def _push(v, P, AP, DP, JC, wxi, agrid, dgrid, na, nd, nyz, nch):
    """Move mass through return shocks and choices (before the exogenous (y, z) transition).
    nch chunks of states are processed in parallel, each into its own buffer.

    For each state st with mass, each shock node k and each stored option r, the mass
    v[st] * wxi[k] * P[st, k, r] moves to next period's (category JC, a' = AP, d' = DP), split by
    lotteries: 2 grid points if no trust (a' only), 4 points if a trust (a' and d'). The exogenous
    state ys is unchanged here; EconomyC.step then applies the (y, z) transition matrix.
    NUMERICS: each parallel chunk writes into its own buffer row, so two cores never add to the
    same number at the same time; the buffers are summed at the end.
    OUTPUT  next-period mass by flat state (before the (y, z) transition).
    """
    N0 = na * nyz
    Ns, nx, ko = P.shape
    buf = np.zeros((nch, v.size))
    step = (Ns + nch - 1) // nch    # states per chunk (rounded up)
    for ch in prange(nch):
        out = buf[ch]
        for st in range(ch * step, min(Ns, (ch + 1) * step)):
            mass = v[st]
            if mass <= 0.0:
                continue
            ys = st % nyz
            for k in range(nx):
                for r in range(ko):
                    pr = P[st, k, r]
                    if pr <= 0.0:
                        continue
                    mw = mass * wxi[k] * pr             # mass taking option r after shock k
                    ia, wa = _lot(agrid, AP[st, k, r])  # liquid saving onto the a grid
                    jcp = JC[st, k, r]
                    if jcp == 0:
                        # no trust next period: two liquid grid points
                        out[ia * nyz + ys] += mw * wa
                        out[(ia + 1) * nyz + ys] += mw * (1.0 - wa)
                    else:
                        # trust next period: four points (2 liquid x 2 trust) in category jcp's block
                        jd, wd = _lot(dgrid, DP[st, k, r])
                        base = N0 + (jcp - 1) * na * nd * nyz
                        out[base + (ia * nd + jd) * nyz + ys] += mw * wa * wd
                        out[base + (ia * nd + jd + 1) * nyz + ys] += mw * wa * (1.0 - wd)
                        out[base + ((ia + 1) * nd + jd) * nyz + ys] += mw * (1.0 - wa) * wd
                        out[base + ((ia + 1) * nd + jd + 1) * nyz + ys] += mw * (1.0 - wa) * (1.0 - wd)
    # add up the chunk buffers
    res = buf[0].copy()
    for ch in range(1, nch):
        res += buf[ch]
    return res


@njit(cache=True)
def _conv_trust_children(CT, H0, H1, Ta_lo, Ta_w, Td_lo, Td_w, q2, role_c, tier_c, catmap, outT):
    """Inheritance for children who already hold a trust (categories >= 1): brute force over the
    child's nonzero states and the nonzero estate cells, with precomputed lottery tables
    (child trusts are sparse, so this beats the factorised BLAS path for them).

    PAPER 3.2 (inheritance). When a parent dies, the child receives the net liquid bequest and the
    trust, divided by the number of children: with probability 1 - q2 the child is an only child
    (sib = 0, gets all), with probability q2 it has one sibling (sib = 1, gets half).
    INPUTS
      CT: (ncat-1, na, nd, nyz) mass of children with a trust, by category, liquid, trust, (y, z).
      H0[jb]: mass of dying parents leaving net liquid bequest agrid[jb] and no trust.
      H1[t, jb, jd]: mass of dying parents leaving agrid[jb] liquid and trust dgrid[jd] of tier t + 1.
      Ta_lo, Ta_w, Td_lo, Td_w: lottery tables from _lot_tables.  q2: probability of a sibling.
    OUTPUT (added in place) outT: same layout as CT. An inherited trust is merged into the child's
      own trust; the child keeps its role and takes the higher tier (module docstring).
    """
    ncm1, na, nd, nyz = CT.shape
    # list the estate cells with positive mass (b0: liquid-only estates)
    nb0 = 0
    for jb in range(na):
        if H0[jb] > 0.0:
            nb0 += 1
    b0 = np.empty(nb0, np.int64); q = 0
    for jb in range(na):
        if H0[jb] > 0.0:
            b0[q] = jb; q += 1
    # ... and (et, eb, ed): estates passing a trust (tier, liquid index, trust index)
    nb1 = 0
    for t in range(2):
        for jb in range(na):
            for jd in range(nd):
                if H1[t, jb, jd] > 0.0:
                    nb1 += 1
    et = np.empty(nb1, np.int64); eb = np.empty(nb1, np.int64); ed = np.empty(nb1, np.int64)
    q = 0
    for t in range(2):
        for jb in range(na):
            for jd in range(nd):
                if H1[t, jb, jd] > 0.0:
                    et[q] = t; eb[q] = jb; ed[q] = jd; q += 1
    for cm in range(ncm1):
        cc = cm + 1                 # the child's category (cm indexes the trust blocks from 0)
        for i in range(na):
            for jc in range(nd):
                for ys in range(nyz):
                    mc = CT[cm, i, jc, ys]
                    if mc <= 0.0:
                        continue
                    for sib in range(2):
                        pr = (1.0 - q2) if sib == 0 else q2
                        # liquid-only estates: the child's liquid wealth rises, trust unchanged
                        for qq in range(nb0):
                            jb = b0[qq]
                            w0 = mc * H0[jb] * pr
                            lo = Ta_lo[i, jb, sib]; wl = Ta_w[i, jb, sib]
                            outT[cm, lo, jc, ys] += w0 * wl
                            outT[cm, lo + 1, jc, ys] += w0 * (1.0 - wl)
                        # estates with a trust: both assets rise; tier = the higher of the two
                        for qq in range(nb1):
                            t = et[qq]; jb = eb[qq]; jd = ed[qq]
                            tgt = catmap[role_c[cc], max(tier_c[cc], t + 1)] - 1
                            w0 = mc * H1[t, jb, jd] * pr
                            lo = Ta_lo[i, jb, sib]; wl = Ta_w[i, jb, sib]
                            ld = Td_lo[jc, jd, sib]; wd = Td_w[jc, jd, sib]
                            outT[tgt, lo, ld, ys] += w0 * wl * wd
                            outT[tgt, lo, ld + 1, ys] += w0 * wl * (1.0 - wd)
                            outT[tgt, lo + 1, ld, ys] += w0 * (1.0 - wl) * wd
                            outT[tgt, lo + 1, ld + 1, ys] += w0 * (1.0 - wl) * (1.0 - wd)


@njit(cache=True)
def _conv_c0_children(C0, H0, H1, Ta_lo, Ta_w, Td_lo, Td_w, q2, catmap, out0, outT):
    """Inheritance for children without a trust (category 0), brute force over the child's nonzero
    states and the nonzero estate cells (cheap when both are sparse, e.g. large-gift pairs).

    As _conv_trust_children, for children in category 0 (C0: (na, nyz) mass). A liquid-only estate
    keeps the child in category 0 (out0); an estate with a trust of tier t + 1 makes the child an
    heir of that tier (catmap[1, t + 1]) holding its share of the trust (outT). PAPER 3.2.
    """
    na, nyz = C0.shape
    nd = H1.shape[2]
    for i in range(na):
        for ys in range(nyz):
            mc = C0[i, ys]
            if mc <= 0.0:
                continue
            for sib in range(2):
                pr = (1.0 - q2) if sib == 0 else q2
                for jb in range(na):
                    h = H0[jb]
                    if h > 0.0:
                        w0 = mc * h * pr
                        lo = Ta_lo[i, jb, sib]; wl = Ta_w[i, jb, sib]
                        out0[lo, ys] += w0 * wl
                        out0[lo + 1, ys] += w0 * (1.0 - wl)
                for t in range(2):
                    tgt = catmap[1, t + 1] - 1          # heir of tier t + 1 (index into outT)
                    for jb in range(na):
                        for jd in range(nd):
                            h = H1[t, jb, jd]
                            if h <= 0.0:
                                continue
                            w0 = mc * h * pr
                            lo = Ta_lo[i, jb, sib]; wl = Ta_w[i, jb, sib]
                            # child's own trust is 0 (dgrid[0]), so use row 0 of the trust table
                            ld = Td_lo[0, jd, sib]; wd = Td_w[0, jd, sib]
                            outT[tgt, lo, ld, ys] += w0 * wl * wd
                            outT[tgt, lo, ld + 1, ys] += w0 * wl * (1.0 - wd)
                            outT[tgt, lo + 1, ld, ys] += w0 * (1.0 - wl) * wd
                            outT[tgt, lo + 1, ld + 1, ys] += w0 * (1.0 - wl) * (1.0 - wd)


@njit(cache=True)
def _lot_tables(agrid, dgrid):
    """Precomputed lotteries for inheritances: own point i plus a 1/(1 + sib) share of point j.
    Ta_lo[i, j, sib], Ta_w[i, j, sib]: lower index and weight of agrid[i] + agrid[j] / (1 + sib)
    on agrid; Td_lo, Td_w: the same for the trust grid. sib = 0 (only child) or 1 (one sibling)."""
    na = agrid.size; nd = dgrid.size
    Ta_lo = np.empty((na, na, 2), np.int64); Ta_w = np.empty((na, na, 2))
    Td_lo = np.empty((nd, nd, 2), np.int64); Td_w = np.empty((nd, nd, 2))
    for sib in range(2):
        for i in range(na):
            for j in range(na):
                lo, w = _lot(agrid, agrid[i] + agrid[j] / (1.0 + sib))
                Ta_lo[i, j, sib] = lo; Ta_w[i, j, sib] = w
        for i in range(nd):
            for j in range(nd):
                lo, w = _lot(dgrid, dgrid[i] + dgrid[j] / (1.0 + sib))
                Td_lo[i, j, sib] = lo; Td_w[i, j, sib] = w
    return Ta_lo, Ta_w, Td_lo, Td_w


@njit(cache=True)
def _push_idx(v, idx, P, AP, DP, JC, wxi, agrid, dgrid, na, nd, nyz):
    """As _push, but only over the listed states (for sparse vectors).
    idx: the flat states with positive mass (a child's or a single pair's vector usually has few),
    so looping over them only is much faster than looping over all states."""
    N0 = na * nyz
    nx = P.shape[1]; ko = P.shape[2]
    out = np.zeros(v.size)
    for ii in range(idx.size):
        st = idx[ii]
        mass = v[st]
        ys = st % nyz
        for k in range(nx):
            for r in range(ko):
                pr = P[st, k, r]
                if pr <= 0.0:
                    continue
                mw = mass * wxi[k] * pr
                ia, wa = _lot(agrid, AP[st, k, r])
                jcp = JC[st, k, r]
                if jcp == 0:
                    out[ia * nyz + ys] += mw * wa
                    out[(ia + 1) * nyz + ys] += mw * (1.0 - wa)
                else:
                    jd, wd = _lot(dgrid, DP[st, k, r])
                    base = N0 + (jcp - 1) * na * nd * nyz
                    out[base + (ia * nd + jd) * nyz + ys] += mw * wa * wd
                    out[base + (ia * nd + jd + 1) * nyz + ys] += mw * wa * (1.0 - wd)
                    out[base + ((ia + 1) * nd + jd) * nyz + ys] += mw * (1.0 - wa) * wd
                    out[base + ((ia + 1) * nd + jd + 1) * nyz + ys] += mw * (1.0 - wa) * (1.0 - wd)
    return out


@njit(cache=True)
def _push_split(v, P, AP, DP, JC, GF, wxi, agrid, dgrid, ggrid, na, nd, nyz):
    """As _push, but returns one next-period vector per gift bin (lottery of each option's gift on
    ggrid), so that a child's gift stays paired with the giving parent's own path.

    Used at the age-55 gift stage (EconomyC.distribution, later in this file). OUTPUT out: (number of gift bins,
    number of states); row gb = the parents' next-period mass that gave (about) ggrid[gb].
    """
    N0 = na * nyz
    Ns, nx, ko = P.shape
    out = np.zeros((ggrid.size, v.size))
    for st in range(Ns):
        mass = v[st]
        if mass <= 0.0:
            continue
        ys = st % nyz
        for k in range(nx):
            for r in range(ko):
                pr = P[st, k, r]
                if pr <= 0.0:
                    continue
                mw0 = mass * wxi[k] * pr
                gb, wg = _lot(ggrid, GF[st, k, r])     # the gift onto the gift grid
                ia, wa = _lot(agrid, AP[st, k, r])
                jcp = JC[st, k, r]
                for h in range(2):                     # h = 0: lower gift bin, h = 1: upper
                    mw = mw0 * (wg if h == 0 else 1.0 - wg)
                    if mw <= 0.0:
                        continue
                    o = out[gb + h]     # PYTHON NOTE: o is a view: adding to o adds to that row
                    if jcp == 0:
                        o[ia * nyz + ys] += mw * wa
                        o[(ia + 1) * nyz + ys] += mw * (1.0 - wa)
                    else:
                        jd, wd = _lot(dgrid, DP[st, k, r])
                        base = N0 + (jcp - 1) * na * nd * nyz
                        o[base + (ia * nd + jd) * nyz + ys] += mw * wa * wd
                        o[base + (ia * nd + jd + 1) * nyz + ys] += mw * wa * (1.0 - wd)
                        o[base + ((ia + 1) * nd + jd) * nyz + ys] += mw * (1.0 - wa) * wd
                        o[base + ((ia + 1) * nd + jd + 1) * nyz + ys] += mw * (1.0 - wa) * (1.0 - wd)
    return out


@njit(cache=True)
def _estate_hist2(v, P, BN, DP, JC, wxi, agrid, dgrid, tier_c):
    """Estates of the dying mass v: H0 over the net liquid bequest (no trust passed on), H1[t]
    over (net liquid bequest, trust assets) for estates passing a trust of tier t + 1.

    v is the mass of parents who die this period (already multiplied by the death probability).
    Charity GC has left the family and is not passed on. OUTPUTS H0: (na,), H1: (2, na, nd),
    in population mass, lotteried onto the grids; they feed the _conv_* functions.
    """
    na = agrid.size; nd = dgrid.size
    H0 = np.zeros(na); H1 = np.zeros((2, na, nd))
    Ns, nx, ko = P.shape
    for st in range(Ns):
        mass = v[st]
        if mass <= 0.0:
            continue
        for k in range(nx):
            for r in range(ko):
                pr = P[st, k, r]
                if pr <= 0.0:
                    continue
                mw = mass * wxi[k] * pr
                bn = BN[st, k, r]; dd = DP[st, k, r]    # net liquid bequest; trust passed on
                if bn < 0.0:                    # no-shield case: tax beyond liquid paid by the trust
                    dd = max(dd + bn, 0.0); bn = 0.0
                jb, wb = _lot(agrid, bn)
                cp = JC[st, k, r]
                if cp == 0:
                    H0[jb] += mw * wb
                    H0[jb + 1] += mw * (1.0 - wb)
                else:
                    t = tier_c[cp] - 1          # 0 for tier 1, 1 for tier 2
                    jd, wd = _lot(dgrid, dd)
                    H1[t, jb, jd] += mw * wb * wd
                    H1[t, jb, jd + 1] += mw * wb * (1.0 - wd)
                    H1[t, jb + 1, jd] += mw * (1.0 - wb) * wd
                    H1[t, jb + 1, jd + 1] += mw * (1.0 - wb) * (1.0 - wd)
    return H0, H1


@njit(cache=True)
def _conv_all(cn, H0, H1, agrid, dgrid, nyz, q2, role_c, tier_c, catmap):
    """Brute-force inheritance convolution (reference implementation for tests).

    Does in one slow, simple loop what EconomyC.conv does fast: for every child state in
    cn (flat vector) and every estate cell in H0 / H1, add the child's share of the estate
    (all if only child, half with a sibling) to the child's wealth and trust, and lottery the result
    onto the grids. Used to check that the fast version gives the same answer.
    """
    na = agrid.size; nd = dgrid.size
    N0 = na * nyz; NT1 = na * nd * nyz      # size of the no-trust block; size of one trust block
    out = np.zeros(cn.size)
    for c in range(cn.size):        # here c is a flat state index (not a category)
        mc = cn[c]
        if mc <= 0.0:
            continue
        # decode the flat index into (liquid index i, (y,z) index rest, category cc, trust index jc_d)
        if c < N0:
            i = c // nyz; rest = c % nyz; cc = 0; jc_d = 0
        else:
            r_ = c - N0
            rest = r_ % nyz; r_ //= nyz     # PYTHON NOTE: `x //= n` means x = x // n
            jc_d = r_ % nd; r_ //= nd
            i = r_ % na; cc = r_ // na + 1
        ac = agrid[i]; dc = dgrid[jc_d] if cc > 0 else 0.0     # the child's own assets
        for sib in range(2):
            pr = (1.0 - q2) if sib == 0 else q2
            # liquid-only estates: category and trust unchanged
            for jb in range(na):
                if H0[jb] <= 0.0:
                    continue
                lo, wl = _lot(agrid, ac + agrid[jb] / (1.0 + sib))
                w0 = mc * H0[jb] * pr
                if cc == 0:
                    out[lo * nyz + rest] += w0 * wl
                    out[(lo + 1) * nyz + rest] += w0 * (1.0 - wl)
                else:
                    base = N0 + (cc - 1) * NT1
                    out[base + (lo * nd + jc_d) * nyz + rest] += w0 * wl
                    out[base + ((lo + 1) * nd + jc_d) * nyz + rest] += w0 * (1.0 - wl)
            # estates with a trust of tier t + 1: the child becomes an heir (if it had no trust) or
            # keeps its role with the higher tier (if it had one)
            for t in range(2):
                ncat = catmap[1, t + 1] if cc == 0 else catmap[role_c[cc], max(tier_c[cc], t + 1)]
                base = N0 + (ncat - 1) * NT1
                for jb in range(na):
                    for jd in range(nd):
                        h = H1[t, jb, jd]
                        if h <= 0.0:
                            continue
                        lo, wl = _lot(agrid, ac + agrid[jb] / (1.0 + sib))
                        ld, wd = _lot(dgrid, dc + dgrid[jd] / (1.0 + sib))
                        w0 = mc * h * pr
                        out[base + (lo * nd + ld) * nyz + rest] += w0 * wl * wd
                        out[base + (lo * nd + ld + 1) * nyz + rest] += w0 * wl * (1.0 - wd)
                        out[base + ((lo + 1) * nd + ld) * nyz + rest] += w0 * (1.0 - wl) * wd
                        out[base + ((lo + 1) * nd + ld + 1) * nyz + rest] += w0 * (1.0 - wl) * (1.0 - wd)
    return out


# ==============================================================================================
# LAYER C, PART 2: THE ECONOMY OBJECT (solve, distribution, moments) AND REPORTING HELPERS
# ==============================================================================================
# What this half of layerC.py does
#   It defines EconomyC, the full model with dynasty trusts. Building one takes four steps:
#   (1) __init__ sets up the grids, household categories, returns (incl. trust returns), fees and
#   medical costs; (2) solve() works backwards from the last age to the first and records, for every
#   state, what households choose (consumption/saving, found / keep / pay out / add to a trust,
#   lifetime gifts); (3) distribution() finds the stationary distribution of households across states,
#   generation after generation, including gifts and inheritances passed from parents to children;
#   (4) moments() turns that distribution into the numbers compared with data (top wealth shares,
#   trust shares, estate-tax rates, ...). The helpers at the bottom (dfa_top_shares, choice_shares,
#   founding_table, upgrade_table) produce extra tables for the paper.
#
# Where it sits
#   Layer A (life cycle: earnings, survival) -> Layer B (EconomyB: fortunes, entrepreneurial returns,
#   bequests; this class inherits from it) -> Layer C (this file: trusts, gifts). Drivers on top:
#   calibrate_C.py (calibration), equilibrium.py / run_c5.py (experiments), run_sweep.py, run_gridcheck.py.
#   The first half of this file holds ParamsC and the fast Numba "kernels" (_decide, _push,
#   _estate_hist2, _conv_*, _lot, ...) that the methods below call.
#
# Reading order
#   EconomyC.__init__ -> _bq_grid -> solve -> step -> conv (with _dlot_matrix) -> distribution
#   (with _anderson) -> moments (with _group_share / _group_ratio, rank_rank, wealth_elasticity)
#   -> buildC -> dfa_top_shares -> choice_shares / founding_table / upgrade_table / cap_diagnostics.
#
# Paper sections implemented here
#   PAPER: 3.2 (household problem, trust options, inheritance) via solve / distribution;
#   PAPER: 3.4 (calibration targets) via moments; PAPER: 3.5 (DFA measurement) via dfa_top_shares;
#   PAPER: Appendix A (computation: backward induction, EGM, distribution, Anderson acceleration).
#
# Words you'll see in this file
#   period      = 5 years. Age index t = 0 for ages 20-24, ..., 13 for ages 85-89 (T = 14 periods).
#   model units = the model's money unit. usd = dollars per model unit; ebar = mean 5-year earnings.
#   a, d        = liquid (personal) wealth; trust assets (at the start of a period).
#   D           = trust assets after this period's return, tax and fees (before the trust choice).
#   m           = cash on hand: liquid wealth after returns, plus income.
#   ys / nyz    = combined (earnings y, return state z) index / number of such combinations.
#   category c  = 0 no trust, 1 heir tier 1, 2 founder tier 1, 3 heir tier 2, 4 founder tier 2.
#   flat state  = one integer numbering every (category, a, d, ys) cell; see "flat state descriptors".
#   lottery     = putting a value that falls between two grid points onto those two points, with
#                 weights that keep the mean right (NUMERICS: the standard way to keep a distribution
#                 on a grid). Done by _lot.
#   pair        = (child vector, parent vector): the factorised way the parent-child joint distribution
#                 is stored (explained in distribution()).
#   xi / wxi    = return-shock nodes / their probabilities (Gauss-Hermite quadrature).
#   ko          = number of most-likely options kept per state and shock for the distribution.
# ==============================================================================================


# ==============================================================================================
# SECTION 1: THE ECONOMY CLASS - SET-UP (grids, categories, returns, fees, medical costs)
# ==============================================================================================
# ----------------------------------------------------------------------------------------------
# PYTHON NOTE: `class EconomyC(EconomyB)` means EconomyC "inherits" from Layer B's EconomyB: it starts
# with everything EconomyB has (earnings process, survival, prices, the liquid-wealth grid agrid,
# the after-tax returns Gt and Rn, ...) and adds or replaces methods for the trust model.
class EconomyC(EconomyB):
    """The Layer C economy: Layer B plus dynasty trusts (tier 1), optionally family offices (tier 2),
    and lifetime gifts. Typical use: EconomyC(p, eflife, surlife).solve().distribution().moments().
    PAPER: Section 3.2 (households, transfers, trusts) and Appendix A (computation)."""

    def __init__(self, p: ParamsC, eflife, surlife):
        """Build the grids and all fixed quantities for given parameters.
        Inputs: p = ParamsC (parameters, mostly in dollars or per-year rates); eflife = age profile of
        labour efficiency (per year of age); surlife = one-year survival probabilities by age.
        Everything stored on self is converted to MODEL UNITS and PER 5-YEAR PERIOD, unless the name
        says otherwise. Nothing is solved here."""
        # Run Layer B's set-up first (earnings, survival, prices, agrid, usd, returns, ...).
        # PYTHON NOTE: super() refers to the parent class (EconomyB); this calls its __init__.
        super().__init__(p, eflife, surlife)
        # PYTHON NOTE: `assert condition, "message"` stops the program with that message if the
        # condition is false. Here: the business scale limit kbar must be off (huge, >= $1e14).
        assert p.kbar_usd >= 1e14, "Layer C assumes no business scale limit (as in Layer B v2)"
        # parent-child pairs are tracked for two overlapping generations only: a parent must be dead
        # by the time the child has children of its own (T <= 2 * gap)
        # (with T = 14 and gap = 7 this holds exactly: a parent is at most 89 when the child is 55.)
        assert p.T <= 2 * p.gap, "T > 2*gap: three overlapping generations are not supported"
        # Number of household categories: 5 with family offices (tier 2) switched on, else 3
        # (0 no trust, 1 heir tier 1, 2 founder tier 1).
        self.ncat = 5 if p.tier2 else 3
        # role_c[c] (0 none, 1 heir, 2 founder) and tier_c[c] (0, 1, 2) for each category c in use.
        # PYTHON NOTE: ROLE[:self.ncat] is "slicing": the first ncat entries. .copy() makes an
        # independent array so later changes would not touch the global ROLE.
        # PYTHON NOTE: several statements on one line are separated by `;`.
        self.role_c = ROLE[:self.ncat].copy(); self.tier_c = TIER[:self.ncat].copy()
        # Trust-asset grid dgrid (model units): the point 0, then nd - 1 points spaced evenly in logs
        # from d_lo_usd ($250k) up to the top of the liquid grid (agrid[-1], $500bn).
        # PYTHON NOTE: np.geomspace(lo, hi, n) gives n points with a constant ratio between neighbours;
        # np.concatenate glues arrays end to end; agrid[-1] is the last element.
        self.dgrid = np.concatenate([[0.0], np.geomspace(p.d_lo_usd / self.usd, self.agrid[-1], p.nd - 1)])
        # UNITS: one-off legal cost of founding a trust (setup, $50k) and of moving to a family office
        # (setup2, $500k), converted from dollars to model units.
        self.setup = p.setup_usd / self.usd; self.setup2 = p.setup2_usd / self.usd
        # UNITS: fixed trustee fee per 5-year period = $10k a year x 5 years, in model units.
        self.Ffix = p.fee_fixed_usd * p.tlen / self.usd
        # Proportional trustee fee as a survival factor over 5 years: (1 - 0.2%)^5. Trust assets are
        # multiplied by ffac each period (see _decide).
        self.ffac = (1 - p.fee_prop) ** p.tlen
        # Tier 2 (family office) running cost: minimum per 5-year period (model units) ...
        self.fo_min = p.fo_min_usd * p.tlen / self.usd
        # ... and the per-dollar rate compounded to 5 years: 1 - (1 - 0.5%)^5 (share of assets).
        self.fo_rate = 1 - (1 - p.fo_rate) ** p.tlen
        # GRAT hurdle as a 5-year gross factor: (1 + IRC 7520 rate)^5. Returns above it pass to the trust.
        self.Hg = (1 + p.hurdle) ** p.tlen
        # Lifetime-gift grid, built in DOLLARS first. `big` = points from gift_pool_usd ($1M) to $50bn,
        # spaced by a ratio of about gift_grid_factor (3). The number of points is
        # ceil(log(50e9 / 1e6) / log 3) + 1 = 11 with the defaults.
        # PYTHON NOTE: int(...) turns the rounded-up float into a whole number for np.geomspace.
        big = np.geomspace(p.gift_pool_usd, 50e9, int(np.ceil(np.log(50e9 / p.gift_pool_usd) / np.log(p.gift_grid_factor))) + 1)
        # Full gift grid: $0, $30k, $100k, $300k, then the `big` points; divided by usd -> model units.
        self.ggrid = np.concatenate([[0.0, 30e3, 100e3, 300e3], big]) / self.usd
        # UNITS: the pooling threshold in model units. Gifts below it are pooled in distribution().
        self.gpool = p.gift_pool_usd / self.usd
        # Gt (from Layer B) is the AFTER-tax gross 5-year return, shape (composite return states, shock
        # nodes). Undo the 18% capital-income tax to get the PRE-tax gross return Gpre:
        # Gt - 1 = (G - 1)(1 - tau_a)  =>  G = 1 + (Gt - 1) / (1 - tau_a).
        self.Gpre = 1 + (self.Gt - 1) / (1 - p.tau_a)          # pre-tax gross return by composite state
        # Gtr = pre-tax gross return on TRUST assets. Equal to Gpre unless the robustness option
        # trust_prem (extra annual return on trust assets) is non-zero; then it is scaled by (1 + prem)^5.
        self.Gtr = self.Gpre * (1 + p.trust_prem) ** p.tlen     # pre-tax gross return on trust assets
        # after-tax gross return on liquid wealth by category (tax drag and behavioral gap by tier)
        # gapf[tier] = share of the behavioural return gap lam_b that applies: full in tier 0,
        # half in tier 1, none in tier 2 (see the part-1 banner). lam_b = 0 by default.
        # PYTHON NOTE: np.array([...]) makes a NumPy array (a fast vector of numbers) from a list.
        gapf = np.array([1.0, 0.5, 0.0])
        # tauc[c] = tax rate on liquid returns (and grantor-trust income) for category c: tau_a2
        # (10%) in tier 2, tau_a (18%) otherwise.
        # PYTHON NOTE: `[f(t) for t in seq]` is a "list comprehension": a compact loop building a list.
        self.tauc = np.array([p.tau_a2 if t == 2 else p.tau_a for t in self.tier_c])
        # taungc[c] = tax rate on NON-grantor trust income (heirs' trusts): tau_ng2 (13.3%) in tier 2,
        # tau_ng (24%) otherwise.
        self.taungc = np.array([p.tau_ng2 if t == 2 else p.tau_ng for t in self.tier_c])
        # Gtc[c, s, k] = after-tax gross 5-year return on LIQUID wealth for category c, composite return
        # state s and shock node k. Shape: (ncat,) + Gt.shape = (ncat, nz, nxi).
        # PYTHON NOTE: (self.ncat,) + self.Gt.shape joins two "tuples" of sizes into one shape.
        self.Gtc = np.zeros((self.ncat,) + self.Gt.shape)
        for c in range(self.ncat):
            t = self.tier_c[c]                                   # NOTE: t here is a tier, not an age
            if t < 2 and p.lam_b == 0.0:
                self.Gtc[c] = self.Gt                            # exactly Layer B
            else:
                # Pre-tax return reduced by the behavioural gap (per year, compounded over 5 years) ...
                Gl = self.Gpre * (1 - p.lam_b * gapf[t]) ** p.tlen
                # ... then taxed at this category's rate: after-tax gross = 1 + (gross - 1)(1 - tau).
                self.Gtc[c] = 1 + (Gl - 1) * (1 - self.tauc[c])
        # Sizes: na liquid grid points, nd trust grid points, nyz = (earnings states) x (composite
        # return states).
        na, nd, nyz = self.agrid.size, p.nd, p.ny * self.nz
        # Size of the flat state space:
        #   N0  = states of category 0 (no trust): one per (a, ys);
        #   NT1 = states of ONE trust category: one per (a, d, ys);
        #   N   = N0 + (ncat - 1) trust blocks. Flat order: category 0 first, then categories 1, 2, ...
        self.N0 = na * nyz; self.NT1 = na * nd * nyz; self.N = self.N0 + (self.ncat - 1) * self.NT1
        # sc = ebar^(1 - sigma): rescales the bequest / charity / gift utility terms, whose shifters
        # are expressed in units of ebar (see the brief's bequest-motive formula).
        self.sc = self.ebar ** (1 - p.sigma)
        # Upper bound on any value: with sigma > 1 and phi1 < 0 the bequest term lies in [0, -phi1*sc)
        # and the charity term in [0, chi*sc), while consumption utility is negative. Used by _add
        # to cap values that come from extrapolating beyond the grid.
        self.vcap = self.sc * (max(-np.min(self.phi1_s), 0.0) + p.chi)   # upper bound on values
        # late-life medical costs (per model period, model units) and the cash-on-hand floor
        nyz_ = p.ny * self.nz
        # med[t, ys] = out-of-pocket medical cost at age t for state ys (model units per 5-year period).
        # Zero unless med_on (off in the v2c calibration, which uses ages 20-89).
        self.med = np.zeros((p.T, nyz_))
        # UNITS: consumption floor per 5-year period = 5 x $2,663 (1998 $) x 1.80 (CPI to 2022) / usd.
        self.cfl = p.tlen * p.cfloor_usd98 * p.cpi_98 / self.usd
        if p.med_on:
            # Distribution over earnings states y at retirement age TL (summing over return states).
            # PYTHON NOTE: reshape(T, ny, nz) views the same numbers as a 3-D block (age, y, z);
            # .sum(axis=2) adds over z; [p.TL] picks the row for age TL.
            piy = self.pi_age.reshape(p.T, p.ny, self.nz).sum(axis=2)[p.TL]
            piy = piy / piy.sum()
            # rank[y] = mid-point percentile of earnings state y in that distribution (between 0 and 1).
            # PYTHON NOTE: np.argsort gives the order that sorts the array; np.cumsum is a running total.
            order = np.argsort(self.y)
            rank = np.empty(p.ny); cum = np.cumsum(piy[order]); rank[order] = cum - 0.5 * piy[order]
            # Medical-cost multiplier rising with permanent-income rank: exp(slope * rank), rescaled so
            # its population mean is one (De Nardi, French & Jones 2010).
            gy = np.exp(p.med_pi_slope * rank); gy = gy / (piy * gy).sum()      # mean one across types
            for t in range(p.T):
                # Mid-point age of period t, e.g. 22 for ages 20-24.
                age = p.age0 + p.tlen * t + 0.5 * (p.tlen - 1)
                if age < p.med_start_age:
                    continue                                     # no medical costs before age 65
                # UNITS: mean ANNUAL cost in 2022 dollars at this age: $1,100 (1998 $) x CPI at 75,
                # growing 10.6% per year of age.
                ann = p.med75_usd98 * p.cpi_98 * np.exp(p.med_growth * (age - 75.0))
                # Per 5-year period, by y type, in model units; np.repeat copies each y value nz times so
                # the vector follows the combined index ys = y * nz + z.
                self.med[t] = np.repeat(p.tlen * ann * gy / self.usd, self.nz)
        # flat state descriptors
        # For every flat state index, record its liquid wealth (fa), trust assets (fd), combined
        # exogenous state (fs), category (fcat), role, tier and return state (fz). These length-N
        # vectors let later code compute "wealth of each state" etc. without decoding indices.
        # Category-0 block: index = i * nyz + s  (i = liquid grid point, s = ys).
        # PYTHON NOTE: np.repeat([x0, x1], 2) = [x0, x0, x1, x1]; np.tile([x0, x1], 2) = [x0, x1, x0, x1].
        a0 = np.repeat(self.agrid, nyz); s0 = np.tile(np.arange(nyz), na)
        # Trust block: index within the block = (i * nd + j) * nyz + s (j = trust grid point).
        iaT = np.repeat(np.arange(na), nd * nyz); jdT = np.tile(np.repeat(np.arange(nd), nyz), na)
        sT = np.tile(np.arange(nyz), na * nd)
        nb = self.ncat - 1                                      # number of trust blocks
        # PYTHON NOTE: [x] * nb is a list holding x nb times; self.agrid[iaT] picks agrid at each index.
        self.fa = np.concatenate([a0] + [self.agrid[iaT]] * nb)
        self.fd = np.concatenate([np.zeros(self.N0)] + [self.dgrid[jdT]] * nb)
        self.fs = np.concatenate([s0] + [sT] * nb)
        self.fcat = np.concatenate([np.zeros(self.N0, int)] + [c * np.ones(self.NT1, int) for c in range(1, self.ncat)])
        self.fjc = self.fcat                                    # backward compatibility
        self.frole = self.role_c[self.fcat]; self.ftier = self.tier_c[self.fcat]
        # Return state of each flat state: fs % nz is the composite return state; zof maps it to
        # 0 = N normal, 1 = L entrepreneur, 2 = H high-growth entrepreneur.
        self.fz = self.zof[self.fs % self.nz]

    # ==========================================================================================
    # SECTION 2: SOLVING THE HOUSEHOLD PROBLEM (backward induction over ages)
    # ==========================================================================================
    def _bq_grid(self):
        """Bequest value and marginal value on the (a', d') grid, by next category and composite s.

        For every end-of-period position a household could die in - next category cp, liquid saving
        a' = agrid[i], trust assets d' = dgrid[j], composite return state z - compute the warm-glow
        value of the estate and its derivative with respect to a'. PAPER: the bequest motive in 3.2,
        phi1*[(1 + (b + theta*d)/phi2)^(1-sigma) - 1] + chi*[1 - (1 + g/phi2c)^(1-sigma)], with the
        estate tax and the optimal charity split worked out by _beq1.
        Returns two arrays (val, mar), each of shape (ncat, na, nd, nz): one block per next category,
        one row per liquid grid point, one column per trust grid point, last axis = return state.
        UNITS: utility units (already multiplied by sc)."""
        p = self.p; na, nd, nz = self.agrid.size, p.nd, self.nz
        val = np.zeros((self.ncat, na, nd, nz)); mar = np.zeros((self.ncat, na, nd, nz))
        # Taxable share of the liquid estate when a valuation discount applies (tier 2, business state).
        kv = 1 - p.disc_v
        for cp in range(self.ncat):
            for i in range(na):
                for j in range(nd):
                    # Category 0 holds no trust, so d' = 0 there whatever j is.
                    d = 0.0 if cp == 0 else self.dgrid[j]
                    # Remaining estate-tax exemption: a founder (role 2) has used part of it on the
                    # trust, approximated by the trust's current value; others have the full E.
                    # APPROXIMATION: counts trust growth as exemption used (see part-1 banner).
                    # Without the shield the trust is taxed in the estate, so the full E applies.
                    Ee = max(self.E - d, 0.0) if (self.role_c[cp] == 2 and p.shield) else self.E
                    for z in range(nz):
                        # Valuation discount only for tier-2 families in a business return state.
                        kap = kv if (self.tier_c[cp] == 2 and self.zof[z] > 0) else 1.0
                        # _beq1 returns (charity g, net bequest b, marginal value, value).
                        # PYTHON NOTE: `_` is a throw-away name for results we don't need.
                        _, _, mr, vl = _beq1(self.agrid[i], d, Ee, kap, self.phi1_s[z], self.sc, p.sigma, self.phi2,
                                             p.chi, self.phi2c, p.tau_b, p.trust_weight, bool(p.shield))
                        val[cp, i, j, z] = vl; mar[cp, i, j, z] = mr
        return val, mar

    def solve(self):
        """Solve the household problem at every age by backward induction (oldest age first).

        NUMERICS (EGM, the "endogenous grid method", in plain words): instead of searching, for each
        cash-on-hand level, for the best amount to save, we fix the amount saved a' on a grid and ask
        "which consumption c makes saving a' optimal?". The Euler equation answers directly:
        u'(c) = (expected discounted marginal value of a'), so c = (that value)^(-1/sigma). Then the
        cash on hand at which this is optimal is simply m = c + a'. Doing this for every a' gives the
        consumption rule on an "endogenous" grid of m values, with no root-finding. When the trust
        choices make that rule non-monotone, an "upper envelope" step (DC-EGM, _envelope_all) keeps
        the best branch.
        The EGM step is done separately for each END-of-period position (next category c', trust d'_j,
        state ys). The discrete choices (found / keep / pay out / add / gift ...) and their logit
        probabilities are then evaluated for each START-of-period state by _decide.

        Output (stored on self, one list entry per age t; flat state index st, shock node k,
        option rank r = 0..ko-1 for the ko most likely options):
          P[t]   (N, nxi, ko)  probability of each kept option (they sum to 1 over r);
          AP[t]  (N, nxi, ko)  next-period liquid wealth a' (model units);
          DP[t]  (N, nxi, ko)  next-period trust assets d' (model units; float32 to save memory);
          JC[t]  (N, nxi, ko)  next-period category c' (int8);
          TX[t]  (N, nxi, ko)  gift tax paid with that option (model units);
          FR[t]  (N, nxi)      GRAT freeze transfer into the trust (tier 2);
          GF[t]  (N, nxi, ko)  lifetime gift, only at the gift age t = gap (else None);
          BN[t], GC[t]         net bequest and charity, filled later on demand by estates();
          Vb[t], Mb[t] (ncat, na, nd, nyz) expected value and marginal value of liquid wealth at the
                               START of age t (before the return shock), used by age t - 1.
        PAPER: 3.2 and Appendix A. Returns self so calls can be chained."""
        p = self.p; T, ny, nz, nx = p.T, p.ny, self.nz, p.nxi
        na, nd = self.agrid.size, p.nd; nyz = ny * nz; s = p.sigma; nc = self.ncat
        a = self.agrid; beta = self.beta                     # beta = beta_annual^5 (per period)
        BQv, BQm = self._bq_grid()
        # The bequest arrays are indexed by return state z; the solver uses the combined index
        # ys = y * nz + z. zs[ys] = z, so BQv[..., zs] copies the z-columns out to all ys columns.
        # PYTHON NOTE: `...` (Ellipsis) means "all the leading axes"; indexing the last axis with an
        # integer array zs repeats columns.
        zs = np.tile(np.arange(nz), ny)
        BQv, BQm = BQv[..., zs], BQm[..., zs]                  # (ncat, na, nd, nyz)
        Ns = self.N
        # Empty lists, one slot per age, filled in the loop below.
        # PYTHON NOTE: [None] * T is a list of T "nothing" placeholders.
        self.P = [None] * T; self.AP = [None] * T; self.DP = [None] * T; self.JC = [None] * T
        self.TX = [None] * T; self.FR = [None] * T; self.BN = [None] * T; self.GC = [None] * T
        self.Vb = [None] * T; self.Mb = [None] * T; self.done = [None] * T; self.GF = [None] * T
        Vn = Mn = None                                         # next age's Vb, Mb (none at the last age)
        fr = np.array(p.fracs, float)                         # contribution menu (shares of cash on hand)
        # Loop over ages from the oldest (T - 1) down to 0. PYTHON NOTE: range(T-1, -1, -1) counts down.
        for t in range(T - 1, -1, -1):
            st = self.sur[t]                                   # probability of surviving to age t + 1
            if t < T - 1:
                # Expected next-period value given today's (y, z): average over tomorrow's (y', z')
                # with the transition matrix Pyz. Vn has shape (ncat, na, nd, nyz); the matrix product
                # acts on the last axis: EV[..., s] = sum over s' of Vn[..., s'] * Pyz[s, s'].
                # PYTHON NOTE: `@` is matrix multiplication; .T is the transpose.
                EV = Vn @ self.Pyz.T; EM = Mn @ self.Pyz.T
            else:
                # Last age: no future (everyone dies at the end; sur[T-1] = 0).
                EV = np.zeros((nc, na, nd, nyz)); EM = np.zeros((nc, na, nd, nyz))
            # W = value of ending the period with (c', a', d') at state ys: survive and continue, or
            # die and leave an estate, discounted by beta. Shape (ncat, na, nd, nyz).
            W = beta * (st * EV + (1 - st) * BQv)
            # Its derivative with respect to a' (floored at a tiny positive number so the power below works).
            Wa = np.maximum(beta * (st * EM + (1 - st) * BQm), 1e-300)
            # NUMERICS (EGM): consumption that makes a' optimal, from u'(c) = c^(-sigma) = Wa.
            cE = Wa ** (-1 / s)
            # Endogenous cash on hand at which (c, a') is optimal: m = c + a'.
            # PYTHON NOTE: a[None, :, None, None] reshapes the 1-D grid so it lines up with axis 1
            # (the a' axis) and "broadcasts" (is repeated automatically) over the other axes.
            mE = cE + a[None, :, None, None]
            # Value at that point: utility of consumption + continuation value.
            vE = cE ** (1 - s) / (1 - s) + W
            # Reorder axes to (c', d', ys, a') so each EGM line (over a') is contiguous in memory,
            # which the Numba kernels need. PYTHON NOTE: transpose(0, 2, 3, 1) moves axis 1 to the end;
            # np.ascontiguousarray makes a real reordered copy.
            cE = np.ascontiguousarray(cE.transpose(0, 2, 3, 1)); mE = np.ascontiguousarray(mE.transpose(0, 2, 3, 1))
            vE = np.ascontiguousarray(vE.transpose(0, 2, 3, 1))
            # Continuation value when saving nothing (a' = 0): used below the first EGM point, where
            # the household is borrowing-constrained and consumes all its cash.
            W0 = np.ascontiguousarray(W[:, 0, :, :])
            # mono[c', j, ys] = True if cash on hand rises strictly with a' along that line (the usual
            # case). Where it does not, the line has kinks/overlaps from the discrete trust choices.
            mono = np.all(np.diff(mE, axis=-1) > 0, axis=-1)
            self.mono_share = mono.mean()                      # diagnostic (ends up holding age 0's value)
            # NUMERICS (DC-EGM upper envelope): replace non-monotone lines by their upper envelope
            # (best value at each m), in place.
            _envelope_all(mE, cE, vE, W0, mono, s)
            # Output arrays for this age (filled in place by _decide).
            Vb = np.zeros((nc, na, nd, nyz)); Mb = np.zeros((nc, na, nd, nyz))
            P = np.zeros((Ns, nx, p.ko)); AP = np.zeros((Ns, nx, p.ko))
            DP = np.zeros((Ns, nx, p.ko), np.float32); JC = np.zeros((Ns, nx, p.ko), np.int8)
            TX = np.zeros((Ns, nx, p.ko), np.float32); FR = np.zeros((Ns, nx), np.float32)
            # Lifetime gifts are decided once, at the parent's age gap (55-59), if gifts are on.
            gift_now = bool(p.gifts) and t == p.gap
            # Gift array: full size at the gift age, a 1x1x1 dummy otherwise (to save memory).
            GF = np.zeros((Ns, nx, p.ko) if gift_now else (1, 1, 1), np.float32)
            # The main decision kernel. For every start-of-period state and shock it computes
            # cash on hand (returns, income, medical costs, trust returns, taxes, fees, family-office
            # cost, GRAT transfer), builds the option menu (stay/found/keep/pay out/add/up- or
            # downgrade, and the gift menu at the gift age), values each option by interpolating the EGM
            # solution, forms logit choice probabilities with the planning friction, and stores the ko
            # most likely options. Arguments are passed in model units / per period. Note the last
            # array argument: a full-size zero array is passed instead of the dummy when there are no
            # gifts, because the kernel writes into it for every state either way.
            _decide(mE, cE, vE, W0, mono, a, self.dgrid, self.inc[t].reshape(-1), self.med[t], self.cfl, self.Rn, self.kbar,
                    self.Gtc, self.Gpre, self.Gtr, self.wxi, nz, self.zof, self.E, self.setup, self.setup2, self.Ffix,
                    self.ffac, self.tauc, self.taungc, p.tau_gift, 1 - p.disc_v, self.fo_min, self.fo_rate,
                    p.grat, self.Hg, fr, p.fric_mu, p.fric_nu, p.fric2_mu, p.fric2_nu, p.nu_d, bool(p.trusts),
                    bool(p.tier2), s, p.ko, self.ebar, bool(p.allow_windup), bool(p.founder_payout),
                    p.d_min_usd / self.usd, self.role_c, self.tier_c, CATMAP, self.vcap, p.lock, gift_now,
                    np.array(p.gift_fracs, float), p.gift_phi, p.gift_excl_usd / self.usd, self.sc, self.phi2,
                    Vb, Mb, P, AP, DP, JC, TX, FR, GF if gift_now else np.zeros((Ns, nx, p.ko), np.float32),
                    p.fric_slope, p.fric_wref_usd / self.usd, 1.0 if p.gift_exempt else 0.0,
                    np.array(p.payout_fracs, dtype=np.float64), bool(p.fund_cap_E))
            # Store this age's policies.
            self.P[t], self.AP[t], self.DP[t], self.JC[t], self.TX[t], self.FR[t] = P, AP, DP, JC, TX, FR
            self.GF[t] = GF if gift_now else None
            if st < 1.0:
                # At ages with deaths, make room for net bequests (BN) and charity (GC) per option;
                # they are computed lazily, only for states that actually carry mass (see estates()).
                # done[st] marks states already computed.
                self.BN[t] = np.zeros((Ns, nx, p.ko)); self.GC[t] = np.zeros((Ns, nx, p.ko))
                self.done[t] = np.zeros(Ns, np.bool_)
            self.Vb[t] = Vb; self.Mb[t] = Mb
            # Diagnostic: remember the first age (counting down) where a value is NaN or infinite,
            # with the EGM arrays, for debugging. PYTHON NOTE: hasattr(obj, 'name') asks whether the
            # attribute already exists; np.isfinite(x).all() is True if every entry is a normal number.
            if not (np.isfinite(Vb).all() and np.isfinite(Mb).all()) and not hasattr(self, '_bad'):
                self._bad = (t, mE, cE, vE, W0, mono)
            Vn, Mn = Vb, Mb                                    # becomes "next age" for age t - 1
        return self

    def estates(self, t, v):
        """Make sure net bequests are computed for all states with mass in v at age t.

        v: a distribution over flat states (length N). For each state with v > 0 not yet done, and each
        stored option, _estates_fill computes the net bequest b after estate tax (into BN[t])
        and charity g (into GC[t]) for a death at the end of age t, using _beq1. Lazy on purpose:
        most states never carry mass, and _beq1 involves a bisection. Model units."""
        p = self.p
        _estates_fill(v, self.P[t], self.AP[t], self.DP[t], self.JC[t], self.BN[t], self.GC[t], self.done[t],
                      self.phi1_s, self.zof, self.nz, p.ny * self.nz, self.E, 1 - p.disc_v, self.role_c,
                      self.tier_c, self.sc, p.sigma, self.phi2, p.chi, self.phi2c, p.tau_b, p.trust_weight,
                      bool(p.shield))

    # ==========================================================================================
    # SECTION 3: MOVING THE DISTRIBUTION FORWARD (one period; inheritances)
    # ==========================================================================================
    # ---- distribution ----
    def step(self, t, v):
        """Next-period distribution of mass v at age t (choices, then exogenous transition).

        Input v: mass on each flat state at age t (length N). Output: mass on each flat state at age
        t + 1, before deaths (callers multiply by survival). Two parts: (1) each state's mass is
        spread over return shocks (weights wxi) and the kept options (probabilities P); the chosen
        (a', d') is placed on the grid by a lottery (_push / _push_idx); (2) the earnings and
        return states move by the Markov matrix Pyz, independently of the choices."""
        p = self.p; nyz = p.ny * self.nz
        # PYTHON NOTE: np.flatnonzero(cond) lists the positions where cond is True.
        idx = np.flatnonzero(v > 0)
        if idx.size > 0.15 * v.size:
            # Dense vector (more than 15% of states occupied): parallel kernel over all states, 2 chunks.
            tmp = _push(v, self.P[t], self.AP[t], self.DP[t], self.JC[t], self.wxi, self.agrid, self.dgrid,
                        self.agrid.size, p.nd, nyz, 2)
            # View as rows = (category, a, d) cells, columns = ys, then apply the (y, z) transition:
            # new[row, s'] = sum over s of tmp[row, s] * Pyz[s, s']. Flatten back to length N.
            # PYTHON NOTE: reshape(-1, nyz) means "nyz columns, as many rows as needed".
            return (tmp.reshape(-1, nyz) @ self.Pyz).reshape(-1)
        # sparse vector (e.g. a child or a large-gift pair): work only where there is mass
        tmp = _push_idx(v, idx, self.P[t], self.AP[t], self.DP[t], self.JC[t], self.wxi, self.agrid, self.dgrid,
                        self.agrid.size, p.nd, nyz).reshape(-1, nyz)
        # Apply the (y, z) transition only to rows that have some mass (saves time for sparse vectors).
        rows = np.flatnonzero(tmp.any(axis=1))
        out = np.zeros_like(tmp)
        out[rows] = tmp[rows] @ self.Pyz
        return out.reshape(-1)

    def _dlot_matrix(self, shift, sib):
        """Lottery matrix (nd, nd): trust bin jd -> grid weights of shift + dgrid[jd] / (1 + sib).

        Used for inheriting a trust. shift = the trust the child already holds (model units, 0 if none);
        dgrid[jd] = the parent's trust; sib = 0 (only child) or 1 (one sibling: the trust is split in
        two). Row jd of L says how the child's new trust value, shift + dgrid[jd] / (1 + sib), is
        spread over the two neighbouring trust grid points (weights sum to 1)."""
        nd = self.p.nd; L = np.zeros((nd, nd))
        for jd in range(nd):
            # _lot returns the lower grid index lo and the weight wl on it (1 - wl goes to lo + 1).
            lo, wl = _lot(self.dgrid, shift + self.dgrid[jd] / (1.0 + sib))
            L[jd, lo] += wl; L[jd, lo + 1] += 1 - wl
        return L

    def conv(self, cn, H0, H1):
        """Child's next-period distribution cn plus inheritances (H0: liquid-only estates; H1[t]:
        estates passing a tier-(t+1) trust). Same result as _conv_all, factorised for speed.

        What it computes (PAPER 3.2, inheritance): the parents of these children have just died; their
        estates are described by histograms on the grid -
          H0 (na,)        mass of estates with net liquid bequest agrid[b] and NO trust;
          H1 (2, na, nd)  mass of estates with net liquid bequest agrid[b] and a trust dgrid[l] of tier
                          1 (H1[0]) or tier 2 (H1[1]).
        Each child receives the estate divided by the number of children: alone with probability
        1 - q2 (sib = 0), or shared with one sibling with probability q2 (sib = 1, half each). The
        child's liquid wealth rises by bequest / (1 + sib); an inherited trust is added to any trust the
        child already has; the child becomes an heir of that tier (a child who founded a trust keeps
        role "founder" and takes the higher tier; APPROXIMATION from the part-1 banner).
        In words, this is a "convolution": every child state is combined with every estate cell.
        Input cn: the child's distribution (length N, at the age just after the parent's death).
        Output: the child's distribution with inheritances added (length N, same total mass as cn
        times the total mass of H0 + H1).

        NUMERICS (factorisation): done naively the cost is (child cells) x (estate cells) x 4 grid
        corners. Two speed-ups: (a) when both are sparse, Numba brute-force kernels with pre-computed
        lottery tables (_conv_c0_children, _conv_trust_children); (b) otherwise, split the child's
        distribution into "slices" (one per category and trust bin), and for each slice do the liquid
        shift (_shiftY) and the trust lottery (_dlot_matrix) separately, then combine them with one
        big matrix product (np.tensordot / np.einsum). This works because, for a given estate, the
        liquid and trust parts of the inheritance are placed on their grids independently."""
        p = self.p; na, nd, nyz = self.agrid.size, p.nd, p.ny * self.nz
        if not hasattr(self, "_Lcache"):
            # Cache every trust-lottery matrix: for each child trust bin jd (the shift) and sib in {0, 1}.
            # PYTHON NOTE: {key: value for ...} is a "dict comprehension" (a lookup table built by a loop).
            self._Lcache = {(jd, sib): self._dlot_matrix(self.dgrid[jd], sib) for jd in range(nd) for sib in (0, 1)}
        out = np.zeros(self.N)
        # Views (not copies) of the output and input split by block:
        #   out0 / C0: category 0, shape (na, nyz);  outT / CT: trust categories, (ncat-1, na, nd, nyz).
        # PYTHON NOTE: reshape of a slice returns a "view": writing into out0 writes into out.
        out0 = out[:self.N0].reshape(na, nyz); outT = out[self.N0:].reshape(self.ncat - 1, na, nd, nyz)
        C0 = cn[:self.N0].reshape(na, nyz); CT = cn[self.N0:].reshape(self.ncat - 1, na, nd, nyz)
        # Bequest bins that carry mass: r0 for liquid-only estates; r1[t] for tier-(t+1) trust estates.
        # PYTHON NOTE: np.nonzero(cond)[0] gives the positions where cond is True.
        r0 = np.nonzero(H0 > 0)[0]
        r1 = [np.nonzero(H1[t].sum(axis=1) > 0)[0] for t in range(2)]
        # Number of non-empty estate cells (used to judge whether brute force is cheap).
        nnz_h = np.count_nonzero(H0) + np.count_nonzero(H1)
        slices = []                                            # child slices left for the factorised path
        if C0.sum() > 0:
            # Children without a trust.
            if np.count_nonzero(C0) * nnz_h < 2e6:
                # Cheap: brute force with lottery tables (built once and cached).
                if not hasattr(self, "_Ltab"):
                    self._Ltab = _lot_tables(self.agrid, self.dgrid)
                # PYTHON NOTE: *self._Ltab "unpacks" the 4 tables as 4 separate arguments.
                _conv_c0_children(np.ascontiguousarray(C0), H0, H1, *self._Ltab, self.q2, CATMAP, out0, outT)
            else:
                slices.append((0, 0, C0))                      # (category 0, trust bin 0, slice)
        if CT.any():
            # children who already hold a trust: brute force with lottery tables when that is cheap
            # (few child trust states x few estate cells), otherwise the factorised BLAS path per slice
            nnz_c = np.count_nonzero(CT)
            if nnz_c * nnz_h < 2e6:
                if not hasattr(self, "_Ltab"):
                    self._Ltab = _lot_tables(self.agrid, self.dgrid)
                _conv_trust_children(np.ascontiguousarray(CT), H0, H1, *self._Ltab, self.q2, self.role_c,
                                     self.tier_c, CATMAP, outT)
            else:
                # One slice per (trust category cc, child trust bin jd) with mass: the (na, nyz)
                # distribution over the child's liquid wealth and exogenous state.
                for cc in range(1, self.ncat):
                    mj = CT[cc - 1].sum(axis=(0, 2))           # mass by child trust bin
                    for jd in np.nonzero(mj > 0)[0]:
                        slices.append((cc, jd, np.ascontiguousarray(CT[cc - 1, :, jd, :])))
        # Factorised path. Outer loop: only child (prob 1 - q2) or one sibling (prob q2).
        for sib, pr in ((0, 1 - self.q2), (1, self.q2)):
            for cc, jd, C in slices:
                if r0.size:
                    # Liquid-only estates. Y[r] = the slice C with liquid wealth shifted up by
                    # agrid[r0[r]] / (1 + sib), on the grid; shape (len(r0), na, nyz).
                    Y = _shiftY(C, r0, self.agrid, sib)
                    # Mix over estates: add = sum over r of H0[r0[r]] * Y[r] -> shape (na, nyz).
                    # PYTHON NOTE: np.tensordot(A, B, axes=(0, 0)) multiplies and sums over the
                    # first axis of both arrays (a matrix product along that axis).
                    add = pr * np.tensordot(H0[r0], Y, axes=(0, 0))
                    if cc == 0:
                        out0 += add                            # child stays in category 0
                    else:
                        outT[cc - 1, :, jd, :] += add          # child keeps its category and trust bin
                for t in range(2):
                    if r1[t].size == 0:
                        continue
                    # Category after inheriting a tier-(t+1) trust: a child with no trust becomes an heir
                    # of that tier; a child with a trust keeps its role and takes the higher tier.
                    # (CATMAP[role, tier] gives the category number; see ParamsC above.)
                    tgt = CATMAP[1, t + 1] if cc == 0 else CATMAP[self.role_c[cc], max(self.tier_c[cc], t + 1)]
                    # Liquid shift for each bequest bin with trust estates.
                    Y = _shiftY(C, r1[t], self.agrid, sib)
                    # Trust part: Dd[b, l'] = mass of estates with liquid bin b that put the child's
                    # trust at grid point l' (parent's trust dgrid[l] / (1 + sib) added to the child's
                    # own dgrid[jd], then lotteried). Matrix (bequest bins) x (nd).
                    Dd = H1[t][r1[t]] @ self._Lcache[(jd, sib)]
                    # Combine: out[a, l', s] += sum over b of Dd[b, l'] * Y[b, a, s].
                    # PYTHON NOTE: np.einsum('bl,bas->als', ...) spells out which axes are multiplied and
                    # summed (b) and which are kept (a, l, s) - a compact way to write this triple sum.
                    outT[tgt - 1] += pr * np.einsum('bl,bas->als', Dd, Y, optimize=True)
        return out

    # ==========================================================================================
    # SECTION 4: THE GENERATION FIXED POINT (stationary distribution) AND ITS ACCELERATION
    # ==========================================================================================
    def _anderson(self, x, fx, mem=6, w=None):
        """Anderson acceleration of the generation fixed point D = F(D) (F is linear in D, and
        heirs' trusts make its slowest mode decay slowly). Keeps D non-negative with its mass.
        w: optional state weights for the least-squares fit. With wealth weights the fit tracks
        the slow mode at the top of the distribution (trust wealth accumulating over dynasties),
        which carries little mass and is ignored by an unweighted fit.

        NUMERICS (Anderson acceleration, in plain words): plain iteration D -> F(D) can crawl when
        some feature (here: trust wealth building up over many generations) changes by only a small
        fraction each round. Anderson's method remembers the last few guesses x and their "residuals"
        g = F(x) - x (how much one more round would change them). It finds the combination of recent
        steps that best cancels the residual (a small least-squares problem) and jumps there,
        extrapolating the slow drift instead of waiting for it.
        Inputs: x = current guess (parents' distribution, length N); fx = F(x), one more generation;
        mem = how many past steps to remember. History lives in self._aa (reset by distribution()).
        Output: the next guess, clipped at zero and rescaled to the total mass of fx."""
        g = fx - x                                             # residual of the current guess
        H = self._aa
        H.append((x.copy(), g.copy()))                         # remember (guess, residual)
        if len(H) > mem + 1:
            H.pop(0)                                           # forget the oldest
        if len(H) < 2:
            return fx                                          # not enough history yet: plain step
        # Columns = changes between consecutive guesses (dX) and between their residuals (dG).
        # PYTHON NOTE: np.stack([...], axis=1) puts the vectors side by side as columns of a matrix.
        dX = np.stack([H[i + 1][0] - H[i][0] for i in range(len(H) - 1)], axis=1)
        dG = np.stack([H[i + 1][1] - H[i][1] for i in range(len(H) - 1)], axis=1)
        # Least squares: coefficients gam so that dG @ gam is as close as possible to g.
        # PYTHON NOTE: np.linalg.lstsq returns several things; `gam, *_ =` keeps the first.
        if w is None:
            gam, *_ = np.linalg.lstsq(dG, g, rcond=None)
        else:
            # Weighted fit: each state's row is multiplied by its weight (more weight on wealthy states).
            gam, *_ = np.linalg.lstsq(dG * w[:, None], g * w, rcond=None)
        # Anderson update: the plain step x + g, corrected by the fitted combination of past steps.
        xn = x + g - (dX + dG) @ gam
        # A distribution cannot be negative: clip, then rescale to the mass that F produced.
        xn = np.maximum(xn, 0.0)
        return xn * (fx.sum() / xn.sum())

    def distribution(self, tol=1e-10, maxgen=400, D0=None, verbose=False, accel=True, wtol=None, mingen=1):
        """Stationary distribution by iterating generations. Stops when the L1 mass change is
        below tol and, if wtol is given, the wealth-weighted change sum|dD|*x / sum(D*x) is below
        wtol (top-wealth moments move with mass too small for the mass test to see). With wtol
        the acceleration fit is wealth-weighted too.

        The idea (PAPER 3.2 inheritance; Appendix A). The unknown is D = the distribution of PARENTS
        at the age their child enters the model (t = gap = 7, ages 55-59), over all flat states.
        Given D, one generation is simulated: each child enters at age 20 with a type drawn given the
        parent's (y, z); children and parents then age side by side; at the gift age the parent may
        give a lifetime gift; when a parent dies the child inherits (conv); by the time the child is
        55 every parent has died, so the children's distribution at 55 is the next generation's D.
        That map D -> F(D) is linear, and we iterate it to a fixed point D = F(D).

        Factorised parent-child bookkeeping, in plain words. The inheritance a child gets depends on
        the parent's wealth, so we need the JOINT distribution of (child state, parent state), which
        is far too big to store (N x N). Trick: while both are alive and no gift links them, a child's
        luck is independent of the parent's luck. So the joint distribution can be stored as a short
        list of "pairs" (c, pw): a child vector c (mass 1) and a parent vector pw, meaning "children
        distributed like c, whose parents are distributed like pw". At entry there is one pair per
        child initial type k0 (exact: the child's type is the only link). Each period both vectors are
        moved forward separately with step(). When some parents of a pair die, their estates are turned
        into a histogram, convolved with the child vector (conv), and that mass leaves the pair and goes
        into O, the "un-paired" children (whose parents are dead). The gift at age 55 does link child
        and parent, so each pair is split by gift size there (small gifts pooled; see below).

        Inputs: tol (mass test), wtol (wealth-weighted test, e.g. 1e-4), maxgen (max generations),
        D0 (starting guess, e.g. from a nearby parameter vector), accel (use Anderson acceleration),
        mingen (minimum generations), verbose (print progress).
        Output (stored on self): dist (T, N) = distribution over flat states at each age, per member
        of a cohort that entered with mass 1 (survival included; cohort-size weights applied later);
        parent_D = the fixed point D; pairs3 / pairs6 = parent-child pairs when the child is 35-39 /
        50-54 (for mobility moments); gens_to_converge. Returns self."""
        # PYTHON NOTE: `import time as _time` loads the standard clock module under a private name.
        import time as _time
        p = self.p; T, gap = p.T, p.gap; nyz = p.ny * self.nz; N = self.N
        self._aa = []                                          # Anderson history, fresh for this call
        if D0 is None:
            # Starting guess: a cohort entering with no wealth, category 0, with the stationary entry
            # distribution of (y, z) (indices 0..nyz-1 are a = 0, category 0), aged to 55 without any
            # inheritances.
            D = np.zeros(N); D[:nyz] = self.pi_age[0]
            for t in range(gap):
                D = self.step(t, D)
        else:
            D = D0.copy()
        _t0 = _time.time()
        # Total wealth a + d of every flat state: the weights for the wealth-weighted test.
        xw = self.fa + self.fd
        for gen in range(maxgen):
            # O[t]     = children at age t whose parent has already died (mass, per entering cohort);
            # child[t] = ALL children at age t (paired + un-paired) -> becomes self.dist.
            O = np.zeros((T, N)); child = np.zeros((T, N))
            Dm = D.reshape(-1, nyz)                            # rows = (cat, a, d) cells, cols = parent ys
            # Build the initial pairs, one per child initial type k0 (= combined (y0, z0) index).
            Cs, Ps = [], []
            for k0 in range(nyz):
                # Parent vector: parents weighted by the probability that their child is of type k0.
                # Kbirth[parent ys, k0] is that probability (Layer B); [None, :] broadcasts it over rows.
                pw = (Dm * self.Kbirth[:, k0][None, :]).reshape(-1)
                if pw.sum() <= 0:
                    continue
                # Child vector: all mass at flat index k0 = (category 0, a = 0, type k0).
                c = np.zeros(N); c[k0] = 1.0
                Cs.append(c); Ps.append(pw)
            store3 = store6 = None
            # Walk the children through their ages t = 0 .. T-2 (parent is at age tp = t + gap).
            for t in range(T - 1):
                tp = t + gap
                if tp <= T - 1:
                    # Some parents may still be alive. Children in pairs carry the mass of their
                    # still-living parents: child[t] gets c * (mass of pw). Plus the un-paired ones.
                    # PYTHON NOTE: zip(Cs, Ps) walks the two lists together, one pair at a time.
                    for c, pw in zip(Cs, Ps):
                        child[t] += c * pw.sum()
                    child[t] += O[t]
                    # Snapshots of the pairs for mobility moments: child 35-39 and child 50-54.
                    # (Only children whose parent is still alive at that point are in the pairs.)
                    if t == (35 - p.age0) // p.tlen:
                        store3 = [(c.copy(), pw.copy()) for c, pw in zip(Cs, Ps)]
                    if t == (50 - p.age0) // p.tlen:
                        store6 = [(c.copy(), pw.copy()) for c, pw in zip(Cs, Ps)]
                    st = self.sur[tp]                          # parent's survival probability this period
                    # Un-paired children move forward one period (children of these ages do not die in
                    # practice: sur = 1 up to age gap, set in Layer B).
                    newO = self.step(t, O[t]) * self.sur[t]
                    newC, newP = [], []
                    if t == 0 and self.GF[tp] is not None:
                        # GIFT STAGE: the child is 20-24 and the parent 55-59, deciding the lifetime gift.
                        # lifetime gifts: split each pair by the parent's gift; the child receives
                        # gift / (number of children) as starting wealth
                        bigacc = {}                      # gift bin -> [child mixture, parent sum, mass]
                        for c, pw in zip(Cs, Ps):
                            cn = self.step(t, c)               # child's own next-period distribution
                            # Parent's next-period vector split by gift bin (rows = gift grid points;
                            # the gift is lotteried onto ggrid), before the (y, z) transition.
                            split = _push_split(pw, self.P[tp], self.AP[tp], self.DP[tp], self.JC[tp], self.GF[tp],
                                                self.wxi, self.agrid, self.dgrid, self.ggrid, self.agrid.size, p.nd, nyz)
                            # small gifts (below gift_pool_usd) are pooled into one pair: the child
                            # gets the pooled gift distribution, independently of which of those
                            # parents gave; large gifts keep their own pair (exact pairing where it
                            # matters for later inheritances, and those vectors are sparse)
                            # APPROXIMATION: pooling breaks the link between a small gift and that
                            # parent's later estate.
                            mass = split.sum(axis=1)           # parent mass by gift bin
                            pool = [gb for gb in range(1, self.ggrid.size) if mass[gb] > 0 and self.ggrid[gb] < self.gpool]
                            big = [gb for gb in range(1, self.ggrid.size) if mass[gb] > 0 and self.ggrid[gb] >= self.gpool]
                            # large-gift bins with little mass are merged into one pooled large-gift pair
                            # (with the default gift_min_pair = 0, every large bin keeps its own pair)
                            thr = p.gift_min_pair * mass.sum()
                            solo = [gb for gb in big if mass[gb] >= thr]
                            rest = [gb for gb in big if mass[gb] < thr]
                            # Groups of gift bins that become one pair each: no gift; pooled small gifts;
                            # each large bin alone; pooled small-mass large bins.
                            groups = ([[0]] if mass[0] > 0 else []) + ([pool] if pool else []) + [[gb] for gb in solo] \
                                + ([rest] if rest else [])
                            for grp in groups:
                                # Parent vector for this group: sum of its gift bins, then the (y, z)
                                # transition and survival (st = 1 at this age).
                                pn = split[grp].sum(axis=0)
                                pn = st * (pn.reshape(-1, nyz) @ self.Pyz).reshape(-1)
                                # H0 = distribution of the gift amount (family total) on the liquid grid,
                                # mass-weighted across the group's bins, normalised to 1.
                                H0 = np.zeros(self.agrid.size)
                                for gb in grp:
                                    lo_, wl_ = _lot(self.agrid, self.ggrid[gb])
                                    H0[lo_] += wl_ * mass[gb]; H0[lo_ + 1] += (1 - wl_) * mass[gb]
                                H0 /= H0.sum()
                                if grp == [0]:
                                    cg = cn                            # no gift: child unchanged
                                else:
                                    # Add the gift to the child's liquid wealth (split by siblings), using
                                    # the inheritance machinery with an empty trust part.
                                    cg = self.conv(cn, H0, np.zeros((2, self.agrid.size, p.nd)))
                                if p.gift_merge_types and len(grp) == 1 and self.ggrid[grp[0]] >= self.gpool:
                                    # large gifts: one pair per gift bin across children's types
                                    # (accumulate over the k0 pairs: mass-weighted child mixture, summed
                                    # parent vector, total mass). APPROXIMATION: within a gift bin, the
                                    # child's initial type is averaged over parents, saving pairs.
                                    mg = pn.sum()
                                    if grp[0] in bigacc:
                                        acc = bigacc[grp[0]]; acc[0] += mg * cg; acc[1] += pn; acc[2] += mg
                                    else:
                                        bigacc[grp[0]] = [mg * cg, pn.copy(), mg]
                                    continue
                                newC.append(cg); newP.append(pn)
                        # Turn the accumulated large-gift groups into pairs (child = mass-weighted mean).
                        # PYTHON NOTE: dict.items() walks over (key, value) entries.
                        for gb, (cm, pm, mg) in bigacc.items():
                            if mg > 0:
                                newC.append(cm / mg); newP.append(pm)
                        Cs, Ps = newC, newP
                        O[t + 1] = newO
                        continue                               # no deaths at the gift age (sur = 1)
                    # ORDINARY AGE: move every pair forward; handle parents who die this period.
                    for c, pw in zip(Cs, Ps):
                        cn = self.step(t, c)
                        if st < 1.0:
                            # Net bequests for the parents' states (lazy fill), then the histogram of the
                            # dying parents' estates: mass pw * (1 - st). H0: liquid only; H1: with trust.
                            self.estates(tp, pw)
                            H0, H1 = _estate_hist2(pw * (1 - st), self.P[tp], self.BN[tp], self.DP[tp],
                                                   self.JC[tp], self.wxi, self.agrid, self.dgrid, self.tier_c)
                            # Their children inherit and leave the pair: into the un-paired mass.
                            newO += self.conv(cn, H0, H1)
                        # Surviving parents (mass st * ...) stay paired with the same child vector.
                        newC.append(cn); newP.append(st * self.step(tp, pw))
                    Cs, Ps = newC, newP
                    O[t + 1] = newO
                else:
                    # All parents are dead by now (they would be older than the last age): only
                    # un-paired children remain.
                    child[t] = O[t]
                    O[t + 1] = self.step(t, O[t]) * self.sur[t]
            child[T - 1] = O[T - 1]
            # Next generation's parents = this generation's children at age gap: Dn = F(D).
            Dn = O[gap].copy()
            # Convergence measures: L1 change in mass, and the change weighted by wealth relative to
            # total wealth (sensitive to the tiny-mass, very rich top of the distribution).
            diff = np.abs(Dn - D).sum()
            wdiff = (np.abs(Dn - D) * xw).sum() / max((Dn * xw).sum(), 1e-300)
            if accel and gen >= 1:
                if wtol is None:
                    D = self._anderson(D, Dn)
                else:
                    # Weights 1 + (wealth / mean wealth under D): the fit cares about rich states too.
                    D = self._anderson(D, Dn, w=1.0 + xw * (D.sum() / max((D * xw).sum(), 1e-300)))
            else:
                D = Dn
            if verbose:
                print(gen, diff, wdiff, round(_time.time() - _t0, 1), flush=True)
            # Stop when both tests pass (diff < tol; wdiff < wtol if given) and at least mingen rounds ran.
            # Note: `child` was computed from the D that went INTO this round, so self.dist matches
            # the pre-update D; the two differ by less than the tolerances at convergence.
            if diff < tol and (wtol is None or wdiff < wtol) and gen + 1 >= mingen:
                break
        self.gens_to_converge = gen + 1
        self.dist = child
        self.parent_D = D
        self.pairs3 = store3; self.pairs6 = store6
        return self

    # ==========================================================================================
    # SECTION 5: MOMENTS (mobility, diagnostics, calibration targets)
    # ==========================================================================================
    # ---- moments ----
    def rank_rank(self, pairs):
        """Rank-rank correlation of parent and child total wealth (ties get their mid-rank).

        pairs: a list of (child vector, parent vector) from distribution() (pairs3 or pairs6). Each
        pair stands for the joint distribution c x pw (outer product), so the joint is never built.
        Ranks are percentile ranks (0 to 1) of total wealth a + d within the child marginal and the
        parent marginal. Returns the Pearson correlation of the two ranks (a unit-free number).
        Note: pairs contain only children whose parent is still alive at that age."""
        x = self.fa + self.fd                                 # total wealth of each flat state
        # Marginal distributions: children (Mc) and parents (Mp) over flat states.
        # PYTHON NOTE: sum(... for ...) adds up the results of a "generator" loop.
        Mc = sum(c * pw.sum() for c, pw in pairs); Mp = sum(pw * c.sum() for c, pw in pairs)
        # PYTHON NOTE: a function defined inside another function ("nested function").
        def ranks(M):
            # Group states with equal wealth (np.unique; inv maps each state to its group), add up
            # mass by group (np.bincount), then mid-rank = cumulative share below + half the group's share.
            u, inv = np.unique(x, return_inverse=True)
            mu = np.bincount(inv, weights=M, minlength=u.size); mu = mu / mu.sum()
            cdf = np.cumsum(mu) - mu / 2
            return cdf[inv]                                    # rank of each flat state
        Rc, Rp = ranks(Mc), ranks(Mp)
        # Moments of the ranks under the joint distribution, pair by pair (sum over pairs of
        # products of child-side and parent-side sums): S total mass, E means, V variances, Ecp cross.
        S = sum(c.sum() * pw.sum() for c, pw in pairs)
        Ec = sum((c * Rc).sum() * pw.sum() for c, pw in pairs) / S
        Ep = sum(c.sum() * (pw * Rp).sum() for c, pw in pairs) / S
        Ecp = sum((c * Rc).sum() * (pw * Rp).sum() for c, pw in pairs) / S
        Vc = sum((c * Rc ** 2).sum() * pw.sum() for c, pw in pairs) / S - Ec ** 2
        Vp = sum(c.sum() * (pw * Rp ** 2).sum() for c, pw in pairs) / S - Ep ** 2
        return (Ecp - Ec * Ep) / np.sqrt(Vc * Vp)             # covariance / (sd x sd)

    def cap_diagnostics(self):
        """Mass-weighted share of trust holders (and of everyone) for whom each safety rule binds:
        grantor tax capped at half of cash, family-office cost capped at a quarter of cash,
        freeze transfer capped at half of cash, and the cash floor.

        A check that the safety caps inside _decide rarely bind. It redoes the cash-on-hand
        arithmetic of _decide for every trust-holding state with mass. Returns a dict of shares
        (unit-free). APPROXIMATION in this diagnostic only: it ignores medical costs and the
        consumption floor, and tests the cash floor before the GRAT transfer and the D < 0 adjustment
        (in _decide those come first), so it is slightly approximate."""
        p = self.p; nyz = p.ny * self.nz
        tot = 0.0; tot_tr = 0.0                               # population mass; trust-holder mass
        hit = {"grantor_tax_cap": 0.0, "fo_cost_cap": 0.0, "grat_cap": 0.0, "cash_floor": 0.0}
        mfloor = 1e-4 * self.ebar                              # same cash floor as _decide
        for t in range(p.T):
            v = self.dist[t] * self.cohort[t]                  # mass at age t (cohort-size weighted)
            tot += v.sum()
            sel = np.nonzero((v > 0) & (self.fcat > 0))[0]     # trust-holding states with mass
            if sel.size == 0:
                continue
            c = self.fcat[sel]; a = self.fa[sel]; d = self.fd[sel]; z = self.fs[sel] % self.nz
            role = self.role_c[c]; tier = self.tier_c[c]; inc = self.inc[t].reshape(-1)[self.fs[sel]]
            for k in range(p.nxi):
                w = v[sel] * self.wxi[k]; tot_tr += w.sum()
                Gp = self.Gtr[z, k]                            # pre-tax gross trust return
                # Cash on hand from liquid wealth (no scale limit) plus income.
                m0 = self.Rn * a + (self.Gtc[c, z, k] - self.Rn) * a + inc
                # Trust after return, tax (heirs: non-grantor rate) and fees, as in _decide.
                # PYTHON NOTE: np.where(cond, x, y) picks x where cond is True, else y, element by element.
                m = m0.copy(); D = np.where(role == 1, d * (1 + (Gp - 1) * (1 - self.taungc[c])) * self.ffac - self.Ffix,
                                            d * Gp * self.ffac - self.Ffix)
                D = np.maximum(D, 0.0)
                # Founders pay the trust's income tax; capped at half of cash (rest paid by the trust).
                gt = np.where(role == 2, self.tauc[c] * (Gp - 1) * d, 0.0)
                gcap = (gt > 0) & (gt > 0.5 * m0)
                gl = np.where(gcap, 0.5 * m0, gt); m -= gl; D -= gt - gl
                # Family-office cost (tier 2), liquid share capped at a quarter of cash.
                fo = np.where(tier == 2, np.maximum(self.fo_min, self.fo_rate * (a + d)), 0.0)
                fl_raw = fo * a / np.maximum(a + d, 1e-300)
                fcap = (tier == 2) & (fl_raw > 0.25 * m0)
                m -= np.minimum(fl_raw, 0.25 * m0) * (tier == 2)
                # GRAT freeze transfer (tier 2), capped at half of remaining cash.
                fr_raw = np.where(tier == 2, p.grat * a * np.maximum(self.Gpre[z, k] - self.Hg, 0.0), 0.0)
                rcap = (tier == 2) & (fr_raw > 0.5 * m) & (fr_raw > 0)
                # Add up the mass where each cap binds. PYTHON NOTE: w[mask] keeps entries where mask is True.
                hit["grantor_tax_cap"] += w[gcap].sum(); hit["fo_cost_cap"] += w[fcap].sum()
                hit["grat_cap"] += w[rcap].sum(); hit["cash_floor"] += w[m < mfloor].sum()
        # PYTHON NOTE: f"{k}_share..." is an "f-string": {k} is replaced by the value of k.
        out = {f"{k}_share_of_trust_holders": v_ / tot_tr if tot_tr > 0 else 0.0 for k, v_ in hit.items()}
        out.update({f"{k}_share_of_population": v_ / tot for k, v_ in hit.items()})
        return out

    def wealth_elasticity(self, floor_units=0.01, pairs=None):
        """Parent-child wealth elasticity: the OLS slope from regressing log child total wealth on log
        parent total wealth, over parent-child pairs where both have wealth above floor_units x ebar
        (0.01 x mean 5-year earnings). pairs defaults to pairs3 (child 35-39, parent 70-74 with
        age0 = 20, parents still alive). Computed from sums over pairs without building the joint.
        Returns a unit-free number."""
        x = self.fa + self.fd
        pos = x > floor_units * self.ebar                      # above the wealth floor
        la = np.where(pos, np.log(np.maximum(x, 1e-300)), 0.0)  # log wealth (0 where excluded)
        # Running sums for OLS: S mass, Sx sum of parent log wealth (regressor), Sy child log wealth,
        # Sxx sum of squared parent log wealth, Sxy cross products.
        S = Sx = Sy = Sxx = Sxy = 0.0
        for c, pw in (self.pairs3 if pairs is None else pairs):
            wc, wp = (c * pos).sum(), (pw * pos).sum()
            ec, ep = (c * pos * la).sum(), (pw * pos * la).sum()
            S += wc * wp; Sx += wc * ep; Sy += ec * wp
            Sxx += wc * (pw * pos * la ** 2).sum(); Sxy += ec * ep
        mx, my = Sx / S, Sy / S
        return (Sxy / S - mx * my) / (Sxx / S - mx ** 2)       # cov(x, y) / var(x)

    # PYTHON NOTE: @staticmethod marks a method that does not use `self`; it is just a helper function
    # kept inside the class.
    @staticmethod
    def _group_share(xs, ws, ind, lo, hi):
        """Mass-weighted mean of ind among the population between top shares lo and hi
        (e.g. lo=0, hi=0.01 is the top 1%), allowing fractional states at the boundaries.

        xs: wealth of each state; ws: population share of each state (sums to 1); ind: the quantity
        to average (e.g. 1 if the state holds a trust). States are ranked from richest down; a state
        straddling the 1% line counts only with the part of its mass inside the group."""
        # Sort from richest to poorest (argsort of -xs). "stable" keeps the original order among ties.
        o = np.argsort(-xs, kind="stable"); w = ws[o]; I = ind[o]
        c1 = np.cumsum(w); c0 = c1 - w                         # each state's [start, end) in rank space
        # Mass of each state inside [lo, hi): overlap of its interval with the group.
        part = np.clip(np.minimum(c1, hi) - np.maximum(c0, lo), 0.0, None)
        return (part * I).sum() / part.sum()

    @staticmethod
    def _group_ratio(xs, ws, num, den, lo, hi):
        """Like _group_share, but a ratio of totals within the group: sum(num) / sum(den) over the
        population between top shares lo and hi (e.g. trust assets / total wealth in the top 1%)."""
        o = np.argsort(-xs, kind="stable"); w = ws[o]
        c1 = np.cumsum(w); c0 = c1 - w
        part = np.clip(np.minimum(c1, hi) - np.maximum(c0, lo), 0.0, None)
        return (part * num[o]).sum() / (part * den[o]).sum()

    def moments(self):
        """All model statistics, as a dict {name: value}. Call after solve() and distribution().

        Population weights: w[state] = share of the whole population (all ages, cohort sizes from
        population growth) in each flat state. UNITS: stocks (K, Kb, ...) in model units per person;
        flows "/Y" are ANNUAL flows over ANNUAL output Y (5-year flows are divided by tlen).

        Calibration targets (PAPER 3.4; final calibration v2c in calibrate_C.py, 11 targets):
          K_gap = (K_supply - Kb) / Kc_demand - 1 = 0: capital market clearing. K_supply = mean total
              wealth (a + d); Kb = wealth of entrepreneurs (z > 0), which they invest in their own
              businesses; the rest must equal Kc_demand, the corporate sector's capital demand at r.
          top1_dfa, top0.1_dfa, top10_dfa (32.5%, 15.0%, 68.9%, Fed DFA Q2 2026): top wealth shares on
              the DFA-consistent measure (dfa_top_shares, PAPER 3.5). Earlier calibrations matched the
              same data to top*_liquid_only = shares of PERSONAL (liquid) wealth a only, as if trusts
              were invisible. "top1" etc. without suffix = shares of total wealth a + d.
          trust_share_of_top1_wealth (0.15) / trust_share_of_top0.1_wealth (0.40): trust assets as a
              share of the total wealth of the top 1% / top 0.1% (ranked by a + d).
          estate_tax/gross_50M+ (0.238, IRS SOI 2024): estate tax paid / gross liquid estate, for
              decedents whose gross liquid estate is $50M or more (trust assets are outside the
              estate, as on an estate-tax return).
          share_deaths_taxable (0.14%, matched in logs): share of deaths with a taxable estate.
          charity/taxable_estates (0.181, IRS SOI): charitable bequests / gross liquid estate, among
              taxable estates.
          wealth_75plus/mean (1.527, SCF 2022, down-weighted): mean wealth at ages 75+ / overall mean.
          inter_vivos_share (0.27, Sabelhaus 2024): lifetime gifts / (gifts + estates).
        Take-up (takeup_*; share holding a trust worth at least d_min, $100k) is reported, not targeted.
        Other entries are reported statistics and checks (output, taxes, mobility, grid checks)."""
        p = self.p; T = p.T
        # Per-state descriptors (length N): liquid wealth, trust, return state, category, role, tier.
        a, d, z, cat, role, tier = self.fa, self.fd, self.fz, self.fcat, self.frole, self.ftier
        x = a + d                                              # total wealth of each state
        # Mass by (age, state), weighted by cohort size (younger cohorts are bigger: population growth).
        # PYTHON NOTE: self.cohort[:, None] turns the age vector into a column so it multiplies each row.
        Wt = self.dist * self.cohort[:, None]
        tot = Wt.sum(); w = Wt.sum(axis=0) / tot               # population share of each state
        K = (w * x).sum(); Kd = (w * d).sum()                  # mean total wealth; mean trust assets
        # Business capital: total wealth of entrepreneurs (return state z > 0). PYTHON NOTE: (z > 0) is
        # a True/False array that counts as 1/0 in arithmetic.
        Kb = (w * x * (z > 0)).sum()
        prem = np.array(p.prem if self.nzr > 1 else (0.0,))   # annual return premium by return state
        ret = self.r_ann + prem[z]                                 # expected annual return by state
        # Business output (annual): capital income plus depreciation on business capital.
        Yb = (w * x * (z > 0) * (self.r_ann + p.delta + prem[z])).sum()

        def shares(xx):
            """Lorenz-curve tools for wealth measure xx: returns (gini, top), where top(f) is the share
            of total xx held by the richest fraction f of the population."""
            o = np.argsort(xx, kind="stable"); xs, ws = xx[o], w[o]      # poorest to richest
            # cw = cumulative population share; L = Lorenz curve (cumulative wealth share).
            cw = np.cumsum(ws); cx = np.cumsum(xs * ws); L = cx / cx[-1]
            Lp = np.concatenate([[0], L[:-1]]); gini = 1 - np.sum(ws * (L + Lp))   # trapezoid area rule
            cw0 = np.concatenate([[0.0], cw]); L0 = np.concatenate([[0.0], L])
            # Top share f = 1 - Lorenz value at population share 1 - f (linear interpolation).
            # PYTHON NOTE: `lambda f: ...` defines a small one-line function; np.interp interpolates.
            top = lambda f: 1.0 - np.interp(1.0 - f, cw0, L0)
            return gini, top
        gini, top = shares(x)                                  # total wealth a + d
        gini_liq, top_liq = shares(a)                          # personal (liquid) wealth only
        dmin = p.d_min_usd / self.usd
        # has = 1 for states holding a trust of at least d_min ($100k); has2 = same, in tier 2.
        has = ((cat > 0) & (d >= dmin)).astype(float)
        has2 = has * (tier == 2)
        # estates and transfer taxes
        # Accumulators (all as mass shares of the population, model units for amounts, per 5 years):
        deaths = taxable = est_val = est_tax = charity = tax_gross = tax_char = 0.0
        big_tot = big_tax = big_trust = 0.0; gift_tax = 0.0; freeze = 0.0
        gifts = givers = 0.0; gift_pop = 1e-300
        # Estate size bins in DOLLARS of gross liquid estate: $10-20M, $20-50M, $50M+, $10M+.
        bins = [(10e6, 20e6), (20e6, 50e6), (50e6, np.inf), (10e6, np.inf)]
        btax = np.zeros(len(bins)); bgross = np.zeros(len(bins))
        for t in range(T):
            mass = self.dist[t] * self.cohort[t] / tot         # population share by state at age t
            # q_all[st, k, r] = population mass in state st, shock k, taking option r.
            q_all = mass[:, None, None] * self.wxi[None, :, None] * self.P[t]
            gift_tax += (q_all * self.TX[t]).sum()             # gift tax (trust funding and gifts)
            if self.GF[t] is not None:
                # Gift age: total lifetime gifts, mass of givers, and the mass at that age.
                gifts += (q_all * self.GF[t]).sum()
                givers += (q_all * (self.GF[t] > 0)).sum(); gift_pop += mass.sum()
            freeze += (mass[:, None] * self.wxi[None, :] * self.FR[t]).sum()   # GRAT transfers
            if self.sur[t] >= 1.0:
                continue                                       # nobody dies at this age
            q = q_all * (1 - self.sur[t])                      # mass dying at the end of age t
            self.estates(t, self.dist[t])                      # make sure BN, GC are filled
            # Per option: liquid estate ap (= a' saved), trust dp, next category cp; charity g, net bequest.
            ap, dp, cp = self.AP[t], self.DP[t].astype(float), self.JC[t]
            g, net = self.GC[t], self.BN[t]
            # Remaining exemption (founders have used part of it on the trust) and valuation discount,
            # exactly as in _estates_fill / _bq_grid.
            Ee = np.where((self.role_c[cp] == 2) & bool(p.shield), np.maximum(self.E - dp, 0.0), self.E)
            kap = np.where((self.tier_c[cp] == 2) & (self.zof[(np.arange(self.N) % (p.ny * self.nz)) % self.nz][:, None, None] > 0),
                           1 - p.disc_v, 1.0)
            deaths += q.sum(); est_val += (q * (ap + dp)).sum()   # number of deaths; value of all estates
            # Taxable estate: discounted liquid estate net of charity above the remaining exemption.
            # Without the shield ("noshield") trust assets count in the taxable estate, as in _beq1
            # (fixed after the first review; KNOWN_ISSUES item 2).
            tx = kap * (ap - g) + (0.0 if p.shield else dp) > Ee
            taxable += q[tx].sum(); tax_gross += (q * ap)[tx].sum(); tax_char += (q * g)[tx].sum()
            et = ap - g - net                                  # estate tax paid = estate - charity - net
            est_tax += (q * et).sum(); charity += (q * g).sum()
            # "Big" estates: total (liquid + trust) above the exemption; trust share and effective tax.
            big = (ap + dp) > self.E
            big_tot += (q * (ap + dp))[big].sum(); big_tax += (q * et)[big].sum(); big_trust += (q * dp)[big].sum()
            apu = ap * self.usd                                # UNITS: gross liquid estate in dollars
            for bi, (lo, hi) in enumerate(bins):
                sel = (apu >= lo) & (apu < hi)
                btax[bi] += (q * et)[sel].sum(); bgross[bi] += (q * ap)[sel].sum()
        # Mean total wealth by age (per person alive at that age), and population mass by age.
        Wage = (self.dist * x[None, :]).sum(axis=1) / self.dist.sum(axis=1)
        mass_age = self.dist.sum(axis=1) * self.cohort
        i75 = (75 - p.age0) // p.tlen                          # age index of 75-79
        # Mean wealth of people aged 75+ divided by mean wealth of everyone (target wealth_75plus/mean).
        old = (Wage[i75:] * mass_age[i75:]).sum() / mass_age[i75:].sum() / K
        retirees = self.age_w[p.TL:].sum()                     # population share aged 65+
        # Earnings distribution of workers (ages before TL): weights ew, gross 5-year earnings ex.
        ew, ex = [], []
        for t in range(p.TL):
            ew.append(self.age_w[t] * self.pi_age[t]); ex.append(self.gross_earn[t].reshape(-1))
        ew = np.concatenate(ew); ex = np.concatenate(ex); ew = ew / ew.sum()
        oe = np.argsort(ex); cwe = np.concatenate([[0.0], np.cumsum(ew[oe])])
        Le = np.concatenate([[0.0], np.cumsum(ew[oe] * ex[oe])]); Le /= Le[-1]
        top1_earn = 1.0 - np.interp(0.99, cwe, Le)             # top-1% earnings share
        # Earnings beyond the standard wage x efficiency x productivity (the entrepreneurs' earnings
        # multiplier earn_mult, all 1.0 by default): counted as business output. Annual (/ tlen).
        base_earn = (np.concatenate([self.age_w[t] * self.pi_age[t] * (p.tlen * self.w * self.eff[t]
                     * np.repeat(np.exp(self.y), self.nz)) for t in range(p.TL)])).sum()
        tot_earn = sum((self.age_w[t] * self.pi_age[t] * self.gross_earn[t].reshape(-1)).sum() for t in range(p.TL))
        extra_earn = (tot_earn - base_earn) / p.tlen
        Yb = Yb + extra_earn
        Y = self.Yc + Yb                                       # total annual output: corporate + business
        # capital income taxes by category: liquid wealth and grantor-trust income at tau_c,
        # non-grantor trust income at tau_ng_c
        tl = self.tauc[cat]; tt = np.where(role == 1, self.taungc[cat], self.tauc[cat])
        cap_tax = (w * ret * (tl * a + tt * d)).sum()          # annual, on expected returns
        # Government revenue (annual): labour tax + capital tax + (estate + gift taxes) per year.
        rev = p.tau_l * (self.w * self.L + extra_earn) + cap_tax + (est_tax + gift_tax) / p.tlen
        pens = retirees * p.pens * self.ebar / p.tlen          # annual pension spending
        # Effective estate-tax rate by gross-estate bin (0 where a bin is empty).
        bin_rate = np.where(bgross > 0, btax / np.maximum(bgross, 1e-300), 0.0)
        return {
            # --- capital market: target K_gap = (K_supply - Kb) / Kc_demand - 1 = 0 (computed by callers)
            "K_supply": K, "Kb": Kb, "Kc_demand": self.Kc, "K_supply-Kb": K - Kb,
            "K/Y (total)": K / Y, "Kb/K": Kb / K,
            # --- top shares of TOTAL wealth a + d
            "gini": gini, "top0.1": top(0.001), "top1": top(0.01), "top5": top(0.05),
            "top10": top(0.10), "top20": top(0.20), "bottom50": 1 - top(0.50),
            # --- top shares of PERSONAL (liquid) wealth a only (targets before v2c)
            "top1_liquid_only": top_liq(0.01), "top0.1_liquid_only": top_liq(0.001),
            "top10_liquid_only": top_liq(0.10),
            # --- DFA-consistent top shares (v2c targets; PAPER 3.5). PYTHON NOTE: **{...} pours the
            # entries of another dict into this one; zip pairs the three names with the three values.
            **{k: v for k, v in zip(("top1_dfa", "top0.1_dfa", "top10_dfa"),
                                    dfa_top_shares(w, a, d, p.dfa_theta, p.scf_trust_share).values())},
            "share_zero": w[x <= 1e-12].sum(),
            "share_entrepreneurs": w[z > 0].sum(),
            # --- estates (targets: share_deaths_taxable in logs, charity/taxable_estates)
            "share_deaths_taxable": taxable / deaths,
            "charity/taxable_estates": tax_char / tax_gross if tax_gross > 0 else 0.0,
            "bequest_flow/Y": est_val / p.tlen / Y,
            "wealth_75plus/mean": old,                       # target
            # --- mobility (not targets)
            "parent_child_wealth_elasticity": self.wealth_elasticity(),
            "elasticity_child_50_54": self.wealth_elasticity(pairs=self.pairs6),
            "rank_rank_child_35_39": self.rank_rank(self.pairs3),
            "rank_rank_child_50_54": self.rank_rank(self.pairs6),
            "earnings_rank_rank": self.earnings_rank_rank(),
            "top1_earnings_share": top1_earn,
            # --- trusts: size, take-up (reported), and trust shares of top wealth (targets)
            "trust_assets/wealth": Kd / K,
            "takeup_all": (w * has).sum(),
            "share_heir_trusts": (w * (role == 1) * has).sum(), "share_founder_trusts": (w * (role == 2) * has).sum(),
            "trust_assets_heirs/trust_assets": (w * d * (role == 1)).sum() / max(Kd, 1e-300),
            "takeup_top0.1": self._group_share(x, w, has, 0.0, 0.001),
            "takeup_top1": self._group_share(x, w, has, 0.0, 0.01),
            "takeup_90_99": self._group_share(x, w, has, 0.01, 0.10),
            "tier2_all": (w * has2).sum(),
            "tier2_top1": self._group_share(x, w, has2, 0.0, 0.01),
            "tier2_top0.1": self._group_share(x, w, has2, 0.0, 0.001),
            "trust_share_of_top1_wealth": self._group_ratio(x, w, d, x, 0.0, 0.01),
            "trust_share_of_top0.1_wealth": self._group_ratio(x, w, d, x, 0.0, 0.001),
            # --- estate taxes (target: estate_tax/gross_50M+)
            "eff_estate_tax_big_estates": big_tax / big_tot if big_tot > 0 else 0.0,
            "trust_share_big_estates": big_trust / big_tot if big_tot > 0 else 0.0,
            "estate_tax/gross_10_20M": bin_rate[0], "estate_tax/gross_20_50M": bin_rate[1],
            "estate_tax/gross_50M+": bin_rate[2], "estate_tax/gross_10M+": bin_rate[3],
            # --- flows relative to annual output (5-year flows divided by tlen)
            "estate_tax/Y": est_tax / p.tlen / Y, "gift_tax/Y": gift_tax / p.tlen / Y,
            "freeze_transfers/Y": freeze / p.tlen / Y,
            "gifts/Y": gifts / p.tlen / Y,
            # --- lifetime gifts (target: inter_vivos_share = gifts / (gifts + all estates incl. trusts))
            "inter_vivos_share": gifts / max(gifts + est_val, 1e-300),
            "share_giving_at_gift_age": givers / gift_pop,
            # --- checks: mean wealth in $ (should be the SCF mean, $1.0637M, when usd is set from it),
            # implied government spending, mass piled at the top of the grids (should be ~0)
            "mean_wealth_usd_check": K * self.usd,
            "implied_g/Y": (rev - pens) / Y,
            "mass_at_grid_top": w[(a >= self.agrid[-3]) | (d >= self.dgrid[-3])].sum(),
            "mean_wealth_by_age": Wage / K,
        }


# ==============================================================================================
# SECTION 6: CONVENIENCE BUILDER
# ==============================================================================================
def buildC(p, eflife, surlife, D0=None, verbose=False):
    """Build, solve and find the stationary distribution in one call; returns the EconomyC.
    D0: optional starting guess for the parents' distribution (speeds up repeated solves).
    Uses distribution()'s default tolerances (tol = 1e-10, no wealth-weighted test)."""
    return EconomyC(p, eflife, surlife).solve().distribution(D0=D0, verbose=verbose)



# ==============================================================================================
# SECTION 7: DFA-CONSISTENT TOP SHARES (PAPER 3.5) AND REPORTING TABLES
# ==============================================================================================
def dfa_top_shares(w, a, d, theta=0.509, scf_share=0.0231, fracs=(0.01, 0.001, 0.10)):
    """Top wealth shares measured the way the Fed's Distributional Financial Accounts measure them.
    The Financial Accounts household sector includes personal trusts; the DFA allocates those totals
    to households by SCF holdings, asset class by asset class. Model analogue: each household's
    personal wealth a, plus the trust assets the SCF records (a share c of trust wealth, set so that
    recorded trusts are scf_share of total wealth, as in SCF 2022), attributed to the holding household,
    plus the remaining trust wealth spread across households in proportion to a^eta, with eta set
    so that the top 1% of personal wealth receives share theta (DFA: the top 1% hold 50.9% of
    corporate equities and mutual fund shares, Q2 2026). Returns {frac: share}.

    PAPER: Section 3.5. Step by step:
      1. A = mean personal wealth, Dt = mean trust wealth; tau = Dt / (A + Dt) = trust share of wealth.
      2. c = min(1, scf_share / tau): the share of each trust the SCF "sees", so that seen trusts are
         2.31% of total wealth. That part is credited to the family that owns the trust.
      3. The unseen part (1 - c) of all trust wealth is not lost: the DFA hands it out the way it
         hands out equities, which are concentrated. Model: in proportion to a^eta.
      4. eta is found by bisection so that the top 1% by personal wealth gets theta = 50.9% of it.
      5. Measured wealth x = a + c*d + (1 - c)*Dt*g(a), with g = a^eta normalised to mean 1.
      6. Top shares of x, from its Lorenz curve.
    Inputs: w = population share of each state (sums to 1); a, d = personal and trust wealth of each
    state (model units; units cancel); fracs = which top shares to return (1%, 0.1%, 10%)."""
    A = (w * a).sum(); Dt = (w * d).sum()                     # step 1: mean personal and trust wealth
    if Dt <= 0:
        x = a                                                  # no trusts: measured wealth = personal
    else:
        tau = Dt / (A + Dt)
        c = min(1.0, scf_share / tau)                          # step 2: SCF-recorded share of trusts
        # Personal-wealth threshold of the top 1%: sort, accumulate population, find where it hits 99%.
        # PYTHON NOTE: np.searchsorted(sorted, v) gives the position where v would be inserted.
        o = np.argsort(a, kind="stable"); cw = np.cumsum(w[o])
        thr = a[o][min(np.searchsorted(cw, 0.99), a.size - 1)]
        # APPROXIMATION: on a grid with lumpy mass, "a >= thr" can be somewhat more than exactly 1%.
        top = a >= thr
        pos = np.where(a > 0, a, 0.0)
        # Step 4: bisection for eta in [0, 4] (60 halvings): the top-1% share of a^eta rises with eta.
        # NUMERICS: bisection = repeatedly halve the interval keeping the half that contains the answer.
        lo, hi = 0.0, 4.0
        for _ in range(60):
            eta = 0.5 * (lo + hi); g = pos ** eta
            sh = (w * g)[top].sum() / (w * g).sum()
            if sh < theta:
                lo = eta
            else:
                hi = eta
        # Step 5: allocation weights with mean 1 (sum of w * g = 1), then measured wealth.
        g = pos ** eta; g = g / (w * g).sum()
        x = a + c * d + (1.0 - c) * Dt * g
    # Step 6: Lorenz curve of x and the requested top shares (1 - Lorenz at 1 - f).
    o = np.argsort(x, kind="stable"); xs, ws = x[o], w[o]
    cw0 = np.concatenate([[0.0], np.cumsum(ws)]); L0 = np.concatenate([[0.0], np.cumsum(xs * ws)]); L0 /= L0[-1]
    return {f: 1.0 - np.interp(1.0 - f, cw0, L0) for f in fracs}


def choice_shares(e):
    """Mass-weighted shares of the trust choices among households holding a trust, by role:
    payout of income, wind-up, large contribution, other (mostly retain).

    Input: a solved EconomyC e with its distribution. Output: {"heirs": {...}, "founders": {...}}
    (None for a role with no mass). Classification of each kept option, comparing the trust after
    the choice d' (DP) with the trust at the START of the period d:
      wind_up_or_terminate: next category 0 (includes forced termination below d_min);
      large_contribution:   d' > 1.02 x 1.5 x d (more than 53% above the starting value);
      payout_income:        d' <= d (trust brought back to its starting value: the income is paid out);
      other_retain:         everything else (retain, small contributions, up/downgrades).
    Caveat: these are thresholds, not option labels. A retained trust after a very good 5-year return
    can exceed 1.53 d and count as "large contribution"; after a loss (D < d) a retained trust can
    count as "payout". Uses only the ko kept options per state."""
    p = e.p; out = {}
    for rl, name in ((1, "heirs"), (2, "founders")):
        tot = pay = ret = con = wind = 0.0
        for t in range(p.T):
            v = e.dist[t] * e.cohort[t]                        # mass by state at age t
            sel = np.nonzero((e.frole == rl) & (v > 0))[0]     # states of this role with mass
            if sel.size == 0:
                continue
            d = e.fd[sel]
            for k in range(p.nxi):
                # Mass taking each kept option: (selected states, ko).
                Pk = e.P[t][sel, k] * v[sel, None] * e.wxi[k]
                DPk = e.DP[t][sel, k]; JCk = e.JC[t][sel, k]
                tot += Pk.sum()
                wind += Pk[JCk == 0].sum()
                keep = JCk > 0
                # PYTHON NOTE: d[:, None] makes d a column so it compares with each of the ko options.
                con += Pk[keep & (DPk > 1.02 * d[:, None] * 1.5)].sum()
                pay += Pk[keep & (DPk <= d[:, None] * 1.0000001)].sum()
        if tot <= 0:
            out[name] = None
            continue
        out[name] = {"payout_income": pay / tot, "wind_up_or_terminate": wind / tot, "large_contribution": con / tot,
                     "other_retain": 1 - (pay + wind + con) / tot}
    return out


def founding_table(e, ages=(2, 5, 8, 10, 12), usd_levels=(5e6, 15e6, 30e6, 60e6, 150e6, 500e6, 2e9, 10e9), z=None, y=None):
    """Probability that a family without a trust founds one this period, by age and liquid wealth
    (averaged over return shocks; median earnings and normal return state by default).

    ages: age indices t (2 = ages 30-34 with age0 = 20); usd_levels: liquid wealth in DOLLARS (the
    nearest grid point is used, and its dollar value is the column label); z, y: composite return
    state and earnings state. Founding = any kept option whose next category is 2 (founder, tier 1).
    Output: {"age 30-34": {"$5M": prob, ...}, ...}. Probabilities are per 5-year period. The label
    arithmetic assumes 5-year periods."""
    p = e.p; nyz = p.ny * e.nz
    y = p.ny // 2 if y is None else y; z = 0 if z is None else z   # PYTHON NOTE: // is whole-number division
    ys = y * e.nz + z                                          # combined (y, z) index
    rows = {}
    for t in ages:
        row = {}
        for usd in usd_levels:
            i = int(np.argmin(np.abs(e.agrid * e.usd - usd)))  # nearest liquid grid point
            st = i * nyz + ys                                  # flat index in the category-0 block
            # Sum over shocks k of P(options leading to category 2), weighted by shock probability.
            pf = sum(e.wxi[k] * e.P[t][st, k][e.JC[t][st, k] == 2].sum() for k in range(p.nxi))
            row[f"${e.agrid[i] * e.usd / 1e6:,.0f}M"] = pf     # label: grid point in $ millions
        rows[f"age {e.p.age0 + 5 * t}-{e.p.age0 + 4 + 5 * t}"] = row
    return rows


def upgrade_table(e, ages=(5, 8, 10, 12), usd_levels=(30e6, 100e6, 300e6, 1e9, 3e9, 10e9), trust_share=0.3, z=None, y=None):
    """Probability that a tier-1 founder moves to tier 2 this period, by age and total wealth
    (trust assets = trust_share of total; median earnings, normal return state by default).

    Only meaningful with tier2 = True (family offices; OFF in all reported results). usd_levels are
    total wealth in DOLLARS, split (1 - trust_share) liquid / trust_share in trust, each placed at its
    nearest grid point. Upgrade = any kept option whose next category is 4 (founder, tier 2).
    Output: {"age 45-49": {"$30M": prob, ...}, ...}, probabilities per 5-year period."""
    p = e.p; nyz = p.ny * e.nz; na, nd = e.agrid.size, p.nd
    y = p.ny // 2 if y is None else y; z = 0 if z is None else z
    ys = y * e.nz + z
    rows = {}
    for t in ages:
        row = {}
        for usd in usd_levels:
            i = int(np.argmin(np.abs(e.agrid * e.usd - usd * (1 - trust_share))))   # liquid grid point
            j = int(np.argmin(np.abs(e.dgrid * e.usd - usd * trust_share)))         # trust grid point
            # Flat index: after the category-0 block, block (c - 1) = 1 for category 2, then (i, j, ys).
            st = e.N0 + ((1 * na + i) * nd + j) * nyz + ys           # category 2 (founder, tier 1)
            pu = sum(e.wxi[k] * e.P[t][st, k][e.JC[t][st, k] == 4].sum() for k in range(p.nxi))
            row[f"${usd / 1e6:,.0f}M"] = pu
        rows[f"age {e.p.age0 + 5 * t}-{e.p.age0 + 4 + 5 * t}"] = row
    return rows
