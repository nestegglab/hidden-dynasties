"""
Layer B of the FODT v2 model: Layer A + fortune creation (reduced-form entrepreneurship).

Return state z in {N (normal), L (entrepreneur), H (high-growth entrepreneur)}:
  * wealth up to a business scale k_bar earns a risky business return:
        gross 5-year return G = (1 + r + pi_z)^5 * exp(s_z * xi - s_z^2 / 2),  s_z = sigma_z * sqrt(5)
    (log-normal: modest mean premium pi_z, high concentrated risk, a right-skewed payoff;
     see Moskowitz & Vissing-Jorgensen 2002; Benhabib, Bisin & Luo 2019)
  * wealth above k_bar, and all wealth in state N, earns the market return r
  * capital income tax tau_a on all returns (losses offset)
  * z follows a Markov chain within a life (N->L, L->H, single exit probability back to N)
    and is partly inherited: a child of an entrepreneur starts in L with probability iota
  * the extra returns come from a private-business sector, so total capital = corporate
    capital K_c + business capital K_b and r is set by the corporate sector (spec, Production)

Everything else (demographics, bequests, charity, estate tax, earnings, GE prices) is as in
Layer A. With nz = 1 the model reduces to Layer A.

Stationary distribution (direct): the parent-child joint distribution is kept in FACTORISED
form. Given the child's initial type (y0, z0), the child's own path is independent of the
parent's path until the parent dies, so the joint distribution is a sum over the (ny*nz)
initial types of (child distribution) x (parent distribution). This is exact and makes the
direct computation cheap even with many states.
"""
# ==============================================================================================
# FILE BANNER: what this file is, in plain words
# ==============================================================================================
# WHAT IT DOES
#   Layer B is the "where do big fortunes come from" layer. It takes the Layer A life-cycle model
#   (work, retire, save, die, leave bequests) and adds a RETURN STATE z for each household:
#   N (normal saver, earns the market return r), L (entrepreneur, earns a risky business return)
#   and H (high-growth entrepreneur: very risky business return AND a large "superstar"
#   multiplier on labour income while working). Households move N -> L -> H by chance and fall
#   back to N with an exit probability. This file sets up prices and the dollar scale, solves
#   the saving problem, computes the long-run (stationary) wealth distribution across
#   generations, and reports moments (top shares, Gini, estates, mobility, ...).
#
# WHERE IT SITS
#   Layer A (layerA.py: life cycle, earnings, bequests)
#     -> Layer B (THIS FILE: fortunes via return states, prices, `usd`, distribution)
#       -> Layer C (layerC.py: dynasty trusts on top of Layer B)
#         -> drivers (calibrate_B2.py, calibrate_C.py, equilibrium.py, run_c5.py, run_sweep.py, ...)
#
# WHAT LAYER C REUSES FROM HERE AND WHAT IT REPLACES
#   ParamsC (layerC.py) inherits every field of ParamsB below, plus ParamsB.r/Pz/Pz0.
#   EconomyC (layerC.py) inherits from EconomyB and:
#     REUSES (calls super().__init__, so all of this runs unchanged for the final model):
#       __init__ (age-efficiency profile, survival incl. "no deaths before 60", earnings chain,
#       return-state chain, child-entry kernel, sibling probability q2, cohort weights,
#       Gauss-Hermite nodes), _exogenous (pi_age, Pyz, Kbirth), _prices (r, wage, ebar, income
#       `inc` and `gross_earn` incl. the superstar multiplier, `usd`, E, the wealth grid `agrid`,
#       after-tax returns Rn and Gt), and earnings_rank_rank.
#     OVERRIDES (has its own version): solve, estates, distribution, wealth_elasticity, moments.
#     DOES NOT USE (Layer B only; Layer C has its own Numba kernels instead): bequest (replaced
#       by _beq1, which reproduces it when there is no trust), coh, dcoh, cfun, euler_errors,
#       lottery, transition, _states, _estate_hist, _convolve, buildB, calibrate_beta_B,
#       monte_carlo_B, _mc_genB.
#
# !!! THE PARAMETER VALUES USED BY THE FINAL MODEL ARE NOT THE DEFAULTS IN ParamsB !!!
#   The final model (calibrate_C.base_params) builds a ParamsB, then overwrites the Layer B
#   parameters with the Layer B calibration stored in results/layerB_v2_params_bestof2.json,
#   via calibrate_B2.to_params. Those calibrated values are (annual unless stated):
#     mean premium over r:       N 0,  L 1% (fixed, not estimated),  H ~0% (2.7e-6)
#     volatility (sd of log return): N 0, L 6.9%, H 61.7%
#     N -> L 6.9% per 5-year period;  L -> H 3.1% per period;  exit back to N 20% per period
#     child of an entrepreneur (L or H parent) starts in L with prob iota = 0.20;
#     child of an N parent starts in L with prob iota0 = p_NL = 0.069
#     H owners earn 12.8 x the normal labour income while working (earn_mult = (1, 1, 12.8))
#   The same file also holds beta_annual, phi1 and chi, but calibrate_C re-estimates those three.
#   calibrate_C also fixes mean_wealth_units = Kc / (1 - 0.305) (about 1.44 * Kc; the default
#   here is 1.4 * Kc) and sets nxi from its FAST settings.
#
# READING ORDER
#   1. ParamsB (all parameters, defaults only)  2. EconomyB.__init__  3. _exogenous  4. _prices
#   5. coh / dcoh (cash on hand)  6. bequest  7. solve + cfun (household problem, EGM)
#   8. lottery, transition, estates, distribution (+ the Numba helpers _estate_hist, _convolve)
#   9. moments and its helpers (_states, wealth_elasticity, earnings_rank_rank)
#   10. drivers: buildB, calibrate_beta_B; check: euler_errors, monte_carlo_B / _mc_genB
#
# PAPER SECTIONS
#   3.1 (demographics, preferences, earnings, entrepreneurial returns): ParamsB, __init__,
#       _exogenous, _prices.  3.3 (firms, government, equilibrium): _prices, moments.
#   3.4 (calibration): the Layer B values come from calibrate_B2.py.
#   Appendix A (computation): solve, distribution and the lottery/convolution helpers.
#
# WORDS YOU'LL SEE
#   period       = 5 years. t = age index, 0 = ages 20-24 ... 13 = ages 85-89 (T = 14).
#   model units  = multiples of model output; `usd` converts to dollars; `ebar` = mean 5-year
#                  earnings of a worker, in model units (preference shifters are scaled by it).
#   y            = earnings (productivity) state, 5 values (Rouwenhorst); exp(y) is productivity.
#   z            = return state 0 N, 1 L, 2 H. With dynastic types (nk = 2, OFF in the final model)
#                  the "composite" state s = z * nk + k also carries the bequest type k.
#   ys / nyz     = combined (y, z) index / count; flattened index = y * nz + z.
#   full state   = (wealth grid point i, y, z), flattened as i * nyz + y * nz + z (N = na * nyz).
#   xi           = return shock (Gauss-Hermite nodes, probability weights wxi).
#   G / Gt / Rn  = gross 5-year return pre-tax / after tax (business) / after tax (market).
#   kbar         = business scale limit (OFF: 1e15 dollars by default and in Layer C).
#   EGM          = endogenous grid method (see `solve`).
#   lottery      = splitting a wealth value between the two nearest grid points (see `lottery`).
#   factorised   = parent-child joint distribution stored as (child vector, parent vector) pairs.
#   q2           = probability a child has a sibling (then the estate is split in two).
# ==============================================================================================

# PYTHON NOTE: `import x` loads a library. numpy (np) = arrays and maths; scipy.sparse (sp) =
# sparse matrices (mostly zeros, stored compactly); numba's njit compiles a Python function to
# fast machine code ("just-in-time"), used for the heavy loops at the bottom of this file.
import math
from dataclasses import dataclass, replace, field
import numpy as np
import scipy.sparse as sp
from numba import njit

# Two helpers from Layer A: rouwenhorst = discretised AR(1) earnings process (grid + transition
# matrix); normal_bins = probabilities of a normal distribution falling into bins around grid points.
from layerA import rouwenhorst, normal_bins


# ==============================================================================================
# PARAMETERS
# ==============================================================================================
# PYTHON NOTE: `@dataclass` turns a class into a simple record of named fields with defaults.
# `ParamsB()` gives all defaults; `ParamsB(phi1=-3.0)` changes one; `replace(p, phi1=-3.0)`
# makes a copy of p with one field changed (used below and in the calibration drivers).
@dataclass
class ParamsB:
    """All parameters of Layer B, with DEFAULT values.

    !!! WARNING: these defaults are NOT the values used in the paper's final model. !!!
    The final model overwrites the return-state block (prem, sig, p_NL, p_LH, iota, iota0,
    earn_mult) with the Layer B calibration in results/layerB_v2_params_bestof2.json (via
    calibrate_B2.to_params), and beta_annual, phi1, chi with the Layer C calibration.
    See the file banner for the calibrated numbers. Many other defaults (taxes, technology,
    earnings, grid) ARE used as they stand.

    UNITS: rates are annual unless the comment says "per 5-year period". Fields ending in
    `_usd` are in dollars and are turned into model units in EconomyB._prices. Fields ending in
    `_units` are multiples of mean 5-year earnings (ebar).
    """
    # ---- demographics ----
    tlen: int = 5                     # years per model period
    T: int = 14                       # number of periods in a life (ages 20-89 with age0 = 20)
    age0: int = 20                    # age at model entry (v3: 25, so that life runs to 94 with T = 14)
    TL: int = 9                       # working periods; retirement after TL periods (age 65)
    gap: int = 7                      # parent's age index when the child enters (t = 7, age 55)
    n_pop: float = 0.01               # annual population growth (sets cohort sizes and q2)
    # ---- preferences ----
    sigma: float = 1.5                # CRRA curvature (inverse intertemporal elasticity)
    # DEFAULT ONLY: final value comes from calibration (Layer C re-estimates beta_annual).
    beta_annual: float = 0.9345       # annual discount factor
    # DEFAULT ONLY: final value comes from calibration (Layer C re-estimates phi1).
    phi1: float = -14.62              # bequest strength (negative with sigma > 1; PAPER: phi1)
    phi2_units: float = 8.0           # bequest luxury shifter phi2, in units of ebar
    # DEFAULT ONLY: final value comes from calibration (Layer C re-estimates chi).
    chi: float = 0.0                  # charity strength (0 = no charitable bequests)
    phi2c_units: float = 8.0          # charity luxury shifter phi2c, in units of ebar
    # ---- taxes ----
    tau_a: float = 0.18               # capital-income tax on all returns (losses offset)
    tau_l: float = 0.301              # labour-income tax
    tau_b: float = 0.40               # estate tax rate above the exemption
    E_usd: float = 30e6               # estate-tax exemption per family, dollars
    scf_mean_nw_usd: float = 1_063_700.0   # SCF mean net worth, dollars (anchors `usd`)
    pens: float = 0.40                # pension per retiree, as a share of ebar (mean 5-year earnings)
    # ---- technology (corporate sector) ----
    alpha: float = 0.36               # capital share in the Cobb-Douglas production function
    delta: float = 0.06               # annual depreciation rate
    KY: float = 3.0                  # corporate K_c / Y_c (sets r = 6%)
    # ---- earnings ----
    rho_y: float = 0.85               # persistence of log earnings, per 5-year period
    sig2_y: float = 0.30              # unconditional variance of log earnings
    rho_h: float = 0.5453             # parent-to-child correlation of log earnings at entry
    sig2_h: float = 0.37              # variance of the child's entry draw around rho_h * parent's y
    ny: int = 5                       # number of earnings states
    # ---- return states (index 0 = N, 1 = L, 2 = H) ----
    # !!! DEFAULT ONLY - the whole block below is overwritten by the Layer B calibration. !!!
    # Final values: prem = (0, 0.01, ~0); sig = (0, 0.069, 0.617); p_NL = 0.069; p_LH = 0.031;
    # q_exit = 0.20 (this one is kept); iota = 0.20; iota0 = p_NL = 0.069.
    nz: int = 3                          # number of return states (1 = Layer A, no entrepreneurs)
    prem: tuple = (0.0, 0.01, 0.03)      # annual mean premium over r
    sig: tuple = (0.0, 0.25, 0.45)       # annual sd of log business return
    p_NL: float = 0.04                   # per 5-year period
    p_LH: float = 0.10                   # per 5-year period: probability L -> H
    q_exit: float = 0.20                 # per 5-year period: probability L or H -> N
    iota: float = 0.30                   # child of L/H parent starts in L
    iota0: float = 0.03                  # child of N parent starts in L
    kbar_usd: float = 1e15               # business scale limit (effectively none by default)
    # ---- bequest types (index 0 = standard, 1 = dynastic); nk = 1 switches types off ----
    # (OFF in the final model: nk = 1, so the next three fields are not used.)
    nk: int = 1
    phi1_dyn: float = -30.0              # bequest intensity of dynastic families
    share_dyn: float = 0.2               # population share of dynastic families
    keep_type: float = 0.9               # probability a child inherits the parent's type
    # labor-earnings multiplier by return state while working ("superstar" owner income,
    # Smith, Zidar & Zwick 2019: most top pass-through profit is a return to the owner's human capital)
    # !!! DEFAULT ONLY: the final model uses (1.0, 1.0, 12.8), i.e. H owners earn 12.8x. !!!
    earn_mult: tuple = (1.0, 1.0, 1.0)
    nxi: int = 7                         # Gauss-Hermite nodes for return shocks
    # ---- grid ----
    na: int = 160                        # number of wealth grid points
    a_lo: float = 0.002                  # units of mean 5-year earnings
    a_max_usd: float = 500e9             # top of the wealth grid, dollars
    mean_wealth_units: float = None      # model mean wealth used for $ conversion (updated by calibration)
    # (None -> _prices uses 1.4 * Kc as a placeholder; the final model passes Kc / (1 - 0.305).)

    def r(self):
        """Annual interest rate implied by the corporate capital-output ratio.

        PAPER 3.3: with Cobb-Douglas, the marginal product of capital is alpha * Y / K, so
        r = alpha / KY - delta (0.36 / 3 - 0.06 = 0.06, i.e. 6% a year by default).
        Note: in the experiments r is re-solved elsewhere (equilibrium.py) by changing KY.
        """
        return self.alpha / self.KY - self.delta

    def Pz(self):
        """Within-life transition matrix of the return state, per 5-year period.

        Output: 3 x 3 array, row = today's state (N, L, H), column = next period's state.
        Each row sums to 1. N -> L with p_NL; L -> H with p_LH; L or H -> N with q_exit.
        There is no direct N -> H and no H -> L. With nz = 1 it is the 1 x 1 matrix [[1]].
        """
        if self.nz == 1:
            return np.ones((1, 1))
        q, a, b = self.q_exit, self.p_NL, self.p_LH       # short names for readability
        return np.array([[1 - a, a, 0.0],          # from N: stay N, or start a business (L)
                         [q, 1 - q - b, b],        # from L: exit to N, stay L, or grow to H
                         [q, 0.0, 1 - q]])         # from H: exit to N, or stay H

    def Pz0(self):
        """Child's starting return state given the parent's state (when the child enters).

        Output: 3 x 3 array, row = parent's state (N, L, H), column = child's starting state.
        Children never start in H. A child of an N parent starts in L with prob iota0, a child
        of an L or H parent with prob iota (the inherited-entrepreneurship channel, PAPER 3.1).
        """
        if self.nz == 1:
            return np.ones((1, 1))
        return np.array([[1 - self.iota0, self.iota0, 0.0],
                         [1 - self.iota, self.iota, 0.0],
                         [1 - self.iota, self.iota, 0.0]])


# ==============================================================================================
# THE ECONOMY: set-up, exogenous chains, prices
# ==============================================================================================
class EconomyB:
    """One Layer B economy for a given parameter set. Layer C's EconomyC inherits from it.

    Typical use: EconomyB(p, eflife, surlife).solve().distribution().moments()
    (each step stores results on `self` and returns `self`, so the calls can be chained).
    """

    def __init__(self, p: ParamsB, eflife, surlife):
        """Build all exogenous objects (demography, earnings, return states) and prices.

        Inputs:
          p        ParamsB (or ParamsC) parameters.
          eflife   1-D array, ANNUAL age-efficiency profile starting at age 20 (one entry per year).
          surlife  1-D array, ANNUAL survival probability from age x to x+1, starting at age 20.
        Creates (the main ones): eff (efficiency per working period), sur (5-year survival by
        age), y/P (earnings grid and chain), Ph (child's entry earnings given parent's), Pz/Pz0
        (return-state chains), q2, mass, cohort, xi/wxi; then calls _exogenous and _prices.
        """
        self.p = p
        T, TL, L = p.T, p.TL, p.tlen          # L here = years per period (not labour supply)
        off = p.age0 - 20                                # eflife and surlife start at age 20
        # PYTHON NOTE: `[f(i) for i in range(...)]` is a list comprehension: a compact for-loop
        # that builds a list. `x[a:b]` is slicing: elements a, a+1, ..., b-1.
        # Sum annual efficiency over the 5 years of each working period (one entry per period).
        eff = np.array([eflife[off + i:off + i + L].sum() for i in range(0, TL * L, L)])
        self.eff = eff * TL / eff.sum()              # normalise: mean efficiency over working life = 1
        assert off + (T - 1) * L <= surlife.size, "survival data too short for this age range"
        # 5-year survival = product of the 5 annual survival rates; last period: survival 0
        # (everyone dies at the end of period T-1, ages 85-89 with age0 = 20).
        sur = np.array([np.prod(surlife[off + i:off + i + L]) for i in range(0, (T - 1) * L, L)] + [0.0])
        # no deaths up to the end of the period in which a parent's child enters the model (the
        # gift stage at that age assumes the parent survives it): ages < 60 with age0 = 20 (as
        # before), < 65 with age0 = 25
        # APPROXIMATION: survival is set to 1 for t = 0..gap, i.e. nobody dies before 60.
        # The distribution code below relies on this (children and orphans are not hit by
        # survival in the early periods).
        sur[:p.gap + 1] = 1.0
        self.sur = sur                                   # (T,): prob. of surviving from t to t+1
        # Earnings: 5-point grid y (log productivity) and 5 x 5 transition matrix P (per period).
        self.y, self.P = rouwenhorst(p.ny, p.rho_y, p.sig2_y)
        # Ph[i, j] = prob. child's entry earnings state is j given parent's state i (normal with
        # mean rho_h * y_i and variance sig2_h, binned onto the grid). Shape (ny, ny).
        self.Ph = np.array([normal_bins(self.y, p.rho_h * yi, math.sqrt(p.sig2_h)) for yi in self.y])
        nk = p.nk
        self.nzr = p.nz                                  # number of return states
        self.nz = p.nz * nk                              # composite state s = z * nk + k
        # NOTE: from here on `self.nz` counts COMPOSITE states (= p.nz when nk = 1, as in the
        # final model), while `p.nz` / `self.nzr` count return states only.
        self.zof = np.arange(self.nz) // nk              # return state of each composite state
        self.kof = np.arange(self.nz) % nk               # bequest type of each composite state
        if nk == 1:
            Tk = np.ones((1, 1))                         # no types: trivial 1 x 1 "type chain"
        else:
            # Child keeps the parent's type with prob keep_type, otherwise draws a new type
            # from the population shares (1 - share_dyn, share_dyn).
            draw = np.array([1 - p.share_dyn, p.share_dyn])
            # PYTHON NOTE: `draw[None, :]` adds a row dimension so the 1-D vector "broadcasts"
            # (is repeated) across the rows of the 2 x 2 identity matrix.
            Tk = p.keep_type * np.eye(2) + (1 - p.keep_type) * draw[None, :]
        # np.kron (Kronecker product) combines two independent chains into one chain on the
        # pair of states. Types never change within a life (identity), so within-life chain =
        # Pz x I; at birth the type follows Tk.
        self.Pz = np.kron(p.Pz(), np.eye(nk))            # (nz, nz) composite within-life chain
        self.Pz0 = np.kron(p.Pz0(), Tk)                  # (nz, nz) parent state -> child's start
        # PYTHON NOTE: np.where(cond, A, B) picks A where cond is true and B elsewhere.
        self.phi1_s = np.where(self.kof == 1, p.phi1_dyn, p.phi1)   # bequest strength by composite state
        # Children per parent: population growth over the 35-year generation gap (k ~ 1.42).
        kids = (1 + p.n_pop) ** (p.tlen * p.gap)
        # PAPER: q2 = 2(k-1)/k = probability a child has a sibling (families have 1 or 2 children,
        # so the average is k). Estates are split equally among the children.
        self.q2 = 2 * (kids - 1.0) / kids
        # mass[t] = share of an entering cohort still alive at age t (mass[0] = 1). Shape (T,).
        self.mass = np.concatenate([[1.0], np.cumprod(sur[:-1])])
        # cohort[t] = size of the age-t cohort relative to the newest one (older cohorts are
        # smaller because of population growth). Shape (T,).
        self.cohort = (1 + p.n_pop) ** (-p.tlen * np.arange(T))
        # NUMERICS: Gauss-Hermite quadrature replaces the continuous normal return shock xi by
        # nxi points and weights, so that E[f(xi)] ~ sum_k wxi[k] * f(xi[k]). hermegauss is
        # the "probabilists'" version (standard normal); dividing by the sum turns the weights
        # into probabilities that add to 1.
        xi, w = np.polynomial.hermite_e.hermegauss(p.nxi)
        self.xi, self.wxi = xi, w / w.sum()
        self._exogenous()
        self._prices()

    # productivity and return-state distributions by age (exogenous)
    def _exogenous(self):
        """Joint (y, z) chains and the exogenous distribution of (y, z) by age.

        Creates:
          Pyz     (nyz, nyz) within-life chain of the combined (y, z) state, per period.
          Kbirth  (nyz, nyz) row = parent's (y, z) at age 55 (t = gap), column = child's
                  starting (y0, z0) at age 20.
          pi_age  (T, nyz) share of each age-t cohort in each (y, z) state. One row per age.
        These do not depend on wealth or choices (survival is the same in every state), so they
        can be computed before the household problem is solved.
        """
        p = self.p
        nyz = p.ny * self.nz
        # joint (y,z) chain and child-entry kernel
        # (y and z move independently, so the joint chain is the Kronecker product; the combined
        # index is y * nz + z.)
        self.Pyz = np.kron(self.P, self.Pz)                         # (y,z) -> (y',z')
        self.Kbirth = np.kron(self.Ph, self.Pz0)                    # parent's (y,z) at 55 -> child's (y0,z0)
        # NUMERICS: find the stationary distribution of newborn types by iteration: start from
        # equal shares, age the cohort gap periods (to 55), draw children, repeat until the
        # newborn distribution stops changing (1e-15), at most 5000 rounds.
        pi0 = np.ones(nyz) / nyz
        for _ in range(5000):
            d = pi0.copy()
            for _t in range(p.gap):
                # PYTHON NOTE: `@` is matrix multiplication. A row vector of shares times a
                # transition matrix gives next period's shares.
                d = d @ self.Pyz
            new = d @ self.Kbirth                                   # children's starting types
            if np.abs(new - pi0).max() < 1e-15:
                break
            pi0 = new
        # Roll the stationary newborn distribution forward through every age.
        self.pi_age = np.zeros((p.T, nyz)); d = pi0
        for t in range(p.T):
            self.pi_age[t] = d; d = d @ self.Pyz

    def _prices(self):
        """Prices, aggregate labour, income by state, the dollar scale and after-tax returns.

        PAPER 3.3 (firms) and 3.1 (earnings, returns). Creates, among others:
          r_ann        annual interest rate.
          KL, w        corporate capital per efficiency unit of labour; wage per unit (annual).
          age_w        (T,) population share of each age (cohort size x survival).
          L, Kc, Yc    aggregate labour, corporate capital demand, corporate output, per capita
                       and annual, in model units.
          ebar         mean 5-year gross earnings of a worker (model units). Preference shifters
                       and the pension are expressed relative to it.
          gross_earn   (T, ny, nz) gross 5-year labour earnings (0 when retired), incl. the
                       superstar multiplier earn_mult.
          inc          (T, ny, nz) after-tax 5-year income: (1 - tau_l) * gross_earn while
                       working, pension pens * ebar when retired.
          usd          dollars per model unit; E, kbar, agrid converted from dollars with it.
          agrid        (na,) wealth grid in model units: 0 then geometric from a_lo*ebar to $500bn.
          Rn           after-tax gross 5-year market return (scalar).
          Gt           (nz, nxi) after-tax gross 5-year business return, one row per return
                       state, one column per shock node.
        """
        p = self.p
        r = p.r(); self.r_ann = r
        # Firm first-order condition r + delta = alpha * (K/L)^(alpha-1)  =>  K/L.
        self.KL = (p.alpha / (r + p.delta)) ** (1 / (1 - p.alpha))
        self.w = (1 - p.alpha) * self.KL ** p.alpha                  # wage = marginal product of labour
        wts = self.cohort * self.mass                                # relative number alive at each age
        self.age_w = wts / wts.sum()                                 # population share by age
        effy = np.exp(self.y)                                        # productivity level by y state
        # PYTHON NOTE: reshape views the same numbers with a new shape: here (T, nyz) becomes
        # (T, ny, nz); summing over axis 2 (z) leaves the share of each earnings state by age.
        piy = self.pi_age.reshape(p.T, p.ny, self.nz).sum(axis=2)
        # Mean efficiency units of labour supplied at each age (0 when retired).
        lab_t = np.array([(piy[t] @ effy) * self.eff[t] if t < p.TL else 0.0 for t in range(p.T)])
        self.L = (self.age_w * lab_t).sum()                          # labour per capita
        # NOTE: L does not include the superstar multiplier; the extra owner earnings are
        # treated as part of business output (see `moments`, extra_earn).
        self.Kc = self.KL * self.L                                   # corporate capital demand per capita
        self.Yc = self.Kc ** p.alpha * self.L ** (1 - p.alpha)       # corporate output (annual)
        workers = self.age_w[:p.TL].sum()                            # share of population working
        # UNITS: ebar = mean gross labour earnings per worker over a 5-YEAR period, model units.
        self.ebar = p.tlen * self.w * self.L / workers
        u = self.ebar                                                # short name: "unit" of earnings
        # Superstar multiplier by composite state (all 1 when there is only one return state).
        mult = np.array(p.earn_mult if self.nzr > 1 else (1.0,))[self.zof]       # by composite state
        self.gross_earn = np.zeros((p.T, p.ny, self.nz))
        inc = np.zeros((p.T, p.ny, self.nz))
        for t in range(p.T):
            if t < p.TL:
                # 5-year gross earnings = 5 * wage * age efficiency * productivity * multiplier.
                # The `[:, None]` / `[None, :]` broadcasting gives one row per y, one column per z.
                self.gross_earn[t] = p.tlen * self.w * self.eff[t] * effy[:, None] * mult[None, :]
                inc[t] = (1 - p.tau_l) * self.gross_earn[t]          # after labour-income tax
            else:
                # APPROXIMATION: flat pension, same for everyone (40% of ebar), not taxed.
                inc[t] = p.pens * u
        self.inc = inc
        self.beta = p.beta_annual ** p.tlen                          # 5-year discount factor
        self.phi2 = p.phi2_units * u                                 # bequest shifter, model units
        self.phi2c = p.phi2c_units * u                               # charity shifter, model units
        # dollars per model unit: SCF mean net worth = model mean wealth
        # UNITS: usd = $ per model unit. mean_wealth_units is the model's mean wealth (supplied
        # by the calibration); if absent, 1.4 * Kc is used as a placeholder.
        mw = p.mean_wealth_units if p.mean_wealth_units is not None else 1.4 * self.Kc
        self.usd = p.scf_mean_nw_usd / mw
        self.E = p.E_usd / self.usd                                  # estate exemption, model units
        self.kbar = p.kbar_usd / self.usd                            # business scale limit, model units
        amax = p.a_max_usd / self.usd                                # grid top ($500bn), model units
        # Wealth grid: a point at 0, then na-1 points spaced geometrically (equal RATIOS, so
        # dense at low wealth and sparse at billions).
        self.agrid = np.concatenate([[0.0], np.geomspace(p.a_lo * u, amax, p.na - 1)])
        # returns
        R5 = (1 + r) ** p.tlen                                       # pre-tax gross 5-year market return
        # Tax only the net return (R5 - 1); "after-tax gross" = 1 + net return * (1 - tau).
        self.Rn = 1 + (R5 - 1) * (1 - p.tau_a)                     # after-tax gross, market return
        G = np.zeros((self.nz, p.nxi))
        for zs in range(self.nz):
            z = self.zof[zs]                                         # return state of composite state zs
            # 5-year volatility: annual sd * sqrt(5) (independent annual shocks).
            s5 = p.sig[z] * math.sqrt(p.tlen) if self.nzr > 1 else 0.0
            pr = p.prem[z] if self.nzr > 1 else 0.0                  # annual premium over r
            # PAPER 3.1: G = (1 + r + pi_z)^5 * exp(s_z * xi - s_z^2 / 2). The "- s^2/2" makes the
            # shock term average exactly 1, so the MEAN 5-year return is (1 + r + pi_z)^5.
            # In state N, s5 = 0 and pr = 0, so G = R5 (no risk).
            G[zs] = (1 + r + pr) ** p.tlen * np.exp(s5 * self.xi - s5 ** 2 / 2)
        # Losses offset: when G < 1 the tax term is negative (a tax credit).
        self.Gt = 1 + (G - 1) * (1 - p.tau_a)                       # after-tax gross business return (nz, nxi)

    # ==========================================================================================
    # CASH ON HAND AND BEQUESTS
    # ==========================================================================================
    # cash on hand from wealth a in return state z with shock node k (before income)
    def coh(self, a, z, k):
        """Wealth after one period's after-tax return (labour income is added separately).

        Inputs: a = wealth (model units, scalar or array); z = composite return state; k = shock
        node. The first min(a, kbar) earns the business return Gt[z, k], the rest earns Rn.
        Written as: everything at Rn, plus the extra (Gt - Rn) on the business part.
        """
        return self.Rn * a + (self.Gt[z, k] - self.Rn) * np.minimum(a, self.kbar)

    def dcoh(self, a, z, k):
        """Derivative of coh with respect to a: the after-tax gross return on the marginal dollar
        (Gt below kbar, Rn above). Used in the Euler equation."""
        return self.Rn + (self.Gt[z, k] - self.Rn) * (a < self.kbar)

    # bequests (same as Layer A)
    def bequest(self, x, phi1=None):
        """Split an estate x between heirs and charity, and return the marginal value of x.

        phi1: scalar or array broadcastable to x (bequest intensity by type).

        PAPER (bequest motive): phi1 * [(1 + b/phi2)^(1-sigma) - 1] + chi * [1 - (1 + g/phi2c)^(1-sigma)]
        (Layer C adds theta * d inside the first bracket), times sc = ebar^(1-sigma), where
        b = net bequest after estate tax and g = charitable gift (charity is deducted before tax).
        Inputs:  x = gross estate (model units, any array shape).
        Outputs (same shape as x): g = charity, bnet(x - g) = net bequest to heirs,
                 and the marginal value of one more unit of estate at the optimal split.
        """
        p = self.p; s = p.sigma; E = self.E
        x = np.asarray(x, float)
        phi1 = p.phi1 if phi1 is None else phi1
        sc = self.ebar ** (1 - s)                     # rescales the warm glow to model units
        # Net bequest after estate tax: tax tau_b on the part above the exemption E.
        def bnet(z):  return z - p.tau_b * np.maximum(z - E, 0.0)
        # Its derivative: 1 below E, 1 - tau_b above.
        def bder(z):  return np.where(z > E, 1 - p.tau_b, 1.0)
        # Marginal warm glow of the net bequest b (positive because phi1 < 0 and 1 - s < 0).
        def phid(b):  return sc * phi1 * (1 - s) / self.phi2 * (1 + b / self.phi2) ** (-s)
        # Marginal warm glow of charity g.
        def psid(g):  return sc * p.chi * (s - 1) / self.phi2c * (1 + g / self.phi2c) ** (-s)
        g = np.zeros_like(x)
        if p.chi > 0:
            # Charity is positive only where the first dollar to charity beats the last dollar
            # to heirs (after estate tax).
            interior = psid(0.0) > phid(bnet(x)) * bder(x)
            # NUMERICS: bisection on g in [0, x], 60 halvings, done for all x at once: find g with
            # marginal value of charity = marginal value of the net bequest.
            lo = np.zeros_like(x); hi = x.copy()
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                f = psid(mid) - phid(bnet(x - mid)) * bder(x - mid)
                # f > 0: charity still more valuable -> give more (raise lo); otherwise lower hi.
                lo = np.where(f > 0, mid, lo); hi = np.where(f > 0, hi, mid)
            g = np.where(interior, 0.5 * (lo + hi), 0.0)
        zz = x - g                                    # estate left for heirs, before estate tax
        return g, bnet(zz), phid(bnet(zz)) * bder(zz)

    # ==========================================================================================
    # HOUSEHOLD PROBLEM
    # ==========================================================================================
    # ---- household problem (EGM) ----
    def solve(self):
        """Solve the saving problem by backward induction with the endogenous grid method.

        (Layer C overrides this with a trust-aware solver.)
        NUMERICS (EGM in plain words): instead of searching for the best saving a' at each cash
        level m, fix a' on the grid, compute the consumption c that satisfies the Euler equation
        u'(c) = beta * E[ survive * return * u'(c next) + die * marginal bequest value ],
        and then the cash that must have been on hand is m = c + a'. This gives c as a function
        of m on an "endogenous" grid of m values, with no root-finding.
        Stores:
          m_nodes, c_nodes  (T, na, ny, nz): for each age and (y, z), the endogenous cash-on-hand
                            points and the consumption there (one row per a' grid point).
          ap  (T, na, ny, nz, nxi): savings a' chosen by someone with wealth agrid[i] at the
              start of age t, earnings y, return state z and return shock node k.
        UNITS: wealth, cash and consumption in model units; c is consumption over 5 years.
        APPROXIMATION (as read from the code): on death the estate equals end-of-period savings
        a' (the bequest term uses Wp at a', with no return in the period of death).
        """
        p = self.p; T, ny, nz, nx = p.T, p.ny, self.nz, p.nxi
        a = self.agrid; s = p.sigma; beta = self.beta
        # Marginal value of leaving a' as an estate, for every grid point and composite state.
        _, _, Wp = self.bequest(a[:, None], self.phi1_s[None, :])        # (na, nz)
        self.m_nodes = np.zeros((T, a.size, ny, nz)); self.c_nodes = np.zeros((T, a.size, ny, nz))
        for t in range(T - 1, -1, -1):                # backwards from the last age to the first
            st = self.sur[t]                          # prob. of surviving to t+1
            EV = np.zeros((a.size, ny, nz))           # expected marginal value of a' if alive (0 at T-1)
            if t < T - 1:
                # marginal value of a' for each (y', z'): E_xi[ dcoh * u'(c_{t+1}) ]
                MV = np.zeros((a.size, ny, nz))
                for zn in range(nz):
                    for k in range(nx):
                        mk = self.coh(a, zn, k); dk = self.dcoh(a, zn, k)   # wealth after return, its slope
                        for yn in range(ny):
                            # next period's consumption at next period's cash on hand (incl. income)
                            c = self.cfun(t + 1, yn, zn, mk + self.inc[t + 1, yn, zn])
                            MV[:, yn, zn] += self.wxi[k] * dk * c ** (-s)   # u'(c) = c^(-sigma)
                # expectation over (y', z') given (y, z)
                EV = (MV.reshape(a.size, ny * nz) @ self.Pyz.T).reshape(a.size, ny, nz)
            # Right side of the Euler equation; the 1e-300 floor avoids dividing by zero below.
            rhs = np.maximum(beta * (st * EV + (1 - st) * Wp[:, None, :]), 1e-300)
            cend = rhs ** (-1 / s)                    # invert u'(c) = rhs  =>  c = rhs^(-1/sigma)
            self.c_nodes[t] = cend
            self.m_nodes[t] = cend + a[:, None, None]  # cash on hand = consumption + savings
        # policies at (a_i, y, z, xi)
        # Turn the policy "c as a function of cash" into "a' as a function of starting wealth":
        # for each grid wealth and each shock node, compute cash on hand and subtract consumption.
        self.ap = np.zeros((T, a.size, ny, nz, nx))
        for t in range(T):
            for z in range(nz):
                for k in range(nx):
                    mk = self.coh(a, z, k)
                    for y in range(ny):
                        m = mk + self.inc[t, y, z]
                        self.ap[t, :, y, z, k] = m - self.cfun(t, y, z, m)
        return self

    def cfun(self, t, y, z, m):
        """Consumption at cash on hand m (array) for age t and state (y, z).

        NUMERICS: linear interpolation between the EGM nodes; linear extrapolation (using the
        last two nodes' slope) above the top node; and c = m (spend everything, save 0) at or
        below the first node, where the no-borrowing constraint binds.
        """
        mn = self.m_nodes[t, :, y, z]; cn = self.c_nodes[t, :, y, z]
        c = np.interp(m, mn, cn)
        hi = m > mn[-1]                               # points above the top node
        if np.any(hi):
            slope = (cn[-1] - cn[-2]) / (mn[-1] - mn[-2])
            c = np.where(hi, cn[-1] + slope * (m - mn[-1]), c)
        return np.where(m <= mn[0], m, c)

    def euler_errors(self):
        """Accuracy check: log10 relative Euler-equation errors at the grid points.

        For every age (except the last), state and shock node where the household saves a
        positive amount, recompute the right side of the Euler equation exactly (with the full
        y, z and xi expectation) and compare the implied consumption with the policy's.
        Output: 1-D array of log10 |1 - c_implied / c|; e.g. -4 means a 0.01% error.
        """
        p = self.p; s = p.sigma; out = []
        for t in range(p.T - 1):
            for z in range(self.nz):
                for k in range(p.nxi):
                    for y in range(p.ny):
                        m = self.coh(self.agrid, z, k) + self.inc[t, y, z]
                        c = self.cfun(t, y, z, m); apy = m - c        # consumption and savings
                        ok = apy > 1e-8 * self.ebar                   # skip constrained points (a' = 0)
                        if not ok.any():
                            continue
                        EV = 0.0
                        for zn in range(self.nz):
                            for kn in range(p.nxi):
                                mk = self.coh(apy, zn, kn); dk = self.dcoh(apy, zn, kn)
                                for yn in range(p.ny):
                                    # joint prob. of (y', z', xi'): independent pieces multiply
                                    pr = self.P[y, yn] * self.Pz[z, zn] * self.wxi[kn]
                                    if pr == 0:
                                        continue
                                    cn = self.cfun(t + 1, yn, zn, mk + self.inc[t + 1, yn, zn])
                                    EV = EV + pr * dk * cn ** (-s)
                        _, _, Wp = self.bequest(apy, self.phi1_s[z])
                        rhs = self.beta * (self.sur[t] * EV + (1 - self.sur[t]) * Wp)
                        out.append(np.abs(1 - rhs ** (-1 / s) / c)[ok])
        e = np.concatenate(out)
        return np.log10(np.maximum(e, 1e-16))         # floor at 1e-16 to avoid log10(0)

    # ==========================================================================================
    # STATIONARY DISTRIBUTION (factorised parent-child method)
    # ==========================================================================================
    # ---- distribution ----
    def lottery(self, x):
        """Place wealth values x onto the grid by splitting each between two neighbours.

        NUMERICS ("lottery"): a value between grid points a_j and a_{j+1} is sent to a_j with
        weight wlo and to a_{j+1} with weight 1 - wlo, chosen so the mean is preserved.
        Values above the grid top are capped at the top (APPROXIMATION).
        Outputs (same shape as x): j = index of the lower grid point; wlo = weight on it.
        """
        a = self.agrid
        x = np.minimum(x, a[-1])
        # searchsorted finds where x would be inserted in the sorted grid; minus 1 = lower neighbour.
        j = np.clip(np.searchsorted(a, x, side="right") - 1, 0, a.size - 2)
        wlo = np.clip((a[j + 1] - x) / (a[j + 1] - a[j]), 0.0, 1.0)
        return j, wlo

    def transition(self, t):
        """Sparse transition matrix of the full state (wealth point, y, z) from age t to t+1.

        Combines: the saving policy ap[t] (via the lottery), the return shock xi (weights wxi)
        and the (y, z) chain Pyz. Survival is NOT included (applied separately).
        Output: N x N sparse matrix M with N = na * nyz; column = today's state, row = tomorrow's,
        so tomorrow's distribution = M @ today's (a column vector of masses).
        """
        p = self.p; na, ny, nz, nx = self.agrid.size, p.ny, self.nz, p.nxi
        nyz = ny * nz
        j, wlo = self.lottery(self.ap[t])                         # (na, ny, nz, nx)
        # Flat index of each current state: i * nyz + y * nz + z.
        src = (np.arange(na)[:, None, None] * nyz + np.arange(ny)[None, :, None] * nz
               + np.arange(nz)[None, None, :])                    # (na, ny, nz)
        cols, rows, vals = [], [], []                             # sparse matrix entries (triplets)
        Pyz = self.Pyz.reshape(ny, nz, ny, nz)                    # Pyz[y, z, y', z']
        for l in range(2):                                        # l = 0 lower, 1 upper grid neighbour
            jj = j + l
            ww = wlo if l == 0 else 1 - wlo
            for k in range(nx):                                   # return shock node
                base = jj[..., k] * nyz                           # (na, ny, nz)
                for yn in range(ny):
                    for zn in range(nz):
                        pr = Pyz[:, :, yn, zn]                    # (ny, nz)
                        # prob. = shock weight x lottery weight x (y, z) transition
                        v = self.wxi[k] * ww[..., k] * pr[None]
                        nzm = v > 0                               # keep only non-zero entries
                        rows.append((base + yn * nz + zn)[nzm]); cols.append(src[nzm]); vals.append(v[nzm])
        N = na * nyz
        # Duplicate (row, col) pairs are summed when the CSR matrix is built.
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))

    def estates(self, t):
        """Gross estate, charity and net bequest for every state and shock node at age t.

        Returns three arrays, each (na, ny, nz, nxi), model units, in this order:
          apt = gross estate (= end-of-period savings a'), g = charity,
          net = net bequest to heirs (after charity and estate tax; before splitting among children).
        (The original one-line docstring listed them as "(net bequest to heirs, charity, gross)";
        the actual return order is gross, charity, net.)
        Layer C overrides this method with a version that also handles trust assets.
        """
        apt = self.ap[t]                                          # (na, ny, nz, nx)
        g, net, _ = self.bequest(apt, self.phi1_s[None, None, :, None])
        return apt, g, net

    def distribution(self, tol=1e-10, maxgen=400, D0=None, verbose=False):
        """Stationary wealth distribution across generations (factorised direct method).

        (Layer C overrides this with a two-asset version plus Anderson acceleration.)
        Idea: D = distribution of parents at age 55 (t = gap), when their child enters. From D,
        follow the children's generation through life, adding inheritances when parents die,
        until the children themselves are 55; that gives a new D. Repeat until D stops changing.
        Factorised pairs: for each possible child starting type k0 = (y0, z0), keep
          c  = the child's distribution (starts as all mass at wealth 0 in type k0), and
          pw = the distribution of the parents of such children.
        Given k0 the two evolve independently until the parent dies, so the joint distribution
        is the sum over k0 of c x pw (outer product), never formed explicitly. Once the parent
        dies, the child's wealth plus inheritance moves into O ("orphans": parent dead).
        Inputs: tol = convergence tolerance on the total absolute change in D (sum over states);
          maxgen = max generations; D0 = starting D (e.g. from an earlier solve), else built from
          newborns with no inheritance; verbose prints the change per generation.
        Stores:
          dist        (T, N): mass of the children's generation at each age (row) and full state
                      (column), not yet weighted by cohort size; rows sum to mass[t].
          parent_D    (N,): converged D.
          pairs3      the (c, pw) pairs when the child is at t = 3 (ages 35-39; parent then at
                      t = 10, ages 70-74), used for the parent-child wealth elasticity.
          gens_to_converge, and _H = (net bequests, charity, gross estates) by age, for moments.
        """
        p = self.p; T, gap, ny, nz, nx = p.T, p.gap, p.ny, self.nz, p.nxi
        na = self.agrid.size; nyz = ny * nz; N = na * nyz
        M = [self.transition(t) for t in range(T)]                # one transition matrix per age
        MT = [m.T.tocsr() for m in M]                             # transposes (not used below)
        # net bequest per (state, xi) at each age -> lottery on the grid
        Hj, Hw, Hval, Gch, Gross = [], [], [], [], []
        for t in range(T):
            apt, g, net = self.estates(t)
            j, wlo = self.lottery(net)
            # Each stored as (N, nxi): one row per full state, one column per shock node.
            Hj.append(j.reshape(N, nx)); Hw.append(wlo.reshape(N, nx))
            Hval.append(net.reshape(N, nx)); Gch.append(g.reshape(N, nx)); Gross.append(apt.reshape(N, nx))
        self._H = (Hval, Gch, Gross)
        if D0 is None:
            # Start: newborns at zero wealth (the first nyz entries = wealth point 0), aged to 55
            # without inheritances.
            D = np.zeros(N); D[:nyz] = self.pi_age[0]
            for t in range(gap):
                D = M[t] @ D
        else:
            D = D0.copy()
        for gen in range(maxgen):
            # O[t] = children aged t whose parent has died ("orphans"), incl. their inheritance.
            # child[t] = all children aged t (child marginal distribution).
            O = np.zeros((T, N)); child = np.zeros((T, N))
            Dm = D.reshape(na, nyz)                               # parents: one row per wealth point
            # factorised pairs: one (child, parent) pair of vectors per child initial type
            Cs, Ps = [], []
            for k0 in range(nyz):
                # Parents' mass times the prob. that their child starts in type k0.
                pw = (Dm * self.Kbirth[:, k0][None, :]).reshape(-1)
                if pw.sum() <= 0:
                    continue
                c = np.zeros(N); c[k0] = 1.0                      # child: zero wealth, type k0
                Cs.append(c); Ps.append(pw)
            store3 = None
            for t in range(T - 1):                                # child's age t -> t+1
                tp = t + gap                                      # parent's age at the same time
                if tp <= T - 1:
                    # Some parents may still be alive.
                    for c, pw in zip(Cs, Ps):
                        child[t] += c * pw.sum()                  # child mass = c x living parents' mass
                    child[t] += O[t]                              # plus children already orphaned
                    if t == (35 - p.age0) // p.tlen:
                        store3 = [(c.copy(), pw.copy()) for c, pw in zip(Cs, Ps)]
                    st = self.sur[tp]                             # parent's survival prob. this period
                    # Orphans move on by themselves. (The child's own survival sur[t] is 1 here
                    # anyway, because nobody dies before 60.)
                    newO = M[t] @ O[t] * self.sur[t]
                    newC, newP = [], []
                    for c, pw in zip(Cs, Ps):
                        cn = M[t] @ c                             # child's next-period distribution
                        if st < 1.0:
                            # estate distribution of dying parents (on the grid)
                            hd = _estate_hist(pw * (1 - st), Hj[tp], Hw[tp], self.wxi, na)
                            # Child's wealth + inheritance (split if a sibling) -> new orphans.
                            newO += _convolve(cn, hd, self.agrid, nyz, self.q2)
                        newC.append(cn); newP.append(st * (M[tp] @ pw))   # surviving parents move on
                    Cs, Ps = newC, newP
                    O[t + 1] = newO
                else:
                    # All parents are dead (survival at the last age is 0): only orphans remain.
                    child[t] = O[t]
                    O[t + 1] = M[t] @ O[t] * self.sur[t]
            child[T - 1] = O[T - 1]
            Dn = O[gap].copy()                                    # the children, now aged 55
            diff = np.abs(Dn - D).sum()                           # total absolute change in mass
            D = Dn
            if verbose:
                print(gen, diff)
            if diff < tol:
                break
        self.gens_to_converge = gen + 1
        self.dist = child
        self.parent_D = D
        self.pairs3 = store3
        return self

    # ==========================================================================================
    # MOMENTS
    # ==========================================================================================
    # ---- moments ----
    def _states(self):
        """Wealth level and return state of every flat state index i * nyz + y * nz + z.

        Outputs: a (N,) wealth in model units; z (N,) return state (0 N, 1 L, 2 H).
        PYTHON NOTE: np.repeat repeats each element (a0 a0 a0 a1 a1 ...); np.tile repeats the
        whole array (0 1 2 0 1 2 ...).
        """
        p = self.p; nyz = p.ny * self.nz
        a = np.repeat(self.agrid, nyz)
        s_ = np.tile(np.arange(self.nz), self.agrid.size * p.ny)
        return a, self.zof[s_]

    def wealth_elasticity(self, floor_units=0.01):
        """Parent-child wealth elasticity: OLS slope of log child wealth on log parent wealth.

        Uses pairs3: children at ages 35-39, their (surviving) parents at ages 70-74. Only
        people with wealth above floor_units * ebar enter (log of zero is undefined).
        NUMERICS: because the joint distribution is a sum over pairs of (child vector) x (parent
        vector), all regression sums (S = mass, Sx, Sy, Sxx, Sxy) can be built from sums over
        each vector separately, without forming the joint distribution.
        Output: scalar slope cov(log parent, log child) / var(log parent).
        """
        a, _ = self._states()
        pos = a > floor_units * self.ebar
        la = np.where(pos, np.log(np.maximum(a, 1e-300)), 0.0)   # log wealth (0 where excluded)
        S = Sx = Sy = Sxx = Sxy = 0.0
        for c, pw in self.pairs3:
            wc, wp = (c * pos).sum(), (pw * pos).sum()            # child mass, parent mass
            ec, ep = (c * pos * la).sum(), (pw * pos * la).sum()  # mass-weighted log wealth sums
            S += wc * wp; Sx += wc * ep; Sy += ec * wp            # x = parent, y = child
            Sxx += wc * (pw * pos * la ** 2).sum(); Sxy += ec * ep
        mx, my = Sx / S, Sy / S                                   # means of log wealth
        return (Sxy / S - mx * my) / (Sxx / S - mx ** 2)

    def earnings_rank_rank(self):
        """Rank-rank slope between parent's earnings state at 55 and child's at 35-39.

        Built from the exogenous chains only (no wealth). J = joint distribution of parent's y
        (rows) and child's y (columns), after summing out z. Ranks are the mid-point percentile
        of each discrete state. Output: scalar regression slope of child rank on parent rank.
        (Layer C reuses this method unchanged.)
        """
        p = self.p
        pyz = self.pi_age[p.gap]                        # parent's (y,z) at 55
        J = pyz[:, None] * self.Kbirth                  # (yz_p, yz_c at 20)
        J = J @ np.linalg.matrix_power(self.Pyz, 3)     # child at 35-39
        J = J.reshape(p.ny, self.nz, p.ny, self.nz).sum(axis=(1, 3))   # (ny parent, ny child)
        def ranks(m):
            # mid-point rank of each state: share below it + half its own share
            c = np.cumsum(m); return c - m / 2
        mp_, mc_ = J.sum(axis=1), J.sum(axis=0)         # parent and child marginals
        rp, rc = ranks(mp_), ranks(mc_)
        a, b = (mp_ * rp).sum(), (mc_ * rc).sum()       # mean ranks (= 0.5)
        cov = (J * (rp[:, None] - a) * (rc[None, :] - b)).sum()
        return cov / (mp_ * (rp - a) ** 2).sum()

    def moments(self):
        """Model moments used for calibration and reporting (Layer C overrides this).

        Weights: dist (by age) x cohort size, normalised so the population sums to 1.
        Returns a dict. Main entries (model units unless "usd"; flows are ANNUAL):
          K_supply  mean household wealth;  Kb  business capital (wealth up to kbar in L/H);
          Kc_demand corporate capital demand;  K_supply-Kb  wealth available to corporations;
          K/Y, Kb/K;  gini, top0.1/top1/top5/top10/top20 wealth shares, bottom50;
          share_zero (at zero wealth), share_entrepreneurs (z > 0) and their wealth share;
          share_deaths_taxable, charity/taxable_estates, bequest_flow/Y, wealth_75plus/mean;
          parent_child_wealth_elasticity, earnings_rank_rank, top1_earnings_share;
          mean_wealth_usd_check (should equal the SCF $1.0637M when usd is consistent);
          implied_g/Y (tax revenue minus pensions, over Y); mass_at_grid_top (grid adequacy);
          mean_wealth_by_age (relative to the mean); max_wealth_usd_with_mass.
        """
        p = self.p; T = p.T
        a, z = self._states()
        W = self.dist * self.cohort[:, None]                      # (T, N) population mass by age and state
        tot = W.sum()
        w = W.sum(axis=0) / tot                                   # (N,) population shares across all ages
        K = (w * a).sum()                                         # mean wealth
        # Business capital: wealth up to kbar held by L and H households.
        Kb = (w * np.minimum(a, self.kbar) * (z > 0)).sum()
        # business output (expected, annual): (r + delta + premium) per unit of business capital
        prem = np.array(p.prem if self.nzr > 1 else (0.0,))
        Yb = (w * np.minimum(a, self.kbar) * (z > 0) * (self.r_ann + p.delta + prem[z])).sum()
        # Lorenz curve of wealth: sort by wealth, cumulate population shares and wealth shares.
        o = np.argsort(a, kind="stable"); xs, ws = a[o], w[o]
        cw = np.cumsum(ws); cx = np.cumsum(xs * ws); L = cx / cx[-1]
        # Gini = 1 - 2 x area under the Lorenz curve (trapezoid rule).
        Lp = np.concatenate([[0], L[:-1]]); gini = 1 - np.sum(ws * (L + Lp))
        cw0 = np.concatenate([[0.0], cw]); L0 = np.concatenate([[0.0], L])
        # PYTHON NOTE: `lambda f: ...` is a one-line unnamed function. top(f) = wealth share of
        # the richest fraction f (1 minus the Lorenz curve at 1 - f, by interpolation).
        top = lambda f: 1.0 - np.interp(1.0 - f, cw0, L0)
        # estates
        Hval, Gch, Gross = self._H
        deaths = taxable = est_val = est_tax = charity = tax_gross = tax_char = 0.0
        for t in range(T):
            # Mass of deaths at age t in each (state, shock node): population share x death prob.
            q = (self.dist[t] * self.cohort[t] / tot * (1 - self.sur[t]))[:, None] * self.wxi[None, :]
            gr, g, net = Gross[t], Gch[t], Hval[t]
            deaths += q.sum(); est_val += (q * gr).sum()          # number of deaths, gross estates
            tx = (gr - g) > self.E                                # estates above the exemption after charity
            taxable += q[tx].sum(); tax_gross += (q * gr)[tx].sum(); tax_char += (q * g)[tx].sum()
            est_tax += (q * (gr - g - net)).sum(); charity += (q * g).sum()   # estate tax paid, charity
        Wage = (self.dist * a[None, :]).sum(axis=1) / self.dist.sum(axis=1)   # mean wealth by age
        mass_age = self.dist.sum(axis=1) * self.cohort
        i75 = (75 - p.age0) // p.tlen                             # age index of 75-79
        old = (Wage[i75:] * mass_age[i75:]).sum() / mass_age[i75:].sum() / K   # mean wealth 75+ / mean
        retirees = self.age_w[p.TL:].sum()
        # earnings distribution of working-age people (exogenous given (y, z) by age)
        ew, ex = [], []                                           # weights and earnings levels
        for t in range(p.TL):
            ew.append(self.age_w[t] * self.pi_age[t]); ex.append(self.gross_earn[t].reshape(-1))
        ew = np.concatenate(ew); ex = np.concatenate(ex); ew = ew / ew.sum()
        oe = np.argsort(ex); cwe = np.concatenate([[0.0], np.cumsum(ew[oe])])
        Le = np.concatenate([[0.0], np.cumsum(ew[oe] * ex[oe])]); Le /= Le[-1]
        top1_earn = 1.0 - np.interp(0.99, cwe, Le)               # top 1% share of gross earnings
        # Earnings WITHOUT the superstar multiplier vs. WITH it: the difference is owner income.
        base_earn = (np.concatenate([self.age_w[t] * self.pi_age[t] * (p.tlen * self.w * self.eff[t]
                     * np.repeat(np.exp(self.y), self.nz)) for t in range(p.TL)])).sum()
        tot_earn = sum((self.age_w[t] * self.pi_age[t] * self.gross_earn[t].reshape(-1)).sum() for t in range(p.TL))
        extra_earn = (tot_earn - base_earn) / p.tlen                 # owner labor income (annual, per capita)
        Yb = Yb + extra_earn                                         # count it as business output
        Y = self.Yc + Yb                                             # total output (annual)
        # Government revenue (annual): labour tax (incl. owner income), capital tax on r * K and on
        # the business excess profit (premium * Kb), plus estate tax (per 5 years -> / tlen).
        rev = p.tau_l * (self.w * self.L + extra_earn) + p.tau_a * (self.r_ann * K + (Yb - extra_earn - (self.r_ann + p.delta) * Kb)) \
            + est_tax / p.tlen
        pens = retirees * p.pens * self.ebar / p.tlen               # pension spending (annual)
        return {
            "K_supply": K, "Kb": Kb, "Kc_demand": self.Kc, "K_supply-Kb": K - Kb,
            "K/Y (total)": K / Y, "Kb/K": Kb / K,
            "gini": gini, "top0.1": top(0.001), "top1": top(0.01), "top5": top(0.05),
            "top10": top(0.10), "top20": top(0.20), "bottom50": 1 - top(0.50),
            "share_zero": w[a <= 1e-12].sum(),
            "share_entrepreneurs": w[z > 0].sum(),
            "wealth_share_entrepreneurs": (w * a)[z > 0].sum() / K,
            "share_deaths_taxable": taxable / deaths,
            "charity/taxable_estates": tax_char / tax_gross if tax_gross > 0 else 0.0,
            "bequest_flow/Y": est_val / p.tlen / Y,              # gross estates per year / output
            "wealth_75plus/mean": old,
            "parent_child_wealth_elasticity": self.wealth_elasticity(),
            "earnings_rank_rank": self.earnings_rank_rank(),
            "top1_earnings_share": top1_earn,
            "mean_wealth_usd_check": K * self.usd,
            "implied_g/Y": (rev - pens) / Y,
            "mass_at_grid_top": w[a >= self.agrid[-3]].sum(),   # mass in the top 3 grid points
            "mean_wealth_by_age": Wage / K,
            "max_wealth_usd_with_mass": a[w > 1e-12].max() * self.usd,
        }


# ==============================================================================================
# NUMBA HELPERS FOR THE DISTRIBUTION (inheritances)
# ==============================================================================================
# PYTHON NOTE: `@njit(cache=True)` compiles the function below to fast machine code with Numba
# the first time it runs, and caches the compiled version on disk. Plain loops are fine here.
@njit(cache=True)
def _estate_hist(pw, Hj, Hw, wxi, na):
    """Histogram of net bequests on the wealth grid.

    Inputs: pw (N,) mass of dying parents by full state; Hj, Hw (N, nxi) lottery index and
    lower weight of the net bequest for each state and shock node; wxi shock weights; na grid size.
    Output: (na,) mass of estates at each wealth grid point (before splitting among siblings).
    """
    out = np.zeros(na)
    N, nx = Hj.shape
    for s in range(N):
        m = pw[s]
        if m <= 0.0:
            continue                                  # skip states with no dying parents
        for k in range(nx):
            mk = m * wxi[k]                           # mass for this shock node
            j = Hj[s, k]
            out[j] += mk * Hw[s, k]                   # lower neighbour
            out[j + 1] += mk * (1.0 - Hw[s, k])       # upper neighbour
    return out


@njit(cache=True)
def _convolve(cn, hd, agrid, nyz, q2):
    """Child's own next wealth (cn, by state) plus inheritance h/m (hd on the grid), m in {1,2}.

    Inputs: cn (N,) child's distribution next period (flat index i * nyz + rest); hd (na,)
    estate histogram from _estate_hist; agrid wealth grid; nyz; q2 = prob. of a sibling.
    For every pair (child wealth point, estate point) the child's new wealth is
    agrid[i] + agrid[j] / m, where m = 1 (prob 1 - q2) or 2 (prob q2), placed on the grid by the
    lottery (binary search for the neighbours). The child's (y, z) part `rest` is unchanged.
    Output: (N,) mass of newly orphaned children by full state.
    APPROXIMATION: estates are first put on the grid, then the sums are put on the grid again
    (two lotteries); values above the grid top are put at the top point.
    """
    na = agrid.size
    out = np.zeros(cn.size)
    amax = agrid[-1]
    hj = np.nonzero(hd > 0.0)[0]                      # grid points that actually hold estates
    for c in range(cn.size):
        mc = cn[c]
        if mc <= 0.0:
            continue
        i = c // nyz; rest = c % nyz                  # wealth point and (y, z) index of the child
        ac = agrid[i]
        for jj in range(hj.size):
            j = hj[jj]
            mh = mc * hd[j]                           # joint mass (independent given the pair)
            for sib in range(2):
                pr = (1.0 - q2) if sib == 0 else q2
                x = ac + agrid[j] / (1.0 + sib)       # own wealth + share of the estate
                if x >= amax:
                    out[(na - 1) * nyz + rest] += mh * pr
                    continue
                # NUMERICS: binary search for the grid interval containing x.
                lo, hi = 0, na - 1
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if agrid[mid] <= x:
                        lo = mid
                    else:
                        hi = mid
                wl = (agrid[lo + 1] - x) / (agrid[lo + 1] - agrid[lo])
                out[lo * nyz + rest] += mh * pr * wl
                out[(lo + 1) * nyz + rest] += mh * pr * (1.0 - wl)
    return out


# ==============================================================================================
# DRIVERS: build, calibrate beta, Monte Carlo check
# ==============================================================================================
def buildB(p, eflife, surlife, D0=None):
    """Convenience: construct, solve and compute the distribution of a Layer B economy."""
    return EconomyB(p, eflife, surlife).solve().distribution(D0=D0)


def calibrate_beta_B(p, eflife, surlife, lo=0.85, hi=0.99, tol=5e-4, verbose=False, D0=None):
    """beta so that household wealth minus business capital = corporate capital demand.
    Also updates the $ conversion so model mean wealth = SCF mean net worth.

    NUMERICS: bisection on beta_annual in [lo, hi], up to 40 steps, stopping when the capital
    gap (K_supply - Kb) / Kc_demand - 1 is below tol in absolute value. Too much capital
    (gap > 0) -> lower beta. Each step reuses the previous parent distribution as a warm start
    and feeds the latest mean wealth into the dollar scale `usd`.
    (Note: the D0 argument is not used; the warm start comes from the previous step.)
    Output: the final EconomyB, rebuilt once more with the converged $ conversion.
    """
    mw = p.mean_wealth_units
    e = None
    for it in range(40):
        mid = 0.5 * (lo + hi)
        e = buildB(replace(p, beta_annual=mid, mean_wealth_units=mw), eflife, surlife,
                   D0=None if e is None else e.parent_D)
        m = e.moments()
        gap = m["K_supply-Kb"] / m["Kc_demand"] - 1             # capital-market excess supply
        mw = m["K_supply"]                                       # update the $ anchor
        if verbose:
            print(f"  beta={mid:.5f} gap={gap:+.4f} Kb/K={m['Kb/K']:.3f} top1={m['top1']:.3f}")
        if abs(gap) < tol:
            break
        if gap > 0:
            hi = mid
        else:
            lo = mid
    # one more build with the converged $ conversion
    e = buildB(replace(p, beta_annual=mid, mean_wealth_units=mw), eflife, surlife, D0=e.parent_D)
    return e


def monte_carlo_B(e: EconomyB, N=200_000, G=14, burn=7, seed=11):
    """Check of the direct distribution by simulating N dynasties for G generations.

    Each generation lives through the solved policies (_mc_genB); when a parent dies the net
    bequest (split in two with prob q2) is handed to that dynasty's child, who receives it
    one period later. The first `burn` generations are discarded. Wealth inputs/outputs in
    model units. Output: dict with gini, top0.1, top1, top10 and mean wealth, comparable to
    `moments()`.
    APPROXIMATION: wealth above the grid top uses the policy at the top grid point.
    """
    rng = np.random.default_rng(seed)                 # random-number generator (reproducible)
    p = e.p; T, gap, ny, nz, nx = p.T, p.gap, p.ny, e.nz, p.nxi
    nyz = ny * nz
    # Cumulative probabilities (for drawing from a discrete distribution with one uniform).
    Pc = np.cumsum(e.Pyz, axis=1); Kc = np.cumsum(e.Kbirth, axis=1); wc = np.cumsum(e.wxi)
    s0 = rng.choice(nyz, size=N, p=e.pi_age[0])       # first generation's starting (y, z) types
    inh_amt = np.zeros(N); inh_t = np.full(N, -1)     # inheritance amount and age it arrives (-1 = none)
    samples = []
    apr = e.ap.reshape(T, e.agrid.size, nyz, nx)      # policies with (y, z) flattened
    for g in range(G):
        U = rng.random((N, T, 3))                     # uniforms: death, (y, z) move, return shock
        A, alive, s7, td, est, sd = _mc_genB(apr, e.agrid, e.sur, Pc, wc, s0, inh_amt, inh_t, U, gap)
        if g >= burn:
            samples.append((A, alive))
        _, net, _ = e.bequest(est, e.phi1_s[sd % e.nz])           # net bequest of each parent
        sib = rng.random(N) < e.q2                    # does the child have a sibling?
        inh_amt = net / np.where(sib, 2.0, 1.0)
        inh_t = td - (gap - 1)                        # child's age one period after the parent's death
        u = rng.random(N)
        # Child's starting type drawn from Kbirth given the parent's (y, z) at 55.
        s0 = np.minimum(np.array([np.searchsorted(Kc[s], uu) for s, uu in zip(s7, u)]), nyz - 1)
    x = np.concatenate([s[0][s[1]] for s in samples])             # wealth of everyone alive
    tt = np.concatenate([np.broadcast_to(np.arange(T), s[0].shape)[s[1]] for s in samples])   # their ages
    w = e.cohort[tt]                                  # weight by cohort size
    o = np.argsort(x); xs, ws = x[o], w[o] / w.sum()
    cw = np.cumsum(ws); cx = np.cumsum(xs * ws); L = cx / cx[-1]
    Lp = np.concatenate([[0], L[:-1]]); gini = 1 - np.sum(ws * (L + Lp))
    cw0 = np.concatenate([[0.0], cw]); L0 = np.concatenate([[0.0], L])
    top = lambda f: 1.0 - np.interp(1.0 - f, cw0, L0)
    return {"gini": gini, "top0.1": top(0.001), "top1": top(0.01), "top10": top(0.10),
            "mean": (xs * ws).sum()}


@njit(cache=True)
def _mc_genB(ap, agrid, sur, Pc, wc, s0, inh_amt, inh_t, U, gap):
    """Simulate one generation of N people through life (Numba).

    Inputs: ap (T, na, nyz, nxi) savings policy; agrid; sur; Pc cumulative (y, z) transition;
    wc cumulative shock weights; s0 starting (y, z); inh_amt / inh_t inheritance and age it
    arrives; U (N, T, 3) uniforms; gap.
    Outputs: A (N, T) wealth at the start of each age; alive (N, T); s7 (y, z) at age gap (for
    drawing the child's type); td age at death; est estate (= savings in the death period);
    sd (y, z) state at death.
    """
    N = s0.size; T = ap.shape[0]; na = agrid.size
    A = np.zeros((N, T)); alive = np.zeros((N, T), np.bool_)
    s7 = np.zeros(N, np.int64); td = np.zeros(N, np.int64); est = np.zeros(N); sd = np.zeros(N, np.int64)
    for i in range(N):
        x = 0.0; s = s0[i]                            # start with zero wealth
        for t in range(T):
            if t == inh_t[i]:
                x += inh_amt[i]                       # inheritance arrives
            if t == gap:
                s7[i] = s
            A[i, t] = x; alive[i, t] = True
            # Draw the return shock node k from its cumulative weights.
            k = 0
            while k < wc.size - 1 and U[i, t, 2] > wc[k]:
                k += 1
            if x >= agrid[-1]:
                xn = ap[t, na - 1, s, k]              # above the grid: use the top point's policy
            else:
                # Binary search, then linear interpolation of the savings policy in wealth.
                lo, hi = 0, na - 1
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if agrid[mid] <= x:
                        lo = mid
                    else:
                        hi = mid
                wl = (agrid[lo + 1] - x) / (agrid[lo + 1] - agrid[lo])
                xn = wl * ap[t, lo, s, k] + (1 - wl) * ap[t, lo + 1, s, k]
            if U[i, t, 0] >= sur[t]:
                td[i] = t; est[i] = xn; sd[i] = s     # dies at the end of age t; estate = savings
                break
            # Draw next period's (y, z) from the cumulative transition row.
            j = 0
            while j < Pc.shape[1] - 1 and U[i, t, 1] > Pc[s, j]:
                j += 1
            s = j; x = xn
    return A, alive, s7, td, est, sd
