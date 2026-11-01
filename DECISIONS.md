# FODT project — decision log

*Public copy (November 2026). A few personal and administrative notes about authorship and the author's credentials have been removed or shortened; every modelling, calibration and results entry is unchanged.*

Read this first in every new session. Newest entries at the bottom of each section.

## Folder layout
- `data/`: model inputs and calibration targets
  - eflife.dat (age-efficiency profile)
  - surlife.dat (survival)
  - markov.dat (unused)
  - kermay01.f90 (original De Nardi Fortran, reference only)
  - targets_v1.csv (every U.S. data target, with source, vintage and URL)
- `paper/`: the original paper PDF; revised sections later
- `code/`:
  - fodt.py (v1 replication of the paper's model)
  - layerA.py (v2 baseline)
- `notebooks/`: runs; each saves its results to `results/`
- `results/`: CSV and JSON outputs, versioned (`_v1`, `_v2`, ...)
- Methodology spec (master copy): Claude Doc "FODT Model — Methodology Spec (v2)", https://claude.ai/code/artifact/7010431a-b1d1-4f82-b974-96bd6774d75e

## Status (2 Oct 2026)
**v1** (`code/fodt.py`, `notebooks/FODT_model_v1.ipynb`, `results/fodt_results_v1.csv`): faithful implementation of Section 3.2 of the paper (eqs. 3–14, Tables 1–2).

Key v1 findings:
- The original notebook reproduces Table 3 rows 2 and 4 exactly (seed 123), but it does not implement the model described in the paper. It assigns FODT to the top 10% by inheritance, uses a fixed 50% transfer, has no threshold or setup cost, no η_trust, and different parameters.
- In the paper's own model, nobody reaches A_th = $100M (max wealth ≈ $8M), so FODT has no effect.
- At A_th = $5M: Gini 0.87, top 1% only 19%.
- The r_D premium alone makes dynastic wealth explode, and there is no stationary distribution.
- Literal eq. 14 weakens bequest saving at the top. Switching utility functions also adds a spurious "level" benefit.
- Irreversible trusts plus inherited FO status mean FO participation grows every generation (non-stationary).
- Calibrated β = 0.923 (K/Y = 3, no FODT); the paper reports 0.95–0.96.

## Modelling assumptions in force (v1)
1. Partial equilibrium: r_L and r_D are fixed. β is calibrated to wealth/annual output = 3, with Y = earnings/(1−α).
2. Model unit = 5 × SSA AWI 2024 ($69,846.57) = $349k.
3. Trust principal is locked; trust income can be distributed. k_FO is charged once on AUM.
4. A family can open an FODT when a ≥ A_th. The trust persists, and heirs inherit it as a trust.
5. Bequest utility: eq. 7 without FODT, eq. 14 with FODT. Both are discounted by β.
6. Inheritance arrives at the parent's death (parent is 30 years older) and is not anticipated. One heir.
7. Productivity: Rouwenhorst, 5 states (ρ = 0.85, σ² = 0.30). Child's productivity ~ N(0.65·parent's, 0.37).
8. g = 23% and labour taxes are not modelled.

## Agreed plan — v2
1. **Fortune creation:** add an entrepreneur / return-heterogeneity pathway (Cagetti–De Nardi 2006; Benhabib–Bisin–Luo 2019; Fagereng et al. 2020). Calibrate to the top 1% share *before* FODT is switched on.
2. **Taxes:** match current U.S. law. Top estate rate 40%. Estate and GST exemption of $15M per person from 2026 (OBBBA, permanent, indexed). The FODT advantage comes mainly through GST/estate avoidance, not a low τ_D.
3. **Heirs:** multiple heirs, with estates split by population growth, so the model has a stationary distribution.
4. Then: a single bequest function (heterogeneous φ instead of eq. 14), a risky return premium that rises with wealth, cost-based FODT entry, GE or an open-economy justification, joint calibration (SMM), and a decomposition table.

Targets: wealth Gini ≈ 0.78–0.83, top 1% ≈ 29–31% (Fed DFA, Q1 2025: 30.8%), K/Y = 3, transfer/wealth ≈ 0.6.

## Spec v2 (2 Oct 2026, later)
The methodology spec is in the Claude Doc "FODT Model — Methodology Spec (v2)":
https://claude.ai/code/artifact/7010431a-b1d1-4f82-b974-96bd6774d75e
- A Fable-model adversarial review left 14 comments in the doc, ranked by severity.
- The spec was revised to cover:
  - FODTs broadened into "wealth-preservation technology" with **two tiers** (dynasty trust; advanced management / family office).
  - **Opt-in that differs by family**, via a planning-friction distribution calibrated to take-up by wealth.
  - **Deductible charitable bequests.**
  - A **main model vs. extensions** scope table.
- Review fixes adopted:
  - Trust assets earn the family's return state.
  - π_D = 0 baseline.
  - Grantor status lasts only during the founder's life; no step-up for trust assets.
  - Unified exemption tracked with a one-bit state.
  - Inheritance ages corrected.
- Decided by Rich (2 Oct):
  - Reduced-form fortune creation.
  - $500B wealth grid.
  - Apply the review fixes for every resolved thread:
    - GE: a private-business sector produces the extra returns.
    - Trust heirs anticipate inheritances (Layer C).
    - ρ_h estimated by SMM to the rank-rank slope of 0.34.
    - Dollar thresholds converted using mean wealth.
    - Fed DFA cited via FRED WFRBST01134 (32.5%, Q2 2026).
    - A lump-sum transfer balances the government budget.
    - Effective estate tax validated against IRS SOI.
    - Trust splitting and wind-up rules across heirs.
    - Timing within a period.
    - Analytic Pareto-tail check before SMM.
  - Added: tier-2 returns come from risk and taxes, not skill (Bach, Calvet & Sodini 2020).
  - Computation: aggregate results and calibration use the directly computed stationary distribution (Young 2010). Monte Carlo is used only for dynasty experiments, oversampling high-return families.
  - Identification:
    - New targets: share of estates above the exemption, bequests by wealth and age (HRS), and the parent–child wealth elasticity (0.37; moved from validation to target).
    - One common φ₂; a single exit probability for the high-return state.
    - Rule: fix or drop any parameter the sensitivity table can't separate.
  - Added: a behavioral return gap λ_b (~1 pp, tied to planning friction; Barber & Odean 2000, Dichev 2007).
- Still pending the reviewer: the open-decisions table (16 rows); the GE accounting for return premia; non-stochastic distribution computation.

## Layer A v1 (2 Oct 2026)
Files: `code/layerA.py`, `notebooks/LayerA_v1.ipynb`, `results/layerA_v1_results.csv`.

Implementation choices:
- Estate exemption: $30M (married couple), converted to model units with SCF 2022 mean net worth ($1,063,700).
- Parent–child gap: **35 years** (spec said 30). With 5-year periods this makes the direct distribution method exact, since grandparents have always died. Children inherit at ages 30–59.
- Government: g is the residual; there is no lump-sum tax on households. The spec's "lump-sum transfer" wording needs to match this.
- Bequest parameters φ₁ and χ are rescaled by mean earnings^(1−σ), because CRRA utility depends on units.

Results:
- β = 0.940 gives K/Y = 3 (r = 6%).
- Gini 0.80; top 1% 15.5% (data 32.5%); top 10% 65%; zero wealth 33%.
- Bequest flow 3.8% of Y; taxable estates 0.01% of deaths; implied g/Y 11.9%.

Tests passed:
- Euler errors: mean 10^-5.5.
- Monte Carlo matches the direct computation to within 0.2 pp.
- Grid 150 vs 250 points: differences of 0.1 pp or less.
- No mass at the top of the grid.

Placeholders: φ₁ and φ₂ not calibrated; χ = 0; τ_l = 0.25 (to source); ρ_h = 0.40 (to estimate); one bequest type; transfer–wealth ratio not yet computed.

Next: pull the data targets, calibrate the bequest block, then Layer B.

## Layer A v2: calibrated (2 Oct 2026)
Files: `data/targets_v1.csv` (every target with source, vintage and URL), `notebooks/LayerA_v2.ipynb`, `results/layerA_v2_*`.

Calibration:
- ρ_h = 0.545 (earnings rank-rank slope 0.341).
- φ₁ = −14.6 (wealth at 75+ = 1.53 × mean, SCF 2022).
- β = 0.9345 (K/Y = 3).
- τ_l = 0.301 (OECD tax wedge, 2024).

Fit:
- Gini 0.83 vs 0.83; top 10% 71% vs 69% (DFA).
- Top 1% 18% vs 32.5%; taxable estates 0.07% vs 0.14% of deaths.
- Zero wealth 36% vs 8%; parent–child wealth elasticity 0.16 vs 0.37.

Data choice: distribution targets come from DFA (Q2 2026), with the SCF 2022 Gini and zero-wealth share as validation. CBO's 27% top-1% figure includes DB pensions, so it is reference only.

Deferred to Layer B, since they need the top tail: χ (charity, 13% of filers' estates), the dynastic bequest type (wealth elasticity), and the taxable-estate share.

Known gap to revisit: too many people with zero wealth.

## Layer B v1: fortune creation (3 Oct 2026)
Files: `code/layerB.py`, `code/calibrate_B.py`, `notebooks/LayerB_v1.ipynb`, `results/layerB_v1_*`, `results/layerB_calibration_log.txt`.

Design:
- Return state N / L / H. Business wealth earns a log-normal return (modest mean premium, high skewed risk).
- Single exit probability back to N; a child of an entrepreneur starts in L with probability ι.
- A private-business sector produces the extra returns; r stays at 6% from the corporate sector.
- No scale limit in v1. The tail is controlled by exit risk, dilution across heirs and the estate tax.
- Distribution computed directly in factorised form: a sum over child entry types of child × parent distributions. Exact, and about 2× faster than Layer A's method.
- Nesting test: with one state, Layer B reproduces Layer A.
- Dynastic bequest type implemented (`nk = 2`) but not yet calibrated.

Calibration: 10 parameters, 8 targets plus capital-market clearing, least squares on the direct distribution.

| Parameter | Value |
|---|---|
| β | 0.949 |
| φ₁ (bequest strength) | −2.63 |
| χ (charity) | 0.21 |
| π_L, σ_L | 4.9%, 6% |
| π_H, σ_H | 9.1%, 75% |
| p_NL, p_LH (per 5 years) | 0.052, 0.048 |
| ι | 0.76 |

Fit:
- Top 1% 32.4% (32.5%); top 10% 69.0% (68.9%).
- Entrepreneurs 20.4% (20%); business capital share 29% (30.5%).
- Taxable deaths 0.17% (0.14%); charity 12.8% (13%); wealth 75+ 1.50 (1.53).
- Gini 0.82 (0.83, not targeted).

Problems:
- The very top is too fat: top 0.1% holds 21% vs about 15% in the SCF (17% with Forbes); Pareto exponent 1.13; top-1% threshold $9M vs $13.6M.
- Parent–child wealth elasticity 0.19 (0.37).
- Zero wealth 23% (8%).
- The L premium (4.9% with low risk) sits uneasily with Moskowitz & Vissing-Jørgensen (2002).

Tests:
- Grid (220 points) and shock nodes (11) change results by under 0.1 pp.
- Euler errors: mean 10^-4.9.
- Monte Carlo seeds scatter (top 1% 32% vs 27%) because the tail is so fat. This confirms the direct-method choice.

Next round:
- Add the top 0.1% share as a target (consider a scale limit or capping σ_H).
- Calibrate the dynastic type to the 0.37 elasticity.
- Consider tying the L premium to evidence.

## Open issues
- Disclose AI assistance according to the target journal's policy.

## Layer B premium-cap test (3 Oct 2026)
File: `results/layerB_premium_cap_test.json`.

Setup: the ordinary entrepreneur premium π_L is fixed at 4.9%, 2%, 1% or 0%. All other Layer B v1 parameters stay as estimated; only β is re-solved so the capital market clears.

| π_L | Top 1% | Top 10% | Top 0.1% | Business wealth share |
|---|---|---|---|---|
| 4.9% | 32.4% | 69.0% | 21.3% | 29.0% |
| 2% | 28.7% | 66.3% | 18.7% | 25.5% |
| 1% | 27.8% | 65.9% | 18.0% | 24.5% |
| 0% | 27.2% | 65.5% | 17.3% | 23.5% |

Takeaways:
- The premium accounts for about 5 points of the top 1% share. The rare high-growth state (H) does most of the work: Layer A's top 1% was 18%.
- Removing the premium moves the top 0.1% share toward the data (about 15–17%).

Next: full recalibration with π_L capped by the evidence, adding the top 0.1% as a target and testing superstar earnings for business owners (Smith, Zidar & Zwick).

## Layer B v2 (3 Oct 2026): capped premium plus superstar owner earnings
Files: `code/calibrate_B2.py` and `code/calibrate_B2b.py` (two starting points), `notebooks/LayerB_v2.ipynb`, `results/layerB_v2*_params.json`, `results/layerB2*_calibration_log.txt`, `results/layerB_v2_results.csv`.

Changes:
- π_L fixed at 1% (Moskowitz & Vissing-Jørgensen 2002; Kartashova 2014).
- H owners earn a multiple of labor earnings while working (Smith, Zidar & Zwick 2019).
- New targets: top 0.1% = 15.0% (DFA Q2 2026); top 1% earnings share = 18% (SCF 2022).

Result (best start, B; sum of squares 56; start A gives 58):

| Parameter | Value |
|---|---|
| π_H | about 0 |
| σ_H | 62% |
| H earnings multiple | 12.8× |
| H population | 1.6% |
| ι | 0.20 (0.44 in start A: weakly identified) |
| β | 0.951 |
| φ₁ | −2.8 |

Fit:
- Top 0.1% 17.0% (15.0); top 1% 32.3% (32.5); top 10% 70.1% (68.9).
- Top 1% earnings 15% (18); business capital share 31% (30.5); charity 13% (13).
- Misses: taxable deaths 0.28% (0.14); wealth 75+ 1.31 (1.53).
- Not targeted: Gini 0.83 (0.83). Pareto exponent 1.36 (v1: 1.13).

Interpretation: with the premium capped, the model sets the high-growth premium to about 0. Concentration comes from superstar owner income plus risky business returns, consistent with the evidence.

Tests: grid 220 vs 160 changes results by under 0.05 pp; Euler errors mean 10^-5.0.

Next: dynastic bequest type (target elasticity 0.37); a target for persistence of business ownership (to pin down ι); look into the excess of taxable estates.

## Spec v2.1 (4 Oct 2026)
The methodology doc was updated to match the model as built (Claude Doc, same link, now titled v2.1):
- A status table.
- The as-built fortune-creation section: log-normal risky returns, the 1% premium cap, superstar owner earnings, no scale limit.
- Two Layer C estate-tax avoidance channels:
  - freeze transfers (GRATs / sales to grantor trusts, an option on the upside above a hurdle);
  - valuation discounts.

  Disciplined by IRS effective rates (about 9.7% for 2021 filers) and the Forbes-vs-estate-return ratio of about 0.5 (Raub, Johnson & Newcomb). Fallback: an effective transfer-tax cap by tier.
- The dynastic bequest type is now calibrated jointly with Layer C.
- A pre-registered Layer C section: targets, untargeted checks, success criteria, and the definition of the main result (switch C off, hold parameters fixed, re-solve r, w and the budget; Shapley decomposition by channel).
- Production status: r re-solving for the experiments is still to build.
- New policy experiments: a 10-year minimum GRAT term; limits on valuation discounts.
- Open decisions 18–21 added (premium cap, source of top fortunes, avoidance channels, dynastic-type timing).

## Literature review v1 (4 Oct 2026)
Claude Doc: "Literature Review — Dynastic Wealth Preservation and Inequality", https://claude.ai/code/artifact/9ea169ec-2285-44bc-b7b3-2e79ff7a958e

Findings:
- Closest competitor found: Luo (NYU job-market paper; publication status to verify). Gifts during life act as estate-tax avoidance, so the top 1% share is U-shaped in the tax rate. Luo has no trusts, GST tax, freeze transfers or valuation discounts.
- No paper found models dynasty trusts or the main avoidance strategies in a calibrated model of the wealth distribution.

To do before claiming novelty: a deeper search of SSRN, job-market papers 2023–26 and law-and-economics journals.

## Layer C1 v1: dynasty trusts, tier 1 (4 Oct 2026)
Files:
- Code: `code/layerC.py`; `code/run_c1.py` runs one configuration.
- Notebook: `notebooks/LayerC1_v1.ipynb`.
- Results: `results/layerC1_*.json`.
- The spec doc has an "As built: stage C1" paragraph.

Model:
- Two assets: liquid wealth a, and trust assets d held outside the estate.
- Three categories: no trust; heir (non-grantor trust, taxed at τ_ng = 24%); founder (grantor; pays trust income tax from liquid wealth).
- Trust assets:
  - earn the family's return state;
  - fees: $10k a year plus 0.2% of assets;
  - setup cost $50k;
  - pass to heirs free of estate and GST tax.
- Funding is tax-free up to the remaining exemption, approximated as max(E − d, 0). Above it, gift tax is 40% per dollar (tax-exclusive).
- Founding friction: a logistic draw each period, as a share of wealth (money-metric). Placeholders: mean 5%, scale 3%. Calibrated in C4.

Decision (Claude, flagged for Rich and the economist): **binding trust rules are the baseline.**
- The founder gets no payouts (completed gift; IRC §2036).
- Heirs may take the income but cannot dissolve the trust.
- Trusts under $100k are terminated (UTC §414).

The first build followed the spec's wind-up option. There, heirs dissolved 86% of trusts and founders took the income, so the effect was nil. Wind-up and founder payouts are kept as robustness cases.

Tests:
- With trusts off, Layer C reproduces Layer B to about 1e-15 on every moment and policy.
- Inheritance convolution checked against a brute-force version (1e-14).

Results: Layer B v2 parameters held fixed, nothing recalibrated, r fixed.

| | Trusts off | C1 baseline | No estate tax, no trusts | Wind-up allowed | No gift tax |
|---|---|---|---|---|---|
| Top 0.1% | 17.0% | 23.3% | 20.7% | 17.1% | 26.3% |
| Top 1% | 32.3% | 38.2% | 36.5% | 32.5% | 41.1% |
| Effective estate tax, estates above exemption | 27.7% | 10.1% | 0 | 18.4% | 7.0% |
| Trust assets / wealth | 0 | 15.7% | 0 | 1.7% | 20.2% |
| Trust share of top-0.1% wealth | 0 | 48% | 0 | 8% | 58% |
| Take-up, top 1% / top 0.1% | 0 | 27% / 41% | 0 | 5% / 14% | 30% / 49% |
| Capital supply (index) | 100 | 114 | 107 | 101 | 120 |

Other runs:
- Founder payouts allowed: about the same as the baseline (top 1% 38.1%).
- τ_ng = 18%: top 1% 40.1%.

Findings:
1. **Binding dynasty trusts raise top concentration more than abolishing the estate tax.** The mechanism is commitment: heirs consume trust income but cannot spend the principal, so dynastic wealth compounds. Tax savings alone (the wind-up case) do almost nothing with the current bequest preferences.
2. The effective estate tax on large estates falls to about 10%, close to the IRS figure of about 9.7% for 2021 filers. It is not targeted yet.
3. The parent–child elasticity does not move (0.18). The elasticity is measured at child age 35–39, before most trust inheritances. This is a reason to look at the measurement timing in C3, not only the dynastic type.
4. Founding mostly happens late in life (ages 70+), except for the very rich, consistent with Kopczuk (2007).

Numerics flagged:
- **Trust-grid sensitivity is material.** nd = 55 gives top 1% 37.4% (vs 38.2%) and top 0.1% 22.2% (vs 23.3%), which is about 15% of the effect.
  - nd = 70 gives 37.7% and 22.5%. Results move by about ±0.5 pp without converging monotonically, which looks like discretisation noise in the trust grid.
  - The C1 effect on the top 1% is therefore +5.1 to +5.9 pp.
  - To fix in C4: interpolate policies in d, or refine the grid where the mass is.
- The generation fixed point needs Anderson acceleration (tolerance 1e-8). One run is about 9 minutes with trusts on, against 30 seconds off. Speed work is needed before the C4 calibration.

Next (C2):
- Tier 2: family office / advanced management.
- Freeze transfers (GRATs) and valuation discounts. The no-gift-tax run bounds the freeze channel.
- Behavioral return gap.

Data still needed: take-up by wealth (SCF 2022 extract), IRS SOI 2024 Table 1, and the size of dynasty-trust assets.

## New data: IRS SOI 2024 and SCF 2022 (5 Oct 2026)
Rich downloaded both files. They are processed into `data/targets_v2.csv`, with the raw files in `data/raw/`.

**IRS SOI, estate tax returns filed in 2024 (Table 1).** Net estate tax as a share of gross estate:

| Gross estate | Net estate tax / gross estate |
|---|---|
| All returns | 9.4% |
| $10–20M | 2.9% |
| $20–50M | 8.9% |
| $50M+ | 12.5% (19.2% net of spousal bequests) |

Charitable deduction / gross estate: 15.1% overall, 23.6% for $50M+.

**SCF 2022 (TRUSTS > 0).** Share of households holding a trust:

| Group | Holding a trust |
|---|---|
| All | 2.7% |
| 90th–99th percentile | 11.3% |
| Top 1% | 12.5% |
| Top 0.1% | 21.4% |

Trust assets / net worth: 2.8% in the top 1%, 4.4% in the top 0.1%. The file is `data/scf2022_trusts_by_wealth.csv`. These are lower bounds: the SCF records trusts in which the household has an interest, so irrevocable dynasty trusts are likely under-reported.

Measurement issue for C4: the DFA and SCF top shares probably exclude much dynasty-trust wealth. The model reports top shares both with trust assets and with liquid wealth only.

## Layer C2 v1: family office tier, freeze transfers, valuation discounts (5 Oct 2026)
Files:
- Code: `code/layerC.py` (`tier2=True`); `code/run_c.py` runs one configuration.
- Notebook: `notebooks/LayerC2_v1.ipynb`.
- Results: `results/layerC_*.json`.
- The spec doc has an "As built: stage C2" paragraph.

Model: there are five categories: none; heir or founder in tier 1; heir or founder in tier 2. Tier 2 (placeholders, to calibrate in C4):
- Entry: $500k setup plus a planning friction (logistic, mean 5% of wealth). Downgrading is free. Heirs inherit the tier.
- Tax drag: τ = 10% on personal wealth and grantor-trust income; 13.3% on non-grantor trust income.
- Running cost: max($1M, 0.5% of assets) a year, shared between personal and trust assets.
- Freeze transfers: half of personal wealth in rolling GRATs. Each period's return above the 5% hurdle passes to the family trust gift-tax free. This applies to founders and heirs.
- Valuation discount: 30% on personal wealth in business states, for estate and gift tax.
- Behavioral gap λ_b: built, default 0.

Fixes made while building C2 (they also restate C1; **C1 v1 numbers are superseded**):
1. A grantor pays trust income tax from personal cash only up to half of cash on hand; the trust reimburses the rest (reimbursement clause / grantor status toggled off). Family-office costs are shared pro rata, with the personal share capped at a quarter of cash on hand.
2. **The planning friction no longer carries option value.** The logistic draw's negative tail had given liquidity-poor trust holders a planning "bonus" without bound, scaled by their wealth. The friction now shifts the choice only; the value is the probability-weighted value of the two branches.
3. Values beyond the top of the EGM grid are extrapolated with the envelope condition and capped at the theoretical upper bound (the warm-glow limit). Extrapolated consumption is kept non-negative.
4. The money-metric scale for logit choices uses the marginal utility at the best free option, not at the first option.

Tests:
- With trusts off, the model reproduces Layer B to 1e-14, with tier 2 on or off.
- The convolution matches a brute-force version (2e-14).
- The budget check on stored choices holds to float32 rounding.

Results (Layer B v2 parameters, nothing recalibrated):

| | Trusts off | Tier 1 only (C1 restated) | Tiers 1+2 | Full use of tier 2 |
|---|---|---|---|---|
| Top 1% | 32.3% | 35.8% | 35.9% | 36.1% |
| Top 0.1% | 17.0% | 19.8% | 19.9% | 20.0% |
| Estate tax / gross estate, $50M+ | 30.3% | 26.9% | 26.5% | 26.1% |
| Trust take-up, top 1% / top 0.1% | 0 | 44% / 64% | 44% / 64% | 44% / 64% |
| Tier 2 take-up, top 1% / top 0.1% | 0 | 0 | 2% / 12% | 11% / 31% |
| Freeze transfers / GDP | 0 | 0 | 0.04% | 0.20% |

"Full use" means no tier-2 friction, cheap family offices ($250k minimum) and all personal wealth in GRATs. Turning GRATs off in the full-use case lowers the top 1% share by 0.15 pp.

Findings:
1. **The binding dynasty trust (tier 1) does almost all the work:** +3.5 pp on the top 1%. Tier 2 and the avoidance channels add at most about 0.3 pp with current preferences.
   - Freeze transfers can be sizeable as flows (0.2% of GDP at full use, about $55B a year). But they mostly move wealth from a family's personal account into its own trust, which raises measured concentration only through later tax savings.
2. **The model's estate tax on the largest estates is still about twice the IRS figure** (26% vs 12.5%; 19% net of spouses). The gap comes mostly from large estates of families that never plan (about a third of the top 0.1% have no trust).
   - C4 has to match this, plus take-up, together. Expect lower frictions and more avoidance at the top, and possibly a larger valuation discount.
   - The model has no spouse, so the right IRS comparison is the 19% figure that nets out spousal bequests.
3. **The parent–child elasticity is still 0.18, against 0.37 in the data.** Avoidance technology alone does not create dynastic persistence in this preference setting, which points to C3 (dynastic bequest type) as the next lever.

Numerics: one C2 run takes about 20–25 minutes (C1 about 11). Speed work is needed before C4.

## Mid-project review and the tax shield × lock-in framing (6 Oct 2026)
- **Fable review.** Claude Doc "FODT Model — Mid-Project Review": https://claude.ai/code/artifact/b52e3f3c-dc9d-420b-9af2-0bfdae6ecd30
  - Verdict: the method is sound, but the +3.5 pp is an upper bound. It scales with a trust stock of 17% of wealth (SCF: 2.3%), take-up 3–4× the SCF, and a fixed r.
  - The doc has a decision table for Rich and the economist.
- **Proposed framing** (from discussion with Rich; added to the spec): every instrument is a point on two dimensions, tax shield and lock-in. Lock-in becomes a parameter ℓ ∈ [0, 1], absorbing the founder valuation weight θ.
- **Proposed main experiment:** a 2×2 of shield on/off × lock-in on/off. It replaces the Shapley decomposition by channel.
- **Model changes proposed:**
  - Family partnerships carry both a valuation discount and illiquidity for heirs.
  - Tier 2 gets soft lock-in, so it should be tested before moving it to an appendix.
- **Status:** awaiting decisions. Quick-win checks are listed in section 5 of the review.

## Quick wins from the review (6 Oct 2026)
Runs: `results/layerC_{c1_base, c1_r5, off_r5, c1_ko5, c1_nd55, c1_nd70, c2_base}.json`. Code additions in `layerC.py`: rank-rank correlation, elasticity at child age 50–54, and cap diagnostics.

**Trust-wealth data** (added to `data/targets_v2.csv` and the lit review):
- CRS R48879 (2026): $5.4–10.8T in non-grantor trusts, from Form 1041 income.
- Galle, Gamage & Lord (2026, U. Penn. Law Review), "Taxing Dynasties": at least $4.5T in perpetual trusts, one-third or more of top-0.1% wealth; 80–90% of wealth set aside for heirs never faces estate tax.
- South Dakota: $800B in trust assets (end-2024).
- Model: 17% of wealth (about $30T), 2–4× the estimates, but 44% of top-0.1% wealth, in line with Galle et al.
- **Proposed targets:** trust share of top-0.1% wealth ≥ 1/3; top-1% share 10–20%. These replace the "2× SCF" bound.

**Interest-rate check:**
- Top 1% effect: +3.5 pp at r = 6%, +3.0 pp at r = 5%.
- With trusts on, the capital market clears at about r = 5.5% (interpolated). Against the r = 6% baseline, the effect stays about +3.5 pp.
- Equilibrium damping of top shares looks small; the trust stock is more sensitive (17% → 14%). A proper r solve is still to build.

**Caps and truncation:**
- The safety caps bind for under 0.02% of trust holders.
- Keeping 5 options per state instead of 3 changes nothing.

**Grid:**
- Top 1% share is 35.8 / 36.1 / 36.1% at nd = 40 / 55 / 70.
- The trust stock is 17.3 / 20.0 / 19.4%, so it needs a finer grid.
- **Decision: use nd = 55 from now on.**

**Parent–child persistence** (trusts off → on):
- Log elasticity: 0.19 → 0.18 at child age 35–39.
- Rank-rank: 0.39 → 0.37 at 35–39, and 0.26 → 0.24 at 50–54.
- The log measure understates persistence because of the low-wealth tail.

Still to do: the Pareto exponent with trusts on; estate-tax bins under a couple/single exemption mix.

## Lifetime gifts added to C3 (7 Oct 2026; agreed with Rich)
Prompted by Mi Luo's job-market paper, "Unintended Effects of Estate Taxation on Wealth Inequality" (NYU 2017; still a working paper; she is now at Peking University, Guanghua). Its finding: higher estate taxes induce more lifetime gifts, and can raise top inequality.

**Design:**
- One gift decision per family, when children form their households (child ages 20–34).
- Warm-glow gift motive, with one new parameter.
- Tax rules: $19k annual exclusion per recipient; tuition and medical bills paid directly are excluded (IRC 2503(e)); above that, the lifetime exemption and then 40% gift tax.
- The gift is received as starting wealth, through the existing convolution.
- Left out: education as a separate asset (already in earnings persistence) and gifts that respond to each child's circumstances.

**Targets** (in `data/targets_v2.csv`):
- Inter vivos share of transfers: 27% (Sabelhaus 2024, SCF 2021).
- 17% of households give in a given year.
- Form 709 filing year 2021: 251k returns, $182.6B in gifts, under 1% taxable.

Placement: replaces heir anticipation in C3. C3 = dynastic type + lifetime gifts.

Order of work: speed work and the r solve, then C3, then the C4 joint calibration.

Note: the IRS website is blocked by this environment's network proxy for direct file downloads, so SOI spreadsheets need to be downloaded by Rich (as with the estate tax table).

## General equilibrium: interest rate re-solved for tier 1 (7 Oct 2026)
Files: `code/equilibrium.py` (secant solve on r, warm starts, one economy in memory at a time); `code/run_ge.py`; `results/layerC_ge_c1.json`.

Parameters are held at Layer B values (no recalibration), with nd = 55. Switching tier-1 trusts on, r clears at **5.49%** (down from 6.00%), after 3 evaluations (47 minutes).

| | Trusts off (r = 6%) | Tier 1, r fixed at 6% | Tier 1, r re-solved (5.49%) |
|---|---|---|---|
| Top 1% | 32.3% | 36.1% | 36.2% |
| Top 0.1% | 17.0% | 20.0% | 19.8% |
| Trust assets / wealth | 0 | 20.0% | 18.1% |
| Trust share of top-0.1% wealth | 0 | 47% | 46% |
| Capital supply | 9.89 | 11.63 | 10.53 |

**Finding:** the equilibrium feedback lowers r by about half a point but leaves the top-share effect intact (+3.9 pp on the top 1%). The review's concern that GE would cut the effect to about 1 pp is not borne out. A lower r hits all wealth, while the trust advantage is mostly the absence of transfer tax.

**Speed:** about 15 minutes per evaluation at nd = 55. Warm starts barely reduce generations (15 vs 18). For C4, a fast configuration is being tested (nxi = 5, ko = 2, tolerance 1e-6), with final results verified at full resolution.

**Fast configuration for calibration (tested 7 Oct):** 5 return-shock nodes, 2 stored options per state, distribution tolerance 1e-6.
- Moments are within 0.15 pp of the full configuration (top 1%: 35.98% vs 36.13% at nd = 55; trust stock 19.7% vs 20.0%).
- Runtime: 8.0 vs 13.5 minutes at nd = 55, and 5.5 vs 9.6 minutes at nd = 40.
- **Decision:** C4 calibrates in the fast configuration at nd = 55, and final results are re-run at full resolution.

## C3 built: lock-in, tax-shield switch, founder trust weight, lifetime gifts (7–8 Oct 2026)
Code (`layerC.py`):
- The decision step is refactored into a menu kernel (`_menu`) with an optional gift stage at parent age 55–59.
- `lock` (ℓ): the share of principal heirs cannot reach.
- `trust_weight` (θ): the founder's weight on trust assets in the bequest motive.
- `shield`: when False, trust assets are in the taxable estate.
- `gifts`: one warm-glow gift decision. $300k per family exempt at that stage; above that, the remaining exemption, then 40%.
- Distribution: gift-split parent–child pairing (`_push_split`); small gifts below `gift_pool_usd` are pooled; sparse push path.

Tests:
- With everything off, the model reproduces Layer B to 9e-16.
- With the C3 features at their defaults, it reproduces C2/C1 exactly (max difference 0.0).
- The sparse push path matches the dense one to 4e-9.

**Shield × lock-in 2×2** (fast configuration, nd = 40, Layer B parameters, r = 6%):

| | Lock-in (ℓ = 1) | No lock-in (ℓ = 0) |
|---|---|---|
| Tax shield | top 1% 35.7% (+3.4 pp) | 32.9% (+0.6) |
| No shield | 31.8% (−0.5) | — (≈ trusts off, 32.3%) |

Intermediate cases:
- ℓ = 0.5: 33.0% (+0.7).
- θ = 0.5: 33.4% (+1.1); trust stock 5.3% of wealth; trust share of top-0.1% wealth 14%.

**Finding:** the effect comes from the combination. Lock-in without the tax shield, or the shield without lock-in, does little. θ is a strong lever for matching the trust stock in C4.

**Lifetime gifts** (gift warm glow φ_g = −2, uncalibrated, β not re-fit):
- Without trusts: top 1% 33.4% (+1.0); inter vivos share 26% (target 27%).
- With trusts: top 1% 43.8–46.0%, trust stock 24–26%, capital supply +30%.
- Parent–child log elasticity rises 0.18 → 0.32 (target 0.37); rank-rank 0.37 → 0.44.
- Gifts are an extra transfer motive, so saving rises. These levels are not interpretable until β and the bequest and gift strengths are re-fit jointly (C4).

Numerics, gifts:
- Exact pairing by gift bin: 62 min per run, against 6 min without gifts.
- Pooling gifts below $1M: 41 min. Persistence measures match (0.321 vs 0.321), but top shares differ by 2 pp. The cause is gift-grid coarseness at the top, not the pooling. **Open:** a finer gift grid above $1M and further speed work before C4.

## Dynastic-type diagnostic (8 Oct 2026) — decision: leave it out of C4, keep as robustness
Settings: 20% of families have a strong bequest motive (φ₁ = −10 vs −2.8); 90% of children inherit their parent's type. Fast configuration, nd = 40, Layer B parameters.

| | Trusts off | + dynastic type | Tier 1 | Tier 1 + dynastic type |
|---|---|---|---|---|
| Top 1% | 32.3% | 35.1% | 35.7% | 39.8% |
| Wealth 75+ / mean (target 1.53) | 1.30 | 1.55 | 1.43 | 1.68 |
| Deaths owing estate tax (target 0.14%) | 0.29% | 0.54% | 0.24% | 0.36% |
| Rank-rank, child 35–39 | 0.39 | 0.37 | 0.37 | 0.34 |
| Trust take-up, top 1% | — | — | 43% | 70% |

**How each feature moves the moments:**
- **The dynastic type** raises old-age wealth, bequest flows, trust take-up and top shares, but *lowers* parent–child persistence and pushes taxable deaths further from target.
- **Gifts** do the opposite on persistence (they raise it) and lower old-age wealth.
- **The trust weight θ** mainly scales trust use.

So the dynastic type does not supply what the data are missing on persistence. Its other effects overlap with the common bequest strength φ₁, θ and the friction, which C4 re-fits anyway.

Kept as a robustness check ("a dynastic minority"); runtime about 2.5× (nyz 15 → 30). Rich agreed to skip it (8 Oct).

## Gift-model speed work and C4 launch (8 Oct 2026)
**Speed:**
- Sparse inheritance kernels (`_conv_c0_children`, `_conv_trust_children`, with lottery tables) are used automatically when cheap.
- Gift grid above $1M: ratio-3 spacing (`gift_grid_factor`).
- **Large-gift pairs are merged across children's initial types** (`gift_merge_types`). Against exact pairing, every moment is within 0.1 pp (top 1% 47.18% vs 47.09%; rank-rank 0.436 vs 0.436). Runtime: 16.5 vs 58 minutes.
- Rejected: pooling thin large-gift bins within a pair (`gift_min_pair`). It changes top shares by 12 pp, because the exact pairing of very large gifts with the giver's later estate drives the top tail.

**C4 launched:** `code/calibrate_C.py`, start A.
- 7 parameters: β, φ₁, χ, friction mean and spread, trust weight θ, gift strength φ_g.
- 11 targets: K gap; top 1%, 0.1% and 10% of personal wealth; trust share of top-1% and top-0.1% wealth; IRS $50M+ rate net of spouses; taxable deaths; charity; wealth 75+; inter vivos share.
- Fast configuration (nxi = 5, ko = 2, nd = 40), with gifts.
- About 17 minutes per evaluation, up to 90 evaluations; log in `results/layerC4_v1_log.txt`.
- The Layer B return process is held fixed in this pass.

**C4 restart (8 Oct, afternoon):**
- The cloud workspace restarted after the first evaluation, which killed the calibration.
- `calibrate_C.py` now writes every evaluation to `results/layerC4_<label>_evals.jsonl` and resumes from the best cached point after a restart.
- Relaunched. Hourly check-ins are scheduled to keep the session active and to relaunch if needed.
- **C4 replay cache (8 Oct, evening):** the cloud workspace restarted again within the hour. Resuming from the best point made `least_squares` recompute a full 8-evaluation Jacobian (about 2 h) after every restart. Each restart now replays the same deterministic optimiser path from start A: evaluations already in the cache return instantly, so a restart costs at most the evaluation in progress. Check-ins are every 30 minutes.
- **C4 hosting (8 Oct, 17:21 ET):** even with 10-minute check-ins, the cloud workspace comes back freshly booted at every check-in, so background jobs don't survive between turns. A run can make progress only during an active turn, or on Rich's computer. The cloud check-ins are paused until Rich chooses one.
- **Measuring trust wealth (9 Oct):** every data source understates trust wealth. Income-based estimates miss assets that throw off no taxable income, grantor trusts are mostly invisible, dynasty trusts never appear on estate tax returns, and offshore holdings are outside US data. So:
  - outside estimates are treated as a floor;
  - C4 targets trust *shares* of top wealth with wide SEs and does not target the dollar stock;
  - C5 will report the Layer C effect at low, medium and high trust shares of top-0.1% wealth (proposed 25/40/55%).
  - The offshore and non-citizen channels are out of scope and flagged as an extension.
  - Written up in the spec ("Measuring trust wealth") and the lit review ("Why every estimate above is a floor").
- **C4 v1 abandoned; v2 convergence fix (9 Oct):** the v1 run on Rich's PC completed 7 evaluations, but the finite-difference Jacobian was dominated by noise.
  - **Cause:** the generation fixed point stopped on the L1 mass change alone (tol 1e-6), after about 14 generations. Trust wealth at the top was still accumulating (slow mode about 0.9 per generation, invisible to the mass test), so results depended on the warm start. For example, the trust share of top-1% wealth was 0.251 cold and 0.266 → 0.276 after 4 and 8 more generations.
  - **Fix in layerC.distribution:** an optional wealth-weighted stopping test `wtol` (sum|dD|·x / sum(D·x)) and a wealth-weighted Anderson fit. Test at start A: converges in 23 generations to a top-1% trust share of 0.311, with 0.3108 after 4 further plain generations (stable).
  - **Calibration changes:** C4 now uses wtol = 1e-4 and maxgen = 120, and warm-starts from the latest converged point. Finite-difference steps are per parameter: beta 0.001 (v1's 0.02 moved beta by 0.019 and the objective from 474 to 66,430).
  - **Implications:** converged moments at start A are higher than v1 reported. Top 1% total is 0.421 (vs 0.397), top 0.1% trust share 0.454, K gap larger. Earlier C1–C3 results used the same loose test, so their levels are understated; the comparisons between variants are probably less affected. To recheck in the final full-resolution runs.
  - **New run:** label v2 (the v1 cache is not reused).
- **C4 v2 result (10 Oct): stopped at evaluation 42 by choice, best point taken as the calibration.** Results in `results/layerC4_v2_params.json`.
  - **Path:** ss 716 → 132 → 68 → 63 → 61 → 60.2. The last three iterations gained about 1 each, and the trust region was shrinking (eval 41 a rejected step), so the run is effectively at a local optimum.
  - **Parameters:** beta 0.9449, phi1 −2.217, chi 0.281, fric_mu 0.0756, fric_nu 0.0389, trust_weight 0.800, gift_phi −0.889.
  - **Fit:** K gap −0.2%; top 1/0.1/10 liquid-only shares 33.0/16.5/71.5% (targets 32.5/15.0/68.9); trust share of top-1/0.1% wealth 26.8/38.3% (15/40).
  - **Misses (share of ss = 60):**
    - wealth of 75+ relative to the mean 1.27 vs 1.53 (26);
    - lifetime gifts' share of transfers 13% vs 27% (12);
    - estate tax as a share of gross estates over $50M 27% vs 19% (7.5);
    - trust share of top-1% wealth (5.5);
    - charity as a share of taxable estates 20% vs 15% (3).
  - **Untargeted:** top 1% total share 38.3%, parent–child wealth elasticity 0.23, rank-rank correlation (child aged 35–39) 0.40, trust take-up in the top 1% 26%.
  - **Interpretation (for the economist):**
    - The 75+ gap is structural. The model has no late-life saving motive beyond warm-glow bequests; out-of-pocket medical risk is the standard candidate (De Nardi, French & Jones 2010).
    - The estate-tax rate gap likely reflects missing at-death avoidance: the marital deduction and couples, plus valuation discounts outside trusts.
    - Under the "measured trust wealth is a floor" decision, the top-1% trust share above 15% may not count as a miss at all.

## C4 v3: changes pre-registered on 13 Oct, before any v3 result

**Diagnosis (v2 best point vs SCF 2022).** Wealth peaks too high at 55–69 and falls too fast after 75.

| Age | 60 | 65 | 70 | 75 | 80 | 85+ |
| --- | --- | --- | --- | --- | --- | --- |
| Model mean ÷ overall mean | 1.91 | 2.11 | 1.82 | 1.52 | 1.21 | 0.91 |
| SCF | 1.58 | 1.73 | 1.62 | 1.54 | 1.69 | 1.31 |

Excluding the top 1%, the 75+ ratio is 0.99 in the model against 1.33 in the SCF, so the shortfall is broad-based and not only at the top. Cause found: the survival data end at 85 and the model kills everyone at 90. In the 2010 SSA table, P(reach 90 | 75) is about 0.35. Late-life medical costs are concentrated after 90 (De Nardi, French & Jones 2010), so adding them would have little effect unless life is extended too.

**Changes:**

1. **Life extended to 99 (T = 16).** `data/surlife_ext99.dat` continues our survival series from age 85, using the age-to-age ratios of the average male/female q(x) in the SSA 2010 period life table (ages 84–94). The last period ends in death with certainty.
2. **Medical costs:** `ParamsC.med_on`; `layerC._decide` subtracts them from cash on hand, with a floor.
   - Mean annual out-of-pocket cost: $1,100 × exp(0.106 × (age − 75)), in 1998 dollars. This is fitted to the means De Nardi–French–Jones report: about $1,100 at 75, $9,200 at 95 and $17,700 at 100.
   - Permanent-income gradient: proportional to exp(2.79 × PI rank), giving a top/bottom-quintile ratio of about 9 at age 95 (their figures: $15,800 vs $1,700). The proxy for permanent income is the earnings state y.
   - Floor on cash on hand: their consumption floor of $2,663 (1998 dollars). Below the floor, assets earn no marginal resources, as with Medicaid/SSI.
   - Prices: CPI 1998 → 2022 factor of 1.80.
   - **Caveat:** the age and income profile is a parametric fit to a few published points, not their full estimated process. It is deterministic given age and income type (no medical-cost shock). Flag for the economist; the full DFJ process is a refinement.
3. **Target corrections (measurement only, IRS SOI 2024 Table 1):**
   - Estate tax ÷ (gross estate − bequests to a surviving spouse), taxable returns over $50M: 15.85B ÷ (82.35B − 15.75B) = **0.238**. **Correction:** v2 used 0.192 and labelled it "net of spousal bequests", but 0.192 is tax ÷ gross *including* spousal bequests. Since the model has no spouse, 0.238 is the comparable figure.
   - Charity ÷ (gross − spousal bequests), all taxable returns: **0.181** (v2: 0.15).
4. **Trust shares become one-sided floors,** per the "measured trust wealth is a floor" decision: top 1% ≥ 0.10 (CRS low end), top 0.1% ≥ 1/3 (Galle, Gamage & Lord). Upper discipline comes from the personal-wealth top shares and the capital market.

**Not changed:**

- No valuation discount at death was added; the re-based target removes the need.
- No couples: the $30M exemption already treats each household as a couple.
- Layer B parameters (returns, earnings) stay at their v2 values.

**Start point:** the v2 best point. **Label:** v3.
- **Amendment (same day, still before any v3 result was seen): age range 25–94 instead of extending to 99.**
  - **Why:** extending to 99 with T = 16 broke the distribution. It tracks parent–child pairs for two overlapping generations, which requires parents to be dead when their children become parents (T ≤ 2·gap = 14). With T = 16, the parent cohort's mass leaked (0.40 → 0.29 in one generation).
  - **Instead:** model entry at age 25 (`ParamsB.age0`), keeping T = 14 so life runs from 25 to 94. Retirement stays at 65 (TL = 8). Survival to 89 uses the SSA-extended series. There is now an assert that T ≤ 2·gap.
  - **Effect:** forced death moves from 90 to 95. P(reach 90 | 75) ≈ 0.35 is now inside the model; P(reach 95 | 75) ≈ 0.14 is not.
  - **Age-indexed moments:** re-indexed from `age0`, so the 75+ ratio, the child 35–39 and 50–54 pairs, and the medical-cost ages keep their meaning.
  - **Gifts:** still made when the child enters the model, now at child age 25 (parent age 60).
- **Amendment 2 (no result seen yet): no deaths up to the end of age 64.**
  - **Why:** the gift step at the parent's age when the child enters (age gap) assumes the parent survives that period. With entry at 25 that is age 60–64, and allowing deaths there dropped those estates.
  - **Rule:** `sur[:gap+1] = 1`. This equals the old "no deaths before 60" when age0 = 20 (checked: survival vector identical), and becomes no deaths before 65 when age0 = 25. Cost: about 5% of mortality at 60–64 is ignored.
- **v3 test results at the v2 best point (13 Oct).**

  | Configuration | 75+ ÷ mean wealth | ss (v3 targets) |
  | --- | --- | --- |
  | Age range 25–94, no medical costs | 1.180 | 68 |
  | With medical costs | 1.196 | 72 |
  | With medical costs, phi2 = 3 (exploratory, not adopted) | 1.175 (excluding top 1%: 1.02, SCF 1.33) | 85 |

  - In every configuration wealth peaks near 2.0× the mean at 65 and falls to about 0.7× by 90 (SCF: flat around 1.5–1.7 through 80).
  - The old-age gap is robust to the life horizon, medical costs and the luxury bequest parameter.
  - **Leading explanation (for the economist):** the model's wealth-to-earnings ratio is about 4.9 (mean wealth $1.06M against model-implied mean earnings of $216k a year, because dollars are mapped by wealth), versus about 7.5 in the SCF. Pensions (40% of mean earnings, consistent with Social Security relative to earnings) are therefore large relative to private wealth, so retirees need little wealth. Housing and other illiquid wealth are absent.
  - Not fixable within this model without restructuring. Report as a limitation.

## C5 main experiments (13 Oct): on the C4 v2 calibration
- **Decision (Rich):** run the main experiments on v2 now. v3 (corrected estate and charity targets, trust floors, life to 94, medical costs) can be recalibrated later and the experiments rerun.
- **Experiments** (`code/run_c5.py`), each with r re-solved so the capital market clears; the PE results at r = 6% are kept too:
  - **base:** as calibrated;
  - **off:** trusts=False, lifetime gifts kept;
  - **lock0:** tax shield kept, lock-in removed (lock = 0);
  - **noshield:** lock-in kept, shield removed (shield = False).
- **Solver settings:** secant on r, |excess| < 0.2% of corporate capital. Fast configuration (nd = 40, nxi = 5, ko = 2), with the wtol convergence check.
- **Regression test** after the v3 code changes (default settings = v2): every reported moment of v2 evaluation 42 reproduced within 0.0003 (e.g. top-1% personal share 0.3294 vs 0.3296; 75+ ratio 1.271 vs 1.272).

## Independent model review (Fable), 14 Oct: findings and response
**Verdict:** careful, well-documented work. The "trusts add wealth rather than relabel it" finding is genuine. But the v2 headline (+4.7 pp on the top 1%) is best read as an upper-bound case, not a point estimate, and v2 numbers should not go in the paper.

**Findings (checked where noted):**

1. **The effect is nearly an accounting identity of the calibrated trust share s of top-1% wealth.** top1_off ≈ top1_on·(1−s)/(1−τ) gives 32.3% against the 33.5% simulated (checked). The effect scales about 0.17 pp per point of s: about +2.1 pp at s = 10%, +3.3 pp at 15%, +4.7 pp at 26.8%. v2's s = 26.8% is above the CRS 10–20% range, so the effect is overstated relative to that evidence.
2. **The 75+ target, which the model cannot hit, probably pushes s up.** Trusts are old people's wealth: the 75+ ratio is 1.11 without trusts and 1.27 with them. Testable with one refit.
3. **Measurement concept:** the DFA scales the SCF to Financial Accounts totals, and assets held "through trusts" are mapped to asset classes (Batty et al., DFA chapter). Trust assets that are in the FA household totals but unreported in the SCF are allocated like other holdings. So the DFA is "total wealth with trust assets partly misallocated", not purely "personal wealth". Also, the model's personal top-1% share divides by personal wealth only; divided by total wealth it is 28.7%. Whether FA households include personal trusts still needs checking against the FA documentation. **Response:** bound it by calibrating total-wealth top shares to the DFA as an alternative, and report both concepts.
4. **Grid inconsistency (checked):** DECISIONS.md (7 Oct) says "C4 calibrates at nd = 55", but calibrate_C.py and the C4/C5 runs use nd = 40. The trust stock moves about ±15% across nd = 40/55/70. Base and off must be verified at full resolution, and the paper must state the grid used.
5. **The lock0 experiment is all-or-nothing:** heirs retain or withdraw everything above the lower bound, with no partial withdrawal. Disclose, or add a partial option.
6. **v3's one-sided trust floors remove all upper discipline on trust wealth.** Pair floors with an upper cap, or use two-sided targets.
7. **Wealth/earnings is about 4.9, since K/Y = 3 forces it.** The zero-wealth share of 31% (vs about 8% in the data) is the same symptom. Add a K/Y ≈ 4.5 robustness run.
8. **Calibration numerics are OK.** The convergence error is below the finite-difference steps, the GE solve is fine, and the 13% capital fall when trusts switch off is just trust wealth leaving.

**Agreed plan (pending Rich and the economist):**

- The C5 runs on v2 continue as a dry run of the pipeline.
- Then recalibrate ("v2b") with:
  - two-sided trust targets at the CRS centre;
  - the 75+ standard error loosened to about 0.15, pre-registered here before the run;
  - the v3 estate and charity target corrections;
  - nd = 55.
- **Main table:** a trust-share sweep (s ≈ 10/15/27/40%), each with β refit, in PE and GE.
- **Bounds:** the total-wealth measurement concept, and K/Y ≈ 4.5.
- **Verification:** at nd = 55/70.
- **Appendix:** tier 2, GRATs, valuation discounts and the dynastic type move results by 0.3 pp or less.
- **C5 results on v2 (15 Oct). Test run of the pipeline; numbers not for the paper.** GE unless noted. Summary: `results/layerC5_v2_summary.csv`.

  | Experiment | r* | Top 1% | Top 0.1% | Trust share of top-1% wealth |
  | --- | --- | --- | --- | --- |
  | Base (shield + lock-in) | 6.01% | 38.5% | 22.5% | 27.1% |
  | Shield only (lock = 0) | 6.33% | 34.7% | 18.0% | 4.6% |
  | Lock-in only (no shield) | 6.29% | 33.5% | 17.0% | 8.6% |
  | No trusts | 6.45% | 33.7% | 17.0% | 0 |

  - **Decomposition (top 1%):** total +4.85 pp = shield alone +1.03 + lock-in alone −0.14 + interaction +3.96. Top 0.1%: +5.48 = +0.99 + 0.01 + 4.48.
  - **Reading:** the effect comes almost entirely from the combination of tax exemption and binding commitment. Lock-in without the shield creates trusts (8.6% of top-1% wealth) that substitute for personal wealth and add no concentration.
  - The PE effects are within 0.2 pp of GE. The parent–child elasticity is roughly unchanged across experiments (0.40–0.41 rank-rank).
  - **Note:** in the base economy, capital supply is very elastic to r (excess +30% at r = 6.5%), because trust accumulation responds strongly to returns. Worth a sentence in the paper and a check in the K/Y robustness run.

## C4 v2b: pre-registered 15 Oct, before any v2b run
Recalibration following the independent review. **Model:** the same as v2 (ages 20–89, no medical costs; the v3 life and medical changes stay a robustness option, since they did not change the fit). **Free parameters, starting point and finite-difference steps:** as v2, starting from the v2 best point.

**Target changes vs v2:**
- 75+ wealth ratio: se 0.05 → **0.15**. It is a structural limitation, and it was pushing trust wealth up.
- Estate tax ÷ (gross − spousal bequests), taxable $50M+: **0.238** (was 0.192, mislabelled).
- Charity ÷ (gross − spousal bequests), taxable returns: **0.181** (was 0.15).
- Trust shares unchanged and **two-sided:** top 1% 0.15 ± 0.05, top 0.1% 0.40 ± 0.07. The v3 one-sided floors are dropped, per the review.
- Trust grid **nd = 55**, implementing the 7 Oct decision.

**Launcher:** `run_calibration_v2b.bat`. **Outputs:** `results/layerC4_v2b_*`.

**Next:**
- Rerun C5 (base / off / lock0 / noshield) on v2b.
- Trust-share sweep (s ≈ 10/15/27/40%).
- Bounds: total-wealth measurement concept; K/Y ≈ 4.5.
- Full-resolution check.
- **v2b progress and grid finding (16 Oct).**
  - **Grid effect:** at the v2 best parameters, moving the trust grid from nd = 40 to nd = 55 raises the trust share of top-1% wealth from 27% to 36% (top 0.1%: 38% → 50%), the total top-1% share from 38.4% to 41.3%, and the K gap from 0 to +7%. nd = 40 understated trust accumulation; this confirms the review's grid concern. An nd = 70 check of the final point is required.
  - **After 19 evaluations (two steps):** ss 96 → 16.4.
    - Trust share of top-1% wealth 20%, of top-0.1% 29%.
    - Personal top 1%/0.1% shares 32.6/15.8%.
    - K gap +0.2%; lifetime gifts 21% of transfers; 75+ ratio 1.18.
    - Total top-1% share 36.2%.
  - **Open tension:** trust wealth may be spread too evenly across the rich. Top 1% is on target while top 0.1% is low (29% vs 40%). Raise with the economist.
- **Compute:** a cloud server or the economist's university cluster was discussed (about $40–60 for the remaining work, using parallel evaluations). Rich's decision: finish the v2b run on his PC first, then decide.
- **C4 v2b result (17 Oct): stopped by choice at evaluation 53 at a plateau; best point (evaluation 51) taken.** Saved as `results/layerC4_v2b_params.json`.
  - **Path:** ss 96 → 28.7 → 16.4 → 14.3 → 14.2 → 13.6 → 13.56.
  - **Parameters:** beta 0.9454, phi1 −1.874, chi 0.205, fric_mu 0.0799, fric_nu 0.0320, trust_weight 0.938, gift_phi −1.927.
  - **Fit:** every target within about 1.5 SE except the down-weighted 75+ ratio (1.13 vs 1.53; 6.9 of ss 13.6).
    - K gap +0.2%;
    - personal top 1%/0.1%/10% shares 32.0/15.1/71.1% (targets 32.5/15.0/68.9);
    - trust share of top-1% wealth 21.3% (target 15 ± 5), of top-0.1% 32.5% (40 ± 7);
    - estate tax rate on $50M+ estates 26.9% (23.8%); charity 19.5% (18.1%); taxable deaths 0.14% (0.14%);
    - lifetime gifts 24.7% of transfers (27%).
  - **Untargeted:** total top-1% share 36.3%; parent–child wealth elasticity 0.28; trust take-up in the top 1% 17%.
  - **Next:** rerun the C5 experiments on v2b (needs `run_c5` pointed at v2b, nd = 55), the nd = 70 check, then the trust-share sweep and the bounds.
- **C5 results on v2b (18 Oct). Main results, fast configuration, nd = 55.** GE, r re-solved. Summary: `results/layerC5_v2b_summary.csv`.

  | Experiment | r* | Top 1% | Top 0.1% | Top 10% | Trust share of top-1% wealth |
  | --- | --- | --- | --- | --- | --- |
  | Base (shield + lock-in) | 6.00% | 36.3% | 19.6% | 72.9% | 21.3% |
  | Shield only (lock = 0) | 6.22% | 33.3% | 16.4% | 71.4% | 3.2% |
  | Lock-in only (no shield) | 6.20% | 32.7% | 15.8% | 71.0% | 4.7% |
  | No trusts | 6.28% | 32.7% | 15.8% | 71.0% | 0 |

  - **Decomposition (top 1%):** total +3.67 pp = shield +0.64 + lock-in +0.00 + interaction +3.02 (82%).
    - Top 0.1%: +3.86 = +0.62 − 0.01 + 3.24.
    - Gini: +1.1 points, of which 0.87 is the interaction.
    - PE effects are within 0.1 pp of GE.
  - **Capital:** trust wealth is mostly additional saving; at r = 6%, capital is 8.3% lower without trusts.
  - **Persistence:** the parent–child wealth elasticity is slightly lower with trusts (0.28 vs 0.30), and rank-rank is unchanged (0.44). Trusts concentrate the top tail rather than raising broad persistence.
  - **Consistency check:** the reviewer's rule (+0.17 pp per point of trust share) predicted about +3.5 pp at s = 21%; the result is +3.67.
  - **Comparison with v2:** the effect fell from +4.85 to +3.67 pp. The pattern (the combination does almost everything) is unchanged.
  - **Still to do before these are final:** the nd = 70 check, the trust-share sweep, the measurement-concept and K/Y bounds, and the full-resolution check.

## Follow-up independent review, 18 Oct (Opus; the Fable monthly limit was reached)
**Verdict:** v2b responds properly to the first review. The qualitative finding is robust: the combination of exemption and lock-in accounts for 82% of the effect in both v2 and v2b. The +3.67 pp point estimate is not yet paper-ready.

**Status of the 14 Oct findings:**
- **Resolved:** the 75+ target (in design), one-sided floors, numerics.
- **Partly resolved:**
  - trust share s (21.3%, still about 1.3 SE above the CRS centre); the effect still scales 0.172 pp per point of s;
  - grid (nd = 55 is used, but the 40 → 55 jump means nd = 70 must be checked).
- **Open:** the measurement-concept bound, partial withdrawal in lock0, wealth/earnings.

**New findings (checked by me where noted):**
1. **The two trust targets can't both be met.** Checked: across all 53 v2b evaluations, s0.1/s1 lies in 1.32–1.54, while the targets imply 2.67. So s1 = 21% is a compromise set by the SEs. The top-1% effect is likely overstated against the CRS centre (about +2.6 pp at s = 15%) and the top-0.1% effect understated. In the model trust wealth is not concentrated enough at the very top: take-up is 30% in the top 0.1% vs 17% in the top 1%. The founding friction or setup cost probably needs to scale more steeply with wealth, or the top-0.1% effect should be reported as a lower bound.
2. **The persistence claim:** in PE the parent–child elasticity is unchanged (0.284 vs 0.283). The GE drop comes from r. Wording: "trusts leave persistence unchanged at fixed prices."
3. **Describe the decomposition as removing one component at a time.** Remove the exemption and the effect is gone; remove lock-in and 0.64 pp remains. Lock-in alone is pure relabelling. The 82% is an upper bound until lock0 allows partial withdrawal; at ℓ = 0.5 the earlier C3 effect was +0.7 vs +3.4.
4. **The reviewer's "single heir per dynasty" concern is not correct (checked in code).** With probability q2 = 0.59 a child has a sibling (n_pop = 1%, 1.42 children per parent), and inherited trust assets are split (sibling-indexed transfer tables in `_conv_trust_children`). Dilution is modelled.

**Priorities before the paper:**
1. Trust-share sweep (s1 ≈ 10/15/21/27%), plus a variant with trust wealth more concentrated at the top (s0.1 ≈ 40% with s1 ≈ 15%).
2. nd = 70 and full-resolution runs of base and off.
3. Total-wealth measurement bound.
4. Partial withdrawal in lock0, or disclose that the 82% is an upper bound.
5. K/Y ≈ 4.5 bound.
6. Wording fixes.

**Drop or move to the appendix:** v3 life and medical costs, dynastic types, tier 2, GRATs and discounts, separate PE tables.

**Draft abstract:** in the reviewer's report (18 Oct); numbers preliminary.
- **Decision (18 Oct): no structural change for the "two trust targets" issue.** In SCF 2022 (our `data/scf2022_trusts_by_wealth.csv`), the top-0.1% to top-1% gradient is 1.6 for trust assets ÷ net worth (4.4% vs 2.8%) and 1.7 for take-up (21.4% vs 12.5%). The model gives 1.5 and 1.8. The 2.7 implied by our targets comes from combining two estimates not measured on a common basis: a CRS/CBO reading for the top 1% and Galle et al. for the top 0.1%. The model's 32.5% top-0.1% share is at Galle's "one-third or more" lower bound. So:
  1. Keep v2b as the main calibration.
  2. Address uncertainty about the level of trust wealth with a **trust-share sweep**: s1 ≈ 10/15/(21 = v2b)/27%. Each point varies fric_mu and refits beta so the capital market clears (other parameters at v2b), then runs base vs off.
  3. Add one **"steeper at the top" robustness variant**: a founding friction that falls with wealth, fmu(W) = fric_mu·(W/$30M)^(−fric_slope), with fric_slope = 0.5 and s1 held at 21%. It shows the top-0.1% result if the SCF understates the gradient.
  4. **Paper caveat:** the SCF may miss more trust wealth at the very top.
- **Trust-share sweep results (19 Oct).** v2b, nd = 55, GE; base is at r = 6% by construction, off has r re-solved. Summary: `results/sweep_summary.csv`.

  | Trust share of top-1% wealth | Top-0.1% trust share | Effect on top 1% | Effect on top 0.1% | r without trusts |
  | --- | --- | --- | --- | --- |
  | 10.1% | 15.9% | +1.39 pp | +1.34 pp | 6.11% |
  | 15.1% | 23.6% | +2.32 pp | +2.33 pp | 6.18% |
  | 21.3% (v2b) | 32.5% | +3.67 pp | +3.86 pp | 6.28% |
  | 27.0% | 40.3% | +5.17 pp | +5.61 pp | 6.40% |

  - The effect rises slightly faster than linearly in s: about 0.18 pp per point at the low end and 0.26 at the high end.
  - Personal top-1% shares stay within 31.8–32.3% across the sweep, close to the 32.5% target.
  - At s = 27% the top-0.1% trust share is 40%, matching Galle et al.'s central figure.
  - **The "steep" variant (fric_slope = 0.5) failed and is discarded.** With the hurdle scaling as (W/$30M)^−0.5, the very rich face almost no friction. The trust share jumped to 56–80% of top-1% wealth, and raising fric_mu (up to 0.28) could not bring it back to 21% because the hurdle is tiny at the top. The solver ended at an invalid point (s1 = 56%). If this robustness check is wanted, redesign it: a smaller slope (about 0.1–0.2), or calibrate fric_mu and fric_slope jointly to the two trust shares. Low priority, since the SCF gradient matches the base model.
- **Grid check passed (19 Oct): nd = 70 vs nd = 55 at the v2b parameters, GE.**

  | | Trust share of top-1% wealth | Top-1% effect | Top-0.1% effect | Personal top 1% |
  | --- | --- | --- | --- | --- |
  | nd = 55 | 21.3% | +3.67 pp | +3.86 pp | 32.0% |
  | nd = 70 | 20.6% | +3.57 pp | +3.78 pp | 32.0% |

  - At nd = 70, base r* is 6.007% (excess −0.3% at r = 6%). The no-trusts economy is unaffected by the trust grid.
  - The 40 → 55 jump (about 9 points of trust share) has shrunk to 0.7 points for 55 → 70, so nd = 55 is adequately converged: the headline moves by 0.1 pp. Report nd = 55, with nd = 70 as a footnote.
- **Measurement-concept check (19 Oct): the model measured the way the Fed's DFA measures the data.**
  - **Facts:**
    - The Financial Accounts household sector explicitly includes personal trusts (Fed Z.1 table guide).
    - The DFA allocates FA totals to households by SCF holdings, asset class by asset class (Batty et al.).
    - The top 1% hold 50.9% of corporate equities and mutual fund shares but 32.5% of net worth (DFA Q2 2026, FRED WFRBST01122, WFRBST01134).
  - **Model-side measure (v2b base, nd = 55):** personal wealth, plus the share c of trust assets the SCF records (attributed to the holding household), plus the remaining trust assets spread by weights a^η, with η set so the top 1% of personal wealth receives share θ. The calibration of c: SCF trusts are 2.3% of net worth against 9.2% in the model, so c ≈ 0.25.
  - **Results (DFA-measured top 1% / top 0.1%):**
    - personal 32.0% / 15.1%; total 36.3% / 19.6%;
    - proportional spreading (θ = 0.325): 31.9–33.9%, depending on c;
    - **central (θ = 0.509, c = 0.25): 34.1% / 17.1%**;
    - range across θ 0.45–0.55 and c 0–0.5: 33.0–35.0%.
  - **Reading:** measured the DFA way, the v2b model overshoots the data's 32.5% by about 1.6 pp (1.6 SE). The pre-set threshold was 0.5 pp, so the gap is material. The "total wealth" bound (36.3%) is the wrong extreme because the DFA cannot attribute unreported trusts to their beneficiaries. The right final approach is to **target the DFA-consistent measure** (θ = 0.509, c = 0.25) in place of personal top shares. Pending Rich's decision.

## C4 v2c, the final calibration: pre-registered 20 Oct, before any v2c run
- **Change from v2b:** the three top-share targets (top 1%/0.1%/10%: 32.5/15.0/68.9%, DFA Q2 2026) are now matched on a **DFA-consistent** measure of model wealth (`layerC.dfa_top_shares`) instead of personal wealth. The measure is:
  - personal wealth;
  - plus SCF-recorded trusts, attributed to the holding household (share c = min(1, 0.0231/τ), so recorded trusts equal 2.31% of wealth as in SCF 2022);
  - plus the remaining trust wealth spread by weights a^η, with η set so the top 1% of personal wealth receives 50.9% (the DFA top-1% share of corporate equities and mutual funds, Q2 2026).
- **Check:** at the v2b point the function reproduces the offline result (top 1% 34.10%, top 0.1% 17.13%), and it returns personal shares when there are no trusts.
- **Everything else as v2b:** targets, standard errors (75+ se 0.15), seven free parameters, nd = 55, steps; start at the v2b best point (`start=V2B`).
- **Then:** rerun C5 (base/off/noshield/lock0) and the trust-share sweep on v2c; K/Y robustness as one run.
- **Launcher:** `run_calibration_v2c.bat`. **Outputs:** `results/layerC4_v2c_*`.
- **C4 v2c, the final calibration (21 Oct): stopped by choice at evaluation 34 at a plateau; best point (evaluation 33) taken.** Saved as `results/layerC4_v2c_params.json`.
  - **Path:** ss 23.2 (the v2b point under the v2c targets) → 18.6 → 18.4 → 17.2 → 17.04.
  - **Parameters:** beta 0.9470, phi1 −1.549, chi 0.149, fric_mu 0.0750, fric_nu 0.0265, trust_weight 1.382, gift_phi −1.793.
  - **Fit:**
    - DFA-measured top 1%/0.1%/10%: 32.6/16.0/71.0% (targets 32.5/15.0/68.9);
    - trust share of top-1% wealth 20.6% (15 ± 5), of top-0.1% 28.4% (40 ± 7);
    - K gap +0.3%; estate tax rate on $50M+ estates 26.7% (23.8%); taxable deaths 0.13% (0.14%); charity 19.6% (18.1%);
    - lifetime gifts 24.0% of transfers (27%);
    - 75+ ratio 1.11 (1.53; 7.8 of ss 17.0).
  - **Untargeted:**
    - personal top 1%/0.1%: 30.7/14.3%; total top 1%/0.1%: 34.2/17.4%;
    - parent–child elasticity 0.28, rank-rank 0.43;
    - trust take-up in the top 1% 22%.
  - **Note for the economist:** trust_weight θ = 1.38 > 1. Founders value a dollar in trust above a dollar bequeathed directly, which can be read as the control or asset-protection value of trusts beyond the tax saving. The fit traded personal concentration down (personal top 1% 32.0 → 30.7%) rather than trust wealth.
  - **Next, one launcher (`run_final_v2c.bat`):**
    1. C5 on v2c (base, off, noshield, lock0; `run_c5.py cal=v2c`);
    2. trust-share sweep on v2c (s15, s10, s27; `run_sweep.py cal=v2c`).
- **Planned extension, after the final runs and the economist's review (21 Oct): dynasty trusts vs Trump accounts.** Both combine lock-in with a tax advantage, but Trump accounts are universal and capped.
  - **Rules (mid-2026):** $1,000 federal seed for births 2025–28; $5,000 per year per child from all contributors combined (employers up to $2,500); "qualified general contributions" by governments and 501(c)(3)s to whole groups of children (Dell pledge: $6.25B, $250 for about 25M children); invested in US equity index funds and locked until 18, then traditional-IRA rules. Sources: Jones Walker (June 2026), NBC (Dec 2025), ABC (Oct 9, 2026).
  - **Scenarios:**
    1. A universal seed at $1k/$5k/$25k/$50k, tax-financed, with r re-solved. The balance is modelled as an endowment at model entry (age 20), which simplifies the lock-in after 18.
    2. Capped private contributions: do richer families fill the accounts?
    3. Rich's idea: philanthropic general contributions by the very top as an alternative to funding dynasty trusts. Both escape the estate tax; one keeps wealth in the dynasty and the other spreads it. Model it as lifetime giving to a universal pool, at Dell scale, 10× Dell, and with a share of trust funding redirected.
  - **Caveat:** the model is weakest at the bottom and middle (31% zero wealth, no housing, no childhood), so bottom-half results would be indicative only.
  - **Philanthropic share gifts (facts as reported, 21 Oct):**
    - **Shotwell (SpaceX president):** plan to give one SpaceX share per child to more than 2M children via Trump accounts, about $320M, prioritising lower-income Texas families (Semafor, July 6, 2026). Treasury has said stock donations may go into the accounts; details are unclear, and the law requires index-tracking investments.
    - **Musk:** pledged one SpaceX share to every child in America in 2027, "the first of many". That is about 72M children × $146 ≈ $10.5B, about 1.1% of his roughly 6.4B SpaceX shares (Newsweek, Oct 19, 2026). The vehicle (Trump accounts or not), eligibility and tax structure are unconfirmed (ROIC, Oct 19, 2026).
    - **Anthropic:** committed 5% of its shares to a new Anthropic Public Benefit Fund, an entity within the company, with more if valuation milestones are met; about 20% of shares are dedicated to public-good uses in total. Uses may include direct distributions to specific populations, e.g. people affected by AI labour-market impacts. Cofounders pledged 80% of their wealth (Fortune, Oct 18, 2026). Disclosure: Claude, which assisted this project, is made by Anthropic.
  - **Modelling note:** giving appreciated shares avoids capital-gains tax and keeps the assets out of the estate, the same tax advantages as dynasty trusts, but dissipates the wealth across millions of households instead of locking it in one dynasty. Scenario idea: the top 0.01% transfer a share p of wealth to universal child accounts (p = 1%, 10%, 50%); outputs are top shares, the bottom 50% and persistence, in GE. Benchmark: if all of Musk's SpaceX stake (~$935B at $146) were distributed, that would be about $13,000 per child.

## Final results on v2c (run 21–22 Oct, `run_final_v2c.bat`; all experiments converged, |excess| ≤ 0.04%)
- **C5 decomposition (GE, r re-solved; pp of total wealth; "measured" = DFA-consistent measure):**

  | Experiment | r* | Top 1% | Top 0.1% | Top 1% measured | Trust share of top-1% wealth |
  | --- | --- | --- | --- | --- | --- |
  | base (trusts) | 5.99% | 34.20% | 17.35% | 32.56% | 20.6% |
  | off (no trusts) | 6.23% | 31.80% | 15.26% | 31.80% | 0 |
  | noshield (lock only) | 6.15% | 31.64% | 15.05% | 31.51% | 8.3% |
  | lock0 (shield only) | 6.21% | 31.88% | 15.25% | 31.88% | 3.1% |

  - **Total effect:** +2.40 pp top 1% (+2.09 pp top 0.1%); measured the DFA way, +0.76 pp (+0.71 pp). **About two-thirds of the true effect is invisible in DFA-style statistics.**
  - **Decomposition (top 1%):** shield alone +0.08, lock alone −0.16, interaction +2.48. Essentially all of the effect comes from shield and lock-in together (v2b: interaction 82%).
  - **Rate effect:** trusts lower r* by 24 bp (more capital supplied).
- **Trust-share sweep on v2c (fric_mu and beta re-solved per point; K_gap ≤ 0.2%):**

  | s1 (trust share of top-1% wealth) | Top-1% effect | Top-0.1% effect | Top-1% measured | Top-0.1% measured |
  | --- | --- | --- | --- | --- |
  | 10.0% | +0.82 pp | +0.48 pp | +0.34 pp | +0.18 pp |
  | 15.3% | +1.38 pp | +0.98 pp | +0.35 pp | +0.27 pp |
  | 20.6% (v2c) | +2.40 pp | +2.09 pp | +0.76 pp | +0.71 pp |
  | 26.9% | +4.08 pp | +3.99 pp | +1.45 pp | +1.43 pp |

  - The effect is convex in the trust share: roughly 0.08 pp per point of s1 at the low end and 0.27 pp at the high end.
  - The measured effect stays at about a third of the true effect throughout.
- **v2c vs v2b:**
  - The headline falls from +3.67 to +2.40 pp.
  - **Cause:** v2c matches the DFA measure, which pushes personal concentration down (personal top 1% 32.0 → 30.7%), so trusts sit on a less concentrated base.
  - **Headline for the paper:** v2c, with the sweep as the range (about +0.8 to +4.1 pp). v2b is reported as the personal-wealth-target alternative.
- **Files:** `results/layerC5_v2c_*.json`, `results/layerC5_sweep_v2c_*`; summary `results/layerC5_v2c_summary.csv`.

## θ check, pre-registered 25 Oct 2026 (before any run)
- **Why:** the final calibration needs a founder's trust weight θ = 1.38 (a trust dollar valued above a dollar bequeathed directly). Two readings: real non-tax motives (control, creditor protection), or a stand-in for the planning tools switched off in tier 1. Rich's point, which the paper will make: GRATs and sales to grantor trusts let families move value into trusts beyond the exemption almost tax-free, while the model charges 40% gift tax above it.
- **Runs (`run_sweep.py cal=v2c theta1_t2 theta1`, launcher `run_theta_check.bat`):** both at θ = 1, with the founding-friction mean and β re-solved so that the trust share of top-1% wealth equals the v2c value (20.58%) and the capital market clears at 6%. The friction mean may go down to −0.05 (the calibration's lower bound). All other parameters at v2c. Then base vs no trusts with r re-solved, as in the sweep.
  1. **theta1:** tier 2 off.
  2. **theta1_t2:** tier 2 on, **restricted to the trust-feeding channel**: freeze transfers (half of liquid wealth, return above a 5% hurdle passes to the family trust gift-tax free; read as sales to grantor trusts, since GST exemption cannot be allocated to a GRAT during its term), with
     - no tax-drag advantage (τ_a2 = τ_a = 18%, τ_ng2 = τ_ng = 24%);
     - no valuation discount (disc_v = 0), because discounts are also used without trusts and would bundle a non-trust channel into the trust effect;
     - legal and administration costs instead of a family office: setup $100k, $25k a year, no proportional fee (assumptions; the C2 family-office costs of $500k and $1M a year are kept for a later full tier-2 run).
     The upgrade friction stays at its default (mean 0.05, scale 0.03).
- **Smoke test (cloud, coarse grids, 3 generations, friction mean 0):** both configurations run; trust share about 22% (theta1) and 23% (theta1_t2); tier-2 take-up 4% of the top 1% and 9% of the top 0.1%; the no-trust economy is unaffected by tier 2.
- **Prediction (stated before the runs):**
  - At the same trust share, the top-1% effect stays within ±0.5 pp of v2c's +2.40 pp in both runs, because the sweep shows the effect is driven by the trust share.
  - Tier 2 shifts trust wealth toward the top 0.1% (its users are the richest families), raising the top-0.1% trust share and the top-0.1% effect a little. Tier-2 take-up stays modest.
  - At θ = 1, hitting the trust share needs a much lower founding friction (mean near zero), so trust take-up below the top 1% may rise. Take-up will be reported.
- **How results will be read:**
  - Effect within ±0.5 pp: θ is a footnote; the paper reports θ = 1 as robustness.
  - Effect outside ±0.5 pp: report it as a finding (the founder's motive matters, not only the trust share).
  - Trust share not reachable with the friction mean at −0.05: report that published trust shares require a non-tax motive or planning tools beyond tier 1, with the effect at the share reached.
  - theta1_t2 vs theta1 measures what the trust-feeding tools add at a fixed trust share (effect, top-0.1% trust share, take-up, freeze flows).
- **Paper change made at the same time:** a functional definition of "dynasty trust" in §2.1 (any vehicle that combines a transfer-tax shield across generations with restricted access to principal), with the instruments it covers and excludes.

## Trust-return sensitivity, pre-registered 25 Oct 2026 (before any run)
- **Why:** the original FODT paper's core mechanism was a return advantage for trust and family-office assets. We left it out of the baseline: it builds the result in, and the evidence cannot pin it down (family-office returns are self-reported; Fagereng et al. 2020 and Smith, Zidar & Zwick attribute most of the return gap of the rich to risk, scale and owners' human capital). Rich's case for it: a "set it and forget it" horizon, keeping concentrated founder positions, and access to higher-alpha opportunities through professional management. So it is reported as a sensitivity, not an assumption.
- **Code (`layerC.py`):** new parameter `trust_prem` (default 0): trust assets earn the family's pre-tax return times (1 + trust_prem)^5 per period. It applies to trust growth, the heir's trust tax and the founder's grantor tax. Freeze transfers still use the family's own return. **Nesting test:** with trust_prem = 0 the new code reproduces the previous code exactly (coarse grid, all 60 moments, maximum difference 0.0). With trust_prem = 0.01 it runs.
- **Runs (`run_sweep.py cal=v2c ret05 ret10`, launcher `run_trust_return.bat`):** premium 0.5% and 1.0% a year. The founding-friction mean and β are re-solved so the trust share of top-1% wealth stays at the v2c value (20.58%) and the market clears at 6%. Then base vs no trusts, with r re-solved.
- **A second reading comes free:** each point's first evaluation is at the v2c parameters with the premium added, before any re-solving, at r = 6%. Compared with the v2c no-trust economy at r = 6% (top 1% 31.83%), it gives the effect when nothing else adjusts and trust wealth is allowed to grow.
- **Prediction (stated before the runs):**
  - At a fixed trust share, the effect rises only modestly, by less than 1 pp at 1%: the trust share still drives the effect, and a premium mainly changes which families found trusts.
  - With parameters fixed, the premium raises the trust share and the effect more, roughly in line with the sweep (about 0.27 pp of effect per point of trust share at the high end).
- **How results will be read:** the paper reports the premium as a sensitivity in §4.5, alongside the θ check. Whichever way it comes out, no change to the headline.
- **Order:** start after `run_theta_check.bat` finishes (both write to `results/sweep_v2c_log.txt`, and running both at once would double run times).

## Paper: literature additions (25 Oct)
- From the original FODT paper's literature review, added to §2.3: the standard wealth-inequality lineage (Aiyagari 1994; Huggett 1996; Castañeda, Díaz-Giménez & Ríos-Rull 2003; Quadrini 2000; Cagetti & De Nardi 2009; De Nardi & Fella 2017), return heterogeneity (Benhabib, Bisin & Zhu 2011; Gabaix et al. 2016) and altruistic vs warm-glow bequests (Becker & Tomes 1979; Barro 1974).
- Added to §5.2 (θ): the family-office literature on governance and control (Glucksberg & Burrows 2016; Rivo-López et al. 2017, 2021; Hayoz, Ge & De Massis 2025). Kaplan & Rauh (2013) is cited for fortune creation.
- Dropped from the original review as off-question: Heer, Laitner, Bossmann et al., Adermon et al., Degan & Thibault.

## Publication copy of the code, and code review findings (25 Oct 2026)
- **Publication copy:** `publish/fodt-dynasty-trusts/` holds the README, requirements, known issues, `code/` (9 files), `data/`, `results/` (final v2c and v2b runs, sweep, grid check) and the launchers.
  - **Comments:** every code file was heavily commented for readers who know the economics but not the numerics, roughly doubling the line count, with search tags PAPER / PYTHON NOTE / NUMERICS / APPROXIMATION / UNITS.
  - **Same program:** each commented file's parsed program structure (ignoring comments and docstrings) was checked against the file that produced the results; all are identical. A coarse test run of the commented copy reproduces the original's output to every printed digit.
  - **Working code left alone:** the code in `code/` that runs on the PC was not touched.
- **Review findings** (details in `publish/fodt-dynasty-trusts/KNOWN_ISSUES.md`):
  1. **Lifetime gifts never use up the estate-tax exemption** (verified). At the age-55 gift stage, a gift above the $300k allowance is taxed only beyond the full exemption. The exemption is not reduced afterwards, either for trust funding in the same period or at death. This makes large gifts too cheap in both the trust and no-trust economies; the direction of the effect on the headline is unclear. **Proposed (to pre-register once Rich agrees):** a bound run in which gifts above the allowance get no exemption, base and no-trust re-solved, after the θ and trust-return checks.
  2. **No-shield experiment:** the count of taxable deaths leaves trust assets out of the taxable estate (the tax amount is correct). This affects one reported moment only.
  3. **Mobility moments** are conditional on the parent being alive; the paper should say so.
  4. **Reporting helpers only:** `choice_shares` labels, the `dfa_top_shares` top-group threshold on a lumpy grid, and `cap_diagnostics` approximations.
  5. **Housekeeping, no effect on results:** see the file.
- **Paper:** §3.2 corrected to say what the code does with lifetime gifts and the exemption.

## θ check results (run 25 Oct 12:40 – 23:28; read 26 Oct)
Both points hit the pre-registered trust share (20.6–20.7% of top-1% wealth) and cleared the market. Friction and β were re-solved; everything else stayed at v2c.

| | v2c (θ = 1.38) | θ = 1 | θ = 1 + freeze transfers (tier 2) |
| --- | --- | --- | --- |
| Friction mean μ_f / β | 0.0750 / 0.9470 | 0.0648 / 0.9467 | 0.0635 / 0.9467 |
| Trust share, top 1% / top 0.1% | 20.6% / 28.4% | 20.6% / 31.7% | 20.7% / 31.7% |
| Take-up, top 1% / top 0.1% / P90–99 | 22% / 33% / 7.4% | 15% / 29% / 4.5% | 15% / 29% / 4.5% |
| Top 1% with → without trusts | 34.20 → 31.80 | 35.45 → 31.85 | 35.43 → 31.84 |
| **Top-1% effect** | **+2.40 pp** | **+3.60 pp** | **+3.58 pp** |
| Top-0.1% effect | +2.09 | +3.65 | +3.60 |
| Top-1% effect, DFA-measured | +0.76 | +1.42 | +1.42 |
| Fit, sum of squares (11 targets, v2c weights) | 17.04 | 18.31 | 18.21 |

- **Reading, by the pre-registered rule:**
  - The effect moved by more than ±0.5 pp, so it is reported as a finding: **the founder's motive matters, not only the trust share.**
  - With θ = 1, fewer families found trusts (take-up in the top 1% 15% vs 22%, and in P90–99 4.5% vs 7.4%). The trusts that exist are larger and more concentrated at the very top: 31.7% of top-0.1% wealth vs 28.4%.
  - With θ > 1, part of trust demand is non-tax and spread among the merely rich, where trusts mostly relabel wealth. At a fixed top-1% trust share, a pure tax motive gives a 50% larger effect.
- **Fit:** θ = 1 fits almost as well (ss 18.3 vs 17.0). It is better on the top-0.1% trust share (Galle et al.), the share of deaths taxed and charity; worse on the DFA top shares (top 1% 33.3% vs 32.5% data).
  - **Caveat:** only μ_f and β were re-solved, so part of the larger effect comes from a more concentrated θ = 1 economy. A full recalibration with θ fixed at 1 would settle it (one more calibration, about 1.5 days).
- **Tier 2 (freeze transfers into the trust) adds nothing:** −0.02 pp, take-up 3% / 6% in the top 1% / 0.1%. It does not stand in for θ.
- **Predictions vs outcome:**
  - **Wrong:** "effect within ±0.5 pp at a fixed share".
  - **Wrong:** "friction falls to near zero", "take-up below the top 1% rises" and "tier 2 shifts trust wealth to the top 0.1%".
  - **Right:** "tier-2 take-up modest".
- **Implication for the paper:** v2c (+2.4 pp) is the best fit and the conservative end. The headline should be stated with this sensitivity: at the same trust share, a pure tax motive (θ = 1) gives +3.6 pp.
- **Next:** `run_trust_return.bat` (pre-registered 25 Oct) can start now.

## Gift-exemption diagnostic, step-up in basis and cloud computing (26 Oct)
- **Diagnostic** (coarse grid: nd 30, na 100; v2c parameters; r = 6%): what the "gifts never use the exemption" approximation gives away.
  - **Who:** about 0.6% of parents at 55 (0.61% with trusts, 0.64% without) make a large gift and later have estates big enough for the exemption to bind. They use on average $18–19M of exemption that is never deducted, of which about $14–15M would have reduced their exemption at death.
  - **How much:** about $5.7–5.8M of estate tax avoided per such family, about 9–10% of their post-gift wealth.
  - **Not a trust channel:** only 0.1% of these families also found a trust, so this is essentially a non-trust avoidance channel. It is nearly identical with and without trusts, so it inflates top concentration in both economies, with a smaller expected effect on the difference.
  - **Calibration context:** the model's effective estate tax on $50M+ estates is already above the IRS figure (26.7% vs 23.8%). Fixing the gift rule alone would widen that gap. The coherent fix is a package: gifts use the exemption, plus valuation discounts for business-owning families, calibrated to the estate-tax target.
- **Step-up in basis and deferral:** not in the model; capital income is taxed on accrual at 18%.
  - **Decision (Rich):** modelling basis is a follow-up project. It needs a further state variable, a realization choice, and rules for step-up, carryover and swaps, multiplying compute several times (a cluster or rented machine).
  - **This paper:** two conclusion paragraphs on basis as future work, plus a limitations bullet. They give a trust advantage per generation of 16–40 points on low-basis stock, note swap powers and carryover basis for gifts, and argue that the multi-generation structure limits how much basis matters.
- **Rented compute** (Google Cloud c3d-highcpu-180 at about $1.55/hr spot; AWS c7a.48xlarge at about $2.90/hr spot) is deferred to the follow-up. The speed-up comes from running evaluations in parallel, not from making one evaluation faster.
- **Queue for this paper:**
  1. the gift-exemption bound test (to set up);
  2. then, depending on its size, either the return test on the current model, or the avoidance package with a recalibration and reruns;
  3. optionally, a full θ = 1 recalibration.

## Gift-exemption bound, pre-registered 26 Oct 2026 (before any run)
- **Why:** in the model, lifetime gifts never use up the exemption (see the 25–26 Oct entries). This test applies the opposite extreme: gifts above the $300k allowance get **no** exemption (40% gift tax on every dollar above the allowance). The truth lies between the current model and this bound.
- **Code:**
  - `layerC.py`: new `ParamsC.gift_exempt` (default True). False sets the exemption available to the gift to zero (`_decide`, `Erem_g`).
  - **Nesting test (coarse grid):** with True, the new code reproduces the previous code exactly (all 60 moments, max difference 0.0). With False, it runs; the gift share of transfers falls from 26.8% to 21.2% on that grid.
  - `run_c5.py`: new experiments `base_giftnoex` and `off_giftnoex` (all v2c parameters fixed, r re-solved).
- **Launcher:** `run_gift_bound.bat` (`py run_c5.py cal=v2c base_giftnoex off_giftnoex`), about 3–4 hours. Outputs: `results/layerC5_v2c_{base,off}_giftnoex.json`; the log is `layerC5_v2c_log.txt`.
- **Prediction (stated before the runs):**
  - Top shares fall a little in both economies (gifts were a cheap non-trust avoidance channel for about 0.6% of families).
  - The trust effect (base − off) stays within ±0.5 pp of +2.40, because the diagnostic found the channel nearly identical with and without trusts.
  - The gift share of transfers falls well below the 27% target.
- **How results will be read:**
  - **Effect within ±0.5 pp of +2.40:** the approximation does not matter for the headline. Keep the current model, disclose the approximation with this bound in the paper, and run `run_trust_return.bat` next.
  - **Effect outside ±0.5 pp:** build the avoidance package (gifts use the exemption, plus valuation discounts for business-owning families), recalibrate to the same targets, and rerun the experiments, sweep, θ check and return test on it.
  - The measured (DFA) effect and the level changes will be reported either way.

## Gift-exemption bound: results (run 26 Oct 09:42 – 12:58)
Rule: gifts above the $300k allowance get no exemption. All v2c parameters fixed, r re-solved; both runs converged.

| | Current (gifts keep the exemption) | Bound (no exemption for gifts) |
| --- | --- | --- |
| Top 1% with → without trusts | 34.20 → 31.80 | 34.41 → 31.55 |
| **Top-1% effect** | **+2.41 pp** | **+2.86 pp** |
| Top-0.1% effect | +2.08 | +2.60 |
| Top-1% effect, DFA-measured | +0.76 | +0.90 |
| Trust share of top-1% / top-0.1% wealth | 20.6% / 28.3% | 22.5% / 30.8% |
| Lifetime-gift share of transfers (with / without trusts) | 24.0% / 27.3% | 18.8% / 22.2% |
| r with / without trusts | 5.99% / 6.23% | 6.07% / 6.34% |

- **Reading, by the pre-registered rule:**
  - The effect moved by +0.45 pp, which is inside ±0.5 pp (close to the edge). **Keep the current model**, disclose the approximation with this bound, and run `run_trust_return.bat` next.
  - The true rule lies between the two, so the effect lies between +2.41 and +2.86; the headline is again the conservative end.
- **Mechanism:** with gifts expensive, families substitute into trusts (trust share up about 2 points, roughly what the sweep's slope implies for +0.45 pp), and the no-trust economy loses an avoidance route (top 1% 31.80 → 31.55).
- **Predictions vs outcome:**
  - **Right:** "the effect stays within ±0.5 pp" (just) and "the gift share falls well below 27%".
  - **Wrong:** "top shares fall in both economies". The top 1% rose in the trust economy because trusts replaced gifts.

## Notes for later: possible policy experiments (26 Oct; paused by Rich)
Our existing experiments already map onto two reforms: "no shield" ≈ periodic taxation of trust assets (Canada deems trusts to dispose of their assets every 21 years), and "no lock-in" ≈ heirs free to draw principal. Candidates for a policy section or follow-up:
1. **Lower GST exemption for trusts** (e.g., $3.5M as in 2009, or $5M). Mostly a parameter change, plus a small code change to separate the trust-funding exemption from the estate exemption.
2. **Duration limit** (e.g., the 90-year limit on GST-exempt status proposed in past federal budgets). The model does not track a trust's age, so approximate it with a per-generation termination probability.
3. **Periodic (e.g., 21-year) deemed realization or tax on trust assets**, Canada-style. Close to the noshield run; could be done at a lower rate or partial shield.
4. **Minimum GRAT term / limits on valuation discounts** (needs tier 2 on).
5. **Trump accounts and philanthropic universal seeding** (already logged 21 Oct).
- Framing for non-academic audiences: "what share of the trust effect would reform X close, and would official statistics notice?"

## Trust-return sensitivity: results (run 26 Oct 15:05 – 19:52; read 27 Oct)
Both points hit the trust share (20.6–20.8% of top-1% wealth) with the market clearing; friction and β re-solved, everything else at v2c.

| | v2c (no premium) | +0.5% a year on trust assets | +1.0% a year |
| --- | --- | --- | --- |
| Friction mean μ_f | 0.0750 | 0.0793 | 0.0839 |
| Trust share, top 1% / top 0.1% | 20.6% / 28.3% | 20.6% / 28.4% | 20.8% / 28.4% |
| **Top-1% effect (GE, r re-solved)** | **+2.41 pp** | **+2.47 pp** | **+2.48 pp** |
| Top-0.1% effect | +2.08 | +2.15 | +2.15 |
| Top-1% effect, DFA-measured | +0.76 | +0.80 | +0.81 |
| *Unadjusted (v2c parameters + premium, r = 6%)*: trust share / top 1% | 20.6% / 34.20% | 22.2% / 34.63% | 24.3% / 35.12% |
| *Unadjusted effect* (vs the no-trust economy at r = 6%, top 1% 31.83%) | +2.37 | +2.80 | +3.29 |

- **Reading:**
  - At a fixed trust share, a return premium of up to 1% a year changes the effect by less than 0.1 pp. Holding everything else fixed, the premium works only by making trust wealth grow: +0.9 pp at 1%, in line with the sweep's slope (about 0.27 pp per point of trust share).
  - Because the calibration pins the trust share to outside estimates, a higher trust return is offset by a higher founding friction. The premium is not separately identified, and it does not change the conclusion.
  - Reported in §4.5 as a sensitivity.
- **Predictions:**
  - **Right:** "at a fixed share the effect rises by less than 1 pp" (it rose by less than 0.1 pp).
  - **Right:** "with parameters fixed, it rises roughly in line with the sweep" (predicted about 1.0, observed 0.9).
- **Status:** all pre-registered checks for this paper are done. **Numbers frozen** for the draft, economist package and SSRN posting:
  - headline +2.4 pp (DFA-measured +0.8);
  - trust-share range +0.8 to +4.1;
  - conservative relative to θ = 1 (+3.6) and the gift bound (+2.9);
  - robust to trust return premia (+2.5).

## Number audit of the paper draft (27 Oct)
An independent agent checked about 185 numbers in the paper (text, tables, figure data, appendix) against the result files; 172 were correct. **No headline number changed.** Corrections made, all from rounding already-rounded shares or from stale wording:
- **Decomposition, from unrounded shares:** shield alone +0.09 (was 0.08), lock-in alone −0.15 (was −0.16), interaction +2.47 (was 2.48), total +2.41.
- **Smaller figures:**
  - DFA-measured top-0.1% effect 0.70 (was 0.71);
  - shield-only trust share 3.0% (was 3.1%).
  - Several earlier entries in this log use the rounded versions; the result files are authoritative.
- **"Measured effect is about a third throughout"** is only true at the calibrated point; across the sweep it is between a quarter and two-fifths (s10 0.42, s15 0.25, v2c 0.32, s27 0.36). The text and the Figure 2 title were fixed.
- **Reporting details:**
  - grid check: trust share moved by 0.6 (not 0.7);
  - main-experiment convergence is "0.04% or less" (lock0 −0.0425%);
  - experiments wall time 6.6 h;
  - generations to converge with trusts 5–19 (13–17 for the calibrated economy).
- **Calibration plateau wording:** the "17.21, 17.16, 17.04" sequence from the params-file status string was not the order of evaluations. The paper now says the best value went from 17.16 (evaluation 21) to 17.04 (evaluation 33), and evaluation 34 was worse.
- **Old-age wealth:** "extending life to 94 moved the ratio by less than 0.02" was not supported. That configuration (entry at 25) gave 1.18 vs about 1.27, so the text now says it "did not raise it".
- **Figures:** Figure 1 and 2 data were updated to match (+2.41 at the calibrated point).

## Mock referee panel, round 1 (27 Oct)
- **Panel:** six independent AI reviewer agents (skill `econ-review-panel`) for JPubE: a macro modeler, public finance, measurement (Sonnet), trusts and estates (Sonnet), Reviewer 2, and an NTA discussant. They saw the paper, code and results, but not this log or the known-issues list.
- **Decision:** major revision (all five referees). Synthesis in the Claude Doc "Hidden Dynasties — mock referee panel, round 1".
- **Consensus issues:**
  1. The headline sits above the trust-share target and the dollar evidence (+1.4 at the 15% target); drop "conservative".
  2. The "invisible" share depends on the capture rate c and the spreading rule; it needs sensitivity at v2c.
  3. All-or-nothing withdrawal in lock0.
  4. Duration limits are mapped to the wrong experiment.
  5. No GST inclusion ratio for funding above the exemption.
  6. **Confirmed bug:** no-shield double-counts founders' trust assets. `_beq1` adds d to the taxable estate while `_bq_grid`, `_estates_fill` and `moments` also reduce the founder's exemption to max(E − d, 0). This affects only the noshield experiment.
  7. Steady state vs a 30-year-old trust stock.
  8. θ is weakly identified, and the v2b-vs-v2c gap is mostly θ, not the measurement change (θ = 1 at v2c targets gives +3.6, against v2b's +3.7). The introduction's attribution must be corrected.
- **Plan** (full list in the doc):
  - **A.** Writing before SSRN.
  - **B.** Four runs before SSRN, to be pre-registered separately: the noshield bug fix, a DFA sensitivity grid, partial withdrawal in lock0, an inclusion-ratio bound.
  - **C.** Before journal submission: full θ = 1 recalibration, a duration-limit experiment, a crude transition, numerical robustness, calibration diagnostics.
  - **D.** Follow-ups.

## Panel runs, pre-registration (27 Oct)
Four checks from the panel's B list, launched with `run_panel_fixes.bat`. The code switches were written and nesting-tested before these predictions. **Disclosure:** a coarse test (nd = 20, r fixed, 3 generations) was run to check that the switches work, and its directional results were seen before the predictions below were written. They are listed here so the predictions can be judged accordingly. Coarse top 1% shares: base 32.77, off 32.17, lock0 32.08, old noshield 32.14, fixed noshield 32.25, lock0 with partial withdrawal 32.14, funding capped at the exemption 32.46.

**Code changes (`layerC.py`; backup `layerC_pre_panel_backup.py`):**
- **noshield bug fix.** When `shield=False`, the founder's exemption is the full E, not max(E − d, 0), in `_estates_fill`, `_bq_grid` and `moments`. The `moments` taxable-estate test now includes d. Shield-on results are untouched.
- **`payout_fracs`** (default `()`): adds heir options to withdraw these shares of the drawable amount. This is partial withdrawal, used only in `lock0_partial`.
- **`fund_cap_E`** (default False): founding and contribution options whose funding exceeds the remaining exemption are dropped. This is an extreme bound for the GST inclusion ratio: no shielded funding above the exemption at all.
- **Nesting test:** with all defaults, the new code reproduces the old exactly (max abs diff 0.0 over all scalar moments). Script: `scratchpad/nest_panel.py`.

**Runs** (v2c, r re-solved; new names, so nothing existing is overwritten):
- `noshield_fix` = {shield: False}
- `lock0_partial` = {lock: 0, payout_fracs: (0.1, 0.25, 0.5)}
- `base_capE` = {fund_cap_E: True}, compared with the existing `off`
- `dfa_grid.py`: re-solves v2c base at r* = 5.9896% (to save the distribution), then recomputes DFA-measured top shares over c ∈ {0, …, 1} × θ_DFA ∈ {0.40, …, 0.80}

**Predictions** (top 1% shares in pp; v2c: base 34.20, off 31.80, noshield 31.64, lock0 31.88):

| Run | Prediction | Reading rule |
|---|---|---|
| noshield_fix | 31.75–32.10: higher than the old noshield, still within 0.5 of off | If noshield_fix − off < 1.2 (half the base effect), the paper's statement "the shield drives the effect" stands, with the corrected number. If it is ≥ 1.2, §4.2 is rewritten. |
| lock0_partial | Within ±0.3 of lock0 (31.58–32.18) | If lock0_partial − off < 1.2, "lock-in is necessary" survives partial withdrawal. Otherwise lock-in is downgraded to "matters under all-or-nothing withdrawal". |
| base_capE | Effect 0.8–1.9, i.e. 35–80% of +2.41; the trust share of top-1% wealth falls | Reported as a lower bound. If the effect is < 1.2, the abstract leads with the range from the capE bound to v2c rather than a single number. |
| DFA grid | At the main values (c ≈ 0.26, θ 0.509) it reproduces +0.76 within 0.02. Across c ∈ [0, 0.5] × θ ∈ [0.45, 0.65], the measured effect is 0.3–1.4, so the invisible share is 40–85%. | The paper reports the range and stops quoting one "invisible" percentage. If the grid point at the main values misses +0.76 by more than 0.05, stop and investigate before using the grid. |

- **Expected time:** about 7 hours (three GE experiments at roughly 2 h each, plus a 45–60 min solve for the grid).

## Take-up gradient check (27 Oct, untargeted; no new model runs)
- **Question:** does trust take-up across the top of the SCF wealth distribution separate θ = 1.38 (v2c) from θ = 1? This is step 1 of the plan to identify θ, before spending a calibration run on it.
- **Data:** SCF 2022 summary extract. "Has a trust" means TRUSTS ≥ $100k. TRUSTS covers trusts and managed investment accounts the household owns, and largely misses discretionary trusts held for beneficiaries. Groups are by net worth, over five implicates. Uncertainty is a household bootstrap (200 draws), which ignores the survey design. Script: `code/scf_takeup_check.py`.

| | Take-up ratio, top 0.1% vs 99–99.9 | Trust-asset share, top 0.1% vs top 1% |
|---|---|---|
| SCF 2022 | 2.11 (sd 0.76; 90% interval 1.13–3.34) | 1.58 (sd 0.20; 90% interval 1.19–1.86) |
| v2c (θ 1.38) | 1.55 | 1.38 |
| θ = 1 | 2.16 | 1.54 |
| v2b (θ 0.94) | 1.90 | 1.53 |
| s15 (θ 1.38) | 1.53 | 1.33 |
| s27 (θ 1.38) | 1.58 | 1.41 |

- **Reading:**
  1. **The ratios measure θ, not the amount of trust wealth.** At fixed θ they barely move from s15 to s27, while θ = 1 moves them a lot. So this is the right kind of moment for identifying θ.
  2. **The SCF is too noisy to decide.** Both economies sit inside the 90% intervals. The point estimates lean towards θ = 1 by about 0.8 sd (take-up) and 1 sd (assets).
  3. **Below the top 0.1% the SCF measure mostly captures other vehicles.** SCF take-up is flat from the 90–99th percentiles (10.2%) to 99–99.9 (10.2%), while the model's falls by two-thirds. Revocable trusts and managed accounts likely dominate there, so that comparison is not used.
- **Decision (proposed):** don't add the SCF ratio as a calibration target, because it would not pin down θ. Report the comparison in §5.2 as weak evidence leaning towards θ = 1, which points to the upper end of the +2.4 to +3.6 range. Keep the θ = 1 recalibration on the journal list as the second bounding case. Data that could identify θ: trust take-up by wealth from IRS fiduciary returns linked to beneficiaries.

## θ = 1 full recalibration, pre-registration (27 Oct)
- **Why:** θ moves the effect by about half and our targets identify it weakly (the 25 Oct θ = 1 check re-solved only β and the friction mean; the SCF take-up check of 27 Oct was too noisy to decide). A full recalibration with θ fixed at 1 gives a second complete economy, so the paper can report two calibrated cases rather than a calibration plus a sensitivity check. This is step 1 of the journal-version to-do (Claude Doc "Hidden Dynasties — journal-version to-do: θ and the transition").
- **Setup:**
  - Label `v2c_theta1`, the same 11 v2c targets and standard errors.
  - θ = `trust_weight` is fixed at 1.0; the other six parameters are estimated.
  - Start `V2C_TH1`: v2c values, with β and the friction mean taken from the θ = 1 check (0.94673, 0.06476).
  - Code: `calibrate_C.py` gains `FIXED_BY_SUFFIX` (labels ending `_theta1` hold θ at 1) and the `V2C_TH1` start. A dummy-objective test confirmed θ = 1 in every evaluation. `run_c5.py` reads the result as calibration `v2c_theta1` (tag `v2c_theta1_`). Backups: `calibrate_C_pre_theta1_backup.py`, `run_c5_pre_theta1_backup.py`. Existing calibrations are unchanged.
  - **Experiments after calibrating:** base and off, r re-solved.
  - **Stopping:** max 90 evaluations, but stop by hand once the best sum of squares has improved by less than 0.1 over six evaluations (as for v2c). Expect 15–25 evaluations, about 12–20 hours.
- **Predictions:**

| Quantity | Prediction |
|---|---|
| Sum of squares | 17.0–18.3 (the partial θ = 1 check gave 18.31; v2c 17.04) |
| Top-1% effect, general equilibrium | +3.2 to +3.8 (partial check: +3.60) |
| DFA-measured effect | +1.2 to +1.6 (partial check: +1.42) |
| Take-up in the top 1% | 13–17% (v2c 22%) |
| Trust share of top-1% wealth | 18–23% |

- **Reading rules:**
  - If the sum of squares is within 0.5 of v2c's 17.04, the targets do not distinguish θ, and the paper reports both economies with equal weight.
  - If it exceeds v2c by more than 1.0, the targets mildly prefer θ = 1.38: report θ = 1 as the alternative, with that statement.
  - Either way, the abstract quotes the calibrated range across the two θ values.

## Panel runs, results (28 Oct)
- **Run:** `run_panel_fixes.bat` finished 27 Oct 16:58. All experiments cleared the capital market within 0.04%.

| Run | Top 1% | Effect vs off | Prediction | Hit? | DFA top 1% | Trust share of top-1% wealth | Estate tax / Y | r* |
|---|---|---|---|---|---|---|---|---|
| off | 31.80 | — | | | 31.80 | — | 0.27% | 6.23% |
| noshield_fix | 31.83 | +0.03 | 31.75–32.10 | yes | 31.51 | 11.4% | 0.27% | 6.12% |
| lock0_partial | 32.04 | +0.24 | 31.58–32.18 | yes | 32.04 | 5.4% | 0.16% | 6.20% |
| base_capE | 32.93 | +1.13 | effect 0.8–1.9 | yes | 32.28 | 13.9% | 0.21% | 6.06% |
| base (v2c) | 34.20 | +2.41 | | | 32.56 | 20.6% | 0.15% | 5.99% |

- **noshield_fix:** the old run's −0.16 was mostly the bug. Lock-in alone now gives +0.03; trusts still form (11.4% of top-1% wealth, take-up 21%) but only relabel wealth. Reading rule: below 1.2 → "the shield drives the effect, with lock-in" stands, with corrected numbers.
- **lock0_partial:** partial withdrawals (10/25/50%) raise the shield-only effect from +0.09 to +0.24. Reading rule: below 1.2 → "lock-in is necessary" survives partial withdrawal.
- **base_capE:** the effect is +1.13 and the trust share falls to 13.9%. That is slightly below the trust-share sweep at the same share (about +1.2 interpolating between s10 and s15). So the cap works mainly by keeping wealth out of trusts. Reading rule triggered (below 1.2): the abstract leads with a range. It already leads with the sweep range (0.8–4.1); the capE point is reported as lying on that range, not as a separate lower bound on the effect.
- **DFA grid** (results/dfa_grid_v2c.json): the main point (c = 0.258, θ_DFA = 0.509) reproduces +0.76 exactly, so the gate passed.
  - At θ_DFA = 0.509 the measured effect is +0.51 (c = 0) to +1.20 (c = 0.5): 50–79% hidden.
  - At c = 0.258 it runs from +0.37 (θ_DFA 0.45) to +1.72 (0.65).
  - Over the full pre-registered box (c 0–0.5, θ_DFA 0.45–0.65): −0.02 to +1.85, i.e. 23% to 100% hidden.
  - **Prediction missed:** I predicted 0.3–1.4 (40–85% hidden); the range is wider at both ends. The spreading share θ_DFA matters much more than the capture rate c.
  - Decision: the paper reports the range at the evidence-based spreading share (0.5–1.2, half to four-fifths hidden) and states the sensitivity to the spreading share.
- **Paper:** brackets filled on 28 Oct (Table 2 gains the partial-withdrawal and funding-cap rows and an estate-tax column; §§4.2, 4.3, 4.5, 5.1, abstract and introduction updated). Figure 1 (waterfall) needs the corrected lock-in-only number.

## Handoff to a new thread (28 Oct)
- **State:**
  - Sections A and B of the mock-panel plan are done, and the paper (Claude Doc "Hidden Dynasties", rev ~146) has no result brackets left.
  - The remaining brackets are housekeeping before SSRN: South Dakota source, CRS and Sabelhaus titles, PC specs, repository link, AI-policy wording, a reference check.
  - Figure 1 has been updated (lock-in alone +0.03; interaction +2.29).
- **Headline numbers** (v2c, general equilibrium):

| Quantity | Value |
|---|---|
| Effect on top-1% share (calibrated, trust share 20.6%) | +2.41 |
| Effect at 15% trust share | +1.38 |
| Range across trust shares 10–27% | +0.82 to +4.08 |
| θ = 1 partial check | +3.60 |
| Funding capped at the exemption | +1.13 (trust share 13.9%) |
| DFA-measured, calibrated point | +0.76 (+0.51 to +1.20 over SCF capture 0–0.5) |
| Shield alone / lock-in alone | +0.09 (+0.24 partial) / +0.03 |

- **Code on the PC:** md5 checked against the cloud copies on 27–28 Oct.
  - layerC.py bc0a0d79…
  - run_c5.py 5474b547…
  - calibrate_C.py 2adcaab6…
  - dfa_grid.py fe095c05…
  - scf_takeup_check.py 5b178cff…
- **Next, in order** (Claude Doc "Hidden Dynasties — journal-version to-do: θ and the transition"):
  0. SSRN housekeeping, then post.
  1. Run `run_calibration_theta1.bat` (pre-registered above; not yet started).
  2. IRS fiduciary trust-size moment.
  3. Transition, version 1.
  4. Decide on version 2.
  5. Transition, version 2.
  6. Recalibration with any new target.
  7. Paper update and second-round panel.
  8. Human review.
- **Also open:**
  - Update the publication package (comments for `trust_prem`, `gift_exempt`, `payout_fracs`, `fund_cap_E`, the θ = 1 option; new results).
  - Appendix A run times for the panel runs.
  - A cover note for the economist; mark the old results package superseded.
  - Parked: basis, policy experiments, Trump accounts, cloud compute.
- **Opening message for the new thread:** "Continuing the Hidden Dynasties dynasty-trust paper. Read DECISIONS.md in my FODT Paper folder from 'Mock referee panel, round 1' to the end, then the to-do doc 'Hidden Dynasties — journal-version to-do: θ and the transition'. Standing practices: pre-register in DECISIONS.md before runs, compute on my PC via .bat launchers, commit to the FODT Paper folder with checksum checks, don't start PC jobs without my OK."

## Plan update: history and the transition's starting point (28 Oct)
- **Rich's point:** trusts that passed wealth with control across generations long predate 1986, so the transition should not start from zero trusts.
- **History** (to be checked against the statutes before coding):
  - Federal estate tax from 1916; gift tax in lasting form from 1932.
  - Before 1976, trusts limited only by the rule against perpetuities skipped estate tax at intermediate generations with no cap on size, while outright transfers faced top rates up to 77% with a $60,000 exemption.
  - 1976: first generation-skipping tax. 1986: replaced by the current one with a $1M exemption; trusts irrevocable on 25 Sep 1985 grandfathered.
  - From 1983 (South Dakota) and the 1990s: perpetuity rules repealed state by state.
  - In short, the law moved from trusts unlimited in size but limited in time to trusts limited in size but potentially unlimited in time.
- **Decisions:**
  1. The introduction gains a paragraph framing transfer-tax avoidance as long-standing (citing Cooper 1979, *A Voluntary Tax?*, Brookings), with today's novelty being a large exemption combined with no time limit, plus wider access.
  2. The to-do doc gains two steps before the transition: historical trust-stock data, and a duration hazard switch (each trust ends with a fixed probability per generation) used for the pre-1986 steady state and the §5.1 duration-limit experiment.
  3. Transition version 1 starts from the pre-1986 steady state, with grandfathered trusts running off. Version 2 adds the regime sequence, the funding-cap decision and a falling founding friction for wider access.
  4. Recalibration of both θ cases happens once, on the final model (to-do step 8).
  5. A century model starting around 1900 is parked as a follow-up paper. Starting earlier would wash out the initial condition, but it would not make 2026 a steady state, because the perpetual, exemption-capped regime is recent.
  6. "Accelerating" is used as motivation only, with sourced evidence. A rising trust stock could be convergence or falling frictions, and the historical series is needed to tell them apart.
- **θ = 1 recalibration:** unaffected; it compares θ on today's model.

## θ = 1 recalibration, results (29 Oct)
- **Run:** 28 evaluations from 28 Oct 08:56 to 29 Oct about 05:30.
- **Stopped by the pre-registered rule:** the best sum of squares went from 16.635 (evaluation 15) to 16.620 (evaluation 24), less than 0.1 over more than six evaluations. The calibration does not write its parameter file when stopped by hand, so `results/layerC4_v2c_theta1_params.json` was written from the evaluation cache (best = evaluation 24).
- **Best point:** β 0.94778, φ1 −1.200, χ 0.126, friction mean 0.0511, friction scale 0.0230, θ 1 (fixed), φ_g −1.795.
- **Fit:** sum of squares **16.62**, against 17.04 for v2c (θ 1.38).
- **Moments vs v2c:**
  - top 1% DFA 32.6% (32.6%)
  - top 0.1% DFA 16.2% (16.0%)
  - trust share of top-1% wealth 19.2% (20.6%)
  - trust share of top-0.1% wealth 30.2% (28.4%)
  - wealth 75+/mean 1.08 (1.11)
- **Prediction:** sum of squares 17.0–18.3. **Missed**: the θ = 1 economy fits slightly better than v2c.
- **Reading rule:** within 0.5 of v2c (difference −0.42), so the targets do not distinguish θ, and the paper reports both economies with equal weight. Effect predictions (+3.2 to +3.8; DFA +1.2 to +1.6; top-1% take-up 13–17%) are pending the experiments.
- **Next:** `run_theta1_experiments.bat` (base, off).

## θ = 1 experiments, results (29 Oct)
- **Run:** `run_theta1_experiments.bat`, 08:43–09:46. Base cleared at r = 6.00% (excess +0.18%, within tolerance) and off at 6.23%.
- **Note:** the calibration window ran on for five more evaluations after the parameter file was written; evaluation 31 reached 16.602 (vs 16.620 at evaluation 24, below the 0.1 stop threshold). The experiments use evaluation 24, as written.

| | θ = 1 (base / off) | v2c, θ 1.38 (base / off) |
|---|---|---|
| Top 1% | 34.53 / 30.94 → **+3.59** | 34.20 / 31.80 → +2.41 |
| Top 0.1% | 18.33 / 14.68 → **+3.65** | 17.35 / 15.26 → +2.09 |
| Top 1%, DFA-measured | 32.57 / 30.94 → **+1.63** | 32.56 / 31.80 → +0.76 |
| Share of the effect hidden | 55% | 68% |
| Trust share of top-1% / top-0.1% wealth | 19.2% / 30.2% | 20.6% / 28.4% |
| Take-up, top 1% / top 0.1% | 12.6% / 26.4% | 22.0% / 32.3% |
| Estate tax / Y (base / off) | 0.15% / 0.24% | 0.15% / 0.27% |
| r* (base / off) | 6.00% / 6.23% | 5.99% / 6.23% |

- **Predictions:**
  - Effect +3.2 to +3.8: **hit** (+3.59).
  - DFA-measured +1.2 to +1.6: **just missed** (+1.63).
  - Take-up in the top 1%, 13–17%: **just missed** (12.6%).
  - Trust share 18–23%: **hit** (19.2%).
- **Reading:** the two calibrations fit about equally well (16.6 vs 17.0), so under the pre-registered rule the paper reports both with equal weight. At the calibrated trust share (about 20%), the effect on the top-1% share is +2.4 to +3.6 points, and the top-0.1% effect +2.1 to +3.7. With θ = 1, fewer families hold trusts but they are richer, so the same trust wealth concentrates more.
- **Next (proposed):**
  1. Paper update: abstract, introduction, §3.4 (second calibration table), §4, §5.2, conclusion.
  2. A seven-parameter restart from the θ = 1 best point (one night) to see where θ settles when free.
  3. Trust-share sweep at θ = 1 (1–2 days), for the journal version.

## Free-θ restart, pre-registration (29 Oct)
- **Why:** v2c estimated θ freely but stopped at θ = 1.38 with a sum of squares of 17.04, while θ = 1 (inside its bounds) reaches 16.60. So v2c did not find the best fit; it was stopped early or sat in a local dip. Before posting anywhere we want to know where θ settles when free, so the paper is written once.
- **Setup:**
  - Label `v2c_free2`: all seven parameters free, the same 11 v2c targets and standard errors.
  - Start `V2C_FREE2`: the best θ = 1 point (evaluation 31 of v2c_theta1, sum of squares 16.602).
  - **Automatic stop:** the run stops once the best sum of squares has improved by less than 0.1 over the last 8 evaluations (the rule applied by hand before). Max 90 evaluations.
  - Then base and off at the result.
  - Code: `calibrate_C.py` gains the `V2C_FREE2` start and the stop rule, active only for labels starting `v2c_free`; tested with a dummy objective, which stopped after 9 evaluations and wrote the parameter file. `run_c5.py` now reads any calibration's parameter file by name; tested. Existing labels behave exactly as before. Backups: `*_pre_free2_backup.py`.
  - **Expected time:** 9–25 evaluations at about 45 minutes, plus about 1.5 hours of experiments.
- **Predictions:**
  - θ ends between 0.9 and 1.3.
  - Sum of squares 16.3–16.6.
  - The effect on the top-1% share lies between the two existing cases (+2.4 to +3.6), close to the θ = 1 value if θ stays near 1.
- **Reading rules:**
  1. θ within 0.85–1.15: the paper's main economy becomes the free estimate (θ ≈ 1), with v2c (θ 1.38) reported as the alternative calibration.
  2. θ at or above 1.25: the objective is flat between 1 and about 1.4, and the two economies stay at equal weight.
  3. In between: the free estimate becomes the main economy, and the range across θ is reported.
  4. In every case the paper says θ is weakly identified. A full profile over θ, the definitive test, stays on the journal list.

## Free-θ restart, results (30 Oct)
- **Run:** 9 evaluations, 29 Oct 13:05 – 30 Oct about 03:00. Then base and off ran at the starting point.
- **What happened: the automatic stop fired after the optimiser's first trial step, so the run is inconclusive.**
  - Evaluation 1 repeated the start (16.602).
  - Evaluations 2–8 were the finite-difference Jacobian, one small change per parameter.
  - Evaluation 9 was the first trial step: θ 1.17, with the friction mean 0.051 → 0.028 and the friction scale 0.023 → 0.010. It fitted worse (17.31).
  - With 7 free parameters, a Jacobian takes 7 evaluations, so a window of 8 evaluations contains only one trial step. A rejected first step therefore triggers the stop, before the optimiser can shrink its step and try again.
  - **The rule was mis-specified for a 7-parameter run; my error.** The θ = 1 run used it by hand with 6 parameters over many more evaluations.
- **What we did learn:**
  - Holding the other parameters at the θ = 1 point, raising θ from 1.00 to 1.03 worsens the fit (16.60 → 16.66).
  - Together with the fully re-estimated points (θ = 1: 16.60; v2c, θ = 1.38: 17.04), the fit appears to worsen as θ rises above 1.
  - Values below 1 have not been tried.
- **Experiments at the starting point** (θ = 1, evaluation 31 of v2c_theta1): top 1% 34.48 vs 30.95 → +3.53; DFA-measured +1.60; trust share of top-1% wealth 19.0%; take-up in the top 1% 12.5%. This essentially reproduces the θ = 1 case (+3.59).
- **Reading:** the pre-registered rules cannot be applied as intended, because θ never got a real chance to move. Taken literally (θ ended at 1.0, within 0.85–1.15), rule 1 says θ ≈ 1 becomes the main case, but that would rest on a stop-rule artefact.
- **Proposed fix (decided after seeing this result, disclosed here):**
  - Stop only when the best fit has improved by less than 0.1 over the last 16 evaluations (two Jacobians plus steps).
  - Rerun as `v2c_free2`. The 9 cached evaluations replay instantly, and the predictions and reading rules are unchanged.
  - The alternative is to accept the local evidence and report θ ≈ 1 as the best-fitting value, with v2c as the alternative.

## Free-θ restart, rerun as v2c_free3 (30 Oct; Rich chose the rerun)
- **Change:**
  - `STOP_N` goes from 8 to 16 in `calibrate_C.py` (backup `calibrate_C_pre_free3_backup.py`).
  - New label `v2c_free3`, so the v2c_free2 results stay as they are.
  - The launcher copies v2c_free2's 9 cached evaluations to the new label, and they replay instantly.
  - Tested with the real cache: the 9 evaluations replay in the same order without stopping, and the first new evaluation is a shorter trial step (θ 1.04, friction mean 0.046).
- **Unchanged:** start point, targets, predictions and reading rules (pre-registration of 29 Oct).

## Free-θ rerun (v2c_free3), results (31 Oct)
- **Calibration:** 26 evaluations (9 replayed). It stopped automatically at 30 Oct ~04:30 by the 16-evaluation rule.
- **Best point (evaluation 10):** θ **1.041**, β 0.94774, φ1 −1.365, χ 0.140, friction mean 0.0462, friction scale 0.0187, φ_g −1.769.
- **Fit:** sum of squares **16.39** (θ = 1: 16.60; v2c, θ 1.38: 17.04).
- **Local search:** raising θ to 1.07 (16.55) and two trial steps at θ 1.04–1.06 were all worse; the search kept shrinking back to θ 1.04. θ below 1 was not probed directly.
- **Numerical noise:** evaluations 24–26 re-ran almost exactly the best point and gave 16.39–16.43, so fit differences below about 0.05 are noise from the solver's warm start.
- **Experiments (general equilibrium):** base r* 5.99%, off 6.21%.

| Quantity | Base | Off | Effect |
|---|---|---|---|
| Top 1% | 34.47 | 31.30 | **+3.17** |
| Top 0.1% | 18.05 | 14.94 | **+3.11** |
| Top 1%, DFA-measured | 32.58 | 31.30 | **+1.28** (60% hidden) |

  Other moments: trust share of top-1% wealth 18.9% and of top-0.1% wealth 29.1%; take-up 12.9% (top 1%) and 26.8% (top 0.1%); estate tax / Y 0.15% vs 0.25%; effective estate-tax rate on large estates 11% vs 26%.
- **Predictions (29 Oct):** θ 0.9–1.3 **hit**; sum of squares 16.3–16.6 **hit**; effect between the two existing cases (+2.4 to +3.6) **hit** (+3.17).
- **Reading rule 1 applies** (θ within 0.85–1.15): the free estimate becomes the main economy ("v2c_free3", θ ≈ 1.04), and v2c (θ 1.38) becomes the alternative calibration. The θ = 1 economy (+3.59) is a further check. The paper still says θ is weakly identified.

## Main-case reruns, pre-registration (31 Oct)
- **Why:** the main tables must come from the main economy. Rerun the decomposition, the funding cap and the DFA grid at v2c_free3. The trust-share sweep stays at v2c for the SSRN version, with a sentence saying so.
- **Runs:**
  - `run_c5.py cal=v2c_free3 lock0 lock0_partial noshield_fix base_capE`
  - `dfa_grid.py cal=v2c_free3`. The script is generalised to any calibration; the default still reproduces the v2c files. Backup `dfa_grid_pre_free3_backup.py`; smoke-tested.
  - About 8–9 hours.
- **Predictions** (effects vs off = 31.30; main effect +3.17):

| Run | Prediction |
|---|---|
| Shield only (lock0) | −0.1 to +0.4 |
| Shield only with partial withdrawals | 0.0 to +0.5 |
| Lock-in only (noshield_fix) | −0.2 to +0.3 |
| Funding cap at the exemption | +0.9 to +1.8, trust share 11–15% |
| DFA grid | reproduces +1.28 within 0.02 at the main point; at θ_DFA 0.509 and c from 0 to 0.5, measured +0.8 to +1.7 |

- **Reading rules:**
  - If shield alone or lock-in alone exceeds +1.6 (half the main effect), the mechanism statement is rewritten.
  - If the DFA grid misses +1.28 by more than 0.05, stop and investigate.

## Main-case reruns, results (1 Nov)
- **Run:** `run_main_reruns.bat`, 31 Oct 15:42 – 1 Nov ~03:00. All experiments cleared within 0.04%.
- **Results at v2c_free3** (θ 1.04; off = 31.30, main effect +3.17):

| Run | Top 1% | Effect | Prediction | Hit? | DFA top 1% | Trust share of top-1% wealth | Estate tax / Y |
|---|---|---|---|---|---|---|---|
| Shield only (lock0) | 31.68 | +0.38 | −0.1 to +0.4 | yes | 31.68 | 2.5% | 0.18% |
| Shield only, partial withdrawals | 31.83 | +0.53 | 0.0 to +0.5 | **no** (just above) | 31.83 | 4.2% | 0.17% |
| Lock-in only (noshield_fix) | 31.55 | +0.25 | −0.2 to +0.3 | yes | 31.54 | 5.3% | 0.27% |
| Funding cap at the exemption | 32.69 | +1.39 | +0.9 to +1.8 | yes | 32.14 | 10.7% | 0.21% |

  Funding cap: the trust share prediction of 11–15% was just missed (10.7%).
- **Decomposition:** shield alone +0.38, lock-in alone +0.25, interaction +2.54 (80% of the effect). Neither feature alone exceeds half the main effect, so the mechanism statement stands, with the new numbers. The single features matter a little more than at θ 1.38, where interaction was 95%.
- **DFA grid:** the main point (c = 0.319, θ_DFA 0.509) reproduces +1.28 exactly, so the gate passed.
  - At θ_DFA 0.509, c from 0 to 0.5: measured +0.62 to +1.74, i.e. 45–80% hidden.
  - At c = 0.319, θ_DFA from 0.45 to 0.65: +0.99 to +1.99.
  - Over the full box (c 0–0.5, θ_DFA 0.45–0.65): +0.18 to +2.27.
  - Prediction (+0.8 to +1.7 at θ_DFA 0.509): **missed** slightly at both ends.
- **Next:** the paper update in one pass, with v2c_free3 as the main case, v2c (θ 1.38) as the alternative and θ = 1 as a check. Then the SSRN PDF rebuild.

## Paper rewrite, authorship and SSRN build (1 Nov)

- **Paper rewrite for the main case:** done in one pass. v2c_free3 (θ 1.04) is the main case, v2c (θ 1.38) the alternative and θ = 1 a check. Updated the abstract, §2.1/§3.2 (the freeze stand-in, reconciling transfers above the exemption with the model), §3.4 (main and alternative columns, "Choice of θ"), Table 2, §4.3 (the level understatement: 34.5% true vs 32.5% measured), §4.4 (the sweep uses the alternative calibration), §5.2, Appendix A.3/A.5 and Figures 1–2.
- **Abstract:** two paragraphs, 194 words. The DFA paragraph now explains that the model knows each trust's beneficiary, so it can measure trust wealth both ways.
- **Authorship:** sole-authored (Richard Berk, Nest Egg Lab, ORCID 0009-0004-5344-4614), with an acknowledgement for collaboration on an earlier draft.
- **Disclosures** on the title page:
  - no external funding;
  - founder of Nest Egg Lab, a nonprofit (Delaware; 501(c)(3) application in progress);
  - AI disclosure, as before.
- **PDF:** paper/ssrn/Hidden_Dynasties_SSRN_draft_2026-11-01e.pdf (md5 465dda2e…), built with pandoc + xelatex. fixauthor.py writes the author block into the tex.
