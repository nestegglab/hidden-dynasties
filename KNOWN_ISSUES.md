# Known issues

A line-by-line review of the code, done while writing the comments (October 2026), found the items
below; later reviews added a few. The code here is exactly the code that produced the results. Each
item says whether it matters for reported numbers and what was done about it. Dates refer to entries
in `DECISIONS.md`.

## Could affect reported results

1. **Lifetime gifts never use up the estate-tax exemption.** *Bounded; the code is unchanged by default.*
   - **What happens:** at the age-55 gift stage, a gift is taxed only above the $300k allowance plus the full remaining exemption (`_decide`, `Erem_g`). But nothing records that the exemption was used. The same household then gets the full exemption again when founding a trust in that period (`_menu` / `_contribs`) and at death (`_beq1`, `_estates_fill`, `moments`).
   - **What the law does:** gifts above the annual exclusions use up the same lifetime exemption as the estate.
   - **Effect:** the model makes large lifetime gifts too cheap, in both the trust and no-trust economies.
   - **What was done (26 Oct):** a pre-registered bound with the switch `gift_exempt = False`, in which gifts above the allowance get no exemption at all (the opposite extreme). On the alternative calibration it raises the top-1% effect from +2.4 to +2.9 points (paper Section 4.5; results `layerC5_v2c_*giftnoex*`). The paper's §3.2 describes what the code does.
2. **Founders' exemption and the taxable-estate count in the no-shield experiment.** *Fixed (27 Oct).*
   - Without the shield, trust assets are taxed in the estate, yet the code still reduced a founder's exemption by the trust's value (counting it twice), and the count of taxable estates left trust assets out.
   - Both are fixed (`_estates_fill`, `EconomyC.estates`, `moments` now check `shield`). Defaults are unchanged, so no other result moves.
   - The reported lock-in-only experiment is the rerun `noshield_fix`. The older `layerC5_v2c_noshield*` files predate the fix and are kept for the record only.

## Affect untargeted or reporting statistics only

3. **Mobility measures are conditional on the parent being alive.** The parent-child elasticity and rank-rank slope use children whose parent is alive when the child is 35–39 (or 50–54).
4. **`choice_shares` labels.** It classifies choices by comparing the trust after the choice with the trust at the start of the period. A retained trust after a very good return can be labelled a contribution, and after a loss a payout. This is used only for descriptive tables.
5. **`dfa_top_shares`.** The top-1% group of personal wealth can hold slightly more than 1% of households when wealth is lumpy on the grid. The search for the allocation exponent also stops at 4 without a warning. Small effect; `dfa_grid.py` shows how the measured effect varies with the two measurement choices.
6. **`cap_diagnostics`.** Approximate: it ignores medical costs (off in all results) and checks the cash floor in a different order from `_decide`.

## Housekeeping (no effect on results)

7. **Parameter files written by hand.** `calibrate_C.main` writes the parameter file only when `least_squares` finishes or the automatic stop rule fires.
   - Two runs were stopped by hand at a plateau, and their files were written from the evaluation cache: `layerC4_v2c_params.json` (best = evaluation 33) and `layerC4_v2c_theta1_params.json` (best = evaluation 24, sum of squares 16.62).
   - The θ = 1 run went on for a few more evaluations after its file was written and reached 16.60 at evaluation 31. The θ = 1 experiments use evaluation 24, as written; the free-θ restart started from evaluation 31.
   - The `*_evals.jsonl` files hold every evaluation.
8. **The main calibration's history.** `v2c_free2` was stopped by the automatic rule after one rejected step because the rule's window (8 evaluations) was shorter than one finite-difference Jacobian (7 evaluations plus a step). The window was raised to 16 *after* seeing this (30 Oct, disclosed in `DECISIONS.md`), and `v2c_free3` replayed `v2c_free2`'s cached evaluations and continued. The main calibration is `v2c_free3`.
9. **`calibrate_B2.py` vs the Layer B parameter file.** `calibrate_B2.py` writes `layerB_v2_params.json`, while the final model reads `layerB_v2_params_bestof2.json`, the better of two Layer B runs, selected by hand (see `DECISIONS.md`).
10. **`run_sweep.py` naming.** `X_V2B` and the saved descriptions say "v2b" even when `cal=v2c` is used; the numbers are v2c's. The theta and trust-return points target 0.2058 while `S0["v2c"]` is 0.2060; the difference is inside the tolerance. The discarded `steep` variant is still in the default list.
11. **`equilibrium.solve_r` is unused.** `run_c5.run` has its own version of the interest-rate search.
12. **Small inconsistencies:**
    - The gift tax ignores the tier-2 valuation discount. Tier 2 is off in all reported results except the freeze-transfer check.
    - `_beq1` guards the bequest value but not its derivative against a negative estate in the no-shield case. No problem was observed: all runs converged.
    - A cash floor of 0.01% of mean earnings can create a negligible amount of money.
    - Layer A's capital-tax revenue uses simple, not compounded, interest. Layer A is not used for results.
