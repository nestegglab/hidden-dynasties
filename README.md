# Hidden Dynasties: code for "Dynasty Trusts and the Concentration of Wealth"

Richard Berk, Nest Egg Lab · richard.berk@nestegglab.org · ORCID 0009-0004-5344-4614

**Paper:** Berk, R. (2026). *Hidden Dynasties: Dynasty Trusts and the Concentration of Wealth*. Working paper, Zenodo. https://doi.org/10.5281/zenodo.23887180

This repository holds the code, input data and results behind the paper (preliminary draft,
November 2026; `paper/`). Every number in the paper comes from a file in `results/`, and every file
in `results/` can be regenerated from the code here.

The code is commented very heavily on purpose. Most files are about half comments, so a reader who
knows the economics but not the numerical methods can follow it line by line. Start with
"How to read the code" below.

---

## What the model does, in one paragraph

Households in an overlapping-generations economy live from 20 to 89 in five-year periods. They earn
labour income, may become entrepreneurs with risky high returns, save, give to their children during
life, and leave bequests and charitable gifts at death. Along the way they pay income, capital, gift
and estate taxes. In any period a household can found a dynasty trust. Assets in the trust pass to
each generation outside the estate tax (the *shield*), but heirs can take only the income, not the
principal (*lock-in*). We calibrate the model to US data, then remove trusts (or only the shield, or
only the lock-in), re-solve the interest rate, and compare top wealth shares. Top shares are measured
both as the model sees them and the way the Federal Reserve's Distributional Financial Accounts (DFA)
would measure them.

## Calibrations: which results are which

| Name in the code | Name in the paper | θ | Sum of squares | Top-1% effect |
| --- | --- | --- | --- | --- |
| `v2c_free3` | **main calibration** | 1.04 | 16.39 | +3.17 pp |
| `v2c` | alternative calibration | 1.38 | 17.04 | +2.41 pp |
| `v2c_theta1` | θ fixed at 1 | 1 | 16.62 | +3.59 pp |
| `v2b` | earlier version (grid check only) | 0.94 | — | — |

`v2c_free2` is the first free-θ restart. It stopped too early (see `DECISIONS.md`, 30 Oct) and is kept
because `v2c_free3` replayed its cached evaluations. The paper reports the range across the three
calibrations as real uncertainty about θ (Section 3.4).

## Folder layout

| Folder / file | What it holds |
| --- | --- |
| `code/` | the model and the scripts that run it (details below) |
| `data/` | inputs: earnings by age (`eflife.dat`), survival by age (`surlife.dat`; `surlife_ext99.dat` for an unused robustness version), calibration targets with sources (`targets_v2.csv`), the SCF 2022 trust tabulation |
| `results/` | everything the runs produce: logs (`*_log.txt`), a record of every model evaluation (`*.jsonl`), final results (`*.json`, `*_summary.csv`), and saved distributions for the DFA grid (`*_dist.npz`) |
| `paper/` | the paper as posted |
| `*.bat` | the one-click launchers actually used on the author's Windows PC, in the order they were run (see below) |
| `DECISIONS.md` | the dated project log: every modelling choice, pre-registration, prediction, miss, failed variant and review |
| `KNOWN_ISSUES.md` | issues found in code reviews, with their status and likely effect |

## How to read the code

Read in this order. Each file starts with a banner that says what it does, how it fits in, and the
order to read its functions.

1. **`code/layerC.py`, the module docstring at the top.** The best single overview of the model.
2. **`code/layerB.py`**: return states (normal, entrepreneur, high-growth), prices, the dollar scale,
   survival.
3. **`code/layerC.py`, `ParamsC`**: every parameter, its units and the value used in the calibrations.
4. **`code/layerC.py`, `_beq1`, `_menu`, `_decide`**: one household's period (paper Section 3.2).
5. **`code/layerC.py`, `EconomyC.solve`, `EconomyC.distribution`, `EconomyC.moments`,
   `dfa_top_shares`**: solving all households, simulating generations, measuring the results.
6. **`code/calibrate_C.py`**: the targets and the calibration (paper Section 3.4).
7. **`code/run_c5.py`, `code/run_sweep.py`, `code/dfa_grid.py`**: the experiments and robustness
   checks (Sections 3.6, 4, 5).

`code/layerA.py` is the original life-cycle model the project started from. The final model uses only
two helper functions from it (the earnings-process discretisation). The rest is kept so that earlier
stages in `DECISIONS.md` can be reproduced.

**Search tags in the comments:**

- `PAPER:` links a line to an equation or section of the paper.
- `PYTHON NOTE:` explains a Python or NumPy idiom the first time it appears.
- `NUMERICS:` explains a numerical method in plain words.
- `APPROXIMATION:` flags a simplification.
- `UNITS:` says whether an amount is in dollars or model units, per year or per five-year period.

**A few words you will meet everywhere:**

- **model units:** money is measured so that model mean wealth equals the SCF mean ($1.06M); `usd` converts.
- **period:** five years; `t = 0` is ages 20–24 and `t = 13` is ages 85–89.
- **`a`, `d`, `m`:** liquid wealth, trust assets, cash on hand.
- **`y`, `z`:** earnings state and return state.
- **category `c`:** 0 no trust, 1 heir, 2 founder (3 and 4 are an advanced-planning tier, switched off in all reported results).

**Switches.** New behaviour is added behind a switch whose default reproduces the earlier code
exactly. The switches used in reported results are `trusts`, `shield`, `lock`, `trust_weight`,
`payout_fracs` (partial withdrawals), `fund_cap_E` (funding capped at the exemption), `gift_exempt`
(the gift-exemption bound), `trust_prem` (extra return on trust assets) and `tier2`. All are
documented in `ParamsC`.

## Map from the paper to the code

| Paper | Code | Results files |
| --- | --- | --- |
| 3.1 Households: demographics, earnings, returns | `layerB.py` (`EconomyB.__init__`, `_exogenous`, `_prices`); `layerA.rouwenhorst` | |
| 3.2 Bequests, gifts, trusts, the household problem (Table 2) | `layerC.py`: `_beq1`, `_decide`, `_menu`, `_contribs`; inheritance in `EconomyC.conv` | |
| 3.3 Firms, government, equilibrium | `layerB._prices`, `equilibrium.ky_for`, `run_c5.run` | |
| 3.4 Calibration (Tables 3, 4) | `calibrate_C.py` (targets `TARGETS_V2C`); Layer B values from `results/layerB_v2_params_bestof2.json` | `layerC4_v2c_free3_*`, `layerC4_v2c_*`, `layerC4_v2c_theta1_*` |
| 3.5 DFA-consistent top shares | `layerC.dfa_top_shares` | |
| 3.6, 4.1 Experiments (Tables 5, 6) | `run_c5.py` (`base`, `off`) | `layerC5_v2c_free3_base*`, `layerC5_v2c_free3_off*` |
| 4.2 Mechanism (Figure 1) | `run_c5.py` (`lock0` = shield only, `noshield_fix` = lock-in only) | `layerC5_v2c_free3_lock0*`, `layerC5_v2c_free3_noshield_fix*` |
| 4.3 What official statistics would show | `layerC.dfa_top_shares`; `dfa_grid.py` for the range over measurement choices | `layerC5_v2c_free3_*`, `dfa_grid_v2c_free3*` |
| 4.4 Trust-share range (Figure 2) | `run_sweep.py` (`s10`, `s15`, `s27`) on the alternative calibration | `layerC5_sweep_v2c_s*`, `sweep_v2c_s*` |
| 4.5 Robustness: θ, freeze transfers | `run_c5.py cal=v2c_theta1`; `run_sweep.py` (`theta1`, `theta1_t2`) | `layerC5_v2c_theta1_*`, `layerC5_sweep_v2c_theta1*` |
| 4.5 Robustness: gift bound, trust return (alternative calibration) | `run_c5.py` (`base_giftnoex`, `off_giftnoex`); `run_sweep.py` (`ret05`, `ret10`) | `layerC5_v2c_*giftnoex*`, `layerC5_sweep_v2c_ret*` |
| 4.5 Robustness: funding cap, partial withdrawals (main calibration) | `run_c5.py` (`base_capE`, `lock0_partial`) | `layerC5_v2c_free3_base_capE*`, `layerC5_v2c_free3_lock0_partial*` |
| 4.5 Robustness: grid | `run_gridcheck.py` (at v2b) | `layerC5_v2b_nd70_*`, `gridcheck_log.txt` |
| 5.2 θ and the SCF take-up check | `scf_takeup_check.py` | printed output; see `DECISIONS.md` |
| Appendix A Computation (Tables A1–A3) | `EconomyC.solve`, `EconomyC.distribution`, `EconomyC._anderson` | the `*_log.txt` and `*.jsonl` files (run times, convergence) |

Figure A1 is a diagram, not a result.

## How to run it

**Requirements:** Python 3.11 or later with NumPy, SciPy and Numba (`pip install -r requirements.txt`).
`scf_takeup_check.py` also needs pandas and the SCF 2022 summary extract (`SCFP2022.csv`, from the
Federal Reserve's SCF page), saved as `data/raw/SCFP2022.csv`. It is not included here. The first run
of any script compiles the model with Numba, which takes a few minutes; after that the compiled code
is cached.

**Time:** one model evaluation with trusts takes 35–65 minutes on a desktop PC (about 2 minutes without
trusts). A calibration takes 25–35 evaluations (one to two days); a set of experiments, a few hours.

**Commands**, in the order they were run (run from inside `code/`; the `.bat` file with the same steps
is in the last column):

| Step | Command | Launcher |
| --- | --- | --- |
| 1. Alternative calibration | `python calibrate_C.py v2c start=V2B` | `run_calibration_v2c.bat` |
| 2. Experiments and trust-share range | `python run_c5.py cal=v2c base off noshield lock0`, then `python run_sweep.py cal=v2c s15 s10 s27` | `run_final_v2c.bat` |
| 3. θ and tier-2 check | `python run_sweep.py cal=v2c theta1_t2 theta1` | `run_theta_check.bat` |
| 4. Trust-return check | `python run_sweep.py cal=v2c ret05 ret10` | `run_trust_return.bat` |
| 5. Gift bound | `python run_c5.py cal=v2c base_giftnoex off_giftnoex` | `run_gift_bound.bat` |
| 6. Review-panel checks (alternative calibration) | `python run_c5.py cal=v2c noshield_fix lock0_partial base_capE`, then `python dfa_grid.py` | `run_panel_fixes.bat` |
| 7. θ = 1 calibration and experiments | `python calibrate_C.py v2c_theta1 start=V2C_TH1`, then `python run_c5.py cal=v2c_theta1 base off` | `run_calibration_theta1.bat`, `run_theta1_experiments.bat` |
| 8. Free-θ restart (stopped early) | `python calibrate_C.py v2c_free2 start=V2C_FREE2` | `run_calibration_free2.bat` |
| 9. **Main calibration** and experiments | copy `layerC4_v2c_free2_evals.jsonl` to `layerC4_v2c_free3_evals.jsonl`, then `python calibrate_C.py v2c_free3 start=V2C_FREE2`, then `python run_c5.py cal=v2c_free3 base off` | `run_calibration_free3.bat` |
| 10. Main-case decomposition, funding cap, DFA grid | `python run_c5.py cal=v2c_free3 lock0 lock0_partial noshield_fix base_capE`, then `python dfa_grid.py cal=v2c_free3` | `run_main_reruns.bat` |
| Grid check (earlier version v2b) | `python run_gridcheck.py` | `run_gridcheck.bat` |
| SCF take-up check (untargeted) | `python scf_takeup_check.py` | |

The launchers print messages such as "tell Claude" because the runs were monitored in AI-assisted
sessions (see "Use of AI tools" below).

**Restarts and reproducing from scratch:** every script keeps a record of each evaluation and skips
finished work, so an interrupted run can simply be restarted. This also means that, with the
`results/` folder as shipped, the scripts will report "already done". To recompute everything from
scratch, empty `results/` except `layerB_v2_params_bestof2.json`, which holds the Layer B parameters
the final model uses. Two parameter files were written by hand from the evaluation cache after a run
was stopped (`layerC4_v2c_params.json`, `layerC4_v2c_theta1_params.json`; see `KNOWN_ISSUES.md`), so a
full rerun of those two calibrations needs the same manual step.

## Checks that the code does what the paper says

- **Nesting:** with trusts switched off, Layer C reproduces Layer B exactly. Every switch added later
  reproduces the earlier code exactly at its default.
- **Grid:** raising the trust grid from 55 to 70 points moves the main effect by 0.1 percentage points.
- **Comments only:** the commented files here were checked against the files that produced the results
  by comparing their parsed program structure, which ignores comments and docstrings. Every file is
  identical as a program.

## License

Code: MIT License (`LICENSE`). The paper in `paper/` is © the author; see the posting for its terms.

## Use of AI tools

The model code, calibration and experiment scripts, and these comments were written and run with the
assistance of Claude, an AI model made by Anthropic, working under the author's direction. All modelling
choices, targets and pre-registrations were approved by the author (recorded in `DECISIONS.md`), and the
author is responsible for the results.
