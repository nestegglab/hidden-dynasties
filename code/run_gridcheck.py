"""
Grid check: base and no-trusts experiments on the C4 v2b calibration with a finer trust grid
(nd = 70 instead of 55), r re-solved. Compares with results/layerC5_v2b_{base,off}.json.
Restart-safe (run_c5 caches). Output: results/layerC5_v2b_nd70_{base,off}.json, log in
results/gridcheck_log.txt.
"""
# ==================================================================================================
# FILE BANNER: run_gridcheck.py -- is the trust-asset grid fine enough? (paper Appendix A)
# ==================================================================================================
# WHAT THIS FILE DOES
#   A numerical-accuracy check. It reruns the base (trusts) and off (no trusts) experiments of
#   run_c5.py with a finer grid for trust assets d (nd = 70 points instead of the usual 55), with r
#   re-solved, and logs the key results side by side with the nd = 55 results. If the effect of
#   trusts barely moves, the 55-point grid is fine enough.
#   NOTE: the check is run on the v2b calibration (CAL["v2b"]), not on the final v2c one; it compares
#   with results/layerC5_v2b_{base,off}.json, which must already exist (from
#   `py run_c5.py cal=v2b base off`), or the comparison step stops with a file-not-found error.
#
# WHERE IT SITS
#   Layer A -> Layer B -> Layer C -> calibrate_C.py -> run_c5.py -> THIS DRIVER (reuses run_c5.run).
#
# HOW IT IS LAUNCHED (author's PC)
#   `py run_gridcheck.py` -- main() takes NO command-line arguments (sys.argv is not read).
#
# FILES (results/ folder)
#   Reads:  data/eflife.dat, data/surlife.dat, results/layerB_v2_params_bestof2.json (via
#           calibrate_C.base_params); results/layerC5_v2b_base.json and layerC5_v2b_off.json
#           (the nd = 55 comparison); the nd = 70 caches/results below if they exist.
#   Writes: results/layerC5_v2b_nd70_base_evals.jsonl, ..._base.json,
#           results/layerC5_v2b_nd70_off_evals.jsonl,  ..._off.json   (through run_c5.run),
#           results/gridcheck_log.txt (appended progress log).
#
# HOW LONG IT RUNS
#   One evaluation with trusts takes 35-60 minutes at nd = 55 and somewhat longer at nd = 70 (more
#   grid points); without trusts about 2 minutes. base needs a handful of evaluations (several
#   hours); off a few minutes to tens of minutes.
#
# READING ORDER
#   ND -> main.
#
# PAPER: Appendix A (computation: grid accuracy).
#
# WORDS YOU'LL SEE
#   nd     number of points on the trust-asset grid d.
#   s1 / s0.1   trust share of top-1% / top-0.1% wealth.
#   effect (pp) base minus off, in percentage points (here for top1 and top0.1 wealth shares).
# ==================================================================================================

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_C as cc
import run_c5 as r5

ND = 70                                       # the finer trust grid used for the check


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():
    """Run base and off at nd = 70 on the v2b calibration, then log both grids' results.
    No arguments; output files are listed in the file banner."""
    cc.use_v2b()                              # v2b set-up (sets nd = 55) ...
    cc.FAST["nd"] = ND                        # ... then override the trust grid to nd = 70
    bp, ef, su = cc.base_params()
    # Point run_c5 at the v2b parameters with its own file tag, so results go to layerC5_v2b_nd70_*.
    r5.cal = dict(x=r5.CAL["v2b"]["x"], tag=f"v2b_nd{ND}_", desc=f"C4 v2b, evaluation 51, nd = {ND}")
    log = open(os.path.join(cc.RES, "gridcheck_log.txt"), "a")
    print(f"# start {time.ctime()} nd={ND}", file=log, flush=True)
    # The two experiments with r re-solved (skipped if already finished; restart-safe caches).
    for exp in ("base", "off"):
        r5.run(exp, bp, ef, su, log)
    # Compare: for the nd = 55 results (tag "v2b_") and the nd = 70 results (tag "v2b_nd70_"), log the
    # trust shares of top wealth, top1 base -> off and its effect, the top0.1 effect, and the top-1%
    # share of personal wealth in base.
    res = {}                                  # collected but not used afterwards
    for tag in ("v2b_", f"v2b_nd{ND}_"):
        b = json.load(open(os.path.join(cc.RES, f"layerC5_{tag}base.json")))["ge_moments"]
        o = json.load(open(os.path.join(cc.RES, f"layerC5_{tag}off.json")))["ge_moments"]
        res[tag] = (b, o)
        print(f"# {tag}: s1={b['trust_share_of_top1_wealth']:.4f} s0.1={b['trust_share_of_top0.1_wealth']:.4f} "
              f"top1 {b['top1']:.4f}->{o['top1']:.4f} (effect {100*(b['top1']-o['top1']):+.2f} pp) "
              f"top0.1 effect {100*(b['top0.1']-o['top0.1']):+.2f} pp personal top1 {b['top1_liquid_only']:.4f}",
              file=log, flush=True)
    print(f"# all done {time.ctime()}", file=log, flush=True)


if __name__ == "__main__":
    main()
