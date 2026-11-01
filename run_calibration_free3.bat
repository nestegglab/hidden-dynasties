@echo off
REM FODT free-theta restart, rerun with a 16-evaluation stop window (DECISIONS.md, 30 Oct 2026).
REM Replays the 9 finished evaluations of v2c_free2, continues, stops automatically, then runs base and off.
REM About 8-15 hours plus 1.5 hours. Safe to restart. Keep the PC awake.
cd /d "%~dp0code"
if not exist ..\results\layerC4_v2c_free3_evals.jsonl copy ..\results\layerC4_v2c_free2_evals.jsonl ..\results\layerC4_v2c_free3_evals.jsonl
echo Free-theta rerun. Progress: results\layerC4_v2c_free3_log.txt
py calibrate_C.py v2c_free3 start=V2C_FREE2
echo.
echo Calibration finished. Experiments (base, off). Progress: results\layerC5_v2c_free3_log.txt
py run_c5.py cal=v2c_free3 base off
echo.
echo Free-theta rerun stopped or finished. Leave this window open and tell Claude.
pause
