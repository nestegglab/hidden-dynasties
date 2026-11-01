@echo off
REM FODT main-case reruns at the free-theta calibration v2c_free3 (pre-registered 31 Oct 2026, DECISIONS.md):
REM decomposition (shield only, partial withdrawals, lock-in only), funding cap, then the DFA grid.
REM About 8-9 hours. Safe to restart: finished work is kept. Keep the PC awake.
cd /d "%~dp0code"
echo Main-case reruns. Progress: results\layerC5_v2c_free3_log.txt
py run_c5.py cal=v2c_free3 lock0 lock0_partial noshield_fix base_capE
echo.
echo DFA grid (about an hour)...
py dfa_grid.py cal=v2c_free3 > ..\results\dfa_grid_v2c_free3_log.txt 2>&1
type ..\results\dfa_grid_v2c_free3_log.txt
echo.
echo Main-case reruns stopped or finished. Leave this window open and tell Claude.
pause
