@echo off
REM FODT mock-panel checks (pre-registered 27 Oct 2026, see DECISIONS.md "Panel runs"):
REM noshield bug fix, lock0 with partial withdrawal, trust funding capped at the exemption,
REM and the DFA measurement grid. About 7 hours. Keep this window open and the PC awake.
REM Safe to restart: finished work is kept.
cd /d "%~dp0code"
echo Panel checks. Progress: results\layerC5_v2c_log.txt
py run_c5.py cal=v2c noshield_fix lock0_partial base_capE
echo.
echo DFA grid (about an hour)...
py dfa_grid.py > ..\results\dfa_grid_v2c_log.txt 2>&1
type ..\results\dfa_grid_v2c_log.txt
echo.
echo Panel checks stopped or finished. Leave this window open and tell Claude.
pause
