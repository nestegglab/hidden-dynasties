@echo off
REM FODT theta check (pre-registered 25 Oct 2026): founder trust weight fixed at 1, with and without
REM the trust-feeding planning tools (tier 2, freeze transfers only). About 10-20 hours.
REM Keep this window open and the PC awake. Safe to restart: finished work is kept.
cd /d "%~dp0code"
echo Theta check. Progress: results\sweep_v2c_log.txt
py run_sweep.py cal=v2c theta1_t2 theta1
echo.
echo Theta check stopped or finished. Leave this window open and tell Claude.
pause
