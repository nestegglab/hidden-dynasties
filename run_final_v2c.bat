@echo off
REM FODT final runs on the v2c calibration: main experiments, then the trust-share range.
REM About 14-18 hours. Keep this window open and the PC awake. Safe to restart: finished work is kept.
cd /d "%~dp0code"
echo Step 1 of 2: main experiments. Progress: results\layerC5_v2c_log.txt
py run_c5.py cal=v2c base off noshield lock0
echo Step 2 of 2: trust-share range. Progress: results\sweep_v2c_log.txt
py run_sweep.py cal=v2c s15 s10 s27
echo.
echo Final runs stopped or finished. Leave this window open and tell Claude.
pause
