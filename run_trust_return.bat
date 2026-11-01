@echo off
REM FODT trust-return sensitivity (pre-registered 25 Oct 2026): trust assets earn an extra 0.5% and 1%
REM a year. Start AFTER run_theta_check.bat has finished. About 10-20 hours.
REM Keep this window open and the PC awake. Safe to restart: finished work is kept.
cd /d "%~dp0code"
echo Trust-return sensitivity. Progress: results\sweep_v2c_log.txt
py run_sweep.py cal=v2c ret05 ret10
echo.
echo Trust-return runs stopped or finished. Leave this window open and tell Claude.
pause
