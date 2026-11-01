@echo off
REM FODT grid check: base and no-trusts at a finer trust grid (70 points). About 3-5 hours.
REM Keep this window open and the PC awake. Safe to restart.
cd /d "%~dp0code"
echo Starting the grid check. Progress is written to results\gridcheck_log.txt
py run_gridcheck.py
echo.
echo Grid check stopped or finished. Leave this window open and tell Claude.
pause
