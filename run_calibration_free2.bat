@echo off
REM FODT free-theta restart (pre-registered 29 Oct 2026, DECISIONS.md): all 7 parameters free, starting
REM from the best theta = 1 point; stops automatically by the stop rule, then runs base and off.
REM About 8-20 hours plus 1.5 hours. Safe to restart: finished evaluations are replayed. Keep the PC awake.
cd /d "%~dp0code"
echo Free-theta restart. Progress: results\layerC4_v2c_free2_log.txt
py calibrate_C.py v2c_free2 start=V2C_FREE2
echo.
echo Calibration finished. Experiments (base, off). Progress: results\layerC5_v2c_free2_log.txt
py run_c5.py cal=v2c_free2 base off
echo.
echo Free-theta runs stopped or finished. Leave this window open and tell Claude.
pause
