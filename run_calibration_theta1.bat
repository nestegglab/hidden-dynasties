@echo off
REM FODT theta = 1 full recalibration (pre-registered 27 Oct 2026, DECISIONS.md), then base and off.
REM Start ONLY after run_panel_fixes.bat has finished: the two would compete for the CPU.
REM About 12-20 hours for the calibration plus about 1.5 hours for the experiments. Safe to restart:
REM finished evaluations are replayed from the cache. Keep the PC awake.
cd /d "%~dp0code"
echo Theta = 1 recalibration. Progress: results\layerC4_v2c_theta1_log.txt
py calibrate_C.py v2c_theta1 start=V2C_TH1
echo.
echo Calibration finished. Experiments (base, off). Progress: results\layerC5_v2c_theta1_log.txt
py run_c5.py cal=v2c_theta1 base off
echo.
echo Theta = 1 runs stopped or finished. Leave this window open and tell Claude.
pause
