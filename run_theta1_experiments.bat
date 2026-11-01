@echo off
REM FODT theta = 1: experiments (base, off) at the recalibrated parameters (29 Oct 2026).
REM Use after closing run_calibration_theta1.bat: the calibration was stopped by the stop rule,
REM and its best point was written to results\layerC4_v2c_theta1_params.json. About 1.5 hours.
cd /d "%~dp0code"
echo Theta = 1 experiments. Progress: results\layerC5_v2c_theta1_log.txt
py run_c5.py cal=v2c_theta1 base off
echo.
echo Theta = 1 experiments stopped or finished. Leave this window open and tell Claude.
pause
