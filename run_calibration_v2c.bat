@echo off
REM FODT Layer C calibration (C4). Runs about a day. Keep this window open and the PC awake.
cd /d "%~dp0code"
echo Checking Python...
py --version
if errorlevel 1 (
  echo Python was not found. Tell Claude and it will walk you through installing it.
  pause
  exit /b
)
echo Installing or updating numpy, scipy, numba...
py -m pip install --user --upgrade numpy scipy numba
echo.
echo Starting calibration. Progress is written to results\layerC4_v2c_log.txt
echo The first line of results should appear within about 30-60 minutes.
py calibrate_C.py v2c start=V2B
echo.
echo Calibration stopped. Leave this window open and tell Claude.
pause
