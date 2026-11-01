@echo off
REM FODT gift-exemption bound (pre-registered 26 Oct 2026): gifts above the $300k allowance get no
REM exemption; economies with and without trusts, interest rate re-solved. About 3-4 hours.
REM Keep this window open and the PC awake. Safe to restart: finished work is kept.
cd /d "%~dp0code"
echo Gift-exemption bound. Progress: results\layerC5_v2c_log.txt
py run_c5.py cal=v2c base_giftnoex off_giftnoex
echo.
echo Gift bound stopped or finished. Leave this window open and tell Claude.
pause
