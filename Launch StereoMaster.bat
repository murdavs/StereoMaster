@echo off
title StereoMaster Launcher
color 0A
echo =============================================================
echo      StereoMaster - Launch Script
echo =============================================================
echo.

REM -------------- Activate the virtual environment --------------
echo Activating the virtual environment: stereomaster_env
call stereomaster_env\Scripts\activate
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Could not activate stereomaster_env.
    pause
    EXIT /B 1
)
echo [OK] Environment activated.
echo.

REM -------------- Start StereoMaster in the background --------------
start "" pythonw StereoMaster.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Could not launch StereoMaster.
    pause
    EXIT /B 1
)
echo [OK] StereoMaster launched. The process will continue running in the background.
echo.
exit
