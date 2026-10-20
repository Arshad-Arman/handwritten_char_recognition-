@echo off
rem Double-click this file (Windows) to run the whole project.
rem Options are passed through, e.g.:  run.bat --mode quick
cd /d "%~dp0"
python run.py %*
echo.
pause
