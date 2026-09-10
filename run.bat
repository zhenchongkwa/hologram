@echo off
rem Launch the Hologram Glitch Band. Double-click it, or run it from a terminal
rem with any of run.py's arguments:  run.bat --effect vhs
setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo.
    echo Python environment not found at .venv
    echo.
    echo Create it with:
    echo     py -3.12 -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

if not exist "models\hand_landmarker.task" (
    echo Hand landmark model missing - downloading it now...
    "%PY%" scripts\download_model.py
    if errorlevel 1 (
        echo.
        echo Model download failed. Check your connection and try again.
        pause
        exit /b 1
    )
    echo.
)

echo  Hologram Glitch Band
echo  ---------------------------------------------------------------
echo   Raise BOTH hands. The shape is drawn through four fingertips:
echo   each hand's thumb and index finger. Move them to reshape it.
echo.
echo   TAP thumb to index on BOTH hands  -  next effect
echo   n  next mode - shape, cube, slit-scan
echo   b  straight to cube mode - a 3D box, open both palms for a rose
echo   g  bloom - bright areas bleed light
echo   [ ]  prev/next      p  one shape per hand      s  screenshot
echo   r  record           ?  all keys                q  quit
echo  ---------------------------------------------------------------
echo.

"%PY%" run.py %*
set "CODE=%ERRORLEVEL%"

rem Keep the window up on failure so a double-click doesn't just vanish.
if not "%CODE%"=="0" (
    echo.
    echo Exited with code %CODE%
    pause
)
exit /b %CODE%
