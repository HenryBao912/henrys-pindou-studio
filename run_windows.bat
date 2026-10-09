@echo off
rem Henry's Pindou Studio launcher for Windows. Double-click to start.
cd /d "%~dp0"
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
%PY% -c "import PIL, numpy, reportlab" >nul 2>nul || (
  echo Installing Pillow, numpy and reportlab - one time only...
  %PY% -m pip install --user pillow numpy reportlab
)
%PY% henrys_pindou_studio.py %*
if errorlevel 1 pause
