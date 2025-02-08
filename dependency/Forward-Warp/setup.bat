@echo off
set "SCRIPT_DIR=%~dp0"

cd /d "%SCRIPT_DIR%\Forward_Warp\cuda"
python setup.py install

cd /d "%SCRIPT_DIR%"
python setup.py install

