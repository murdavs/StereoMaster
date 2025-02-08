@echo off
setlocal enableextensions enabledelayedexpansion

echo ============================================================
echo  StereoMaster Dependency Installation Script
echo ============================================================
echo.

:: ------------------------------------------------------------
:: Step 1: Change to the script's directory
:: ------------------------------------------------------------
cd /d "%~dp0"

:: ------------------------------------------------------------
:: Step 2: Check if Python is real (not a Store alias)
:: ------------------------------------------------------------
echo Checking for a real Python installation...
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [WARNING] "python" not found in PATH.
    goto :PYTHON_INSTALL_PROMPT
) else (
    echo [INFO] "python" appears in PATH. Checking if it's real...
    set "PYTHON_VERSION_OUTPUT="
    for /f "delims=" %%v in ('python --version 2^>nul') do (
        set "PYTHON_VERSION_OUTPUT=%%v"
    )
    if not defined PYTHON_VERSION_OUTPUT (
        echo [WARNING] Python might be a Microsoft Store alias.
        goto :PYTHON_INSTALL_PROMPT
    ) else (
        echo [OK] Found a real Python: !PYTHON_VERSION_OUTPUT!
        set "PYTHON_EXE=python"
        goto :AFTER_PYTHON_CHECK
    )
)

:PYTHON_INSTALL_PROMPT
echo [WARNING] Python is not truly installed (or is the Store alias).
echo.

:ASK_INSTALL_PYTHON
set /P "INSTALL_PYTHON=Do you want to install Python 3.10 now? (Y/N): "
if /I "%INSTALL_PYTHON%"=="Y" goto :INSTALL_PYTHON
if /I "%INSTALL_PYTHON%"=="N" (
    echo Python will not be installed. Exiting...
    pause
    exit /B 0
)
echo Invalid choice. Type Y or N.
goto :ASK_INSTALL_PYTHON

:INSTALL_PYTHON
echo.
echo [INFO] Installing Python 3.10. Please wait...

set "PYTHON_VERSION=3.10.9"
set "PYTHON_INSTALLER=python-3.10.9-amd64.exe"
set "PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/%PYTHON_INSTALLER%"

if not exist "%PYTHON_INSTALLER%" (
    echo Downloading %PYTHON_INSTALLER%...
    curl -k -L -O "%PYTHON_URL%"
    if not exist "%PYTHON_INSTALLER%" (
        echo [ERROR] Download failed or file not found after download.
        pause
        exit /B 1
    ) else (
        echo [OK] Downloaded successfully.
    )
)

echo Launching Python 3.10 installer (interactive mode)...
echo [INFO] Run as Administrator if you want "Install for all users".
start /wait "" "%PYTHON_INSTALLER%" InstallAllUsers=1 PrependPath=1 TargetDir="C:\Python310"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python 3.10 setup failed.
    pause
    exit /B 1
)

:: Check if the new python.exe exists in C:\Python310
if exist "C:\Python310\python.exe" (
    echo [OK] Python 3.10 was successfully installed at C:\Python310.
    set "PYTHON_EXE=C:\Python310\python.exe"
    echo Removing the installer "%PYTHON_INSTALLER%"...
    if exist "%PYTHON_INSTALLER%" del /f /q "%PYTHON_INSTALLER%"
) else (
    echo [WARNING] Could not confirm Python 3.10 in C:\Python310.
    echo [WARNING] Will try "python" from PATH.
    set "PYTHON_EXE=python"
)

:AFTER_PYTHON_CHECK
echo.

:: ------------------------------------------------------------
:: Step 3: Check if Git is installed
:: ------------------------------------------------------------
echo Checking for Git...
where git >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [WARNING] "git" is not installed or not in PATH.
    goto :GIT_INSTALL_PROMPT
) else (
    echo [OK] Git is already installed.
    goto :AFTER_GIT_CHECK
)

:GIT_INSTALL_PROMPT
echo [WARNING] Git is not installed.
echo.

:ASK_INSTALL_GIT
set /P "INSTALL_GIT=Do you want to install Git for Windows now? (Y/N): "
if /I "%INSTALL_GIT%"=="Y" goto :INSTALL_GIT
if /I "%INSTALL_GIT%"=="N" (
    echo Git will not be installed. Continuing...
    goto :AFTER_GIT_CHECK
)
echo Invalid choice. Type Y or N.
goto :ASK_INSTALL_GIT

:INSTALL_GIT
echo.
echo [INFO] Installing Git for Windows. Please wait...

set "GIT_VERSION=2.39.2"
set "GIT_INSTALLER=Git-%GIT_VERSION%-64-bit.exe"
set "GIT_URL=https://github.com/git-for-windows/git/releases/download/v%GIT_VERSION%.windows.1/%GIT_INSTALLER%"

if not exist "%GIT_INSTALLER%" (
    echo Downloading %GIT_INSTALLER%...
    curl -k -L -O "%GIT_URL%"
    if not exist "%GIT_INSTALLER%" (
        echo [ERROR] Download failed or file not found after download.
        pause
        exit /B 1
    ) else (
        echo [OK] Downloaded successfully.
    )
)

echo Launching Git installer (interactive mode)...
start /wait "" "%GIT_INSTALLER%"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Git setup failed.
    pause
    exit /B 1
)

where git >nul 2>&1
if %ERRORLEVEL%==0 (
    echo [OK] Git appears to be installed correctly.
    echo Removing the installer "%GIT_INSTALLER%"...
    if exist "%GIT_INSTALLER%" del /f /q "%GIT_INSTALLER%"
) else (
    echo [WARNING] Could not confirm Git installation in PATH.
    echo [WARNING] Please verify your Git installation manually.
)

:AFTER_GIT_CHECK
echo.

:: ------------------------------------------------------------
:: Step 4: Check or create the virtual environment
:: ------------------------------------------------------------
echo Checking if "stereomaster_env" exists...
if exist "stereomaster_env\Scripts\activate.bat" (
    echo [OK] "stereomaster_env" already exists.
) else (
    echo [INFO] Creating virtual environment...
    "%PYTHON_EXE%" -m venv "stereomaster_env"
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Could not create virtual environment.
        pause
        exit /B 1
    )
    echo [OK] Virtual environment created.
)

:: ------------------------------------------------------------
:: Step 5: Activate the environment
:: ------------------------------------------------------------
echo Activating virtual environment...
call "stereomaster_env\Scripts\activate.bat"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to activate the environment.
    pause
    exit /B 1
)
echo [OK] Environment activated.
echo.

:: ------------------------------------------------------------
:: Step 6: Upgrade pip
:: ------------------------------------------------------------
echo Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to upgrade pip.
    pause
    exit /B 1
)
echo [OK] pip upgraded successfully.
echo.

:: ------------------------------------------------------------
:: Step 7: Install packages (requirements + Triton wheel)
:: ------------------------------------------------------------
echo Updating pip, setuptools, and wheel...
pip install --upgrade pip setuptools wheel
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to update pip/setuptools/wheel.
    pause
    exit /B 1
)



echo Installing packages from requirements.txt with PEP 517...
pip install --use-pep517 --no-cache-dir -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install packages from requirements.txt.
    pause
    exit /B 1
)

echo Installing Triton wheel...
curl -k -L -o assets/triton-2.1.0-cp310-cp310-win_amd64.whl https://huggingface.co/spaces/Murdavs/StereoMaster/resolve/main/triton-2.1.0-cp310-cp310-win_amd64.whl?download=true
pip install .\assets\triton-2.1.0-cp310-cp310-win_amd64.whl
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install the Triton wheel.
    pause
    exit /B 1
)

echo [OK] Packages installed successfully.
echo.


:: ------------------------------------------------------------
:: Step 8: Download essential model weights
:: ------------------------------------------------------------
cd /d "%~dp0"
echo Creating "weights" directory if it doesn't exist...
if not exist "weights" (
    mkdir "weights"
)

pushd "weights"

echo Installing Git LFS...
git lfs install
if %ERRORLEVEL% neq 0 (
    echo [ERROR] "git lfs install" failed.
    pause
    exit /B 1
)
echo [OK] Git LFS installed.

echo Cloning "stable-video-diffusion-img2vid-xt-1-1" repository...
git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to clone "stable-video-diffusion-img2vid-xt-1-1".
    pause
    exit /B 1
)
echo [OK] Cloned "stable-video-diffusion-img2vid-xt-1-1".

echo Cloning "DepthCrafter" repository...
git clone https://huggingface.co/tencent/DepthCrafter
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to clone "DepthCrafter".
    pause
    exit /B 1
)
echo [OK] Cloned "DepthCrafter".

echo Cloning "StereoCrafter" repository...
git clone https://huggingface.co/TencentARC/StereoCrafter
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to clone "StereoCrafter".
    pause
    exit /B 1
)
echo [OK] Cloned "StereoCrafter".

popd
echo.

:: ------------------------------------------------------------
:: Step 9: Download Video-Depth-Anything checkpoints
:: ------------------------------------------------------------
cd /d "%~dp0"
echo Creating "checkpoints" directory if it doesn't exist...
if not exist "checkpoints" (
    mkdir "checkpoints"
)

pushd "checkpoints"

echo Downloading "video_depth_anything_vits.pth"...
curl -k -L -O https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
if not exist "video_depth_anything_vits.pth" (
    echo [ERROR] Failed to download "video_depth_anything_vits.pth".
    pause
    exit /B 1
)
echo [OK] Downloaded "video_depth_anything_vits.pth".

echo Downloading "video_depth_anything_vitl.pth"...
curl -k -L -O https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth
if not exist "video_depth_anything_vitl.pth" (
    echo [ERROR] Failed to download "video_depth_anything_vitl.pth".
    pause
    exit /B 1
)
echo [OK] Downloaded "video_depth_anything_vitl.pth".

popd
echo.


:: ------------------------------------------------------------
:: Step 10: Forward-Warp dependencies
:: ------------------------------------------------------------
cd /d "%~dp0"
echo Installing Forward-Warp dependencies...
call "dependency\Forward-Warp\setup.bat"
echo [OK] Forward-Warp dependencies installed.
echo.

echo ============================================================
echo  All dependencies for StereoMaster are installed!
echo ============================================================
pause
exit /B
