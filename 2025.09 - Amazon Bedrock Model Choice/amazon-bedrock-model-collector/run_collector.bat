@echo off
REM Amazon Bedrock Model Collector Runner Script
REM Windows version

setlocal enabledelayedexpansion

echo 🚀 Amazon Bedrock Model Collector - Windows Runner
echo ====================================================
echo Comprehensive model database with enhanced features
echo.

REM Define paths
set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%venv"
set "REQUIREMENTS_FILE=%SCRIPT_DIR%requirements.txt"
set "MAIN_SCRIPT=%SCRIPT_DIR%main.py"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is required but not installed. Please install Python 3.8 or higher.
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [INFO] Using Python %PYTHON_VERSION%

REM Create virtual environment if it doesn't exist
if not exist "%VENV_DIR%" (
    echo [INFO] Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

REM Install requirements
echo [INFO] Installing requirements...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ❌ Failed to upgrade pip
    exit /b 1
)

python -m pip install -r "%REQUIREMENTS_FILE%"
if errorlevel 1 (
    echo ❌ Failed to install requirements
    exit /b 1
)

REM Create directories
if not exist "%SCRIPT_DIR%out" mkdir "%SCRIPT_DIR%out"
if not exist "%SCRIPT_DIR%logs" mkdir "%SCRIPT_DIR%logs"

REM Check AWS credentials
echo [INFO] Checking AWS credentials for profile: avelizf+main-Admin
python -c "import boto3; boto3.Session(profile_name='avelizf+main-Admin')" 2>nul
if errorlevel 1 (
    echo ⚠️  AWS credentials may not be configured for profile 'avelizf+main-Admin'
)

REM Run collector
echo [INFO] Starting Amazon Bedrock Model Collector...
echo.
python main.py

if errorlevel 1 (
    echo ❌ Collection failed. Check logs for details.
    call "%VENV_DIR%\Scripts\deactivate.bat"
    exit /b 1
)

echo [SUCCESS] Model collection completed successfully!
echo [INFO] Output files: %SCRIPT_DIR%out\
echo [INFO] Log files: %SCRIPT_DIR%logs\

call "%VENV_DIR%\Scripts\deactivate.bat"
echo [SUCCESS] Script completed!
pause