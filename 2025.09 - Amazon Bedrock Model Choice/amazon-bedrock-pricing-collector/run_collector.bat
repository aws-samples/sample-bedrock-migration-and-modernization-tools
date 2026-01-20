@echo off
setlocal EnableDelayedExpansion

REM Amazon Bedrock Pricing Collector Runner Script
REM Windows version

echo 🚀 Amazon Bedrock Pricing Collector - Windows Runner
echo ========================================================

REM Define paths
set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%venv"
set "REQUIREMENTS_FILE=%SCRIPT_DIR%requirements.txt"
set "MAIN_SCRIPT=%SCRIPT_DIR%main.py"

REM Function definitions using labels (Windows batch doesn't have functions)
goto :main

:print_status
echo [INFO] %~1
goto :eof

:print_success
echo [SUCCESS] %~1
goto :eof

:print_warning
echo [WARNING] %~1
goto :eof

:print_error
echo [ERROR] %~1
goto :eof

:main
REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Python is required but not installed. Please install Python 3.8 or higher."
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
call :print_status "Using Python %PYTHON_VERSION%"

REM Check if virtual environment exists
if not exist "%VENV_DIR%" (
    call :print_status "Creating virtual environment..."
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        call :print_error "Failed to create virtual environment"
        pause
        exit /b 1
    )
    call :print_success "Virtual environment created at %VENV_DIR%"
) else (
    call :print_status "Virtual environment already exists"
)

REM Activate virtual environment
call :print_status "Activating virtual environment..."
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    call :print_error "Failed to activate virtual environment"
    pause
    exit /b 1
)

REM Upgrade pip
call :print_status "Upgrading pip..."
python -m pip install --upgrade pip

REM Install/upgrade requirements
if exist "%REQUIREMENTS_FILE%" (
    call :print_status "Installing requirements from %REQUIREMENTS_FILE%..."
    pip install -r "%REQUIREMENTS_FILE%"
    if errorlevel 1 (
        call :print_error "Failed to install requirements"
        pause
        exit /b 1
    )
    call :print_success "Requirements installed successfully"
) else (
    call :print_warning "requirements.txt not found. Installing basic dependencies..."
    pip install boto3 requests
)

REM Ensure output and logs directories exist
if not exist "%SCRIPT_DIR%out" mkdir "%SCRIPT_DIR%out"
if not exist "%SCRIPT_DIR%logs" mkdir "%SCRIPT_DIR%logs"

REM Check for AWS credentials
call :print_status "Checking AWS credentials..."
python -c "import boto3; boto3.Session(profile_name='avelizf+main-Admin')" >nul 2>&1
if errorlevel 1 (
    call :print_warning "AWS credentials may not be configured properly"
    call :print_warning "Make sure your AWS profile 'avelizf+main-Admin' is set up"
)

REM Run the pricing collector
call :print_status "Starting Amazon Bedrock Pricing Collector..."
echo.

cd /d "%SCRIPT_DIR%"
python main.py

REM Check exit code
if errorlevel 1 (
    call :print_error "Pricing collection failed. Check the logs for details."
    pause
    exit /b 1
) else (
    call :print_success "Pricing collection completed successfully!"
    echo.
    call :print_status "Output files are available in: %SCRIPT_DIR%out\"
    call :print_status "Log files are available in: %SCRIPT_DIR%logs\"
)

REM Deactivate virtual environment
call "%VENV_DIR%\Scripts\deactivate.bat"

call :print_success "Script completed successfully!"
echo.
echo Press any key to exit...
pause >nul