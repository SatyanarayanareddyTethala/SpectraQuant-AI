@echo off
REM SpectraQuant-AI Installation Script for Windows
REM Run this from Command Prompt or PowerShell

echo ========================================
echo SpectraQuant-AI Installation
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found
    echo Please install Python 3.11+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [INFO] Python found
python --version

REM Check Git
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found
    echo Please install Git from https://git-scm.com/download/win
    pause
    exit /b 1
)

echo [INFO] Git found
echo.

REM Ask installation type
echo Select installation type:
echo 1) Core only (basic features)
echo 2) Full (core + intelligence layer)
set /p INSTALL_TYPE="Enter choice [1-2]: "

if not "%INSTALL_TYPE%"=="1" if not "%INSTALL_TYPE%"=="2" (
    echo [ERROR] Invalid choice
    pause
    exit /b 1
)

echo.
echo ========================================
echo Starting Installation
echo ========================================
echo.

REM Create virtual environment
echo [INFO] Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
echo [SUCCESS] Virtual environment created

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [SUCCESS] Virtual environment activated

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip (continuing anyway)
) else (
    echo [SUCCESS] pip upgraded
)

REM Install core dependencies
echo [INFO] Installing core dependencies (this may take 5-15 minutes)...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install core dependencies
    pause
    exit /b 1
)
echo [SUCCESS] Core dependencies installed

REM Install intelligence layer if requested
if "%INSTALL_TYPE%"=="2" (
    echo [INFO] Installing intelligence layer dependencies...
    pip install -r trading_assistant\requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install intelligence layer dependencies
        pause
        exit /b 1
    )
    echo [SUCCESS] Intelligence layer dependencies installed
)

REM Install package
echo [INFO] Installing SpectraQuant package...
pip install -e .
if errorlevel 1 (
    echo [ERROR] Failed to install SpectraQuant package
    pause
    exit /b 1
)
echo [SUCCESS] SpectraQuant package installed

echo.
echo ========================================
echo Verifying Installation
echo ========================================
echo.

REM Verify CLI
echo [INFO] Checking CLI...
spectraquant --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] CLI not found in PATH (may need to restart shell)
) else (
    echo [SUCCESS] CLI is available
)

REM Run doctor
echo [INFO] Running environment diagnostics...
spectraquant doctor

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.

echo [SUCCESS] SpectraQuant-AI has been successfully installed
echo.
echo Next steps:
echo   1. Activate virtual environment: .venv\Scripts\activate.bat
echo   2. View help: spectraquant --help
echo   3. Configure: edit config.yaml
echo   4. Download data: spectraquant download
echo   5. Run pipeline: spectraquant refresh
echo.

if "%INSTALL_TYPE%"=="2" (
    echo Intelligence Layer installed. To setup:
    echo   python scripts\bootstrap_intelligence.py
    echo.
)

echo For more information, see:
echo   - README.md - Main documentation
echo   - INSTALLATION.md - Detailed installation guide
echo   - DEPENDENCIES.md - Dependency reference
echo.

echo [SUCCESS] Happy trading! 📈🚀
echo.
pause
