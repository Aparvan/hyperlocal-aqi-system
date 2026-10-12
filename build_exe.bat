@echo off
REM ============================================================================
REM AeroSense AI — One-Click Windows Desktop Executable (.exe) Builder
REM ============================================================================
REM Usage:
REM   build_exe.bat            -> Builds fast-startup folder bundle: dist\AeroSense\AeroSense.exe
REM   build_exe.bat --onefile  -> Builds single-file portable executable: dist\AeroSense.exe
REM ============================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo [1/5] Checking Python environment...
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not on PATH. Install Python 3.10+ from https://www.python.org/
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [2/5] Creating virtual environment .venv ...
    python -m venv .venv
) else (
    echo [2/5] Using existing virtual environment .venv ...
)

call ".venv\Scripts\activate.bat"

echo [3/5] Installing runtime and packaging dependencies (pywebview + pyinstaller)...
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pywebview pyinstaller

if not exist "saved_models\multimodal_model.pt" (
    echo [3.5/5] Pre-training Multimodal Deep Learning weights for bundle...
    python models\train_eval.py
)

if not exist "assets\app_icon.ico" (
    echo [3.8/5] Generating application icon...
    python tools\make_icon.py
)

if /I "%~1"=="--onefile" (
    echo [4/5] Building single-file portable executable ^(dist\AeroSense.exe^)...
    pyinstaller --noconfirm --clean --onefile --windowed ^
        --name AeroSense ^
        --icon "assets\app_icon.ico" ^
        --add-data "static;static" ^
        --add-data "saved_models;saved_models" ^
        --add-data "data_store;data_store" ^
        --add-data "assets;assets" ^
        desktop_app.py
    echo.
    echo [5/5] Build complete! Single-file executable:
    echo       %~dp0dist\AeroSense.exe
) else (
    echo [4/5] Building fast-startup desktop application bundle ^(dist\AeroSense\AeroSense.exe^)...
    pyinstaller --noconfirm --clean AeroSense.spec
    echo.
    echo [5/5] Build complete! Launch the desktop app by double-clicking:
    echo       %~dp0dist\AeroSense\AeroSense.exe
)

endlocal
