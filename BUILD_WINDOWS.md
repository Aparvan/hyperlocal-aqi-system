# Building & Running AeroSense AI as a Windows Desktop App (`.exe`)

AeroSense AI includes a native desktop wrapper ([`desktop_app.py`](desktop_app.py)) and a one-click PyInstaller build pipeline ([`AeroSense.spec`](AeroSense.spec) + [`build_exe.bat`](build_exe.bat)) so you can run or distribute the platform as a standalone Windows `.exe` with no terminal window and no external browser required.

---

## Option A — Run the Desktop Window Directly from Python (No Build Step)

```powershell
# 1. Install dependencies (optional: add pywebview for an embedded WebView2 native window)
pip install -r requirements.txt
pip install pywebview

# 2. Launch the native desktop window
python desktop_app.py
```

### How Window Selection Works
1. **`pywebview` (Preferred)**: Opens a native Windows Edge WebView2 window (`AeroSense AI — Hyperlocal Air Quality Prediction & Mapping System`) with no address bar or browser tabs.
2. **Edge / Chrome App Mode (Automatic Fallback)**: If `pywebview` is not installed, `desktop_app.py` automatically detects Microsoft Edge (`msedge.exe`, pre-installed on Windows 10/11) or Google Chrome and launches a dedicated chromeless `--app` window that shuts down the backend cleanly when closed.
3. **Standard Browser Mode**: Pass `--browser` (`python desktop_app.py --browser`) to open in your default browser tab, or `--headless` to run the REST API server only.

---

## Option B — Build a Standalone `AeroSense.exe` on Windows

### One-Click Build (Recommended)
Open **Command Prompt** or **PowerShell** in the project root and run:

```bat
.\build_exe.bat
```

This script automatically:
1. Creates/activates `.venv` and installs `requirements.txt` + `pywebview` + `pyinstaller`.
2. Ensures `saved_models/multimodal_model.pt` and `assets/app_icon.ico` exist.
3. Packages everything using `AeroSense.spec` into:
   ```text
   dist\AeroSense\AeroSense.exe
   ```
Double-click `dist\AeroSense\AeroSense.exe` (or create a Desktop shortcut to it) to launch the application with zero console popup.

### Single-File Portable `.exe`
If you want a single `.exe` file instead of a folder bundle:

```bat
.\build_exe.bat --onefile
```
Output:
```text
dist\AeroSense.exe
```
*(Note: Because PyTorch DLLs are ~250 MB, the folder bundle `dist\AeroSense\AeroSense.exe` starts in ~1–2 seconds, whereas `--onefile` unpacks to `%TEMP%` on each launch and takes ~8–12 seconds to start.)*

---

## Frozen Path & Log Architecture

When running as `AeroSense.exe`:
- **Read-only bundled assets** (`static/`, `saved_models/`, `data_store/`) are loaded from PyInstaller's `sys._MEIPASS` directory via `config.RESOURCE_DIR`.
- **Writable runtime files** (`cache/` and `aerosense.log`) are written next to `AeroSense.exe` via `config.RUNTIME_DIR` (or automatically fall back to `%LOCALAPPDATA%\AeroSense` if installed under a protected path like `C:\Program Files`).
