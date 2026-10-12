# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification for building the standalone Windows desktop executable:
    dist/AeroSense/AeroSense.exe  (fast-startup one-folder bundle, recommended for PyTorch)

To build on Windows:
    pyinstaller --noconfirm --clean AeroSense.spec
or simply double-click / run:
    build_exe.bat
"""

from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).resolve()

# Bundle read-only data directories alongside the executable
datas = [
    (str(ROOT / "static"), "static"),
    (str(ROOT / "saved_models"), "saved_models"),
    (str(ROOT / "data_store"), "data_store"),
]
if (ROOT / "assets").exists():
    datas.append((str(ROOT / "assets"), "assets"))

hiddenimports = [
    "config",
    "api.app",
    "data.generator",
    "data.weather_fetcher",
    "data.station_fetcher",
    "data.satellite_fetcher",
    "data.preprocessor",
    "models.baselines",
    "models.spatial_cnn",
    "models.temporal_lstm",
    "models.multimodal_fusion",
    "models.st_gat",
    "models.pinn_loss",
    "models.xai_engine",
    "services.grid_engine",
    "services.hotspot_detector",
    "services.advisor_agent",
    "services.scenario_simulator",
    "services.route_optimizer",
    "services.multi_agent_war_room",
    "werkzeug.serving",
]

# Include pywebview backends if pywebview is installed in the build environment
try:
    import webview  # noqa: F401
    hiddenimports += collect_submodules("webview")
except ImportError:
    pass

icon_file = ROOT / "assets" / "app_icon.ico"
icon_arg = str(icon_file) if icon_file.exists() else None

a = Analysis(
    [str(ROOT / "desktop_app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib.tests", "scipy.spatial.cKDTree"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AeroSense",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Windowed desktop app (no black terminal popup on Windows)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_arg,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AeroSense",
)
