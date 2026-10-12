"""
AeroSense AI — Standalone Desktop Application Launcher.

Starts the Flask GeoAI backend on an available local loopback port in a background
daemon thread, waits until `/api/status` reports healthy, and opens the interactive
dashboard inside a dedicated native desktop window:

  1. Preferred: `pywebview` (uses native Windows Edge WebView2 / WinForms / GTK / Cocoa
     with zero external browser chrome — feels like a native `.exe` application).
  2. Fallback A: Microsoft Edge or Google Chrome in `--app=<url>` chromeless window mode
     (pre-installed on every Windows 10/11 PC), blocking until the app window closes.
  3. Fallback B: Default system browser tab (`webbrowser.open`), keeping the server
     alive until Ctrl+C or process termination.

Works identically from source (`python desktop_app.py`) and when frozen into a
Windows executable (`AeroSense.exe`) via PyInstaller (`AeroSense.spec`).
"""

import os
import sys
import time
import socket
import shutil
import logging
import argparse
import threading
import subprocess
import webbrowser
import urllib.request
from pathlib import Path

# Ensure project root is importable
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import RUNTIME_DIR, RESOURCE_DIR  # noqa: E402

# Configure logging to file when running as a windowless .exe (console=False)
LOG_FILE = RUNTIME_DIR / "aerosense.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        *(
            [logging.StreamHandler(sys.stdout)]
            if sys.stdout is not None and hasattr(sys.stdout, "write")
            else []
        ),
    ],
)
log = logging.getLogger("AeroSenseDesktop")

APP_TITLE = "AeroSense AI — Hyperlocal Air Quality Prediction & Mapping System"
WINDOW_WIDTH = 1480
WINDOW_HEIGHT = 920


def find_free_port(preferred: int = 5000) -> int:
    """Return `preferred` if available on 127.0.0.1, otherwise an OS-assigned free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_flask_thread(host: str, port: int) -> threading.Thread:
    """Import the Flask app and run it in a background daemon thread."""
    from api.app import app
    from werkzeug.serving import make_server

    class ServerThread(threading.Thread):
        def __init__(self):
            super().__init__(daemon=True)
            self.srv = make_server(host, port, app, threaded=True)
            self.ctx = app.app_context()
            self.ctx.push()

        def run(self):
            log.info("Flask GeoAI server listening on http://%s:%d", host, port)
            self.srv.serve_forever()

        def shutdown(self):
            try:
                self.srv.shutdown()
            except Exception:
                pass

    t = ServerThread()
    t.start()
    return t


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Poll `/api/status` until the server responds with HTTP 200 or timeout expires."""
    status_url = f"{url.rstrip('/')}/api/status"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(status_url, timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.15)
    return False


def launch_pywebview(url: str) -> bool:
    """Attempt to open a native OS window via `pywebview`. Returns True if used."""
    try:
        import webview  # type: ignore
    except ImportError:
        return False

    log.info("Launching native desktop window via pywebview -> %s", url)
    try:
        webview.create_window(
            APP_TITLE,
            url,
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
            min_size=(1024, 680),
            text_select=True,
        )
        webview.start()
        return True
    except Exception as exc:
        log.warning("pywebview failed (%s); falling back to chromeless browser app mode.", exc)
        return False


def find_chromium_binary() -> str | None:
    """Locate Microsoft Edge or Google Chrome on Windows / macOS / Linux."""
    candidates: list[str] = []
    if sys.platform.startswith("win"):
        prog_files = [
            os.environ.get("PROGRAMFILES", r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
            os.environ.get("LOCALAPPDATA", ""),
        ]
        for base in prog_files:
            if not base:
                continue
            candidates.extend([
                os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
                os.path.join(base, "Google", "Chrome", "Application", "chrome.exe"),
            ])
    elif sys.platform == "darwin":
        candidates.extend([
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        ])
    else:
        for name in ("microsoft-edge", "google-chrome", "chromium", "chromium-browser"):
            found = shutil.which(name)
            if found:
                candidates.append(found)

    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def launch_browser_app_mode(url: str) -> bool:
    """
    Launch Edge/Chrome in standalone `--app=<url>` mode with an isolated profile so the
    process stays alive for the lifetime of the window (blocking until the user closes it).
    """
    binary = find_chromium_binary()
    if not binary:
        return False

    profile_dir = RUNTIME_DIR / "cache" / "app_window_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        binary,
        f"--app={url}",
        f"--window-size={WINDOW_WIDTH},{WINDOW_HEIGHT}",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
    ]
    log.info("Launching standalone app window via %s -> %s", Path(binary).name, url)
    try:
        proc = subprocess.Popen(cmd)
        proc.wait()
        return True
    except Exception as exc:
        log.warning("Browser app-mode launch failed (%s).", exc)
        return False


def main():
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=0, help="Bind port (default: 5000 or next free)")
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open in default web browser instead of native pywebview window",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Start the backend server only without opening a GUI window",
    )
    args = parser.parse_args()

    port = args.port if args.port > 0 else find_free_port(5000)
    url = f"http://{args.host}:{port}"

    log.info("Starting %s", APP_TITLE)
    log.info("Resource dir: %s | Runtime dir: %s", RESOURCE_DIR, RUNTIME_DIR)

    server_thread = start_flask_thread(args.host, port)
    if not wait_for_server(url, timeout=30.0):
        log.error("Backend failed to respond within 30s at %s", url)
        sys.exit(1)

    log.info("Backend ready at %s", url)

    try:
        if args.headless:
            log.info("Running in headless mode. Press Ctrl+C to stop.")
            while True:
                time.sleep(1.0)
        elif not args.browser and launch_pywebview(url):
            pass
        elif not args.browser and launch_browser_app_mode(url):
            pass
        else:
            log.info("Opening dashboard in default system browser: %s", url)
            webbrowser.open(url)
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        log.info("Shutting down AeroSense AI...")
    finally:
        server_thread.shutdown()


if __name__ == "__main__":
    main()
