"""Tiny local web server for the UI (standard library only, no extra installs).

    python -m task_organizer --serve            # http://127.0.0.1:8000
    python -m task_organizer --serve --port 9000

The page is served from task_organizer/static/index.html and talks to
POST /api/organize. Your API key stays on the server side.
"""

import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .graph import organize_tasks
from .llm import ConfigError, has_api_key

STATIC_DIR = Path(__file__).parent / "static"
MAX_BODY_BYTES = 20_000


class Handler(BaseHTTPRequestHandler):
    server_version = "TodoBestie/1.0"

    # -- helpers ----------------------------------------------------------
    def _send(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict):
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            html = (STATIC_DIR / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif path == "/health":
            self._json(200, {"ok": True, "api_key_configured": has_api_key()})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/organize":
            return self._json(404, {"error": "not found"})

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._json(400, {"error": "bad Content-Length"})
        if length <= 0 or length > MAX_BODY_BYTES:
            return self._json(400, {"error": "message is empty or too long"})

        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            text = data.get("text", "") if isinstance(data, dict) else ""
        except (ValueError, UnicodeDecodeError):
            return self._json(400, {"error": "request must be valid JSON"})

        if not isinstance(text, str) or not text.strip():
            return self._json(400, {"error": "type a few tasks first, bestie 🫶"})

        try:
            state = organize_tasks(text)
        except ConfigError as e:
            return self._json(500, {"error": str(e)})
        except Exception as e:  # upstream API / network trouble
            return self._json(502, {"error": f"the AI had a moment: {e}"})

        self._json(200, {"tasks": state.get("final_tasks", [])})


def make_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    try:
        server = make_server(host, port)
    except OSError as e:
        print(f"Couldn't start on port {port}: {e}\nTry another one with --port 9000",
              file=sys.stderr)
        raise SystemExit(1)

    url = f"http://{host}:{port}"
    print(f"✨ to-do bestie is live at {url}   (Ctrl+C to stop)")
    if not has_api_key():
        print("⚠️  No API key found: add GROQ_API_KEY or XAI_API_KEY to your .env.",
              file=sys.stderr)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye bestie 💖")
    finally:
        server.server_close()
