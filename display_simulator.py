import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

HOST = "127.0.0.1"
PORT = 8765
DISPLAY_RATE_LIMIT_SECONDS = 1.5

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sign display simulator</title>
  <style>
    :root { color-scheme: dark; font-family: Arial, sans-serif; }
    body { margin: 0; padding: 2rem 1rem; background: #020617; color: #f8fafc; text-align: center; }
    main { width: min(42rem, 100%); margin: auto; }
    .card { margin: 1.25rem 0; padding: 1.5rem; border: 1px solid #334155; border-radius: 1rem; background: #0f172a; }
    .sign { min-height: 2.5rem; color: #fbbf24; font-size: 2rem; font-weight: bold; }
    .matrix { height: 6rem; overflow: hidden; display: flex; align-items: center; border: .5rem solid #334155; border-radius: .75rem; background: #020403; }
    .message { display: inline-block; min-width: 100%; color: #4ade80; font: bold 2.25rem/1 monospace; white-space: nowrap; text-shadow: 0 0 .6rem #22c55e; }
    .message.scroll { animation: scroll 8s linear infinite; }
    @keyframes scroll { from { transform: translateX(100%); } to { transform: translateX(-100%); } }
    .status { color: #94a3b8; }
  </style>
</head>
<body>
  <main>
    <h1>Sign-language display simulator</h1>
    <p class="status">Local ESP8266 simulator — no hardware or Wi-Fi connection needed.</p>
    <section class="card">
      <h2>Latest recognized sign</h2>
      <div id="sign" class="sign">Waiting for a sign...</div>
    </section>
    <section class="card">
      <h2>Four-module LED matrix preview</h2>
      <div class="matrix"><div id="message" class="message">READY</div></div>
    </section>
    <p class="status">Keep this page open while the webcam app is running.</p>
  </main>
  <script>
    async function refresh() {
      try {
        const response = await fetch("/state", { cache: "no-store" });
        if (!response.ok) throw new Error("HTTP " + response.status);
        const state = await response.json();
        document.getElementById("sign").textContent = state.recognized || "Waiting for a sign...";
        const message = document.getElementById("message");
        message.textContent = state.display || "READY";
        message.classList.toggle("scroll", (state.display || "").length > 4);
      } catch (_) {
        document.getElementById("sign").textContent = "Simulator connection lost";
      }
    }
    refresh();
    setInterval(refresh, 300);
  </script>
</body>
</html>
""".encode("utf-8")


class DisplayState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.display_text = "READY"
        self.recognized_sign = ""
        self.last_update = 0.0

    def update(self, message: str, source: str) -> bool:
        now = time.monotonic()
        with self.lock:
            if now - self.last_update < DISPLAY_RATE_LIMIT_SECONDS:
                return False
            self.last_update = now
            self.display_text = message.upper()
            if source == "sign":
                self.recognized_sign = message
            return True

    def snapshot(self) -> dict[str, str]:
        with self.lock:
            return {
                "display": self.display_text,
                "recognized": self.recognized_sign,
            }


STATE = DisplayState()


class DisplaySimulatorHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        request = urlsplit(self.path)

        if request.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
            return

        if request.path == "/recognized":
            self._send_text(STATE.snapshot()["recognized"] or "Waiting for a sign...")
            return

        if request.path == "/state":
            body = json.dumps(STATE.snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if request.path == "/text":
            self._handle_text(request.query)
            return

        self._send_text("Not found", status=404)

    def _handle_text(self, query: str) -> None:
        arguments = parse_qs(query, keep_blank_values=True)
        message = arguments.get("msg", [""])[0].strip()
        if not message:
            self._send_text("Missing msg", status=400)
            return

        source = arguments.get("source", [""])[0]
        if not STATE.update(message, source):
            self._send_text("SKIP")
            return
        self._send_text("OK")

    def _send_text(self, message: str, status: int = 200) -> None:
        body = message.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.log_date_time_string()} - {format % args}")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), DisplaySimulatorHandler)
    print(f"Display simulator: http://{HOST}:{PORT}/")
    print("Use this as SIGN_DISPLAY_URL while running test.py.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping display simulator.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
