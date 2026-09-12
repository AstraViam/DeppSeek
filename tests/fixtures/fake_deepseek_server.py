"""A minimal fake of DeepSeek's streaming chat-completions endpoint.

Exists to exercise the real HTTP and server-sent-events path: tool-call
arguments arrive as fragments indexed by position and have to be reassembled,
and usage metadata arrives in a final chunk that carries no choices. Both are
easy to get wrong and invisible to a test that stubs the provider out.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar


def sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def chunk(delta: dict, finish: str | None = None) -> dict:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "deepseek-flash",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


class Handler(BaseHTTPRequestHandler):
    # Class attributes because BaseHTTPRequestHandler instantiates a new handler
    # per request; the test sets these on the class to script the responses.
    script: ClassVar[list[list[dict]]] = []
    requests: ClassVar[list[dict]] = []

    def log_message(self, *args):
        """Silence the default per-request logging to stderr."""
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        Handler.requests.append(body)

        frames = Handler.script.pop(0) if Handler.script else [chunk({"content": "done"}, "stop")]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for frame in frames:
            self.wfile.write(sse(frame))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def start() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}"
