"""A minimal MCP server over stdio, used to exercise the client end to end."""

import json
import sys

TOOLS = [
    {
        "name": "echo",
        "description": "Echo the supplied text back.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "boom",
        "description": "Always reports an error.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def reply(request_id, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}) + "\n")
    sys.stdout.flush()


def main():
    # Noise on stderr, to check the client drains it instead of deadlocking.
    print("fake server starting", file=sys.stderr, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        method = message.get("method")
        request_id = message.get("id")

        if method == "initialize":
            reply(request_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "0.1"},
            })
        elif method == "notifications/initialized":
            continue  # notifications carry no id and expect no reply
        elif method == "tools/list":
            reply(request_id, {"tools": TOOLS})
        elif method == "tools/call":
            params = message.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {})
            if name == "echo":
                reply(request_id, {"content": [{"type": "text", "text": f"echo: {args.get('text','')}"}]})
            elif name == "boom":
                reply(request_id, {"content": [{"type": "text", "text": "it broke"}], "isError": True})
            else:
                sys.stdout.write(json.dumps({
                    "jsonrpc": "2.0", "id": request_id,
                    "error": {"code": -32601, "message": f"no such tool: {name}"},
                }) + "\n")
                sys.stdout.flush()
        elif request_id is not None:
            reply(request_id, {})


if __name__ == "__main__":
    main()
