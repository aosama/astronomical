"""Serve signed Sparkle acceptance artifacts on an ephemeral loopback port."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--port-file", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    if not arguments.directory.is_dir():
        raise ValueError(f"Serve directory does not exist: {arguments.directory}")

    request_handler = partial(SimpleHTTPRequestHandler, directory=str(arguments.directory))
    with ThreadingHTTPServer(("127.0.0.1", 0), request_handler) as server:
        arguments.port_file.write_text(f"{server.server_port}\n", encoding="utf-8")
        print(f"Sparkle acceptance server listening on 127.0.0.1:{server.server_port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
