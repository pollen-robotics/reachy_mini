"""Generate the OpenAPI JSON spec from the FastAPI app without starting the server.

Usage:
    python scripts/generate_openapi.py

The script imports the FastAPI app, extracts its OpenAPI schema, and writes it
to docs/source/API/openapi.json. This file is committed to the repo so the
REST API documentation can be browsed without a running daemon.

This is a *static* copy of the spec. The live, interactive version (Swagger UI)
is served by the daemon itself at:

    http://localhost:8000/docs       (Lite — daemon on your machine)
    http://reachy-mini.local:8000/docs  (Wireless — daemon on the robot)

The live version lets you send real requests to the robot.  This static copy is
for offline reference and for the hosted documentation site.

Why this is safe to run offline
-------------------------------
``create_app(Args())`` with default arguments (``wireless_version=False``)
only registers FastAPI routes and instantiates lightweight Python objects.
All hardware initialisation happens inside the ``lifespan`` context manager,
which is only entered when ``uvicorn`` actually starts serving — so this
script never touches motors, serial ports, or audio devices.
"""

import json
from pathlib import Path

from reachy_mini.daemon.app.main import Args, create_app

OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent / "docs" / "source" / "API" / "openapi.json"
)


def main() -> None:
    """Extract the OpenAPI schema from the FastAPI app and write it to docs."""
    # create_app(Args()) builds the FastAPI app with all routes registered,
    # but does NOT start the lifespan (no hardware, no server).
    app = create_app(Args())

    # .openapi() returns the same schema served at /openapi.json (and used by
    # the live Swagger UI at /docs) when the daemon is running.
    schema = app.openapi()

    schema["info"]["title"] = "Reachy Mini REST API"
    schema["info"]["description"] = (
        "HTTP and WebSocket API exposed by the Reachy Mini daemon. "
        "Interactive Swagger docs are available at `/docs` when the daemon is running."
    )

    output = json.dumps(schema, indent=2) + "\n"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(output)
    print(f"OpenAPI spec written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
