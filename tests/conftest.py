import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[0] / "src"

# The package under test, and the fixtures directory, both need to be importable
# before collection. Fixtures are plain modules rather than a package so that
# fake_mcp_server.py can also be run directly as a subprocess entry point.
for path in (SRC, HERE / "fixtures"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
