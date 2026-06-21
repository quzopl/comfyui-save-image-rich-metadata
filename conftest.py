"""Make the module root importable as a package so pytest's Package.setup() succeeds."""
import sys
from pathlib import Path

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
