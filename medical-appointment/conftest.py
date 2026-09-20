"""Shared pytest setup.

Keeps the project root importable and stops the serving path from loading model
weights at import time during tests.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MEDAPP_SKIP_WARMUP", "1")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
