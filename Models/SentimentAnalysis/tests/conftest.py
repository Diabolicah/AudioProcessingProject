"""Put the package directory on sys.path.

Every module in this project imports its siblings flatly (`from Preprocess
import ...`), so the tests need `Models/SentimentAnalysis` importable no matter
which directory pytest was launched from.
"""

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
