"""Pytest configuration for test path modifications."""
import sys
from pathlib import Path

# Add scripts directory to Python path for imports
scripts_path = Path(__file__).parent.parent / "scripts"
if str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path))
