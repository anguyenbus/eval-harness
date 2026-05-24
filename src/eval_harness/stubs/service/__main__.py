"""
Entry point for python -m eval_harness.stubs.service.

This module enables invoking the stub service CLI via:
    uv run python -m eval_harness.stubs.service --config <path>
"""

from __future__ import annotations

from eval_harness.stubs.service.main import main

if __name__ == "__main__":
    main()
