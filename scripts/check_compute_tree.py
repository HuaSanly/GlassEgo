#!/usr/bin/env python3
"""Compatibility entry point for the compute projection tree policy."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.compute_projection import main


if __name__ == "__main__":
    raise SystemExit(main(["check", *sys.argv[1:]]))
