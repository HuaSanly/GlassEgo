#!/usr/bin/env python3
"""Run pip check while allowing only documented research-package metadata gaps."""

from __future__ import annotations

import subprocess
import sys


ALLOWED_PREFIXES = (
    "hamer 0.0.0 requires detectron2",
    "hamer 0.0.0 requires mmcv",
    "hamer 0.0.0 requires opencv-python",
    "orient-anything 2.0.0 requires opencv-python",
    "ultralytics ",
    "pyrender 0.1.45 has requirement PyOpenGL==3.1.0",
)


def main() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    unexpected = [
        line
        for line in result.stdout.splitlines()
        if line and not line.startswith(ALLOWED_PREFIXES)
    ]
    if unexpected:
        print("Unexpected pip dependency errors:", file=sys.stderr)
        print("\n".join(unexpected), file=sys.stderr)
        return 1
    print("pip dependency check: OK (documented metadata gaps ignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
