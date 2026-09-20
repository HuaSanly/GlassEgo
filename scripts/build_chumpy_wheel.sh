#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${1:-$PROJECT_ROOT/environment/wheels}"
SOURCE_URL="https://files.pythonhosted.org/packages/01/f7/865755c8bdb837841938de622e6c8b5cb6b1c933bde3bd3332f0cd4574f1/chumpy-0.70.tar.gz"
SOURCE_SHA256="a0275c2018784ca1302875567dc81761f5fd469fab9f3ac0f3e7c39e9180350a"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

curl -fsSL "$SOURCE_URL" -o "$BUILD_DIR/chumpy-0.70.tar.gz"
echo "$SOURCE_SHA256  $BUILD_DIR/chumpy-0.70.tar.gz" | sha256sum --check --status
tar -xzf "$BUILD_DIR/chumpy-0.70.tar.gz" -C "$BUILD_DIR"
patch -d "$BUILD_DIR/chumpy-0.70" -p1 < "$PROJECT_ROOT/environment/chumpy-py311.patch"
mkdir -p "$OUTPUT_DIR"
python -m pip wheel --no-deps --no-build-isolation \
  "$BUILD_DIR/chumpy-0.70" --wheel-dir "$OUTPUT_DIR"
