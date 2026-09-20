#!/usr/bin/env python3
"""Download and validate the model assets used by offline preprocessing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "environment" / "model-assets.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--checksums-out", type=Path, default=None)
    args = parser.parse_args()
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    resolved = []
    for asset in manifest["assets"]:
        common = {
            "repo_id": asset["repo_id"],
            "revision": asset["revision"],
            "cache_dir": str(args.cache_dir) if args.cache_dir else None,
            "local_files_only": args.local_files_only,
        }
        if asset.get("snapshot"):
            path = Path(snapshot_download(**common))
            resolved.append({"repo_id": asset["repo_id"], "path": str(path)})
            print(f"ok {asset['repo_id']} snapshot: {path}")
            continue
        for filename in asset["files"]:
            path = Path(hf_hub_download(filename=filename, **common))
            resolved.append(
                {
                    "repo_id": asset["repo_id"],
                    "revision": asset["revision"],
                    "filename": filename,
                    "sha256": sha256(path),
                    "path": str(path),
                }
            )
            print(f"ok {asset['repo_id']}/{filename}: {path}")

    if args.checksums_out is not None:
        args.checksums_out.parent.mkdir(parents=True, exist_ok=True)
        args.checksums_out.write_text(
            json.dumps({"schema_version": 1, "resolved": resolved}, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
