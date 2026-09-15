#!/usr/bin/env python3
"""Fetch MaleCNS v1.0 connectome data into data/raw/ with resume + SHA256 verification.

Usage:
    uv run python tools/fetch_connectome.py [--only NAME] [--verify-only] [--write-checksums]
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = [
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
CHECKSUMS_PATH = Path(__file__).resolve().parent / "checksums.json"
CHUNK_SIZE = 1024 * 1024


def load_checksums() -> dict:
    with open(CHECKSUMS_PATH) as f:
        return json.load(f)


def save_checksums(checksums: dict) -> None:
    with open(CHECKSUMS_PATH, "w") as f:
        json.dump(checksums, f, indent=2)
        f.write("\n")


def remote_size(name: str) -> int:
    req = urllib.request.Request(BASE_URL + name, method="HEAD")
    with urllib.request.urlopen(req) as resp:
        return int(resp.headers["Content-Length"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def progress(name: str, done: int, total: int) -> None:
    pct = 100.0 * done / total if total else 0.0
    print(f"\r{name}: {done}/{total} bytes ({pct:.1f}%)", end="", file=sys.stderr, flush=True)


def download(name: str, expected_sha256: str | None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / name
    total = remote_size(name)

    existing = dest.stat().st_size if dest.exists() else 0
    if existing == total:
        if expected_sha256 is None or sha256_file(dest) == expected_sha256:
            print(f"{name}: already complete, skipping", file=sys.stderr)
            return
        print(f"{name}: checksum mismatch, redownloading", file=sys.stderr)
        existing = 0

    if existing > total:
        existing = 0

    mode = "ab" if existing else "wb"
    req = urllib.request.Request(BASE_URL + name)
    if existing:
        req.add_header("Range", f"bytes={existing}-")

    with urllib.request.urlopen(req) as resp, open(dest, mode) as out:
        done = existing
        progress(name, done, total)
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            progress(name, done, total)
    print(file=sys.stderr)

    actual = sha256_file(dest)
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError(f"{name}: SHA256 mismatch after download (got {actual})")


def verify(name: str, expected_sha256: str | None) -> bool:
    dest = DATA_DIR / name
    if not dest.exists():
        print(f"{name}: missing", file=sys.stderr)
        return False
    if expected_sha256 is None:
        print(f"{name}: no recorded checksum to verify against", file=sys.stderr)
        return False
    actual = sha256_file(dest)
    ok = actual == expected_sha256
    print(f"{name}: {'OK' if ok else f'MISMATCH (got {actual})'}", file=sys.stderr)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Fetch only this file name")
    parser.add_argument(
        "--verify-only", action="store_true", help="Verify existing files, no download"
    )
    parser.add_argument(
        "--write-checksums", action="store_true", help="Record SHA256 of downloaded files"
    )
    args = parser.parse_args()

    names = [args.only] if args.only else FILES
    for name in names:
        if name not in FILES:
            print(f"unknown file: {name}", file=sys.stderr)
            return 1

    checksums = load_checksums()

    if args.verify_only:
        ok = all(verify(name, checksums.get(name)) for name in names)
        return 0 if ok else 1

    for name in names:
        download(name, checksums.get(name))
        if args.write_checksums:
            checksums[name] = sha256_file(DATA_DIR / name)
            save_checksums(checksums)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
