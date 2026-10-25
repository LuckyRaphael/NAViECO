"""Versioned, checksum-verified NAViECO reference acquisition."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile

from .reference_registry import REFERENCE_ASSETS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract plain files only, refusing paths outside destination."""
    with ZipFile(archive) as zf:
        files = zf.infolist()
        if not files:
            raise ValueError("NAViECO reference archive is empty")
        for info in files:
            name = info.filename
            parts = Path(name).parts
            if not name or name.startswith(("/", "\\")) or ".." in parts or ":" in name:
                raise ValueError(f"Unsafe reference entry: {name}")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Reference archive must not contain symlinks: {name}")
        for info in files:
            target = destination / info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out, length=8 * 1024 * 1024)


def get_reference(name: str = "NAViECO_v1") -> Path:
    """Return a local reference directory, downloading a named release on first use.

    An explicitly configured NAVIECO_REFERENCE_DIR takes precedence. Downloaded
    versions are cached under $XDG_CACHE_HOME/navieco/references (or ~/.cache).
    """
    override = os.environ.get("NAVIECO_REFERENCE_DIR")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"NAVIECO_REFERENCE_DIR is not a directory: {path}")
        return path

    if name not in REFERENCE_ASSETS:
        raise ValueError(
            f"No published download manifest for reference {name!r}. "
            "Publish the versioned reference release and registry before public use."
        )

    spec = REFERENCE_ASSETS[name]
    url, expected = spec["url"], spec["sha256"].lower()
    if len(expected) != 64 or any(x not in "0123456789abcdef" for x in expected):
        raise ValueError("Invalid packaged SHA-256 for NAViECO reference")
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    parent = cache_root.expanduser() / "navieco" / "references"
    parent.mkdir(parents=True, exist_ok=True)
    dest = parent / name
    seal = dest / ".navieco_reference_integrity.json"

    def ready():
        if not seal.is_file():
            return False
        try:
            return json.loads(seal.read_text())["sha256"] == expected
        except (OSError, ValueError, KeyError):
            return False

    if ready():
        os.environ["NAVIECO_REFERENCE_DIR"] = str(dest)
        return dest

    lock = parent / (name + ".lock")
    acquired = False
    for _ in range(600):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            if ready():
                os.environ["NAVIECO_REFERENCE_DIR"] = str(dest)
                return dest
            time.sleep(1)
    if not acquired:
        raise TimeoutError(f"Timed out waiting for NAViECO reference cache lock: {lock}")

    try:
        if not ready():
            with tempfile.TemporaryDirectory(dir=parent, prefix=f".{name}-") as tmp:
                temp = Path(tmp)
                archive = temp / "reference.zip"
                print(f"[NAViECO] Downloading {name} to local cache: {dest}", flush=True)
                request = Request(url, headers={"User-Agent": "NAViECO-reference-client/1"})
                with urlopen(request, timeout=90) as stream, archive.open("wb") as out:
                    shutil.copyfileobj(stream, out, length=8 * 1024 * 1024)
                if _sha256(archive) != expected:
                    raise ValueError("Reference SHA-256 mismatch; refusing unverified data")
                staging = temp / "extract"
                staging.mkdir()
                _safe_extract(archive, staging)
                if not any(staging.rglob("*.h5ad")):
                    raise ValueError("Reference archive has no H5AD; verify release contents")
                (staging / ".navieco_reference_integrity.json").write_text(
                    json.dumps({"name": name, "sha256": expected, "url": url}, indent=2)
                )
                # Never replace an existing cache without a separate explicit action.
                if dest.exists():
                    if ready():
                        os.environ["NAVIECO_REFERENCE_DIR"] = str(dest)
                        return dest
                    raise RuntimeError(f"Reference cache exists but is not verified: {dest}")
                staging.rename(dest)
        os.environ["NAVIECO_REFERENCE_DIR"] = str(dest)
        return dest
    finally:
        lock.unlink(missing_ok=True)
