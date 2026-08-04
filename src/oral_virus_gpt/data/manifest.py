from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class ManifestEntry:
    relative_path: str
    sha256: str
    size_bytes: int


def compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_manifest(entries: Iterable[ManifestEntry], output: Path) -> None:
    payload = {
        "version": "1.0",
        "entries": [asdict(e) for e in sorted(entries, key=lambda x: x.relative_path)],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_manifest(path: Path) -> list[ManifestEntry]:
    payload = json.loads(path.read_text())
    return [ManifestEntry(**entry) for entry in payload.get("entries", [])]
