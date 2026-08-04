from __future__ import annotations

from pathlib import Path

from oral_virus_gpt.data.manifest import (
    ManifestEntry,
    compute_sha256,
    load_manifest,
    write_manifest,
)


def test_manifest_roundtrip(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    a.write_bytes(b"hello")
    b = tmp_path / "b.bin"
    b.write_bytes(b"world")
    entries = [
        ManifestEntry(relative_path="a.bin", sha256=compute_sha256(a), size_bytes=a.stat().st_size),
        ManifestEntry(relative_path="b.bin", sha256=compute_sha256(b), size_bytes=b.stat().st_size),
    ]
    out = tmp_path / "manifest.json"
    write_manifest(entries, out)
    loaded = load_manifest(out)
    assert loaded == sorted(entries, key=lambda x: x.relative_path)


def test_manifest_invariant_under_reordering(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    a.write_bytes(b"x")
    b = tmp_path / "b.bin"
    b.write_bytes(b"y")
    e1 = [
        ManifestEntry("a.bin", compute_sha256(a), a.stat().st_size),
        ManifestEntry("b.bin", compute_sha256(b), b.stat().st_size),
    ]
    e2 = list(reversed(e1))
    out_a = tmp_path / "m1.json"
    out_b = tmp_path / "m2.json"
    write_manifest(e1, out_a)
    write_manifest(e2, out_b)
    assert out_a.read_text() == out_b.read_text()
