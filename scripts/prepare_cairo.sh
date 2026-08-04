#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-./data/cairo}"
mkdir -p "$DEST"

cat <<EOM
[cairo] Place dataset at $DEST and write a manifest.json describing splits.
Source: British Dental Journal 2025 — request from authors
License: TBD
EOM
