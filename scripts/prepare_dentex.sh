#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-./data/dentex}"
mkdir -p "$DEST"

cat <<EOM
[dentex] Place dataset at $DEST and write a manifest.json describing splits.
Source: https://dentex.grand-challenge.org/
License: CC-BY-NC-SA 4.0
EOM
