#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-./data/tufts}"
mkdir -p "$DEST"

cat <<EOM
[tufts] Place dataset at $DEST and write a manifest.json describing splits.
Source: https://tdd.ece.tufts.edu/ (request access)
License: TBD
EOM
