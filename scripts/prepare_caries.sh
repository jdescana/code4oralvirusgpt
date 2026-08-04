#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-./data/caries}"
mkdir -p "$DEST"

cat <<EOM
[caries] Place dataset at $DEST and write a manifest.json describing splits.
Source: Zenodo annotated caries dataset
License: open
EOM
