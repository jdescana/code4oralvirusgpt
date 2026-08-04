#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-./data/mmoral}"
mkdir -p "$DEST"

cat <<EOM
[mmoral] Place dataset at $DEST and write a manifest.json describing splits.
Source: https://github.com/isbrycee/OralGPT
License: TBD
EOM
