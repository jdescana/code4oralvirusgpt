#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-./data/code}"
mkdir -p "$DEST"

cat <<EOM
[code] Place dataset at $DEST and write a manifest.json describing splits.
Source: HuggingFace dataset OralVirusGPT/CODe (DOI 10.57967/hf/6421)
License: CC-BY
EOM
