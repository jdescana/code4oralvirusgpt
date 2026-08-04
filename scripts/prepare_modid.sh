#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-./data/modid}"
mkdir -p "$DEST"

cat <<EOM
[modid] Place dataset at $DEST and write a manifest.json describing splits.
Source: Dryad / Zenodo multispectral oral disease image dataset
License: TBD
EOM
