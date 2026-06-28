#!/bin/bash
# Run the full pipeline from the repository root (package imports require it).
cd "$(dirname "$0")/.." || exit 1
python -m experts_fusion.run "$@"
