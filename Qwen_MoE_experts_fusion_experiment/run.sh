#!/bin/bash
files=$(find . -maxdepth 1 -name "*.py" -type f | sed 's|^\./||' | sort)

while IFS= read -r file; do
    python "$file"
done <<< "$files"