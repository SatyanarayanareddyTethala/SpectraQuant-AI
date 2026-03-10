#!/usr/bin/env bash
set -euo pipefail

python --version
pip --version

pip freeze | head -n 30 || true

git status -sb
git rev-parse --abbrev-ref HEAD

for dir in models data reports logs; do
  if [ -d "$dir" ]; then
    echo "Listing $dir/"
    ls -lah "$dir" | head -n 20
  else
    echo "$dir/ (missing)"
  fi
done
