#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-}"

if [[ -z "${VERSION}" ]]; then
  echo "Usage: $0 <version>"
  exit 1
fi

python - <<PY
from pathlib import Path

root = Path("${ROOT_DIR}")
pyproject = root / "pyproject.toml"
package_init = root / "src" / "spectraquant" / "__init__.py"
changelog = root / "CHANGELOG.md"
version = "${VERSION}"

py_lines = pyproject.read_text().splitlines()
in_project = False
out_lines = []
for line in py_lines:
    stripped = line.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        in_project = stripped == "[project]"
        out_lines.append(line)
        continue
    if in_project and stripped.startswith("version"):
        out_lines.append(f'version = "{version}"')
    else:
        out_lines.append(line)
pyproject.write_text("\n".join(out_lines) + "\n")

init_lines = package_init.read_text().splitlines()
updated = []
for line in init_lines:
    if line.startswith("__version__"):
        updated.append(f'__version__ = "{version}"')
    else:
        updated.append(line)
package_init.write_text("\n".join(updated) + "\n")

if changelog.exists():
    content = changelog.read_text()
    if f"[{version}]" not in content:
        raise SystemExit(f"CHANGELOG.md missing entry for [{version}].")
PY

python "${ROOT_DIR}/scripts/release_validate.py" --expected-version "${VERSION}"

if [[ -d "${ROOT_DIR}/models/latest" ]]; then
  mkdir -p "${ROOT_DIR}/models/promoted"
  if [[ -f "${ROOT_DIR}/models/latest/model.txt" ]]; then
    cp "${ROOT_DIR}/models/latest/model.txt" "${ROOT_DIR}/models/promoted/model.txt"
  fi
  if [[ -f "${ROOT_DIR}/models/latest/model.pkl" ]]; then
    cp "${ROOT_DIR}/models/latest/model.pkl" "${ROOT_DIR}/models/promoted/model.pkl"
  fi
fi

git add "${ROOT_DIR}/pyproject.toml" "${ROOT_DIR}/src/spectraquant/__init__.py" "${ROOT_DIR}/CHANGELOG.md"

if git diff --cached --quiet; then
  echo "No version changes detected."
else
  git commit -m "chore: release ${VERSION}"
fi

git tag -a "v${VERSION}" -m "v${VERSION}"
echo "Release tag v${VERSION} created."
