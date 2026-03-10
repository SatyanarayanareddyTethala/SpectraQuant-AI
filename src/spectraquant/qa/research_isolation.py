"""Production/research isolation checks."""
from __future__ import annotations

import ast
from pathlib import Path


def check_no_research_imports(root: Path) -> None:
    violations: list[str] = []
    for path in root.rglob("*.py"):
        if "research" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("spectraquant.research"):
                    violations.append(f"{path}: {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("spectraquant.research"):
                        violations.append(f"{path}: {alias.name}")
    if violations:
        raise AssertionError("Research imports found in production code:\n" + "\n".join(violations))
