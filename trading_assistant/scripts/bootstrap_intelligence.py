#!/usr/bin/env python3
"""
Wrapper script for bootstrap_intelligence.py

This wrapper allows running the bootstrap script from either:
  - Repository root: python scripts/bootstrap_intelligence.py
  - trading_assistant dir: python scripts/bootstrap_intelligence.py
  - From anywhere: python trading_assistant/scripts/bootstrap_intelligence.py

The actual bootstrap script is located at: scripts/bootstrap_intelligence.py (repo root)
"""
import sys
from pathlib import Path

# Find the repository root
script_path = Path(__file__).resolve()
repo_root = script_path.parent.parent.parent

# The real bootstrap script location
real_bootstrap = repo_root / "scripts" / "bootstrap_intelligence.py"

if not real_bootstrap.exists():
    print(f"Error: Bootstrap script not found at {real_bootstrap}")
    print(f"Expected location: {repo_root}/scripts/bootstrap_intelligence.py")
    sys.exit(1)

# Add repo root to path
sys.path.insert(0, str(repo_root))

# Execute the real bootstrap script
with open(real_bootstrap) as f:
    code = compile(f.read(), str(real_bootstrap), 'exec')
    exec(code, {'__file__': str(real_bootstrap)})
