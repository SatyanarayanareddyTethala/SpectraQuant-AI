"""Optional dependency management with lazy imports and graceful fallbacks."""
from __future__ import annotations

import sys
from typing import Any


class MissingDependencyError(RuntimeError):
    """Raised when an optional dependency is required but not available."""
    pass


def require_lightgbm() -> Any:
    """
    Import and return lightgbm module, or raise helpful error if not available.
    
    Returns:
        lightgbm module
        
    Raises:
        MissingDependencyError: If lightgbm cannot be imported
    """
    try:
        import lightgbm
        return lightgbm
    except ImportError as e:
        raise MissingDependencyError(
            "LightGBM is required for training and prediction commands.\n"
            "Install with: pip install lightgbm\n"
            "\n"
            "Note: On macOS Apple Silicon with Python 3.13+, you may encounter issues.\n"
            "Recommended: Use Python 3.11 or 3.12 instead.\n"
            "  brew install python@3.11\n"
            "  python3.11 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  pip install -e .\n"
        ) from e


def require_sklearn() -> Any:
    """
    Import and return sklearn module, or raise helpful error if not available.
    
    Returns:
        sklearn module
        
    Raises:
        MissingDependencyError: If sklearn cannot be imported
    """
    try:
        import sklearn
        return sklearn
    except ImportError as e:
        raise MissingDependencyError(
            "scikit-learn is required for training and prediction commands.\n"
            "Install with: pip install scikit-learn\n"
            "\n"
            "Note: On macOS Apple Silicon with Python 3.13+, you may encounter issues.\n"
            "Recommended: Use Python 3.11 or 3.12 instead.\n"
        ) from e


def check_optional_dependency(name: str) -> tuple[bool, str | None]:
    """
    Check if an optional dependency is available.
    
    Args:
        name: Name of the dependency to check
        
    Returns:
        Tuple of (is_available, error_message)
    """
    try:
        if name == "lightgbm":
            import lightgbm
            return True, None
        elif name == "sklearn" or name == "scikit-learn":
            import sklearn
            return True, None
        elif name == "scipy":
            import scipy
            return True, None
        elif name == "numpy":
            import numpy
            return True, None
        elif name == "pandas":
            import pandas
            return True, None
        elif name == "yaml":
            import yaml
            return True, None
        elif name == "transformers":
            import transformers
            return True, None
        elif name == "torch":
            import torch
            return True, None
        else:
            return False, f"Unknown dependency: {name}"
    except ImportError as e:
        return False, str(e)


def get_dependency_status() -> dict[str, tuple[bool, str | None]]:
    """
    Get status of all important dependencies.
    
    Returns:
        Dictionary mapping dependency name to (is_available, error_message)
    """
    deps = [
        "numpy",
        "pandas", 
        "yaml",
        "lightgbm",
        "sklearn",
        "scipy",
        "transformers",
        "torch",
    ]
    
    return {dep: check_optional_dependency(dep) for dep in deps}
