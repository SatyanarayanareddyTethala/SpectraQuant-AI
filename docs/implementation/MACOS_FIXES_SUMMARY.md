# macOS Apple Silicon Runability Fixes - Summary

## Changes Made

### 1. Lazy Loading of ML Dependencies

**Files Modified:**
- `src/spectraquant/cli/main.py` - Removed top-level imports of `lightgbm` and `sklearn.metrics`
  - Moved imports inside functions that need them (`_train_gbdt_model`, `_load_lgbm_model`)
  - Changed type hints from specific types to `Any` to avoid import-time dependencies

**Files Created:**
- `src/spectraquant/utils/__init__.py` - New utils package
- `src/spectraquant/utils/optional_deps.py` - Helper module for lazy dependency loading
  - `require_lightgbm()` - Lazy import with helpful error messages
  - `require_sklearn()` - Lazy import with helpful error messages
  - `check_optional_dependency()` - Check if a dependency is available
  - `get_dependency_status()` - Get status of all important dependencies
  - `MissingDependencyError` - Custom exception for missing dependencies

**Result:** 
- `spectraquant --help` starts instantly without importing heavy ML libraries
- Commands that don't need ML (download, universe-stats, features, etc.) run without scipy/sklearn/lightgbm
- Train and predict commands load ML deps only when invoked
- Clear error messages guide users to install missing dependencies

### 2. Environment Doctor Command

**Added:** `cmd_doctor()` function in `src/spectraquant/cli/main.py`

**Features:**
- Detects Python version and warns about 3.13+ compatibility issues
- Identifies macOS Apple Silicon platform
- Tests imports of all core and optional dependencies
- Provides color-coded status output (✓ OK / ✗ MISSING)
- Gives actionable recommendations for fixing issues
- Platform-specific guidance (especially for macOS Apple Silicon)

**Usage:**
```bash
spectraquant doctor
```

### 3. Bootstrap Script Wrapper

**Files Created:**
- `trading_assistant/scripts/bootstrap_intelligence.py` - Wrapper script

**Features:**
- Works from any directory (repo root, trading_assistant/, or anywhere)
- Locates the real bootstrap script at `scripts/bootstrap_intelligence.py`
- Properly sets up Python path before execution
- Clear error messages if script not found

**Usage:**
```bash
# From repo root
python scripts/bootstrap_intelligence.py

# From trading_assistant directory
cd trading_assistant
python scripts/bootstrap_intelligence.py

# Direct path from anywhere
python trading_assistant/scripts/bootstrap_intelligence.py
```

### 4. Documentation Consolidation

**Files Removed:**
- `README_OLD.md` - Outdated documentation
- `QUICK_INSTALL.md` - Redundant quick reference
- `INSTALLATION.md` - Redundant installation guide

**Files Modified:**
- `README.md` - Comprehensive consolidated documentation with:
  - macOS Apple Silicon specific installation instructions
  - Python version recommendations (3.11/3.12, NOT 3.13+)
  - Doctor command documentation
  - Bootstrap script usage from multiple directories
  - Enhanced troubleshooting section
  - Updated Python badge to 3.11+

## Python Version Compatibility

### Recommended: Python 3.11 or 3.12

**Why NOT Python 3.13+:**
- Limited wheel availability for scientific packages
- SciPy, LightGBM, and other packages may require compilation from source
- On macOS Apple Silicon, this can fail or be very slow
- Python 3.13 is very new (released late 2024), ecosystem still catching up

**Why Python 3.11/3.12:**
- Excellent wheel availability for all dependencies
- Well-tested with scientific Python stack
- Fast installation on all platforms including Apple Silicon
- Recommended by most scientific Python projects

## Installation Commands for macOS Apple Silicon

### Quick Setup:

```bash
# 1. Install Python 3.11 via Homebrew
brew install python@3.11

# 2. Clone repository
git clone https://github.com/satyanarayanar17-dev/SpectraQuant-AI.git
cd SpectraQuant-AI

# 3. Create virtual environment with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 5. Verify installation
spectraquant --help
spectraquant doctor
```

### If LightGBM Build Fails:

```bash
# Install required tools
brew install libomp cmake

# Try building from source
pip install --no-binary lightgbm lightgbm
```

## Testing Results

### Manual Testing Performed:

1. ✅ `spectraquant --help` - Works instantly without importing ML libraries
2. ✅ `spectraquant doctor` - Successfully checks environment and provides recommendations
3. ✅ Bootstrap wrapper - Works from both repo root and trading_assistant directory
4. ✅ Lazy loading - Verified ML imports only happen in train/predict functions

### CI/CD Impact:

- No breaking changes to existing functionality
- All commands still work exactly as before
- Enhanced startup performance for help and non-ML commands
- Better error messages when dependencies missing

## Files Changed Summary

```
Modified:
- src/spectraquant/cli/main.py (lazy imports, doctor command, updated usage)
- README.md (consolidated docs, macOS instructions)

Created:
- src/spectraquant/utils/__init__.py
- src/spectraquant/utils/optional_deps.py
- trading_assistant/scripts/bootstrap_intelligence.py

Deleted:
- README_OLD.md
- QUICK_INSTALL.md
- INSTALLATION.md
```

## Developer Notes

### How Lazy Loading Works:

Before:
```python
from lightgbm import Booster, LGBMClassifier, LGBMRegressor, early_stopping
from sklearn.metrics import mean_squared_error, roc_auc_score
```

After:
```python
# No top-level imports

def _train_gbdt_model(dataset, label_col, config):
    from spectraquant.utils.optional_deps import require_lightgbm, require_sklearn
    lgb = require_lightgbm()  # Imports lightgbm only when function is called
    sklearn = require_sklearn()
    from sklearn.metrics import mean_squared_error, roc_auc_score
    
    model = lgb.LGBMClassifier(**params)  # Use via module reference
    # ... rest of function
```

### Benefits:
1. **Fast Startup**: Help and basic commands start in <1 second
2. **Graceful Degradation**: Missing deps only error when actually needed
3. **Clear Errors**: Helpful messages tell users exactly what to install
4. **Platform-Aware**: Recommendations tailored to user's platform

### Testing Lazy Loading:

To verify ML libraries aren't imported prematurely:
```python
import sys
sys.path.insert(0, 'src')

# This should be fast and not import lightgbm
from spectraquant.cli.main import main

# Check if lightgbm was imported
print('lightgbm' in sys.modules)  # Should be False

# Now run doctor (still shouldn't import lightgbm)
sys.argv = ['spectraquant', 'doctor']
main()

print('lightgbm' in sys.modules)  # Still False
```

## Known Limitations

1. **Other modules may still have top-level imports**: This fix only addresses the CLI entry point. Other modules like `spectraquant.analysis.feature_pruning` may still have direct imports.

2. **Python 3.13+ issues persist**: While we've improved the situation, users on Python 3.13+ may still face issues. The doctor command helps diagnose this.

3. **Type hints use `Any`**: To avoid import-time dependencies, some type hints were changed from specific types to `Any`. This is a trade-off for startup performance.

## Future Improvements

1. Extend lazy loading to other modules
2. Add more dependency checks to doctor command
3. Create pre-built wheels for common platforms
4. Add automated tests for lazy loading behavior
