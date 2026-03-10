# Getting Started with SpectraQuant-AI

**Complete guide to get up and running quickly.**

---

## What You Need

### Required
- **Python 3.11+** (3.11 or 3.12 recommended)
- **Git** (for cloning repository)
- **Internet connection** (for downloading dependencies and data)

### Recommended
- **Build tools** (for compiling LightGBM)
  - Linux: `build-essential`, `cmake`, `libomp-dev`
  - macOS: `cmake`, `libomp` (via Homebrew)
  - Windows: Visual Studio Build Tools

### Optional
- **PostgreSQL** (for Intelligence Layer with database)
- **Docker** (for containerized deployment)

---

## Installation Options

### Option 1: Automated Install (Recommended)

**Easiest method** - One command does everything:

**Linux/macOS:**
```bash
git clone https://github.com/satyanarayanar17-dev/SpectraQuant-AI.git
cd SpectraQuant-AI
./install.sh
```

**Windows:**
```cmd
git clone https://github.com/satyanarayanar17-dev/SpectraQuant-AI.git
cd SpectraQuant-AI
install.bat
```

The script will:
1. ✅ Check prerequisites (Python, Git)
2. ✅ Create virtual environment
3. ✅ Install core dependencies
4. ✅ Optionally install Intelligence Layer
5. ✅ Verify installation
6. ✅ Show next steps

⏱️ **Time:** 5-15 minutes

### Option 2: Manual Install

**For more control:**

```bash
# 1. Clone
git clone https://github.com/satyanarayanar17-dev/SpectraQuant-AI.git
cd SpectraQuant-AI

# 2. Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .

# 4. Verify
spectraquant doctor
```

⏱️ **Time:** 5-15 minutes

### Option 3: Docker

**For containerized deployment:**

```bash
git clone https://github.com/satyanarayanar17-dev/SpectraQuant-AI.git
cd SpectraQuant-AI
docker-compose up -d
```

⏱️ **Time:** 10-20 minutes (including Docker image build)

---

## First Steps After Installation

### 1. Activate Environment

Every time you work with SpectraQuant:

```bash
cd SpectraQuant-AI
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Check Installation

```bash
spectraquant doctor
```

Should show ✅ for all checks.

### 3. Review Configuration

```bash
cat config.yaml
```

Edit as needed for your use case.

### 4. Download Data

```bash
spectraquant download
```

Downloads market data for configured universe.

### 5. Run Pipeline

```bash
spectraquant refresh
```

Runs complete pipeline: download → train → predict → signals → portfolio

---

## What Gets Installed?

### Core Components

**Python Packages** (from `requirements.txt`):
- `numpy`, `pandas` - Data processing
- `yfinance` - Market data
- `scikit-learn`, `lightgbm` - Machine learning
- `torch`, `transformers` - Deep learning
- `streamlit` - Dashboard
- `pytest` - Testing

**Size:** ~2-3 GB

### Intelligence Layer (Optional)

**Python Packages** (from `trading_assistant/requirements.txt`):
- `SQLAlchemy`, `alembic` - Database
- `fastapi`, `uvicorn` - Web API
- `APScheduler` - Job scheduling
- `sentence-transformers` - NLP
- `xgboost` - Additional ML

**Additional Size:** ~1-2 GB

---

## Verification Checklist

After installation, verify:

- [ ] `python --version` shows 3.11+
- [ ] Virtual environment activated (see `(.venv)` in prompt)
- [ ] `spectraquant --help` works
- [ ] `spectraquant doctor` shows all green ✅
- [ ] `pytest -q` passes (or shows expected failures)
- [ ] Can view `config.yaml`
- [ ] Can run `spectraquant download --help`

If all checks pass, you're ready to go! 🚀

---

## Common Issues & Quick Fixes

### Command not found: spectraquant
```bash
# Fix: Install package
pip install -e .
```

### Python version too old
```bash
# Fix: Install Python 3.11 or 3.12
# See INSTALLATION.md for platform-specific instructions
```

### LightGBM build errors (macOS)
```bash
# Fix: Install dependencies
brew install libomp cmake
pip install --no-binary lightgbm lightgbm
```

### Import errors
```bash
# Fix: Activate environment and reinstall
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Need Python 3.11 on macOS Apple Silicon
```bash
# Fix: Install specific version
brew install python@3.11
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## Next Steps

### Learn the Basics

1. **Read documentation:**
   - [README.md](README.md) - Full documentation
   - [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command reference

2. **Explore commands:**
   ```bash
   spectraquant --help
   spectraquant download --help
   spectraquant train --help
   ```

3. **Run tests:**
   ```bash
   pytest -v
   ```

### Setup Intelligence Layer (Optional)

For automated trading workflows:

```bash
# 1. Install dependencies (if not done)
pip install -r trading_assistant/requirements.txt

# 2. Run bootstrap wizard
python scripts/bootstrap_intelligence.py

# 3. Start scheduler
python -m spectraquant.intelligence.scheduler
```

### Launch Dashboard

```bash
cd dashboard
streamlit run app.py
```

Visit: http://localhost:8501

### Try a Sample Workflow

```bash
# 1. Configure universe
nano config.yaml  # or your text editor

# 2. Download data
spectraquant download

# 3. Run full pipeline
spectraquant refresh

# 4. View results
ls reports/
```

---

## Documentation Guide

### For Installation Help
- **[INSTALLATION.md](INSTALLATION.md)** - Complete installation guide
  - Prerequisites checklist
  - Platform-specific instructions
  - Troubleshooting

### For Quick Reference
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command cheat sheet
  - Common commands
  - Configuration snippets
  - Quick fixes

### For Dependency Details
- **[DEPENDENCIES.md](DEPENDENCIES.md)** - Detailed dependency list
  - System requirements
  - Python packages
  - Version requirements

### For Using the System
- **[README.md](README.md)** - Main documentation
  - Features overview
  - Complete usage guide
  - Configuration reference

### For Advanced Features
- **[README_INTELLIGENCE.md](README_INTELLIGENCE.md)** - Intelligence Layer
- **[SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)** - System architecture
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contributing guidelines

---

## Getting Help

**If you get stuck:**

1. **Run diagnostics:**
   ```bash
   spectraquant doctor
   ```

2. **Check documentation:**
   - This file for getting started
   - [INSTALLATION.md](INSTALLATION.md) for installation issues
   - [README.md](README.md) for usage questions

3. **Search existing issues:**
   - [GitHub Issues](https://github.com/satyanarayanar17-dev/SpectraQuant-AI/issues)

4. **Create new issue:**
   - Include output of `spectraquant doctor`
   - Include error messages
   - Specify OS and Python version

5. **Join discussions:**
   - [GitHub Discussions](https://github.com/satyanarayanar17-dev/SpectraQuant-AI/discussions)

---

## Installation Time Estimates

| Component | Time | Size |
|-----------|------|------|
| Prerequisites check | < 1 min | - |
| Repository clone | 1-2 min | ~50 MB |
| Virtual environment | < 1 min | ~20 MB |
| Core dependencies | 5-10 min | 2-3 GB |
| Intelligence Layer | 3-5 min | 1-2 GB |
| Verification | 1-2 min | - |
| **Total (Core only)** | **7-15 min** | **~2-3 GB** |
| **Total (Full)** | **10-20 min** | **~3-5 GB** |

*Times vary based on internet speed and system performance*

---

## Platform-Specific Notes

### Linux (Ubuntu/Debian)
✅ Best supported platform  
✅ All features work out of the box  
📝 Install build tools: `sudo apt install build-essential cmake libomp-dev`

### macOS (Intel)
✅ Fully supported  
📝 Install via Homebrew recommended  
📝 Install build tools: `brew install cmake libomp`

### macOS (Apple Silicon)
⚠️ Use Python 3.11 or 3.12 (not 3.13+)  
📝 May need to build LightGBM from source  
📝 Follow Apple Silicon section in INSTALLATION.md

### Windows
✅ Works with manual installation  
⚠️ WSL recommended for best experience  
📝 Visual Studio Build Tools may be needed

---

## Ready to Start?

1. ✅ **Read** this guide (you're doing it!)
2. ✅ **Choose** installation method (automated recommended)
3. ✅ **Install** following the steps above
4. ✅ **Verify** with `spectraquant doctor`
5. ✅ **Explore** with `spectraquant --help`
6. ✅ **Learn** from [README.md](README.md)
7. ✅ **Build** your first trading strategy!

---

**Welcome to SpectraQuant-AI! Happy trading! 📈🚀**
