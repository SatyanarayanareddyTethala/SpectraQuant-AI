# Dependencies Reference - SpectraQuant-AI

Complete reference of all dependencies required for SpectraQuant-AI with installation commands.

> 📘 **For step-by-step installation guide, see [INSTALLATION.md](INSTALLATION.md)**

---

## Quick Reference

### What You Need

**Prerequisites:**
- Python 3.11+ (Python 3.11 or 3.12 recommended)
- Git
- pip (Python package manager)
- Build tools (gcc, cmake, libomp)

**Python Packages:**
- Core: `pip install -r requirements.txt`
- Intelligence Layer (optional): `pip install -r trading_assistant/requirements.txt`
- Package: `pip install -e .`

**Optional:**
- PostgreSQL (for Intelligence Layer database)
- Docker & Docker Compose (for containerized deployment)

---

## Quick Summary

### Minimum Installation (Core Features Only)
```bash
git clone https://github.com/satyanarayanar17-dev/SpectraQuant-AI.git
cd SpectraQuant-AI
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Full Installation (Core + Intelligence Layer)
```bash
git clone https://github.com/satyanarayanar17-dev/SpectraQuant-AI.git
cd SpectraQuant-AI
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd trading_assistant
pip install -r requirements.txt
cd ..
pip install -e .
```

---

## Detailed Dependency List

### System Prerequisites

#### Python (Required)
- **Version**: Python 3.11 or higher

**Installation:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# macOS (Homebrew)
brew install python@3.11

# Windows - Download from https://www.python.org/downloads/
```

#### Git (Required)
**Installation:**
```bash
# Ubuntu/Debian
sudo apt install -y git

# macOS (Homebrew)
brew install git

# Windows - Download from https://git-scm.com/download/win
```

#### Build Tools (Recommended for some packages)
```bash
# Ubuntu/Debian
sudo apt install -y build-essential

# macOS - Xcode Command Line Tools
xcode-select --install
```

---

### Core Python Dependencies

These are installed via `pip install -r requirements.txt`:

1. **numpy** - Numerical computing library
2. **pandas** - Data manipulation and analysis
3. **PyYAML** - YAML parser and emitter
4. **yfinance** - Yahoo Finance market data downloader
5. **streamlit** - Web application framework for dashboards
6. **pyarrow** - Apache Arrow Python bindings for data serialization
7. **pytest** - Testing framework
8. **scikit-learn** - Machine learning library
9. **lightgbm** - Gradient boosting framework
10. **transformers** - Hugging Face transformers for NLP
11. **torch** - PyTorch deep learning framework

**Installation Command:**
```bash
pip install -r requirements.txt
```

**Or install individually:**
```bash
pip install numpy pandas PyYAML yfinance streamlit pyarrow pytest scikit-learn lightgbm transformers torch
```

---

### Intelligence Layer Python Dependencies

These are installed via `pip install -r trading_assistant/requirements.txt`:

#### Core
- **numpy>=1.24.0**
- **pandas>=2.0.0**
- **PyYAML>=6.0**

#### Database
- **SQLAlchemy>=2.0.0** - SQL toolkit and ORM
- **alembic>=1.12.0** - Database migrations
- **psycopg2-binary>=2.9.0** - PostgreSQL adapter

#### Machine Learning & AI
- **scikit-learn>=1.3.0**
- **lightgbm>=4.0.0**
- **xgboost>=2.0.0** - Extreme Gradient Boosting

#### NLP & Embeddings
- **sentence-transformers>=2.2.0** - Sentence embeddings
- **transformers>=4.30.0**
- **torch>=2.0.0**

#### Web API
- **fastapi>=0.104.0** - Modern web framework
- **uvicorn>=0.24.0** - ASGI server
- **streamlit>=1.28.0**
- **pydantic>=2.0.0** - Data validation

#### Scheduling
- **APScheduler>=3.10.0** - Advanced Python Scheduler

#### Email
- **jinja2>=3.1.0** - Template engine

#### Utilities
- **python-dotenv>=1.0.0** - Environment variable management
- **requests>=2.31.0** - HTTP library
- **pyarrow>=13.0.0**

#### Testing
- **pytest>=7.4.0**
- **pytest-asyncio>=0.21.0** - Async testing support

#### Market Data
- **yfinance>=0.2.0**

#### Monitoring
- **psutil>=5.9.0** - System and process utilities

**Installation Command:**
```bash
cd trading_assistant
pip install -r requirements.txt
```

**Or install individually:**
```bash
pip install numpy>=1.24.0 pandas>=2.0.0 PyYAML>=6.0 SQLAlchemy>=2.0.0 alembic>=1.12.0 psycopg2-binary>=2.9.0 scikit-learn>=1.3.0 lightgbm>=4.0.0 xgboost>=2.0.0 sentence-transformers>=2.2.0 transformers>=4.30.0 torch>=2.0.0 fastapi>=0.104.0 uvicorn>=0.24.0 streamlit>=1.28.0 pydantic>=2.0.0 APScheduler>=3.10.0 jinja2>=3.1.0 python-dotenv>=1.0.0 requests>=2.31.0 pyarrow>=13.0.0 pytest>=7.4.0 pytest-asyncio>=0.21.0 yfinance>=0.2.0 psutil>=5.9.0
```

---

### Optional System Dependencies

#### PostgreSQL Database (Required for Intelligence Layer)
```bash
# Ubuntu/Debian
sudo apt install -y postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql

# macOS (Homebrew)
brew install postgresql
brew services start postgresql

# Windows - Download from https://www.postgresql.org/download/windows/
```

#### Docker & Docker Compose (Optional, for containerized deployment)
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
sudo apt install -y docker-compose

# macOS (Homebrew)
brew install docker docker-compose

# Windows - Download Docker Desktop from https://www.docker.com/products/docker-desktop
```

---

## Platform-Specific Notes

### macOS (Apple Silicon)

Some packages may require additional setup on Apple Silicon Macs:

```bash
# Install libomp for LightGBM
brew install libomp

# If LightGBM installation fails, build from source
pip install --no-binary lightgbm lightgbm

# For faster PyTorch installation (CPU only)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Ubuntu/Debian

Additional build tools may be needed:
```bash
sudo apt install -y build-essential python3-dev libssl-dev libffi-dev
```

### Windows

- Use PowerShell or Command Prompt as Administrator
- Consider using Windows Subsystem for Linux (WSL) for better compatibility
- Virtual environment activation: `.venv\Scripts\activate` (not `source .venv/bin/activate`)

---

## Complete Installation Script

Here's a complete bash script that installs everything:

```bash
#!/bin/bash

# Exit on error
set -e

echo "==> Installing SpectraQuant-AI Dependencies"

# 1. Clone repository
echo "==> Step 1: Cloning repository..."
git clone https://github.com/satyanarayanar17-dev/SpectraQuant-AI.git
cd SpectraQuant-AI

# 2. Create virtual environment
echo "==> Step 2: Creating virtual environment..."
python3 -m venv .venv

# 3. Activate virtual environment
echo "==> Step 3: Activating virtual environment..."
source .venv/bin/activate

# 4. Upgrade pip
echo "==> Step 4: Upgrading pip..."
pip install --upgrade pip

# 5. Install core dependencies
echo "==> Step 5: Installing core dependencies..."
pip install -r requirements.txt

# 6. Install intelligence layer dependencies
echo "==> Step 6: Installing intelligence layer dependencies..."
cd trading_assistant
pip install -r requirements.txt
cd ..

# 7. Install package in editable mode
echo "==> Step 7: Installing SpectraQuant package..."
pip install -e .

# 8. Verify installation
echo "==> Step 8: Verifying installation..."
spectraquant --help

echo ""
echo "✅ Installation complete!"
echo ""
echo "To get started:"
echo "  1. Activate virtual environment: source .venv/bin/activate"
echo "  2. Run: spectraquant --help"
echo "  3. See README.md for usage instructions"
```

Save this as `install.sh`, make it executable with `chmod +x install.sh`, and run with `./install.sh`.

---

## Verification Commands

After installation, verify everything is working:

```bash
# Check Python version
python --version

# Verify virtual environment is activated
which python  # Should show path to .venv

# Check installed packages
pip list | grep -E "numpy|pandas|lightgbm|torch|fastapi"

# Test CLI
spectraquant --help

# Run tests
pytest

# Test API (if intelligence layer is running)
curl http://localhost:8000/health
```

---

## Troubleshooting

### Common Issues

**Issue: `ImportError: No module named 'spectraquant'`**
```bash
pip install -e .
```

**Issue: LightGBM build errors**
```bash
# macOS
brew install libomp
pip install --no-binary lightgbm lightgbm

# Linux
sudo apt install -y build-essential
```

**Issue: Torch takes too long to install**
```bash
# Install CPU-only version (smaller, faster)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Issue: PostgreSQL connection errors**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql  # Linux
brew services list  # macOS

# Test connection
psql -h localhost -U spectraquant_user -d spectraquant
```

---

## Summary Table

| Component | Installation Command | Purpose |
|-----------|---------------------|---------|
| Python 3.11+ | `sudo apt install python3.11` | Runtime environment |
| Git | `sudo apt install git` | Version control |
| Core deps | `pip install -r requirements.txt` | Basic functionality |
| Intelligence deps | `pip install -r trading_assistant/requirements.txt` | Advanced features |
| Package | `pip install -e .` | Install CLI tools |
| PostgreSQL | `sudo apt install postgresql` | Database (optional) |
| Docker | `curl -fsSL https://get.docker.com | sh` | Containerization (optional) |

---

## Next Steps

After installing all dependencies:

1. **Configure the system**: Edit `config.yaml`
2. **Setup universe data**: Add ticker files to `data/universe/`
3. **Download data**: Run `spectraquant download`
4. **Run pipeline**: Execute `spectraquant refresh`
5. **Launch dashboard**: Run `streamlit run dashboard/app.py`

For more information:
- [INSTALLATION.md](INSTALLATION.md) - Detailed installation guide
- [README.md](README.md) - Main documentation
- [README_INTELLIGENCE.md](README_INTELLIGENCE.md) - Intelligence layer guide

---

**Installation complete! Happy trading! 📈🚀**
