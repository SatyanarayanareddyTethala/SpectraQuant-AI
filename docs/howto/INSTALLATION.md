# Installation Guide - SpectraQuant-AI

Complete installation guide for SpectraQuant-AI with all dependencies and prerequisites.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Installation](#quick-installation)
- [Detailed Installation Steps](#detailed-installation-steps)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Optional Components](#optional-components)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before installing SpectraQuant-AI, ensure you have the following prerequisites installed on your system.

### Required Prerequisites

#### 1. Python 3.11 or Higher

**Recommended:** Python 3.11 or 3.12  
**Minimum:** Python 3.11  
**⚠️ Warning:** Python 3.13+ has limited wheel availability for scientific packages, especially on macOS Apple Silicon

**Check if Python is installed:**
```bash
python --version
# or
python3 --version
```

**Installation:**

- **Ubuntu/Debian:**
  ```bash
  sudo apt update
  sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
  ```

- **macOS (Homebrew):**
  ```bash
  brew install python@3.11
  # or
  brew install python@3.12
  ```

- **Windows:**
  Download and install from [python.org](https://www.python.org/downloads/)
  - During installation, check "Add Python to PATH"

#### 2. Git

**Check if Git is installed:**
```bash
git --version
```

**Installation:**

- **Ubuntu/Debian:**
  ```bash
  sudo apt install -y git
  ```

- **macOS (Homebrew):**
  ```bash
  brew install git
  ```
  
  Or install Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```

- **Windows:**
  Download from [git-scm.com](https://git-scm.com/download/win)

#### 3. pip (Python Package Manager)

Usually comes with Python. Verify:
```bash
pip --version
# or
pip3 --version
```

If not installed:
```bash
# Ubuntu/Debian
sudo apt install -y python3-pip

# macOS/Windows
# pip is typically included with Python
```

### Recommended Prerequisites

#### Build Tools (for compiling packages like LightGBM)

- **Ubuntu/Debian:**
  ```bash
  sudo apt update
  sudo apt install -y build-essential cmake libomp-dev
  ```

- **macOS (Homebrew):**
  ```bash
  brew install cmake libomp
  ```

- **Windows:**
  - Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
  - Or use [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install)

### Optional Prerequisites

#### PostgreSQL (for Intelligence Layer)

Required only if you plan to use the Intelligence Layer (trading assistant) with PostgreSQL.

- **Ubuntu/Debian:**
  ```bash
  sudo apt install -y postgresql postgresql-contrib
  sudo systemctl start postgresql
  sudo systemctl enable postgresql
  ```

- **macOS (Homebrew):**
  ```bash
  brew install postgresql
  brew services start postgresql
  ```

- **Windows:**
  Download from [postgresql.org](https://www.postgresql.org/download/windows/)

#### Docker & Docker Compose (for containerized deployment)

- **Ubuntu/Debian:**
  ```bash
  curl -fsSL https://get.docker.com -o get-docker.sh
  sudo sh get-docker.sh
  sudo usermod -aG docker $USER
  sudo apt install -y docker-compose-plugin
  ```

- **macOS:**
  Download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)

- **Windows:**
  Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)

---

## Quick Installation

### Automated Installation (Recommended)

For Linux and macOS, use the automated installation script:

```bash
# Clone repository
git clone https://github.com/satyanarayanar17-dev/SpectraQuant-AI.git
cd SpectraQuant-AI

# Run installation script
./install.sh
```

For Windows, use the batch script:

```cmd
REM Clone repository
git clone https://github.com/satyanarayanar17-dev/SpectraQuant-AI.git
cd SpectraQuant-AI

REM Run installation script
install.bat
```

The script will:
- Check prerequisites
- Create virtual environment
- Install dependencies
- Verify installation
- Provide next steps

### Manual Installation

#### Minimal Installation (Core Features Only)

```bash
# 1. Clone repository
git clone https://github.com/satyanarayanar17-dev/SpectraQuant-AI.git
cd SpectraQuant-AI

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# 4. Install core dependencies
pip install -r requirements.txt

# 5. Install package
pip install -e .

# 6. Verify installation
spectraquant --help
spectraquant doctor
```

### Full Installation (Core + Intelligence Layer)

```bash
# 1. Clone repository
git clone https://github.com/satyanarayanar17-dev/SpectraQuant-AI.git
cd SpectraQuant-AI

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# 4. Install core dependencies
pip install -r requirements.txt

# 5. Install intelligence layer dependencies (ARCHIVED — skip for active use)
# pip install -r trading_assistant/requirements.txt

# 6. Install package
pip install -e .

# 7. Verify installation
spectraquant --help
spectraquant doctor
```

---

## Detailed Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/satyanarayanar17-dev/SpectraQuant-AI.git
cd SpectraQuant-AI
```

### Step 2: Create Virtual Environment

A virtual environment isolates your Python packages from the system Python.

**Create virtual environment:**
```bash
python -m venv .venv
# or specify Python version explicitly
python3.11 -m venv .venv
```

**Activate virtual environment:**

- **Linux/macOS:**
  ```bash
  source .venv/bin/activate
  ```

- **Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

When activated, your command prompt should show `(.venv)` prefix.

### Step 3: Upgrade pip and Build Tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

This ensures you have the latest package management tools.

### Step 4: Install Core Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `PyYAML` - Configuration file parsing
- `yfinance` - Market data download
- `streamlit` - Dashboard framework
- `pyarrow` - Data serialization
- `pytest` - Testing framework
- `scikit-learn` - Machine learning
- `lightgbm` - Gradient boosting
- `transformers` - NLP models
- `torch` - Deep learning framework

**Installation time:** 5-15 minutes depending on your internet speed and system.

### Step 5: Install Intelligence Layer Dependencies (Archived — Skip)

> ⚠️ **The `trading_assistant/` application is archived and not actively maintained.**
> Skip this step for active V3 or V2 use.  See `trading_assistant/ARCHIVED.md`.

```bash
# ARCHIVED — do not run for active platform setup
# pip install -r trading_assistant/requirements.txt
```

### Step 6: Install SpectraQuant Package

```bash
pip install -e .
```

The `-e` flag installs in "editable" mode, meaning changes to source code are immediately reflected without reinstalling.

This installs the `spectraquant` command-line tool.

### Step 7: Verify Installation

```bash
# Check CLI is available
spectraquant --help

# Run environment diagnostics
spectraquant doctor

# Run tests
pytest -q
```

If `spectraquant doctor` shows all green checkmarks, you're ready to go!

---

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

**Complete installation script:**

```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip \
    git build-essential cmake libomp-dev

# Clone repository
git clone https://github.com/satyanarayanar17-dev/SpectraQuant-AI.git
cd SpectraQuant-AI

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
# pip install -r trading_assistant/requirements.txt  # ARCHIVED — skip

# Install package
pip install -e .

# Verify
spectraquant doctor
```

### macOS

#### Intel Mac

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11 git cmake libomp

# Clone repository
git clone https://github.com/satyanarayanar17-dev/SpectraQuant-AI.git
cd SpectraQuant-AI

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
# pip install -r trading_assistant/requirements.txt  # ARCHIVED — skip

# Install package
pip install -e .

# Verify
spectraquant doctor
```

#### Apple Silicon (M1/M2/M3)

**⚠️ Important:** Python 3.13+ may have issues with scientific packages on Apple Silicon. Use Python 3.11 or 3.12.

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11 (recommended)
brew install python@3.11

# Install build tools
brew install cmake libomp

# Clone repository
git clone https://github.com/satyanarayanar17-dev/SpectraQuant-AI.git
cd SpectraQuant-AI

# Create virtual environment with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate

# Verify Python version
python --version  # Should show 3.11.x

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
# pip install -r trading_assistant/requirements.txt  # ARCHIVED — skip

# Install package
pip install -e .

# Verify
spectraquant doctor
```

**If LightGBM fails to install:**
```bash
brew install libomp cmake
pip install --no-binary lightgbm lightgbm
```

### Windows

#### Using Windows (Native)

```powershell
# 1. Install Python 3.11 from https://www.python.org/downloads/
#    Make sure to check "Add Python to PATH" during installation

# 2. Open PowerShell or Command Prompt

# 3. Clone repository
git clone https://github.com/satyanarayanar17-dev/SpectraQuant-AI.git
cd SpectraQuant-AI

# 4. Create virtual environment
python -m venv .venv

# 5. Activate virtual environment
.venv\Scripts\Activate.ps1  # PowerShell
# or
.venv\Scripts\activate.bat  # Command Prompt

# 6. Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# 7. Install dependencies
pip install -r requirements.txt
# pip install -r trading_assistant\requirements.txt  # ARCHIVED — skip

# 8. Install package
pip install -e .

# 9. Verify
spectraquant --help
spectraquant doctor
```

#### Using WSL (Recommended for Windows)

Windows Subsystem for Linux provides better compatibility:

```bash
# 1. Install WSL2
wsl --install

# 2. Open WSL terminal and follow Linux instructions above
```

---

## Optional Components

### Intelligence Layer Setup (Archived)

> ⚠️ **The `trading_assistant/` standalone application is archived and not actively maintained.**
> The steps below are preserved for historical reference only.
> See `trading_assistant/ARCHIVED.md` for details.

For the automated trading assistant with scheduled workflows:

1. **Install dependencies** (if not done already):
   ```bash
   # ARCHIVED — do not run for active platform setup
   # pip install -r trading_assistant/requirements.txt
   ```

2. **Setup database:**
   
   **Option A: SQLite (for testing/development)**
   ```bash
   # No setup needed, uses local file
   ```
   
   **Option B: PostgreSQL (for production)**
   ```bash
   # Install PostgreSQL (see Prerequisites section)
   
   # Create database and user
   sudo -u postgres psql
   CREATE DATABASE trading_assistant;
   CREATE USER spectraquant_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE trading_assistant TO spectraquant_user;
   \q
   ```

3. **Run bootstrap wizard:**
   ```bash
   python scripts/bootstrap_intelligence.py
   ```
   
   This interactive wizard will:
   - Configure market settings (timezone, trading hours)
   - Setup database connection
   - Configure news sources (NewsAPI)
   - Setup email alerts (optional)
   - Set risk parameters
   - Initialize database tables

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

5. **Run database migrations:**
   ```bash
   alembic upgrade head
   ```

6. **Start the scheduler:**
   ```bash
   python -m spectraquant.intelligence.scheduler
   ```

7. **Verify:**
   ```bash
   curl http://localhost:8000/health
   ```

### Dashboard Setup

The Streamlit dashboard is included in core dependencies, no additional setup needed:

```bash
cd dashboard
streamlit run app.py
```

Access at: http://localhost:8501

### Docker Deployment (Optional)

If you prefer containerized deployment:

1. **Install Docker** (see Prerequisites section)

2. **Build and run:**
   ```bash
   docker-compose up -d
   ```

3. **Check status:**
   ```bash
   docker-compose ps
   ```

4. **View logs:**
   ```bash
   docker-compose logs -f
   ```

---

## Verification

After installation, verify everything is working correctly:

### 1. Check Python Version

```bash
python --version
```

Should show Python 3.11.x or 3.12.x

### 2. Verify Virtual Environment

```bash
which python  # Linux/macOS
where python  # Windows
```

Should show path inside `.venv` directory.

### 3. Check Installed Packages

```bash
pip list | grep -E "numpy|pandas|lightgbm|torch|fastapi"
```

Should show all key packages installed.

### 4. Test CLI

```bash
spectraquant --help
```

Should display help message with available commands.

### 5. Run Environment Diagnostics

```bash
spectraquant doctor
```

Should show:
- ✅ Python version check
- ✅ All dependency imports
- ✅ Platform information
- Warnings for any issues

### 6. Run Test Suite

```bash
# Quick test run
pytest -q

# Verbose output
pytest -v

# Specific test
pytest tests/test_golden_pipeline.py -v
```

### 7. Test Basic Workflow

```bash
# Download sample data
spectraquant download --help

# Check configuration
cat config.yaml
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Command not found: `spectraquant`

**Solution:**
```bash
# Make sure virtual environment is activated
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate.bat     # Windows (Command Prompt)

# Reinstall package
pip install -e .

# If still not working, use full path
python -m spectraquant.cli.main --help
```

#### Issue: `ImportError: No module named 'spectraquant'`

**Solution:**
```bash
pip install -e .
```

#### Issue: LightGBM build errors

**macOS:**
```bash
brew install libomp cmake
pip install --no-binary lightgbm lightgbm
```

**Linux:**
```bash
sudo apt install -y build-essential cmake libomp-dev
pip install --upgrade pip
pip install lightgbm
```

**Windows:**
Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) and retry.

#### Issue: PyTorch installation is very slow

**Solution:** Install CPU-only version (smaller, faster):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### Issue: Python 3.13+ on macOS Apple Silicon

**Symptoms:** Import errors, slow performance, build failures

**Solution:** Switch to Python 3.11:
```bash
# Remove existing virtual environment
rm -rf .venv

# Install Python 3.11
brew install python@3.11

# Create new virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Reinstall dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .

# Verify
spectraquant doctor
```

#### Issue: Permission denied errors (Linux/macOS)

**Solution:**
```bash
# Don't use sudo with pip in virtual environment
# Make sure virtual environment is activated
source .venv/bin/activate

# Check ownership
ls -la .venv/
# If owned by root, remove and recreate without sudo
```

#### Issue: PostgreSQL connection errors

**Check PostgreSQL is running:**
```bash
# Linux
sudo systemctl status postgresql

# macOS
brew services list

# Windows
# Check Windows Services
```

**Test connection:**
```bash
psql -h localhost -U postgres -d postgres
```

**Fix connection:**
```bash
# Linux/macOS
sudo -u postgres psql
ALTER USER postgres PASSWORD 'new_password';

# Update .env file with correct credentials
```

#### Issue: Virtual environment activation fails (Windows PowerShell)

**Error:** "execution of scripts is disabled on this system"

**Solution:**
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try activating again
.venv\Scripts\Activate.ps1
```

#### Issue: Git clone fails (authentication)

**Solution:**
```bash
# Use HTTPS with personal access token
git clone https://github.com/satyanarayanar17-dev/SpectraQuant-AI.git

# Or setup SSH keys
# See: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
```

### Getting Help

If you encounter issues not covered here:

1. **Check existing documentation:**
   - [README.md](README.md) - Main documentation
   - [DEPENDENCIES.md](DEPENDENCIES.md) - Detailed dependency information
   - [README_INTELLIGENCE.md](README_INTELLIGENCE.md) - Intelligence layer guide

2. **Run diagnostics:**
   ```bash
   spectraquant doctor
   ```

3. **Search existing issues:**
   - [GitHub Issues](https://github.com/satyanarayanar17-dev/SpectraQuant-AI/issues)

4. **Create a new issue:**
   - Include output of `spectraquant doctor`
   - Include error messages
   - Specify your OS and Python version

5. **Community discussions:**
   - [GitHub Discussions](https://github.com/satyanarayanar17-dev/SpectraQuant-AI/discussions)

---

## Next Steps

After successful installation:

1. **Configure the system:**
   ```bash
   # Review and edit configuration
   cat config.yaml
   ```

2. **Setup universe data:**
   - Download ticker universe files or create custom lists
   - See README.md for instructions

3. **Run first pipeline:**
   ```bash
   # Download market data
   spectraquant download
   
   # Run full pipeline
   spectraquant refresh
   ```

4. **Explore features:**
   ```bash
   # List all commands
   spectraquant --help
   
   # View documentation
   cat README.md
   ```

5. **Setup Intelligence Layer (optional):**
   ```bash
   # Run bootstrap wizard
   python scripts/bootstrap_intelligence.py
   ```

---

## Summary Checklist

Use this checklist to ensure all prerequisites are met:

- [ ] Python 3.11 or 3.12 installed
- [ ] Git installed
- [ ] pip installed and upgraded
- [ ] Virtual environment created and activated
- [ ] Build tools installed (for your platform)
- [ ] Core dependencies installed (`requirements.txt`)
- [ ] Intelligence Layer dependencies installed (optional)
- [ ] SpectraQuant package installed (`pip install -e .`)
- [ ] `spectraquant doctor` shows all green
- [ ] Tests pass (`pytest -q`)
- [ ] PostgreSQL installed and running (if using Intelligence Layer)
- [ ] Environment variables configured (`.env` file)

---

**Installation complete! Happy trading! 📈🚀**

For more information, see:
- [README.md](README.md) - Main documentation
- [DEPENDENCIES.md](DEPENDENCIES.md) - Detailed dependency list
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [README_INTELLIGENCE.md](README_INTELLIGENCE.md) - Intelligence layer guide
