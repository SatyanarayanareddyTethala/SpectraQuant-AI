# Quick Reference - SpectraQuant-AI

One-page reference for installation and common commands.

---

## Installation (3 Steps)

### 1. Prerequisites
- Python 3.11+ → [python.org](https://www.python.org/downloads/)
- Git → [git-scm.com](https://git-scm.com/downloads)

### 2. Install

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

**Manual:**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 3. Verify
```bash
spectraquant doctor
```

---

## Common Commands

### Setup
```bash
# Activate environment
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Check system
spectraquant doctor

# Configure
nano config.yaml
```

### Core Pipeline
```bash
# Full pipeline (all steps)
spectraquant refresh

# Or run individually:
spectraquant download        # 1. Download price data
spectraquant build-dataset   # 2. Build training dataset
spectraquant train           # 3. Train models
spectraquant predict         # 4. Generate predictions
spectraquant signals         # 5. Generate signals
spectraquant portfolio       # 6. Simulate portfolio
spectraquant execute         # 7. Execute paper trades
```

### Intelligence Layer
```bash
# Bootstrap
python scripts/bootstrap_intelligence.py

# Start scheduler
python -m spectraquant.intelligence.scheduler

# Check health
curl http://localhost:8000/health
```

### Dashboard
```bash
cd dashboard
streamlit run app.py
```

### Testing
```bash
pytest                       # All tests
pytest -q                    # Quiet mode
pytest tests/test_file.py    # Specific test
```

### Model Management
```bash
spectraquant list-models     # List trained models
spectraquant promote-model 5 # Promote model version
spectraquant retrain         # Trigger retraining
```

---

## File Structure

```
SpectraQuant-AI/
├── requirements.txt         # Core dependencies
├── trading_assistant/
│   └── requirements.txt    # Intelligence Layer deps
├── config.yaml             # Main configuration
├── .env                    # Environment variables (create from .env.example)
├── data/
│   ├── universe/          # Ticker lists
│   ├── prices/            # Downloaded data
│   └── processed/         # Processed datasets
├── models/                # Trained models
└── reports/              # Output artifacts
    ├── predictions/
    ├── signals/
    ├── portfolio/
    └── execution/
```

---

## Configuration Quick Reference

### config.yaml Key Sections

```yaml
universe:
  india:
    source: nse
    tickers_file: data/universe/nifty_500.csv

portfolio:
  rebalance: monthly
  weighting: equal
  top_k: 20
  horizon: "1d"

predictions:
  daily_horizons: [1d, 5d, 20d]
  intraday_horizons: [5m, 30m, 60m]

intraday:
  enabled: true
  signal_thresholds:
    buy: 0.6
    sell: 0.4
```

---

## Troubleshooting

### Can't find spectraquant command
```bash
pip install -e .
```

### Import errors
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### LightGBM build errors (macOS)
```bash
brew install libomp cmake
pip install --no-binary lightgbm lightgbm
```

### Python version issues (macOS Apple Silicon)
```bash
brew install python@3.11
python3.11 -m venv .venv
```

### Slow imports
Already fixed in latest version - CLI uses lazy loading

---

## Environment Variables (.env)

```bash
# Database (for Intelligence Layer)
DATABASE_URL=postgresql://user:pass@localhost:5432/trading_assistant

# NewsAPI
NEWSAPI_KEY=your_api_key

# Email (optional)
SMTP_USERNAME=your_email@example.com
SMTP_PASSWORD=your_password
```

---

## Getting Help

- **Documentation**: [README.md](README.md), [INSTALLATION.md](INSTALLATION.md)
- **Issues**: [github.com/satyanarayanar17-dev/SpectraQuant-AI/issues](https://github.com/satyanarayanar17-dev/SpectraQuant-AI/issues)
- **Diagnostics**: `spectraquant doctor`

---

## Quick Test

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Check system
spectraquant doctor

# 3. View help
spectraquant --help

# 4. Test download (dry run)
spectraquant download --help

# 5. Run tests
pytest -q
```

---

**For complete documentation, see:**
- [INSTALLATION.md](INSTALLATION.md) - Detailed installation guide
- [README.md](README.md) - Full documentation
- [DEPENDENCIES.md](DEPENDENCIES.md) - Dependency reference
