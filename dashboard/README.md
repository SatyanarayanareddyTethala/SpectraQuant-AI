# SpectraQuant Dashboard

## Setup (macOS or Linux)

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the pipeline (download → build → train → predict)

```bash
python -m src.spectraquant.cli.main download
python -m src.spectraquant.cli.main build-dataset
python -m src.spectraquant.cli.main train
python -m src.spectraquant.cli.main predict
```

Optional artifacts used in Operations:

```bash
python -m src.spectraquant.cli.main signals
python -m src.spectraquant.cli.main portfolio
```

## Run the dashboard

```bash
streamlit run dashboard/app.py --server.port 8501
```

Open <http://localhost:8501> in your browser.

## Required artifacts

The dashboard reads from pipeline outputs if present. Missing artifacts are handled gracefully and surfaced with the CLI command that generates them.

- **Predictions**: `reports/predictions/predictions_*.csv`
- **Signals**: `reports/signals/*.csv`
- **Portfolio weights**: `reports/portfolio/*weights*.csv`
- **Portfolio returns**: `reports/portfolio/*returns*.csv`
- **Model metadata**: `models/training_metadata.json`
- **Price history**: `data/prices/<TICKER>.parquet` or `.csv`
- **Config universe**: `config.yaml` (resolved into `data.tickers` / `universe.tickers`)

## Notes

- The main dashboard KPIs and Investment Simulator are driven by the latest predictions and price history.
- If only 1-day predictions are available, longer horizons are compounded from 1-day returns and labeled accordingly.
