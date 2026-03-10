import json
from pathlib import Path

import pandas as pd

from spectraquant.dataset.builder import build_dataset
from spectraquant.dataset.io import load_dataset


def test_dataset_builder_ordering(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prices_dir = Path("data/prices")
    prices_dir.mkdir(parents=True)

    dates = pd.date_range("2023-01-01", periods=80, freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": 100 + pd.Series(range(80)).values,
            "high": 101 + pd.Series(range(80)).values,
            "low": 99 + pd.Series(range(80)).values,
            "close": 100 + pd.Series(range(80)).values,
            "volume": 1000 + pd.Series(range(80)).values,
        }
    )
    df.to_csv(prices_dir / "TEST.csv", index=False)

    dataset_path = build_dataset(["TEST"], start="2023-01-01", end="2023-12-31", horizons=[5, 20])
    assert dataset_path.exists()

    dataset = load_dataset(dataset_path)
    assert isinstance(dataset.index, pd.MultiIndex)
    dates_index = dataset.index.get_level_values(0)
    assert dates_index.is_monotonic_increasing


def test_dataset_builder_falls_back_to_csv_when_parquet_unavailable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prices_dir = Path("data/prices")
    prices_dir.mkdir(parents=True)

    dates = pd.date_range("2023-01-01", periods=80, freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": 100 + pd.Series(range(80)).values,
            "high": 101 + pd.Series(range(80)).values,
            "low": 99 + pd.Series(range(80)).values,
            "close": 100 + pd.Series(range(80)).values,
            "volume": 1000 + pd.Series(range(80)).values,
        }
    )
    df.to_csv(prices_dir / "TEST.csv", index=False)

    def _raise_import_error(*args, **kwargs):
        raise ImportError("pyarrow not installed")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _raise_import_error)

    dataset_path = build_dataset(["TEST"], start="2023-01-01", end="2023-12-31", horizons=[5, 20])
    assert dataset_path.suffix == ".csv"
    assert dataset_path.exists()

    dataset = load_dataset(dataset_path)
    assert isinstance(dataset.index, pd.MultiIndex)

    manifest_paths = list(Path("reports/run").glob("*/manifest.json"))
    assert manifest_paths
    payload = json.loads(manifest_paths[0].read_text())
    dataset_meta = payload["dataset"]
    assert dataset_meta["format"] == "csv"
    assert dataset_meta["path"] == str(dataset_path)
