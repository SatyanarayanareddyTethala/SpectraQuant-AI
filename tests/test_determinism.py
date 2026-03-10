from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _run_replay(tmp_path: Path, run_id: str) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.yaml"
    universe_path = tmp_path / "universe_small.csv"
    prices_dir = tmp_path / "data" / "prices"
    config_path.write_text((FIXTURES / "config.yaml").read_text())
    universe_path.write_text((FIXTURES / "universe_small.csv").read_text())
    prices_dir.mkdir(parents=True, exist_ok=True)
    for file in (FIXTURES / "prices").glob("*.csv"):
        (prices_dir / file.name).write_text(file.read_text())
    intraday_dir = prices_dir / "intraday"
    intraday_dir.mkdir(parents=True, exist_ok=True)
    for file in (FIXTURES / "prices" / "intraday").glob("*.csv"):
        (intraday_dir / file.name).write_text(file.read_text())

    subprocess.check_call(
        [
            "python",
            str(ROOT / "scripts" / "replay_run.py"),
            "--config",
            str(config_path),
            "--universe",
            str(universe_path),
            "--prices-dir",
            str(prices_dir),
        ],
        cwd=tmp_path,
    )
    manifests = sorted((tmp_path / "reports" / "manifests").glob("replay_manifest_*.json"))
    assert manifests, "Replay manifest not generated"
    return json.loads(manifests[-1].read_text())


def test_replay_determinism(tmp_path: Path) -> None:
    manifest1 = _run_replay(tmp_path / "run1", "run1")
    manifest2 = _run_replay(tmp_path / "run2", "run2")
    assert manifest1["outputs"] == manifest2["outputs"]
