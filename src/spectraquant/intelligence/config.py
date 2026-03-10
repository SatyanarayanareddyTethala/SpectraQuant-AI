"""Configuration loading for the Intelligence layer.

Reads ``config/intelligence.yaml`` (searched upward from CWD) and resolves
``*_env`` keys to environment variables so that secrets never appear in source.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MarketConfig:
    timezone: str = "America/New_York"
    open_time: str = "09:30"
    close_time: str = "16:00"
    premarket_offset_minutes: int = 60
    universe_file: str = "universe.md"


@dataclass
class DatabaseConfig:
    url: str = "sqlite:///intelligence.db"
    url_env: str = ""
    pool_size: int = 5
    max_overflow: int = 10


@dataclass
class NewsConfig:
    provider: str = "rss"
    api_key_env: str = ""
    feeds: list = field(default_factory=list)
    risk_score_threshold: float = 0.7


@dataclass
class EmailConfig:
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    username_env: str = "SMTP_USERNAME"
    password_env: str = "SMTP_PASSWORD"
    from_addr: str = ""
    to_addrs: list = field(default_factory=list)
    enabled: bool = False


@dataclass
class RiskConfig:
    equity_base: float = 100_000.0
    alpha_risk_fraction: float = 0.01
    beta_max: float = 500.0
    adv_participation_cap: float = 0.02
    max_daily_loss: float = 2_000.0
    max_gross_exposure: float = 1.0
    max_name_exposure: float = 0.10
    max_sector_exposure: float = 0.30


@dataclass
class CostsConfig:
    commission_per_share: float = 0.005
    base_slippage_bps: float = 2.0
    spread_weight: float = 0.5
    volatility_weight: float = 0.3
    participation_weight: float = 0.2


@dataclass
class SimulationConfig:
    enabled: bool = True
    seed: int = 42


@dataclass
class ExecutionConfig:
    live_enabled: bool = False
    broker: str = "paper"
    api_key_env: str = ""


@dataclass
class LearningConfig:
    retrain_day: str = "Sunday"
    retrain_hour: int = 18
    lookback_days: int = 252
    min_samples: int = 100


@dataclass
class SchedulerConfig:
    premarket_cron: str = "0 28 8 * * 1-5"
    hourly_news_cron: str = "0 5 * * * 1-5"
    intraday_interval_seconds: int = 60
    nightly_cron: str = "0 0 17 * * 1-5"
    active_start: str = "09:30"
    active_end: str = "16:00"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


@dataclass
class IntelligenceConfig:
    market: MarketConfig = field(default_factory=MarketConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from *start* until we find a directory containing
    ``config/`` or ``.git``.  Falls back to CWD."""
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / ".git").exists() or (parent / "config").is_dir():
            return parent
    return Path.cwd()


def _resolve_env(data: dict) -> dict:
    """For every key ending in ``_env`` whose value is a non-empty string,
    replace the *base* key with the environment variable's value."""
    resolved: Dict[str, Any] = {}
    for key, value in data.items():
        if key.endswith("_env") and isinstance(value, str) and value:
            base_key = key[:-4]  # strip '_env'
            env_val = os.environ.get(value, "")
            if env_val:
                resolved[base_key] = env_val
            # keep the _env key for reference
            resolved[key] = value
        elif isinstance(value, dict):
            resolved[key] = _resolve_env(value)
        else:
            resolved[key] = value
    return resolved


def _dict_to_dataclass(cls: type, data: dict) -> Any:
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: Optional[str] = None) -> IntelligenceConfig:
    """Load intelligence configuration from YAML with env-var resolution.

    Parameters
    ----------
    path : str, optional
        Explicit path to the YAML file.  When *None* the loader searches
        for ``config/intelligence.yaml`` relative to the repository root.

    Returns
    -------
    IntelligenceConfig
    """
    # Best-effort .env loading
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
        load_dotenv()
    except ImportError:
        pass

    if path is None:
        root = _find_repo_root()
        path = str(root / "config" / "intelligence.yaml")

    cfg_path = Path(path)
    if not cfg_path.exists():
        logger.warning("Config file %s not found — using defaults", cfg_path)
        return IntelligenceConfig()

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("PyYAML not installed — using defaults")
        return IntelligenceConfig()

    with open(cfg_path, "r") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    raw = _resolve_env(raw)

    section_map = {
        "market": MarketConfig,
        "database": DatabaseConfig,
        "news": NewsConfig,
        "email": EmailConfig,
        "risk": RiskConfig,
        "costs": CostsConfig,
        "simulation": SimulationConfig,
        "execution": ExecutionConfig,
        "learning": LearningConfig,
        "scheduler": SchedulerConfig,
        "logging": LoggingConfig,
    }

    kwargs: Dict[str, Any] = {}
    for section, cls in section_map.items():
        section_data = raw.get(section, {})
        if isinstance(section_data, dict):
            kwargs[section] = _dict_to_dataclass(cls, section_data)
        else:
            kwargs[section] = cls()

    return IntelligenceConfig(**kwargs)
