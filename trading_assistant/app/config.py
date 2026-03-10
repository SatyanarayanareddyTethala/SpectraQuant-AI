"""
Configuration management for trading assistant.
Loads and validates YAML configuration.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dataclasses import dataclass, field


@dataclass
class MarketConfig:
    """Market timing configuration"""
    timezone: str
    open_time: str
    close_time: str
    premarket_plan_offset_minutes: int
    intraday_poll_seconds: int


@dataclass
class UniverseConfig:
    """Universe configuration"""
    source: str
    symbols: list
    tickers_file: Optional[str]
    min_adv: float
    max_spread_bps: float
    min_market_cap: Optional[float] = None


@dataclass
class CostsConfig:
    """Transaction costs configuration"""
    commission_per_trade: float
    slippage_model: Dict[str, float]


@dataclass
class RiskConfig:
    """Risk management configuration"""
    equity_base: float
    alpha_risk_fraction: float
    b_max: float
    max_daily_loss: float
    max_gross_exposure: float
    max_name_exposure: float
    max_sector_exposure: float
    turnover_cap: float
    adv_participation_cap: float
    mae_threshold: float


@dataclass
class NewsConfig:
    """News configuration"""
    providers: list
    refresh_minutes: int
    risk_tags_thresholds: Dict[str, list]
    embedding_model: str
    min_relevance_score: float


@dataclass
class EmailConfig:
    """Email notification configuration"""
    smtp_host: str
    smtp_port: int
    use_tls: bool
    username_env: str
    password_env: str
    from_email: str
    to_emails: list
    subject_prefix: str
    cooldown_minutes: Dict[str, int]


@dataclass
class LearningConfig:
    """Machine learning configuration"""
    window_months: int
    nightly_recalibrate: bool
    weekly_retrain_day: int
    ranking_model: Dict[str, Any]
    failure_model: Dict[str, Any]
    calibration: Dict[str, str]
    ensemble_size: int
    drift: Dict[str, Any]
    promotion_gates: Dict[str, float]
    rollback_rules: Dict[str, Any]


@dataclass
class FeaturesConfig:
    """Feature engineering configuration"""
    technical: Dict[str, Any]
    liquidity: Dict[str, list]
    cross_sectional: Dict[str, Any]
    regime: Dict[str, int]


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str
    port: int
    database: str
    username_env: str
    password_env: str
    pool_size: int
    max_overflow: int


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str
    file: str
    max_bytes: int
    backup_count: int
    format: str


@dataclass
class SchedulerConfig:
    """Scheduler configuration"""
    timezone: str
    tasks: Dict[str, Any]


@dataclass
class SimulationConfig:
    """Simulation mode configuration"""
    enabled: bool
    mock_data_days: int
    seed: int


@dataclass
class APIConfig:
    """API configuration"""
    enabled: bool
    host: str
    port: int
    reload: bool


@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    enabled: bool
    port: int


@dataclass
class Config:
    """Main configuration object"""
    market: MarketConfig
    universe: UniverseConfig
    costs: CostsConfig
    risk: RiskConfig
    news: NewsConfig
    email: EmailConfig
    learning: LearningConfig
    features: FeaturesConfig
    database: DatabaseConfig
    logging: LoggingConfig
    scheduler: SchedulerConfig
    simulation: SimulationConfig
    api: APIConfig
    dashboard: DashboardConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary"""
        return cls(
            market=MarketConfig(**config_dict['market']),
            universe=UniverseConfig(**config_dict['universe']),
            costs=CostsConfig(**config_dict['costs']),
            risk=RiskConfig(**config_dict['risk']),
            news=NewsConfig(**config_dict['news']),
            email=EmailConfig(**config_dict['email']),
            learning=LearningConfig(**config_dict['learning']),
            features=FeaturesConfig(**config_dict['features']),
            database=DatabaseConfig(**config_dict['database']),
            logging=LoggingConfig(**config_dict['logging']),
            scheduler=SchedulerConfig(**config_dict['scheduler']),
            simulation=SimulationConfig(**config_dict['simulation']),
            api=APIConfig(**config_dict['api']),
            dashboard=DashboardConfig(**config_dict['dashboard']),
        )
    
    @classmethod
    def load(cls, config_path: str = None) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file. If None, looks for config.yaml
                        in standard locations.
        
        Returns:
            Config object
        """
        if config_path is None:
            # Search for config in standard locations
            search_paths = [
                'config/config.yaml',
                'config.yaml',
                '../config/config.yaml',
            ]
            for path in search_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if config_path is None:
                raise FileNotFoundError(
                    "Config file not found. Searched: " + ", ".join(search_paths)
                )
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    def validate(self):
        """Validate configuration values"""
        errors = []
        
        # Validate risk parameters
        if self.risk.equity_base <= 0:
            errors.append("risk.equity_base must be positive")
        if not 0 < self.risk.alpha_risk_fraction < 1:
            errors.append("risk.alpha_risk_fraction must be between 0 and 1")
        if self.risk.b_max > self.risk.equity_base:
            errors.append("risk.b_max cannot exceed equity_base")
        
        # Validate learning parameters
        if self.learning.ensemble_size < 1:
            errors.append("learning.ensemble_size must be at least 1")
        if not 0 <= self.learning.weekly_retrain_day <= 6:
            errors.append("learning.weekly_retrain_day must be 0-6 (Monday-Sunday)")
        
        # Validate email
        if not self.email.to_emails:
            errors.append("email.to_emails cannot be empty")
        
        if errors:
            raise ValueError("Configuration validation errors:\n" + "\n".join(errors))


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load and validate configuration.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Validated Config object
    """
    config = Config.load(config_path)
    config.validate()
    return config
