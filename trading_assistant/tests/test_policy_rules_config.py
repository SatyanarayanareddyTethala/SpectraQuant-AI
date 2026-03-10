"""
Test PolicyRules initialization with Config dataclass.
Tests for the fix of the AttributeError when passing config.__dict__ with nested dataclasses.
"""
import pytest
from dataclasses import asdict
from app.config import Config, MarketConfig, UniverseConfig, RiskConfig, EmailConfig, NewsConfig
from app.config import CostsConfig, LearningConfig, FeaturesConfig, DatabaseConfig, LoggingConfig
from app.config import SchedulerConfig, SimulationConfig, APIConfig, DashboardConfig
from app.policy.rules import PolicyRules


def create_test_config():
    """Create a minimal test config"""
    return Config(
        market=MarketConfig(
            timezone='America/New_York',
            open_time='09:30',
            close_time='16:00',
            premarket_plan_offset_minutes=60,
            intraday_poll_seconds=300
        ),
        universe=UniverseConfig(
            source='file',
            symbols=['AAPL', 'MSFT'],
            tickers_file=None,
            min_adv=1000000,
            max_spread_bps=50
        ),
        costs=CostsConfig(
            commission_per_trade=1.0,
            slippage_model={'fixed': 0.0001}
        ),
        risk=RiskConfig(
            equity_base=100000,
            alpha_risk_fraction=0.02,
            b_max=10000,
            max_daily_loss=2000,
            max_gross_exposure=1.0,
            max_name_exposure=0.1,
            max_sector_exposure=0.3,
            turnover_cap=1.0,
            adv_participation_cap=0.1,
            mae_threshold=0.02
        ),
        news=NewsConfig(
            providers=['test'],
            refresh_minutes=60,
            risk_tags_thresholds={},
            embedding_model='test',
            min_relevance_score=0.5
        ),
        email=EmailConfig(
            smtp_host='localhost',
            smtp_port=587,
            use_tls=True,
            username_env='EMAIL_USER',
            password_env='EMAIL_PASS',
            from_email='test@example.com',
            to_emails=['recipient@example.com'],
            subject_prefix='[Test]',
            cooldown_minutes={'execute_now': 15, 'news_update': 60}
        ),
        learning=LearningConfig(
            window_months=6,
            nightly_recalibrate=True,
            weekly_retrain_day=0,
            ranking_model={},
            failure_model={},
            calibration={},
            ensemble_size=3,
            drift={},
            promotion_gates={},
            rollback_rules={}
        ),
        features=FeaturesConfig(
            technical={},
            liquidity={},
            cross_sectional={},
            regime={}
        ),
        database=DatabaseConfig(
            host='localhost',
            port=5432,
            database='test',
            username_env='DB_USER',
            password_env='DB_PASS',
            pool_size=5,
            max_overflow=10
        ),
        logging=LoggingConfig(
            level='INFO',
            file='test.log',
            max_bytes=1000000,
            backup_count=3,
            format='%(message)s'
        ),
        scheduler=SchedulerConfig(
            timezone='America/New_York',
            tasks={}
        ),
        simulation=SimulationConfig(
            enabled=False,
            mock_data_days=30,
            seed=42
        ),
        api=APIConfig(
            enabled=False,
            host='localhost',
            port=8000,
            reload=False
        ),
        dashboard=DashboardConfig(
            enabled=False,
            port=8501
        )
    )


def test_policy_rules_with_asdict():
    """Test that PolicyRules can be initialized with asdict(config)"""
    config = create_test_config()
    
    # This should work without AttributeError
    policy_rules = PolicyRules(asdict(config))
    
    # Verify that attributes are properly set
    assert policy_rules.min_adv == 1000000
    assert policy_rules.max_spread_bps == 50
    assert policy_rules.max_daily_loss == 2000


def test_policy_rules_with_config_dict_fails():
    """Test that PolicyRules fails with config.__dict__ (the bug we're fixing)"""
    config = create_test_config()
    
    # Using config.__dict__ should cause AttributeError because nested dataclasses
    # are not converted to dicts
    with pytest.raises(AttributeError):
        policy_rules = PolicyRules(config.__dict__)
        # Try to access an attribute to trigger the error
        _ = policy_rules.min_adv


def test_policy_rules_nested_dict_access():
    """Test that PolicyRules correctly accesses nested dictionary values"""
    config = create_test_config()
    config_dict = asdict(config)
    
    policy_rules = PolicyRules(config_dict)
    
    # Verify that nested dictionary access works
    assert 'universe' in config_dict
    assert isinstance(config_dict['universe'], dict)
    assert 'min_adv' in config_dict['universe']
    assert config_dict['universe']['min_adv'] == policy_rules.min_adv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
