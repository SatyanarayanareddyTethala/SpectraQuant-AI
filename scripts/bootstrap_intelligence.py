#!/usr/bin/env python3
"""
Bootstrap Wizard for SpectraQuant Intelligence Layer

Interactive first-run configuration wizard that:
1. Detects existing config
2. Asks for all required settings
3. Validates connectivity (SMTP, News, DB)
4. Writes config files and .env
5. Runs database migrations
6. Executes smoke tests
7. Provides next steps

Usage:
    python scripts/bootstrap_intelligence.py
"""
import os
import sys
import yaml
import getpass
import smtplib
import psycopg2
from pathlib import Path
from datetime import datetime, time
from typing import Dict, Any, List, Optional
import subprocess
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class Colors:
    """Terminal colors for better UX"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def ask_question(question: str, default: Optional[str] = None, secret: bool = False) -> str:
    """Ask user a question and return answer"""
    if default:
        prompt = f"{Colors.OKBLUE}{question} [{default}]:{Colors.ENDC} "
    else:
        prompt = f"{Colors.OKBLUE}{question}:{Colors.ENDC} "
    
    if secret:
        answer = getpass.getpass(prompt)
    else:
        answer = input(prompt)
    
    return answer.strip() if answer.strip() else default


def ask_yes_no(question: str, default: bool = True) -> bool:
    """Ask yes/no question"""
    default_str = "Y/n" if default else "y/N"
    answer = ask_question(f"{question} ({default_str})", default="y" if default else "n")
    return answer.lower() in ['y', 'yes', '']


def check_existing_config(config_path: Path, env_path: Path) -> bool:
    """Check if config already exists"""
    if config_path.exists() or env_path.exists():
        print_warning("Existing configuration detected!")
        if config_path.exists():
            print_info(f"  - Config file: {config_path}")
        if env_path.exists():
            print_info(f"  - Environment file: {env_path}")
        
        if ask_yes_no("Do you want to overwrite existing configuration?", default=False):
            return True
        else:
            print_info("Exiting. Use existing configuration or delete files manually.")
            sys.exit(0)
    return False


def configure_market() -> Dict[str, Any]:
    """Configure market settings"""
    print_header("MARKET CONFIGURATION")
    
    print_info("Configure your market hours and timezone")
    print_info("Examples: Asia/Kolkata (NSE), America/New_York (NYSE), Europe/London (LSE)")
    
    timezone = ask_question("Market timezone", default="Asia/Kolkata")
    
    print_info("\nMarket hours (HH:MM:SS format)")
    open_time = ask_question("Market open time", default="09:15:00")
    close_time = ask_question("Market close time", default="15:30:00")
    
    premarket_offset = int(ask_question("Premarket plan offset (minutes before open)", default="60"))
    intraday_poll = int(ask_question("Intraday polling interval (seconds)", default="60"))
    
    return {
        'timezone': timezone,
        'open_time': open_time,
        'close_time': close_time,
        'premarket_plan_offset_minutes': premarket_offset,
        'intraday_poll_seconds': intraday_poll
    }


def configure_universe() -> Dict[str, Any]:
    """Configure trading universe"""
    print_header("UNIVERSE CONFIGURATION")
    
    print_info("Configure your trading universe (stocks to trade)")
    
    source = ask_question("Universe source (nse/ftse/custom)", default="nse")
    
    tickers_file = ask_question(
        "Path to tickers file (CSV with 'ticker' column)",
        default="data/universe/nse_500.csv"
    )
    
    min_adv = int(ask_question("Minimum average daily volume (shares)", default="1000000"))
    max_spread_bps = int(ask_question("Maximum spread (basis points)", default="50"))
    
    return {
        'source': source,
        'symbols': [],
        'tickers_file': tickers_file,
        'min_adv': min_adv,
        'max_spread_bps': max_spread_bps,
        'min_market_cap': None
    }


def configure_costs() -> Dict[str, Any]:
    """Configure cost model"""
    print_header("COST MODEL CONFIGURATION")
    
    print_info("Configure transaction costs and slippage model")
    
    commission = float(ask_question("Commission per trade (local currency)", default="20.0"))
    
    print_info("\nSlippage model weights (should sum to ~1.0)")
    base_bps = int(ask_question("Base slippage (basis points)", default="5"))
    spread_weight = float(ask_question("Spread weight", default="0.5"))
    vol_weight = float(ask_question("Volatility weight", default="0.3"))
    participation_weight = float(ask_question("Participation weight", default="0.2"))
    
    return {
        'commission_per_trade': commission,
        'slippage_model': {
            'base_bps': base_bps,
            'spread_weight': spread_weight,
            'vol_weight': vol_weight,
            'participation_weight': participation_weight
        }
    }


def configure_risk() -> Dict[str, Any]:
    """Configure risk limits"""
    print_header("RISK MANAGEMENT CONFIGURATION")
    
    print_info("Configure position sizing and risk limits")
    
    equity_base = float(ask_question("Total equity base (local currency)", default="1000000"))
    alpha_risk_fraction = float(ask_question("Risk fraction per trade", default="0.02"))
    b_max = float(ask_question("Maximum position size", default="50000"))
    max_daily_loss = float(ask_question("Maximum daily loss limit", default="20000"))
    max_gross_exposure = float(ask_question("Maximum gross exposure", default="500000"))
    max_name_exposure = float(ask_question("Maximum per-name exposure", default="100000"))
    max_sector_exposure = float(ask_question("Maximum per-sector exposure", default="200000"))
    turnover_cap = float(ask_question("Daily turnover cap (fraction)", default="0.5"))
    adv_participation_cap = float(ask_question("ADV participation cap", default="0.1"))
    mae_threshold = float(ask_question("MAE threshold (fraction)", default="0.03"))
    
    return {
        'equity_base': equity_base,
        'alpha_risk_fraction': alpha_risk_fraction,
        'b_max': b_max,
        'max_daily_loss': max_daily_loss,
        'max_gross_exposure': max_gross_exposure,
        'max_name_exposure': max_name_exposure,
        'max_sector_exposure': max_sector_exposure,
        'turnover_cap': turnover_cap,
        'adv_participation_cap': adv_participation_cap,
        'mae_threshold': mae_threshold
    }


def configure_news() -> tuple[Dict[str, Any], Dict[str, str]]:
    """Configure news providers"""
    print_header("NEWS CONFIGURATION")
    
    env_vars = {}
    
    print_info("Configure news sources for sentiment and risk assessment")
    
    # NewsAPI
    use_newsapi = ask_yes_no("Enable NewsAPI?", default=True)
    newsapi_key = ""
    if use_newsapi:
        newsapi_key = ask_question("NewsAPI key", secret=True)
        if newsapi_key:
            env_vars['NEWSAPI_KEY'] = newsapi_key
    
    # RSS feeds
    use_rss = ask_yes_no("Enable RSS feeds?", default=True)
    rss_urls = []
    if use_rss:
        print_info("Enter RSS feed URLs (one per line, empty line to finish):")
        while True:
            url = ask_question("RSS URL", default="")
            if not url:
                break
            rss_urls.append(url)
    
    refresh_minutes = int(ask_question("News refresh interval (minutes)", default="60"))
    
    providers = []
    if use_newsapi:
        providers.append({
            'type': 'newsapi',
            'api_key_env': 'NEWSAPI_KEY',
            'enabled': True
        })
    if use_rss:
        providers.append({
            'type': 'rss',
            'urls': rss_urls if rss_urls else ['https://economictimes.indiatimes.com/rssfeedstopstories.cms'],
            'enabled': True
        })
    
    return {
        'providers': providers,
        'refresh_minutes': refresh_minutes,
        'risk_tags_thresholds': {
            'high_risk': ['bankruptcy', 'fraud', 'lawsuit', 'investigation'],
            'medium_risk': ['warning', 'downgrade', 'loss', 'decline'],
            'catalyst': ['upgrade', 'beat', 'growth', 'expansion']
        }
    }, env_vars


def configure_email() -> tuple[Dict[str, Any], Dict[str, str]]:
    """Configure email notifications"""
    print_header("EMAIL CONFIGURATION")
    
    env_vars = {}
    
    print_info("Configure SMTP for email notifications")
    print_info("For Gmail, use: smtp.gmail.com:587 with app-specific password")
    
    smtp_host = ask_question("SMTP host", default="smtp.gmail.com")
    smtp_port = int(ask_question("SMTP port", default="587"))
    smtp_username = ask_question("SMTP username (email address)")
    smtp_password = ask_question("SMTP password", secret=True)
    
    env_vars['SMTP_USERNAME'] = smtp_username
    env_vars['SMTP_PASSWORD'] = smtp_password
    
    from_email = ask_question("From email address", default=smtp_username)
    
    # To addresses
    to_emails = []
    print_info("Enter recipient email addresses (one per line, empty line to finish):")
    while True:
        email = ask_question("Email address", default="")
        if not email:
            break
        to_emails.append(email)
    
    if not to_emails:
        to_emails = [smtp_username]
    
    subject_prefix = ask_question("Email subject prefix", default="[SpectraQuant]")
    
    return {
        'smtp_host': smtp_host,
        'smtp_port': smtp_port,
        'username_env': 'SMTP_USERNAME',
        'password_env': 'SMTP_PASSWORD',
        'from': from_email,
        'to': to_emails,
        'subject_prefix': subject_prefix
    }, env_vars


def configure_database() -> tuple[Dict[str, Any], Dict[str, str]]:
    """Configure database"""
    print_header("DATABASE CONFIGURATION")
    
    env_vars = {}
    
    print_info("Configure database connection")
    print_info("PostgreSQL recommended for production; SQLite available for simulation.")
    
    use_sqlite = ask_yes_no("Use SQLite simulation fallback (no PostgreSQL required)?", default=False)
    
    if use_sqlite:
        database_url = "sqlite:///intelligence.db"
        env_vars['DATABASE_URL'] = database_url
        return {
            'host': '',
            'port': 0,
            'name': 'intelligence.db',
            'backend': 'sqlite',
            'user_env': '',
            'password_env': ''
        }, env_vars
    
    use_docker = ask_yes_no("Use Docker for PostgreSQL?", default=True)
    
    if use_docker:
        db_host = "localhost"
        db_port = 5432
        db_name = "trading_assistant"
        db_user = "postgres"
        db_pass = ask_question("PostgreSQL password", default="postgres", secret=True)
    else:
        db_host = ask_question("Database host", default="localhost")
        db_port = int(ask_question("Database port", default="5432"))
        db_name = ask_question("Database name", default="trading_assistant")
        db_user = ask_question("Database username", default="postgres")
        db_pass = ask_question("Database password", secret=True)
    
    database_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    
    env_vars['DB_USERNAME'] = db_user
    env_vars['DB_PASSWORD'] = db_pass
    env_vars['DATABASE_URL'] = database_url
    
    return {
        'host': db_host,
        'port': db_port,
        'name': db_name,
        'backend': 'postgresql',
        'user_env': 'DB_USERNAME',
        'password_env': 'DB_PASSWORD'
    }, env_vars


def configure_learning() -> Dict[str, Any]:
    """Configure learning parameters"""
    print_header("LEARNING CONFIGURATION")
    
    print_info("Configure online learning and model retraining")
    
    window_months = int(ask_question("Training window (months)", default="12"))
    nightly_recalibrate = ask_yes_no("Enable nightly recalibration?", default=True)
    weekly_retrain_day = int(ask_question("Weekly retrain day (0=Mon, 6=Sun)", default="6"))
    
    print_info("\nDrift detection thresholds")
    psi_threshold = float(ask_question("PSI threshold", default="0.25"))
    ks_pvalue_threshold = float(ask_question("KS p-value threshold", default="0.05"))
    adwin_delta = float(ask_question("ADWIN delta", default="0.002"))
    
    print_info("\nModel promotion gates")
    min_net_improvement = float(ask_question("Min net improvement", default="0.02"))
    max_drawdown_increase = float(ask_question("Max drawdown increase", default="0.05"))
    min_calibration_improvement = float(ask_question("Min calibration improvement", default="0.01"))
    max_turnover_increase = float(ask_question("Max turnover increase", default="0.1"))
    
    print_info("\nRollback rules")
    underperform_days = int(ask_question("Underperform days threshold", default="5"))
    dd_breach = float(ask_question("Drawdown breach threshold", default="0.1"))
    
    return {
        'window_months': window_months,
        'nightly_recalibrate': nightly_recalibrate,
        'weekly_retrain_day': weekly_retrain_day,
        'drift': {
            'psi_threshold': psi_threshold,
            'ks_pvalue_threshold': ks_pvalue_threshold,
            'adwin_delta': adwin_delta
        },
        'promotion_gates': {
            'min_net_improvement': min_net_improvement,
            'max_drawdown_increase': max_drawdown_increase,
            'min_calibration_improvement': min_calibration_improvement,
            'max_turnover_increase': max_turnover_increase
        },
        'rollback_rules': {
            'underperform_days': underperform_days,
            'dd_breach': dd_breach
        }
    }


def configure_simulation() -> Dict[str, Any]:
    """Configure simulation mode"""
    print_header("SIMULATION CONFIGURATION")
    
    print_info("Simulation mode for safe research and paper trading")
    print_info("Live execution is NOT supported — simulation is always the default.")
    
    enabled = ask_yes_no("Enable simulation mode?", default=True)
    
    if enabled:
        data_provider = ask_question("Data provider (yfinance/local)", default="yfinance")
        mock_data_days = int(ask_question("Mock data days", default="252"))
        seed = int(ask_question("Random seed", default="42"))
        execute_orders = ask_yes_no("Execute simulated orders?", default=False)
        
        return {
            'enabled': True,
            'data_provider': data_provider,
            'mock_data_days': mock_data_days,
            'seed': seed,
            'execute_orders': execute_orders
        }
    else:
        print_warning("Simulation mode is strongly recommended. No live trading is supported.")
        return {
            'enabled': True,  # Force simulation on regardless
            'data_provider': 'yfinance',
            'execute_orders': False
        }


def configure_logging() -> Dict[str, Any]:
    """Configure logging"""
    print_header("LOGGING CONFIGURATION")
    
    print_info("Configure application logging")
    
    level = ask_question("Log level (DEBUG/INFO/WARNING/ERROR)", default="INFO")
    log_file = ask_question("Log file path", default="logs/trading_assistant.log")
    
    return {
        'level': level.upper(),
        'file': log_file,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }


def configure_features() -> Dict[str, Any]:
    """Configure feature settings"""
    return {
        'lookback_days': 252,
        'min_history_days': 60,
        'technical_indicators': True,
        'liquidity_features': True,
        'regime_features': True,
        'cross_sectional_features': True
    }


def configure_models() -> Dict[str, Any]:
    """Configure model settings"""
    return {
        'ensemble_size': 3,
        'ranking_model': 'lightgbm',
        'failure_model': 'xgboost',
        'calibration_method': 'isotonic',
        'cross_validation_folds': 5
    }


def configure_scheduler() -> Dict[str, Any]:
    """Configure scheduler timing"""
    return {
        'premarket_time': '08:15:00',
        'nightly_time': '18:00:00',
        'weekly_retrain_time': '02:00:00'
    }


def validate_timezone(timezone: str) -> bool:
    """Validate timezone string"""
    print_info(f"Validating timezone: {timezone}")
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(timezone)
        print_success(f"Timezone '{timezone}' is valid!")
        return True
    except (KeyError, Exception) as e:
        print_error(f"Invalid timezone '{timezone}': {e}")
        return False


def validate_smtp(config: Dict[str, Any], env_vars: Dict[str, str]) -> bool:
    """Validate SMTP connection"""
    print_info("Validating SMTP connection...")
    
    try:
        host = config['smtp_host']
        port = config['smtp_port']
        username = env_vars.get('SMTP_USERNAME', '')
        password = env_vars.get('SMTP_PASSWORD', '')
        
        with smtplib.SMTP(host, port, timeout=10) as server:
            server.starttls()
            server.login(username, password)
        
        print_success("SMTP connection successful!")
        return True
    except Exception as e:
        print_error(f"SMTP connection failed: {e}")
        return False


def validate_database(env_vars: Dict[str, str]) -> bool:
    """Validate database connection"""
    print_info("Validating database connection...")
    
    try:
        database_url = env_vars.get('DATABASE_URL', '')
        # Parse URL
        match = re.match(r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', database_url)
        if not match:
            print_error("Invalid database URL format")
            return False
        
        user, password, host, port, dbname = match.groups()
        
        # Try to connect
        conn = psycopg2.connect(
            host=host,
            port=int(port),
            user=user,
            password=password,
            dbname='postgres',  # Connect to default DB first
            connect_timeout=10
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if target database exists, create if not
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
        if not cursor.fetchone():
            print_info(f"Creating database '{dbname}'...")
            cursor.execute(f'CREATE DATABASE {dbname}')
            print_success(f"Database '{dbname}' created!")
        
        cursor.close()
        conn.close()
        
        # Connect to target database
        conn = psycopg2.connect(
            host=host,
            port=int(port),
            user=user,
            password=password,
            dbname=dbname,
            connect_timeout=10
        )
        conn.close()
        
        print_success("Database connection successful!")
        return True
    except Exception as e:
        print_error(f"Database connection failed: {e}")
        print_info("If using Docker, make sure PostgreSQL container is running:")
        print_info("  cd trading_assistant && docker-compose up -d postgres")
        return False


def validate_news_api(env_vars: Dict[str, str]) -> bool:
    """Validate NewsAPI connection"""
    newsapi_key = env_vars.get('NEWSAPI_KEY', '')
    if not newsapi_key:
        print_warning("NewsAPI key not provided, skipping validation")
        return True
    
    print_info("Validating NewsAPI connection...")
    
    try:
        import requests
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            'apiKey': newsapi_key,
            'country': 'us',
            'pageSize': 1
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            print_success("NewsAPI connection successful!")
            return True
        else:
            print_error(f"NewsAPI validation failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"NewsAPI validation failed: {e}")
        return False


def write_config_file(config: Dict[str, Any], config_path: Path):
    """Write configuration to YAML file"""
    print_info(f"Writing configuration to {config_path}...")
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print_success("Configuration file written!")


def write_env_file(env_vars: Dict[str, str], env_path: Path):
    """Write environment variables to .env file"""
    print_info(f"Writing secrets to {env_path}...")
    
    with open(env_path, 'w') as f:
        f.write("# SpectraQuant Intelligence Layer - Environment Variables\n")
        f.write("# DO NOT COMMIT THIS FILE\n\n")
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    print_success("Environment file written!")
    print_warning(f"IMPORTANT: Never commit {env_path} to version control!")


def run_migrations(trading_assistant_dir: Path) -> bool:
    """Run Alembic migrations"""
    print_info("Running database migrations...")
    
    try:
        os.chdir(trading_assistant_dir)
        result = subprocess.run(
            ['alembic', 'upgrade', 'head'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print_success("Database migrations completed!")
            print_info(result.stdout)
            return True
        else:
            print_error("Migration failed!")
            print_error(result.stderr)
            return False
    except Exception as e:
        print_error(f"Migration error: {e}")
        return False


def run_smoke_tests(repo_root: Path) -> bool:
    """Run basic smoke tests using the intelligence layer"""
    print_info("Running smoke tests...")
    
    try:
        src_dir = str(repo_root / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        
        from spectraquant.intelligence.config import load_config
        from spectraquant.intelligence.premarket import premarket_plan
        from spectraquant.intelligence.hourly_news import hourly_news
        from spectraquant.intelligence.intraday import intraday_monitor
        from spectraquant.intelligence.learning import nightly_update
        
        print_success("Core function imports successful!")
        
        # Run each function in simulation mode
        print_info("Running premarket_plan(simulation=True)...")
        result = premarket_plan(simulation=True)
        print_success(f"premarket_plan: {result.get('status', 'ok')}")
        
        print_info("Running hourly_news()...")
        result = hourly_news()
        print_success(f"hourly_news: {result.get('status', 'ok')}")
        
        print_info("Running intraday_monitor()...")
        result = intraday_monitor()
        print_success(f"intraday_monitor: {result.get('status', 'ok')}")
        
        print_info("Running nightly_update()...")
        result = nightly_update()
        print_success(f"nightly_update: {result.get('status', 'ok')}")
        
        print_success("All smoke tests passed!")
        return True
    except Exception as e:
        print_error(f"Smoke test failed: {e}")
        return False


def create_run_marker(repo_root: Path):
    """Create RUN_OK marker file"""
    marker_path = repo_root / "config" / "RUN_OK"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    with open(marker_path, 'w') as f:
        f.write(f"Bootstrap completed at {datetime.now()}\n")
    print_success(f"Created run marker: {marker_path}")


def print_next_steps():
    """Print next steps for the user"""
    print_header("SETUP COMPLETE!")
    
    print(f"{Colors.OKGREEN}Bootstrap wizard completed successfully!{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}Next Steps:{Colors.ENDC}\n")
    
    print("1. Review configuration:")
    print("   cat config/intelligence.yaml\n")
    
    print("2. Start the intelligence scheduler:")
    print("   python -m spectraquant.intelligence.scheduler\n")
    
    print("3. Check health:")
    print("   curl http://localhost:8000/health\n")
    
    print("4. View scheduled jobs:")
    print("   curl http://localhost:8000/scheduler/jobs\n")
    
    print("5. (Optional) Start the trading assistant:")
    print("   cd trading_assistant && docker-compose up -d\n")
    
    print(f"{Colors.WARNING}IMPORTANT:{Colors.ENDC}")
    print("- System starts in SIMULATION mode by default")
    print("- No live trading is supported — paper trading only")
    print("- yFinance provides periodic/near-real-time data")
    print("- Monitor the system for at least a week in simulation")
    print("- Read README_INTELLIGENCE.md for complete documentation\n")


def main():
    """Main bootstrap wizard"""
    print_header("SPECTRAQUANT INTELLIGENCE LAYER - BOOTSTRAP WIZARD")
    
    print(f"{Colors.OKCYAN}Welcome to the SpectraQuant Intelligence Layer setup wizard!{Colors.ENDC}")
    print(f"{Colors.OKCYAN}This wizard will guide you through the initial configuration.{Colors.ENDC}\n")
    
    # Determine paths
    repo_root = Path(__file__).parent.parent
    config_dir = repo_root / "config"
    config_path = config_dir / "intelligence.yaml"
    env_path = repo_root / ".env"
    
    # Check existing config
    check_existing_config(config_path, env_path)
    
    # Collect all configuration
    all_env_vars = {}
    
    config = {}
    config['market'] = configure_market()
    config['universe'] = configure_universe()
    config['costs'] = configure_costs()
    config['risk'] = configure_risk()
    
    news_config, news_env = configure_news()
    config['news'] = news_config
    all_env_vars.update(news_env)
    
    email_config, email_env = configure_email()
    config['email'] = email_config
    all_env_vars.update(email_env)
    
    db_config, db_env = configure_database()
    config['database'] = db_config
    all_env_vars.update(db_env)
    
    config['learning'] = configure_learning()
    config['simulation'] = configure_simulation()
    config['execution'] = {'live_enabled': False}
    config['logging'] = configure_logging()
    config['features'] = configure_features()
    config['models'] = configure_models()
    config['scheduler'] = configure_scheduler()
    
    # Validation
    print_header("VALIDATION")
    
    validations = []
    
    # Timezone validation
    validations.append(('Timezone', validate_timezone(config['market']['timezone'])))
    
    # SMTP validation
    if ask_yes_no("Validate SMTP connection?", default=True):
        validations.append(('SMTP', validate_smtp(email_config, all_env_vars)))
    else:
        validations.append(('SMTP', None))
    
    # Database validation (skip for SQLite)
    if config['database'].get('backend') == 'sqlite':
        print_success("SQLite selected — no connection validation needed.")
        validations.append(('Database', True))
    elif ask_yes_no("Validate database connection?", default=True):
        validations.append(('Database', validate_database(all_env_vars)))
    else:
        validations.append(('Database', None))
    
    # News API validation
    if all_env_vars.get('NEWSAPI_KEY'):
        if ask_yes_no("Validate NewsAPI connection?", default=True):
            validations.append(('NewsAPI', validate_news_api(all_env_vars)))
        else:
            validations.append(('NewsAPI', None))
    
    # Check validation results
    failures = [name for name, result in validations if result is False]
    if failures:
        print_warning(f"Some validations failed: {', '.join(failures)}")
        if not ask_yes_no("Continue anyway?", default=False):
            print_error("Setup aborted.")
            sys.exit(1)
    
    # Write files
    print_header("WRITING CONFIGURATION")
    
    write_config_file(config, config_path)
    write_env_file(all_env_vars, env_path)
    
    # Database setup
    print_header("DATABASE SETUP")
    
    if config['database'].get('backend') == 'sqlite':
        print_info("Initializing SQLite database...")
        try:
            sys.path.insert(0, str(repo_root / "src"))
            from spectraquant.intelligence.db.session import init_db
            from spectraquant.intelligence.db.models import Base
            os.environ['DATABASE_URL'] = all_env_vars.get('DATABASE_URL', 'sqlite:///intelligence.db')
            engine = init_db(os.environ['DATABASE_URL'])
            Base.metadata.create_all(engine)
            print_success("SQLite database initialized!")
        except Exception as e:
            print_error(f"SQLite init failed: {e}")
    else:
        if ask_yes_no("Run database migrations now?", default=True):
            trading_assistant_dir = repo_root / "trading_assistant"
            if not run_migrations(trading_assistant_dir):
                print_warning("Migrations failed. You can run them manually later:")
                print_info("  cd trading_assistant && alembic upgrade head")
        else:
            print_info("Skipping migrations. Run manually with:")
            print_info("  cd trading_assistant && alembic upgrade head")
    
    # Smoke tests
    print_header("SMOKE TESTS")
    
    if ask_yes_no("Run smoke tests?", default=True):
        run_smoke_tests(repo_root)
    
    # Create run marker
    create_run_marker(repo_root)
    
    # Print next steps
    print_next_steps()
    
    print(f"{Colors.OKGREEN}{Colors.BOLD}SpectraQuant Intelligence Layer initialized successfully.{Colors.ENDC}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Setup interrupted by user.{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
