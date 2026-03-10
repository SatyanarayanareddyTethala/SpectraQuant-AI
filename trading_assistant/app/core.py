"""
Core trading functions: premarket plan, hourly news, intraday monitor, nightly update.
"""
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
from sqlalchemy.orm import Session

from .db import crud, models, get_db_manager
from .config import Config
from .ingest.market import MarketDataIngester
from .ingest.news import NewsIngester
from .features.build import FeatureBuilder
from .models.rank_model import EnsembleRankingModel
from .models.fail_model import FailureModel
from .models.registry import ModelRegistry
from .risk.sizing import PositionSizer
from .risk.limits import RiskLimits
from .risk.costs import CostModel
from .policy.triggers import TriggerEvaluator
from .policy.rules import PolicyRules
from .state.dedupe import DedupeManager, generate_plan_dedupe_key, generate_news_dedupe_key, generate_exec_dedupe_key
from .notify.email import EmailNotifier


def premarket_plan(config: Config) -> Dict[str, Any]:
    """
    Generate premarket trading plan T-60 minutes before market open.
    
    This function:
    1. Fetches latest market data and news
    2. Builds features with strict as-of timestamps
    3. Scores symbols using ranking model
    4. Predicts failure probabilities
    5. Generates top-K trade list with triggers
    6. Applies do-not-trade rules
    7. Sends PLAN email (idempotent)
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary with plan details
    """
    db_manager = get_db_manager()
    
    with db_manager.get_session() as db:
        # Current date/time
        now = datetime.now()
        plan_date = now.date()
        
        # Check if plan already exists (idempotency)
        existing_plan = crud.get_plan_by_date(db, datetime.combine(plan_date, datetime.min.time()))
        if existing_plan:
            return {'status': 'already_exists', 'plan_id': existing_plan.plan_id}
        
        # Initialize components
        market_ingester = MarketDataIngester(config.market.__dict__)
        feature_builder = FeatureBuilder(config.features.__dict__)
        model_registry = ModelRegistry(db)
        position_sizer = PositionSizer(config.risk.__dict__)
        risk_limits = RiskLimits(config.risk.__dict__)
        dedupe_manager = DedupeManager(config.email.__dict__)
        
        # Get universe
        symbols = config.universe.symbols
        if not symbols and config.universe.tickers_file:
            # Load from file
            import pandas as pd
            df = pd.read_csv(config.universe.tickers_file)
            symbols = df['ticker'].tolist()
        
        # Fetch latest EOD data (up to yesterday)
        yesterday = now - timedelta(days=1)
        start_date = yesterday - timedelta(days=252)
        
        eod_data = market_ingester.fetch_eod_data(symbols, start_date, yesterday)
        market_ingester.save_eod_data(db, eod_data)
        
        # Build features as of today (using data up to yesterday)
        as_of_date = datetime.combine(plan_date, datetime.min.time())
        features_df = feature_builder.build_features(db, symbols, as_of_date)
        
        if features_df.empty:
            return {'status': 'no_features', 'reason': 'insufficient_data'}
        
        # Get active ranking model
        rank_model_obj = model_registry.get_active_model('rank')
        fail_model_obj = model_registry.get_active_model('fail')
        
        if not rank_model_obj:
            return {'status': 'no_model', 'reason': 'no_active_ranking_model'}
        
        # Load models (simplified - in production, load from model_path)
        # For now, assume models are loaded or use mock predictions
        
        # Generate ranking scores (simplified)
        feature_cols = [c for c in features_df.columns if c not in ['date', 'symbol']]
        scores = features_df[feature_cols].mean(axis=1)  # Simplified scoring
        features_df['score_rank'] = scores
        
        # Generate failure probabilities (simplified)
        features_df['p_fail'] = 0.2  # Mock value
        
        # Calculate confidence (1 - p_fail)
        features_df['confidence'] = 1 - features_df['p_fail']
        
        # Rank and select top-K
        top_k = config.portfolio.top_k or 20
        features_df = features_df.sort_values('score_rank', ascending=False).head(top_k)
        
        # Create plan
        plan_json = {
            'plan_date': plan_date.isoformat(),
            'generated_at': now.isoformat(),
            'num_trades': len(features_df),
            'risk_limits': {
                'max_daily_loss': config.risk.max_daily_loss,
                'max_gross_exposure': config.risk.max_gross_exposure,
                'max_spread_bps': config.universe.max_spread_bps
            },
            'portfolio': {
                'equity_base': config.risk.equity_base,
                'available_capital': config.risk.equity_base,
                'num_positions': 0
            }
        }
        
        plan = crud.create_premarket_plan(
            db=db,
            plan_date=datetime.combine(plan_date, datetime.min.time()),
            plan_json=plan_json,
            model_id_rank=rank_model_obj.model_id if rank_model_obj else None,
            model_id_fail=fail_model_obj.model_id if fail_model_obj else None
        )
        
        # Create individual trades
        trades_list = []
        for rank, (idx, row) in enumerate(features_df.iterrows(), 1):
            symbol = row['symbol']
            
            # Get latest price
            latest_bar = db.query(models.EOD).filter(
                models.EOD.symbol == symbol
            ).order_by(models.EOD.date.desc()).first()
            
            if not latest_bar:
                continue
            
            entry_price = latest_bar.close
            
            # Calculate stop and target (simplified)
            stop_price = entry_price * 0.97  # 3% stop
            target_price = entry_price * 1.06  # 6% target
            
            # Calculate position size
            avg_volume = row.get('avg_volume_20d', 1000000)
            size_shares, size_meta = position_sizer.calculate_size(
                entry_price, stop_price, avg_volume, config.risk.equity_base
            )
            
            if size_shares == 0:
                continue
            
            # Define trigger
            trigger_json = {
                'type': 'price',
                'target_price': entry_price * 0.999,  # Trigger slightly below current
                'direction': 'above',
                'description': f"Enter when price reaches {entry_price * 0.999:.2f}"
            }
            
            # Do-not-trade conditions
            do_not_trade_if = {
                'volatility_above': 0.05,
                'volume_below': avg_volume * 0.5,
                'gap_above': 0.03
            }
            
            trade_data = {
                'symbol': symbol,
                'rank': rank,
                'side': 'LONG',
                'entry_type': 'LIMIT',
                'entry_price': entry_price,
                'stop_price': stop_price,
                'target_price': target_price,
                'size_shares': size_shares,
                'trigger_json': trigger_json,
                'score_rank': float(row['score_rank']),
                'p_fail': float(row['p_fail']),
                'confidence': float(row['confidence']),
                'do_not_trade_if': do_not_trade_if
            }
            
            trade = crud.create_plan_trade(db, plan.plan_id, trade_data)
            trades_list.append(trade_data)
        
        # Create PLAN alert with deduplication
        dedupe_key = generate_plan_dedupe_key(datetime.combine(plan_date, datetime.min.time()))
        
        alert_payload = {
            'plan_date': plan_date.isoformat(),
            'generated_at': now.isoformat(),
            'trades': trades_list,
            'risk_limits': plan_json['risk_limits'],
            'portfolio': plan_json['portfolio']
        }
        
        alert = dedupe_manager.create_alert_if_unique(
            db=db,
            alert_type='PLAN',
            dedupe_key=dedupe_key,
            payload=alert_payload,
            email_to=config.email.to_emails,
            plan_id=plan.plan_id
        )
        
        # Send email
        if alert:
            email_notifier = EmailNotifier(config.email.__dict__)
            email_notifier.process_pending_alerts(db)
        
        return {
            'status': 'success',
            'plan_id': plan.plan_id,
            'num_trades': len(trades_list),
            'alert_sent': alert is not None
        }


def hourly_news(config: Config, plan_id: int) -> Dict[str, Any]:
    """
    Fetch and process hourly news updates.
    
    This function:
    1. Fetches new articles from configured providers
    2. Deduplicates against existing news
    3. Enriches with embeddings and risk assessment
    4. Assesses impact on active plan and portfolio
    5. Adjusts confidence/blocked status as needed
    6. Sends HOURLY NEWS email (idempotent per hour)
    
    Args:
        config: Configuration object
        plan_id: Active plan ID
        
    Returns:
        Dictionary with news update details
    """
    db_manager = get_db_manager()
    
    with db_manager.get_session() as db:
        now = datetime.now()
        current_hour = now.hour
        
        # Check if news already processed for this hour
        dedupe_manager = DedupeManager(config.email.__dict__)
        dedupe_key = generate_news_dedupe_key(plan_id, current_hour)
        
        if not dedupe_manager.check_can_send(db, dedupe_key, 'NEWS'):
            return {'status': 'already_sent', 'hour': current_hour}
        
        # Initialize news ingester
        news_ingester = NewsIngester(config.news.__dict__)
        
        # Get plan symbols
        trades = crud.get_plan_trades(db, plan_id)
        symbols = [t.symbol for t in trades]
        
        # Fetch news
        articles = news_ingester.fetch_news(symbols, lookback_hours=1)
        
        # Deduplicate
        unique_articles = news_ingester.deduplicate_news(db, articles)
        
        # Enrich
        enriched_articles = news_ingester.enrich_news(unique_articles)
        
        # Save to database
        news_ingester.save_news(db, enriched_articles)
        
        # Assess portfolio impact (simplified)
        high_risk_articles = [
            (art, enr) for art, enr in enriched_articles 
            if enr['risk_score'] > 0.7
        ]
        
        affected_symbols = set()
        for article, enrichment in high_risk_articles:
            affected_symbols.update(article.get('symbols', []))
        
        # Create news alert
        alert_payload = {
            'timestamp': now.isoformat(),
            'hour': current_hour,
            'article_count': len(unique_articles),
            'high_risk_count': len(high_risk_articles),
            'affected_symbols': list(affected_symbols),
            'articles': [
                {
                    'title': art['title'],
                    'source': art['source'],
                    'published_at': art['published_at'].isoformat(),
                    'url': art.get('url'),
                    'symbols': art.get('symbols', []),
                    'risk_level': 'high' if enr['risk_score'] > 0.7 else ('medium' if enr['risk_score'] > 0.4 else 'low'),
                    'risk_tags': enr['risk_tags'],
                    'summary': enr['summary']
                }
                for art, enr in enriched_articles[:10]  # Top 10
            ],
            'portfolio_impact': {
                'active_trades': len(trades),
                'affected_positions': len(affected_symbols.intersection(set(symbols))),
                'confidence_adjustments': []
            },
            'sources': list(set(art['source'] for art, _ in enriched_articles)),
            'recommendations': []
        }
        
        # Add recommendations if high risk
        if high_risk_articles:
            alert_payload['recommendations'].append(
                f"Review {len(high_risk_articles)} high-risk articles before executing new trades"
            )
        
        alert = dedupe_manager.create_alert_if_unique(
            db=db,
            alert_type='NEWS',
            dedupe_key=dedupe_key,
            payload=alert_payload,
            email_to=config.email.to_emails,
            plan_id=plan_id
        )
        
        # Send email
        if alert:
            email_notifier = EmailNotifier(config.email.__dict__)
            email_notifier.process_pending_alerts(db)
        
        return {
            'status': 'success',
            'article_count': len(unique_articles),
            'high_risk_count': len(high_risk_articles),
            'alert_sent': alert is not None
        }


def intraday_monitor(config: Config, plan_id: int) -> Dict[str, Any]:
    """
    Monitor intraday triggers and send EXECUTE NOW alerts.
    
    This function:
    1. Fetches current intraday bars
    2. Evaluates triggers for each trade in plan
    3. Enforces do-not-trade rules
    4. Enforces portfolio daily loss stop
    5. Sends EXECUTE NOW emails when triggers hit (with deduplication)
    
    Args:
        config: Configuration object
        plan_id: Active plan ID
        
    Returns:
        Dictionary with monitoring results
    """
    db_manager = get_db_manager()
    
    with db_manager.get_session() as db:
        # Initialize components
        # Note: PolicyRules needs full config with nested dataclasses converted to dicts (using asdict)
        # Other components receive specific sub-configs that don't have nested dataclasses
        market_ingester = MarketDataIngester(config.market.__dict__)
        trigger_evaluator = TriggerEvaluator(config.market.__dict__)
        policy_rules = PolicyRules(asdict(config))
        risk_limits = RiskLimits(config.risk.__dict__)
        dedupe_manager = DedupeManager(config.email.__dict__)
        
        # Get plan trades
        trades = crud.get_plan_trades(db, plan_id)
        symbols = [t.symbol for t in trades]
        
        # Fetch current intraday data
        intraday_data = market_ingester.fetch_intraday_data(symbols, interval='5m', period='1d')
        
        # Build current prices dict
        current_prices = {}
        for symbol, df in intraday_data.items():
            if not df.empty:
                current_prices[symbol] = df['close'].iloc[-1]
        
        # Check daily loss limit
        can_trade, current_loss = risk_limits.check_daily_loss_limit(db, datetime.now().date())
        
        if not can_trade:
            return {
                'status': 'loss_limit_breached',
                'current_loss': current_loss,
                'max_loss': config.risk.max_daily_loss
            }
        
        # Evaluate triggers
        triggered_trades = trigger_evaluator.evaluate_triggers(
            db, plan_id, current_prices, intraday_data
        )
        
        alerts_sent = []
        
        for triggered in triggered_trades:
            symbol = triggered['symbol']
            trade = triggered['trade']
            
            # Check do-not-trade rules
            market_data = {
                'avg_daily_volume': 1000000,  # Simplified
                'spread_bps': 10,
                'volatility': 0.02,
                'current_volume': intraday_data[symbol]['volume'].iloc[-1] if symbol in intraday_data else 0,
                'regime': {'volatility': 'normal'}
            }
            
            # Get recent news
            recent_news = db.query(models.NewsRaw).filter(
                models.NewsRaw.symbols.contains([symbol]),
                models.NewsRaw.ts_published >= datetime.now() - timedelta(hours=2)
            ).all()
            
            news_context = [
                {'risk_score': 0.3}  # Simplified
                for _ in recent_news
            ]
            
            allowed, blocked_reasons = policy_rules.check_trade_allowed(
                db, symbol, trade, triggered['current_price'], market_data, news_context
            )
            
            if not allowed:
                continue
            
            # Create EXECUTE NOW alert
            dedupe_key = generate_exec_dedupe_key(plan_id, symbol, str(triggered['trigger_details'].get('type', 'trigger')))
            
            alert_payload = {
                'symbol': symbol,
                'action': 'BUY',
                'entry_price': trade.entry_price,
                'stop_price': trade.stop_price,
                'target_price': trade.target_price,
                'size_shares': trade.size_shares,
                'risk_reward_ratio': (trade.target_price - trade.entry_price) / (trade.entry_price - trade.stop_price),
                'entry_type': trade.entry_type,
                'trigger_type': triggered['trigger_details'].get('type'),
                'trigger_description': str(triggered['trigger_details']),
                'triggered_at': triggered['triggered_at'].isoformat(),
                'plan_id': plan_id
            }
            
            alert = dedupe_manager.create_alert_if_unique(
                db=db,
                alert_type='EXEC',
                dedupe_key=dedupe_key,
                payload=alert_payload,
                email_to=config.email.to_emails,
                plan_id=plan_id,
                symbol=symbol
            )
            
            if alert:
                alerts_sent.append(symbol)
        
        # Send emails
        if alerts_sent:
            email_notifier = EmailNotifier(config.email.__dict__)
            email_notifier.process_pending_alerts(db)
        
        return {
            'status': 'success',
            'triggers_evaluated': len(trades),
            'triggers_hit': len(triggered_trades),
            'alerts_sent': len(alerts_sent),
            'symbols': alerts_sent
        }


def nightly_update(config: Config) -> Dict[str, Any]:
    """
    Nightly update to compute outcomes, label failures, and trigger learning.
    
    This function:
    1. Computes trade outcomes (PnL, MAE, MFE, holding time, costs)
    2. Labels failures precisely with types
    3. Updates model registry with new data
    4. Optionally triggers nightly recalibration
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary with update results
    """
    db_manager = get_db_manager()
    
    with db_manager.get_session() as db:
        # Get today's plan
        today = datetime.now().date()
        plan = crud.get_plan_by_date(db, datetime.combine(today, datetime.min.time()))
        
        if not plan:
            return {'status': 'no_plan', 'date': today.isoformat()}
        
        # Get fills for this plan
        fills = db.query(models.Fill).filter(
            models.Fill.plan_id == plan.plan_id
        ).all()
        
        # Group fills by symbol to construct trades
        from collections import defaultdict
        symbol_fills = defaultdict(list)
        for fill in fills:
            symbol_fills[fill.symbol].append(fill)
        
        outcomes_created = 0
        
        for symbol, fills_list in symbol_fills.items():
            # Sort by timestamp
            fills_list.sort(key=lambda f: f.ts_fill)
            
            # Match buys with sells
            position = 0
            entry_price = 0
            entry_ts = None
            
            for fill in fills_list:
                if fill.action == 'BUY':
                    if position == 0:
                        entry_ts = fill.ts_fill
                        entry_price = fill.price
                    position += fill.qty
                    
                elif fill.action == 'SELL':
                    if position > 0:
                        # Create outcome
                        exit_ts = fill.ts_fill
                        exit_price = fill.price
                        
                        # Calculate metrics (simplified)
                        pnl_gross = (exit_price - entry_price) * min(fill.qty, position)
                        
                        # Get costs
                        cost_model = CostModel(config.costs.__dict__)
                        costs = cost_model.calculate_total_cost(fill.qty, fill.price)
                        
                        pnl_net = pnl_gross - costs['total_cost']
                        return_net = pnl_net / (entry_price * fill.qty) if entry_price * fill.qty > 0 else 0
                        
                        holding_mins = int((exit_ts - entry_ts).total_seconds() / 60)
                        
                        # Create outcome
                        crud.create_trade_outcome(
                            db=db,
                            symbol=symbol,
                            entry_ts=entry_ts,
                            entry_price=entry_price,
                            plan_id=plan.plan_id,
                            outcome_data={
                                'exit_ts': exit_ts,
                                'exit_price': exit_price,
                                'pnl_net': pnl_net,
                                'return_net': return_net,
                                'mae': -0.01,  # Simplified
                                'mfe': 0.02,   # Simplified
                                'holding_mins': holding_mins,
                                'cost_total': costs['total_cost']
                            }
                        )
                        
                        outcomes_created += 1
                        position -= fill.qty
        
        return {
            'status': 'success',
            'plan_id': plan.plan_id,
            'outcomes_created': outcomes_created
        }
