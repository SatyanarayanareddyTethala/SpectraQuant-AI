"""Initial schema for trading assistant

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create bars_5m table
    op.create_table(
        'bars_5m',
        sa.Column('ts', sa.DateTime(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('open', sa.Float(), nullable=False),
        sa.Column('high', sa.Float(), nullable=False),
        sa.Column('low', sa.Float(), nullable=False),
        sa.Column('close', sa.Float(), nullable=False),
        sa.Column('volume', sa.Float(), nullable=False),
        sa.Column('vwap', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('ts', 'symbol')
    )
    op.create_index('idx_bars_5m_symbol', 'bars_5m', ['symbol'])
    op.create_index('idx_bars_5m_ts', 'bars_5m', ['ts'])

    # Create eod table
    op.create_table(
        'eod',
        sa.Column('date', sa.DateTime(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('open', sa.Float(), nullable=False),
        sa.Column('high', sa.Float(), nullable=False),
        sa.Column('low', sa.Float(), nullable=False),
        sa.Column('close', sa.Float(), nullable=False),
        sa.Column('volume', sa.Float(), nullable=False),
        sa.Column('adj_close', sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint('date', 'symbol')
    )
    op.create_index('idx_eod_symbol', 'eod', ['symbol'])
    op.create_index('idx_eod_date', 'eod', ['date'])

    # Create news_raw table
    op.create_table(
        'news_raw',
        sa.Column('news_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('ts_published', sa.DateTime(), nullable=False),
        sa.Column('source', sa.String(100), nullable=False),
        sa.Column('url', sa.Text(), nullable=True),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('body', sa.Text(), nullable=True),
        sa.Column('symbols', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('hash', sa.String(64), nullable=False),
        sa.PrimaryKeyConstraint('news_id'),
        sa.UniqueConstraint('hash')
    )
    op.create_index('idx_news_raw_ts', 'news_raw', ['ts_published'])
    op.create_index('idx_news_raw_hash', 'news_raw', ['hash'])

    # Create news_enriched table
    op.create_table(
        'news_enriched',
        sa.Column('news_id', sa.Integer(), nullable=False),
        sa.Column('embedding', postgresql.BYTEA(), nullable=True),
        sa.Column('risk_tags', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('risk_score', sa.Float(), nullable=True),
        sa.Column('summary_3bul', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['news_id'], ['news_raw.news_id']),
        sa.PrimaryKeyConstraint('news_id')
    )

    # Create features_daily table
    op.create_table(
        'features_daily',
        sa.Column('date', sa.DateTime(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('feature_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint('date', 'symbol')
    )
    op.create_index('idx_features_daily_symbol', 'features_daily', ['symbol'])
    op.create_index('idx_features_daily_date', 'features_daily', ['date'])

    # Create model_registry table
    op.create_table(
        'model_registry',
        sa.Column('model_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('data_window', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('metrics_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('model_path', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('model_id')
    )
    op.create_index('idx_model_registry_status', 'model_registry', ['status'])
    op.create_index('idx_model_registry_created', 'model_registry', ['created_at'])

    # Create premarket_plan table
    op.create_table(
        'premarket_plan',
        sa.Column('plan_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('plan_date', sa.DateTime(), nullable=False),
        sa.Column('generated_at', sa.DateTime(), nullable=False),
        sa.Column('model_id_rank', sa.Integer(), nullable=True),
        sa.Column('model_id_fail', sa.Integer(), nullable=True),
        sa.Column('plan_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(['model_id_rank'], ['model_registry.model_id']),
        sa.ForeignKeyConstraint(['model_id_fail'], ['model_registry.model_id']),
        sa.PrimaryKeyConstraint('plan_id'),
        sa.UniqueConstraint('plan_date')
    )
    op.create_index('idx_premarket_plan_date', 'premarket_plan', ['plan_date'])

    # Create plan_trades table
    op.create_table(
        'plan_trades',
        sa.Column('plan_id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('rank', sa.Integer(), nullable=False),
        sa.Column('side', sa.String(10), nullable=False),
        sa.Column('entry_type', sa.String(20), nullable=False),
        sa.Column('entry_price', sa.Float(), nullable=False),
        sa.Column('stop_price', sa.Float(), nullable=False),
        sa.Column('target_price', sa.Float(), nullable=False),
        sa.Column('size_shares', sa.Integer(), nullable=False),
        sa.Column('trigger_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('score_rank', sa.Float(), nullable=False),
        sa.Column('p_fail', sa.Float(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('do_not_trade_if', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(['plan_id'], ['premarket_plan.plan_id']),
        sa.PrimaryKeyConstraint('plan_id', 'symbol')
    )
    op.create_index('idx_plan_trades_rank', 'plan_trades', ['rank'])
    op.create_index('idx_plan_trades_symbol', 'plan_trades', ['symbol'])

    # Create alerts table
    op.create_table(
        'alerts',
        sa.Column('alert_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('plan_id', sa.Integer(), nullable=True),
        sa.Column('symbol', sa.String(20), nullable=True),
        sa.Column('alert_type', sa.String(50), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('dedupe_key', sa.String(200), nullable=False),
        sa.Column('payload_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('email_to', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('email_status', sa.String(20), nullable=False),
        sa.Column('sent_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['plan_id'], ['premarket_plan.plan_id']),
        sa.PrimaryKeyConstraint('alert_id'),
        sa.UniqueConstraint('dedupe_key')
    )
    op.create_index('idx_alerts_type', 'alerts', ['alert_type'])
    op.create_index('idx_alerts_created', 'alerts', ['created_at'])
    op.create_index('idx_alerts_dedupe', 'alerts', ['dedupe_key'])

    # Create fills table
    op.create_table(
        'fills',
        sa.Column('fill_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('plan_id', sa.Integer(), nullable=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('ts_fill', sa.DateTime(), nullable=False),
        sa.Column('action', sa.String(10), nullable=False),
        sa.Column('qty', sa.Integer(), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('fees', sa.Float(), nullable=False),
        sa.Column('slippage_bps', sa.Float(), nullable=False),
        sa.Column('venue', sa.String(50), nullable=True),
        sa.Column('meta_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['plan_id'], ['premarket_plan.plan_id']),
        sa.PrimaryKeyConstraint('fill_id')
    )
    op.create_index('idx_fills_symbol', 'fills', ['symbol'])
    op.create_index('idx_fills_ts', 'fills', ['ts_fill'])

    # Create trade_outcomes table
    op.create_table(
        'trade_outcomes',
        sa.Column('trade_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('plan_id', sa.Integer(), nullable=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('entry_ts', sa.DateTime(), nullable=False),
        sa.Column('entry_price', sa.Float(), nullable=False),
        sa.Column('exit_ts', sa.DateTime(), nullable=True),
        sa.Column('exit_price', sa.Float(), nullable=True),
        sa.Column('pnl_net', sa.Float(), nullable=True),
        sa.Column('return_net', sa.Float(), nullable=True),
        sa.Column('mae', sa.Float(), nullable=True),
        sa.Column('mfe', sa.Float(), nullable=True),
        sa.Column('holding_mins', sa.Integer(), nullable=True),
        sa.Column('cost_total', sa.Float(), nullable=True),
        sa.Column('outcome_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['plan_id'], ['premarket_plan.plan_id']),
        sa.PrimaryKeyConstraint('trade_id')
    )
    op.create_index('idx_trade_outcomes_symbol', 'trade_outcomes', ['symbol'])
    op.create_index('idx_trade_outcomes_entry_ts', 'trade_outcomes', ['entry_ts'])

    # Create failure_labels table
    op.create_table(
        'failure_labels',
        sa.Column('trade_id', sa.Integer(), nullable=False),
        sa.Column('label', sa.String(50), nullable=False),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('details_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['trade_id'], ['trade_outcomes.trade_id']),
        sa.PrimaryKeyConstraint('trade_id', 'label')
    )

    # Create learning_runs table
    op.create_table(
        'learning_runs',
        sa.Column('run_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
        sa.Column('data_range', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('drift_flags', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('candidate_models', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('promoted_model_id', sa.Integer(), nullable=True),
        sa.Column('decision', sa.String(20), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['promoted_model_id'], ['model_registry.model_id']),
        sa.PrimaryKeyConstraint('run_id')
    )
    op.create_index('idx_learning_runs_started', 'learning_runs', ['started_at'])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index('idx_learning_runs_started', table_name='learning_runs')
    op.drop_table('learning_runs')
    
    op.drop_table('failure_labels')
    
    op.drop_index('idx_trade_outcomes_entry_ts', table_name='trade_outcomes')
    op.drop_index('idx_trade_outcomes_symbol', table_name='trade_outcomes')
    op.drop_table('trade_outcomes')
    
    op.drop_index('idx_fills_ts', table_name='fills')
    op.drop_index('idx_fills_symbol', table_name='fills')
    op.drop_table('fills')
    
    op.drop_index('idx_alerts_dedupe', table_name='alerts')
    op.drop_index('idx_alerts_created', table_name='alerts')
    op.drop_index('idx_alerts_type', table_name='alerts')
    op.drop_table('alerts')
    
    op.drop_index('idx_plan_trades_symbol', table_name='plan_trades')
    op.drop_index('idx_plan_trades_rank', table_name='plan_trades')
    op.drop_table('plan_trades')
    
    op.drop_index('idx_premarket_plan_date', table_name='premarket_plan')
    op.drop_table('premarket_plan')
    
    op.drop_index('idx_model_registry_created', table_name='model_registry')
    op.drop_index('idx_model_registry_status', table_name='model_registry')
    op.drop_table('model_registry')
    
    op.drop_index('idx_features_daily_date', table_name='features_daily')
    op.drop_index('idx_features_daily_symbol', table_name='features_daily')
    op.drop_table('features_daily')
    
    op.drop_table('news_enriched')
    
    op.drop_index('idx_news_raw_hash', table_name='news_raw')
    op.drop_index('idx_news_raw_ts', table_name='news_raw')
    op.drop_table('news_raw')
    
    op.drop_index('idx_eod_date', table_name='eod')
    op.drop_index('idx_eod_symbol', table_name='eod')
    op.drop_table('eod')
    
    op.drop_index('idx_bars_5m_ts', table_name='bars_5m')
    op.drop_index('idx_bars_5m_symbol', table_name='bars_5m')
    op.drop_table('bars_5m')
