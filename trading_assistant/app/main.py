"""
Main application entry point.
"""
import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from .config import load_config
from .db import init_db, get_db, get_db_manager
from .db import crud, models
from .scheduler import TradingScheduler


# Setup logging
def setup_logging(config):
    """Setup logging configuration"""
    log_level = getattr(logging, config.logging.level, logging.INFO)
    log_format = config.logging.format
    
    # Create logs directory
    log_file = Path(config.logging.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


# Load configuration
config = load_config()

# Setup logging
setup_logging(config)

logger = logging.getLogger(__name__)

# Global scheduler instance
scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global scheduler
    
    # Startup
    logger.info("Starting trading assistant application")
    
    # Initialize database
    logger.info("Initializing database...")
    init_db(config.database.__dict__)
    
    # Create tables if they don't exist
    db_manager = get_db_manager()
    db_manager.create_tables()
    logger.info("Database initialized")
    
    # Start scheduler if enabled
    if config.scheduler.tasks:
        logger.info("Starting scheduler...")
        scheduler = TradingScheduler(config)
        scheduler.start()
        logger.info("Scheduler started")
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    
    if scheduler:
        logger.info("Stopping scheduler...")
        scheduler.stop()
        logger.info("Scheduler stopped")
    
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="SpectraQuant Trading Assistant",
    description="Production-grade AI-driven daily trading assistant",
    version="1.0.0",
    lifespan=lifespan
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


# Plan endpoints
@app.get("/plan/today")
async def get_today_plan(db: Session = Depends(get_db)):
    """Get today's premarket plan"""
    from datetime import datetime
    
    today = datetime.now().date()
    plan = crud.get_plan_by_date(db, datetime.combine(today, datetime.min.time()))
    
    if not plan:
        raise HTTPException(status_code=404, detail="No plan for today")
    
    # Get trades
    trades = crud.get_plan_trades(db, plan.plan_id)
    
    return {
        "plan_id": plan.plan_id,
        "plan_date": plan.plan_date.isoformat(),
        "generated_at": plan.generated_at.isoformat(),
        "num_trades": len(trades),
        "trades": [
            {
                "symbol": t.symbol,
                "rank": t.rank,
                "entry_price": t.entry_price,
                "stop_price": t.stop_price,
                "target_price": t.target_price,
                "size_shares": t.size_shares,
                "score_rank": t.score_rank,
                "confidence": t.confidence,
                "p_fail": t.p_fail
            }
            for t in trades
        ]
    }


@app.get("/plan/{plan_id}")
async def get_plan(plan_id: int, db: Session = Depends(get_db)):
    """Get specific plan by ID"""
    plan = db.query(models.PremarketPlan).filter(
        models.PremarketPlan.plan_id == plan_id
    ).first()
    
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")
    
    trades = crud.get_plan_trades(db, plan_id)
    
    return {
        "plan_id": plan.plan_id,
        "plan_date": plan.plan_date.isoformat(),
        "generated_at": plan.generated_at.isoformat(),
        "plan_json": plan.plan_json,
        "num_trades": len(trades)
    }


# Alert endpoints
@app.get("/alerts")
async def list_alerts(
    limit: int = 50,
    alert_type: str = None,
    db: Session = Depends(get_db)
):
    """List recent alerts"""
    query = db.query(models.Alert).order_by(models.Alert.created_at.desc())
    
    if alert_type:
        query = query.filter(models.Alert.alert_type == alert_type)
    
    alerts = query.limit(limit).all()
    
    return {
        "count": len(alerts),
        "alerts": [
            {
                "alert_id": a.alert_id,
                "alert_type": a.alert_type,
                "symbol": a.symbol,
                "created_at": a.created_at.isoformat(),
                "email_status": a.email_status,
                "sent_at": a.sent_at.isoformat() if a.sent_at else None
            }
            for a in alerts
        ]
    }


# Model endpoints
@app.get("/models")
async def list_models(
    model_type: str = None,
    status: str = None,
    db: Session = Depends(get_db)
):
    """List models in registry"""
    from .models.registry import ModelRegistry
    
    registry = ModelRegistry(db)
    models_list = registry.list_models(model_type, status, limit=50)
    
    return {
        "count": len(models_list),
        "models": [
            {
                "model_id": m.model_id,
                "model_type": m.model_type,
                "status": m.status,
                "created_at": m.created_at.isoformat(),
                "metrics": m.metrics_json
            }
            for m in models_list
        ]
    }


@app.get("/models/{model_id}")
async def get_model(model_id: int, db: Session = Depends(get_db)):
    """Get specific model details"""
    from .models.registry import ModelRegistry
    
    registry = ModelRegistry(db)
    model = db.query(models.ModelRegistry).filter(
        models.ModelRegistry.model_id == model_id
    ).first()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "model_id": model.model_id,
        "model_type": model.model_type,
        "status": model.status,
        "created_at": model.created_at.isoformat(),
        "data_window": model.data_window,
        "metrics": model.metrics_json,
        "model_path": model.model_path
    }


# Metrics endpoints
@app.get("/metrics/summary")
async def get_metrics_summary(db: Session = Depends(get_db)):
    """Get summary metrics"""
    from datetime import datetime, timedelta
    
    # Get last 30 days of outcomes
    start_date = datetime.now() - timedelta(days=30)
    outcomes = db.query(models.TradeOutcome).filter(
        models.TradeOutcome.entry_ts >= start_date,
        models.TradeOutcome.exit_ts.isnot(None)
    ).all()
    
    if not outcomes:
        return {
            "period_days": 30,
            "total_trades": 0,
            "metrics": {}
        }
    
    # Calculate metrics
    total_pnl = sum(o.pnl_net or 0 for o in outcomes)
    winning_trades = sum(1 for o in outcomes if (o.pnl_net or 0) > 0)
    losing_trades = sum(1 for o in outcomes if (o.pnl_net or 0) < 0)
    
    return {
        "period_days": 30,
        "total_trades": len(outcomes),
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": winning_trades / len(outcomes) if outcomes else 0,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / len(outcomes) if outcomes else 0,
        "avg_holding_mins": sum(o.holding_mins or 0 for o in outcomes) / len(outcomes) if outcomes else 0
    }


# Scheduler endpoints
@app.get("/scheduler/jobs")
async def list_scheduled_jobs():
    """List all scheduled jobs"""
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not running")
    
    jobs = scheduler.list_jobs()
    return {"jobs": jobs}


# Run application
if __name__ == "__main__":
    import uvicorn
    from datetime import datetime
    
    host = config.api.host if config.api.enabled else "127.0.0.1"
    port = config.api.port if config.api.enabled else 8000
    reload = config.api.reload if config.api.enabled else False
    
    logger.info(f"Starting FastAPI server on {host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)
