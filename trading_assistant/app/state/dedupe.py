"""
Alert deduplication system to prevent spam.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from ..db import crud, models


class DedupeManager:
    """Manages alert deduplication with cooldown periods"""
    
    def __init__(self, config: dict):
        """
        Initialize deduplication manager.
        
        Args:
            config: Email configuration with cooldown settings
        """
        self.config = config
        self.cooldown_minutes = config.get('cooldown_minutes', {})
    
    def check_can_send(
        self,
        db: Session,
        dedupe_key: str,
        alert_type: str
    ) -> bool:
        """
        Check if an alert can be sent based on deduplication rules.
        
        Args:
            db: Database session
            dedupe_key: Unique key for this alert
            alert_type: Type of alert (for cooldown lookup)
            
        Returns:
            True if alert can be sent, False if duplicate
        """
        # Check if dedupe key exists
        existing = db.query(models.Alert).filter(
            models.Alert.dedupe_key == dedupe_key
        ).first()
        
        if existing:
            # Check cooldown period
            cooldown = self.cooldown_minutes.get(alert_type, 0)
            
            if cooldown > 0:
                cutoff_time = datetime.utcnow() - timedelta(minutes=cooldown)
                
                if existing.created_at > cutoff_time:
                    # Still in cooldown period
                    return False
                else:
                    # Cooldown expired, can send but should update dedupe key
                    # Actually, with unique constraint, we can't resend same key
                    # So we'd need to modify the key slightly (e.g., add timestamp bucket)
                    return False
            else:
                # No cooldown, dedupe key prevents resending forever
                return False
        
        return True
    
    def generate_dedupe_key(
        self,
        alert_type: str,
        date: datetime,
        symbol: Optional[str] = None,
        trigger_id: Optional[str] = None
    ) -> str:
        """
        Generate standardized dedupe key.
        
        Args:
            alert_type: Type of alert (PLAN, NEWS, EXEC)
            date: Date for the alert
            symbol: Optional symbol
            trigger_id: Optional trigger identifier
            
        Returns:
            Dedupe key string
        """
        date_str = date.strftime('%Y%m%d')
        
        if alert_type == 'PLAN':
            # One plan per day
            return f"PLAN:{date_str}"
        
        elif alert_type == 'NEWS':
            # One news update per hour per plan
            hour = date.strftime('%H')
            return f"NEWS:{date_str}{hour}"
        
        elif alert_type == 'EXEC':
            # One execute alert per symbol per trigger
            if not symbol or not trigger_id:
                raise ValueError("EXEC alerts require symbol and trigger_id")
            return f"EXEC:{date_str}:{symbol}:{trigger_id}"
        
        else:
            # Generic
            return f"{alert_type}:{date_str}"
    
    def create_alert_if_unique(
        self,
        db: Session,
        alert_type: str,
        dedupe_key: str,
        payload: Dict[str, Any],
        email_to: list,
        plan_id: Optional[int] = None,
        symbol: Optional[str] = None
    ) -> Optional[models.Alert]:
        """
        Create alert only if dedupe key is unique.
        
        Args:
            db: Database session
            alert_type: Type of alert
            dedupe_key: Deduplication key
            payload: Alert payload
            email_to: List of email recipients
            plan_id: Optional plan ID
            symbol: Optional symbol
            
        Returns:
            Alert object if created, None if duplicate
        """
        if not self.check_can_send(db, dedupe_key, alert_type):
            return None
        
        alert = crud.create_alert(
            db=db,
            alert_type=alert_type,
            dedupe_key=dedupe_key,
            payload=payload,
            email_to=email_to,
            plan_id=plan_id,
            symbol=symbol
        )
        
        return alert
    
    def get_recent_alerts(
        self,
        db: Session,
        symbol: Optional[str] = None,
        alert_type: Optional[str] = None,
        lookback_hours: int = 24
    ) -> list:
        """
        Get recent alerts with optional filters.
        
        Args:
            db: Database session
            symbol: Optional symbol filter
            alert_type: Optional alert type filter
            lookback_hours: Hours to look back
            
        Returns:
            List of Alert objects
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        query = db.query(models.Alert).filter(
            models.Alert.created_at >= cutoff_time
        )
        
        if symbol:
            query = query.filter(models.Alert.symbol == symbol)
        
        if alert_type:
            query = query.filter(models.Alert.alert_type == alert_type)
        
        return query.order_by(models.Alert.created_at.desc()).all()


def generate_plan_dedupe_key(plan_date: datetime) -> str:
    """Generate dedupe key for premarket plan"""
    return f"PLAN:{plan_date.strftime('%Y%m%d')}"


def generate_news_dedupe_key(plan_id: int, hour: int) -> str:
    """Generate dedupe key for hourly news update"""
    return f"NEWS:{plan_id}:{hour:02d}"


def generate_exec_dedupe_key(
    plan_id: int,
    symbol: str,
    trigger_id: str
) -> str:
    """Generate dedupe key for execute now alert"""
    return f"EXEC:{plan_id}:{symbol}:{trigger_id}"
