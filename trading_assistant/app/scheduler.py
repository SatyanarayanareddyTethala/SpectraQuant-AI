"""
Task scheduler for trading assistant.
"""
import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import pytz

from .config import Config
from .core import premarket_plan, hourly_news, intraday_monitor, nightly_update
from .db import init_db


logger = logging.getLogger(__name__)


class TradingScheduler:
    """Manages scheduled tasks for trading assistant"""
    
    def __init__(self, config: Config):
        """
        Initialize scheduler.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.scheduler = None
        self.timezone = pytz.timezone(config.scheduler.timezone)
        self.current_plan_id = None
    
    def start(self):
        """Start the scheduler"""
        if self.scheduler is not None:
            logger.warning("Scheduler already running")
            return
        
        self.scheduler = BackgroundScheduler(timezone=self.timezone)
        
        # Add scheduled tasks based on configuration
        self._add_tasks()
        
        # Start scheduler
        self.scheduler.start()
        logger.info("Trading scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        if self.scheduler:
            self.scheduler.shutdown(wait=True)
            self.scheduler = None
            logger.info("Trading scheduler stopped")
    
    def _add_tasks(self):
        """Add all scheduled tasks"""
        tasks_config = self.config.scheduler.tasks
        
        # Premarket plan task
        if tasks_config.get('premarket_plan', {}).get('enabled', True):
            time_str = tasks_config['premarket_plan'].get('time', '08:15:00')
            hour, minute, second = map(int, time_str.split(':'))
            
            self.scheduler.add_job(
                func=self._run_premarket_plan,
                trigger=CronTrigger(
                    hour=hour,
                    minute=minute,
                    second=second,
                    timezone=self.timezone
                ),
                id='premarket_plan',
                name='Premarket Plan Generation',
                replace_existing=True
            )
            logger.info(f"Scheduled premarket plan for {time_str}")
        
        # Hourly news task
        if tasks_config.get('hourly_news', {}).get('enabled', True):
            interval_minutes = tasks_config['hourly_news'].get('interval_minutes', 60)
            active_hours = tasks_config['hourly_news'].get('active_hours', list(range(9, 16)))
            
            # Add job for each active hour
            for hour in active_hours:
                self.scheduler.add_job(
                    func=self._run_hourly_news,
                    trigger=CronTrigger(
                        hour=hour,
                        minute=0,
                        timezone=self.timezone
                    ),
                    id=f'hourly_news_{hour}',
                    name=f'Hourly News Update - {hour}:00',
                    replace_existing=True
                )
            
            logger.info(f"Scheduled hourly news for hours: {active_hours}")
        
        # Intraday monitoring task
        if tasks_config.get('intraday_monitor', {}).get('enabled', True):
            interval_seconds = tasks_config['intraday_monitor'].get('interval_seconds', 60)
            active_hours = tasks_config['intraday_monitor'].get('active_hours', list(range(9, 16)))
            
            # Use interval trigger that checks if we're in active hours
            self.scheduler.add_job(
                func=self._run_intraday_monitor,
                trigger=IntervalTrigger(
                    seconds=interval_seconds,
                    timezone=self.timezone
                ),
                id='intraday_monitor',
                name='Intraday Monitoring',
                replace_existing=True
            )
            logger.info(f"Scheduled intraday monitoring every {interval_seconds} seconds")
        
        # Nightly update task
        if tasks_config.get('nightly_update', {}).get('enabled', True):
            time_str = tasks_config['nightly_update'].get('time', '18:00:00')
            hour, minute, second = map(int, time_str.split(':'))
            
            self.scheduler.add_job(
                func=self._run_nightly_update,
                trigger=CronTrigger(
                    hour=hour,
                    minute=minute,
                    second=second,
                    timezone=self.timezone
                ),
                id='nightly_update',
                name='Nightly Update',
                replace_existing=True
            )
            logger.info(f"Scheduled nightly update for {time_str}")
        
        # Weekly retrain task
        if tasks_config.get('weekly_retrain', {}).get('enabled', True):
            day_of_week = tasks_config['weekly_retrain'].get('day_of_week', 6)
            time_str = tasks_config['weekly_retrain'].get('time', '02:00:00')
            hour, minute, second = map(int, time_str.split(':'))
            
            self.scheduler.add_job(
                func=self._run_weekly_retrain,
                trigger=CronTrigger(
                    day_of_week=day_of_week,
                    hour=hour,
                    minute=minute,
                    second=second,
                    timezone=self.timezone
                ),
                id='weekly_retrain',
                name='Weekly Model Retrain',
                replace_existing=True
            )
            logger.info(f"Scheduled weekly retrain for day {day_of_week} at {time_str}")
    
    def _run_premarket_plan(self):
        """Run premarket plan generation"""
        try:
            logger.info("Starting premarket plan generation")
            result = premarket_plan(self.config)
            
            if result['status'] == 'success':
                self.current_plan_id = result['plan_id']
                logger.info(f"Premarket plan generated successfully: Plan ID {self.current_plan_id}")
            else:
                logger.warning(f"Premarket plan generation failed: {result}")
        
        except Exception as e:
            logger.error(f"Error in premarket plan: {e}", exc_info=True)
    
    def _run_hourly_news(self):
        """Run hourly news update"""
        if self.current_plan_id is None:
            logger.warning("No active plan for hourly news")
            return
        
        try:
            logger.info("Starting hourly news update")
            result = hourly_news(self.config, self.current_plan_id)
            logger.info(f"Hourly news update completed: {result}")
        
        except Exception as e:
            logger.error(f"Error in hourly news: {e}", exc_info=True)
    
    def _run_intraday_monitor(self):
        """Run intraday monitoring"""
        # Check if we're in active hours
        now = datetime.now(self.timezone)
        active_hours = self.config.scheduler.tasks.get('intraday_monitor', {}).get('active_hours', [])
        
        if now.hour not in active_hours:
            return
        
        if self.current_plan_id is None:
            logger.debug("No active plan for intraday monitoring")
            return
        
        try:
            logger.debug("Running intraday monitor")
            result = intraday_monitor(self.config, self.current_plan_id)
            
            if result.get('alerts_sent'):
                logger.info(f"Intraday alerts sent: {result}")
        
        except Exception as e:
            logger.error(f"Error in intraday monitor: {e}", exc_info=True)
    
    def _run_nightly_update(self):
        """Run nightly update"""
        try:
            logger.info("Starting nightly update")
            result = nightly_update(self.config)
            logger.info(f"Nightly update completed: {result}")
            
            # Clear current plan ID
            self.current_plan_id = None
        
        except Exception as e:
            logger.error(f"Error in nightly update: {e}", exc_info=True)
    
    def _run_weekly_retrain(self):
        """Run weekly model retraining"""
        try:
            logger.info("Starting weekly model retrain")
            # Placeholder for actual retraining logic
            logger.info("Weekly retrain completed (placeholder)")
        
        except Exception as e:
            logger.error(f"Error in weekly retrain: {e}", exc_info=True)
    
    def list_jobs(self):
        """List all scheduled jobs"""
        if self.scheduler:
            jobs = []
            for job in self.scheduler.get_jobs():
                jobs.append({
                    'id': job.id,
                    'name': job.name,
                    'next_run_time': job.next_run_time,
                    'trigger': str(job.trigger)
                })
            return jobs
        return []
