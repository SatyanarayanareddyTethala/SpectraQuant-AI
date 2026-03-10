"""
Email notification system with SMTP.
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from sqlalchemy.orm import Session

from ..db import crud


class EmailNotifier:
    """Handles email notifications via SMTP"""
    
    def __init__(self, config: dict):
        """
        Initialize email notifier.
        
        Args:
            config: Email configuration dictionary
        """
        self.config = config
        self.smtp_host = config.get('smtp_host')
        self.smtp_port = config.get('smtp_port', 587)
        self.use_tls = config.get('use_tls', True)
        self.from_email = config.get('from_email')
        self.subject_prefix = config.get('subject_prefix', '[SpectraQuant]')
        
        # Get credentials from environment
        username_env = config.get('username_env', 'SMTP_USERNAME')
        password_env = config.get('password_env', 'SMTP_PASSWORD')
        self.username = os.getenv(username_env)
        self.password = os.getenv(password_env)
        
        # Setup Jinja2 template environment
        template_dir = Path(__file__).parent / 'templates'
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    def send_email(
        self,
        to_emails: List[str],
        subject: str,
        html_body: str,
        text_body: Optional[str] = None
    ) -> bool:
        """
        Send email via SMTP.
        
        Args:
            to_emails: List of recipient emails
            subject: Email subject
            html_body: HTML email body
            text_body: Plain text email body (optional)
            
        Returns:
            True if sent successfully
        """
        if not self.username or not self.password:
            print("SMTP credentials not configured. Skipping email.")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"{self.subject_prefix} {subject}"
            msg['From'] = self.from_email
            msg['To'] = ', '.join(to_emails)
            
            # Add text and HTML parts
            if text_body:
                part1 = MIMEText(text_body, 'plain')
                msg.attach(part1)
            
            part2 = MIMEText(html_body, 'html')
            msg.attach(part2)
            
            # Connect to SMTP server
            if self.use_tls:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port)
            
            # Login and send
            server.login(self.username, self.password)
            server.sendmail(self.from_email, to_emails, msg.as_string())
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def send_plan_email(
        self,
        db: Session,
        plan_data: Dict[str, Any],
        to_emails: List[str]
    ) -> bool:
        """
        Send premarket plan email.
        
        Args:
            db: Database session
            plan_data: Plan data dictionary
            to_emails: Recipient emails
            
        Returns:
            True if sent successfully
        """
        template = self.jinja_env.get_template('plan_email.j2')
        html_body = template.render(plan=plan_data)
        
        subject = f"Premarket Plan - {plan_data['plan_date']}"
        
        success = self.send_email(to_emails, subject, html_body)
        
        return success
    
    def send_execute_now_email(
        self,
        db: Session,
        alert_data: Dict[str, Any],
        to_emails: List[str]
    ) -> bool:
        """
        Send EXECUTE NOW alert email.
        
        Args:
            db: Database session
            alert_data: Alert data dictionary
            to_emails: Recipient emails
            
        Returns:
            True if sent successfully
        """
        template = self.jinja_env.get_template('execute_now_email.j2')
        html_body = template.render(alert=alert_data)
        
        symbol = alert_data.get('symbol', 'UNKNOWN')
        subject = f"EXECUTE NOW - {symbol}"
        
        success = self.send_email(to_emails, subject, html_body)
        
        return success
    
    def send_hourly_news_email(
        self,
        db: Session,
        news_data: Dict[str, Any],
        to_emails: List[str]
    ) -> bool:
        """
        Send hourly news update email.
        
        Args:
            db: Database session
            news_data: News data dictionary
            to_emails: Recipient emails
            
        Returns:
            True if sent successfully
        """
        template = self.jinja_env.get_template('hourly_news_email.j2')
        html_body = template.render(news=news_data)
        
        hour = news_data.get('hour', 'Unknown')
        subject = f"Hourly News Update - {hour}"
        
        success = self.send_email(to_emails, subject, html_body)
        
        return success
    
    def process_pending_alerts(self, db: Session) -> int:
        """
        Process all pending alerts in database.
        
        Args:
            db: Database session
            
        Returns:
            Number of alerts processed
        """
        pending_alerts = crud.get_pending_alerts(db)
        processed = 0
        
        for alert in pending_alerts:
            success = False
            
            if alert.alert_type == 'PLAN':
                success = self.send_plan_email(
                    db, alert.payload_json, alert.email_to
                )
            elif alert.alert_type == 'EXEC':
                success = self.send_execute_now_email(
                    db, alert.payload_json, alert.email_to
                )
            elif alert.alert_type == 'NEWS':
                success = self.send_hourly_news_email(
                    db, alert.payload_json, alert.email_to
                )
            
            # Update alert status
            if success:
                crud.update_alert_status(
                    db, alert.alert_id, 'sent', datetime.utcnow()
                )
                processed += 1
            else:
                crud.update_alert_status(db, alert.alert_id, 'failed')
        
        return processed
