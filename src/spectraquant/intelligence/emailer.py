"""Email notification for the Intelligence layer."""
from __future__ import annotations

import logging
import os
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Send alert emails over SMTP/TLS.

    Credentials are read from environment variables whose names are stored in
    the ``EmailConfig`` (``username_env``, ``password_env``).
    """

    def send_alert(
        self,
        subject: str,
        body: str,
        config: Dict[str, Any],
    ) -> bool:
        """Send a single email alert.

        Parameters
        ----------
        subject : str
            Email subject line.
        body : str
            Plain-text body (HTML also accepted – wrapped in ``<pre>``).
        config : dict
            Must contain ``smtp_host``, ``smtp_port``, ``username_env``,
            ``password_env``, ``from_addr``, ``to_addrs``.

        Returns
        -------
        bool
            *True* if the email was sent successfully.
        """
        smtp_host: str = config.get("smtp_host", "smtp.gmail.com")
        smtp_port: int = int(config.get("smtp_port", 587))
        username: str = os.environ.get(config.get("username_env", ""), "")
        password: str = os.environ.get(config.get("password_env", ""), "")
        from_addr: str = config.get("from_addr", username)
        to_addrs: List[str] = config.get("to_addrs", [])

        if not username or not password:
            logger.warning("SMTP credentials not set — skipping email send")
            return False
        if not to_addrs:
            logger.warning("No recipients configured — skipping email send")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_addrs)
        msg.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(username, password)
                server.sendmail(from_addr, to_addrs, msg.as_string())
            logger.info("Email sent: %s → %s", subject, to_addrs)
            return True
        except Exception:
            logger.exception("Failed to send email: %s", subject)
            return False

    def process_pending(
        self,
        session: "Session",  # noqa: F821
        config: Dict[str, Any],
    ) -> int:
        """Send all pending (unsent) alerts and update their status.

        Parameters
        ----------
        session : sqlalchemy.orm.Session
            Active DB session.
        config : dict
            Email configuration.

        Returns
        -------
        int
            Number of alerts successfully sent.
        """
        from spectraquant.intelligence.db.models import Alert

        pending: List[Any] = (
            session.query(Alert)
            .filter(Alert.status == "pending")
            .order_by(Alert.created_at)
            .all()
        )

        sent_count = 0
        now = datetime.now(tz=timezone.utc)

        for alert in pending:
            subject = f"[{alert.category}] {alert.symbol or 'System'}"
            body = alert.payload or ""
            ok = self.send_alert(subject, body, config)
            if ok:
                alert.status = "sent"
                alert.sent_at = now
                sent_count += 1
            else:
                alert.status = "failed"
                alert.sent_at = now

        if pending:
            session.commit()

        logger.info("Processed %d pending alerts, sent %d", len(pending), sent_count)
        return sent_count
