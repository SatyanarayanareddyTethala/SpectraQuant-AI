"""Database session management — supports PostgreSQL and SQLite."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker] = None


def get_engine(url: Optional[str] = None) -> Engine:
    """Return (and cache) a SQLAlchemy engine.

    Parameters
    ----------
    url : str, optional
        Database URL.  Defaults to an in-memory SQLite database.
    """
    global _engine
    if _engine is not None:
        return _engine

    db_url = url or "sqlite:///intelligence.db"
    connect_args = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    _engine = create_engine(
        db_url,
        pool_pre_ping=True,
        connect_args=connect_args,
    )
    logger.info("Database engine created: %s", db_url.split("@")[-1])
    return _engine


def get_session_factory(engine: Optional[Engine] = None) -> sessionmaker:
    """Return (and cache) a session factory bound to *engine*."""
    global _SessionFactory
    if _SessionFactory is not None:
        return _SessionFactory

    eng = engine or get_engine()
    _SessionFactory = sessionmaker(bind=eng, expire_on_commit=False)
    return _SessionFactory


def init_db(url: Optional[str] = None) -> Engine:
    """Initialise the engine and session factory.

    Also creates all tables defined in :mod:`spectraquant.intelligence.db.models`.
    """
    from spectraquant.intelligence.db.models import Base

    engine = get_engine(url)
    get_session_factory(engine)
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised (tables created)")
    return engine


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager yielding a scoped session with auto-commit/rollback."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI-compatible dependency that yields a database session."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
