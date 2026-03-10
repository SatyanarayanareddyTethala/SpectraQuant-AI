"""
Database session management and connection handling.
"""
import os
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

from .models import Base


class DatabaseManager:
    """Manages database connections and sessions"""
    
    def __init__(self, config: dict):
        """
        Initialize database manager with configuration.
        
        Args:
            config: Database configuration dictionary
        """
        self.config = config
        self.engine = None
        self.SessionLocal = None
        self._setup_engine()
    
    def _setup_engine(self):
        """Setup database engine and session factory"""
        # Get credentials from environment variables
        db_user = os.getenv(
            self.config.get('username_env', 'DB_USERNAME'),
            'postgres'
        )
        db_password = os.getenv(
            self.config.get('password_env', 'DB_PASSWORD'),
            'postgres'
        )
        
        # Build connection URL
        db_url = (
            f"postgresql://{db_user}:{db_password}"
            f"@{self.config.get('host', 'localhost')}"
            f":{self.config.get('port', 5432)}"
            f"/{self.config.get('database', 'trading_assistant')}"
        )
        
        # Create engine with connection pooling
        self.engine = create_engine(
            db_url,
            pool_size=self.config.get('pool_size', 5),
            max_overflow=self.config.get('max_overflow', 10),
            pool_pre_ping=True,  # Verify connections before using
            echo=False
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        Base.metadata.drop_all(bind=self.engine)
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.
        
        Yields:
            SQLAlchemy session
            
        Example:
            with db_manager.get_session() as session:
                result = session.query(Model).all()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_session_maker(self) -> sessionmaker:
        """
        Get the session factory for dependency injection.
        
        Returns:
            SQLAlchemy sessionmaker
        """
        return self.SessionLocal


# Global database manager instance (will be initialized by config)
_db_manager = None


def init_db(config: dict) -> DatabaseManager:
    """
    Initialize global database manager.
    
    Args:
        config: Database configuration dictionary
        
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    _db_manager = DatabaseManager(config)
    return _db_manager


def get_db_manager() -> DatabaseManager:
    """
    Get global database manager instance.
    
    Returns:
        DatabaseManager instance
        
    Raises:
        RuntimeError: If database not initialized
    """
    if _db_manager is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _db_manager


def get_db() -> Generator[Session, None, None]:
    """
    Dependency injection function for FastAPI.
    
    Yields:
        SQLAlchemy session
        
    Example:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    if _db_manager is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    with _db_manager.get_session() as session:
        yield session
