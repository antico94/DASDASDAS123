# data/db_session.py
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from config import Config

# Create base class for all models
Base = declarative_base()


class DatabaseSession:
    """Database session manager for SQLAlchemy."""

    _engine = None
    _session_factory = None
    _logger = logging.getLogger(__name__)

    @classmethod
    def initialize(cls, database_uri=None):
        """Initialize database engine and session factory.

        Args:
            database_uri (str, optional): Database connection URI.
                                         Defaults to Config.DATABASE_URI.
        """
        if database_uri is None:
            database_uri = Config.DATABASE_URI

        # Use standard logging first before db_logger is available
        cls._logger.info(f"Initializing database connection")

        try:
            # Create engine with appropriate settings
            cls._engine = create_engine(
                database_uri,
                pool_pre_ping=True,  # Test connections before using them
                pool_recycle=3600,  # Recycle connections after an hour
                echo=False,  # Disable SQL logging regardless of log level
                echo_pool=False  # Disable connection pool logging
            )

            # Create session factory
            cls._session_factory = scoped_session(
                sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    bind=cls._engine
                )
            )

            cls._logger.debug("Database session initialized successfully")

            # Once initialization is complete, we can try to use DBLogger
            # But we need to avoid a circular import during bootstrapping
            from db_logger.db_logger import DBLogger

            try:
                DBLogger.log_event("INFO", "Database session initialized successfully", "DatabaseSession")
            except ImportError:
                # If DBLogger is not yet available, we've already logged with standard logging
                pass

        except Exception as e:
            cls._logger.error(f"Failed to initialize database: {str(e)}")

            # Try to use DBLogger if available
            from db_logger.db_logger import DBLogger

            try:
                DBLogger.log_error("DatabaseSession", "Failed to initialize database", exception=e)
            except ImportError:
                # DBLogger not available during bootstrapping - we've already logged with standard logging
                pass

            raise

    @classmethod
    def create_tables(cls):
        """Create all tables defined in SQLAlchemy models."""
        if cls._engine is None:
            cls.initialize()

        cls._logger.info("Creating database tables...")
        Base.metadata.create_all(cls._engine)
        cls._logger.info("Database tables created successfully")

        # Try to use DBLogger if available
        try:
            from db_logger.db_logger import DBLogger
            DBLogger.log_event("INFO", "Database tables created successfully", "DatabaseSession")
        except ImportError:
            # DBLogger not available - we've already logged with standard logging
            pass

    @classmethod
    def get_session(cls):
        """Get a scoped database session.

        Returns:
            sqlalchemy.orm.Session: A scoped database session
        """
        if cls._session_factory is None:
            cls.initialize()

        return cls._session_factory()

    @classmethod
    def close_session(cls):
        """Close the current scoped session."""
        if cls._session_factory is not None:
            cls._session_factory.remove()
            cls._logger.debug("Database session closed")

            # Try to use DBLogger if available
            try:
                from db_logger.db_logger import DBLogger
                DBLogger.log_event("DEBUG", "Database session closed", "DatabaseSession")
            except ImportError:
                # DBLogger not available - we've already logged with standard logging
                pass