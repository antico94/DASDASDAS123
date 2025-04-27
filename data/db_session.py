# data/db_session.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from config import Config
from custom_logging.logger import app_logger

# Create base class for all models
Base = declarative_base()


class DatabaseSession:
    """Database session manager for SQLAlchemy."""

    _engine = None
    _session_factory = None

    @classmethod
    def initialize(cls, database_uri=None):
        """Initialize database engine and session factory.

        Args:
            database_uri (str, optional): Database connection URI.
                                         Defaults to Config.DATABASE_URI.
        """
        if database_uri is None:
            database_uri = Config.DATABASE_URI

        app_logger.info(f"Initializing database connection")

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

            app_logger.debug("Database session initialized successfully")

        except Exception as e:
            app_logger.error(f"Failed to initialize database: {str(e)}")
            raise

    @classmethod
    def create_tables(cls):
        """Create all tables defined in SQLAlchemy models."""
        if cls._engine is None:
            cls.initialize()

        app_logger.info("Creating database tables...")
        Base.metadata.create_all(cls._engine)
        app_logger.info("Database tables created successfully")

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
            app_logger.debug("Database session closed")