from sqlalchemy import inspect, schema
from data.db_session import DatabaseSession
from db_logger.models import Base


def ensure_log_schema_exists():
    """Ensure the 'logs' schema exists in the database."""
    engine = DatabaseSession._engine
    inspector = inspect(engine)

    if 'logs' not in inspector.get_schema_names():
        with engine.begin() as conn:
            conn.execute(schema.CreateSchema('logs'))
        print("Created 'logs' schema")


def create_log_tables():
    """Create log tables if they don't exist."""
    ensure_log_schema_exists()
    Base.metadata.create_all(DatabaseSession._engine)
    print("Created log tables")


def initialize_logging():
    """Initialize the logging system by creating schema and tables."""
    try:
        # Ensure we have a database connection
        session = DatabaseSession.get_session()
        session.close()

        # Create schema and tables
        ensure_log_schema_exists()
        create_log_tables()
        return True
    except Exception as e:
        print(f"Error initializing logging tables: {str(e)}")
        return False