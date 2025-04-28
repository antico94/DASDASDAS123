# db_logger/setup.py
import json # Keep import in case it's needed elsewhere, though not used in provided snippet
import traceback # Import traceback for detailed error printing
from sqlalchemy import inspect, schema, exc as sa_exc # Import sa_exc for specific SQLAlchemy errors
from data.db_session import DatabaseSession # Assume DatabaseSession provides _engine
# Assuming Base is imported from models.py in the same package
from db_logger.models import Base 


def ensure_log_schema_exists():
    """Ensure the 'logs' schema exists in the database."""
    print("DB_LOGGER_SETUP: Attempting to ensure 'logs' schema exists...") # Debug print
    engine = DatabaseSession._engine
    inspector = inspect(engine)

    if 'logs' not in inspector.get_schema_names():
        print("DB_LOGGER_SETUP: 'logs' schema not found. Creating...") # Debug print
        try:
            # Use isolation_level='AUTOCOMMIT' if needed for schema creation outside transaction
            # but engine.begin() is usually sufficient
            with engine.begin() as conn:
                 # Check schema existence again inside the transaction to be safe in some DBs
                 # (though inspect is usually fine outside)
                 # Example check if needed:
                 # result = conn.execute(text("SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = 'logs'"))
                 # if result.fetchone() is None:
                 conn.execute(schema.CreateSchema('logs'))
            print("DB_LOGGER_SETUP: Successfully created 'logs' schema") # Debug print
        except Exception as e:
             print(f"DB_LOGGER_SETUP: ERROR: Failed to create 'logs' schema: {e}") # Debug print
             print(traceback.format_exc()) # Print traceback
             raise # Re-raise the exception to halt initialization if schema creation fails
    else:
        print("DB_LOGGER_SETUP: 'logs' schema already exists") # Debug print


def create_log_tables():
    """Create log tables if they don't exist."""
    print("DB_LOGGER_SETUP: Attempting to create log tables...") # Debug print
    try:
        # Ensure schema exists first
        ensure_log_schema_exists()

        # Create all tables defined in Base.metadata within the specified schema
        # This requires models to correctly define __tablename__ and __table_args__={'schema': 'logs'}
        Base.metadata.create_all(DatabaseSession._engine)
        print("DB_LOGGER_SETUP: Successfully called Base.metadata.create_all. Tables created (or they already existed).") # Debug print
    except Exception as e:
        print(f"DB_LOGGER_SETUP: ERROR: Failed to create log tables: {e}") # Debug print
        print(traceback.format_exc()) # Print traceback
        raise # Re-raise the exception


def initialize_logging():
    """Initialize the logging system by creating schema and tables."""
    print("DB_LOGGER_SETUP: Attempting to initialize logging schema and tables...") # Debug print
    # Removed the initial session check - Base.metadata.create_all connects directly
    try:
        # Create schema and tables
        create_log_tables()
        print("DB_LOGGER_SETUP: Logging initialization process completed.") # Debug print
        return True
    except Exception as e:
        # Catch specific SQLAlchemy errors during creation if needed, but general Exception is fine
        print(f"DB_LOGGER_SETUP: FATAL ERROR during logging initialization: {str(e)}")
        print("DB_LOGGER_SETUP: Printing traceback for initialization error:")
        print(traceback.format_exc()) # Print the full traceback for debugging
        # Do NOT re-raise here if main.py is designed to continue without logging
        # If logging is critical, you might want to sys.exit(1) or re-raise
        return False # Return False indicating initialization failed