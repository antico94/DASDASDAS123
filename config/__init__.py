# config/__init__.py
import os
from config.development import DevelopmentConfig
from config.production import ProductionConfig
from config.testing import TestingConfig

# Create a mapping of environment names to config classes
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}

# Get the current environment from environment variable or default to development
ENV = os.getenv('TRADING_BOT_ENV', 'development')

# Create the active configuration instance
Config = config_by_name[ENV]()