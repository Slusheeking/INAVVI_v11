"""
Database Configuration for the Autonomous Trading System.

This module provides configuration for database connection.
"""

import os
import logging

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """
    Configuration for database connection.
    """
    
    def __init__(self):
        """
        Initialize the database configuration.
        """
        self.host = os.environ.get('TIMESCALEDB_HOST', 'localhost')
        self.port = os.environ.get('TIMESCALEDB_PORT', '5432')
        self.database = os.environ.get('TIMESCALEDB_DATABASE', 'ats_db')
        self.user = os.environ.get('TIMESCALEDB_USER', 'ats_user')
        self.password = os.environ.get('TIMESCALEDB_PASSWORD', 'ats_password')
        
        logger.info(f"Database configuration initialized: host={self.host}, port={self.port}, database={self.database}, user={self.user}")
    
    def get_connection_string(self):
        """
        Get the database connection string.
        
        Returns:
            str: Database connection string
        """
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    def get_connection_params(self):
        """
        Get the database connection parameters.
        
        Returns:
            dict: Database connection parameters
        """
        return {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.user,
            'password': self.password
        }
    
    def update_config(self, **kwargs):
        """
        Update the database configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated database configuration: {key}={value}")
            else:
                logger.warning(f"Unknown database configuration parameter: {key}")
