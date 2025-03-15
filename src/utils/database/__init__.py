"""
Database utilities for the trading system.

This module provides utilities for interacting with the database.
"""

import os
import logging
from typing import Dict, Any, Optional

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for interacting with TimescaleDB."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the database manager.
        
        Args:
            connection_string: Database connection string. If None, it will be constructed from environment variables.
        """
        if connection_string is None:
            # Construct connection string from environment variables
            host = os.environ.get("TIMESCALEDB_HOST", "localhost")
            port = os.environ.get("TIMESCALEDB_PORT", "5432")
            database = os.environ.get("TIMESCALEDB_DATABASE", "ats_db")
            user = os.environ.get("TIMESCALEDB_USER", "ats_user")
            password = os.environ.get("TIMESCALEDB_PASSWORD", "ats_password")
            
            connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        
        self.connection_string = connection_string
        self.engine = create_engine(connection_string)
        logger.info(f"Initialized database manager with connection to {host}:{port}/{database}")
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return the results as a DataFrame.
        
        Args:
            query: SQL query to execute
            params: Parameters for the query
            
        Returns:
            DataFrame with query results
        """
        try:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                
                # Convert to DataFrame
                df = pd.DataFrame(result.fetchall())
                
                # Set column names
                if not df.empty:
                    df.columns = result.keys()
                
                return df
        
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            # Return empty DataFrame on error
            return pd.DataFrame()
    
    def execute_statement(self, statement: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Execute a SQL statement (INSERT, UPDATE, DELETE, etc.).
        
        Args:
            statement: SQL statement to execute
            params: Parameters for the statement
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                if params:
                    conn.execute(text(statement), params)
                else:
                    conn.execute(text(statement))
                
                conn.commit()
                
                return True
        
        except Exception as e:
            logger.error(f"Error executing statement: {e}")
            return False
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = "append") -> bool:
        """
        Insert a DataFrame into a database table.
        
        Args:
            df: DataFrame to insert
            table_name: Name of the table
            if_exists: What to do if the table exists ('fail', 'replace', or 'append')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            df.to_sql(
                name=table_name,
                con=self.engine,
                if_exists=if_exists,
                index=False,
                method="multi",
                chunksize=10000
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Error inserting DataFrame into {table_name}: {e}")
            return False
    
    def check_connection(self) -> bool:
        """
        Check if the database connection is working.
        
        Returns:
            True if connection is working, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False

# Singleton instance
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """
    Get the database manager instance.
    
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
    
    return _db_manager