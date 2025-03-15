"""
Database utilities for the Autonomous Trading System.

This module provides utilities for database operations, including
connection management, query execution, and data manipulation.
"""

import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import pandas as pd
import psycopg2
from psycopg2 import pool
from psycopg2.extras import DictCursor, execute_batch, execute_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.utils.concurrency.concurrency_utils import ThreadSafeDict
from src.utils.logging import get_logger

logger = get_logger("utils.database.database_utils")

# Global connection pools
_pg_pools = ThreadSafeDict()
_sqlalchemy_engines = ThreadSafeDict()


def get_connection_string(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    driver: str = "postgresql",
) -> str:
    """
    Get a database connection string.
    
    Args:
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        password: Database password
        driver: Database driver
        
    Returns:
        Connection string
    """
    return f"{driver}://{user}:{password}@{host}:{port}/{database}"


def get_connection_string_from_env(
    prefix: str = "DB",
    driver: str = "postgresql",
) -> str:
    """
    Get a database connection string from environment variables.
    
    Args:
        prefix: Environment variable prefix
        driver: Database driver
        
    Returns:
        Connection string
    """
    host = os.getenv(f"{prefix}_HOST", "localhost")
    port = os.getenv(f"{prefix}_PORT", "5432")
    database = os.getenv(f"{prefix}_NAME", "postgres")
    user = os.getenv(f"{prefix}_USER", "postgres")
    password = os.getenv(f"{prefix}_PASSWORD", "postgres")
    
    return get_connection_string(host, int(port), database, user, password, driver)


def get_connection_pool(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    min_connections: int = 1,
    max_connections: int = 10,
    pool_key: Optional[str] = None,
) -> pool.ThreadedConnectionPool:
    """
    Get a connection pool.
    
    Args:
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        password: Database password
        min_connections: Minimum number of connections
        max_connections: Maximum number of connections
        pool_key: Key to identify the pool (default: host:port/database)
        
    Returns:
        Connection pool
    """
    # Generate a key for the pool if not provided
    if pool_key is None:
        pool_key = f"{host}:{port}/{database}"
    
    # Check if the pool already exists
    if pool_key in _pg_pools:
        return _pg_pools[pool_key]
    
    # Create a new pool
    try:
        new_pool = pool.ThreadedConnectionPool(
            min_connections,
            max_connections,
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
        )
        _pg_pools[pool_key] = new_pool
        logger.info(f"Created connection pool for {pool_key}")
        return new_pool
    except Exception as e:
        logger.error(f"Error creating connection pool for {pool_key}: {e}")
        raise


def get_connection_pool_from_env(
    prefix: str = "DB",
    min_connections: int = 1,
    max_connections: int = 10,
    pool_key: Optional[str] = None,
) -> pool.ThreadedConnectionPool:
    """
    Get a connection pool from environment variables.
    
    Args:
        prefix: Environment variable prefix
        min_connections: Minimum number of connections
        max_connections: Maximum number of connections
        pool_key: Key to identify the pool
        
    Returns:
        Connection pool
    """
    host = os.getenv(f"{prefix}_HOST", "localhost")
    port = int(os.getenv(f"{prefix}_PORT", "5432"))
    database = os.getenv(f"{prefix}_NAME", "postgres")
    user = os.getenv(f"{prefix}_USER", "postgres")
    password = os.getenv(f"{prefix}_PASSWORD", "postgres")
    
    return get_connection_pool(
        host, port, database, user, password, min_connections, max_connections, pool_key
    )


@contextmanager
def get_connection(
    pool_or_key: Union[pool.ThreadedConnectionPool, str],
    autocommit: bool = False,
) -> Generator[psycopg2.extensions.connection, None, None]:
    """
    Get a connection from a pool.
    
    Args:
        pool_or_key: Connection pool or pool key
        autocommit: Whether to enable autocommit
        
    Yields:
        Database connection
    """
    # Get the pool
    if isinstance(pool_or_key, str):
        if pool_or_key in _pg_pools:
            pool_obj = _pg_pools[pool_or_key]
        else:
            raise ValueError(f"Pool {pool_or_key} not found")
    else:
        pool_obj = pool_or_key
    
    # Get a connection from the pool
    conn = None
    try:
        conn = pool_obj.getconn()
        if autocommit:
            conn.autocommit = True
        yield conn
    finally:
        if conn is not None:
            pool_obj.putconn(conn)


@contextmanager
def get_cursor(
    conn_or_pool: Union[psycopg2.extensions.connection, pool.ThreadedConnectionPool, str],
    cursor_factory: Optional[type] = DictCursor,
    autocommit: bool = False,
) -> Generator[psycopg2.extensions.cursor, None, None]:
    """
    Get a cursor.
    
    Args:
        conn_or_pool: Connection, connection pool, or pool key
        cursor_factory: Cursor factory
        autocommit: Whether to enable autocommit
        
    Yields:
        Database cursor
    """
    # Check if we need to get a connection from a pool
    if isinstance(conn_or_pool, (pool.ThreadedConnectionPool, str)):
        with get_connection(conn_or_pool, autocommit) as conn:
            with conn.cursor(cursor_factory=cursor_factory) as cursor:
                yield cursor
    else:
        # Use the provided connection
        conn = conn_or_pool
        if autocommit:
            conn.autocommit = True
        with conn.cursor(cursor_factory=cursor_factory) as cursor:
            yield cursor


def execute_query(
    conn_or_pool: Union[psycopg2.extensions.connection, pool.ThreadedConnectionPool, str],
    query: str,
    params: Optional[Union[Tuple, Dict[str, Any], List[Dict[str, Any]]]] = None,
    fetch: bool = True,
    cursor_factory: Optional[type] = DictCursor,
    autocommit: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """
    Execute a query.
    
    Args:
        conn_or_pool: Connection, connection pool, or pool key
        query: SQL query
        params: Query parameters
        fetch: Whether to fetch results
        cursor_factory: Cursor factory
        autocommit: Whether to enable autocommit
        
    Returns:
        Query results or None
    """
    with get_cursor(conn_or_pool, cursor_factory, autocommit) as cursor:
        cursor.execute(query, params)
        
        if fetch:
            return cursor.fetchall()
        
        return None


def execute_batch_query(
    conn_or_pool: Union[psycopg2.extensions.connection, pool.ThreadedConnectionPool, str],
    query: str,
    params_list: List[Union[Tuple, Dict[str, Any]]],
    page_size: int = 1000,
    cursor_factory: Optional[type] = DictCursor,
    autocommit: bool = False,
) -> None:
    """
    Execute a batch query.
    
    Args:
        conn_or_pool: Connection, connection pool, or pool key
        query: SQL query
        params_list: List of query parameters
        page_size: Number of rows per batch
        cursor_factory: Cursor factory
        autocommit: Whether to enable autocommit
    """
    with get_cursor(conn_or_pool, cursor_factory, autocommit) as cursor:
        execute_batch(cursor, query, params_list, page_size=page_size)


def execute_values_query(
    conn_or_pool: Union[psycopg2.extensions.connection, pool.ThreadedConnectionPool, str],
    query: str,
    values: List[Tuple],
    template: Optional[str] = None,
    page_size: int = 1000,
    cursor_factory: Optional[type] = DictCursor,
    autocommit: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """
    Execute a query with values.
    
    Args:
        conn_or_pool: Connection, connection pool, or pool key
        query: SQL query
        values: List of value tuples
        template: Template string
        page_size: Number of rows per batch
        cursor_factory: Cursor factory
        autocommit: Whether to enable autocommit
        
    Returns:
        Query results or None
    """
    with get_cursor(conn_or_pool, cursor_factory, autocommit) as cursor:
        result = execute_values(
            cursor, query, values, template=template, page_size=page_size, fetch=True
        )
        return result


def get_sqlalchemy_engine(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    driver: str = "postgresql",
    engine_key: Optional[str] = None,
    **kwargs,
) -> Engine:
    """
    Get a SQLAlchemy engine.
    
    Args:
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        password: Database password
        driver: Database driver
        engine_key: Key to identify the engine (default: host:port/database)
        **kwargs: Additional arguments to pass to create_engine
        
    Returns:
        SQLAlchemy engine
    """
    # Generate a key for the engine if not provided
    if engine_key is None:
        engine_key = f"{host}:{port}/{database}"
    
    # Check if the engine already exists
    if engine_key in _sqlalchemy_engines:
        return _sqlalchemy_engines[engine_key]
    
    # Create a new engine
    try:
        connection_string = get_connection_string(host, port, database, user, password, driver)
        engine = create_engine(connection_string, **kwargs)
        _sqlalchemy_engines[engine_key] = engine
        logger.info(f"Created SQLAlchemy engine for {engine_key}")
        return engine
    except Exception as e:
        logger.error(f"Error creating SQLAlchemy engine for {engine_key}: {e}")
        raise


def get_sqlalchemy_engine_from_env(
    prefix: str = "DB",
    driver: str = "postgresql",
    engine_key: Optional[str] = None,
    **kwargs,
) -> Engine:
    """
    Get a SQLAlchemy engine from environment variables.
    
    Args:
        prefix: Environment variable prefix
        driver: Database driver
        engine_key: Key to identify the engine
        **kwargs: Additional arguments to pass to create_engine
        
    Returns:
        SQLAlchemy engine
    """
    host = os.getenv(f"{prefix}_HOST", "localhost")
    port = int(os.getenv(f"{prefix}_PORT", "5432"))
    database = os.getenv(f"{prefix}_NAME", "postgres")
    user = os.getenv(f"{prefix}_USER", "postgres")
    password = os.getenv(f"{prefix}_PASSWORD", "postgres")
    
    return get_sqlalchemy_engine(
        host, port, database, user, password, driver, engine_key, **kwargs
    )


def get_sqlalchemy_session(
    engine_or_key: Union[Engine, str],
) -> Session:
    """
    Get a SQLAlchemy session.
    
    Args:
        engine_or_key: SQLAlchemy engine or engine key
        
    Returns:
        SQLAlchemy session
    """
    # Get the engine
    if isinstance(engine_or_key, str):
        if engine_or_key in _sqlalchemy_engines:
            engine = _sqlalchemy_engines[engine_or_key]
        else:
            raise ValueError(f"Engine {engine_or_key} not found")
    else:
        engine = engine_or_key
    
    # Create a session
    Session = sessionmaker(bind=engine)
    return Session()


@contextmanager
def session_scope(
    engine_or_key: Union[Engine, str],
) -> Generator[Session, None, None]:
    """
    Context manager for SQLAlchemy sessions.
    
    Args:
        engine_or_key: SQLAlchemy engine or engine key
        
    Yields:
        SQLAlchemy session
    """
    session = get_sqlalchemy_session(engine_or_key)
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Error in session: {e}")
        raise
    finally:
        session.close()


def execute_sqlalchemy_query(
    engine_or_key: Union[Engine, str],
    query: str,
    params: Optional[Dict[str, Any]] = None,
    return_dataframe: bool = True,
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Execute a query using SQLAlchemy.
    
    Args:
        engine_or_key: SQLAlchemy engine or engine key
        query: SQL query
        params: Query parameters
        return_dataframe: Whether to return a DataFrame
        
    Returns:
        Query results as a DataFrame or list of dictionaries
    """
    # Get the engine
    if isinstance(engine_or_key, str):
        if engine_or_key in _sqlalchemy_engines:
            engine = _sqlalchemy_engines[engine_or_key]
        else:
            raise ValueError(f"Engine {engine_or_key} not found")
    else:
        engine = engine_or_key
    
    # Execute the query
    with engine.connect() as conn:
        result = conn.execute(text(query), params or {})
        
        if return_dataframe:
            return pd.DataFrame(result.fetchall(), columns=result.keys())
        else:
            return [dict(row) for row in result.fetchall()]


def dataframe_to_sql(
    df: pd.DataFrame,
    table_name: str,
    engine_or_key: Union[Engine, str],
    schema: Optional[str] = None,
    if_exists: str = "append",
    index: bool = False,
    chunksize: Optional[int] = None,
    **kwargs,
) -> None:
    """
    Write a DataFrame to a SQL table.
    
    Args:
        df: DataFrame to write
        table_name: Table name
        engine_or_key: SQLAlchemy engine or engine key
        schema: Schema name
        if_exists: What to do if the table exists ('fail', 'replace', 'append')
        index: Whether to include the index
        chunksize: Number of rows per chunk
        **kwargs: Additional arguments to pass to to_sql
    """
    # Get the engine
    if isinstance(engine_or_key, str):
        if engine_or_key in _sqlalchemy_engines:
            engine = _sqlalchemy_engines[engine_or_key]
        else:
            raise ValueError(f"Engine {engine_or_key} not found")
    else:
        engine = engine_or_key
    
    # Write the DataFrame to the table
    df.to_sql(
        table_name,
        engine,
        schema=schema,
        if_exists=if_exists,
        index=index,
        chunksize=chunksize,
        **kwargs,
    )


def sql_to_dataframe(
    query: str,
    engine_or_key: Union[Engine, str],
    params: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Read a SQL query into a DataFrame.
    
    Args:
        query: SQL query
        engine_or_key: SQLAlchemy engine or engine key
        params: Query parameters
        **kwargs: Additional arguments to pass to read_sql
        
    Returns:
        DataFrame
    """
    # Get the engine
    if isinstance(engine_or_key, str):
        if engine_or_key in _sqlalchemy_engines:
            engine = _sqlalchemy_engines[engine_or_key]
        else:
            raise ValueError(f"Engine {engine_or_key} not found")
    else:
        engine = engine_or_key
    
    # Read the query into a DataFrame
    return pd.read_sql(query, engine, params=params, **kwargs)


def create_table_if_not_exists(
    conn_or_pool: Union[psycopg2.extensions.connection, pool.ThreadedConnectionPool, str],
    table_name: str,
    columns: Dict[str, str],
    schema: Optional[str] = None,
    primary_key: Optional[Union[str, List[str]]] = None,
    indexes: Optional[Dict[str, List[str]]] = None,
    autocommit: bool = True,
) -> None:
    """
    Create a table if it doesn't exist.
    
    Args:
        conn_or_pool: Connection, connection pool, or pool key
        table_name: Table name
        columns: Dictionary mapping column names to column types
        schema: Schema name
        primary_key: Primary key column(s)
        indexes: Dictionary mapping index names to column lists
        autocommit: Whether to enable autocommit
    """
    # Build the table name
    full_table_name = f"{schema}.{table_name}" if schema else table_name
    
    # Build the column definitions
    column_defs = [f"{name} {type_}" for name, type_ in columns.items()]
    
    # Add the primary key
    if primary_key:
        if isinstance(primary_key, str):
            column_defs.append(f"PRIMARY KEY ({primary_key})")
        else:
            column_defs.append(f"PRIMARY KEY ({', '.join(primary_key)})")
    
    # Build the CREATE TABLE query
    query = f"CREATE TABLE IF NOT EXISTS {full_table_name} ({', '.join(column_defs)})"
    
    # Execute the query
    execute_query(conn_or_pool, query, autocommit=autocommit, fetch=False)
    
    # Create indexes
    if indexes:
        for index_name, index_columns in indexes.items():
            index_query = (
                f"CREATE INDEX IF NOT EXISTS {index_name} "
                f"ON {full_table_name} ({', '.join(index_columns)})"
            )
            execute_query(conn_or_pool, index_query, autocommit=autocommit, fetch=False)


def create_hypertable(
    conn_or_pool: Union[psycopg2.extensions.connection, pool.ThreadedConnectionPool, str],
    table_name: str,
    time_column: str,
    schema: Optional[str] = None,
    chunk_time_interval: Optional[str] = None,
    autocommit: bool = True,
) -> None:
    """
    Create a TimescaleDB hypertable.
    
    Args:
        conn_or_pool: Connection, connection pool, or pool key
        table_name: Table name
        time_column: Time column name
        schema: Schema name
        chunk_time_interval: Chunk time interval
        autocommit: Whether to enable autocommit
    """
    # Build the table name
    full_table_name = f"{schema}.{table_name}" if schema else table_name
    
    # Build the CREATE HYPERTABLE query
    query = f"SELECT create_hypertable('{full_table_name}', '{time_column}'"
    
    if chunk_time_interval:
        query += f", chunk_time_interval => interval '{chunk_time_interval}'"
    
    query += ", if_not_exists => TRUE)"
    
    # Execute the query
    execute_query(conn_or_pool, query, autocommit=autocommit, fetch=False)


def add_compression_policy(
    conn_or_pool: Union[psycopg2.extensions.connection, pool.ThreadedConnectionPool, str],
    table_name: str,
    compress_after: str,
    schema: Optional[str] = None,
    autocommit: bool = True,
) -> None:
    """
    Add a TimescaleDB compression policy.
    
    Args:
        conn_or_pool: Connection, connection pool, or pool key
        table_name: Table name
        compress_after: Compression interval
        schema: Schema name
        autocommit: Whether to enable autocommit
    """
    # Build the table name
    full_table_name = f"{schema}.{table_name}" if schema else table_name
    
    # Enable compression
    enable_query = f"ALTER TABLE {full_table_name} SET (timescaledb.compress = TRUE)"
    execute_query(conn_or_pool, enable_query, autocommit=autocommit, fetch=False)
    
    # Add compression policy
    policy_query = (
        f"SELECT add_compression_policy('{full_table_name}', "
        f"interval '{compress_after}')"
    )
    execute_query(conn_or_pool, policy_query, autocommit=autocommit, fetch=False)


def add_retention_policy(
    conn_or_pool: Union[psycopg2.extensions.connection, pool.ThreadedConnectionPool, str],
    table_name: str,
    drop_after: str,
    schema: Optional[str] = None,
    autocommit: bool = True,
) -> None:
    """
    Add a TimescaleDB retention policy.
    
    Args:
        conn_or_pool: Connection, connection pool, or pool key
        table_name: Table name
        drop_after: Retention interval
        schema: Schema name
        autocommit: Whether to enable autocommit
    """
    # Build the table name
    full_table_name = f"{schema}.{table_name}" if schema else table_name
    
    # Add retention policy
    query = (
        f"SELECT add_retention_policy('{full_table_name}', "
        f"interval '{drop_after}')"
    )
    execute_query(conn_or_pool, query, autocommit=autocommit, fetch=False)


def create_continuous_aggregate(
    conn_or_pool: Union[psycopg2.extensions.connection, pool.ThreadedConnectionPool, str],
    view_name: str,
    query: str,
    refresh_interval: str,
    start_offset: str,
    end_offset: str,
    schema: Optional[str] = None,
    autocommit: bool = True,
) -> None:
    """
    Create a TimescaleDB continuous aggregate.
    
    Args:
        conn_or_pool: Connection, connection pool, or pool key
        view_name: View name
        query: Aggregate query
        refresh_interval: Refresh interval
        start_offset: Start offset
        end_offset: End offset
        schema: Schema name
        autocommit: Whether to enable autocommit
    """
    # Build the view name
    full_view_name = f"{schema}.{view_name}" if schema else view_name
    
    # Create continuous aggregate
    create_query = (
        f"CREATE MATERIALIZED VIEW IF NOT EXISTS {full_view_name} "
        f"WITH (timescaledb.continuous) AS {query}"
    )
    execute_query(conn_or_pool, create_query, autocommit=autocommit, fetch=False)
    
    # Add refresh policy
    policy_query = (
        f"SELECT add_continuous_aggregate_policy('{full_view_name}', "
        f"start_offset => interval '{start_offset}', "
        f"end_offset => interval '{end_offset}', "
        f"schedule_interval => interval '{refresh_interval}')"
    )
    execute_query(conn_or_pool, policy_query, autocommit=autocommit, fetch=False)


def get_table_columns(
    conn_or_pool: Union[psycopg2.extensions.connection, pool.ThreadedConnectionPool, str],
    table_name: str,
    schema: Optional[str] = "public",
) -> List[Dict[str, Any]]:
    """
    Get the columns of a table.
    
    Args:
        conn_or_pool: Connection, connection pool, or pool key
        table_name: Table name
        schema: Schema name
        
    Returns:
        List of column information
    """
    query = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """
    
    return execute_query(conn_or_pool, query, (schema, table_name))


def get_table_size(
    conn_or_pool: Union[psycopg2.extensions.connection, pool.ThreadedConnectionPool, str],
    table_name: str,
    schema: Optional[str] = "public",
) -> Dict[str, Any]:
    """
    Get the size of a table.
    
    Args:
        conn_or_pool: Connection, connection pool, or pool key
        table_name: Table name
        schema: Schema name
        
    Returns:
        Table size information
    """
    query = """
        SELECT
            pg_size_pretty(pg_total_relation_size(%s)) AS total_size,
            pg_size_pretty(pg_relation_size(%s)) AS table_size,
            pg_size_pretty(pg_indexes_size(%s)) AS index_size,
            pg_size_pretty(pg_total_relation_size(%s) - pg_relation_size(%s)) AS external_size
    """
    
    full_table_name = f"{schema}.{table_name}"
    params = (full_table_name, full_table_name, full_table_name, full_table_name, full_table_name)
    
    result = execute_query(conn_or_pool, query, params)
    return result[0] if result else {}


def get_table_row_count(
    conn_or_pool: Union[psycopg2.extensions.connection, pool.ThreadedConnectionPool, str],
    table_name: str,
    schema: Optional[str] = "public",
    approximate: bool = True,
) -> int:
    """
    Get the number of rows in a table.
    
    Args:
        conn_or_pool: Connection, connection pool, or pool key
        table_name: Table name
        schema: Schema name
        approximate: Whether to use an approximate count
        
    Returns:
        Number of rows
    """
    full_table_name = f"{schema}.{table_name}"
    
    if approximate:
        query = """
            SELECT reltuples::bigint AS approximate_row_count
            FROM pg_class
            JOIN pg_namespace ON pg_namespace.oid = pg_class.relnamespace
            WHERE pg_class.relname = %s AND pg_namespace.nspname = %s
        """
        params = (table_name, schema)
    else:
        query = f"SELECT COUNT(*) AS exact_row_count FROM {full_table_name}"
        params = None
    
    result = execute_query(conn_or_pool, query, params)
    return result[0][0] if result else 0


def get_database_size(
    conn_or_pool: Union[psycopg2.extensions.connection, pool.ThreadedConnectionPool, str],
) -> Dict[str, Any]:
    """
    Get the size of the database.
    
    Args:
        conn_or_pool: Connection, connection pool, or pool key
        
    Returns:
        Database size information
    """
    query = """
        SELECT
            pg_size_pretty(pg_database_size(current_database())) AS database_size,
            pg_database_size(current_database()) AS database_size_bytes
    """
    
    result = execute_query(conn_or_pool, query)
    return result[0] if result else {}


def get_connection_info(
    conn_or_pool: Union[psycopg2.extensions.connection, pool.ThreadedConnectionPool, str],
) -> Dict[str, Any]:
    """
    Get information about the database connection.
    
    Args:
        conn_or_pool: Connection, connection pool, or pool key
        
    Returns:
        Connection information
    """
    query = """
        SELECT
            current_database() AS database,
            current_schema() AS schema,
            current_user AS user,
            version() AS version,
            inet_server_addr() AS server_address,
            inet_server_port() AS server_port
    """
    
    result = execute_query(conn_or_pool, query)
    return result[0] if result else {}


def wait_for_database(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    max_retries: int = 10,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> bool:
    """
    Wait for a database to become available.
    
    Args:
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        password: Database password
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase the delay by after each retry
        
    Returns:
        True if the database is available, False otherwise
    """
    retry_count = 0
    delay = retry_delay
    
    while retry_count < max_retries:
        try:
            # Try to connect to the database
            conn = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
                connect_timeout=5,
            )
            conn.close()
            logger.info(f"Database {database} on {host}:{port} is available")
            return True
        except Exception as e:
            logger.warning(
                f"Database {database} on {host}:{port} is not available yet: {e}"
            )
            retry_count += 1
            
            if retry_count < max_retries:
                logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                delay *= backoff_factor
    
    logger.error(
        f"Database {database} on {host}:{port} is not available after {max_retries} retries"
    )
    return False


def wait_for_database_from_env(
    prefix: str = "DB",
    max_retries: int = 10,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> bool:
    """
    Wait for a database to become available using environment variables.
    
    Args:
        prefix: Environment variable prefix
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase the delay by after each retry
        
    Returns:
        True if the database is available, False otherwise
    """
    host = os.getenv(f"{prefix}_HOST", "localhost")
    port = int(os.getenv(f"{prefix}_PORT", "5432"))
    database = os.getenv(f"{prefix}_NAME", "postgres")
    user = os.getenv(f"{prefix}_USER", "postgres")
    password = os.getenv(f"{prefix}_PASSWORD", "postgres")
    
    return wait_for_database(
        host, port, database, user, password, max_retries, retry_delay, backoff_factor
    )


def close_all_connections() -> None:
    """Close all database connections."""
    # Close all psycopg2 connection pools
    for key, pool_obj in _pg_pools.items():
        try:
            pool_obj.closeall()
            logger.info(f"Closed connection pool {key}")
        except Exception as e:
            logger.error(f"Error closing connection pool {key}: {e}")
    
    # Clear the pools dictionary
    _pg_pools.clear()
    
    # Close all SQLAlchemy engines
    for key, engine in _sqlalchemy_engines.items():
        try:
            engine.dispose()
            logger.info(f"Closed SQLAlchemy engine {key}")
        except Exception as e:
            logger.error(f"Error closing SQLAlchemy engine {key}: {e}")
    
    # Clear the engines dictionary
    _sqlalchemy_engines.clear()