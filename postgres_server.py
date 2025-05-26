from typing import Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from mcp.server.fastmcp import FastMCP
import sys
import logging
import os
import argparse
import signal
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, # Changed to DEBUG to see more logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('postgres-mcp-server')

# Retry configuration from environment variables
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", 3))
RETRY_WAIT_MULTIPLIER = int(os.getenv("RETRY_WAIT_MULTIPLIER", 1)) # in seconds
RETRY_WAIT_MAX = int(os.getenv("RETRY_WAIT_MAX", 10)) # in seconds

logger.info(
    f"Retry configuration: Attempts={RETRY_ATTEMPTS}, Wait Multiplier={RETRY_WAIT_MULTIPLIER}s, Max Wait={RETRY_WAIT_MAX}s"
)

def log_retry_attempt(retry_state):
    """Log details of a retry attempt."""
    logger.warning(
        f"Retrying {retry_state.fn.__name__} due to {retry_state.outcome.exception()}, "
        f"attempt {retry_state.attempt_number}/{RETRY_ATTEMPTS}..."
    )

# Initialize server with capabilities
mcp = FastMCP(
    "PostgreSQL Explorer",
    capabilities={
        "tools": True,      # Enable tool support
        "logging": True,    # Enable logging support
        "resources": False, # We don't use resources
        "prompts": False   # We don't use prompts
    }
)

# Connection string from --conn flag or POSTGRES_CONNECTION_STRING env var
parser = argparse.ArgumentParser(description="PostgreSQL Explorer MCP server")
parser.add_argument(
    "--conn",
    dest="conn",
    default=os.getenv("POSTGRES_CONNECTION_STRING"),
    help="PostgreSQL connection string or DSN"
)
args, _ = parser.parse_known_args()
CONNECTION_STRING: Optional[str] = args.conn

logger.info(
    "Starting PostgreSQL MCP server – connection %s",
    ("to " + CONNECTION_STRING.split('@')[1]) if CONNECTION_STRING and '@' in CONNECTION_STRING else "(not set)"
)

# Initialize connection pool
pool = None
if CONNECTION_STRING:
    try:
        MIN_CONNS = int(os.getenv("POSTGRES_POOL_MIN_CONNS", 1))
        MAX_CONNS = int(os.getenv("POSTGRES_POOL_MAX_CONNS", 5))
        pool = SimpleConnectionPool(MIN_CONNS, MAX_CONNS, CONNECTION_STRING)
        logger.info(f"Connection pool initialized: min_conns={MIN_CONNS}, max_conns={MAX_CONNS}")
    except Exception as e:
        logger.error(f"Failed to initialize connection pool: {str(e)}")
        pool = None # Ensure pool is None if initialization fails
else:
    logger.warning("CONNECTION_STRING not set. Connection pooling will not be used.")


@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=RETRY_WAIT_MULTIPLIER, max=RETRY_WAIT_MAX),
    before_sleep=log_retry_attempt,
    retry=lambda retry_state: isinstance(retry_state.outcome.exception(), (psycopg2.OperationalError, psycopg2.pool.PoolError)),
    reraise=True  # Reraise the exception if all retries fail
)
def get_connection():
    if not pool:
        # Fallback to old behavior if pool is not initialized
        if not CONNECTION_STRING:
            # This case should ideally not be retried as it's a configuration issue.
            # However, the retry decorator is applied to the whole function.
            # For a non-retryable config error, it will try RETRY_ATTEMPTS times then raise.
            logger.error("POSTGRES_CONNECTION_STRING is not set and pool is not initialized.")
            raise RuntimeError(
                "POSTGRES_CONNECTION_STRING is not set and pool is not initialized. "
                "Provide --conn DSN or export POSTGRES_CONNECTION_STRING."
            )
        try:
            logger.warning("Connection pool not available, creating a new connection.")
            conn = psycopg2.connect(CONNECTION_STRING)
            logger.debug("Database connection established successfully (non-pooled)")
            return conn
        except psycopg2.OperationalError as e: # Specific exception for retry
            logger.error(f"Failed to establish database connection (non-pooled): {str(e)}")
            raise # Reraise to trigger retry
        except Exception as e: # Other non-retryable exceptions
            logger.error(f"Non-retryable error establishing database connection (non-pooled): {str(e)}")
            raise

    try:
        conn = pool.getconn()
        logger.debug("Connection retrieved from pool")
        return conn
    except psycopg2.pool.PoolError as e: # Specific exception for retry on pool errors
        logger.error(f"Failed to get connection from pool: {str(e)}")
        raise # Reraise to trigger retry
    except Exception as e: # Other non-retryable exceptions
        logger.error(f"Non-retryable error getting connection from pool: {str(e)}")
        raise


@mcp.tool()
def query(sql: str, parameters: Optional[list] = None) -> str:
    """Execute a SQL query against the PostgreSQL database."""
    conn = None
    try:
        conn = get_connection()
    except RetryError as e: # Catch tenacity's RetryError after all attempts for get_connection failed
        logger.error(f"Failed to get connection after {RETRY_ATTEMPTS} attempts: {str(e)}")
        return f"Failed to connect to database after multiple retries: {str(e.last_attempt.exception())}"
    except RuntimeError as e: # Catch RuntimeError from get_connection if not configured (e.g. no CONN_STRING)
        logger.error(f"Failed to get connection (configuration error): {str(e)}")
        return str(e)
    except Exception as e: # Catch any other unexpected errors from get_connection
        logger.error(f"An unexpected error occurred while getting connection: {str(e)}")
        return f"Error connecting to database: {str(e)}"

    logger.info(f"Executing query: {sql[:100]}{'...' if len(sql) > 100 else ''}")

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_WAIT_MULTIPLIER, max=RETRY_WAIT_MAX),
        before_sleep=log_retry_attempt,
        retry=lambda retry_state: isinstance(retry_state.outcome.exception(), psycopg2.OperationalError) and \
                                  "read-only transaction" not in str(retry_state.outcome.exception()).lower() and \
                                  "syntax error" not in str(retry_state.outcome.exception()).lower(), # Add more non-retryable conditions if needed
        reraise=True
    )
    def execute_query_with_retry(current_conn, sql_query, params):
        # Use RealDictCursor for better handling of special characters in column names
        with current_conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Properly escape the query using mogrify
            if params:
                query_string = cur.mogrify(sql_query, params).decode('utf-8')
                logger.debug(f"Query with parameters: {query_string}")
            else:
                query_string = sql_query
            
            # Execute the escaped query
            cur.execute(query_string)
            
            # For non-SELECT queries
            if cur.description is None:
                current_conn.commit()
                affected_rows = cur.rowcount
                logger.info(f"Non-SELECT query executed successfully. Rows affected: {affected_rows}")
                return f"Query executed successfully. Rows affected: {affected_rows}"
            
            # For SELECT queries
            rows = cur.fetchall()
            if not rows:
                logger.info("Query returned no results")
                return "No results found"
            
            logger.info(f"Query returned {len(rows)} rows")
            
            # Format results with proper string escaping
            result_lines = ["Results:", "--------"]
            for row in rows:
                try:
                    # Convert each value to string safely
                    line_items = []
                    for key, val in row.items():
                        if val is None:
                            formatted_val = "NULL"
                        elif isinstance(val, (bytes, bytearray)):
                            formatted_val = val.decode('utf-8', errors='replace')
                        else:
                            formatted_val = str(val).replace('%', '%%')
                        line_items.append(f"{key}: {formatted_val}")
                    result_lines.append(" | ".join(line_items))
                except Exception as row_error:
                    error_msg = f"Error formatting row: {str(row_error)}"
                    logger.error(error_msg)
                    result_lines.append(error_msg)
                    continue
            
            return "\n".join(result_lines)

    try:
        return execute_query_with_retry(conn, sql, parameters)
    except RetryError as e: # Catch tenacity's RetryError after all attempts for execute_query_with_retry failed
        logger.error(f"Query failed after {RETRY_ATTEMPTS} attempts: {str(e.last_attempt.exception())}\nQuery: {sql}")
        if conn: conn.rollback() # Rollback on final query execution error
        return f"Query error after multiple retries: {str(e.last_attempt.exception())}\nQuery: {sql}"
    except psycopg2.Error as pe: # Catch specific psycopg2 errors that were not retried or failed after retry
        error_msg = f"Query error (psycopg2): {str(pe)}\nQuery: {sql}"
        logger.error(error_msg)
        if conn: conn.rollback()
        return error_msg
    except Exception as exec_error: # Catch any other unexpected errors during query execution
        error_msg = f"Unexpected query error: {str(exec_error)}\nQuery: {sql}"
        logger.error(error_msg)
        if conn: conn.rollback() # Rollback on query execution error
        return error_msg
    finally:
        if conn:
            if pool:
                try:
                    pool.putconn(conn)
                    logger.debug("Connection returned to pool")
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {str(e)}")
                    # If returning to pool fails, close the connection directly
                    try:
                        conn.close()
                        logger.warning("Connection closed directly after failing to return to pool.")
                    except Exception as close_err:
                        logger.error(f"Error closing connection directly: {str(close_err)}")
            else:
                # If pool was never initialized, close the connection as before
                try:
                    conn.close()
                    logger.debug("Database connection closed (non-pooled)")
                except Exception as e:
                    logger.error(f"Error closing non-pooled connection: {str(e)}")


@mcp.tool()
def list_schemas() -> str:
    """List all schemas in the database."""
    logger.info("Listing database schemas")
    return query("SELECT schema_name FROM information_schema.schemata ORDER BY schema_name")

@mcp.tool()
def list_tables(db_schema: str = 'public') -> str:
    """List all tables in a specific schema.
    
    Args:
        db_schema: The schema name to list tables from (defaults to 'public')
    """
    logger.info(f"Listing tables in schema: {db_schema}")
    sql = """
    SELECT table_name, table_type
    FROM information_schema.tables
    WHERE table_schema = %s
    ORDER BY table_name
    """
    return query(sql, [db_schema])

@mcp.tool()
def describe_table(table_name: str, db_schema: str = 'public') -> str:
    """Get detailed information about a table.
    
    Args:
        table_name: The name of the table to describe
        db_schema: The schema name (defaults to 'public')
    """
    logger.info(f"Describing table: {db_schema}.{table_name}")
    sql = """
    SELECT 
        column_name,
        data_type,
        is_nullable,
        column_default,
        character_maximum_length
    FROM information_schema.columns
    WHERE table_schema = %s AND table_name = %s
    ORDER BY ordinal_position
    """
    return query(sql, [db_schema, table_name])

@mcp.tool()
def get_foreign_keys(table_name: str, db_schema: str = 'public') -> str:
    """Get foreign key information for a table.
    
    Args:
        table_name: The name of the table to get foreign keys from
        db_schema: The schema name (defaults to 'public')
    """
    logger.info(f"Getting foreign keys for table: {db_schema}.{table_name}")
    sql = """
    SELECT 
        tc.constraint_name,
        kcu.column_name as fk_column,
        ccu.table_schema as referenced_schema,
        ccu.table_name as referenced_table,
        ccu.column_name as referenced_column
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
        AND tc.table_schema = kcu.table_schema
    JOIN information_schema.referential_constraints rc
        ON tc.constraint_name = rc.constraint_name
    JOIN information_schema.constraint_column_usage ccu
        ON rc.unique_constraint_name = ccu.constraint_name
    WHERE tc.constraint_type = 'FOREIGN KEY'
        AND tc.table_schema = %s
        AND tc.table_name = %s
    ORDER BY tc.constraint_name, kcu.ordinal_position
    """
    return query(sql, [db_schema, table_name])

@mcp.tool()
def find_relationships(table_name: str, db_schema: str = 'public') -> str:
    """Find both explicit and implied relationships for a table.
    
    Args:
        table_name: The name of the table to analyze relationships for
        db_schema: The schema name (defaults to 'public')
    """
    logger.info(f"Finding relationships for table: {db_schema}.{table_name}")
    try:
        # First get explicit foreign key relationships
        fk_sql = """
        SELECT 
            kcu.column_name,
            ccu.table_name as foreign_table,
            ccu.column_name as foreign_column,
            'Explicit FK' as relationship_type,
            1 as confidence_level
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu 
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
            ON ccu.constraint_name = tc.constraint_name
            AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = %s
            AND tc.table_name = %s
        """
        
        logger.debug("Querying explicit foreign key relationships")
        explicit_results = query(fk_sql, [db_schema, table_name])
        
        # Then look for implied relationships based on common patterns
        logger.debug("Querying implied relationships")
        implied_sql = """
        WITH source_columns AS (
            -- Get all ID-like columns from our table
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = %s 
            AND table_name = %s
            AND (
                column_name LIKE '%%id' 
                OR column_name LIKE '%%_id'
                OR column_name LIKE '%%_fk'
            )
        ),
        potential_references AS (
            -- Find tables that might be referenced by our ID columns
            SELECT DISTINCT
                sc.column_name as source_column,
                sc.data_type as source_type,
                t.table_name as target_table,
                c.column_name as target_column,
                c.data_type as target_type,
                CASE
                    -- Highest confidence: column matches table_id pattern and types match
                    WHEN sc.column_name = t.table_name || '_id' 
                        AND sc.data_type = c.data_type THEN 2
                    -- High confidence: column ends with _id and types match
                    WHEN sc.column_name LIKE '%%_id' 
                        AND sc.data_type = c.data_type THEN 3
                    -- Medium confidence: column contains table name and types match
                    WHEN sc.column_name LIKE '%%' || t.table_name || '%%'
                        AND sc.data_type = c.data_type THEN 4
                    -- Lower confidence: column ends with id and types match
                    WHEN sc.column_name LIKE '%%id'
                        AND sc.data_type = c.data_type THEN 5
                END as confidence_level
            FROM source_columns sc
            CROSS JOIN information_schema.tables t
            JOIN information_schema.columns c 
                ON c.table_schema = t.table_schema 
                AND c.table_name = t.table_name
                AND (c.column_name = 'id' OR c.column_name = sc.column_name)
            WHERE t.table_schema = %s
                AND t.table_name != %s  -- Exclude self-references
        )
        SELECT 
            source_column as column_name,
            target_table as foreign_table,
            target_column as foreign_column,
            CASE 
                WHEN confidence_level = 2 THEN 'Strong implied relationship (exact match)'
                WHEN confidence_level = 3 THEN 'Strong implied relationship (_id pattern)'
                WHEN confidence_level = 4 THEN 'Likely implied relationship (name match)'
                ELSE 'Possible implied relationship'
            END as relationship_type,
            confidence_level
        FROM potential_references
        WHERE confidence_level IS NOT NULL
        ORDER BY confidence_level, source_column;
        """
        implied_results = query(implied_sql, [db_schema, table_name])
        
        return "Explicit Relationships:\n" + explicit_results + "\n\nImplied Relationships:\n" + implied_results
        
    except Exception as e:
        error_msg = f"Error finding relationships: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Signal handler for graceful shutdown
def graceful_shutdown(signum, frame):
    signal_name = signal.Signals(signum).name
    logger.info(f"Received shutdown signal ({signal_name}), initiating graceful shutdown...")
    if pool:
        try:
            pool.closeall()
            logger.info("Connection pool closed successfully during graceful shutdown.")
        except Exception as e:
            logger.error(f"Error closing connection pool during graceful shutdown: {str(e)}")
    else:
        logger.info("No active connection pool to close during graceful shutdown.")
    
    # It's important to call mcp.stop() to allow FastMCP to shutdown its components if needed.
    # This might involve stopping the WebSocket server, etc.
    # We'll assume mcp.stop() is a blocking call or handles its own threading for shutdown.
    if mcp and hasattr(mcp, 'stop') and callable(mcp.stop):
        try:
            logger.info("Attempting to stop MCP server...")
            mcp.stop() # Assuming FastMCP has a stop method for clean shutdown
            logger.info("MCP server stop requested.")
        except Exception as e:
            logger.error(f"Error during mcp.stop(): {str(e)}")

    logger.info("Graceful shutdown complete. Exiting.")
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    try:
        logger.info("Starting MCP Postgres server...")
        mcp.run()  # This is typically a blocking call
    except SystemExit as e:
        # This allows sys.exit(0) from graceful_shutdown to be caught and handled cleanly.
        logger.info(f"SystemExit caught with code {e.code}. Application terminating.")
        # The finally block will still execute.
    except Exception as e:
        logger.error(f"Unhandled server error: {str(e)}", exc_info=True)
    finally:
        # This block now primarily serves as a fallback for unexpected exits
        # or if mcp.run() finishes without a sys.exit (e.g. if it's non-blocking and main thread ends)
        # Graceful shutdown via signals should handle pool closure before this.
        logger.info("Executing main finally block...")
        if pool:
            # Check if pool might have already been closed by signal handler to avoid errors
            # Simple check: if pool object still has a closeall method (it might be None or replaced)
            # A more robust check might involve a flag set by the signal handler.
            # For now, we assume pool.closeall() is idempotent or handles being called multiple times.
            try:
                # If pool.closeall() was already called, this might do nothing or error
                # depending on psycopg2's pool implementation. Let's assume it's safe.
                logger.info("Main finally block: Ensuring connection pool is closed.")
                pool.closeall() # Attempt to close again, just in case.
                logger.info("Main finally block: Connection pool closeall called.")
            except Exception as e:
                logger.error(f"Main finally block: Error closing connection pool: {str(e)}")
        else:
            logger.info("Main finally block: No active connection pool to close.")
        
        # If sys.exit(0) was called by graceful_shutdown, we don't want to override with sys.exit(1)
        # This is tricky because sys.exit() raises SystemExit.
        # The current structure will exit with 0 if graceful_shutdown completed.
        # If an error occurred before graceful_shutdown, it would exit with 1 (default for sys.exit() in python 3.8+ if no arg).
        # If an error occurred in mcp.run(), it would also fall through to here.
        # We only want to force exit(1) if it's not already a clean exit.
        # This part of logic is a bit complex due to SystemExit.
        # The SystemExit catch block helps clarify this.
        # If we reached here due to an error (not SystemExit(0)), then an exit code != 0 is appropriate.
        # The sys.exit(1) was originally here which is fine for error cases.
        # If graceful_shutdown ran, sys.exit(0) already happened.
        logger.info("Main finally block finished.")
        # The original sys.exit(1) is removed; exit code is handled by SystemExit or natural end of program.


@mcp.tool()
def health_check() -> str:
    """Performs a health check on the database connection and a simple query."""
    logger.info("Performing health check...")
    conn = None
    try:
        # 1. Attempt to acquire a database connection
        conn = get_connection()
        logger.debug("Health check: Connection acquired.")

        # 2. Execute a simple query
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT 1 AS healthy")
            result = cur.fetchone()
            if result and result['healthy'] == 1:
                logger.info("Health check OK: Database connection and query successful.")
                # For non-SELECT queries, commit is needed. SELECT 1 is technically a SELECT.
                # No explicit commit needed for SELECT on most PostgreSQL setups by default.
                return "Health check OK: Database connection successful."
            else:
                logger.error("Health check FAILED: Query execution did not return expected result.")
                conn.rollback() # Rollback if query result is unexpected
                return "Health check FAILED: Query execution error (unexpected result)."

    except RetryError as e: # Catch tenacity's RetryError if get_connection fails after all retries
        logger.error(f"Health check FAILED: Could not connect to database after multiple retries - {str(e.last_attempt.exception())}")
        return f"Health check FAILED: Could not connect to database after retries ({str(e.last_attempt.exception())})."
    except psycopg2.Error as db_err: # Catch specific database errors (includes OperationalError, PoolError not caught by RetryError)
        logger.error(f"Health check FAILED: Database error - {str(db_err)}")
        if conn: conn.rollback() # Rollback on database error
        return f"Health check FAILED: Database error ({str(db_err)})."
    except Exception as e:
        logger.error(f"Health check FAILED: An unexpected error occurred - {str(e)}")
        if conn: conn.rollback() # Rollback on unexpected error
        return f"Health check FAILED: An unexpected error occurred ({str(e)})."
    finally:
        if conn:
            if pool:
                try:
                    pool.putconn(conn)
                    logger.debug("Health check: Connection returned to pool.")
                except Exception as e:
                    logger.error(f"Health check: Error returning connection to pool - {str(e)}")
                    try:
                        conn.close()
                        logger.warning("Health check: Connection closed directly after failing to return to pool.")
                    except Exception as close_err:
                        logger.error(f"Health check: Error closing connection directly - {str(close_err)}")
            else:
                try:
                    conn.close()
                    logger.debug("Health check: Non-pooled connection closed.")
                except Exception as e:
                    logger.error(f"Health check: Error closing non-pooled connection - {str(e)}")
