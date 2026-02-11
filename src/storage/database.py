"""SQLite database connection and schema management (async version)."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from src.core.logging import get_logger

logger = get_logger("storage.database")

# Connection pool settings
DEFAULT_POOL_SIZE = 5
CONNECTION_TIMEOUT = 30.0

SCHEMA = """
-- Agent requests table
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    input_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    completed_at TEXT,
    final_output TEXT,
    error TEXT
);

-- Agent decisions table
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    selected_tool TEXT,
    tool_input TEXT,
    FOREIGN KEY (request_id) REFERENCES requests(id)
);

-- Tool executions table
CREATE TABLE IF NOT EXISTS executions (
    id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    tool_input TEXT NOT NULL,
    status TEXT NOT NULL,
    output TEXT,
    error TEXT,
    duration_ms INTEGER,
    FOREIGN KEY (decision_id) REFERENCES decisions(id),
    FOREIGN KEY (request_id) REFERENCES requests(id)
);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    level TEXT NOT NULL,
    component TEXT NOT NULL,
    message TEXT NOT NULL,
    context TEXT,
    request_id TEXT,
    FOREIGN KEY (request_id) REFERENCES requests(id)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_requests_status ON requests(status);
CREATE INDEX IF NOT EXISTS idx_requests_created ON requests(created_at);
CREATE INDEX IF NOT EXISTS idx_decisions_request ON decisions(request_id);
CREATE INDEX IF NOT EXISTS idx_executions_request ON executions(request_id);
CREATE INDEX IF NOT EXISTS idx_executions_tool ON executions(tool_name);
CREATE INDEX IF NOT EXISTS idx_audit_level ON audit_logs(level);
CREATE INDEX IF NOT EXISTS idx_audit_request ON audit_logs(request_id);
"""


class AsyncDatabase:
    """Async SQLite database manager with connection pooling."""

    def __init__(self, db_path: str | Path, pool_size: int = DEFAULT_POOL_SIZE):
        """Initialize database with path and connection pool.

        Args:
            db_path: Path to SQLite database file
            pool_size: Maximum number of connections in the pool
        """
        self.db_path = Path(db_path)
        self._pool_size = pool_size
        self._pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(maxsize=pool_size)
        self._pool_initialized = False
        self._initialized = False
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new configured database connection."""
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.execute("PRAGMA journal_mode = WAL")
        return conn

    async def _init_pool(self) -> None:
        """Initialize the connection pool."""
        if self._pool_initialized:
            return

        for _ in range(self._pool_size):
            conn = await self._create_connection()
            await self._pool.put(conn)

        self._pool_initialized = True
        logger.debug(f"Connection pool initialized with {self._pool_size} connections")

    async def initialize(self) -> None:
        """Initialize database schema and connection pool."""
        if self._initialized:
            return

        # Create schema with a temporary connection
        conn = await self._create_connection()
        try:
            await conn.executescript(SCHEMA)
            await conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
        finally:
            await conn.close()

        # Initialize connection pool
        await self._init_pool()
        self._initialized = True

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Get a database connection from the pool.

        Yields:
            Async SQLite connection configured for the agent
        """
        # Ensure pool is initialized
        if not self._pool_initialized:
            await self._init_pool()

        # Get connection from pool with timeout
        try:
            conn = await asyncio.wait_for(
                self._pool.get(),
                timeout=CONNECTION_TIMEOUT,
            )
        except TimeoutError:
            # Pool exhausted, create a new connection
            logger.warning("Connection pool exhausted, creating new connection")
            conn = await self._create_connection()

        try:
            yield conn
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
        finally:
            # Return connection to pool if pool has room
            try:
                self._pool.put_nowait(conn)
            except asyncio.QueueFull:
                # Pool is full, close this connection
                await conn.close()

    async def close(self) -> None:
        """Close all connections in the pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                await conn.close()
            except asyncio.QueueEmpty:
                break

        self._pool_initialized = False
        self._initialized = False
        logger.info("Database connections closed")


# Global database instance (initialized lazily)
_db: AsyncDatabase | None = None


async def get_database() -> AsyncDatabase:
    """Get the global database instance."""
    global _db
    if _db is None:
        from src.core.config import get_settings

        settings = get_settings()
        # Extract path from sqlite:/// URL
        db_url = settings.database_url
        if db_url.startswith("sqlite:///"):
            db_path = db_url[10:]  # Remove "sqlite:///"
        else:
            db_path = "./data/agent.db"

        _db = AsyncDatabase(db_path)
        await _db.initialize()

    return _db


def reset_database() -> None:
    """Reset the global database instance (for testing).

    Note: For async cleanup, use close_database() instead.
    This function resets the global reference without waiting for connections to close.
    """
    global _db
    if _db is not None:
        # Schedule connection cleanup in background if there's an event loop
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_db.close())
        except RuntimeError:
            # No running loop, just reset the reference
            pass
    _db = None


async def close_database() -> None:
    """Close the global database instance and all connections."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None
        logger.info("Global database instance closed")
