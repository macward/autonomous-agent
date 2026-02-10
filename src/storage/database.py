"""SQLite database connection and schema management."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from src.core.logging import get_logger

logger = get_logger("storage.database")

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


class Database:
    """SQLite database manager with connection pooling."""

    def __init__(self, db_path: str | Path):
        """Initialize database with path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._ensure_directory()
        self._initialized = False

    def _ensure_directory(self) -> None:
        """Ensure database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        with self.connection() as conn:
            conn.executescript(SCHEMA)
            logger.info(f"Database initialized at {self.db_path}")

        self._initialized = True

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with automatic cleanup.

        Yields:
            SQLite connection configured for the agent
        """
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")

        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# Global database instance (initialized lazily)
_db: Database | None = None


def get_database() -> Database:
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

        _db = Database(db_path)
        _db.initialize()

    return _db


def reset_database() -> None:
    """Reset the global database instance (for testing)."""
    global _db
    _db = None
