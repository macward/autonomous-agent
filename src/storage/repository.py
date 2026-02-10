"""Repository for audit record CRUD operations."""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from src.core.logging import get_logger
from src.storage.database import Database, get_database
from src.storage.models import (
    AgentDecision,
    AgentRequest,
    AuditLog,
    ExecutionStatus,
    RequestStatus,
    ToolExecution,
)

logger = get_logger("storage.repository")


def _generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


def _to_json(data: dict[str, Any] | None) -> str | None:
    """Convert dict to JSON string."""
    if data is None:
        return None
    return json.dumps(data)


def _from_json(data: str | None) -> dict[str, Any] | None:
    """Convert JSON string to dict."""
    if data is None:
        return None
    return json.loads(data)


class AuditRepository:
    """Repository for all audit-related operations."""

    def __init__(self, db: Database | None = None):
        """Initialize repository with database.

        Args:
            db: Database instance (uses global if not provided)
        """
        self._db = db

    @property
    def db(self) -> Database:
        """Get database instance."""
        if self._db is None:
            self._db = get_database()
        return self._db

    # Request operations

    def create_request(self, input_text: str) -> AgentRequest:
        """Create a new agent request record.

        Args:
            input_text: Original user input

        Returns:
            Created request record
        """
        request = AgentRequest(
            id=_generate_id(),
            input_text=input_text,
        )

        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO requests (id, created_at, input_text, status)
                VALUES (?, ?, ?, ?)
                """,
                (request.id, request.created_at.isoformat(), request.input_text, request.status),
            )

        logger.info(f"Created request {request.id}")
        return request

    def update_request_status(
        self,
        request_id: str,
        status: RequestStatus,
        final_output: str | None = None,
        error: str | None = None,
    ) -> None:
        """Update request status.

        Args:
            request_id: Request ID to update
            status: New status
            final_output: Final output if completed
            error: Error message if failed
        """
        completed_at = datetime.now(UTC).isoformat() if status in (
            RequestStatus.COMPLETED,
            RequestStatus.FAILED,
        ) else None

        with self.db.connection() as conn:
            conn.execute(
                """
                UPDATE requests
                SET status = ?, completed_at = ?, final_output = ?, error = ?
                WHERE id = ?
                """,
                (status, completed_at, final_output, error, request_id),
            )

        logger.info(f"Updated request {request_id} to {status}")

    def get_request(self, request_id: str) -> AgentRequest | None:
        """Get a request by ID.

        Args:
            request_id: Request ID

        Returns:
            Request record or None
        """
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM requests WHERE id = ?",
                (request_id,),
            ).fetchone()

        if row is None:
            return None

        return AgentRequest(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            input_text=row["input_text"],
            status=RequestStatus(row["status"]),
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            final_output=row["final_output"],
            error=row["error"],
        )

    # Decision operations

    def create_decision(
        self,
        request_id: str,
        reasoning: str,
        selected_tool: str | None = None,
        tool_input: dict[str, Any] | None = None,
    ) -> AgentDecision:
        """Create a new decision record.

        Args:
            request_id: Parent request ID
            reasoning: LLM reasoning
            selected_tool: Tool selected by LLM
            tool_input: Input for the tool

        Returns:
            Created decision record
        """
        decision = AgentDecision(
            id=_generate_id(),
            request_id=request_id,
            reasoning=reasoning,
            selected_tool=selected_tool,
            tool_input=tool_input,
        )

        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO decisions (id, request_id, created_at, reasoning, selected_tool, tool_input)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.id,
                    decision.request_id,
                    decision.created_at.isoformat(),
                    decision.reasoning,
                    decision.selected_tool,
                    _to_json(decision.tool_input),
                ),
            )

        logger.info(f"Created decision {decision.id} for request {request_id}")
        return decision

    def get_decisions_for_request(self, request_id: str) -> list[AgentDecision]:
        """Get all decisions for a request.

        Args:
            request_id: Request ID

        Returns:
            List of decision records
        """
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE request_id = ? ORDER BY created_at",
                (request_id,),
            ).fetchall()

        return [
            AgentDecision(
                id=row["id"],
                request_id=row["request_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                reasoning=row["reasoning"],
                selected_tool=row["selected_tool"],
                tool_input=_from_json(row["tool_input"]),
            )
            for row in rows
        ]

    # Execution operations

    def create_execution(
        self,
        decision_id: str,
        request_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
        status: ExecutionStatus,
        output: str | None = None,
        error: str | None = None,
        duration_ms: int | None = None,
    ) -> ToolExecution:
        """Create a new tool execution record.

        Args:
            decision_id: Parent decision ID
            request_id: Root request ID
            tool_name: Name of executed tool
            tool_input: Input provided to tool
            status: Execution result status
            output: Tool output
            error: Error message if failed
            duration_ms: Execution duration

        Returns:
            Created execution record
        """
        execution = ToolExecution(
            id=_generate_id(),
            decision_id=decision_id,
            request_id=request_id,
            tool_name=tool_name,
            tool_input=tool_input,
            status=status,
            output=output,
            error=error,
            duration_ms=duration_ms,
        )

        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO executions
                (id, decision_id, request_id, created_at, tool_name, tool_input, status, output, error, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution.id,
                    execution.decision_id,
                    execution.request_id,
                    execution.created_at.isoformat(),
                    execution.tool_name,
                    _to_json(execution.tool_input),
                    execution.status,
                    execution.output,
                    execution.error,
                    execution.duration_ms,
                ),
            )

        logger.info(f"Created execution {execution.id} for tool {tool_name}")
        return execution

    def get_executions_for_request(self, request_id: str) -> list[ToolExecution]:
        """Get all executions for a request.

        Args:
            request_id: Request ID

        Returns:
            List of execution records
        """
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM executions WHERE request_id = ? ORDER BY created_at",
                (request_id,),
            ).fetchall()

        return [
            ToolExecution(
                id=row["id"],
                decision_id=row["decision_id"],
                request_id=row["request_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                tool_name=row["tool_name"],
                tool_input=_from_json(row["tool_input"]),
                status=ExecutionStatus(row["status"]),
                output=row["output"],
                error=row["error"],
                duration_ms=row["duration_ms"],
            )
            for row in rows
        ]

    # Audit log operations

    def log(
        self,
        level: str,
        component: str,
        message: str,
        context: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> AuditLog:
        """Create an audit log entry.

        Args:
            level: Log level
            component: Component that generated the log
            message: Log message
            context: Additional context
            request_id: Associated request ID

        Returns:
            Created audit log record
        """
        audit = AuditLog(
            id=_generate_id(),
            level=level,
            component=component,
            message=message,
            context=context,
            request_id=request_id,
        )

        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs (id, created_at, level, component, message, context, request_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit.id,
                    audit.created_at.isoformat(),
                    audit.level,
                    audit.component,
                    audit.message,
                    _to_json(audit.context),
                    audit.request_id,
                ),
            )

        return audit

    def get_audit_logs(
        self,
        request_id: str | None = None,
        level: str | None = None,
        limit: int = 100,
    ) -> list[AuditLog]:
        """Query audit logs with optional filters.

        Args:
            request_id: Filter by request ID
            level: Filter by log level
            limit: Maximum records to return

        Returns:
            List of audit log records
        """
        query = "SELECT * FROM audit_logs WHERE 1=1"
        params: list[Any] = []

        if request_id is not None:
            query += " AND request_id = ?"
            params.append(request_id)

        if level is not None:
            query += " AND level = ?"
            params.append(level)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self.db.connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            AuditLog(
                id=row["id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                level=row["level"],
                component=row["component"],
                message=row["message"],
                context=_from_json(row["context"]),
                request_id=row["request_id"],
            )
            for row in rows
        ]

    # Query methods

    def get_recent_requests(self, limit: int = 10) -> list[AgentRequest]:
        """Get recent requests.

        Args:
            limit: Maximum records to return

        Returns:
            List of recent requests
        """
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM requests ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [
            AgentRequest(
                id=row["id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                input_text=row["input_text"],
                status=RequestStatus(row["status"]),
                completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                final_output=row["final_output"],
                error=row["error"],
            )
            for row in rows
        ]

    def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics.

        Returns:
            Dictionary with execution stats
        """
        with self.db.connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
            by_status = conn.execute(
                "SELECT status, COUNT(*) as count FROM executions GROUP BY status"
            ).fetchall()
            by_tool = conn.execute(
                "SELECT tool_name, COUNT(*) as count FROM executions GROUP BY tool_name"
            ).fetchall()
            avg_duration = conn.execute(
                "SELECT AVG(duration_ms) FROM executions WHERE duration_ms IS NOT NULL"
            ).fetchone()[0]

        return {
            "total_executions": total,
            "by_status": {row["status"]: row["count"] for row in by_status},
            "by_tool": {row["tool_name"]: row["count"] for row in by_tool},
            "avg_duration_ms": round(avg_duration) if avg_duration else None,
        }
