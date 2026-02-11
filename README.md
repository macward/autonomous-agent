# Autonomous Agent

Autonomous agent that runs persistently on a local server, capable of interpreting instructions, making decisions, and executing predefined tools safely without direct human intervention at each step.

## Security by Design

This agent is potentially dangerous if not properly constrained. Security measures include:

- **Allowlist-only tool execution** - Only registered tools can be executed
- **Permission enforcement** - Block dangerous permissions (network, execute) by default
- **Isolated workspace** - All file operations restricted to workspace directory
- **Path traversal protection** - Attempts to escape workspace are blocked
- **JSON Schema validation** - All tool inputs validated against schemas
- **Complete audit trail** - Every request, decision, and execution logged
- **Rate limiting** - Configurable request limits per minute
- **CORS protection** - Restrict allowed origins
- **Request size limits** - Prevent oversized payloads
- **Timeouts and limits** - All operations have configurable timeouts and output limits

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Anthropic API key

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd autonomous-agent

# Install dependencies
uv sync

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Required: AGENT_API_KEY and LLM_API_KEY
```

### Configuration

Edit `.env` with your settings:

```env
# API Security (required)
AGENT_API_KEY=your-secret-api-key-here

# LLM Configuration (required)
LLM_PROVIDER=anthropic  # or 'groq'
LLM_API_KEY=your-api-key
LLM_MODEL=claude-sonnet-4-20250514  # optional, defaults based on provider

# Server (optional)
HOST=127.0.0.1
PORT=8000
DEBUG=false

# Security limits (optional)
WORKSPACE_ROOT=./workspace
MAX_TOOL_TIMEOUT=30
MAX_OUTPUT_SIZE=10000

# Permission enforcement (optional)
# Comma-separated list: read, write, execute, network
# Default blocks network and execute for security
BLOCKED_PERMISSIONS=network,execute

# Rate limiting (optional)
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=10  # requests per minute

# CORS (optional)
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Request limits (optional)
MAX_REQUEST_SIZE=1048576  # 1MB default
```

### Running the Agent

```bash
# Start the server
uv run python -m src.main

# Or with uvicorn directly
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000
```

The agent will be available at `http://127.0.0.1:8000`.

## API Reference

### Health Check

```http
GET /health
```

Returns agent health status. Does not require authentication.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Run Agent

```http
POST /agent/run
Content-Type: application/json
X-API-Key: your-api-key

{
  "input": "List the files in the workspace"
}
```

Executes the agent with the given input. The agent will:
1. Analyze the input using the LLM
2. Select and execute appropriate tools
3. Return a structured response

**Request Headers:**
- `X-API-Key` (required): Your API key

**Request Body:**
- `input` (string, required): User instruction (1-10000 characters)

**Response:**
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "success",
  "output": "The workspace contains 3 files...",
  "error": null,
  "tool_executions": [
    {
      "tool_name": "list_dir",
      "status": "success",
      "duration_ms": 5,
      "output": {"entries": [...], "count": 3}
    }
  ],
  "created_at": "2024-01-15T10:30:00Z",
  "duration_ms": 1234
}
```

**Status Values:**
- `success`: Agent completed successfully
- `error`: Agent encountered an error
- `tool_error`: A tool execution failed

**Rate Limiting:**

The `/agent/run` endpoint is rate limited to 10 requests per minute by default. When exceeded, the API returns:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 60

{
  "error": "Rate limit exceeded: 10 per 1 minute"
}
```

### Interactive Documentation

When the server is running:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## LLM Providers

The agent supports multiple LLM providers:

| Provider | Default Model | Environment Variable |
|----------|---------------|---------------------|
| `anthropic` | `claude-sonnet-4-20250514` | `LLM_API_KEY` (Anthropic API key) |
| `groq` | `llama-3.3-70b-versatile` | `LLM_API_KEY` (Groq API key) |

### Using Groq

Groq provides fast inference for open-source models:

```env
LLM_PROVIDER=groq
LLM_API_KEY=your-groq-api-key
LLM_MODEL=llama-3.3-70b-versatile  # optional
```

Available Groq models with tool use support:
- `llama-3.3-70b-versatile` (default)
- `llama-3.1-70b-versatile`
- `llama-3.1-8b-instant`
- `mixtral-8x7b-32768`

## Available Tools

### health_check

Check the agent's health status and system information.

**Input:** None required
**Output:** System status, platform info, Python version

### list_dir

List files and directories in a path within the workspace.

**Input:**
- `path` (string, optional): Relative path within workspace (default: ".")

**Output:** List of entries with name, type, and size

### read_file

Read the contents of a text file within the workspace.

**Input:**
- `path` (string, required): Relative path to file
- `max_lines` (integer, optional): Maximum lines to read (1-1000)

**Output:** File content, size, line count

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent API                             │
│           (FastAPI + Auth + Rate Limiting + CORS)            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   Agent Orchestrator                         │
│              (LLM reasoning loop)                            │
└──────┬──────────────┼───────────────────────┬───────────────┘
       │              │                       │
┌──────▼──────┐ ┌─────▼─────┐ ┌───────────────▼───────────────┐
│    LLM      │ │   Tool    │ │         Storage               │
│  Connector  │ │  Executor │ │  (Async SQLite + Pool)        │
│ (Anthropic) │ │ +Perms    │ │                               │
└─────────────┘ └─────┬─────┘ └───────────────────────────────┘
                      │
              ┌───────▼───────┐
              │ Tool Registry │
              │ (Allowlist)   │
              └───────────────┘
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test file
uv run pytest tests/test_integration.py -v
```

### Code Quality

```bash
# Run linter
uv run ruff check src tests

# Run formatter
uv run ruff format src tests
```

## Production Deployment

### Security Checklist

1. **API Key**: Generate a strong, random API key
2. **Network**: Bind to localhost or use reverse proxy with TLS
3. **User**: Run as unprivileged system user
4. **Workspace**: Create isolated directory with minimal permissions
5. **Permissions**: Keep `BLOCKED_PERMISSIONS=network,execute` in production
6. **Rate Limiting**: Enable rate limiting (`RATE_LIMIT_ENABLED=true`)
7. **CORS**: Restrict `CORS_ORIGINS` to trusted domains only
8. **Logs**: Configure log rotation and monitoring
9. **Limits**: Adjust timeouts, output limits, and request size for your use case

### Docker (Example)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install uv && uv sync --frozen

# Create unprivileged user
RUN useradd -m agent && chown -R agent:agent /app
USER agent

# Create workspace
RUN mkdir -p /app/workspace /app/data

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Systemd Service (Example)

```ini
[Unit]
Description=Autonomous Agent
After=network.target

[Service]
Type=simple
User=agent
WorkingDirectory=/opt/autonomous-agent
ExecStart=/opt/autonomous-agent/.venv/bin/uvicorn src.main:app --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Audit Trail

All operations are logged to SQLite for complete audit trail:

- **Requests**: Input text, status, timestamps
- **Decisions**: LLM reasoning, selected tools
- **Executions**: Tool name, input, output, duration, errors

Query the database:

```bash
sqlite3 data/agent.db "SELECT * FROM requests ORDER BY created_at DESC LIMIT 10"
```

## License

MIT
