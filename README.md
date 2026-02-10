# Autonomous Agent

Autonomous agent that runs persistently on a local server, capable of interpreting instructions, making decisions, and executing predefined tools safely.

## Security by Design

This agent is potentially dangerous if not properly constrained. Security measures include:

- Allowlist-only tool execution (no arbitrary commands)
- Unprivileged system user
- Isolated workspace directories
- Strong JSON Schema validation on all inputs
- Complete auditable logs
- Timeouts and output limits on all operations

## Setup

```bash
# Install dependencies
uv sync

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the agent
uv run python -m src.main
```

## API

- `POST /agent/run` - Execute agent with text input (requires API key)
- `GET /health` - Health check endpoint

## Architecture

1. **Agent API** - FastAPI HTTP service
2. **LLM Connector** - Adapter for reasoning model
3. **Tool Registry** - Catalog of available tools
4. **Tool Executor** - Secure tool execution with validation
5. **Storage/Audit** - SQLite persistence for complete audit trail
