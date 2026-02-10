# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Configuration

- vibe: autonomous-agent
- branch: main

## MCP Usage

**Always use vibeMCP tools** for task management, plans, and session logs. Never modify task files directly in `~/.vibe/autonomous-agent/` - use the MCP tools instead:
- `mcp__vibeMCP__list_tasks` - list tasks
- `mcp__vibeMCP__read_doc` - read documents
- `mcp__vibeMCP__get_plan` - get execution plan
- `mcp__vibeMCP__tool_update_task_status` - update task status
- `mcp__vibeMCP__tool_log_session` - log session progress
- `mcp__vibeMCP__tool_create_task` - create new tasks

## Project Overview

Autonomous agent that runs persistently on a local server, capable of interpreting instructions, making decisions, and executing predefined tools safely without direct human intervention at each step.

**Key principle**: Security by design. The agent is potentially dangerous if not properly constrained.

## Architecture

Five core components:

1. **Agent API** - FastAPI HTTP service exposing `POST /agent/run`
2. **LLM Connector** - Adapter for reasoning model (remote or local)
3. **Tool Registry** - Catalog of available tools with name, description, JSON Schema, permissions
4. **Tool Executor** - Executes tools with input validation, timeouts, resource limits, logging
5. **Storage/Audit** - SQLite persistence for requests, decisions, tool executions, outputs, errors

## Security Constraints (Non-negotiable)

- Allowlist-only tool execution (no arbitrary commands)
- Unprivileged system user
- Isolated workspace directories
- Strong JSON Schema validation on all inputs
- Complete auditable logs
- Timeouts and output limits on all operations

## MVP Scope

- Single endpoint: `POST /agent/run` with text input
- API Key authentication
- Initial tools: `health_check`, `list_dir`, `read_file`
- SQLite storage

## Tool Definition

A tool is a controlled, explicit, validated action:
- Clear purpose
- Typed inputs only (JSON Schema)
- Restricted environment operation
- Structured JSON output
- Safe failure modes

The LLM chooses which tool to use; it does not execute code directly.

## Reference Documents

- `references/documento_agente_autonomo_local.md` - Full project objectives and scope document
