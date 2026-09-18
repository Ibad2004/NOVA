# NOVA

Local-first, extensible AI computer assistant.

## Initial architecture

- `app/core/` — orchestration, state, permissions, and shared application logic
- `app/llm/` — Ollama integration and future model routing
- `app/agents/` — agent/workflow logic
- `app/tools/` — tool abstractions and local tool helpers
- `app/mcp/` — MCP client and tool discovery/integration
- `app/memory/` — memory and retrieval layer
- `config/` — application configuration
- `tests/` — automated tests
- `logs/` — runtime logs
- `data/knowledge/` — future RAG knowledge sources
- `data/workspace/` — NOVA-controlled working area
