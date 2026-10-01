# NOVA

Local-first, extensible AI computer assistant.

NOVA is designed as a modular AI assistant that can interact with a computer through tools and MCP servers. The architecture is designed to remain extensible so that future capabilities such as Gmail, GitHub, Calendar, Browser, and other integrations can be added without redesigning the core.

## Current Architecture

```text
User
  ↓
NOVA Core / Orchestrator
  ↓
Model Router
  ↓
Local LLM (Ollama)
  ↓
MCP Client
  ↓
MCP Servers
  ↓
Tools
````

## Project Structure

* `app/core/` — orchestration, routing, state, permissions, and shared application logic
* `app/llm/` — LLM integration and model communication
* `app/agents/` — agent/workflow logic
* `app/tools/` — tool abstractions and local tool helpers
* `app/mcp/` — MCP client, MCP servers, and tool integration
* `app/memory/` — future memory and retrieval layer
* `config/` — application configuration
* `tests/` — automated tests and model benchmarks
* `logs/` — runtime logs
* `data/knowledge/` — future RAG knowledge sources
* `data/workspace/` — NOVA-controlled working area

## Day 2 — Agent and MCP Foundation

Day 2 focused on building the core agent and tool-use architecture.

### Implemented

* Ollama LLM integration
* Qwen3 4B Instruct as the current local model
* Model Router
* NOVA Orchestrator
* MCP Client
* Computer MCP Server
* Dynamic MCP tool discovery
* Native LLM tool calling
* Multi-step tool execution
* Tool execution retry handling
* Maximum tool-step limit
* Basic sensitive-file protection
* Safe non-recursive file searching
* Text-file discovery

### Current Computer MCP Tools

NOVA currently has the following computer tools:

* `get_current_directory()` — returns the current working directory
* `list_files()` — lists files and folders in a directory
* `list_text_files()` — finds supported text files in a directory
* `read_file()` — reads text files
* `search_files()` — searches for files and folders by name

## Tool Execution Flow

```text
User Request
     ↓
NOVA Orchestrator
     ↓
Model Router
     ↓
Local LLM
     ↓
Select MCP Tool
     ↓
MCP Client
     ↓
Computer MCP Server
     ↓
Execute Tool
     ↓
Tool Result
     ↓
LLM
     ↓
Final Response
```

## Safety

NOVA currently includes basic protections for sensitive configuration files such as:

* `.env`
* `.env.local`
* `.env.production`

Unrestricted recursive file searching is also not currently exposed to the model.

A more advanced permission and security layer will be implemented in a later stage.

## Configuration

Model and execution settings are configured through environment variables.

Example:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b-instruct
OLLAMA_TEMPERATURE=0.2
OLLAMA_MAX_TOKENS=2048

MAX_TOOL_STEPS=5
MAX_TOOL_RETRIES=2
```

Sensitive values should be stored in `.env` and must not be committed to Git.

## Running NOVA

Start NOVA with:

```powershell
python -m app.main
```

NOVA accepts a natural-language request and can dynamically select available MCP tools.

## Testing

The project contains tests for:

* Ollama communication
* Model routing
* MCP tool discovery
* MCP tool calling
* Orchestrator behavior
* Multi-step tool execution
* Model benchmarking

## Future Architecture

NOVA is designed to remain model- and integration-independent.

Future model providers may include:

```text
Local
├── Ollama
└── Larger local models

Cloud
├── OpenAI
├── Anthropic
└── Other providers
```

Future MCP integrations may include:

```text
Computer MCP
Gmail MCP
GitHub MCP
Calendar MCP
Browser MCP
Documents MCP
```

The goal is to allow new capabilities to be added as independent integrations without redesigning NOVA Core.

## Roadmap

### Completed

* Foundation
* Local LLM integration
* Model routing
* Agent orchestration
* MCP client/server architecture
* Basic computer/file tools

### Next

* Permission and security layer
* Safe file creation and modification
* Multi-step computer workflows
* Persistent memory
* Model-provider abstraction
* Cloud LLM support
* Gmail integration
* GitHub integration
* Calendar integration
* Browser integration

### Long-Term Research

* Advanced agent planning
* Vision
* Voice
* Improved memory and RAG
* Agent evaluation
* Experience collection
* Reinforcement learning / self-improvement