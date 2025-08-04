# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a local RAG (Retrieval-Augmented Generation) system with Google Drive integration built using modern AI frameworks. The system uses trunk-based development with a comprehensive test-driven development (TDD) approach.

## Development Workflow

### Trunk-Based Development
- All development happens on the main branch
- Keep changes small and merge frequently
- Use feature flags for incomplete features rather than long-lived branches
- Run `just check` before committing to ensure all tests and linting pass

### Common Development Commands

```bash
# Project management (using just)
just quickstart       # Complete setup wizard + run
just dev             # Install deps + setup services
just run             # Run the main application
just run-folder ID   # Run with specific Google Drive folder
just chat            # Chat only mode (skip indexing)
just index           # Index only mode (no chat)

# Quality checks (always run before committing)
just check           # Run all tests + linting (pytest + ruff + mypy)
just check-all       # Tests + linting + formatting checks
just test            # Run all tests
just test-file test_config.py  # Run specific test file
just test-coverage   # Run tests with coverage report
just format          # Auto-format code with black + isort

# Services management
just setup           # Start Qdrant + Ollama, pull models
just services-up     # Start Docker services
just services-status # Check if services are running
just logs           # View all logs
```

### Running Individual Tests
```bash
# Run a specific test
uv run pytest tests/test_config.py::TestConfig::test_singleton_pattern -v

# Run tests matching a pattern
uv run pytest -k "test_load_documents" -v

# Run with debugging output
uv run pytest -s tests/test_document_loader.py
```

## Architecture

### Core System Flow
```
Google Drive → Document Loader → Indexing Pipeline → Vector DB (Qdrant) → Query Pipeline → Ollama LLM → Chat Interface
```

### Key Architectural Decisions

1. **Singleton Configuration Pattern**: The `Config` class uses a singleton pattern to ensure consistent configuration across all components. All configuration is loaded from environment variables.

2. **Pipeline Architecture**: The system uses Haystack pipelines for both indexing and querying:
   - **IndexingPipeline**: Document splitting → Embedding generation → Vector storage
   - **QueryPipeline**: Query embedding → Vector retrieval → LLM generation with context

3. **Document Processing**: Documents are loaded from Google Drive using service account authentication, then chunked with configurable overlap for better context preservation.

4. **Local-First Design**: All AI processing happens locally using Ollama, ensuring data privacy. Qdrant runs in Docker for vector storage.

### Component Interactions

- **Config**: Singleton that validates and provides configuration to all components
- **GoogleDriveLoader**: Authenticates with Google API and downloads documents with metadata
- **IndexingPipeline**: Creates embeddings and stores in Qdrant with metadata preservation
- **QueryPipeline**: Retrieves relevant chunks and generates answers using local LLM
- **ChatInterface**: Rich terminal UI that manages conversation state and commands

### Environment Variables

The system relies on `.env` configuration (see `.env.example`):
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to Google service account JSON
- `QDRANT_URL`: Vector database URL (default: http://localhost:6333)
- `OLLAMA_BASE_URL`: Ollama API URL (default: http://localhost:11434)
- `OLLAMA_MODEL_NAME`: LLM model (default: llama3.2:latest)
- `OLLAMA_EMBEDDING_MODEL`: Embedding model (default: mxbai-embed-large)

### Test-Driven Development

All components have comprehensive test coverage (70+ tests). Tests use mocking for external dependencies:
- Google Drive API calls are mocked in `test_document_loader.py`
- Qdrant operations are mocked in `test_indexing_pipeline.py`
- Ollama calls are mocked in `test_query_pipeline.py`

When adding new features:
1. Write failing tests first
2. Implement minimal code to pass
3. Refactor for quality
4. Ensure `just check` passes

## Google Drive Integration

The system uses service account authentication:
1. Service account JSON must be at path specified in `GOOGLE_APPLICATION_CREDENTIALS`
2. The Google Drive folder must be shared with the service account email
3. Documents are loaded with full metadata (name, MIME type, modification time)
4. Supports filtering by MIME types and document count limits

## Key Implementation Details

- **Haystack Version**: Uses Haystack 2.x with its new pipeline API
- **Document Store**: Qdrant in local mode (Docker container)
- **Embedding Dimension**: 1024 (mxbai-embed-large)
- **Chunk Processing**: Configurable size/overlap via environment variables
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Logging**: Rich logging with configurable levels
- **Code Formatting**: Black (line-length: 100) + isort (black profile)
