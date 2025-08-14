# RAG Knowledge System

A local RAG (Retrieval-Augmented Generation) system with Google Drive integration, built using modern AI frameworks.

## Features

- 📁 **Google Drive Integration**: Load documents directly from your Google Drive
- 🐙 **GitHub Repository Ingestion**: Batch ingest markdown documentation from multiple GitHub repos
- 🔒 **Local LLM**: Uses Ollama for private, local AI processing
- 🔍 **Vector Search**: Powered by Qdrant for fast semantic search
- 🚀 **Modern Framework**: Built with Haystack for flexible AI pipelines
- 💻 **Rich Terminal UI**: Beautiful command-line interface using Rich
- ✅ **Test-Driven**: Comprehensive test suite (110+ tests) with TDD methodology

## Tech Stack

- **Google Drive API**: Document loading with `llama-index-readers-google`
- **GitHub CLI**: Repository cloning and authentication for private repos
- **Haystack**: AI pipeline orchestration
- **Qdrant**: Vector database for document storage and retrieval
- **Ollama**: Local LLM hosting (Llama 3.2, Mistral, etc.)
- **Rich**: Terminal UI and formatting
- **UV**: Fast Python package management

## Quick Start

### 1. Prerequisites

Install required services:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull llama3.2:latest
ollama pull mxbai-embed-large

# Install GitHub CLI (for repository ingestion)
# macOS
brew install gh

# Linux
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh

# Authenticate with GitHub (for private repos)
gh auth login
```

### 2. Setup Project

```bash
# Clone and setup
git clone https://github.com/yourusername/rag-knowledge-system.git
cd rag-knowledge-system

# Install dependencies with UV
uv sync

# Start services
docker-compose up -d
```

### 3. Configure Google Drive API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google Drive API
4. Create service account credentials
5. Download JSON key file as `credentials.json`
6. Share your Google Drive folder with the service account email

### 4. Run the System

#### Google Drive Ingestion

```bash
# Full setup and chat
uv run python run.py

# Or with specific options
uv run python run.py --folder-id YOUR_FOLDER_ID --max-docs 10

# Chat only (if data already indexed)
uv run python run.py --chat-only

# Setup only (no chat)
uv run python run.py --setup-only
```

#### GitHub Repository Ingestion

The system can batch ingest markdown documentation from multiple GitHub repositories:

```bash
# Create a list of repositories to ingest (one per line)
cat > repos.txt << EOF
owner/repo1
owner/repo2
myorg/private-repo
EOF

# Ingest repositories (checks ~/Coding first, clones if needed)
just ingest-github repos.txt

# Force fresh clone (ignore local copies)
just ingest-github-fresh repos.txt

# Clear store and reingest
just reingest-github repos.txt

# Use custom local directory
just ingest-github repos.txt ~/Projects
```

**Note**: Only `.md` files are processed to focus on documentation and README files.

## Configuration

The system uses environment variables (`.env` file):

```bash
# Google Drive API Configuration
GOOGLE_APPLICATION_CREDENTIALS=credentials.json

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=documents

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=llama3.2:latest
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large

# Application Configuration
LOG_LEVEL=INFO
MAX_DOCUMENTS_PER_BATCH=10
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# GitHub Repository Ingestion Configuration
GITHUB_LOCAL_REPOS_DIR=~/Coding
GITHUB_MAX_FILE_SIZE_MB=10
GITHUB_FILE_EXTENSIONS=.md
```

## Usage Examples

### Command Line Options

```bash
# Basic usage
uv run python run.py

# Load from specific folder
uv run python run.py --folder-id 1BxYz_ABC123...

# Limit document count
uv run python run.py --max-docs 5

# Show configuration
uv run python run.py --info

# Setup without chat
uv run python run.py --setup-only

# Chat with existing data
uv run python run.py --chat-only
```

### Chat Commands

Once in chat mode:

```
You: What is Python?
Assistant: [AI-generated answer based on your documents]

Available commands:
/help     - Show help message
/history  - Show conversation history
/clear    - Clear screen and history
/quit     - Exit application
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test file
uv run pytest tests/test_config.py -v
```

### Project Structure

```
rag-knowledge-system/
├── .env                    # Environment configuration
├── credentials.json        # Google Drive API credentials
├── docker-compose.yml      # Qdrant service
├── pyproject.toml         # UV project configuration
├── run.py                 # Main entry point
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── document_loader.py # Google Drive integration
│   ├── indexing_pipeline.py # Document indexing
│   ├── query_pipeline.py  # RAG query processing
│   ├── chat_interface.py  # Terminal UI
│   └── main.py           # System orchestration
├── tests/                 # Comprehensive test suite
│   ├── test_config.py
│   ├── test_document_loader.py
│   ├── test_indexing_pipeline.py
│   ├── test_query_pipeline.py
│   ├── test_chat_interface.py
│   └── test_integration.py
└── data/
    └── qdrant_storage/    # Vector database storage
```

### Architecture

The system follows a modular architecture with clear separation of concerns:

1. **Configuration Layer**: Centralized config management with validation
2. **Document Loading**: Google Drive API integration for document retrieval
3. **Indexing Pipeline**: Haystack-based pipeline for document processing and embedding
4. **Query Pipeline**: RAG pipeline combining retrieval and generation
5. **User Interface**: Rich terminal interface for interaction

## Troubleshooting

### Common Issues

1. **Google Drive Authentication**
   ```
   Error: Google credentials file not found
   ```
   - Ensure `credentials.json` exists and is valid
   - Check service account has access to Drive folders

2. **Ollama Connection**
   ```
   Error: Failed to connect to Ollama
   ```
   - Start Ollama: `ollama serve`
   - Pull required models: `ollama pull llama3.2:latest`

3. **Qdrant Connection**
   ```
   Error: Failed to connect to Qdrant
   ```
   - Start Qdrant: `docker-compose up -d`
   - Check service status: `docker-compose ps`

### Logs

Check logs for detailed error information:
```bash
tail -f rag_system.log
```

## Contributing

This project was built using Test-Driven Development (TDD). When contributing:

1. Write tests first (Red phase)
2. Implement minimal code to pass tests (Green phase)
3. Refactor for quality (Refactor phase)
4. Ensure all tests pass: `uv run pytest`

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built with [Haystack](https://github.com/deepset-ai/haystack) for AI pipelines
- [Qdrant](https://github.com/qdrant/qdrant) for vector storage
- [Ollama](https://github.com/ollama/ollama) for local LLM hosting
- [Rich](https://github.com/Textualize/rich) for terminal UI
- [UV](https://github.com/astral-sh/uv) for Python package management
