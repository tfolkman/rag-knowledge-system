#!/usr/bin/env just --justfile

# Default recipe - show available commands
default:
    @just --list

# Install project dependencies
install:
    uv venv
    uv pip install -e .

# Start required services (Qdrant in Docker, Ollama runs system-wide)
services-up:
    docker compose up -d
    @echo "Waiting for Qdrant to start..."
    @sleep 5
    @echo "✅ Qdrant started"
    @echo "ℹ️  Using system-wide Ollama at localhost:11434"

# Stop services
services-down:
    docker compose down

# Pull required Ollama models
pull-models:
    @echo "Pulling Ollama models..."
    ollama pull llama3.2:latest
    ollama pull mxbai-embed-large
    @echo "✅ Models pulled"

# Complete setup (services + models)
setup: services-up pull-models
    @echo "✅ Setup complete!"

# Run the main application with hierarchical processing
run:
    uv run python run_hierarchical.py

# Run with specific Google Drive folder (old non-hierarchical)
run-folder-old folder_id:
    uv run python run.py --folder-id {{folder_id}}

# Run with hierarchical Google Drive folder traversal
run-folder folder_id:
    uv run python run_hierarchical.py --folder-id {{folder_id}}

# Run chat only (skip indexing)
chat:
    uv run python run_hierarchical.py --chat-only

# Run setup only (no chat) - hierarchical indexing
index:
    uv run python run_hierarchical.py

# Clear store and re-index with hierarchical processing
reindex:
    uv run python run_hierarchical.py --clear-store

# Show system configuration
info:
    uv run python run.py --info

# Run all tests
test:
    uv run pytest -v

# Run tests with coverage
test-coverage:
    uv run pytest --cov=src --cov-report=term-missing

# Run specific test file
test-file file:
    uv run pytest tests/{{file}} -v

# Run tests in watch mode (requires pytest-watch)
test-watch:
    uv run pytest-watch

# Format code with black and isort
format:
    uv run black src tests
    uv run isort src tests

# Lint code
lint:
    uv run ruff check src tests
    uv run mypy src

# Run all tests and linting checks
check: test lint
    @echo "✅ All tests and linting checks passed!"

# Run tests, linting, and formatting check
check-all: test lint
    @echo "Checking code formatting..."
    @uv run black --check src tests || (echo "❌ Code needs formatting. Run 'just format'" && exit 1)
    @uv run isort --check-only src tests || (echo "❌ Imports need sorting. Run 'just format'" && exit 1)
    @echo "✅ All checks passed!"

# Clean up generated files and caches
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name ".coverage" -delete
    rm -rf .pytest_cache
    rm -rf .mypy_cache
    rm -rf .ruff_cache
    rm -rf .pre-commit-trivy-cache
    rm -rf .trivy-cache

# Install pre-commit hooks
pre-commit-install:
    pre-commit install
    @echo "✅ Pre-commit hooks installed!"

# Run all pre-commit hooks on all files
pre-commit-run:
    pre-commit run --all-files

# Update pre-commit hooks to latest versions
pre-commit-update:
    pre-commit autoupdate

# Run security scan with Trivy
security-scan:
    @echo "🔍 Running Trivy security scan..."
    docker run --rm -v "$(pwd)":/src aquasec/trivy:latest fs /src \
        --skip-dirs .venv \
        --skip-dirs __pycache__ \
        --skip-dirs .git \
        --skip-dirs data \
        --severity CRITICAL,HIGH,MEDIUM

# Update secrets baseline
update-secrets-baseline:
    detect-secrets scan --update .secrets.baseline
    @echo "✅ Secrets baseline updated!"

# Audit secrets in baseline
audit-secrets:
    detect-secrets audit .secrets.baseline

# Full security check (tests + linting + security)
security-check: check security-scan
    @echo "🔒 All security checks passed!"

# Full clean (including data and services)
clean-all: clean services-down
    rm -rf data/qdrant_storage/*
    @echo "✅ Full cleanup complete"

# Check if services are running
services-status:
    @echo "Checking services..."
    @docker ps | grep qdrant > /dev/null && echo "✅ Qdrant: Running (Docker)" || echo "❌ Qdrant: Not running"
    @curl -s http://localhost:11434/api/version > /dev/null && echo "✅ Ollama: Running (System)" || echo "❌ Ollama: Not running"

# View logs
logs:
    docker compose logs -f

# View Qdrant logs
logs-qdrant:
    docker logs -f qdrant

# View Ollama logs
logs-ollama:
    docker logs -f ollama

# Quick development setup
dev: install setup
    @echo "✅ Development environment ready!"

# Run with example data (for testing)
demo:
    @echo "Setting up demo environment..."
    @cp .env.example .env 2>/dev/null || true
    @echo "GOOGLE_DRIVE_FOLDER_ID=demo_folder" >> .env
    uv run python run.py --max-docs 5


# Interactive setup wizard
wizard:
    @echo "🧙 RAG Knowledge System Setup Wizard"
    @echo "===================================="
    @echo ""
    @echo "Step 1: Checking Docker..."
    @command -v docker >/dev/null 2>&1 && echo "✅ Docker is installed" || echo "❌ Docker is not installed - please install Docker first"
    @echo ""
    @echo "Step 2: Checking Ollama..."
    @command -v ollama >/dev/null 2>&1 && echo "✅ Ollama is installed" || echo "❌ Ollama is not installed - please install Ollama first"
    @echo ""
    @echo "Step 3: Checking credentials.json..."
    @test -f credentials.json && echo "✅ credentials.json exists" || echo "❌ credentials.json not found - run 'just create-credentials' to create template"
    @echo ""
    @echo "Step 4: Checking .env file..."
    @test -f .env && echo "✅ .env exists" || (cp .env.example .env && echo "✅ Created .env from template")
    @echo ""
    @echo "Ready to run 'just dev' to complete setup!"

# Benchmark the system
benchmark:
    @echo "Running system benchmark..."
    uv run python -c "from src.main import benchmark_system; benchmark_system()"

# Update all dependencies
update-deps:
    uv pip install --upgrade -r requirements.txt

# Shell into Qdrant container
shell-qdrant:
    docker exec -it qdrant sh

# Shell into Ollama container
shell-ollama:
    docker exec -it ollama sh

# Run a quick health check
health:
    @echo "🏥 System Health Check"
    @echo "====================="
    @just services-status
    @echo ""
    @echo "Python Environment:"
    @uv run python --version
    @echo ""
    @echo "Dependencies:"
    @uv pip list | grep -E "(haystack|qdrant|ollama|rich)" || echo "❌ Dependencies not installed"
    @echo ""
    @echo "Credentials:"
    @test -f credentials.json && echo "✅ credentials.json exists" || echo "❌ credentials.json missing"
    @test -f .env && echo "✅ .env exists" || echo "❌ .env missing"

# Quick start (all-in-one command)
quickstart: wizard dev run
    @echo "🚀 System is running!"
