"""Main orchestration module for the RAG system."""

import logging
import sys
from typing import Any, Dict, List, Optional

from googleapiclient.errors import HttpError  # type: ignore
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from .chat_interface import ChatInterface
from .config import Config
from .document_loader import GoogleDriveLoader
from .indexing_pipeline import IndexingPipeline
from .query_pipeline import QueryPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()


class RAGSystem:
    """Main orchestrator for the RAG system."""

    def __init__(self):
        self.config = Config()
        self.console = console
        self.indexing_pipeline = None
        self.query_pipeline = None

    def initialize_pipelines(self) -> bool:
        """Initialize the indexing and query pipelines."""
        try:
            self.console.print("[cyan]Initializing pipelines...[/cyan]")
            self.indexing_pipeline = IndexingPipeline(self.config)
            self.query_pipeline = QueryPipeline(self.config)
            # Initialize both pipeline components
            self.indexing_pipeline.initialize()
            self.query_pipeline.initialize()
            self.console.print("[green]✅ Pipelines initialized successfully[/green]")
            return True
        except Exception as e:
            self.console.print(f"[red]❌ Failed to initialize pipelines: {e}[/red]")
            logger.exception("Pipeline initialization failed")
            return False

    def load_and_index_documents(
        self, folder_id: Optional[str] = None, max_docs: Optional[int] = None
    ) -> bool:
        """Load documents from Google Drive and index them."""
        try:
            # Use provided folder_id or fall back to config
            actual_folder_id = folder_id or self.config.GOOGLE_DRIVE_FOLDER_ID
            if not actual_folder_id:
                self.console.print("[red]❌ No Google Drive folder ID provided![/red]")
                self.console.print(
                    "[yellow]Set GOOGLE_DRIVE_FOLDER_ID in .env or use --folder-id flag[/yellow]"
                )
                return False

            self.console.print(
                f"[cyan]Loading documents from Google Drive folder: {actual_folder_id}[/cyan]"
            )

            # Load documents
            loader = GoogleDriveLoader(self.config)
            loader.authenticate()  # Authenticate before loading documents
            documents = loader.load_documents(actual_folder_id, max_documents=max_docs)

            if not documents:
                self.console.print("[yellow]⚠️  No documents found in the specified folder[/yellow]")
                return False

            self.console.print(f"[green]✅ Loaded {len(documents)} documents[/green]")

            # Display loaded documents
            self._display_loaded_documents(documents)

            # Index documents
            self.console.print("\n[cyan]Indexing documents...[/cyan]")
            self.indexing_pipeline.process_documents(documents)
            self.console.print("[green]✅ Documents indexed successfully[/green]")

            return True

        except HttpError as e:
            if e.resp.status == 404:
                self.console.print(f"[red]❌ Folder not found: {actual_folder_id}[/red]")
                self.console.print(
                    "[yellow]Please check the folder ID and ensure it's shared with your service account[/yellow]"
                )
            else:
                self.console.print(f"[red]❌ Google Drive API error: {e}[/red]")
            return False
        except Exception as e:
            self.console.print(f"[red]❌ Failed to load/index documents: {e}[/red]")
            logger.exception("Document loading/indexing failed")
            return False

    def _display_loaded_documents(self, documents: List[Dict[str, Any]]):
        """Display a table of loaded documents."""
        table = Table(title="Loaded Documents", show_header=True, header_style="bold magenta")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Type", style="green")
        table.add_column("Size", style="yellow")

        for doc in documents[:10]:  # Show first 10
            # Check for name in metadata or root level
            name = doc.get("name") or doc.get("metadata", {}).get("name", "Unknown")
            mime_type = doc.get("mime_type") or doc.get("metadata", {}).get("mimeType", "Unknown")

            table.add_row(
                name[:50],
                mime_type,
                f"{len(doc.get('content', ''))} chars",
            )

        if len(documents) > 10:
            table.add_row("...", "...", f"({len(documents) - 10} more)")

        self.console.print(table)

    def start_chat(self):
        """Start the interactive chat interface."""
        if not self.query_pipeline:
            self.console.print("[red]❌ Query pipeline not initialized![/red]")
            return

        chat = ChatInterface(self.config, self.query_pipeline)
        chat.run()

    def show_info(self):
        """Display system configuration and status."""
        panel = Panel.fit(
            f"""[bold cyan]RAG Knowledge System Configuration[/bold cyan]

[yellow]Environment:[/yellow]
• Google Drive Folder: {self.config.GOOGLE_DRIVE_FOLDER_ID or '[red]Not Set[/red]'}
• Credentials Path: {self.config.CREDENTIALS_PATH}
• Qdrant URL: {self.config.QDRANT_URL}
• Ollama URL: {self.config.OLLAMA_URL}

[yellow]Models:[/yellow]
• Chat Model: {self.config.CHAT_MODEL}
• Embedding Model: {self.config.EMBEDDING_MODEL}

[yellow]Processing:[/yellow]
• Chunk Size: {self.config.CHUNK_SIZE}
• Chunk Overlap: {self.config.CHUNK_OVERLAP}
• Collection Name: {self.config.COLLECTION_NAME}
""",
            title="System Info",
            border_style="cyan",
        )
        self.console.print(panel)


def benchmark_system():
    """Run a simple benchmark of the system."""
    console = Console()
    console.print("[cyan]Running system benchmark...[/cyan]")

    try:
        # Test imports
        console.print("✅ All imports successful")

        # Test configuration
        Config()
        console.print("✅ Configuration loaded")

        # Test pipeline initialization
        config = Config()
        IndexingPipeline(config)
        console.print("✅ Indexing pipeline created")

        QueryPipeline(config)
        console.print("✅ Query pipeline created")

        console.print("\n[green]All systems operational![/green]")

    except Exception as e:
        console.print(f"\n[red]❌ Benchmark failed: {e}[/red]")
        raise


def main():
    """Main entry point for the application."""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Knowledge System")
    parser.add_argument("--folder-id", help="Google Drive folder ID to load documents from")
    parser.add_argument("--max-docs", type=int, help="Maximum number of documents to load")
    parser.add_argument(
        "--chat-only", action="store_true", help="Skip indexing and go straight to chat"
    )
    parser.add_argument(
        "--setup-only", action="store_true", help="Only run setup and indexing, no chat"
    )
    parser.add_argument("--info", action="store_true", help="Show system configuration")

    args = parser.parse_args()

    # Create system instance
    system = RAGSystem()

    # Show info and exit if requested
    if args.info:
        system.show_info()
        return

    # Show banner
    console.print(
        Panel.fit(
            "[bold cyan]🚀 RAG Knowledge System[/bold cyan]\n"
            "Chat with your Google Drive documents using local AI",
            border_style="cyan",
        )
    )

    # Initialize pipelines
    if not system.initialize_pipelines():
        console.print("[red]Failed to initialize system. Please check your configuration.[/red]")
        sys.exit(1)

    # Load and index documents unless chat-only mode
    if not args.chat_only:
        if not system.load_and_index_documents(args.folder_id, args.max_docs):
            console.print("[red]Failed to load documents. Please check your configuration.[/red]")
            sys.exit(1)
    else:
        console.print("[cyan]Skipping document indexing (--chat-only mode)[/cyan]")

    # Start chat unless setup-only mode
    if not args.setup_only:
        console.print("\n[cyan]Starting chat interface...[/cyan]\n")
        system.start_chat()
    else:
        console.print(
            "\n[green]✅ Setup complete! Run without --setup-only to start chatting.[/green]"
        )


if __name__ == "__main__":
    main()
