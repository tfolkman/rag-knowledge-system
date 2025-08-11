#!/usr/bin/env python
"""Run script for hierarchical RAG system with Google Drive folder traversal."""

import argparse
import logging
import sys

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from src.chat_interface import ChatInterface
from src.config import Config
from src.hierarchical_loader import HierarchicalDocumentLoader
from src.indexing_pipeline import IndexingPipeline
from src.query_pipeline import QueryPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()


def load_google_drive_folder(folder_id: str, config: Config, max_docs: int = None):
    """Load documents from Google Drive with hierarchical categorization."""
    console.print(f"[cyan]Loading documents from Google Drive folder: {folder_id}[/cyan]")
    console.print("[cyan]Traversing folder hierarchy...[/cyan]")

    # Initialize hierarchical loader
    loader = HierarchicalDocumentLoader()

    try:
        # Load documents with hierarchy from Google Drive
        documents = loader.load_from_google_drive(config, folder_id, max_documents=max_docs)

        if not documents:
            console.print("[yellow]⚠️  No documents found in the specified folder[/yellow]")
            return None

        console.print(f"[green]✅ Loaded {len(documents)} documents[/green]")

        # Display loaded documents with categories
        table = Table(title="Loaded Documents", show_header=True, header_style="bold magenta")
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Subcategory", style="blue")
        table.add_column("File Name", style="green")
        table.add_column("Path", style="yellow")
        table.add_column("Size", style="magenta")

        for doc in documents[:15]:  # Show first 15
            table.add_row(
                doc.meta.get("category", "root"),
                doc.meta.get("subcategory", "-"),
                doc.meta.get("file_name", "unknown")[:30],
                doc.meta.get("hierarchy_path", "")[:40],
                f"{len(doc.content)} chars",
            )

        if len(documents) > 15:
            table.add_row("...", "...", f"({len(documents) - 15} more)", "...", "...")

        console.print(table)

        # Show category distribution
        categories = {}
        for doc in documents:
            cat = doc.meta.get("category", "root")
            categories[cat] = categories.get(cat, 0) + 1

        console.print("\n[bold]Category Distribution:[/bold]")
        for cat, count in sorted(categories.items()):
            console.print(f"  • {cat}: {count} documents")

        return documents

    except Exception as e:
        console.print(f"[red]❌ Error loading from Google Drive: {e}[/red]")
        logger.exception("Failed to load from Google Drive")
        return None


def index_documents(documents, config: Config):
    """Index documents with hierarchical splitting."""
    console.print("\n[cyan]Indexing documents with hierarchical splitting...[/cyan]")

    # Initialize indexing pipeline
    pipeline = IndexingPipeline(config)
    pipeline.setup_document_store()

    # Process with hierarchical splitting
    result = pipeline.process_documents_hierarchical(documents)

    console.print(f"[green]✅ Created {result['chunks_created']} chunks[/green]")
    console.print(f"[green]✅ Indexed {result['chunks_written']} chunks successfully[/green]")

    # Show collection info
    info = pipeline.get_collection_info()
    console.print("\n[bold]Collection Info:[/bold]")
    console.print(f"  • Collection: {info['collection_name']}")
    console.print(f"  • Total documents: {info['document_count']}")
    console.print(f"  • Qdrant URL: {info['url']}")

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hierarchical RAG System with Google Drive")
    parser.add_argument(
        "--folder-id",
        type=str,
        help="Google Drive folder ID to load documents from",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        help="Maximum number of documents to load",
    )
    parser.add_argument(
        "--chat-only",
        action="store_true",
        help="Skip indexing and start chat interface",
    )
    parser.add_argument(
        "--clear-store",
        action="store_true",
        help="Clear the vector store before indexing",
    )

    args = parser.parse_args()

    # Initialize configuration
    config = Config()

    try:
        # Validate configuration
        config.validate()
    except ValueError as e:
        console.print(f"[red]❌ Configuration error: {e}[/red]")
        console.print("[yellow]Please check your .env file and credentials[/yellow]")
        sys.exit(1)

    # Chat only mode
    if args.chat_only:
        console.print("[cyan]Starting chat interface...[/cyan]")
        query_pipeline = QueryPipeline(config)
        query_pipeline.initialize()
        chat = ChatInterface(config, query_pipeline)
        chat.run()
        return

    # Determine folder ID
    folder_id = args.folder_id or config.GOOGLE_DRIVE_FOLDER_ID

    if not folder_id:
        console.print("[red]❌ No Google Drive folder ID specified[/red]")
        console.print(
            "[yellow]Please specify a folder ID with --folder-id or set GOOGLE_DRIVE_FOLDER_ID in .env[/yellow]"
        )
        console.print(
            "[yellow]Example: python run_hierarchical.py --folder-id YOUR_FOLDER_ID[/yellow]"
        )
        sys.exit(1)

    # Load documents from Google Drive
    documents = load_google_drive_folder(folder_id, config, args.max_docs)

    if documents:
        # Clear store if requested
        if args.clear_store:
            console.print("[yellow]Clearing existing vector store...[/yellow]")
            pipeline = IndexingPipeline(config)
            pipeline.setup_document_store()
            # Recreate the collection
            pipeline.document_store.recreate_index = True
            pipeline.setup_document_store()

        # Index documents
        index_documents(documents, config)

        # Ask if user wants to start chat
        console.print("\n[cyan]Documents indexed successfully![/cyan]")
        response = console.input("[yellow]Start chat interface? (y/n): [/yellow]")

        if response.lower() == "y":
            query_pipeline = QueryPipeline(config)
            query_pipeline.initialize()
            chat = ChatInterface(config, query_pipeline)
            chat.run()


if __name__ == "__main__":
    main()
