#!/usr/bin/env python
"""Batch ingestion script for GitHub repositories into RAG system."""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.config import Config
from src.github_loader import GitHubRepositoryLoader
from src.indexing_pipeline import IndexingPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()


def main():
    """Main entry point for GitHub batch ingestion."""
    parser = argparse.ArgumentParser(description="Batch ingest GitHub repositories into RAG system")
    parser.add_argument(
        "--repos-file",
        type=str,
        required=True,
        help="Path to file containing repository list (one per line)",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="~/Coding",
        help="Directory to check for existing repos and clone new ones (default: ~/Coding)",
    )
    parser.add_argument(
        "--force-clone",
        action="store_true",
        help="Force fresh clone even if repository exists locally",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Don't update existing repositories with git pull",
    )
    parser.add_argument(
        "--clear-store",
        action="store_true",
        help="Clear the vector store before indexing",
    )
    parser.add_argument(
        "--max-docs-per-repo",
        type=int,
        help="Maximum number of documents to load per repository",
    )
    parser.add_argument(
        "--max-file-size",
        type=float,
        default=10.0,
        help="Maximum file size in MB to process (default: 10MB)",
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

    # Resolve paths
    repos_file = Path(args.repos_file)
    local_dir = Path(args.local_dir).expanduser()

    if not repos_file.exists():
        console.print(f"[red]❌ Repository list file not found: {repos_file}[/red]")
        sys.exit(1)

    # Read repository list
    with open(repos_file, "r") as f:
        repo_list = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not repo_list:
        console.print(f"[red]❌ No repositories found in {repos_file}[/red]")
        sys.exit(1)

    console.print(f"[cyan]Loading {len(repo_list)} repositories from {repos_file}[/cyan]")
    console.print(f"[cyan]Local directory: {local_dir}[/cyan]")
    console.print()

    # Initialize loader
    loader = GitHubRepositoryLoader(config)
    loader.local_repos_dir = local_dir
    loader.max_file_size_mb = args.max_file_size

    # Process repositories
    all_documents = []
    repo_stats = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing repositories...", total=len(repo_list))

        for idx, repo_identifier in enumerate(repo_list, 1):
            progress.update(
                task, description=f"[{idx}/{len(repo_list)}] Processing {repo_identifier}..."
            )

            try:
                # Process repository
                documents, status = loader.process_repository(
                    repo_identifier,
                    local_dir=local_dir,
                    force_clone=args.force_clone,
                    update_existing=not args.no_update,
                    max_documents=args.max_docs_per_repo,
                )

                all_documents.extend(documents)
                repo_stats.append(
                    {
                        "repo": repo_identifier,
                        "status": "✅ Success",
                        "documents": len(documents),
                        "message": status,
                    }
                )

            except Exception as e:
                logger.error(f"Error processing {repo_identifier}: {e}")
                repo_stats.append(
                    {
                        "repo": repo_identifier,
                        "status": "❌ Failed",
                        "documents": 0,
                        "message": str(e),
                    }
                )

            progress.advance(task)

    # Display results summary
    console.print()
    table = Table(
        title="Repository Processing Summary", show_header=True, header_style="bold magenta"
    )
    table.add_column("Repository", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Documents", style="yellow")
    table.add_column("Details", style="white")

    for stat in repo_stats:
        table.add_row(
            stat["repo"],
            stat["status"],
            str(stat["documents"]),
            stat["message"][:50] + "..." if len(stat["message"]) > 50 else stat["message"],
        )

    console.print(table)

    # Statistics
    successful_repos = sum(1 for s in repo_stats if s["status"] == "✅ Success")
    console.print()
    console.print("[bold]Statistics:[/bold]")
    console.print(f"  • Successful repositories: {successful_repos}/{len(repo_list)}")
    console.print(f"  • Total documents loaded: {len(all_documents)}")

    if not all_documents:
        console.print("[yellow]⚠️  No documents were loaded. Exiting.[/yellow]")
        sys.exit(0)

    # Index documents
    console.print()
    console.print("[cyan]Indexing documents into vector store...[/cyan]")

    # Initialize indexing pipeline
    pipeline = IndexingPipeline(config)

    # Clear store if requested
    if args.clear_store:
        console.print("[yellow]Clearing existing vector store...[/yellow]")
        pipeline.document_store = None
        pipeline.setup_document_store()
        # Force recreate
        pipeline.document_store.recreate_index = True

    pipeline.setup_document_store()

    # Process with hierarchical splitting
    result = pipeline.process_documents_hierarchical(all_documents)

    console.print(f"[green]✅ Created {result['chunks_created']} chunks[/green]")
    console.print(f"[green]✅ Indexed {result['chunks_written']} chunks successfully[/green]")

    # Show collection info
    info = pipeline.get_collection_info()
    console.print()
    console.print("[bold]Collection Info:[/bold]")
    console.print(f"  • Collection: {info['collection_name']}")
    console.print(f"  • Total documents: {info['document_count']}")
    console.print(f"  • Qdrant URL: {info['url']}")

    # Show category distribution
    categories = {}
    for doc in all_documents:
        cat = doc.meta.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    console.print()
    console.print("[bold]Repository Distribution in Vector Store:[/bold]")
    for cat, count in sorted(categories.items()):
        console.print(f"  • {cat}: {count} documents")

    console.print()
    console.print("[green]✅ GitHub repository ingestion completed successfully![/green]")
    console.print("[cyan]You can now use 'just chat' to query the indexed repositories.[/cyan]")


if __name__ == "__main__":
    main()
