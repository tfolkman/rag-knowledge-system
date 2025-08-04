from typing import Any, Dict, List

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from src.config import Config
from src.query_pipeline import QueryPipeline


class ChatInterface:
    """Terminal chat interface for RAG system using Rich."""

    def __init__(self, config: Config, query_pipeline: QueryPipeline):
        """Initialize the chat interface.

        Args:
            config: Configuration object
            query_pipeline: Query pipeline for processing questions
        """
        self.config = config
        self.query_pipeline = query_pipeline
        self.console = Console()
        self.history: List[Dict[str, str]] = []

    def display_welcome(self) -> None:
        """Display welcome message."""
        title = Text("🧠 RAG Knowledge System", style="bold blue")
        welcome_text = f"""
Welcome to your RAG-powered Knowledge Assistant!

Model: {self.config.ollama_model_name}
Collection: {self.config.qdrant_collection_name}

Type your questions and I'll search through your documents to provide answers.
Type /help for available commands.
        """

        panel = Panel(welcome_text.strip(), title=title, border_style="blue", padding=(1, 2))

        self.console.print()
        self.console.print(panel)
        self.console.print()

    def display_help(self) -> None:
        """Display help message with available commands."""
        help_text = """
Available commands:

/help     - Show this help message
/history  - Show conversation history
/clear    - Clear screen and conversation history
/quit     - Exit the application
/exit     - Exit the application

Simply type your questions to get answers based on your documents.
        """

        panel = Panel(help_text.strip(), title="💡 Help", border_style="green", padding=(1, 2))

        self.console.print(panel)
        self.console.print()

    def get_user_input(self) -> str:
        """Get user input with prompt.

        Returns:
            User input string
        """
        try:
            return Prompt.ask("[bold cyan]You[/bold cyan]").strip()
        except KeyboardInterrupt:
            return "/quit"
        except EOFError:
            return "/quit"

    def format_sources(self, sources: List[Dict[str, Any]], max_length: int = 100) -> str:
        """Format sources for display.

        Args:
            sources: List of source documents
            max_length: Maximum length for content preview

        Returns:
            Formatted sources string
        """
        if not sources:
            return "No sources found."

        formatted_sources = []
        for i, source in enumerate(sources, 1):
            content = source.get("content", "")
            metadata = source.get("metadata", {})
            name = metadata.get("name", f"Document {i}")

            # Truncate content for display
            if len(content) > max_length:
                content = content[:max_length] + "..."

            formatted_sources.append(f"[bold]{i}. {name}[/bold]\n{escape(content)}")

        return "\n\n".join(formatted_sources)

    def display_answer(self, query_result: Dict[str, Any]) -> None:
        """Display the answer and sources.

        Args:
            query_result: Result from query pipeline
        """
        answer = query_result.get("answer", "No answer generated.")
        sources = query_result.get("sources", [])

        # Display answer
        answer_panel = Panel(
            escape(answer), title="🤖 Assistant", border_style="green", padding=(1, 2)
        )
        self.console.print(answer_panel)

        # Display sources if available
        if sources:
            self.console.print()
            sources_text = self.format_sources(sources)
            sources_panel = Panel(
                sources_text,
                title=f"📚 Sources ({len(sources)} found)",
                border_style="yellow",
                padding=(1, 2),
            )
            self.console.print(sources_panel)

        self.console.print()

    def display_history(self) -> None:
        """Display conversation history."""
        if not self.history:
            self.console.print("[yellow]No history available.[/yellow]")
            self.console.print()
            return

        history_text = []
        for i, entry in enumerate(self.history, 1):
            history_text.append(f"[bold cyan]{i}. Q:[/bold cyan] {escape(entry['query'])}")
            history_text.append(
                f"[bold green]   A:[/bold green] {escape(entry['answer'][:200])}{'...' if len(entry['answer']) > 200 else ''}"
            )
            history_text.append("")

        panel = Panel(
            "\n".join(history_text).strip(),
            title=f"📋 History ({len(self.history)} entries)",
            border_style="magenta",
            padding=(1, 2),
        )

        self.console.print(panel)
        self.console.print()

    def process_query(self, user_input: str) -> bool:
        """Process user query or command.

        Args:
            user_input: User input string

        Returns:
            True to continue chat, False to exit
        """
        if not user_input:
            return True

        # Handle commands
        if user_input.startswith("/"):
            command = user_input.lower()

            if command == "/help":
                self.display_help()
                return True

            elif command == "/quit" or command == "/exit":
                self.console.print("[yellow]Goodbye! 👋[/yellow]")
                return False

            elif command == "/history":
                self.display_history()
                return True

            elif command == "/clear":
                self.console.clear()
                self.history.clear()
                self.display_welcome()
                return True

            else:
                self.console.print(f"[red]Unknown command: {user_input}[/red]")
                self.console.print("[yellow]Type /help for available commands.[/yellow]")
                self.console.print()
                return True

        # Process regular query
        try:
            with self.console.status(
                "[bold green]Searching and generating answer...", spinner="dots"
            ):
                result = self.query_pipeline.query(user_input)

            self.display_answer(result)

            # Add to history
            self.history.append({"query": result["query"], "answer": result["answer"]})

        except Exception as e:
            self.console.print(f"[red]Error processing query: {str(e)}[/red]")
            self.console.print("[yellow]Please try again or check your configuration.[/yellow]")
            self.console.print()

        return True

    def start(self) -> None:
        """Start the chat interface."""
        self.display_welcome()

        while True:
            try:
                user_input = self.get_user_input()
                if not self.process_query(user_input):
                    break
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Unexpected error: {str(e)}[/red]")
                self.console.print("[yellow]Please try again.[/yellow]")
                continue

    def run(self) -> None:
        """Run the chat interface (alias for start)."""
        self.start()
