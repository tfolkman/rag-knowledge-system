from unittest.mock import Mock, patch

from src.chat_interface import ChatInterface
from src.config import Config


class TestChatInterface:
    """Test terminal chat interface for RAG system."""

    def setup_method(self):
        """Setup test fixtures."""
        # Reset singleton instance
        Config._instance = None
        self.config = Config()
        self.config.ollama_model_name = "llama3.2:latest"

        # Mock query pipeline
        self.mock_query_pipeline = Mock()
        self.mock_query_pipeline.query.return_value = {
            "query": "test question",
            "answer": "test answer",
            "sources": [
                {"content": "source 1", "metadata": {"name": "doc1.txt"}},
                {"content": "source 2", "metadata": {"name": "doc2.txt"}},
            ],
        }

    def test_interface_initialization(self):
        """Test that chat interface initializes correctly."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        assert interface.config == self.config
        assert interface.query_pipeline == self.mock_query_pipeline
        assert interface.console is not None
        assert interface.history == []

    def test_format_sources_with_sources(self):
        """Test formatting sources for display."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        sources = [
            {"content": "This is content 1", "metadata": {"name": "doc1.txt"}},
            {
                "content": "This is content 2 which is much longer and should be truncated when displayed",
                "metadata": {"name": "doc2.pdf"},
            },
        ]

        formatted = interface.format_sources(sources)

        assert "doc1.txt" in formatted
        assert "This is content 1" in formatted
        assert "doc2.pdf" in formatted
        # Content should be truncated
        assert len(formatted) < len(sources[1]["content"]) + 100

    def test_format_sources_empty(self):
        """Test formatting empty sources."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        formatted = interface.format_sources([])

        assert "No sources found" in formatted

    def test_display_welcome_message(self):
        """Test welcome message display."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        with patch.object(interface.console, "print") as mock_print:
            interface.display_welcome()

            # Should print welcome message
            assert mock_print.called
            # Check that print was called multiple times (panels)
            assert len(mock_print.call_args_list) >= 2

    def test_display_help(self):
        """Test help message display."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        with patch.object(interface.console, "print") as mock_print:
            interface.display_help()

            # Should print help commands
            assert mock_print.called
            # Check that print was called (at least for panel and newline)
            assert len(mock_print.call_args_list) >= 1

    @patch("builtins.input", return_value="test question")
    def test_get_user_input_normal(self, mock_input):
        """Test getting normal user input."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        user_input = interface.get_user_input()

        assert user_input == "test question"

    @patch("builtins.input", return_value="/help")
    def test_get_user_input_command(self, mock_input):
        """Test getting command user input."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        user_input = interface.get_user_input()

        assert user_input == "/help"

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_get_user_input_keyboard_interrupt(self, mock_input):
        """Test handling keyboard interrupt during input."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        user_input = interface.get_user_input()

        assert user_input == "/quit"

    def test_process_query_normal(self):
        """Test processing normal query."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        with patch.object(interface, "display_answer") as mock_display:
            result = interface.process_query("What is Python?")

            assert result is True  # Continue chatting
            self.mock_query_pipeline.query.assert_called_once_with("What is Python?")
            mock_display.assert_called_once()

            # Check history
            assert len(interface.history) == 1
            assert interface.history[0]["query"] == "test question"  # From mock result
            assert interface.history[0]["answer"] == "test answer"

    def test_process_query_help_command(self):
        """Test processing help command."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        with patch.object(interface, "display_help") as mock_help:
            result = interface.process_query("/help")

            assert result is True  # Continue chatting
            mock_help.assert_called_once()
            self.mock_query_pipeline.query.assert_not_called()

    def test_process_query_quit_command(self):
        """Test processing quit command."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        result = interface.process_query("/quit")

        assert result is False  # Stop chatting
        self.mock_query_pipeline.query.assert_not_called()

    def test_process_query_exit_command(self):
        """Test processing exit command."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        result = interface.process_query("/exit")

        assert result is False  # Stop chatting
        self.mock_query_pipeline.query.assert_not_called()

    def test_process_query_history_command(self):
        """Test processing history command."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        # Add some history
        interface.history = [
            {"query": "question 1", "answer": "answer 1"},
            {"query": "question 2", "answer": "answer 2"},
        ]

        with patch.object(interface, "display_history") as mock_history:
            result = interface.process_query("/history")

            assert result is True  # Continue chatting
            mock_history.assert_called_once()
            self.mock_query_pipeline.query.assert_not_called()

    def test_process_query_clear_command(self):
        """Test processing clear command."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        # Add some history
        interface.history = [{"query": "test", "answer": "test"}]

        with patch.object(interface.console, "clear") as mock_clear:
            result = interface.process_query("/clear")

            assert result is True  # Continue chatting
            mock_clear.assert_called_once()
            assert len(interface.history) == 0  # History cleared

    def test_process_query_empty_input(self):
        """Test processing empty input."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        result = interface.process_query("")

        assert result is True  # Continue chatting
        self.mock_query_pipeline.query.assert_not_called()

    def test_process_query_with_error(self):
        """Test processing query when pipeline raises error."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        # Mock pipeline to raise exception
        self.mock_query_pipeline.query.side_effect = Exception("Pipeline error")

        with patch.object(interface.console, "print") as mock_print:
            result = interface.process_query("test question")

            assert result is True  # Continue chatting despite error
            # Should display error message
            assert mock_print.called
            call_args = str(mock_print.call_args_list)
            assert "error" in call_args.lower() or "Error" in call_args

    def test_display_answer(self):
        """Test displaying answer with sources."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        query_result = {
            "query": "test question",
            "answer": "test answer",
            "sources": [{"content": "source content", "metadata": {"name": "doc.txt"}}],
        }

        with patch.object(interface.console, "print") as mock_print:
            interface.display_answer(query_result)

            assert mock_print.called
            # Check that print was called for answer and sources panels
            assert len(mock_print.call_args_list) >= 3  # Answer panel, newline, sources panel

    def test_display_history_with_history(self):
        """Test displaying history when history exists."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        interface.history = [
            {"query": "question 1", "answer": "answer 1"},
            {"query": "question 2", "answer": "answer 2"},
        ]

        with patch.object(interface.console, "print") as mock_print:
            interface.display_history()

            assert mock_print.called
            # Check that print was called for history panel
            assert len(mock_print.call_args_list) >= 1

    def test_display_history_empty(self):
        """Test displaying history when no history exists."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        with patch.object(interface.console, "print") as mock_print:
            interface.display_history()

            assert mock_print.called
            call_args = str(mock_print.call_args_list)
            assert "no history" in call_args.lower() or "No history" in call_args

    @patch("builtins.input")
    def test_start_chat_loop(self, mock_input):
        """Test the main chat loop."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        # Simulate user input sequence: question -> quit
        mock_input.side_effect = ["test question", "/quit"]

        with patch.object(interface, "display_welcome") as mock_welcome:
            with patch.object(
                interface, "process_query", side_effect=[True, False]
            ) as mock_process:
                interface.start()

                mock_welcome.assert_called_once()
                assert mock_process.call_count == 2
                mock_process.assert_any_call("test question")
                mock_process.assert_any_call("/quit")

    def test_run_method_delegates_to_start(self):
        """Test that run method delegates to start."""
        interface = ChatInterface(self.config, self.mock_query_pipeline)

        with patch.object(interface, "start") as mock_start:
            interface.run()
            mock_start.assert_called_once()
