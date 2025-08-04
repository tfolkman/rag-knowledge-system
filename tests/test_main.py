"""Tests for the main orchestration module."""

from unittest.mock import Mock, patch

import pytest

from src.main import RAGSystem, benchmark_system


class TestRAGSystem:
    """Test the RAG system."""

    @pytest.fixture
    def orchestrator(self):
        """Create a RAG system instance."""
        with patch("os.path.exists", return_value=True):
            with patch("src.main.Config"):
                return RAGSystem()

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.config is not None
        assert orchestrator.console is not None
        assert orchestrator.indexing_pipeline is None
        assert orchestrator.query_pipeline is None

    @patch("src.main.QueryPipeline")
    @patch("src.main.IndexingPipeline")
    def test_initialize_pipelines_success(self, mock_indexing, mock_query, orchestrator):
        """Test successful pipeline initialization."""
        # Setup mocks
        mock_indexing_instance = Mock()
        mock_indexing_instance.document_store = Mock()
        mock_indexing.return_value = mock_indexing_instance

        mock_query_instance = Mock()
        mock_query.return_value = mock_query_instance

        # Initialize pipelines
        result = orchestrator.initialize_pipelines()

        # Verify
        assert result is True
        assert orchestrator.indexing_pipeline == mock_indexing_instance
        assert orchestrator.query_pipeline == mock_query_instance
        mock_indexing.assert_called_once_with(orchestrator.config)
        mock_query.assert_called_once_with(orchestrator.config)

    @patch("src.main.IndexingPipeline")
    def test_initialize_pipelines_failure(self, mock_indexing, orchestrator):
        """Test pipeline initialization failure."""
        # Setup mock to raise exception
        mock_indexing.side_effect = Exception("Pipeline error")

        # Initialize pipelines
        result = orchestrator.initialize_pipelines()

        # Verify
        assert result is False
        assert orchestrator.indexing_pipeline is None
        assert orchestrator.query_pipeline is None

    def test_load_and_index_documents_no_pipelines(self, orchestrator):
        """Test document loading without initialized pipelines."""
        # Ensure pipelines are not initialized
        orchestrator.indexing_pipeline = None
        orchestrator.query_pipeline = None

        result = orchestrator.load_and_index_documents()
        assert result is False

    @patch("src.main.GoogleDriveLoader")
    def test_load_and_index_documents_no_documents(self, mock_loader, orchestrator):
        """Test document loading with no documents found."""
        # Setup pipelines
        orchestrator.indexing_pipeline = Mock()
        orchestrator.query_pipeline = Mock()

        # Setup loader mock
        mock_loader_instance = Mock()
        mock_loader_instance.load_documents.return_value = []
        mock_loader.return_value = mock_loader_instance

        # Load documents
        result = orchestrator.load_and_index_documents("folder_id")

        # Verify
        assert result is False
        mock_loader.assert_called_once_with(orchestrator.config)
        mock_loader_instance.load_documents.assert_called_once_with("folder_id", max_documents=None)

    @patch("src.main.GoogleDriveLoader")
    def test_load_and_index_documents_success(self, mock_loader, orchestrator):
        """Test successful document loading and indexing."""
        # Setup pipelines
        orchestrator.indexing_pipeline = Mock()
        orchestrator.indexing_pipeline.index_documents = Mock()
        orchestrator.query_pipeline = Mock()

        # Setup loader mock
        mock_loader_instance = Mock()
        mock_loader_instance.load_documents.return_value = [
            {"name": "doc1.txt", "content": "content1", "mime_type": "text/plain"},
            {"name": "doc2.pdf", "content": "content2", "mime_type": "application/pdf"},
        ]
        mock_loader.return_value = mock_loader_instance

        # Load documents
        with patch.object(orchestrator, "_display_loaded_documents"):
            result = orchestrator.load_and_index_documents("folder_id", max_docs=10)

        # Verify
        assert result is True
        mock_loader.assert_called_once_with(orchestrator.config)
        mock_loader_instance.load_documents.assert_called_once_with("folder_id", max_documents=10)
        orchestrator.indexing_pipeline.index_documents.assert_called_once()

    def test_display_loaded_documents(self, orchestrator):
        """Test document display table."""
        documents = [
            {"name": "doc1.txt", "content": "content1", "mime_type": "text/plain"},
            {"name": "doc2.pdf", "content": "content2", "mime_type": "application/pdf"},
        ]

        # Should not raise any exceptions
        orchestrator._display_loaded_documents(documents)

    def test_display_loaded_documents_many(self, orchestrator):
        """Test document display table with many documents."""
        documents = [
            {"name": f"doc{i}.txt", "content": f"content{i}", "mime_type": "text/plain"}
            for i in range(15)
        ]

        # Should not raise any exceptions and truncate display
        orchestrator._display_loaded_documents(documents)

    @patch("src.main.ChatInterface")
    def test_start_chat_no_pipelines(self, mock_chat, orchestrator):
        """Test starting chat without pipelines."""
        orchestrator.start_chat()
        mock_chat.assert_not_called()

    @patch("src.main.ChatInterface")
    def test_start_chat_success(self, mock_chat, orchestrator):
        """Test starting chat successfully."""
        # Setup pipelines
        orchestrator.indexing_pipeline = Mock()
        orchestrator.query_pipeline = Mock()

        # Setup chat mock
        mock_chat_instance = Mock()
        mock_chat.return_value = mock_chat_instance

        # Start chat
        orchestrator.start_chat()

        # Verify
        mock_chat.assert_called_once_with(orchestrator.query_pipeline)
        mock_chat_instance.run.assert_called_once()

    def test_display_system_info(self, orchestrator):
        """Test system info display."""
        # Check that system has the required components
        assert hasattr(orchestrator, "config")
        assert hasattr(orchestrator, "console")
        assert hasattr(orchestrator, "indexing_pipeline")
        assert hasattr(orchestrator, "query_pipeline")


class TestBenchmarkSystem:
    """Test the benchmark function."""

    @patch("src.main.Console")
    @patch("src.main.QueryPipeline")
    @patch("src.main.IndexingPipeline")
    @patch("src.main.Config")
    def test_benchmark_success(self, mock_config, mock_indexing, mock_query, mock_console):
        """Test successful benchmark."""
        # Setup mocks
        mock_config.return_value = Mock()
        mock_indexing.return_value = Mock()
        mock_query.return_value = Mock()

        # Run benchmark
        benchmark_system()

        # Verify all components were tested
        assert mock_config.call_count == 2  # Called in both RAGSystem() and in benchmark_system()
        mock_indexing.assert_called_once()
        mock_query.assert_called_once()

    @patch("src.main.Console")
    @patch("src.main.Config")
    def test_benchmark_failure(self, mock_config, mock_console):
        """Test benchmark failure."""
        # Setup mock to raise exception
        mock_config.side_effect = Exception("Config error")

        # Run benchmark - should raise exception
        with pytest.raises(Exception, match="Config error"):
            benchmark_system()
