from unittest.mock import Mock, patch

import pytest

from src.config import Config
from src.indexing_pipeline import IndexingPipeline


class TestIndexingPipeline:
    """Test indexing pipeline for RAG system."""

    def setup_method(self):
        """Setup test fixtures."""
        # Reset singleton instance
        Config._instance = None
        self.config = Config()
        self.config.qdrant_url = "http://localhost:6333"
        self.config.qdrant_collection_name = "test_collection"
        self.config.ollama_embedding_model = "mxbai-embed-large"
        self.config.chunk_size = 500
        self.config.chunk_overlap = 50

    def test_pipeline_initialization(self):
        """Test that indexing pipeline initializes correctly."""
        pipeline = IndexingPipeline(self.config)

        assert pipeline.config == self.config
        assert pipeline.document_store is None
        assert pipeline.pipeline is None

    @patch("src.indexing_pipeline.QdrantDocumentStore")
    def test_setup_document_store(self, mock_qdrant):
        """Test document store setup."""
        mock_store = Mock()
        mock_qdrant.return_value = mock_store

        pipeline = IndexingPipeline(self.config)
        pipeline.setup_document_store()

        assert pipeline.document_store == mock_store
        mock_qdrant.assert_called_once_with(
            url=self.config.qdrant_url,
            index=self.config.qdrant_collection_name,
            embedding_dim=1024,  # mxbai-embed-large dimension
            wait_result_from_api=True,
            recreate_index=False,
        )

    @patch("src.indexing_pipeline.Pipeline")
    @patch("src.indexing_pipeline.OllamaDocumentEmbedder")
    @patch("src.indexing_pipeline.DocumentSplitter")
    @patch("src.indexing_pipeline.DocumentWriter")
    def test_create_indexing_pipeline(
        self, mock_writer, mock_splitter, mock_embedder, mock_pipeline
    ):
        """Test creation of indexing pipeline."""
        # Mock components
        mock_embedder_instance = Mock()
        mock_embedder.return_value = mock_embedder_instance
        mock_splitter_instance = Mock()
        mock_splitter.return_value = mock_splitter_instance
        mock_writer_instance = Mock()
        mock_writer.return_value = mock_writer_instance
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        pipeline = IndexingPipeline(self.config)
        pipeline.document_store = Mock()  # Mock document store
        pipeline.create_indexing_pipeline()

        assert pipeline.pipeline == mock_pipeline_instance

        # Verify component creation
        mock_embedder.assert_called_once_with(
            model="mxbai-embed-large", url="http://localhost:11434"
        )
        mock_splitter.assert_called_once_with(split_by="word", split_length=500, split_overlap=50)
        mock_writer.assert_called_once_with(document_store=pipeline.document_store)

        # Verify pipeline connections
        assert mock_pipeline_instance.add_component.call_count == 3
        assert mock_pipeline_instance.connect.call_count == 2

    def test_create_pipeline_without_document_store(self):
        """Test that creating pipeline without document store raises error."""
        pipeline = IndexingPipeline(self.config)

        with pytest.raises(ValueError, match="Document store not initialized"):
            pipeline.create_indexing_pipeline()

    def test_process_documents_without_pipeline(self):
        """Test that processing documents without pipeline raises error."""
        pipeline = IndexingPipeline(self.config)

        with pytest.raises(ValueError, match="Pipeline not initialized"):
            pipeline.process_documents([])

    def test_convert_raw_documents_to_haystack_format(self):
        """Test conversion of raw documents to Haystack format."""
        pipeline = IndexingPipeline(self.config)

        raw_documents = [
            {
                "content": b"This is test content 1",
                "metadata": {
                    "name": "test1.txt",
                    "id": "1",
                    "mimeType": "text/plain",
                    "source": "google_drive",
                },
            },
            {
                "content": b"This is test content 2",
                "metadata": {
                    "name": "test2.txt",
                    "id": "2",
                    "mimeType": "text/plain",
                    "source": "google_drive",
                },
            },
        ]

        haystack_documents = pipeline.convert_documents(raw_documents)

        assert len(haystack_documents) == 2
        assert haystack_documents[0].content == "This is test content 1"
        assert haystack_documents[0].meta["name"] == "test1.txt"
        assert haystack_documents[0].meta["source"] == "google_drive"
        assert haystack_documents[1].content == "This is test content 2"
        assert haystack_documents[1].meta["name"] == "test2.txt"

    def test_convert_non_text_documents(self):
        """Test handling of non-text documents."""
        pipeline = IndexingPipeline(self.config)

        raw_documents = [
            {
                "content": b"PDF binary content here",
                "metadata": {
                    "name": "test.pdf",
                    "id": "1",
                    "mimeType": "application/pdf",
                    "source": "google_drive",
                },
            }
        ]

        # PDF is now supported as a text document type
        haystack_documents = pipeline.convert_documents(raw_documents)
        assert len(haystack_documents) == 1  # PDF is now processed

    @patch("src.indexing_pipeline.QdrantDocumentStore")
    def test_process_documents_success(self, mock_qdrant):
        """Test successful document processing."""
        mock_store = Mock()
        mock_qdrant.return_value = mock_store

        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {"writer": {"documents_written": 2}}

        pipeline = IndexingPipeline(self.config)
        pipeline.setup_document_store()
        pipeline.pipeline = mock_pipeline

        raw_documents = [
            {
                "content": b"Test content 1",
                "metadata": {
                    "name": "test1.txt",
                    "mimeType": "text/plain",
                    "source": "google_drive",
                },
            },
            {
                "content": b"Test content 2",
                "metadata": {
                    "name": "test2.txt",
                    "mimeType": "text/plain",
                    "source": "google_drive",
                },
            },
        ]

        result = pipeline.process_documents(raw_documents)

        assert result["documents_processed"] == 2
        assert result["documents_written"] == 2
        mock_pipeline.run.assert_called_once()

    def test_get_collection_info_without_document_store(self):
        """Test getting collection info without document store raises error."""
        pipeline = IndexingPipeline(self.config)

        with pytest.raises(ValueError, match="Document store not initialized"):
            pipeline.get_collection_info()

    @patch("src.indexing_pipeline.QdrantDocumentStore")
    def test_get_collection_info_success(self, mock_qdrant):
        """Test getting collection information."""
        mock_store = Mock()
        mock_store.count_documents.return_value = 42
        mock_qdrant.return_value = mock_store

        pipeline = IndexingPipeline(self.config)
        pipeline.setup_document_store()

        info = pipeline.get_collection_info()

        assert info["collection_name"] == "test_collection"
        assert info["document_count"] == 42
        assert info["url"] == "http://localhost:6333"

    def test_initialize_complete_pipeline(self):
        """Test complete pipeline initialization."""
        with patch("src.indexing_pipeline.QdrantDocumentStore") as mock_qdrant:
            mock_store = Mock()
            mock_qdrant.return_value = mock_store

            with patch.object(IndexingPipeline, "create_indexing_pipeline") as mock_create:
                pipeline = IndexingPipeline(self.config)
                pipeline.initialize()

                # Should setup document store and create pipeline
                assert pipeline.document_store == mock_store
                mock_create.assert_called_once()

    def test_cleanup(self):
        """Test pipeline cleanup."""
        pipeline = IndexingPipeline(self.config)
        pipeline.document_store = Mock()
        pipeline.pipeline = Mock()

        pipeline.cleanup()

        assert pipeline.document_store is None
        assert pipeline.pipeline is None
