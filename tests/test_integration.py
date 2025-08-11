from unittest.mock import Mock, patch

import pytest

from src.chat_interface import ChatInterface
from src.config import Config
from src.document_loader import GoogleDriveLoader
from src.indexing_pipeline import IndexingPipeline
from src.query_pipeline import QueryPipeline


class TestIntegration:
    """Integration tests for the complete RAG system."""

    def setup_method(self):
        """Setup test fixtures."""
        # Reset singleton instance
        Config._instance = None
        self.config = Config()
        self.config.google_credentials_path = "test_credentials.json"
        self.config.qdrant_url = "http://localhost:6333"
        self.config.qdrant_collection_name = "test_collection"
        self.config.ollama_base_url = "http://localhost:11434"
        self.config.ollama_model_name = "llama3.2:latest"
        self.config.ollama_embedding_model = "mxbai-embed-large"

    def test_config_integration(self):
        """Test that all components can use the same config."""
        # Test that each component can initialize with the config
        with patch("src.document_loader.service_account.Credentials"):
            loader = GoogleDriveLoader(self.config)
            assert loader.config == self.config

        indexing_pipeline = IndexingPipeline(self.config)
        assert indexing_pipeline.config == self.config

        query_pipeline = QueryPipeline(self.config)
        assert query_pipeline.config == self.config

        mock_query_pipeline = Mock()
        chat_interface = ChatInterface(self.config, mock_query_pipeline)
        assert chat_interface.config == self.config

    @patch("src.indexing_pipeline.QdrantDocumentStore")
    @patch("src.document_loader.service_account.Credentials")
    @patch("src.document_loader.build")
    def test_document_loading_to_indexing_integration(
        self, mock_build, mock_credentials, mock_qdrant
    ):
        """Test integration between document loader and indexing pipeline."""
        # Setup mocks
        mock_creds = Mock()
        mock_credentials.from_service_account_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service

        # Mock document listing and downloading
        mock_files = Mock()
        mock_service.files.return_value = mock_files
        mock_list = Mock()
        mock_files.list.return_value = mock_list
        mock_list.execute.return_value = {
            "files": [{"id": "1", "name": "test.txt", "mimeType": "text/plain"}]
        }

        # Mock document download
        with patch("src.document_loader.MediaIoBaseDownload") as mock_download:
            with patch("src.document_loader.io.BytesIO") as mock_bytesio:
                mock_buffer = Mock()
                mock_bytesio.return_value = mock_buffer
                mock_buffer.getvalue.return_value = b"This is test document content."

                mock_downloader = Mock()
                mock_download.return_value = mock_downloader
                mock_downloader.next_chunk.side_effect = [(True, Mock(progress=lambda: 1.0))]

                # Mock document store
                mock_store = Mock()
                mock_qdrant.return_value = mock_store

                # Create components
                loader = GoogleDriveLoader(self.config)
                loader.authenticate()

                indexing_pipeline = IndexingPipeline(self.config)
                indexing_pipeline.setup_document_store()

                # Load documents
                raw_documents = loader.load_documents(max_documents=1)

                # Verify documents were loaded
                assert len(raw_documents) == 1
                assert raw_documents[0]["content"] == "This is test document content."
                assert raw_documents[0]["metadata"]["name"] == "test.txt"

                # Convert documents to Haystack format
                haystack_documents = indexing_pipeline.convert_documents(raw_documents)

                # Verify conversion
                assert len(haystack_documents) == 1
                assert haystack_documents[0].content == "This is test document content."
                assert haystack_documents[0].meta["name"] == "test.txt"

    @patch("src.query_pipeline.QdrantDocumentStore")
    @patch("src.indexing_pipeline.QdrantDocumentStore")
    def test_indexing_to_query_integration(self, mock_indexing_qdrant, mock_query_qdrant):
        """Test integration between indexing and query pipelines."""
        # Mock document stores
        mock_indexing_store = Mock()
        mock_indexing_qdrant.return_value = mock_indexing_store
        mock_query_store = Mock()
        mock_query_qdrant.return_value = mock_query_store

        # Create pipelines
        indexing_pipeline = IndexingPipeline(self.config)
        indexing_pipeline.setup_document_store()

        query_pipeline = QueryPipeline(self.config)
        query_pipeline.setup_document_store()

        # Verify both use same collection configuration
        mock_indexing_qdrant.assert_called_once_with(
            url=self.config.qdrant_url,
            index=self.config.qdrant_collection_name,
            embedding_dim=1024,
            wait_result_from_api=True,
            recreate_index=False,
        )

        mock_query_qdrant.assert_called_once_with(
            url=self.config.qdrant_url,
            index=self.config.qdrant_collection_name,
            embedding_dim=1024,
            wait_result_from_api=True,
            recreate_index=False,
        )

        # Test collection info compatibility
        mock_indexing_store.count_documents.return_value = 42
        mock_query_store.count_documents.return_value = 42

        indexing_info = indexing_pipeline.get_collection_info()
        query_info = query_pipeline.get_collection_info()

        assert indexing_info["collection_name"] == query_info["collection_name"]
        assert indexing_info["document_count"] == query_info["document_count"]
        assert indexing_info["url"] == query_info["url"]

    @patch("src.query_pipeline.QdrantDocumentStore")
    def test_query_to_chat_integration(self, mock_qdrant):
        """Test integration between query pipeline and chat interface."""
        # Mock document store and pipeline
        mock_store = Mock()
        mock_qdrant.return_value = mock_store

        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {
            "llm": {"replies": ["This is a test answer about Python programming."]},
            "retriever": {
                "documents": [
                    Mock(
                        content="Python is a programming language",
                        meta={"name": "python.txt"},
                    )
                ]
            },
        }

        # Create query pipeline
        query_pipeline = QueryPipeline(self.config)
        query_pipeline.setup_document_store()

        # Mock components for the new direct-call approach
        mock_embedder = Mock()
        mock_embedder.run.return_value = {"embedding": [0.1, 0.2, 0.3]}

        mock_retriever = Mock()
        mock_retriever.run.return_value = {
            "documents": [
                Mock(
                    content="Python is a programming language",
                    meta={"name": "python.txt"},
                )
            ]
        }

        mock_prompt_builder = Mock()
        mock_prompt_builder.run.return_value = {"prompt": "Generated prompt"}

        mock_generator = Mock()
        mock_generator.run.return_value = {
            "replies": ["This is a test answer about Python programming."]
        }

        query_pipeline.embedder = mock_embedder
        query_pipeline.retriever = mock_retriever
        query_pipeline.prompt_builder = mock_prompt_builder
        query_pipeline.generator = mock_generator
        # No need to set pipeline - we use direct component calls

        # Create chat interface
        chat_interface = ChatInterface(self.config, query_pipeline)

        # Test query through chat interface
        with patch.object(chat_interface, "display_answer") as mock_display:
            result = chat_interface.process_query("What is Python?")

            assert result is True  # Continue chatting
            mock_display.assert_called_once()

            # Verify query was processed
            call_args = mock_display.call_args[0][0]
            assert call_args["answer"] == "This is a test answer about Python programming."
            assert len(call_args["sources"]) == 1
            assert call_args["sources"][0]["content"] == "Python is a programming language"

    @patch("src.query_pipeline.QdrantDocumentStore")
    @patch("src.indexing_pipeline.QdrantDocumentStore")
    @patch("src.document_loader.service_account.Credentials")
    @patch("src.document_loader.build")
    def test_end_to_end_document_flow_simulation(
        self, mock_build, mock_credentials, mock_indexing_qdrant, mock_query_qdrant
    ):
        """Test simulated end-to-end document flow from loading to querying."""
        # Setup document loader mocks
        mock_creds = Mock()
        mock_credentials.from_service_account_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service

        # Mock document listing
        mock_files = Mock()
        mock_service.files.return_value = mock_files
        mock_list = Mock()
        mock_files.list.return_value = mock_list
        mock_list.execute.return_value = {
            "files": [{"id": "1", "name": "python_guide.txt", "mimeType": "text/plain"}]
        }

        # Mock document download
        with patch("src.document_loader.MediaIoBaseDownload") as mock_download:
            with patch("src.document_loader.io.BytesIO") as mock_bytesio:
                mock_buffer = Mock()
                mock_bytesio.return_value = mock_buffer
                mock_buffer.getvalue.return_value = b"Python is a high-level programming language known for its simplicity and readability."

                mock_downloader = Mock()
                mock_download.return_value = mock_downloader
                mock_downloader.next_chunk.side_effect = [(True, Mock(progress=lambda: 1.0))]

                # Mock document stores
                mock_indexing_store = Mock()
                mock_indexing_qdrant.return_value = mock_indexing_store
                mock_query_store = Mock()
                mock_query_qdrant.return_value = mock_query_store

                # Mock indexing pipeline
                mock_indexing_pipeline = Mock()
                mock_indexing_pipeline.run.return_value = {"writer": {"documents_written": 1}}

                # Mock query pipeline
                mock_query_pipeline = Mock()
                mock_query_pipeline.run.return_value = {
                    "llm": {"replies": ["Python is a programming language known for simplicity."]},
                    "retriever": {
                        "documents": [
                            Mock(
                                content="Python is a high-level programming language",
                                meta={"name": "python_guide.txt"},
                            )
                        ]
                    },
                }

                # Step 1: Load documents
                loader = GoogleDriveLoader(self.config)
                loader.authenticate()
                raw_documents = loader.load_documents(max_documents=1)

                # Step 2: Index documents
                indexing_pipeline = IndexingPipeline(self.config)
                indexing_pipeline.setup_document_store()
                indexing_pipeline.pipeline = mock_indexing_pipeline

                indexing_result = indexing_pipeline.process_documents(raw_documents)

                # Step 3: Query documents
                query_pipeline = QueryPipeline(self.config)
                query_pipeline.setup_document_store()

                # Mock components for direct call approach
                mock_embedder = Mock()
                mock_embedder.run.return_value = {"embedding": [0.1, 0.2, 0.3]}

                mock_retriever = Mock()
                mock_retriever.run.return_value = {
                    "documents": [
                        Mock(
                            content="Python is a high-level programming language",
                            meta={"name": "python_guide.txt"},
                        )
                    ]
                }

                mock_prompt_builder = Mock()
                mock_prompt_builder.run.return_value = {"prompt": "Generated prompt"}

                mock_generator = Mock()
                mock_generator.run.return_value = {
                    "replies": ["Python is a programming language known for simplicity."]
                }

                query_pipeline.embedder = mock_embedder
                query_pipeline.retriever = mock_retriever
                query_pipeline.prompt_builder = mock_prompt_builder
                query_pipeline.generator = mock_generator
                # No need to set pipeline - we use direct component calls

                query_result = query_pipeline.query("What is Python?")

                # Verify end-to-end flow
                assert len(raw_documents) == 1
                assert indexing_result["documents_processed"] == 1
                assert indexing_result["documents_written"] == 1
                assert (
                    query_result["answer"]
                    == "Python is a programming language known for simplicity."
                )
                assert len(query_result["sources"]) == 1

    def test_error_handling_integration(self):
        """Test error handling across components."""
        # Test config validation
        with pytest.raises(ValueError, match="Google credentials file not found"):
            config = Config()
            config.google_credentials_path = None
            config.validate()

        # Test document loader without authentication
        loader = GoogleDriveLoader(self.config)
        with pytest.raises(ValueError, match="Not authenticated"):
            loader.list_documents()

        # Test indexing pipeline without document store
        indexing_pipeline = IndexingPipeline(self.config)
        with pytest.raises(ValueError, match="Document store not initialized"):
            indexing_pipeline.create_indexing_pipeline()

        # Test query pipeline without document store
        query_pipeline = QueryPipeline(self.config)
        with pytest.raises(ValueError, match="Document store not initialized"):
            query_pipeline.create_query_pipeline()

    def test_configuration_consistency_across_components(self):
        """Test that configuration values are consistent across all components."""
        # All components should use the same Qdrant configuration
        components = []

        # Document loader uses config for credentials
        with patch("src.document_loader.service_account.Credentials"):
            loader = GoogleDriveLoader(self.config)
            components.append(loader)

        # Indexing pipeline uses config for Qdrant and Ollama
        indexing_pipeline = IndexingPipeline(self.config)
        components.append(indexing_pipeline)

        # Query pipeline uses config for Qdrant and Ollama
        query_pipeline = QueryPipeline(self.config)
        components.append(query_pipeline)

        # Chat interface uses config for display
        mock_query_pipeline = Mock()
        chat_interface = ChatInterface(self.config, mock_query_pipeline)
        components.append(chat_interface)

        # Verify all components share the same config instance
        for component in components:
            assert component.config == self.config
            assert component.config.qdrant_url == "http://localhost:6333"
            assert component.config.qdrant_collection_name == "test_collection"
