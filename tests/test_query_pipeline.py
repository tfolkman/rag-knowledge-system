from unittest.mock import Mock, patch

import pytest

from src.config import Config
from src.query_pipeline import QueryPipeline


class TestQueryPipeline:
    """Test query pipeline for RAG system."""

    def setup_method(self):
        """Setup test fixtures."""
        # Reset singleton instance
        Config._instance = None
        self.config = Config()
        self.config.qdrant_url = "http://localhost:6333"
        self.config.qdrant_collection_name = "test_collection"
        self.config.ollama_base_url = "http://localhost:11434"
        self.config.ollama_model_name = "llama3.2:latest"
        self.config.ollama_embedding_model = "mxbai-embed-large"

    def test_pipeline_initialization(self):
        """Test that query pipeline initializes correctly."""
        pipeline = QueryPipeline(self.config)

        assert pipeline.config == self.config
        assert pipeline.document_store is None
        assert pipeline.pipeline is None

    @patch("src.query_pipeline.QdrantDocumentStore")
    def test_setup_document_store(self, mock_qdrant):
        """Test document store setup."""
        mock_store = Mock()
        mock_qdrant.return_value = mock_store

        pipeline = QueryPipeline(self.config)
        pipeline.setup_document_store()

        assert pipeline.document_store == mock_store
        mock_qdrant.assert_called_once_with(
            url=self.config.qdrant_url,
            index=self.config.qdrant_collection_name,
            embedding_dim=1024,  # mxbai-embed-large dimension
            wait_result_from_api=True,
            recreate_index=False,
        )

    @patch("src.query_pipeline.Pipeline")
    @patch("src.query_pipeline.OllamaGenerator")
    @patch("src.query_pipeline.OllamaTextEmbedder")
    @patch("src.query_pipeline.QdrantEmbeddingRetriever")
    @patch("src.query_pipeline.PromptBuilder")
    def test_create_query_pipeline(
        self,
        mock_prompt_builder,
        mock_retriever,
        mock_embedder,
        mock_generator,
        mock_pipeline,
    ):
        """Test creation of query pipeline."""
        # Mock components
        mock_embedder_instance = Mock()
        mock_embedder.return_value = mock_embedder_instance
        mock_retriever_instance = Mock()
        mock_retriever.return_value = mock_retriever_instance
        mock_generator_instance = Mock()
        mock_generator.return_value = mock_generator_instance
        mock_prompt_builder_instance = Mock()
        mock_prompt_builder.return_value = mock_prompt_builder_instance
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        pipeline = QueryPipeline(self.config)
        pipeline.document_store = Mock()  # Mock document store
        pipeline.create_query_pipeline()

        assert pipeline.pipeline == mock_pipeline_instance

        # Verify component creation
        mock_embedder.assert_called_once_with(
            model="mxbai-embed-large", url="http://localhost:11434"
        )
        mock_retriever.assert_called_once_with(document_store=pipeline.document_store, top_k=5)
        mock_generator.assert_called_once_with(
            model="llama3.2:latest", url="http://localhost:11434"
        )
        mock_prompt_builder.assert_called_once()

        # Verify pipeline connections
        assert mock_pipeline_instance.add_component.call_count == 4
        assert mock_pipeline_instance.connect.call_count == 3

    def test_create_pipeline_without_document_store(self):
        """Test that creating pipeline without document store raises error."""
        pipeline = QueryPipeline(self.config)

        with pytest.raises(ValueError, match="Document store not initialized"):
            pipeline.create_query_pipeline()

    def test_query_without_pipeline(self):
        """Test that querying without pipeline raises error."""
        pipeline = QueryPipeline(self.config)

        with pytest.raises(ValueError, match="Pipeline not initialized"):
            pipeline.query("test query")

    @patch("src.query_pipeline.QdrantDocumentStore")
    def test_query_success(self, mock_qdrant):
        """Test successful query execution."""
        mock_store = Mock()
        mock_qdrant.return_value = mock_store

        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {
            "llm": {"replies": ["This is the generated answer based on the context."]},
            "retriever": {
                "documents": [
                    Mock(content="Retrieved document 1", meta={"name": "doc1.txt"}),
                    Mock(content="Retrieved document 2", meta={"name": "doc2.txt"}),
                ]
            },
        }

        pipeline = QueryPipeline(self.config)
        pipeline.setup_document_store()
        pipeline.pipeline = mock_pipeline

        result = pipeline.query("What is the capital of France?")

        assert result["answer"] == "This is the generated answer based on the context."
        assert len(result["sources"]) == 2
        assert result["sources"][0]["content"] == "Retrieved document 1"
        assert result["sources"][0]["metadata"]["name"] == "doc1.txt"
        assert result["query"] == "What is the capital of France?"

        mock_pipeline.run.assert_called_once()

    @patch("src.query_pipeline.QdrantDocumentStore")
    def test_query_with_no_sources(self, mock_qdrant):
        """Test query execution when no sources are retrieved."""
        mock_store = Mock()
        mock_qdrant.return_value = mock_store

        # Mock pipeline with no retrieved documents
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {
            "llm": {"replies": ["I could not find relevant information to answer your question."]},
            "retriever": {"documents": []},
        }

        pipeline = QueryPipeline(self.config)
        pipeline.setup_document_store()
        pipeline.pipeline = mock_pipeline

        result = pipeline.query("What is the capital of Mars?")

        assert result["answer"] == "I could not find relevant information to answer your question."
        assert len(result["sources"]) == 0
        assert result["query"] == "What is the capital of Mars?"

    def test_get_default_prompt_template(self):
        """Test default prompt template generation."""
        pipeline = QueryPipeline(self.config)
        template = pipeline.get_default_prompt_template()

        assert "context" in template
        assert "question" in template
        assert "You are a helpful assistant" in template

    @patch("src.query_pipeline.QdrantDocumentStore")
    def test_query_with_custom_top_k(self, mock_qdrant):
        """Test query with custom top_k parameter."""
        mock_store = Mock()
        mock_qdrant.return_value = mock_store

        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {
            "llm": {"replies": ["Answer"]},
            "retriever": {"documents": []},
        }

        pipeline = QueryPipeline(self.config)
        pipeline.setup_document_store()
        pipeline.pipeline = mock_pipeline

        pipeline.query("test query", top_k=10)

        # Check that pipeline was called with correct parameters
        call_args = mock_pipeline.run.call_args[0][0]
        assert call_args["retriever"]["top_k"] == 10

    def test_get_collection_info_without_document_store(self):
        """Test getting collection info without document store raises error."""
        pipeline = QueryPipeline(self.config)

        with pytest.raises(ValueError, match="Document store not initialized"):
            pipeline.get_collection_info()

    @patch("src.query_pipeline.QdrantDocumentStore")
    def test_get_collection_info_success(self, mock_qdrant):
        """Test getting collection information."""
        mock_store = Mock()
        mock_store.count_documents.return_value = 42
        mock_qdrant.return_value = mock_store

        pipeline = QueryPipeline(self.config)
        pipeline.setup_document_store()

        info = pipeline.get_collection_info()

        assert info["collection_name"] == "test_collection"
        assert info["document_count"] == 42
        assert info["url"] == "http://localhost:6333"

    def test_initialize_complete_pipeline(self):
        """Test complete pipeline initialization."""
        with patch("src.query_pipeline.QdrantDocumentStore") as mock_qdrant:
            mock_store = Mock()
            mock_qdrant.return_value = mock_store

            with patch.object(QueryPipeline, "create_query_pipeline") as mock_create:
                pipeline = QueryPipeline(self.config)
                pipeline.initialize()

                # Should setup document store and create pipeline
                assert pipeline.document_store == mock_store
                mock_create.assert_called_once()

    def test_cleanup(self):
        """Test pipeline cleanup."""
        pipeline = QueryPipeline(self.config)
        pipeline.document_store = Mock()
        pipeline.pipeline = Mock()

        pipeline.cleanup()

        assert pipeline.document_store is None
        assert pipeline.pipeline is None

    @patch("src.query_pipeline.QdrantDocumentStore")
    def test_query_with_context_formatting(self, mock_qdrant):
        """Test that query properly calls pipeline with question and retriever parameters."""
        mock_store = Mock()
        mock_qdrant.return_value = mock_store

        # Mock documents with content
        mock_docs = [
            Mock(content="Document 1 content", meta={"name": "doc1.txt"}),
            Mock(content="Document 2 content", meta={"name": "doc2.txt"}),
        ]

        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {
            "llm": {"replies": ["Generated answer"]},
            "retriever": {"documents": mock_docs},
        }

        pipeline = QueryPipeline(self.config)
        pipeline.setup_document_store()
        pipeline.pipeline = mock_pipeline

        pipeline.query("test query")

        # Verify that pipeline was called with correct parameters
        call_args = mock_pipeline.run.call_args[0][0]
        assert "embedder" in call_args
        assert call_args["embedder"]["text"] == "test query"
        assert "prompt_builder" in call_args
        assert call_args["prompt_builder"]["question"] == "test query"
        assert "retriever" in call_args
        assert call_args["retriever"]["top_k"] == 5
