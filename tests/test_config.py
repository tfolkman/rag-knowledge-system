import os
from unittest.mock import patch

import pytest

from src.config import Config


class TestConfig:
    """Test configuration module for RAG system."""

    def test_config_loads_from_environment(self):
        """Test that configuration loads values from environment variables."""
        # Reset singleton instance for this test
        Config._instance = None

        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": "test_credentials.json",
                "QDRANT_URL": "http://test:6333",
                "QDRANT_COLLECTION_NAME": "test_collection",
                "OLLAMA_BASE_URL": "http://test:11434",
                "OLLAMA_MODEL_NAME": "test_model",
                "OLLAMA_EMBEDDING_MODEL": "test_embedding",
                "LOG_LEVEL": "DEBUG",
                "MAX_DOCUMENTS_PER_BATCH": "5",
                "CHUNK_SIZE": "1000",
                "CHUNK_OVERLAP": "100",
            },
        ):
            config = Config()

            assert config.google_credentials_path == "test_credentials.json"
            assert config.qdrant_url == "http://test:6333"
            assert config.qdrant_collection_name == "test_collection"
            assert config.ollama_base_url == "http://test:11434"
            assert config.ollama_model_name == "test_model"
            assert config.ollama_embedding_model == "test_embedding"
            assert config.log_level == "DEBUG"
            assert config.max_documents_per_batch == 5
            assert config.chunk_size == 1000
            assert config.chunk_overlap == 100

    def test_config_has_default_values(self):
        """Test that configuration has sensible default values."""
        # Reset singleton instance for this test
        Config._instance = None

        with patch.dict(os.environ, {}, clear=True):
            config = Config()

            assert config.qdrant_url == "http://localhost:6333"
            assert config.qdrant_collection_name == "documents"
            assert config.ollama_base_url == "http://localhost:11434"
            assert config.ollama_model_name == "llama3.2:latest"
            assert config.ollama_embedding_model == "mxbai-embed-large"
            assert config.log_level == "INFO"
            assert config.max_documents_per_batch == 10
            assert config.chunk_size == 500
            assert config.chunk_overlap == 50

    def test_config_validates_required_fields(self):
        """Test that configuration validates required fields."""
        # Reset singleton instance for this test
        Config._instance = None

        with patch.dict(os.environ, {}, clear=True):
            config = Config()

            # Should raise exception if Google credentials path is not set
            with pytest.raises(ValueError, match="Google credentials file not found"):
                config.validate()

    def test_config_validates_google_credentials_file_exists(self):
        """Test that configuration validates Google credentials file exists."""
        # Reset singleton instance for this test
        Config._instance = None

        with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "nonexistent.json"}):
            config = Config()

            with pytest.raises(ValueError, match="Google credentials file not found"):
                config.validate()

    def test_config_singleton_pattern(self):
        """Test that Config implements singleton pattern."""
        # Reset singleton instance for this test
        Config._instance = None

        config1 = Config()
        config2 = Config()

        assert config1 is config2

    def test_config_to_dict(self):
        """Test that configuration can be converted to dictionary."""
        # Reset singleton instance for this test
        Config._instance = None

        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": "test.json",
                "QDRANT_URL": "http://test:6333",
            },
        ):
            config = Config()
            config_dict = config.to_dict()

            assert isinstance(config_dict, dict)
            assert "qdrant_url" in config_dict
            assert "ollama_base_url" in config_dict
            assert config_dict["qdrant_url"] == "http://test:6333"
