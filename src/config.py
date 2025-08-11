import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for RAG system using singleton pattern."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Google Drive API Configuration
        self.google_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.CREDENTIALS_PATH = self.google_credentials_path  # Alias for main.py compatibility
        self.GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")  # New field for main.py

        # Qdrant Configuration
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.QDRANT_URL = self.qdrant_url  # Alias for main.py compatibility
        self.qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME", "documents")
        self.COLLECTION_NAME = self.qdrant_collection_name  # Alias for main.py compatibility

        # Ollama Configuration
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.OLLAMA_URL = self.ollama_base_url  # Alias for main.py compatibility
        self.ollama_model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3.2:latest")
        self.CHAT_MODEL = self.ollama_model_name  # Alias for main.py compatibility
        self.ollama_embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
        self.EMBEDDING_MODEL = self.ollama_embedding_model  # Alias for main.py compatibility

        # Application Configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.max_documents_per_batch = int(os.getenv("MAX_DOCUMENTS_PER_BATCH", "10"))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        self.CHUNK_SIZE = self.chunk_size  # Alias for main.py compatibility
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        self.CHUNK_OVERLAP = self.chunk_overlap  # Alias for main.py compatibility

        self._initialized = True

    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.google_credentials_path:
            raise ValueError(
                "Google credentials file not found: GOOGLE_APPLICATION_CREDENTIALS not set"
            )

        credentials_path = Path(self.google_credentials_path)
        if not credentials_path.exists():
            raise ValueError(f"Google credentials file not found: {self.google_credentials_path}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "google_credentials_path": self.google_credentials_path,
            "google_drive_folder_id": self.GOOGLE_DRIVE_FOLDER_ID,
            "qdrant_url": self.qdrant_url,
            "qdrant_collection_name": self.qdrant_collection_name,
            "ollama_base_url": self.ollama_base_url,
            "ollama_model_name": self.ollama_model_name,
            "ollama_embedding_model": self.ollama_embedding_model,
            "log_level": self.log_level,
            "max_documents_per_batch": self.max_documents_per_batch,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
