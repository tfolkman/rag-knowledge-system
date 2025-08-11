from typing import Any, Dict, List, Optional, cast

from haystack import Document, Pipeline
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from src.config import Config
from src.hierarchical_splitter import HierarchicalDocumentSplitter


class IndexingPipeline:
    """Indexing pipeline for RAG system using Haystack and Qdrant."""

    # Supported MIME types for document processing
    SUPPORTED_MIME_TYPES = [
        "text/plain",
        "application/vnd.google-apps.document",  # Google Docs
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # Word docs
    ]

    def __init__(self, config: Config):
        """Initialize the indexing pipeline.

        Args:
            config: Configuration object
        """
        self.config = config
        self.document_store: Optional[QdrantDocumentStore] = None
        self.pipeline: Optional[Pipeline] = None
        self.use_hierarchical = False  # Flag to enable hierarchical processing

    def setup_document_store(self) -> None:
        """Setup Qdrant document store."""
        self.document_store = QdrantDocumentStore(
            url=self.config.qdrant_url,
            index=self.config.qdrant_collection_name,
            embedding_dim=1024,  # mxbai-embed-large dimension
            wait_result_from_api=True,
            recreate_index=False,
        )

    def create_indexing_pipeline(self) -> None:
        """Create the indexing pipeline with all components."""
        if not self.document_store:
            raise ValueError("Document store not initialized. Call setup_document_store() first.")

        # Initialize components
        embedder = OllamaDocumentEmbedder(
            model=self.config.ollama_embedding_model, url=self.config.ollama_base_url
        )

        splitter = DocumentSplitter(
            split_by="word",  # Split by word count
            split_length=self.config.chunk_size,
            split_overlap=self.config.chunk_overlap,
        )

        writer = DocumentWriter(document_store=self.document_store)

        # Create pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component("splitter", splitter)
        self.pipeline.add_component("embedder", embedder)
        self.pipeline.add_component("writer", writer)

        # Connect components
        self.pipeline.connect("splitter", "embedder")
        self.pipeline.connect("embedder", "writer")

    def convert_documents(self, raw_documents: List[Dict[str, Any]]) -> List[Document]:
        """Convert raw documents to Haystack Document format.

        Args:
            raw_documents: List of raw documents with content and metadata

        Returns:
            List of Haystack Document objects
        """
        haystack_documents = []

        for raw_doc in raw_documents:
            # Process text-based documents
            mime_type = raw_doc["metadata"].get("mimeType", "")

            # Skip non-text documents
            if mime_type and mime_type not in self.SUPPORTED_MIME_TYPES:
                continue

            # Convert bytes to string if needed
            content = raw_doc["content"]
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")

            # Create Haystack Document
            doc = Document(content=content, meta=raw_doc["metadata"])
            haystack_documents.append(doc)

        return haystack_documents

    def process_documents(self, raw_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process documents through the indexing pipeline.

        Args:
            raw_documents: List of raw documents to process

        Returns:
            Processing results
        """
        if not self.pipeline:
            raise ValueError("Pipeline not initialized. Call create_indexing_pipeline() first.")

        # Convert to Haystack format
        documents = self.convert_documents(raw_documents)

        if not documents:
            return {
                "documents_processed": 0,
                "documents_written": 0,
                "message": "No text documents to process",
            }

        # Run pipeline
        result = self.pipeline.run({"splitter": {"documents": documents}})

        return {
            "documents_processed": len(documents),
            "documents_written": result.get("writer", {}).get("documents_written", 0),
            "result": result,
        }

    def process_documents_hierarchical(self, documents: List[Document]) -> Dict[str, Any]:
        """Process documents with hierarchical splitting.

        Args:
            documents: List of Haystack Document objects with hierarchical metadata

        Returns:
            Processing results
        """
        if not self.document_store:
            raise ValueError("Document store not initialized. Call setup_document_store() first.")

        # Use hierarchical splitter
        hierarchical_splitter = HierarchicalDocumentSplitter(
            parent_chunk_size=2000,
            child_chunk_size=500,
            grandchild_chunk_size=150,
            chunk_overlap=50,
        )

        # Split documents hierarchically
        chunks = hierarchical_splitter.split_documents(documents)

        # Embed chunks
        embedder = OllamaDocumentEmbedder(
            model=self.config.ollama_embedding_model, url=self.config.ollama_base_url
        )

        # Embed all chunks
        embedded_result = embedder.run(chunks)
        embedded_docs = cast(List[Document], embedded_result["documents"])

        # Write to document store
        writer = DocumentWriter(document_store=self.document_store)
        result = writer.run(documents=embedded_docs)

        return {
            "documents_processed": len(documents),
            "chunks_created": len(chunks),
            "chunks_written": result.get("documents_written", 0),
            "result": result,
        }

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the document collection.

        Returns:
            Collection information
        """
        if not self.document_store:
            raise ValueError("Document store not initialized. Call setup_document_store() first.")

        return {
            "collection_name": self.config.qdrant_collection_name,
            "document_count": self.document_store.count_documents(),
            "url": self.config.qdrant_url,
        }

    def initialize(self) -> None:
        """Initialize the complete indexing pipeline."""
        self.setup_document_store()
        self.create_indexing_pipeline()

    def cleanup(self) -> None:
        """Cleanup pipeline resources."""
        self.document_store = None
        self.pipeline = None
