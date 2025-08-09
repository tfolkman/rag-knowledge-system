from typing import Any, Dict, Optional

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from src.config import Config


class QueryPipeline:
    """Query pipeline for RAG system using Haystack, Qdrant, and Ollama."""

    def __init__(self, config: Config):
        """Initialize the query pipeline.

        Args:
            config: Configuration object
        """
        self.config = config
        self.document_store: Optional[QdrantDocumentStore] = None
        self.pipeline: Optional[Pipeline] = None

    def setup_document_store(self) -> None:
        """Setup Qdrant document store."""
        self.document_store = QdrantDocumentStore(
            url=self.config.qdrant_url,
            index=self.config.qdrant_collection_name,
            embedding_dim=1024,  # mxbai-embed-large dimension
            wait_result_from_api=True,
            recreate_index=False,
        )

    def get_default_prompt_template(self) -> str:
        """Get the default prompt template for RAG.

        Returns:
            Default prompt template string
        """
        return """You are a helpful assistant. Answer the question based on the context if available, otherwise use your general knowledge.

Context from documents:
{{ context }}

Question: {{ question }}

If the context is empty or irrelevant, still provide a helpful answer based on your general knowledge, but mention that the information wasn't found in the indexed documents.

Answer:"""

    def create_query_pipeline(self) -> None:
        """Create the query pipeline with all components."""
        if not self.document_store:
            raise ValueError("Document store not initialized. Call setup_document_store() first.")

        # Initialize components
        embedder = OllamaTextEmbedder(
            model=self.config.ollama_embedding_model, url=self.config.ollama_base_url
        )

        retriever = QdrantEmbeddingRetriever(document_store=self.document_store, top_k=5)

        prompt_builder = PromptBuilder(template=self.get_default_prompt_template())

        generator = OllamaGenerator(
            model=self.config.ollama_model_name, url=self.config.ollama_base_url
        )

        # Create pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component("embedder", embedder)
        self.pipeline.add_component("retriever", retriever)
        self.pipeline.add_component("prompt_builder", prompt_builder)
        self.pipeline.add_component("llm", generator)

        # Connect components
        self.pipeline.connect("embedder.embedding", "retriever.query_embedding")
        self.pipeline.connect("retriever.documents", "prompt_builder.context")
        self.pipeline.connect("prompt_builder.prompt", "llm.prompt")

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Execute a query against the RAG system.

        Args:
            question: The question to ask
            top_k: Number of documents to retrieve

        Returns:
            Query results with answer and sources
        """
        if not self.pipeline:
            raise ValueError("Pipeline not initialized. Call create_query_pipeline() first.")

        # Format context from retrieved documents
        result = self.pipeline.run(
            {
                "embedder": {"text": question},
                "retriever": {"top_k": top_k},
                "prompt_builder": {
                    "question": question,
                    # context will be filled by retrieved documents via pipeline connection
                },
            }
        )

        # Extract results
        answer = ""
        if "llm" in result and "replies" in result["llm"] and result["llm"]["replies"]:
            answer = result["llm"]["replies"][0]

        sources = []
        if "retriever" in result and "documents" in result["retriever"]:
            for doc in result["retriever"]["documents"]:
                sources.append({"content": doc.content, "metadata": doc.meta})

        return {
            "query": question,
            "answer": answer,
            "sources": sources,
            "raw_result": result,
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
        """Initialize the complete query pipeline."""
        self.setup_document_store()
        self.create_query_pipeline()

    def cleanup(self) -> None:
        """Cleanup pipeline resources."""
        self.document_store = None
        self.pipeline = None
