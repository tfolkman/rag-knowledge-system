import logging
from typing import Any, Dict, Optional

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from src.config import Config

logger = logging.getLogger(__name__)


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
        # Store components for direct access
        self.embedder: Optional[OllamaTextEmbedder] = None
        self.retriever: Optional[QdrantEmbeddingRetriever] = None
        self.prompt_builder: Optional[PromptBuilder] = None
        self.generator: Optional[OllamaGenerator] = None

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

        # Initialize and store components for direct access
        self.embedder = OllamaTextEmbedder(
            model=self.config.ollama_embedding_model, url=self.config.ollama_base_url
        )

        self.retriever = QdrantEmbeddingRetriever(document_store=self.document_store, top_k=5)

        self.prompt_builder = PromptBuilder(template=self.get_default_prompt_template())

        self.generator = OllamaGenerator(
            model=self.config.ollama_model_name, url=self.config.ollama_base_url
        )

        # Note: We're using direct component calls instead of pipeline execution
        # This gives us better control over the data flow and error handling
        self.pipeline = None  # Not using Haystack pipeline execution anymore

        logger.info("Query pipeline components initialized successfully")

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Execute a query against the RAG system.

        Args:
            question: The question to ask
            top_k: Number of documents to retrieve

        Returns:
            Query results with answer and sources
        """
        if not self.embedder or not self.retriever or not self.prompt_builder or not self.generator:
            raise ValueError("Components not initialized. Call create_query_pipeline() first.")

        try:
            logger.debug(f"Processing query: {question[:100]}...")  # Log first 100 chars

            # Step 1: Generate embedding for the question
            logger.debug("Generating embedding for query")
            embedding_result = self.embedder.run(text=question)
            query_embedding = embedding_result["embedding"]

            # Step 2: Retrieve relevant documents
            # Ensure query_embedding is a list[float]
            if isinstance(query_embedding, dict):
                query_embedding = list(query_embedding.values())  # type: ignore

            logger.debug(f"Retrieving top {top_k} documents")
            retrieval_result = self.retriever.run(query_embedding=query_embedding, top_k=top_k)
            documents = retrieval_result.get("documents", [])
            logger.info(f"Retrieved {len(documents)} documents")

            # Step 3: Format retrieved documents into context string
            context_parts = []
            sources = []
            for i, doc in enumerate(documents, 1):
                # Format each document chunk
                doc_text = f"[Document {i}]\n{doc.content}\n"
                if doc.meta:
                    doc_text += f"Source: {doc.meta.get('name', 'Unknown')}\n"
                context_parts.append(doc_text)
                sources.append({"content": doc.content, "metadata": doc.meta})

            # Join all document chunks into context
            context = (
                "\n---\n".join(context_parts) if context_parts else "No relevant documents found."
            )
            logger.debug(f"Formatted context with {len(context_parts)} document parts")

            # Step 4: Build the prompt
            logger.debug("Building prompt with context")
            prompt_result = self.prompt_builder.run(question=question, context=context)
            prompt = prompt_result["prompt"]

            # Step 5: Generate the answer
            logger.debug("Generating answer with LLM")
            generation_result = self.generator.run(prompt=prompt)
            answer = (
                generation_result.get("replies", [""])[0]
                if generation_result.get("replies")
                else ""
            )

            logger.info(f"Successfully generated answer for query (length: {len(answer)} chars)")

            return {
                "query": question,
                "answer": answer,
                "sources": sources,
                "raw_result": {
                    "embedder": embedding_result,
                    "retriever": retrieval_result,
                    "prompt_builder": prompt_result,
                    "generator": generation_result,
                },
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise

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
        logger.debug("Cleaning up query pipeline resources")
        self.document_store = None
        self.pipeline = None
        self.embedder = None
        self.retriever = None
        self.prompt_builder = None
        self.generator = None
