"""Hierarchical Document Splitter for multi-level chunking."""

import hashlib
import logging
from typing import Dict, List, Optional

from haystack import Document

logger = logging.getLogger(__name__)


class HierarchicalDocumentSplitter:
    """
    Split documents into hierarchical chunks for auto-merging retrieval.

    Creates a tree structure of chunks where:
    - Parent chunks are large (e.g., 2000 words)
    - Child chunks are medium (e.g., 500 words)
    - Grandchild chunks are small (e.g., 150 words)
    """

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 500,
        grandchild_chunk_size: int = 150,
        chunk_overlap: int = 50,
        split_by: str = "word",
    ):
        """
        Initialize the hierarchical splitter.

        Args:
            parent_chunk_size: Size of parent chunks in words/tokens
            child_chunk_size: Size of child chunks in words/tokens
            grandchild_chunk_size: Size of grandchild chunks in words/tokens
            chunk_overlap: Number of words/tokens to overlap between chunks
            split_by: Unit to split by ("word" or "token")
        """
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.grandchild_chunk_size = grandchild_chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_by = split_by

        # Validate chunk sizes
        if not (parent_chunk_size > child_chunk_size > grandchild_chunk_size):
            raise ValueError("Chunk sizes must follow: parent > child > grandchild")

    def split_documents(
        self,
        documents: List[Document],
        levels: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Split documents into hierarchical chunks.

        Args:
            documents: List of documents to split
            levels: Which levels to create (["parent", "child", "grandchild"])
                   If None, creates all levels

        Returns:
            List of chunked documents with hierarchy metadata
        """
        if levels is None:
            levels = ["parent", "child", "grandchild"]

        all_chunks = []

        for doc_idx, doc in enumerate(documents):
            if not doc.content:
                logger.warning(f"Skipping empty document: {doc.meta.get('file_name', 'unknown')}")
                continue

            # Split content into words
            words = doc.content.split() if self.split_by == "word" else doc.content.split()

            if not words:
                continue

            # Generate document ID for tracking
            doc_id = self._generate_doc_id(doc, doc_idx)

            # Create chunks at each level
            doc_chunks = []
            parent_chunks = []
            child_chunks = []

            # Determine if document is small
            doc_word_count = len(words)

            # If document is smaller than smallest chunk size, create single chunk
            if doc_word_count <= self.grandchild_chunk_size:
                chunk = self._create_single_chunk(
                    words=words,
                    level="grandchild",
                    doc_id=doc_id,
                    original_meta=doc.meta,
                )
                doc_chunks.append(chunk)
            else:
                # Create parent chunks
                if "parent" in levels:
                    parent_chunks = self._create_chunks(
                        words=words,
                        chunk_size=self.parent_chunk_size,
                        overlap=self.chunk_overlap,
                        level="parent",
                        doc_id=doc_id,
                        original_meta=doc.meta,
                    )
                    doc_chunks.extend(parent_chunks)

                # Create child chunks
                if "child" in levels:
                    child_chunks = self._create_chunks(
                        words=words,
                        chunk_size=self.child_chunk_size,
                        overlap=self.chunk_overlap,
                        level="child",
                        doc_id=doc_id,
                        original_meta=doc.meta,
                    )

                    # Assign parent IDs to child chunks
                    if "parent" in levels and parent_chunks:
                        self._assign_parent_ids(child_chunks, parent_chunks)
                    else:
                        # If no parent level, set parent_id to a virtual parent
                        for chunk in child_chunks:
                            chunk.meta["parent_id"] = f"{doc_id}_virtual_parent"

                    doc_chunks.extend(child_chunks)

                # Create grandchild chunks
                if "grandchild" in levels:
                    grandchild_chunks = self._create_chunks(
                        words=words,
                        chunk_size=self.grandchild_chunk_size,
                        overlap=self.chunk_overlap,
                        level="grandchild",
                        doc_id=doc_id,
                        original_meta=doc.meta,
                    )

                    # Assign parent IDs to grandchild chunks
                    if "child" in levels and child_chunks:
                        self._assign_parent_ids(grandchild_chunks, child_chunks)
                    elif "parent" in levels and parent_chunks:
                        self._assign_parent_ids(grandchild_chunks, parent_chunks)
                    else:
                        # If no parent levels, set parent_id to a virtual parent
                        for chunk in grandchild_chunks:
                            chunk.meta["parent_id"] = f"{doc_id}_virtual_parent"

                    doc_chunks.extend(grandchild_chunks)

            all_chunks.extend(doc_chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    def _create_chunks(
        self,
        words: List[str],
        chunk_size: int,
        overlap: int,
        level: str,
        doc_id: str,
        original_meta: Dict,
    ) -> List[Document]:
        """
        Create chunks of a specific size from words.

        Args:
            words: List of words to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            level: Hierarchy level (parent/child/grandchild)
            doc_id: Document ID
            original_meta: Original document metadata

        Returns:
            List of chunk documents
        """
        chunks = []
        total_words = len(words)

        # Calculate step size (chunk_size - overlap)
        step = max(1, chunk_size - overlap)

        # Create chunks
        chunk_index = 0
        start = 0

        while start < total_words:
            end = min(start + chunk_size, total_words)

            # Extract chunk words
            chunk_words = words[start:end]
            chunk_content = " ".join(chunk_words)

            # Generate chunk ID
            chunk_id = self._generate_chunk_id(doc_id, level, chunk_index)

            # Create chunk metadata
            chunk_meta = original_meta.copy()
            chunk_meta.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_level": level,
                    "chunk_index": chunk_index,
                    "chunk_start": start,
                    "chunk_end": end,
                    "total_chunks": -1,  # Will be updated later
                    "parent_id": None,  # Will be assigned later if applicable
                    "doc_id": doc_id,
                }
            )

            # Create chunk document
            chunk = Document(content=chunk_content, meta=chunk_meta)
            chunks.append(chunk)

            chunk_index += 1

            # Move to next chunk position
            start += step

            # If we've reached the end, break
            if end >= total_words:
                break

        # Update total chunks count
        for chunk in chunks:
            chunk.meta["total_chunks"] = len(chunks)

        return chunks

    def _create_single_chunk(
        self,
        words: List[str],
        level: str,
        doc_id: str,
        original_meta: Dict,
    ) -> Document:
        """
        Create a single chunk for very small documents.

        Args:
            words: List of words
            level: Hierarchy level
            doc_id: Document ID
            original_meta: Original document metadata

        Returns:
            Single chunk document
        """
        chunk_content = " ".join(words)
        chunk_id = self._generate_chunk_id(doc_id, level, 0)

        chunk_meta = original_meta.copy()
        chunk_meta.update(
            {
                "chunk_id": chunk_id,
                "chunk_level": level,
                "chunk_index": 0,
                "chunk_start": 0,
                "chunk_end": len(words),
                "total_chunks": 1,
                "parent_id": None,
                "doc_id": doc_id,
            }
        )

        return Document(content=chunk_content, meta=chunk_meta)

    def _assign_parent_ids(
        self,
        child_chunks: List[Document],
        parent_chunks: List[Document],
    ):
        """
        Assign parent IDs to child chunks based on overlap.

        Args:
            child_chunks: List of child chunks
            parent_chunks: List of parent chunks
        """
        for child in child_chunks:
            child_start = child.meta["chunk_start"]
            child_end = child.meta["chunk_end"]

            # Find the parent chunk that best contains this child
            best_parent = None
            best_overlap = 0

            for parent in parent_chunks:
                parent_start = parent.meta["chunk_start"]
                parent_end = parent.meta["chunk_end"]

                # Calculate overlap
                overlap_start = max(child_start, parent_start)
                overlap_end = min(child_end, parent_end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_parent = parent

            if best_parent:
                child.meta["parent_id"] = best_parent.meta["chunk_id"]

    def _generate_doc_id(self, doc: Document, doc_idx: int) -> str:
        """
        Generate a unique document ID.

        Args:
            doc: Document
            doc_idx: Document index

        Returns:
            Document ID
        """
        # Use file name if available, otherwise use content hash
        if "file_name" in doc.meta:
            base = doc.meta["file_name"]
        else:
            # Create hash of content for uniqueness
            content = doc.content if doc.content else ""
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            base = f"doc_{content_hash}"

        return f"{base}_{doc_idx}"

    def _generate_chunk_id(self, doc_id: str, level: str, chunk_index: int) -> str:
        """
        Generate a unique chunk ID.

        Args:
            doc_id: Document ID
            level: Chunk level
            chunk_index: Chunk index

        Returns:
            Chunk ID
        """
        return f"{doc_id}_{level}_{chunk_index}"
