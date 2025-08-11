"""Tests for the HierarchicalDocumentSplitter."""

from haystack import Document

from src.hierarchical_splitter import HierarchicalDocumentSplitter


class TestHierarchicalDocumentSplitter:
    """Test suite for HierarchicalDocumentSplitter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.splitter = HierarchicalDocumentSplitter(
            parent_chunk_size=2000,
            child_chunk_size=500,
            grandchild_chunk_size=150,
            chunk_overlap=50,
        )

    def test_splitter_initialization(self):
        """Test splitter initializes with correct parameters."""
        assert self.splitter.parent_chunk_size == 2000
        assert self.splitter.child_chunk_size == 500
        assert self.splitter.grandchild_chunk_size == 150
        assert self.splitter.chunk_overlap == 50

    def test_single_level_split(self):
        """Test splitting a document into single level chunks."""
        # Create a document with ~1000 words
        content = " ".join(["word" + str(i) for i in range(1000)])
        doc = Document(
            content=content,
            meta={"file_name": "test.txt", "category": "Test"},
        )

        # Split into child chunks only (500 words each)
        chunks = self.splitter.split_documents([doc], levels=["child"])

        # Should create 2 chunks with overlap
        assert len(chunks) >= 2

        # Check chunk metadata
        for chunk in chunks:
            assert chunk.meta["chunk_level"] == "child"
            assert chunk.meta["parent_id"] is not None
            assert chunk.meta["file_name"] == "test.txt"
            assert chunk.meta["category"] == "Test"

    def test_multi_level_split(self):
        """Test splitting a document into multiple hierarchy levels."""
        # Create a large document with ~3000 words
        content = " ".join(["word" + str(i) for i in range(3000)])
        doc = Document(
            content=content,
            meta={"file_name": "large.txt", "category": "Test"},
        )

        # Split into all three levels
        chunks = self.splitter.split_documents([doc])

        # Separate chunks by level
        parent_chunks = [c for c in chunks if c.meta["chunk_level"] == "parent"]
        child_chunks = [c for c in chunks if c.meta["chunk_level"] == "child"]
        grandchild_chunks = [c for c in chunks if c.meta["chunk_level"] == "grandchild"]

        # Verify we have chunks at each level
        assert len(parent_chunks) >= 1  # At least 1 parent chunk (2000 words)
        assert len(child_chunks) >= 4  # At least 4 child chunks (500 words each)
        assert len(grandchild_chunks) >= 10  # At least 10 grandchild chunks (150 words)

        # Verify parent-child relationships
        for child in child_chunks:
            assert child.meta["parent_id"] is not None
            # Parent ID should match one of the parent chunks
            parent_ids = [p.meta["chunk_id"] for p in parent_chunks]
            assert child.meta["parent_id"] in parent_ids

    def test_parent_child_relationships(self):
        """Test that parent-child relationships are correctly established."""
        content = " ".join(["word" + str(i) for i in range(1500)])
        doc = Document(content=content, meta={"file_name": "test.txt"})

        chunks = self.splitter.split_documents([doc])

        # Get parent and child chunks
        parents = [c for c in chunks if c.meta["chunk_level"] == "parent"]
        children = [c for c in chunks if c.meta["chunk_level"] == "child"]

        # Each child should have a valid parent
        for child in children:
            parent_id = child.meta["parent_id"]
            parent_exists = any(p.meta["chunk_id"] == parent_id for p in parents)
            assert parent_exists, f"Child chunk has invalid parent_id: {parent_id}"

    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        # Create content where we can verify overlap
        words = [f"word{i:04d}" for i in range(600)]  # Unique words
        content = " ".join(words)
        doc = Document(content=content)

        chunks = self.splitter.split_documents([doc], levels=["child"])

        # With 600 words and 500-word chunks with 50-word overlap
        # First chunk: words 0-499
        # Second chunk: words 450-599 (50 word overlap)
        assert len(chunks) == 2

        # Verify overlap exists
        first_chunk_words = chunks[0].content.split()
        second_chunk_words = chunks[1].content.split()

        # Last 50 words of first chunk should overlap with first 50 of second
        overlap_from_first = set(first_chunk_words[-50:])
        overlap_from_second = set(second_chunk_words[:50])

        # Should have common words
        assert len(overlap_from_first & overlap_from_second) > 0

    def test_metadata_preservation(self):
        """Test that document metadata is preserved in chunks."""
        doc = Document(
            content=" ".join(["word"] * 1000),
            meta={
                "file_name": "test.txt",
                "category": "Health",
                "subcategory": "Nutrition",
                "author": "Test Author",
                "created_date": "2024-01-01",
            },
        )

        chunks = self.splitter.split_documents([doc])

        # All chunks should preserve original metadata
        for chunk in chunks:
            assert chunk.meta["file_name"] == "test.txt"
            assert chunk.meta["category"] == "Health"
            assert chunk.meta["subcategory"] == "Nutrition"
            assert chunk.meta["author"] == "Test Author"
            assert chunk.meta["created_date"] == "2024-01-01"
            # Plus new chunk-specific metadata
            assert "chunk_id" in chunk.meta
            assert "chunk_level" in chunk.meta
            assert "chunk_index" in chunk.meta

    def test_empty_document_handling(self):
        """Test handling of empty documents."""
        doc = Document(content="", meta={"file_name": "empty.txt"})

        chunks = self.splitter.split_documents([doc])

        # Should return empty list for empty document
        assert len(chunks) == 0

    def test_small_document_handling(self):
        """Test handling of documents smaller than chunk size."""
        # Document with only 50 words (smaller than grandchild size)
        content = " ".join(["word"] * 50)
        doc = Document(content=content, meta={"file_name": "small.txt"})

        chunks = self.splitter.split_documents([doc])

        # Should create one chunk at the smallest level
        assert len(chunks) == 1
        assert chunks[0].meta["chunk_level"] == "grandchild"
        assert chunks[0].content == content

    def test_multiple_documents(self):
        """Test splitting multiple documents at once."""
        docs = [
            Document(
                content=" ".join(["doc1word"] * 600),
                meta={"file_name": "doc1.txt", "category": "Cat1"},
            ),
            Document(
                content=" ".join(["doc2word"] * 800),
                meta={"file_name": "doc2.txt", "category": "Cat2"},
            ),
        ]

        chunks = self.splitter.split_documents(docs, levels=["child"])

        # Get chunks for each document
        doc1_chunks = [c for c in chunks if c.meta["file_name"] == "doc1.txt"]
        doc2_chunks = [c for c in chunks if c.meta["file_name"] == "doc2.txt"]

        # Each document should have appropriate number of chunks
        assert len(doc1_chunks) >= 1
        assert len(doc2_chunks) >= 1

        # Categories should be preserved
        assert all(c.meta["category"] == "Cat1" for c in doc1_chunks)
        assert all(c.meta["category"] == "Cat2" for c in doc2_chunks)

    def test_chunk_id_uniqueness(self):
        """Test that chunk IDs are unique."""
        doc = Document(content=" ".join(["word"] * 3000))

        chunks = self.splitter.split_documents([doc])

        # Extract all chunk IDs
        chunk_ids = [c.meta["chunk_id"] for c in chunks]

        # All IDs should be unique
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_auto_merging_preparation(self):
        """Test that chunks are prepared for auto-merging retrieval."""
        doc = Document(
            content=" ".join(["word"] * 2000),
            meta={"file_name": "test.txt"},
        )

        chunks = self.splitter.split_documents([doc])

        # All chunks should have auto-merging metadata
        for chunk in chunks:
            assert "chunk_id" in chunk.meta
            assert "chunk_level" in chunk.meta
            assert "parent_id" in chunk.meta or chunk.meta["chunk_level"] == "parent"
            assert "chunk_index" in chunk.meta
            assert "total_chunks" in chunk.meta

    def test_custom_chunk_sizes(self):
        """Test using custom chunk sizes."""
        custom_splitter = HierarchicalDocumentSplitter(
            parent_chunk_size=1000,
            child_chunk_size=250,
            grandchild_chunk_size=100,
            chunk_overlap=25,
        )

        doc = Document(content=" ".join(["word"] * 1200))
        chunks = custom_splitter.split_documents([doc])

        # Verify chunk sizes respect custom settings
        parent_chunks = [c for c in chunks if c.meta["chunk_level"] == "parent"]
        child_chunks = [c for c in chunks if c.meta["chunk_level"] == "child"]
        grandchild_chunks = [c for c in chunks if c.meta["chunk_level"] == "grandchild"]

        # Should have appropriate number of chunks
        assert len(parent_chunks) >= 1  # At least 1 parent chunk (1000 words each)
        assert len(child_chunks) >= 4  # At least 4 child chunks (250 words each)
        assert len(grandchild_chunks) >= 10  # At least 10 grandchild chunks (100 words each)

        # First parent chunk should be close to 1000 words
        if parent_chunks:
            first_parent_words = len(parent_chunks[0].content.split())
            # With 1200 total words and 1000 chunk size, first chunk should be 1000
            assert 950 <= first_parent_words <= 1000

    def test_word_boundary_splitting(self):
        """Test that splitting happens at word boundaries."""
        # Create content with clear word boundaries
        content = "This is a test. " * 200  # 800 words total
        doc = Document(content=content)

        chunks = self.splitter.split_documents([doc], levels=["child"])

        # All chunks should end at word boundaries (no partial words)
        for chunk in chunks:
            # Check that content doesn't end mid-word
            assert not chunk.content.endswith(" i")  # Partial "is"
            assert not chunk.content.endswith(" tes")  # Partial "test"
            # Content should end with complete word or punctuation
            assert chunk.content[-1] in ". " or chunk.content.endswith("test")
