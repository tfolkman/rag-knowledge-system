"""Tests for the HierarchicalDocumentLoader."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.hierarchical_loader import HierarchicalDocumentLoader


class TestHierarchicalDocumentLoader:
    """Test suite for HierarchicalDocumentLoader."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = HierarchicalDocumentLoader()

    def test_loader_initialization(self):
        """Test loader initializes correctly."""
        assert self.loader is not None
        assert hasattr(self.loader, "load_from_directory")
        assert hasattr(self.loader, "load_from_google_drive")

    def test_folder_traversal_local(self):
        """Test recursive folder traversal for local files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test directory structure
            root = Path(tmpdir)

            # Create Health folder with subfolders
            health_dir = root / "Health"
            health_dir.mkdir()
            (health_dir / "Nutrition").mkdir()
            (health_dir / "Exercise").mkdir()

            # Create Content Creation folder
            content_dir = root / "Content Creation"
            content_dir.mkdir()
            (content_dir / "Video").mkdir()

            # Create test files
            (health_dir / "health_overview.txt").write_text("Health content")
            (health_dir / "Nutrition" / "diet_plan.txt").write_text("Diet information")
            (health_dir / "Exercise" / "workout.txt").write_text("Workout routines")
            (content_dir / "Video" / "editing_tips.txt").write_text("Video editing tips")

            # Load documents
            documents = self.loader.load_from_directory(root)

            # Verify documents were loaded
            assert len(documents) == 4

            # Verify metadata
            doc_names = [doc.meta["file_name"] for doc in documents]
            assert "health_overview.txt" in doc_names
            assert "diet_plan.txt" in doc_names
            assert "workout.txt" in doc_names
            assert "editing_tips.txt" in doc_names

    def test_metadata_extraction(self):
        """Test that proper metadata is extracted from folder structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create nested structure
            health_dir = root / "Health"
            nutrition_dir = health_dir / "Nutrition"
            nutrition_dir.mkdir(parents=True)

            # Create a test file
            test_file = nutrition_dir / "vitamins.txt"
            test_file.write_text("Vitamin information")

            # Load documents
            documents = self.loader.load_from_directory(root)

            assert len(documents) == 1
            doc = documents[0]

            # Check metadata
            assert doc.meta["category"] == "Health"
            assert doc.meta["subcategory"] == "Nutrition"
            assert doc.meta["file_name"] == "vitamins.txt"
            assert doc.meta["hierarchy_level"] == 2
            assert "Health/Nutrition" in doc.meta["hierarchy_path"]
            assert doc.content == "Vitamin information"

    def test_category_assignment(self):
        """Test that categories are correctly assigned based on folder names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create multiple top-level categories
            categories = ["Health", "Finance", "Technology", "Personal"]
            for category in categories:
                cat_dir = root / category
                cat_dir.mkdir()
                (cat_dir / f"{category.lower()}_doc.txt").write_text(f"{category} content")

            # Load documents
            documents = self.loader.load_from_directory(root)

            # Verify categories
            doc_categories = {doc.meta["category"] for doc in documents}
            assert doc_categories == set(categories)

    def test_empty_folders_handled(self):
        """Test that empty folders are handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create empty folders
            (root / "EmptyFolder").mkdir()
            (root / "AnotherEmpty" / "Nested").mkdir(parents=True)

            # Create one folder with content
            content_dir = root / "WithContent"
            content_dir.mkdir()
            (content_dir / "file.txt").write_text("Some content")

            # Load documents
            documents = self.loader.load_from_directory(root)

            # Should only load the one file
            assert len(documents) == 1
            assert documents[0].meta["category"] == "WithContent"

    def test_deep_nesting_levels(self):
        """Test handling of deeply nested folder structures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create deeply nested structure
            deep_path = root / "Level1" / "Level2" / "Level3" / "Level4" / "Level5"
            deep_path.mkdir(parents=True)

            # Create file at deep level
            deep_file = deep_path / "deep_file.txt"
            deep_file.write_text("Deep content")

            # Load documents
            documents = self.loader.load_from_directory(root)

            assert len(documents) == 1
            doc = documents[0]

            # Check hierarchy metadata
            assert doc.meta["category"] == "Level1"
            assert doc.meta["subcategory"] == "Level2"
            assert doc.meta["hierarchy_level"] == 5
            assert "Level1/Level2/Level3/Level4/Level5" in doc.meta["hierarchy_path"]

    def test_file_type_filtering(self):
        """Test that only supported file types are loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create files of different types
            (root / "text_file.txt").write_text("Text content")
            (root / "markdown_file.md").write_text("# Markdown content")
            (root / "pdf_file.pdf").write_bytes(b"PDF content")  # Mock PDF
            (root / "image_file.jpg").write_bytes(b"Image data")  # Should be ignored
            (root / "binary_file.exe").write_bytes(b"Binary data")  # Should be ignored

            # Load documents with file type filtering
            documents = self.loader.load_from_directory(
                root, allowed_extensions=[".txt", ".md", ".pdf"]
            )

            # Should only load supported types
            loaded_files = {doc.meta["file_name"] for doc in documents}
            assert "text_file.txt" in loaded_files
            assert "markdown_file.md" in loaded_files
            assert "pdf_file.pdf" in loaded_files
            assert "image_file.jpg" not in loaded_files
            assert "binary_file.exe" not in loaded_files

    def test_google_drive_integration(self):
        """Test loading hierarchical structure from Google Drive."""
        with patch("src.hierarchical_loader.GoogleDriveLoader") as mock_gdrive:
            # Mock Google Drive loader
            mock_loader = MagicMock()
            mock_gdrive.return_value = mock_loader
            mock_loader.authenticate.return_value = None

            # Mock the _get_all_folders_recursive method
            with patch.object(self.loader, "_get_all_folders_recursive") as mock_get_folders:
                mock_get_folders.return_value = [
                    {"id": "folder1", "name": "Health", "parents": ["root_folder_id"]},
                    {"id": "folder2", "name": "Nutrition", "parents": ["folder1"]},
                ]

                # Mock load_documents to return different results per folder
                # First call for root_folder_id, second for folder1, third for folder2
                mock_loader.load_documents.side_effect = [
                    [],  # root folder - no documents
                    [],  # folder1 (Health) - no documents
                    [  # folder2 (Nutrition) - has the document
                        {
                            "content": "Diet content",
                            "metadata": {
                                "name": "diet.txt",
                                "id": "doc1",
                                "mimeType": "text/plain",
                            },
                        }
                    ],
                ]

                # Load from Google Drive
                config = Mock()
                documents = self.loader.load_from_google_drive(config, "root_folder_id")

                assert len(documents) == 1
                doc = documents[0]
                # The document is in folder2 which should map to Health/Nutrition
                assert doc.meta["source"] == "google_drive"

    def test_metadata_preservation(self):
        """Test that existing metadata is preserved when adding hierarchical metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create a file
            (root / "test.txt").write_text("Test content")

            # Load with additional metadata
            documents = self.loader.load_from_directory(
                root, additional_metadata={"author": "Test Author", "version": "1.0"}
            )

            assert len(documents) == 1
            doc = documents[0]

            # Check both hierarchical and additional metadata
            assert doc.meta["author"] == "Test Author"
            assert doc.meta["version"] == "1.0"
            assert "file_name" in doc.meta
            assert "hierarchy_path" in doc.meta

    def test_error_handling_invalid_path(self):
        """Test error handling for invalid directory paths."""
        with pytest.raises(ValueError, match="Directory does not exist"):
            self.loader.load_from_directory(Path("/nonexistent/path"))

    def test_error_handling_permission_denied(self):
        """Test error handling for permission denied scenarios."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            restricted_dir = root / "restricted"
            restricted_dir.mkdir()

            # Create a file and then restrict permissions
            (restricted_dir / "file.txt").write_text("Content")

            # Mock permission error
            with patch("pathlib.Path.iterdir", side_effect=PermissionError("Access denied")):
                with pytest.raises(PermissionError, match="Access denied"):
                    self.loader.load_from_directory(root)

    def test_large_file_handling(self):
        """Test handling of large files with size limits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create a large file (mock)
            large_file = root / "large.txt"
            large_file.write_text("x" * 10_000_000)  # 10MB of text

            # Create a normal file
            normal_file = root / "normal.txt"
            normal_file.write_text("Normal content")

            # Load with size limit
            documents = self.loader.load_from_directory(root, max_file_size_mb=5)  # 5MB limit

            # Should only load the normal file
            assert len(documents) == 1
            assert documents[0].meta["file_name"] == "normal.txt"

    def test_duplicate_file_names_different_folders(self):
        """Test handling of duplicate file names in different folders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create duplicate file names in different folders
            (root / "Folder1").mkdir()
            (root / "Folder2").mkdir()

            (root / "Folder1" / "readme.txt").write_text("Folder1 readme")
            (root / "Folder2" / "readme.txt").write_text("Folder2 readme")

            # Load documents
            documents = self.loader.load_from_directory(root)

            assert len(documents) == 2

            # Both should be loaded with distinct metadata
            categories = {doc.meta["category"] for doc in documents}
            assert categories == {"Folder1", "Folder2"}

            # Each should have unique content
            contents = {doc.content for doc in documents}
            assert len(contents) == 2
