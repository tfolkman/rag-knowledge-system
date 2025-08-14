"""Tests for GitHub repository loader."""

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from haystack import Document

from src.config import Config
from src.github_loader import GitHubRepositoryLoader


class TestGitHubRepositoryLoader:
    """Test GitHub repository loader functionality."""

    def test_loader_initialization(self):
        """Test that the loader initializes with correct defaults."""
        config = Mock(spec=Config)
        loader = GitHubRepositoryLoader(config)

        assert loader.config == config
        assert loader.local_repos_dir == Path.home() / "Coding"
        assert loader.max_file_size_mb == 10.0
        assert loader.allowed_extensions == [".md"]
        assert loader.hierarchical_loader is not None

    def test_parse_repo_name_valid(self):
        """Test parsing valid repository identifiers."""
        config = Mock(spec=Config)
        loader = GitHubRepositoryLoader(config)

        owner, repo = loader.parse_repo_name("owner/repo")
        assert owner == "owner"
        assert repo == "repo"

    def test_parse_repo_name_invalid(self):
        """Test parsing invalid repository identifiers."""
        config = Mock(spec=Config)
        loader = GitHubRepositoryLoader(config)

        with pytest.raises(ValueError) as exc_info:
            loader.parse_repo_name("invalid-format")
        assert "owner/repo" in str(exc_info.value)

    def test_get_local_repo_path_default(self):
        """Test getting local repository path with default directory."""
        config = Mock(spec=Config)
        loader = GitHubRepositoryLoader(config)

        path = loader.get_local_repo_path("owner/repo")
        assert path == Path.home() / "Coding" / "repo"

    def test_get_local_repo_path_custom(self):
        """Test getting local repository path with custom directory."""
        config = Mock(spec=Config)
        loader = GitHubRepositoryLoader(config)

        custom_dir = Path("/custom/dir")
        path = loader.get_local_repo_path("owner/repo", custom_dir)
        assert path == custom_dir / "repo"

    def test_check_local_repo_exists(self, tmp_path):
        """Test checking if a valid git repository exists."""
        config = Mock(spec=Config)
        loader = GitHubRepositoryLoader(config)

        # Create a mock git repository
        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()

        assert loader.check_local_repo(repo_path) is True

    def test_check_local_repo_not_exists(self, tmp_path):
        """Test checking if repository doesn't exist."""
        config = Mock(spec=Config)
        loader = GitHubRepositoryLoader(config)

        repo_path = tmp_path / "non-existent"
        assert loader.check_local_repo(repo_path) is False

    def test_check_local_repo_not_git(self, tmp_path):
        """Test checking directory that's not a git repository."""
        config = Mock(spec=Config)
        loader = GitHubRepositoryLoader(config)

        # Create directory without .git
        repo_path = tmp_path / "not-git"
        repo_path.mkdir()

        assert loader.check_local_repo(repo_path) is False

    @patch("subprocess.run")
    def test_clone_repo_success(self, mock_run, tmp_path):
        """Test successful repository cloning."""
        config = Mock(spec=Config)
        loader = GitHubRepositoryLoader(config)

        mock_run.return_value = Mock(returncode=0, stderr="")

        target_dir = tmp_path / "test-repo"
        result = loader.clone_repo("owner/repo", target_dir)

        assert result is True
        mock_run.assert_called_once_with(
            ["gh", "repo", "clone", "owner/repo", str(target_dir)],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("subprocess.run")
    def test_clone_repo_failure(self, mock_run, tmp_path):
        """Test failed repository cloning."""
        config = Mock(spec=Config)
        loader = GitHubRepositoryLoader(config)

        mock_run.side_effect = subprocess.CalledProcessError(1, "gh", stderr="Error")

        target_dir = tmp_path / "test-repo"
        result = loader.clone_repo("owner/repo", target_dir)

        assert result is False

    @patch("subprocess.run")
    def test_update_repo_success(self, mock_run, tmp_path):
        """Test successful repository update."""
        config = Mock(spec=Config)
        loader = GitHubRepositoryLoader(config)

        mock_run.return_value = Mock(returncode=0, stderr="")

        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()
        result = loader.update_repo(repo_path)

        assert result is True
        mock_run.assert_called_once_with(
            ["git", "-C", str(repo_path), "pull", "--ff-only"],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("subprocess.run")
    def test_update_repo_failure(self, mock_run, tmp_path):
        """Test failed repository update."""
        config = Mock(spec=Config)
        loader = GitHubRepositoryLoader(config)

        mock_run.side_effect = subprocess.CalledProcessError(1, "git", stderr="Error")

        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()
        result = loader.update_repo(repo_path)

        assert result is False

    def test_load_repository_not_exists(self):
        """Test loading from non-existent repository."""
        config = Mock(spec=Config)
        loader = GitHubRepositoryLoader(config)

        docs = loader.load_repository("owner/repo", Path("/non/existent"))
        assert docs == []

    def test_load_repository_success(self, tmp_path):
        """Test successful repository loading."""
        config = Mock(spec=Config)
        loader = GitHubRepositoryLoader(config)

        # Setup mock documents
        mock_docs = [
            Document(content="Test content", meta={"category": "src", "subcategory": "module"}),
            Document(content="README", meta={"category": "root"}),
        ]
        loader.hierarchical_loader.load_from_directory = Mock(return_value=mock_docs)

        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()

        docs = loader.load_repository("owner/repo", repo_path, max_documents=10)

        assert len(docs) == 2
        # Check that category was overridden to repo name
        assert docs[0].meta["category"] == "repo"
        assert docs[1].meta["category"] == "repo"
        # Check that original categories were preserved as subcategories
        assert "src" in docs[0].meta["subcategory"]

    @patch.object(GitHubRepositoryLoader, "check_local_repo")
    @patch.object(GitHubRepositoryLoader, "clone_repo")
    @patch.object(GitHubRepositoryLoader, "load_repository")
    def test_process_repository_local_exists(self, mock_load, mock_clone, mock_check, tmp_path):
        """Test processing repository that exists locally."""
        config = Mock(spec=Config)
        loader = GitHubRepositoryLoader(config)

        mock_check.return_value = True
        mock_docs = [Document(content="Test", meta={})]
        mock_load.return_value = mock_docs

        docs, status = loader.process_repository("owner/repo", tmp_path)

        assert len(docs) == 1
        assert "Found local" in status
        mock_clone.assert_not_called()

    @patch.object(GitHubRepositoryLoader, "check_local_repo")
    @patch.object(GitHubRepositoryLoader, "clone_repo")
    @patch.object(GitHubRepositoryLoader, "load_repository")
    def test_process_repository_needs_clone(self, mock_load, mock_clone, mock_check, tmp_path):
        """Test processing repository that needs cloning."""
        config = Mock(spec=Config)
        loader = GitHubRepositoryLoader(config)

        mock_check.return_value = False
        mock_clone.return_value = True
        mock_docs = [Document(content="Test", meta={})]
        mock_load.return_value = mock_docs

        docs, status = loader.process_repository("owner/repo", tmp_path)

        assert len(docs) == 1
        assert "Cloned" in status
        mock_clone.assert_called_once()

    def test_process_repositories_from_file(self, tmp_path):
        """Test processing multiple repositories from file."""
        config = Mock(spec=Config)
        loader = GitHubRepositoryLoader(config)

        # Create repos file
        repos_file = tmp_path / "repos.txt"
        repos_file.write_text("owner1/repo1\n# comment\nowner2/repo2\n")

        # Mock process_repository
        with patch.object(loader, "process_repository") as mock_process:
            mock_process.side_effect = [
                ([Document(content="Doc1", meta={})], "Status1"),
                ([Document(content="Doc2", meta={})], "Status2"),
            ]

            docs, statuses = loader.process_repositories_from_file(repos_file, tmp_path)

            assert len(docs) == 2
            assert len(statuses) == 2
            assert mock_process.call_count == 2

    def test_process_repositories_from_file_not_exists(self):
        """Test processing from non-existent file."""
        config = Mock(spec=Config)
        loader = GitHubRepositoryLoader(config)

        with pytest.raises(ValueError) as exc_info:
            loader.process_repositories_from_file(Path("/non/existent.txt"))
        assert "not found" in str(exc_info.value)

    def test_markdown_only_filtering(self):
        """Test that only markdown files are configured for processing."""
        config = Mock(spec=Config)
        loader = GitHubRepositoryLoader(config)

        # Verify only .md is in allowed extensions
        assert loader.allowed_extensions == [".md"]
        assert ".py" not in loader.allowed_extensions
        assert ".js" not in loader.allowed_extensions
        assert ".txt" not in loader.allowed_extensions
