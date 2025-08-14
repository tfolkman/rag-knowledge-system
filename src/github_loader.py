"""GitHub Repository Loader for batch ingestion into RAG system."""

import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from haystack import Document

from .config import Config
from .hierarchical_loader import HierarchicalDocumentLoader

logger = logging.getLogger(__name__)


class GitHubRepositoryLoader:
    """Load documents from GitHub repositories with local caching."""

    def __init__(self, config: Config):
        """Initialize the GitHub repository loader.

        Args:
            config: Configuration object
        """
        self.config = config
        self.hierarchical_loader = HierarchicalDocumentLoader()

        # Default configurations
        self.local_repos_dir = Path.home() / "Coding"
        self.max_file_size_mb = 10.0
        self.allowed_extensions = [
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".md",
            ".txt",
            ".rst",
            ".yaml",
            ".yml",
            ".json",
            ".toml",
            ".sh",
            ".bash",
            ".zsh",
            ".go",
            ".rs",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".html",
            ".css",
            ".scss",
            ".sql",
            ".graphql",
            ".dockerfile",
            ".dockerignore",
            ".gitignore",
            ".env.example",
        ]

    def parse_repo_name(self, repo_identifier: str) -> Tuple[str, str]:
        """Parse repository identifier to extract owner and repo name.

        Args:
            repo_identifier: Repository in format "owner/repo"

        Returns:
            Tuple of (owner, repo_name)

        Raises:
            ValueError: If repo_identifier is not in expected format
        """
        parts = repo_identifier.strip().split("/")
        if len(parts) != 2:
            raise ValueError(
                f"Repository identifier must be in format 'owner/repo', got: {repo_identifier}"
            )
        return parts[0], parts[1]

    def get_local_repo_path(self, repo_identifier: str, local_dir: Optional[Path] = None) -> Path:
        """Get the expected local path for a repository.

        Args:
            repo_identifier: Repository in format "owner/repo"
            local_dir: Directory to check for repos (default: ~/Coding)

        Returns:
            Path where the repository should exist locally
        """
        local_dir = local_dir or self.local_repos_dir
        _, repo_name = self.parse_repo_name(repo_identifier)
        return local_dir / repo_name

    def check_local_repo(self, repo_path: Path) -> bool:
        """Check if a repository exists locally and is a valid git repo.

        Args:
            repo_path: Path to check for repository

        Returns:
            True if valid git repository exists at path
        """
        if not repo_path.exists():
            return False

        # Check if it's a valid git repository
        git_dir = repo_path / ".git"
        return git_dir.exists() and git_dir.is_dir()

    def clone_repo(self, repo_identifier: str, target_dir: Path) -> bool:
        """Clone a repository using GitHub CLI.

        Args:
            repo_identifier: Repository in format "owner/repo"
            target_dir: Directory to clone into

        Returns:
            True if cloning succeeded
        """
        try:
            # Ensure target directory exists
            target_dir.parent.mkdir(parents=True, exist_ok=True)

            # Use gh CLI to clone
            cmd = ["gh", "repo", "clone", repo_identifier, str(target_dir)]
            logger.info(f"Cloning repository: {repo_identifier} to {target_dir}")

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if result.returncode == 0:
                logger.info(f"Successfully cloned: {repo_identifier}")
                return True
            else:
                logger.error(f"Failed to clone {repo_identifier}: {result.stderr}")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"Error cloning {repo_identifier}: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error cloning {repo_identifier}: {e}")
            return False

    def update_repo(self, repo_path: Path) -> bool:
        """Update an existing repository by pulling latest changes.

        Args:
            repo_path: Path to the repository

        Returns:
            True if update succeeded
        """
        try:
            cmd = ["git", "-C", str(repo_path), "pull", "--ff-only"]
            logger.info(f"Updating repository: {repo_path}")

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if result.returncode == 0:
                logger.info(f"Successfully updated: {repo_path}")
                return True
            else:
                logger.warning(f"Could not fast-forward update {repo_path}: {result.stderr}")
                return False

        except subprocess.CalledProcessError as e:
            logger.warning(f"Error updating {repo_path}: {e.stderr}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error updating {repo_path}: {e}")
            return False

    def load_repository(
        self, repo_identifier: str, repo_path: Path, max_documents: Optional[int] = None
    ) -> List[Document]:
        """Load documents from a repository with hierarchical metadata.

        Args:
            repo_identifier: Repository in format "owner/repo"
            repo_path: Local path to the repository
            max_documents: Maximum number of documents to load

        Returns:
            List of Document objects with repository metadata
        """
        if not repo_path.exists():
            logger.error(f"Repository path does not exist: {repo_path}")
            return []

        logger.info(f"Loading documents from repository: {repo_identifier} at {repo_path}")

        # Extract repository name for category
        _, repo_name = self.parse_repo_name(repo_identifier)

        # Use hierarchical loader to load documents
        documents = self.hierarchical_loader.load_from_directory(
            root_path=repo_path,
            allowed_extensions=self.allowed_extensions,
            max_file_size_mb=self.max_file_size_mb,
            additional_metadata={
                "repository": repo_identifier,
                "source": "github",
                "local_path": str(repo_path),
            },
        )

        # Override category to be repository name
        for doc in documents:
            # Keep the original category as subcategory if it wasn't root
            original_category = doc.meta.get("category", "root")
            if original_category != "root":
                doc.meta["subcategory"] = f"{original_category}/{doc.meta.get('subcategory', '')}"
            doc.meta["category"] = repo_name

        # Limit documents if specified
        if max_documents and len(documents) > max_documents:
            documents = documents[:max_documents]

        logger.info(f"Loaded {len(documents)} documents from {repo_identifier}")
        return documents

    def process_repository(
        self,
        repo_identifier: str,
        local_dir: Optional[Path] = None,
        force_clone: bool = False,
        update_existing: bool = True,
        max_documents: Optional[int] = None,
    ) -> Tuple[List[Document], str]:
        """Process a single repository - check locally, clone if needed, and load documents.

        Args:
            repo_identifier: Repository in format "owner/repo"
            local_dir: Directory to check/clone repos (default: ~/Coding)
            force_clone: Force fresh clone even if exists locally
            update_existing: Update existing repos with git pull
            max_documents: Maximum number of documents to load

        Returns:
            Tuple of (documents, status_message)
        """
        local_dir = local_dir or self.local_repos_dir
        repo_path = self.get_local_repo_path(repo_identifier, local_dir)

        # Check if repository exists locally
        if self.check_local_repo(repo_path) and not force_clone:
            logger.info(f"Found local repository: {repo_identifier} at {repo_path}")

            # Optionally update the repository
            if update_existing:
                self.update_repo(repo_path)

            status = f"Found local: {repo_identifier} at {repo_path}"
        else:
            # Clone the repository
            if force_clone and repo_path.exists():
                logger.info(f"Force clone requested, removing existing: {repo_path}")
                import shutil

                shutil.rmtree(repo_path)

            success = self.clone_repo(repo_identifier, repo_path)
            if not success:
                return [], f"Failed to clone: {repo_identifier}"

            status = f"Cloned: {repo_identifier} to {repo_path}"

        # Load documents from the repository
        documents = self.load_repository(repo_identifier, repo_path, max_documents)

        return documents, status

    def process_repositories_from_file(
        self,
        repos_file: Path,
        local_dir: Optional[Path] = None,
        force_clone: bool = False,
        update_existing: bool = True,
        max_documents_per_repo: Optional[int] = None,
    ) -> Tuple[List[Document], List[str]]:
        """Process multiple repositories from a file.

        Args:
            repos_file: Path to file containing repository identifiers (one per line)
            local_dir: Directory to check/clone repos (default: ~/Coding)
            force_clone: Force fresh clone even if exists locally
            update_existing: Update existing repos with git pull
            max_documents_per_repo: Maximum documents per repository

        Returns:
            Tuple of (all_documents, status_messages)
        """
        if not repos_file.exists():
            raise ValueError(f"Repository list file not found: {repos_file}")

        # Read repository list
        with open(repos_file, "r") as f:
            repo_list = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        logger.info(f"Processing {len(repo_list)} repositories from {repos_file}")

        all_documents = []
        status_messages = []

        for idx, repo_identifier in enumerate(repo_list, 1):
            logger.info(f"[{idx}/{len(repo_list)}] Processing: {repo_identifier}")

            try:
                documents, status = self.process_repository(
                    repo_identifier, local_dir, force_clone, update_existing, max_documents_per_repo
                )

                all_documents.extend(documents)
                status_messages.append(
                    f"[{idx}/{len(repo_list)}] {status} - {len(documents)} documents"
                )

            except Exception as e:
                error_msg = f"[{idx}/{len(repo_list)}] Error processing {repo_identifier}: {e}"
                logger.error(error_msg)
                status_messages.append(error_msg)
                continue

        logger.info(
            f"Processed {len(repo_list)} repositories, loaded {len(all_documents)} total documents"
        )
        return all_documents, status_messages
