"""Hierarchical Document Loader for folder-based categorization."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from haystack import Document

from .config import Config
from .document_loader import GoogleDriveLoader

logger = logging.getLogger(__name__)


class HierarchicalDocumentLoader:
    """Load documents with hierarchical folder metadata for categorization."""

    def __init__(self):
        """Initialize the hierarchical document loader."""
        self.supported_extensions = {".txt", ".md", ".pdf", ".doc", ".docx"}

    def load_from_directory(
        self,
        root_path: Path,
        allowed_extensions: Optional[List[str]] = None,
        max_file_size_mb: Optional[float] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Load documents from a directory with hierarchical metadata.

        Args:
            root_path: Root directory to traverse
            allowed_extensions: List of allowed file extensions (e.g., ['.txt', '.pdf'])
            max_file_size_mb: Maximum file size in megabytes
            additional_metadata: Additional metadata to add to all documents

        Returns:
            List of Document objects with hierarchical metadata

        Raises:
            ValueError: If directory does not exist
            PermissionError: If access is denied to directory
        """
        if not root_path.exists():
            raise ValueError(f"Directory does not exist: {root_path}")

        if not root_path.is_dir():
            raise ValueError(f"Path is not a directory: {root_path}")

        # Use provided extensions or defaults
        extensions = set(allowed_extensions) if allowed_extensions else self.supported_extensions

        documents = []
        max_size_bytes = (max_file_size_mb * 1024 * 1024) if max_file_size_mb else None

        # Recursively traverse directory
        for file_path in self._traverse_directory(root_path):
            # Check file extension
            if file_path.suffix.lower() not in extensions:
                logger.debug(f"Skipping unsupported file type: {file_path}")
                continue

            # Check file size
            if max_size_bytes and file_path.stat().st_size > max_size_bytes:
                logger.warning(f"Skipping large file: {file_path} (size > {max_file_size_mb}MB)")
                continue

            # Load file content
            try:
                content = self._read_file_content(file_path)
                if not content:
                    logger.warning(f"Empty file skipped: {file_path}")
                    continue

                # Extract hierarchical metadata
                metadata = self._extract_hierarchical_metadata(file_path, root_path)

                # Add additional metadata if provided
                if additional_metadata:
                    metadata.update(additional_metadata)

                # Create Document
                document = Document(content=content, meta=metadata)
                documents.append(document)
                logger.debug(
                    f"Loaded document: {file_path} with category: {metadata.get('category')}"
                )

            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
                continue

        logger.info(f"Loaded {len(documents)} documents from {root_path}")
        return documents

    def load_from_google_drive(
        self, config: Config, folder_id: str, max_documents: Optional[int] = None
    ) -> List[Document]:
        """
        Load documents from Google Drive with hierarchical folder metadata.

        Args:
            config: Configuration object
            folder_id: Google Drive folder ID
            max_documents: Maximum number of documents to load

        Returns:
            List of Document objects with hierarchical metadata
        """
        loader = GoogleDriveLoader(config)
        loader.authenticate()

        # Get all folders recursively under the root folder
        logger.info(f"Loading folder structure from Google Drive: {folder_id}")
        all_folders = self._get_all_folders_recursive(loader, folder_id)

        # Build folder hierarchy map
        folder_map = self._build_folder_hierarchy_from_drive(all_folders, folder_id)

        # Load documents from all folders (root + subfolders)
        all_documents = []
        folders_to_process = [folder_id] + [f["id"] for f in all_folders]

        logger.info(f"Processing {len(folders_to_process)} folders...")
        for idx, current_folder_id in enumerate(folders_to_process):
            folder_name = folder_map.get(current_folder_id, {}).get("name", "root")
            logger.debug(
                f"Loading documents from folder {idx+1}/{len(folders_to_process)}: {folder_name}"
            )

            # Load documents from this specific folder
            raw_documents = loader.load_documents(current_folder_id, max_documents=None)

            for doc_data in raw_documents:
                # Create Document object from raw data
                content = doc_data.get("content", "")
                metadata = doc_data.get("metadata", {})

                # Add hierarchical metadata based on folder
                hierarchical_meta = self._get_google_drive_hierarchy(
                    [current_folder_id], folder_map
                )
                metadata.update(hierarchical_meta)
                metadata["source"] = "google_drive"
                metadata["file_name"] = metadata.get("name", "unknown")
                metadata["folder_id"] = current_folder_id

                # Create Document
                document = Document(content=content, meta=metadata)
                all_documents.append(document)

                # Stop if we've reached max documents
                if max_documents and len(all_documents) >= max_documents:
                    break

            if max_documents and len(all_documents) >= max_documents:
                break

        logger.info(f"Loaded {len(all_documents)} documents from Google Drive with hierarchy")
        return all_documents

    def _traverse_directory(self, directory: Path) -> List[Path]:
        """
        Recursively traverse directory and return all file paths.

        Args:
            directory: Directory to traverse

        Returns:
            List of file paths
        """
        file_paths = []

        try:
            for item in directory.iterdir():
                if item.is_dir():
                    # Recursively traverse subdirectories
                    file_paths.extend(self._traverse_directory(item))
                elif item.is_file():
                    file_paths.append(item)
        except PermissionError as e:
            logger.error(f"Permission denied accessing: {directory}")
            raise e

        return file_paths

    def _read_file_content(self, file_path: Path) -> str:
        """
        Read content from a file.

        Args:
            file_path: Path to the file

        Returns:
            File content as string
        """
        try:
            if file_path.suffix.lower() == ".pdf":
                # For PDF files, we'd use a PDF reader library
                # For now, return placeholder
                logger.debug(f"PDF file detected: {file_path}")
                return file_path.read_bytes().decode("utf-8", errors="ignore")[:1000]
            else:
                # Text-based files
                return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ""

    def _extract_hierarchical_metadata(self, file_path: Path, root_path: Path) -> Dict[str, Any]:
        """
        Extract hierarchical metadata from file path.

        Args:
            file_path: Path to the file
            root_path: Root directory path

        Returns:
            Dictionary of metadata
        """
        # Get relative path from root
        relative_path = file_path.relative_to(root_path)
        path_parts = relative_path.parts[:-1]  # Exclude the filename

        # Extract category and subcategory
        category = path_parts[0] if len(path_parts) > 0 else "root"
        subcategory = path_parts[1] if len(path_parts) > 1 else None

        # Build hierarchy path
        hierarchy_path = "/".join(path_parts) if path_parts else "root"

        # Get file stats
        file_stats = file_path.stat()

        metadata = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "category": category,
            "subcategory": subcategory,
            "hierarchy_path": hierarchy_path,
            "hierarchy_level": len(path_parts),
            "file_type": file_path.suffix.lower(),
            "file_size_bytes": file_stats.st_size,
            "modified_date": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "created_date": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "source": "local",
        }

        return metadata

    def _get_all_folders_recursive(
        self, loader: GoogleDriveLoader, folder_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all folders recursively under a root folder.

        Args:
            loader: GoogleDriveLoader instance
            folder_id: Root folder ID to start from

        Returns:
            List of folder dictionaries with id, name, and parents
        """
        if not loader.service:
            return []

        all_folders = []
        folders_to_process = [folder_id]
        processed_folders = set()

        while folders_to_process:
            current_folder = folders_to_process.pop(0)
            if current_folder in processed_folders:
                continue

            processed_folders.add(current_folder)

            try:
                # Query for subfolders in current folder
                query = f"'{current_folder}' in parents and mimeType='application/vnd.google-apps.folder'"
                response = (
                    loader.service.files()
                    .list(q=query, fields="files(id, name, parents)", pageSize=1000)
                    .execute()
                )

                subfolders = response.get("files", [])
                for folder in subfolders:
                    all_folders.append(folder)
                    folders_to_process.append(folder["id"])

            except Exception as e:
                logger.warning(f"Error getting subfolders for {current_folder}: {e}")

        return all_folders

    def _build_folder_hierarchy_from_drive(
        self, folders: List[Dict[str, Any]], root_folder_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build a hierarchy map from Google Drive folder structure.

        Args:
            folders: List of folder dictionaries from Google Drive
            root_folder_id: The root folder ID

        Returns:
            Dictionary mapping folder IDs to folder info with hierarchy
        """
        # Initialize with root folder
        folder_map: Dict[str, Dict[str, Any]] = {
            root_folder_id: {
                "name": "root",
                "parents": [],
                "path": "root",
            }
        }

        # Add all folders to map
        for folder in folders:
            folder_id = folder.get("id")
            if folder_id:
                folder_map[folder_id] = {
                    "name": folder.get("name"),
                    "parents": folder.get("parents", []),
                    "path": None,  # Will be computed
                }

        # Compute full paths for each folder
        for folder_id, folder_info in folder_map.items():
            if folder_id == root_folder_id:
                continue  # Skip root, already has path

            path_parts = [folder_info["name"]]
            current_parents = folder_info["parents"]

            # Traverse up the hierarchy
            while current_parents:
                parent_id = current_parents[0]
                if parent_id in folder_map:
                    parent_info = folder_map[parent_id]
                    if parent_info["name"] != "root":
                        path_parts.insert(0, parent_info["name"])
                    current_parents = parent_info["parents"]
                else:
                    break

            folder_info["path"] = "/".join(path_parts)

        return folder_map

    def _build_folder_hierarchy(self, folders: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Build a hierarchy map from Google Drive folder structure.

        Args:
            folders: List of folder dictionaries from Google Drive

        Returns:
            Dictionary mapping folder IDs to folder info with hierarchy
        """
        folder_map: Dict[str, Dict[str, Any]] = {}

        for folder in folders:
            folder_id = folder.get("id")
            if folder_id:
                folder_map[folder_id] = {
                    "name": folder.get("name"),
                    "parents": folder.get("parents", []),
                    "path": None,  # Will be computed
                }

        # Compute full paths for each folder
        for folder_id, folder_info in folder_map.items():
            path_parts = [folder_info["name"]]
            current_parents = folder_info["parents"]

            # Traverse up the hierarchy
            while current_parents and current_parents[0] in folder_map:
                parent_id = current_parents[0]
                parent_info = folder_map[parent_id]
                path_parts.insert(0, parent_info["name"])
                current_parents = parent_info["parents"]

            folder_info["path"] = "/".join(path_parts)

        return folder_map

    def _get_google_drive_hierarchy(
        self, parent_ids: List[str], folder_map: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get hierarchical metadata for a Google Drive file.

        Args:
            parent_ids: List of parent folder IDs
            folder_map: Folder hierarchy map

        Returns:
            Dictionary of hierarchical metadata
        """
        if not parent_ids or not folder_map:
            return {
                "category": "root",
                "subcategory": None,
                "hierarchy_path": "root",
                "hierarchy_level": 0,
            }

        # Get immediate parent folder
        parent_id = parent_ids[0]
        if parent_id not in folder_map:
            return {
                "category": "unknown",
                "subcategory": None,
                "hierarchy_path": "unknown",
                "hierarchy_level": 0,
            }

        folder_info = folder_map[parent_id]
        path = folder_info.get("path", "")
        path_parts = path.split("/") if path else []

        return {
            "category": path_parts[0] if path_parts else "root",
            "subcategory": path_parts[1] if len(path_parts) > 1 else None,
            "hierarchy_path": path,
            "hierarchy_level": len(path_parts),
        }
