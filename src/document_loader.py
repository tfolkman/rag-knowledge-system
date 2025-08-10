import io
from typing import Any, Dict, List, Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build  # type: ignore
from googleapiclient.http import MediaIoBaseDownload  # type: ignore

from src.config import Config


class GoogleDriveLoader:
    """Google Drive document loader for RAG system."""

    SUPPORTED_MIME_TYPES = [
        "text/plain",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.google-apps.document",
    ]

    def __init__(self, config: Config):
        """Initialize the Google Drive loader.

        Args:
            config: Configuration object
        """
        self.config = config
        self.credentials_path = config.google_credentials_path
        self.service = None
        self._credentials = None

    def authenticate(self) -> None:
        """Authenticate with Google Drive API."""
        try:
            scopes = ["https://www.googleapis.com/auth/drive.readonly"]
            self._credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path, scopes=scopes
            )
            self.service = build("drive", "v3", credentials=self._credentials)
        except Exception as e:
            raise Exception(f"Authentication failed: {e}")

    def list_documents(self, folder_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List documents from Google Drive.

        Args:
            folder_id: Optional folder ID to filter documents

        Returns:
            List of document metadata
        """
        if not self.service:
            raise ValueError("Not authenticated. Call authenticate() first.")

        # Build query
        mime_type_query = " or ".join([f"mimeType='{mt}'" for mt in self.SUPPORTED_MIME_TYPES])

        if folder_id:
            query = f"trashed=false and '{folder_id}' in parents and ({mime_type_query})"
        else:
            query = f"trashed=false and ({mime_type_query})"

        try:
            results = (
                self.service.files()
                .list(q=query, fields="files(id, name, mimeType, modifiedTime)")
                .execute()
            )

            return results.get("files", [])
        except Exception as e:
            raise Exception(f"Failed to list documents: {e}")

    def download_document(self, file_id: str) -> bytes:
        """Download document content.

        Args:
            file_id: Google Drive file ID

        Returns:
            Document content as bytes
        """
        if not self.service:
            raise ValueError("Not authenticated. Call authenticate() first.")

        try:
            # First, get file metadata to check MIME type
            file_metadata = self.service.files().get(fileId=file_id).execute()
            mime_type = file_metadata.get("mimeType", "")

            # Handle Google Docs (need to export as text)
            if mime_type == "application/vnd.google-apps.document":
                request = self.service.files().export_media(fileId=file_id, mimeType="text/plain")
            else:
                # For other files, download directly
                request = self.service.files().get_media(fileId=file_id)

            # Download file content
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)

            done = False
            while not done:
                done, _ = downloader.next_chunk()

            return buffer.getvalue()

        except Exception as e:
            raise Exception(f"Failed to download document {file_id}: {e}")

    def load_documents(
        self, folder_id: Optional[str] = None, max_documents: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load documents from Google Drive.

        Args:
            folder_id: Optional folder ID to filter documents
            max_documents: Maximum number of documents to load

        Returns:
            List of documents with content and metadata
        """
        if not self.service:
            raise ValueError("Not authenticated. Call authenticate() first.")

        # List documents
        doc_list = self.list_documents(folder_id)

        # Limit number of documents
        if max_documents:
            doc_list = doc_list[:max_documents]

        documents = []
        for doc_meta in doc_list:
            try:
                content = self.download_document(doc_meta["id"])
                documents.append(
                    {
                        "content": content,
                        "metadata": {
                            "id": doc_meta["id"],
                            "name": doc_meta["name"],
                            "mimeType": doc_meta["mimeType"],
                            "source": "google_drive",
                        },
                    }
                )
            except Exception as e:
                print(f"Failed to download document {doc_meta['name']}: {e}")
                continue

        return documents

    def get_supported_mime_types(self) -> List[str]:
        """Get supported MIME types.

        Returns:
            List of supported MIME types
        """
        return self.SUPPORTED_MIME_TYPES.copy()
