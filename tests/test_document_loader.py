from unittest.mock import Mock, patch

import pytest

from src.config import Config
from src.document_loader import GoogleDriveLoader


class TestGoogleDriveLoader:
    """Test Google Drive document loader."""

    def setup_method(self):
        """Setup test fixtures."""
        # Reset singleton instance
        Config._instance = None
        self.config = Config()
        self.config.google_credentials_path = "test_credentials.json"

    def test_loader_initialization(self):
        """Test that loader initializes correctly."""
        loader = GoogleDriveLoader(self.config)

        assert loader.config == self.config
        assert loader.credentials_path == "test_credentials.json"
        assert loader.service is None

    @patch("src.document_loader.build")
    @patch("src.document_loader.service_account.Credentials")
    def test_authenticate_success(self, mock_credentials, mock_build):
        """Test successful authentication with Google Drive API."""
        # Mock credentials
        mock_creds = Mock()
        mock_credentials.from_service_account_file.return_value = mock_creds

        # Mock Drive service
        mock_service = Mock()
        mock_build.return_value = mock_service

        loader = GoogleDriveLoader(self.config)
        loader.authenticate()

        assert loader.service == mock_service
        mock_credentials.from_service_account_file.assert_called_once_with(
            "test_credentials.json",
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        mock_build.assert_called_once_with("drive", "v3", credentials=mock_creds)

    @patch("src.document_loader.service_account.Credentials")
    def test_authenticate_invalid_credentials(self, mock_credentials):
        """Test authentication failure with invalid credentials."""
        mock_credentials.from_service_account_file.side_effect = Exception("Invalid credentials")

        loader = GoogleDriveLoader(self.config)

        with pytest.raises(Exception, match="Invalid credentials"):
            loader.authenticate()

    def test_list_documents_without_authentication(self):
        """Test that listing documents without authentication raises error."""
        loader = GoogleDriveLoader(self.config)

        with pytest.raises(ValueError, match="Not authenticated"):
            loader.list_documents()

    @patch("src.document_loader.build")
    @patch("src.document_loader.service_account.Credentials")
    def test_list_documents_success(self, mock_credentials, mock_build):
        """Test successful document listing."""
        # Setup authentication mocks
        mock_creds = Mock()
        mock_credentials.from_service_account_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service

        # Mock API response
        mock_files = Mock()
        mock_service.files.return_value = mock_files
        mock_list = Mock()
        mock_files.list.return_value = mock_list
        mock_list.execute.return_value = {
            "files": [
                {"id": "1", "name": "test1.txt", "mimeType": "text/plain"},
                {"id": "2", "name": "test2.pdf", "mimeType": "application/pdf"},
                {
                    "id": "3",
                    "name": "test3.docx",
                    "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                },
            ]
        }

        loader = GoogleDriveLoader(self.config)
        loader.authenticate()
        documents = loader.list_documents()

        assert len(documents) == 3
        assert documents[0]["name"] == "test1.txt"
        assert documents[1]["name"] == "test2.pdf"
        assert documents[2]["name"] == "test3.docx"

        mock_files.list.assert_called_once_with(
            q="trashed=false and (mimeType='text/plain' or mimeType='application/pdf' or mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document' or mimeType='application/vnd.google-apps.document')",
            fields="files(id, name, mimeType, modifiedTime)",
        )

    @patch("src.document_loader.build")
    @patch("src.document_loader.service_account.Credentials")
    def test_list_documents_with_folder_filter(self, mock_credentials, mock_build):
        """Test document listing with folder filter."""
        # Setup authentication mocks
        mock_creds = Mock()
        mock_credentials.from_service_account_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service

        # Mock API response
        mock_files = Mock()
        mock_service.files.return_value = mock_files
        mock_list = Mock()
        mock_files.list.return_value = mock_list
        mock_list.execute.return_value = {"files": []}

        loader = GoogleDriveLoader(self.config)
        loader.authenticate()
        loader.list_documents(folder_id="test_folder_id")

        expected_query = "trashed=false and 'test_folder_id' in parents and (mimeType='text/plain' or mimeType='application/pdf' or mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document' or mimeType='application/vnd.google-apps.document')"
        mock_files.list.assert_called_once_with(
            q=expected_query, fields="files(id, name, mimeType, modifiedTime)"
        )

    def test_download_document_without_authentication(self):
        """Test downloading document without authentication raises error."""
        loader = GoogleDriveLoader(self.config)

        with pytest.raises(ValueError, match="Not authenticated"):
            loader.download_document("test_id")

    @patch("src.document_loader.build")
    @patch("src.document_loader.service_account.Credentials")
    @patch("src.document_loader.io.BytesIO")
    def test_download_document_success(self, mock_bytesio, mock_credentials, mock_build):
        """Test successful document download."""
        # Setup authentication mocks
        mock_creds = Mock()
        mock_credentials.from_service_account_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service

        # Mock download
        mock_files = Mock()
        mock_service.files.return_value = mock_files
        mock_get = Mock()
        mock_files.get.return_value = mock_get
        mock_media = Mock()
        mock_get.get_media.return_value = mock_media

        # Mock BytesIO
        mock_buffer = Mock()
        mock_bytesio.return_value = mock_buffer
        mock_buffer.getvalue.return_value = b"Test document content"

        # Mock MediaIoBaseDownload
        with patch("src.document_loader.MediaIoBaseDownload") as mock_download:
            mock_downloader = Mock()
            mock_download.return_value = mock_downloader
            mock_downloader.next_chunk.side_effect = [
                (False, Mock(progress=lambda: 0.5)),
                (True, Mock(progress=lambda: 1.0)),
            ]

            loader = GoogleDriveLoader(self.config)
            loader.authenticate()
            content = loader.download_document("test_file_id")

            assert content == b"Test document content"
            mock_files.get.assert_called_once_with(fileId="test_file_id")
            mock_get.get_media.assert_called_once()

    @patch("src.document_loader.build")
    @patch("src.document_loader.service_account.Credentials")
    def test_load_documents_batch(self, mock_credentials, mock_build):
        """Test loading documents in batches."""
        # Setup authentication mocks
        mock_creds = Mock()
        mock_credentials.from_service_account_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service

        loader = GoogleDriveLoader(self.config)
        loader.authenticate()

        # Mock list_documents
        with patch.object(loader, "list_documents") as mock_list:
            mock_list.return_value = [
                {"id": "1", "name": "test1.txt", "mimeType": "text/plain"},
                {"id": "2", "name": "test2.txt", "mimeType": "text/plain"},
            ]

            # Mock download_document
            with patch.object(loader, "download_document") as mock_download:
                mock_download.side_effect = [b"Content 1", b"Content 2"]

                documents = loader.load_documents(max_documents=2)

                assert len(documents) == 2
                assert documents[0]["content"] == b"Content 1"
                assert documents[0]["metadata"]["name"] == "test1.txt"
                assert documents[1]["content"] == b"Content 2"
                assert documents[1]["metadata"]["name"] == "test2.txt"

    def test_get_supported_mime_types(self):
        """Test that loader returns supported MIME types."""
        loader = GoogleDriveLoader(self.config)
        mime_types = loader.get_supported_mime_types()

        expected_types = [
            "text/plain",
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.google-apps.document",
        ]

        assert set(mime_types) == set(expected_types)
