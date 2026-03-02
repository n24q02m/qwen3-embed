"""Tests for model management utility functions."""

import io
import json
import tarfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.hf_api import RepoFile

from qwen3_embed.common.model_description import BaseModelDescription, ModelSource
from qwen3_embed.common.model_management import ModelManagement

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def make_repo_file(path: str, size: int = 100, oid: str = "abc123") -> RepoFile:
    """Create a RepoFile instance for testing."""
    return RepoFile(path=path, size=size, oid=oid)


def make_tar_gz(tmp_path: Path, inner_name: str = "model.onnx") -> Path:
    """Create a minimal valid .tar.gz archive and return its path."""
    tar_path = tmp_path / "model.tar.gz"
    inner_file = tmp_path / inner_name
    inner_file.write_bytes(b"dummy content")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(inner_file), arcname=inner_name)
    tar_path.write_bytes(buf.getvalue())
    return tar_path


def make_model_description(
    model: str = "test/model",
    hf: str | None = "org/repo",
    url: str | None = None,
    model_file: str = "model.onnx",
    additional_files: list[str] | None = None,
    deprecated_tar_struct: bool = False,
) -> BaseModelDescription:
    """Build a minimal BaseModelDescription for testing."""
    source = ModelSource(hf=hf, url=url, _deprecated_tar_struct=deprecated_tar_struct)
    return BaseModelDescription(
        model=model,
        sources=source,
        model_file=model_file,
        description="Test model",
        license="MIT",
        size_in_GB=0.1,
        additional_files=additional_files or [],
    )


class ConcreteModelManagement(ModelManagement):
    """Concrete subclass for testing abstract base class methods."""

    _models: list[BaseModelDescription] = []

    @classmethod
    def _list_supported_models(cls) -> list[BaseModelDescription]:
        return cls._models

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        return [{"model": m.model} for m in cls._models]

    @classmethod
    def add_custom_model(cls, *args: Any, **kwargs: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# TestAbstractMethods
# ---------------------------------------------------------------------------


class TestAbstractMethods:
    """Tests for methods that raise NotImplementedError in the base class."""

    def test_list_supported_models_raises(self):
        with pytest.raises(NotImplementedError):
            ModelManagement.list_supported_models()

    def test_add_custom_model_raises(self):
        with pytest.raises(NotImplementedError):
            ModelManagement.add_custom_model()

    def test_list_supported_models_private_raises(self):
        with pytest.raises(NotImplementedError):
            ModelManagement._list_supported_models()


# ---------------------------------------------------------------------------
# TestGetModelDescription
# ---------------------------------------------------------------------------


class TestGetModelDescription:
    """Tests for _get_model_description."""

    def setup_method(self):
        ConcreteModelManagement._models = [
            make_model_description(model="Alpha"),
            make_model_description(model="Beta"),
        ]

    def test_found_exact_case(self):
        result = ConcreteModelManagement._get_model_description("Alpha")
        assert result.model == "Alpha"

    def test_found_case_insensitive(self):
        result = ConcreteModelManagement._get_model_description("BETA")
        assert result.model == "Beta"

    def test_not_found_raises_value_error(self):
        with pytest.raises(ValueError, match="Gamma is not supported"):
            ConcreteModelManagement._get_model_description("Gamma")


# ---------------------------------------------------------------------------
# TestDownloadFileFromGcs
# ---------------------------------------------------------------------------


class TestDownloadFileFromGcs:
    """Tests for download_file_from_gcs."""

    def test_returns_existing_file(self, tmp_path):
        """If the file already exists, return immediately without HTTP request."""
        existing = tmp_path / "model.onnx"
        existing.write_bytes(b"data")
        result = ModelManagement.download_file_from_gcs(
            "http://example.com/model.onnx", str(existing)
        )
        assert result == str(existing)

    @patch("qwen3_embed.common.model_management.requests.get")
    def test_403_raises_permission_error(self, mock_get, tmp_path):
        response = Mock()
        response.status_code = 403
        mock_get.return_value = response

        output = tmp_path / "new_model.onnx"
        expected_msg = (
            r"^Authentication Error: You do not have permission to access this resource\. "
            r"Please check your credentials\.$"
        )
        with pytest.raises(PermissionError, match=expected_msg):
            ModelManagement.download_file_from_gcs("http://example.com/x.onnx", str(output))

        assert not output.exists()

    @patch("qwen3_embed.common.model_management.requests.get")
    def test_missing_content_length_prints_warning(self, mock_get, tmp_path, capsys):
        response = Mock()
        response.status_code = 200
        response.headers = {"content-length": "0"}
        response.iter_content.return_value = [b"hello"]
        mock_get.return_value = response

        output = tmp_path / "out.onnx"
        result = ModelManagement.download_file_from_gcs(
            "http://example.com/out.onnx", str(output), show_progress=False
        )
        assert result == str(output)
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    @patch("qwen3_embed.common.model_management.requests.get")
    def test_downloads_file_with_content_length(self, mock_get, tmp_path):
        chunk = b"A" * 1024
        response = Mock()
        response.status_code = 200
        response.headers = {"content-length": str(len(chunk))}
        response.iter_content.return_value = [chunk]
        mock_get.return_value = response

        output = tmp_path / "model.onnx"
        result = ModelManagement.download_file_from_gcs(
            "http://example.com/model.onnx", str(output), show_progress=True
        )
        assert result == str(output)
        assert output.read_bytes() == chunk

    @patch("qwen3_embed.common.model_management.requests.get")
    def test_skips_keepalive_empty_chunks(self, mock_get, tmp_path):
        """Empty chunks must be filtered out (keep-alive frames)."""
        response = Mock()
        response.status_code = 200
        response.headers = {"content-length": "5"}
        response.iter_content.return_value = [b"", b"hello", b""]
        mock_get.return_value = response

        output = tmp_path / "out.onnx"
        ModelManagement.download_file_from_gcs("http://example.com/out.onnx", str(output))
        assert output.read_bytes() == b"hello"

    @patch("qwen3_embed.common.model_management.requests.get")
    def test_show_progress_false_when_no_content_length(self, mock_get, tmp_path):
        """Progress bar disabled when content-length is missing."""
        response = Mock()
        response.status_code = 200
        response.headers = {}
        response.iter_content.return_value = [b"data"]
        mock_get.return_value = response

        output = tmp_path / "out.onnx"
        # Should not raise even with show_progress=True but no content-length
        ModelManagement.download_file_from_gcs(
            "http://example.com/out.onnx", str(output), show_progress=True
        )
        assert output.exists()


# ---------------------------------------------------------------------------
# TestDecompressToCache
# ---------------------------------------------------------------------------


class TestDecompressToCache:
    """Tests for decompress_to_cache method."""

    def test_decompress_nonexistent_file(self, tmp_path):
        nonexistent_file = tmp_path / "nonexistent.tar.gz"
        with pytest.raises(ValueError, match="does not exist or is not a file"):
            ModelManagement.decompress_to_cache(str(nonexistent_file), str(tmp_path))

    def test_decompress_invalid_extension(self, tmp_path):
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.touch()
        with pytest.raises(ValueError, match="is not a .tar.gz file"):
            ModelManagement.decompress_to_cache(str(invalid_file), str(tmp_path))

    def test_decompress_directory(self, tmp_path):
        directory = tmp_path / "directory.tar.gz"
        directory.mkdir()
        with pytest.raises(ValueError, match="does not exist or is not a file"):
            ModelManagement.decompress_to_cache(str(directory), str(tmp_path))

    def test_decompress_corrupted_tar_gz(self, tmp_path):
        corrupted_file = tmp_path / "corrupted.tar.gz"
        corrupted_file.write_text("not a tar file")

        cache_dir = tmp_path / "cache_tmp"
        cache_dir.mkdir()

        with pytest.raises(ValueError, match="An error occurred while decompressing"):
            ModelManagement.decompress_to_cache(str(corrupted_file), str(cache_dir))

    def test_decompress_success(self, tmp_path):
        """Valid .tar.gz extracts successfully and returns cache_dir."""
        tar_path = make_tar_gz(tmp_path, inner_name="model.onnx")
        cache_dir = tmp_path / "out"
        cache_dir.mkdir()

        result = ModelManagement.decompress_to_cache(str(tar_path), str(cache_dir))
        assert result == str(cache_dir)
        assert (cache_dir / "model.onnx").exists()

    def test_decompress_error_with_tmp_in_path_removes_cache(self, tmp_path):
        """Cache dir containing 'tmp' in its path is removed on TarError."""
        cache_dir = tmp_path / "tmp_cache"
        cache_dir.mkdir()

        corrupted = tmp_path / "bad.tar.gz"
        corrupted.write_text("not valid")

        with pytest.raises(ValueError, match="An error occurred while decompressing"):
            ModelManagement.decompress_to_cache(str(corrupted), str(cache_dir))

        assert not cache_dir.exists()


# ---------------------------------------------------------------------------
# TestDownloadFilesFromHuggingFace
# ---------------------------------------------------------------------------


class TestDownloadFilesFromHuggingFace:
    """Tests for download_files_from_huggingface."""

    def _make_repo_files(self) -> list[RepoFile]:
        return [
            make_repo_file("model.onnx", size=500, oid="aaa"),
            make_repo_file("config.json", size=100, oid="bbb"),
        ]

    @patch("qwen3_embed.common.model_management.disable_progress_bars")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_local_files_only_no_metadata(self, mock_snap, mock_disable, tmp_path):
        """local_files_only=True with no metadata file just calls snapshot_download."""
        mock_snap.return_value = str(tmp_path / "result")

        result = ModelManagement.download_files_from_huggingface(
            hf_source_repo="org/repo",
            cache_dir=str(tmp_path),
            extra_patterns=["model.onnx"],
            local_files_only=True,
        )
        mock_disable.assert_called_once()
        mock_snap.assert_called_once()
        assert result == str(tmp_path / "result")

    @patch("qwen3_embed.common.model_management.disable_progress_bars")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_local_files_only_with_valid_metadata(self, mock_snap, mock_disable, tmp_path):
        """local_files_only=True with valid metadata: snapshot_download is called."""
        snapshot_dir = tmp_path / "models--org--repo"
        snapshot_dir.mkdir(parents=True)
        inner_file = snapshot_dir / "model.onnx"
        inner_file.write_bytes(b"x" * 500)

        metadata = {
            "model.onnx": {"size": 500, "blob_id": "aaa"},
        }
        (snapshot_dir / ModelManagement.METADATA_FILE).write_text(json.dumps(metadata))
        mock_snap.return_value = str(snapshot_dir)

        result = ModelManagement.download_files_from_huggingface(
            hf_source_repo="org/repo",
            cache_dir=str(tmp_path),
            extra_patterns=["model.onnx"],
            local_files_only=True,
        )
        mock_snap.assert_called_once()
        assert result == str(snapshot_dir)

    @patch("qwen3_embed.common.model_management.disable_progress_bars")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_local_files_only_with_invalid_metadata_warns(self, mock_snap, mock_disable, tmp_path):
        """local_files_only=True with mismatched file sizes logs a warning but continues."""
        snapshot_dir = tmp_path / "models--org--repo"
        snapshot_dir.mkdir(parents=True)
        inner_file = snapshot_dir / "model.onnx"
        inner_file.write_bytes(b"x" * 200)  # size mismatch

        metadata = {"model.onnx": {"size": 500, "blob_id": "aaa"}}
        (snapshot_dir / ModelManagement.METADATA_FILE).write_text(json.dumps(metadata))
        mock_snap.return_value = str(snapshot_dir)

        # Should NOT raise, just warn
        result = ModelManagement.download_files_from_huggingface(
            hf_source_repo="org/repo",
            cache_dir=str(tmp_path),
            extra_patterns=["model.onnx"],
            local_files_only=True,
        )
        mock_snap.assert_called_once()
        assert result == str(snapshot_dir)

    @patch("qwen3_embed.common.model_management.list_repo_tree")
    @patch("qwen3_embed.common.model_management.model_info")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_online_no_cached_metadata(self, mock_snap, mock_info, mock_tree, tmp_path):
        """Online mode: no prior metadata -> downloads, collects metadata, verifies."""
        snapshot_dir = tmp_path / "models--org--repo"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "model.onnx").write_bytes(b"x" * 500)

        repo_files = [make_repo_file("model.onnx", size=500, oid="aaa")]
        mock_info.return_value = Mock(sha="rev123")
        mock_tree.return_value = repo_files
        mock_snap.return_value = str(snapshot_dir)

        result = ModelManagement.download_files_from_huggingface(
            hf_source_repo="org/repo",
            cache_dir=str(tmp_path),
            extra_patterns=["model.onnx"],
        )
        assert result == str(snapshot_dir)
        meta_file = snapshot_dir / ModelManagement.METADATA_FILE
        assert meta_file.exists()

    @patch("qwen3_embed.common.model_management.list_repo_tree")
    @patch("qwen3_embed.common.model_management.model_info")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_online_cached_metadata_verified(self, mock_snap, mock_info, mock_tree, tmp_path):
        """Online mode with valid cached metadata: disables progress bars."""
        snapshot_dir = tmp_path / "models--org--repo"
        snapshot_dir.mkdir(parents=True)
        inner_file = snapshot_dir / "model.onnx"
        inner_file.write_bytes(b"x" * 500)

        repo_files = [make_repo_file("model.onnx", size=500, oid="aaa")]
        metadata = {"model.onnx": {"size": 500, "blob_id": "aaa"}}
        (snapshot_dir / ModelManagement.METADATA_FILE).write_text(json.dumps(metadata))

        mock_info.return_value = Mock(sha="rev123")
        mock_tree.return_value = repo_files
        mock_snap.return_value = str(snapshot_dir)

        with patch("qwen3_embed.common.model_management.disable_progress_bars") as mock_dis:
            result = ModelManagement.download_files_from_huggingface(
                hf_source_repo="org/repo",
                cache_dir=str(tmp_path),
                extra_patterns=["model.onnx"],
            )
        mock_dis.assert_called_once()
        assert result == str(snapshot_dir)

    @patch("qwen3_embed.common.model_management.list_repo_tree")
    @patch("qwen3_embed.common.model_management.model_info")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_online_empty_repo_tree(self, mock_snap, mock_info, mock_tree, tmp_path):
        """Online mode with empty repo tree: treats repo_files as empty list."""
        snapshot_dir = tmp_path / "models--org--repo"
        snapshot_dir.mkdir(parents=True)

        mock_info.return_value = Mock(sha="rev123")
        mock_tree.return_value = []
        mock_snap.return_value = str(snapshot_dir)

        result = ModelManagement.download_files_from_huggingface(
            hf_source_repo="org/repo",
            cache_dir=str(tmp_path),
            extra_patterns=["model.onnx"],
        )
        assert result == str(snapshot_dir)

    @patch("qwen3_embed.common.model_management.list_repo_tree")
    @patch("qwen3_embed.common.model_management.model_info")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_online_download_failure_raises_value_error(
        self, mock_snap, mock_info, mock_tree, tmp_path
    ):
        """If post-download offline verification fails (size mismatch), raises ValueError."""
        snapshot_dir = tmp_path / "models--org--repo"
        snapshot_dir.mkdir(parents=True)
        # File exists but with wrong size -> offline verify will fail
        (snapshot_dir / "model.onnx").write_bytes(b"x" * 100)

        repo_files = [make_repo_file("model.onnx", size=500, oid="aaa")]
        mock_info.return_value = Mock(sha="rev123")
        mock_tree.return_value = repo_files
        mock_snap.return_value = str(snapshot_dir)

        with pytest.raises(ValueError, match="Files have been corrupted"):
            ModelManagement.download_files_from_huggingface(
                hf_source_repo="org/repo",
                cache_dir=str(tmp_path),
                extra_patterns=["model.onnx"],
            )

    @patch("qwen3_embed.common.model_management.list_repo_tree")
    @patch("qwen3_embed.common.model_management.model_info")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_metadata_save_oserror_is_swallowed(self, mock_snap, mock_info, mock_tree, tmp_path):
        """OSError while saving metadata is logged but does not raise."""
        snapshot_dir = tmp_path / "models--org--repo"
        snapshot_dir.mkdir(parents=True)
        inner_file = snapshot_dir / "model.onnx"
        inner_file.write_bytes(b"x" * 500)

        repo_files = [make_repo_file("model.onnx", size=500, oid="aaa")]
        mock_info.return_value = Mock(sha="rev123")
        mock_tree.return_value = repo_files
        mock_snap.return_value = str(snapshot_dir)

        # Patch Path.write_text to raise OSError on metadata write
        original_write_text = Path.write_text

        def patched_write_text(self, data, *args, **kwargs):
            if self.name == ModelManagement.METADATA_FILE:
                raise OSError("disk full")
            return original_write_text(self, data, *args, **kwargs)

        with patch.object(Path, "write_text", patched_write_text):
            # Should not raise
            result = ModelManagement.download_files_from_huggingface(
                hf_source_repo="org/repo",
                cache_dir=str(tmp_path),
                extra_patterns=["model.onnx"],
            )
        assert result == str(snapshot_dir)

    @patch("qwen3_embed.common.model_management.list_repo_tree")
    @patch("qwen3_embed.common.model_management.model_info")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_verify_files_oserror_returns_false(self, mock_snap, mock_info, mock_tree, tmp_path):
        """KeyError in metadata causes _verify_files_from_metadata to return False."""
        snapshot_dir = tmp_path / "models--org--repo"
        snapshot_dir.mkdir(parents=True)

        # metadata with broken entry (missing 'size' key) -> KeyError in verify
        metadata = {"model.onnx": {"blob_id": "aaa"}}  # 'size' key missing
        (snapshot_dir / ModelManagement.METADATA_FILE).write_text(json.dumps(metadata))
        inner_file = snapshot_dir / "model.onnx"
        inner_file.write_bytes(b"x" * 500)

        repo_files = [make_repo_file("model.onnx", size=500, oid="aaa")]
        mock_info.return_value = Mock(sha="rev123")
        mock_tree.return_value = repo_files
        mock_snap.return_value = str(snapshot_dir)

        # With broken metadata, verified_metadata=False -> re-collects + re-verifies
        # Since the actual file matches (size 500), final offline verify passes
        result = ModelManagement.download_files_from_huggingface(
            hf_source_repo="org/repo",
            cache_dir=str(tmp_path),
            extra_patterns=["model.onnx"],
        )
        assert result == str(snapshot_dir)


# ---------------------------------------------------------------------------
# TestRetrieveModelGcs
# ---------------------------------------------------------------------------


class TestRetrieveModelGcs:
    """Tests for retrieve_model_gcs."""

    def test_returns_existing_populated_model_dir(self, tmp_path):
        """If model_dir already exists and has files, return it without downloading."""
        model_dir = tmp_path / "name"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_bytes(b"data")

        with patch.object(ModelManagement, "download_file_from_gcs") as mock_dl:
            result = ModelManagement.retrieve_model_gcs(
                model_name="model/name",
                source_url="http://example.com/model.tar.gz",
                cache_dir=str(tmp_path),
            )
        mock_dl.assert_not_called()
        assert result == model_dir

    def test_local_files_only_raises_value_error(self, tmp_path):
        """local_files_only=True when model dir is absent raises ValueError."""
        with pytest.raises(ValueError, match="local_files_only=True"):
            ModelManagement.retrieve_model_gcs(
                model_name="test/model",
                source_url="http://example.com/model.tar.gz",
                cache_dir=str(tmp_path),
                local_files_only=True,
            )

    def test_removes_stale_tmp_dir_before_download(self, tmp_path):
        """Pre-existing model_tmp_dir is removed before download attempt."""
        model_tmp_dir = tmp_path / "tmp" / "model"
        model_tmp_dir.mkdir(parents=True)
        (model_tmp_dir / "stale.onnx").write_bytes(b"stale")

        def fake_download(url: str, output_path: str, **kwargs) -> str:
            Path(output_path).write_bytes(b"fake tar")
            return output_path

        def fake_decompress(targz_path: str, cache_dir: str) -> str:
            # Simulate extraction creating model_tmp_dir again
            out = Path(cache_dir) / "model"
            out.mkdir(parents=True, exist_ok=True)
            (out / "model.onnx").write_bytes(b"fresh")
            return cache_dir

        with (
            patch.object(ModelManagement, "download_file_from_gcs", side_effect=fake_download),
            patch.object(ModelManagement, "decompress_to_cache", side_effect=fake_decompress),
        ):
            result = ModelManagement.retrieve_model_gcs(
                model_name="model",
                source_url="http://example.com/model.tar.gz",
                cache_dir=str(tmp_path),
            )
        assert result == tmp_path / "model"

    def test_deletes_tar_gz_after_extraction(self, tmp_path):
        """The downloaded .tar.gz is removed after extraction."""
        model_name = "mymodel"

        def fake_download(url: str, output_path: str, **kwargs) -> str:
            Path(output_path).write_bytes(b"fake tar")
            return output_path

        def fake_decompress(targz_path: str, cache_dir: str) -> str:
            out = Path(cache_dir) / model_name
            out.mkdir(parents=True, exist_ok=True)
            (out / "model.onnx").write_bytes(b"data")
            return cache_dir

        with (
            patch.object(ModelManagement, "download_file_from_gcs", side_effect=fake_download),
            patch.object(ModelManagement, "decompress_to_cache", side_effect=fake_decompress),
        ):
            result = ModelManagement.retrieve_model_gcs(
                model_name=model_name,
                source_url="http://example.com/mymodel.tar.gz",
                cache_dir=str(tmp_path),
            )
        tar_gz = tmp_path / f"{model_name}.tar.gz"
        assert not tar_gz.exists()
        assert result == tmp_path / model_name

    def test_raises_if_tmp_dir_missing_after_extraction(self, tmp_path):
        """If model_tmp_dir is absent after decompress, raises ValueError."""
        with (
            patch.object(ModelManagement, "download_file_from_gcs"),
            patch.object(
                ModelManagement, "decompress_to_cache", return_value=str(tmp_path / "tmp")
            ),
            pytest.raises(ValueError, match="Could not find"),
        ):
            ModelManagement.retrieve_model_gcs(
                model_name="missing",
                source_url="http://example.com/missing.tar.gz",
                cache_dir=str(tmp_path),
            )

    def test_deprecated_tar_struct_prefixes_fast(self, tmp_path):
        """deprecated_tar_struct=True prepends 'fast-' to the model dir name."""
        model_name = "org/mymodel"

        def fake_download(url: str, output_path: str, **kwargs) -> str:
            Path(output_path).write_bytes(b"fake tar")
            return output_path

        def fake_decompress(targz_path: str, cache_dir: str) -> str:
            out = Path(cache_dir) / "fast-mymodel"
            out.mkdir(parents=True, exist_ok=True)
            (out / "model.onnx").write_bytes(b"data")
            return cache_dir

        with (
            patch.object(ModelManagement, "download_file_from_gcs", side_effect=fake_download),
            patch.object(ModelManagement, "decompress_to_cache", side_effect=fake_decompress),
        ):
            result = ModelManagement.retrieve_model_gcs(
                model_name=model_name,
                source_url="http://example.com/fast-mymodel.tar.gz",
                cache_dir=str(tmp_path),
                deprecated_tar_struct=True,
            )
        assert result.name == "fast-mymodel"


# ---------------------------------------------------------------------------
# TestDownloadModel
# ---------------------------------------------------------------------------


class TestDownloadModel:
    """Tests for download_model."""

    def test_specific_model_path_returns_immediately(self, tmp_path):
        """specific_model_path kwarg bypasses all download logic."""
        model = make_model_description()
        result = ModelManagement.download_model(
            model, cache_dir=str(tmp_path), specific_model_path="/custom/path"
        )
        assert result == Path("/custom/path")

    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    def test_hf_source_cache_hit_returns_path(self, mock_enable, tmp_path):
        """HF local cache hit returns immediately without retry loop."""
        model = make_model_description(hf="org/repo")

        with patch.object(
            ModelManagement,
            "download_files_from_huggingface",
            return_value=str(tmp_path / "cached"),
        ) as mock_hf:
            result = ModelManagement.download_model(model, cache_dir=str(tmp_path))

        mock_hf.assert_called_once()
        assert result == Path(tmp_path / "cached")
        mock_enable.assert_called()

    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    def test_hf_source_cache_miss_falls_through_to_retry(self, mock_enable, tmp_path):
        """HF local cache miss leads to online retry, then success."""
        model = make_model_description(hf="org/repo")
        online_path = str(tmp_path / "online")

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("local_files_only"):
                raise Exception("not in cache")
            return online_path

        with patch.object(
            ModelManagement, "download_files_from_huggingface", side_effect=side_effect
        ):
            result = ModelManagement.download_model(model, cache_dir=str(tmp_path), retries=1)

        assert result == Path(online_path)
        assert call_count == 2

    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    def test_hf_online_fails_falls_back_to_url(self, mock_enable, tmp_path):
        """HF online failure falls back to GCS/URL source."""
        model = make_model_description(hf="org/repo", url="http://example.com/model.tar.gz")
        gcs_path = tmp_path / "gcs_model"

        def hf_side_effect(*args, **kwargs):
            raise OSError("network error")

        with (
            patch.object(
                ModelManagement, "download_files_from_huggingface", side_effect=hf_side_effect
            ),
            patch.object(ModelManagement, "retrieve_model_gcs", return_value=gcs_path) as mock_gcs,
        ):
            result = ModelManagement.download_model(model, cache_dir=str(tmp_path), retries=1)

        mock_gcs.assert_called_once()
        assert result == gcs_path

    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    def test_url_only_source_uses_gcs(self, mock_enable, tmp_path):
        """Model with only url source uses retrieve_model_gcs directly."""
        model = make_model_description(hf=None, url="http://example.com/model.tar.gz")
        gcs_path = tmp_path / "gcs_model"

        with patch.object(
            ModelManagement, "retrieve_model_gcs", return_value=gcs_path
        ) as mock_gcs:
            result = ModelManagement.download_model(model, cache_dir=str(tmp_path), retries=1)

        mock_gcs.assert_called_once()
        assert result == gcs_path

    @patch("qwen3_embed.common.model_management.time.sleep")
    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    def test_all_sources_fail_raises_value_error(self, mock_enable, mock_sleep, tmp_path):
        """All retry attempts exhausted -> ValueError."""
        model = make_model_description(hf="org/repo", url="http://example.com/model.tar.gz")

        with (
            patch.object(
                ModelManagement,
                "download_files_from_huggingface",
                side_effect=OSError("network error"),
            ),
            patch.object(
                ModelManagement, "retrieve_model_gcs", side_effect=Exception("gcs error")
            ),
            pytest.raises(ValueError, match="Could not load model"),
        ):
            ModelManagement.download_model(model, cache_dir=str(tmp_path), retries=2)

    @patch("qwen3_embed.common.model_management.time.sleep")
    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    def test_sleep_called_between_retries(self, mock_enable, mock_sleep, tmp_path):
        """time.sleep is called between failed retry attempts."""
        model = make_model_description(hf="org/repo", url="http://example.com/model.tar.gz")

        with (
            patch.object(
                ModelManagement,
                "download_files_from_huggingface",
                side_effect=OSError("fail"),
            ),
            patch.object(ModelManagement, "retrieve_model_gcs", side_effect=Exception("fail")),
            pytest.raises(ValueError),
        ):
            ModelManagement.download_model(model, cache_dir=str(tmp_path), retries=2)

        assert mock_sleep.call_count >= 1

    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    def test_local_files_only_breaks_on_failure(self, mock_enable, tmp_path):
        """local_files_only=True does not retry and raises ValueError."""
        model = make_model_description(hf="org/repo", url="http://example.com/model.tar.gz")

        with (
            patch.object(
                ModelManagement,
                "download_files_from_huggingface",
                side_effect=Exception("not cached"),
            ),
            patch.object(
                ModelManagement,
                "retrieve_model_gcs",
                side_effect=Exception("not cached"),
            ),
            pytest.raises(ValueError, match="Could not load model"),
        ):
            ModelManagement.download_model(model, cache_dir=str(tmp_path), local_files_only=True)

    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    def test_repository_not_found_error_handled(self, mock_enable, tmp_path):
        """RepositoryNotFoundError in HF online -> falls back to GCS."""
        model = make_model_description(hf="org/repo", url="http://example.com/model.tar.gz")
        gcs_path = tmp_path / "gcs_model"

        def hf_side_effect(*args, **kwargs):
            if kwargs.get("local_files_only"):
                raise Exception("not cached")
            raise RepositoryNotFoundError("not found", response=MagicMock())

        with (
            patch.object(
                ModelManagement, "download_files_from_huggingface", side_effect=hf_side_effect
            ),
            patch.object(ModelManagement, "retrieve_model_gcs", return_value=gcs_path) as mock_gcs,
        ):
            result = ModelManagement.download_model(model, cache_dir=str(tmp_path), retries=1)

        mock_gcs.assert_called_once()
        assert result == gcs_path

    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    def test_extra_patterns_built_from_model_file_and_additional_files(
        self, mock_enable, tmp_path
    ):
        """extra_patterns includes model_file and additional_files."""
        model = make_model_description(
            hf="org/repo", model_file="weights.onnx", additional_files=["vocab.txt"]
        )
        captured_patterns: list[list[str]] = []

        def capture_hf(*args, **kwargs):
            captured_patterns.append(kwargs.get("extra_patterns", []))
            return str(tmp_path / "cached")

        with patch.object(
            ModelManagement, "download_files_from_huggingface", side_effect=capture_hf
        ):
            ModelManagement.download_model(model, cache_dir=str(tmp_path))

        patterns = captured_patterns[0]
        assert "weights.onnx" in patterns
        assert "vocab.txt" in patterns
