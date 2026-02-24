import json

from qwen3_embed.common.model_management import ModelManagement


# Mock RepoFile since we don't want to rely on external library details for unit testing logic
class MockRepoFile:
    def __init__(self, path, size, blob_id):
        self.path = path
        self.size = size
        self.blob_id = blob_id


class TestModelManagementInternals:
    def test_verify_files_from_metadata_offline_success(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        file_path = model_dir / "config.json"
        file_path.write_text("content")

        metadata = {"config.json": {"size": len("content"), "blob_id": "123"}}

        # Verify offline (repo_files=[])
        result = ModelManagement._verify_files_from_metadata(model_dir, metadata, repo_files=[])
        assert result is True

    def test_verify_files_from_metadata_offline_failure_missing(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        # File missing
        metadata = {"config.json": {"size": 10, "blob_id": "123"}}
        result = ModelManagement._verify_files_from_metadata(model_dir, metadata, repo_files=[])
        assert result is False

    def test_verify_files_from_metadata_offline_failure_size(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        file_path = model_dir / "config.json"
        file_path.write_text("content")

        metadata = {"config.json": {"size": len("content") + 1, "blob_id": "123"}}

        result = ModelManagement._verify_files_from_metadata(model_dir, metadata, repo_files=[])
        assert result is False

    def test_verify_files_from_metadata_online_success(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        file_path = model_dir / "config.json"
        file_path.write_text("content")

        metadata = {"config.json": {"size": len("content"), "blob_id": "blob_1"}}

        repo_files = [MockRepoFile(path="config.json", size=len("content"), blob_id="blob_1")]

        result = ModelManagement._verify_files_from_metadata(
            model_dir, metadata, repo_files=repo_files
        )
        assert result is True

    def test_verify_files_from_metadata_online_failure_blob_id(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        file_path = model_dir / "config.json"
        file_path.write_text("content")

        metadata = {"config.json": {"size": len("content"), "blob_id": "blob_1"}}

        repo_files = [MockRepoFile(path="config.json", size=len("content"), blob_id="blob_2")]

        result = ModelManagement._verify_files_from_metadata(
            model_dir, metadata, repo_files=repo_files
        )
        assert result is False

    def test_collect_file_metadata(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        file_path = model_dir / "config.json"
        file_path.write_text("content")

        # Create metadata file which should be ignored (assuming METADATA_FILE is files_metadata.json)
        (model_dir / "files_metadata.json").write_text("ignore me")

        repo_files = [MockRepoFile(path="config.json", size=len("content"), blob_id="blob_1")]

        # We need to ensure the extracted method uses the class attribute METADATA_FILE
        # Since we haven't extracted it yet, this test will fail if we run it now unless we implement it or stub it.
        # But for now, we write the test assuming the method exists.

        meta = ModelManagement._collect_file_metadata(model_dir, repo_files)

        assert "config.json" in meta
        assert meta["config.json"]["size"] == len("content")
        assert meta["config.json"]["blob_id"] == "blob_1"
        assert "files_metadata.json" not in meta

    def test_save_file_metadata(self, tmp_path):
        model_dir = tmp_path / "model"
        # Directory doesn't exist yet, should be created

        meta = {"config.json": {"size": 123, "blob_id": "blob_1"}}

        ModelManagement._save_file_metadata(model_dir, meta)

        metadata_file = model_dir / "files_metadata.json"
        assert metadata_file.exists()
        loaded_meta = json.loads(metadata_file.read_text())
        assert loaded_meta == meta
