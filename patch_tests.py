import re

with open("tests/test_model_management.py", "r") as f:
    content = f.read()

# Replace test_decompress_error_with_tmp_in_path_removes_cache
pattern = re.compile(
    r"    def test_decompress_error_with_tmp_in_path_removes_cache\(self, tmp_path\):\n"
    r"        \"\"\"Cache dir containing 'tmp' in its path is removed on TarError\.\"\"\"\n"
    r"        cache_dir = tmp_path / \"tmp_cache\"\n"
    r"        cache_dir\.mkdir\(\)\n"
    r"\n"
    r"        corrupted = tmp_path / \"bad\.tar\.gz\"\n"
    r"        corrupted\.write_text\(\"not valid\"\)\n"
    r"\n"
    r"        with pytest\.raises\(ValueError, match=\"An error occurred while decompressing\"\):\n"
    r"            ModelManagement\.decompress_to_cache\(str\(corrupted\), str\(cache_dir\)\)"
)

replacement = """    def test_decompress_error_removes_cache_unconditionally(self, tmp_path):
        \"\"\"Cache dir is removed unconditionally on TarError.\"\"\"
        cache_dir = tmp_path / "cache_dir"
        cache_dir.mkdir()

        tar_path = make_tar_gz(tmp_path, inner_name="model.onnx")

        with (
            patch.object(tarfile.TarFile, "extractall", side_effect=tarfile.TarError("Mid-extraction failure")),
            pytest.raises(ValueError, match="An error occurred while decompressing")
        ):
            ModelManagement.decompress_to_cache(str(tar_path), str(cache_dir))

        assert not cache_dir.exists()"""

new_content = pattern.sub(replacement, content)

with open("tests/test_model_management.py", "w") as f:
    f.write(new_content)
