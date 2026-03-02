with open("tests/test_model_management.py") as f:
    content = f.read()

# Replace nonexistent file test
old_str_1 = '    def test_decompress_nonexistent_file(self, tmp_path):\n        nonexistent_file = tmp_path / "nonexistent.tar.gz"\n        with pytest.raises(ValueError, match="does not exist or is not a file"):'
new_str_1 = '    def test_decompress_nonexistent_file(self, tmp_path):\n        import re\n        nonexistent_file = tmp_path / "nonexistent.tar.gz"\n        with pytest.raises(ValueError, match=re.escape(f"{nonexistent_file} does not exist or is not a file.")):'
content = content.replace(old_str_1, new_str_1)

# Replace directory test
old_str_2 = '    def test_decompress_directory(self, tmp_path):\n        directory = tmp_path / "directory.tar.gz"\n        directory.mkdir()\n        with pytest.raises(ValueError, match="does not exist or is not a file"):'
new_str_2 = '    def test_decompress_directory(self, tmp_path):\n        import re\n        directory = tmp_path / "directory.tar.gz"\n        directory.mkdir()\n        with pytest.raises(ValueError, match=re.escape(f"{directory} does not exist or is not a file.")):'
content = content.replace(old_str_2, new_str_2)

with open("tests/test_model_management.py", "w") as f:
    f.write(content)
