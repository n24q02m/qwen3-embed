import sys
import os

with open('qwen3_embed/common/model_management.py', 'r') as f:
    lines = f.readlines()

# Update imports
for i, line in enumerate(lines):
    if line.startswith('from typing import Any, Generic, TypeVar'):
        lines[i] = line.replace('Any, Generic, TypeVar', 'Any, Generator, Generic, Iterable, TypeVar')
        break

# Find _validate_tar_member end and decompress_to_cache start
validate_end = -1
for i, line in enumerate(lines):
    if 'def _validate_tar_member' in line:
        # Search for the end of the method (next @classmethod or def)
        for j in range(i + 1, len(lines)):
            if lines[j].startswith('    @classmethod') or lines[j].startswith('    def '):
                validate_end = j
                break
        break

if validate_end == -1:
    print("Could not find end of _validate_tar_member")
    sys.exit(1)

new_methods = [
    '    @classmethod\n',
    '    def _get_safe_members(\n',
    '        cls, tar: tarfile.TarFile, target_dir: str\n',
    '    ) -> Generator[tarfile.TarInfo, None, None]:\n',
    '        total_size = 0\n',
    '        max_uncompressed_size = 20 * 1024 * 1024 * 1024  # 20 GB\n',
    '        for member in tar:\n',
    '            cls._validate_tar_member(member, target_dir)\n',
    '            total_size += member.size\n',
    '            if total_size > max_uncompressed_size:\n',
    '                raise tarfile.TarError(\n',
    '                    f"Decompression bomb detected: total uncompressed size exceeds {max_uncompressed_size} bytes"\n',
    '                )\n',
    '            yield member\n',
    '\n',
    '    @classmethod\n',
    '    def _manual_extract(\n',
    '        cls, tar: tarfile.TarFile, cache_dir: str, members: Iterable[tarfile.TarInfo]\n',
    '    ) -> None:\n',
    '        for member in members:\n',
    '            # Sanitize metadata to mimic "data" filter\n',
    '            member.mode &= 0o777\n',
    '            member.uid = 0\n',
    '            member.gid = 0\n',
    '            member.uname = ""\n',
    '            member.gname = ""\n',
    '            tar.extract(member, path=cache_dir)\n',
    '\n'
]

lines[validate_end:validate_end] = new_methods

# Find decompress_to_cache and replace its body
start = -1
end = -1
for i, line in enumerate(lines):
    if 'def decompress_to_cache' in line:
        start = i
        # Find the end of the method
        for j in range(i + 1, len(lines)):
            if lines[j].startswith('    @classmethod') or lines[j].startswith('    def '):
                end = j
                break
        break

if start == -1 or end == -1:
    print("Could not find decompress_to_cache")
    sys.exit(1)

new_decompress_body = [
    '    def decompress_to_cache(cls, targz_path: str, cache_dir: str) -> str:\n',
    '        """\n',
    '        Decompresses a .tar.gz file to a cache directory.\n',
    '\n',
    '        Args:\n',
    '            targz_path (str): Path to the .tar.gz file.\n',
    '            cache_dir (str): Path to the cache directory.\n',
    '\n',
    '        Returns:\n',
    '            cache_dir (str): Path to the cache directory.\n',
    '        """\n',
    '        # Check if targz_path exists and is a file\n',
    '        if not os.path.isfile(targz_path):\n',
    '            raise ValueError(f"{targz_path} does not exist or is not a file.")\n',
    '\n',
    '        # Check if targz_path is a .tar.gz file\n',
    '        if not targz_path.endswith(".tar.gz"):\n',
    '            raise ValueError(f"{targz_path} is not a .tar.gz file.")\n',
    '\n',
    '        try:\n',
    '            # Open the tar.gz file\n',
    '            with tarfile.open(targz_path, "r:gz") as tar:\n',
    '                # Extract all files into the cache directory securely\n',
    '                target_dir = os.path.abspath(cache_dir)\n',
    '                safe_members = cls._get_safe_members(tar, target_dir)\n',
    '\n',
    '                if hasattr(tarfile, "data_filter"):\n',
    '                    tar.extractall(path=cache_dir, members=safe_members, filter="data")\n',
    '                else:\n',
    '                    cls._manual_extract(tar, cache_dir, safe_members)\n',
    '        except tarfile.TarError as e:\n',
    '            # If decompression fails, remove the partially extracted directory\n',
    '            shutil.rmtree(cache_dir, ignore_errors=True)\n',
    '            logger.error(f"Failed to decompress {targz_path}: {e}")\n',
    '            raise e\n',
    '\n',
    '        return cache_dir\n',
    '\n'
]

lines[start:end] = new_decompress_body

with open('qwen3_embed/common/model_management.py', 'w') as f:
    f.writelines(lines)
