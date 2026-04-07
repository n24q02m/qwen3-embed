from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError
import os

try:
    snapshot_download(repo_id="non-existent/repo", local_files_only=True)
except LocalEntryNotFoundError:
    print("Caught LocalEntryNotFoundError")
except Exception as e:
    print(f"Caught other exception: {type(e).__name__}")
