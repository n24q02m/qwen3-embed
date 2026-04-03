import base64
import contextlib
import hashlib
import json
import os
import shutil
import tarfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Generic, TypeVar

import requests
from huggingface_hub import list_repo_tree, model_info, snapshot_download
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    disable_progress_bars,
    enable_progress_bars,
)
from loguru import logger
from tqdm import tqdm

from qwen3_embed.common.model_description import BaseModelDescription

T = TypeVar("T", bound=BaseModelDescription)


class ModelManagement(Generic[T]):
    METADATA_FILE = "files_metadata.json"

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """Lists the supported models.

        Returns:
            list[T]: A list of dictionaries containing the model information.
        """
        raise NotImplementedError()

    @classmethod
    def add_custom_model(
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add a custom model to the existing embedding classes based on the passed model descriptions

        Model description dict should contain the fields same as in one of the model descriptions presented
         in qwen3_embed.common.model_description

         E.g. for BaseModelDescription:
              model: str
              sources: ModelSource
              model_file: str
              description: str
              license: str
              size_in_GB: float
              additional_files: list[str]

        Returns:
            None
        """
        raise NotImplementedError()

    @classmethod
    def _list_supported_models(cls) -> list[T]:
        raise NotImplementedError()

    @classmethod
    def _get_model_description(cls, model_name: str) -> T:
        """
        Gets the model description from the model_name.

        Args:
            model_name (str): The name of the model.

        raises:
            ValueError: If the model_name is not supported.

        Returns:
            T: The model description.
        """
        model_name_lower = model_name.lower()
        for model in cls._list_supported_models():
            if model_name_lower == model.model.lower():
                return model

        raise ValueError(f"Model {model_name} is not supported in {cls.__name__}.")

    @staticmethod
    def _get_expected_md5(headers: Any) -> str | None:
        if "x-goog-hash" in headers:
            x_goog_hash = headers["x-goog-hash"]
            for part in x_goog_hash.split(","):
                part = part.strip()
                if part.startswith("md5="):
                    return base64.b64decode(part[4:]).hex()
        return None

    @staticmethod
    def _download_and_hash_file(
        response: Any, output_path: str, total_size_in_bytes: int, show_progress: bool
    ) -> str:
        show_progress = bool(total_size_in_bytes and show_progress)
        md5_hash = hashlib.md5(
            usedforsecurity=False
        )  # SECURITY: MD5 is used solely for non-cryptographic file integrity checking (GCS checksums).
        with (
            tqdm(
                total=total_size_in_bytes,
                unit="iB",
                unit_scale=True,
                disable=not show_progress,
                desc=f"Downloading {Path(output_path).name}",
            ) as progress_bar,
            open(output_path, "wb") as file,
        ):
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # Filter out keep-alive new chunks
                    progress_bar.update(len(chunk))
                    file.write(chunk)
                    md5_hash.update(chunk)
        return md5_hash.hexdigest()

    @classmethod
    def download_file_from_gcs(cls, url: str, output_path: str, show_progress: bool = True) -> str:
        """
        Downloads a file from Google Cloud Storage.

        Args:
            url (str): The URL to download the file from.
            output_path (str): The path to save the downloaded file to.
            show_progress (bool, optional): Whether to show a progress bar. Defaults to True.

        Returns:
            str: The path to the downloaded file.
        """

        if not url.startswith(
            ("http://storage.googleapis.com/", "https://storage.googleapis.com/")
        ):
            raise ValueError(
                f"Invalid URL: {url}. Only URLs from Google Cloud Storage are allowed."
            )
        if os.path.exists(output_path):
            return output_path
        # SECURITY: Explicitly enforce TLS verification to prevent accidental or malicious bypass via environment variables (like REQUESTS_CA_BUNDLE).
        response = requests.get(url, stream=True, timeout=10, verify=True)

        # Handle HTTP errors
        if response.status_code == 403:
            raise PermissionError(
                "Authentication Error: You do not have permission to access this resource. "
                "Please check your credentials."
            )
        response.raise_for_status()

        expected_md5 = cls._get_expected_md5(response.headers)

        total_size_in_bytes = int(response.headers.get("content-length", 0))
        if total_size_in_bytes == 0:
            logger.warning(f"Content-length header is missing or zero in the response from {url}.")

        calculated_md5 = cls._download_and_hash_file(
            response, output_path, total_size_in_bytes, show_progress
        )

        if expected_md5 and expected_md5 != calculated_md5:
            os.remove(output_path)
            raise ValueError(
                f"File integrity check failed: expected MD5 {expected_md5}, got {calculated_md5}"
            )

        return output_path

    @classmethod
    def _verify_files_from_metadata(
        cls, model_dir: Path, stored_metadata: dict[str, Any], repo_files: list[RepoFile]
    ) -> bool:
        try:
            for rel_path, meta in stored_metadata.items():
                file_path = model_dir / rel_path

                if not file_path.exists():
                    return False

                if repo_files:  # online verification
                    file_info = next((f for f in repo_files if f.path == file_path.name), None)
                    if (
                        not file_info
                        or file_info.size != meta["size"]
                        or file_info.blob_id != meta["blob_id"]
                    ):
                        return False

                else:  # offline verification
                    if file_path.stat().st_size != meta["size"]:
                        return False
            return True
        except (OSError, KeyError) as e:
            logger.error(f"Error verifying files: {str(e)}")
            return False

    @classmethod
    def _collect_file_metadata(
        cls, model_dir: Path, repo_files: list[RepoFile]
    ) -> dict[str, dict[str, int | str]]:
        meta: dict[str, dict[str, int | str]] = {}
        file_info_map = {f.path: f for f in repo_files}
        for file_path in model_dir.rglob("*"):
            if file_path.is_file() and file_path.name != cls.METADATA_FILE:
                repo_file = file_info_map.get(file_path.name)
                if repo_file:
                    meta[str(file_path.relative_to(model_dir))] = {
                        "size": repo_file.size,
                        "blob_id": repo_file.blob_id,
                    }
        return meta

    @classmethod
    def _save_file_metadata(cls, model_dir: Path, meta: dict[str, dict[str, int | str]]) -> None:
        try:
            if not model_dir.exists():
                model_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            (model_dir / cls.METADATA_FILE).write_text(json.dumps(meta))
        except (OSError, ValueError) as e:
            logger.exception(e)
            logger.warning("Failed to save metadata file. Next load may take longer to verify.")

    @classmethod
    def download_files_from_huggingface(
        cls,
        hf_source_repo: str,
        cache_dir: str,
        extra_patterns: list[str],
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Downloads a model from HuggingFace Hub.
        Args:
            hf_source_repo (str): Name of the model on HuggingFace Hub, e.g. "qdrant/all-MiniLM-L6-v2-onnx".
            cache_dir (Optional[str]): The path to the cache directory.
            extra_patterns (list[str]): extra patterns to allow in the snapshot download, typically
                includes the required model files.
            local_files_only (bool, optional): Whether to only use local files. Defaults to False.
        Returns:
            Path: The path to the model directory.
        """

        allow_patterns = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "preprocessor_config.json",
        ]

        allow_patterns.extend(extra_patterns)

        snapshot_dir = Path(cache_dir) / f"models--{hf_source_repo.replace('/', '--')}"
        metadata_file = snapshot_dir / cls.METADATA_FILE

        if local_files_only:
            disable_progress_bars()
            if metadata_file.exists():
                metadata = json.loads(metadata_file.read_text())
                verified = cls._verify_files_from_metadata(snapshot_dir, metadata, repo_files=[])
                if not verified:
                    logger.warning(
                        "Local file sizes do not match the metadata."
                    )  # do not raise, still make an attempt to load the model
            result = snapshot_download(  # nosec B615
                repo_id=hf_source_repo,
                allow_patterns=allow_patterns,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                **kwargs,
            )
            return result

        repo_revision = model_info(hf_source_repo).sha
        repo_tree = list(list_repo_tree(hf_source_repo, revision=repo_revision, repo_type="model"))

        allowed_extensions = {".json", ".onnx", ".gguf", ".txt"}
        repo_files = (
            [
                f
                for f in repo_tree
                if isinstance(f, RepoFile) and Path(f.path).suffix in allowed_extensions
            ]
            if repo_tree
            else []
        )

        verified_metadata = False

        if snapshot_dir.exists() and metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
            verified_metadata = cls._verify_files_from_metadata(snapshot_dir, metadata, repo_files)

        if verified_metadata:
            disable_progress_bars()

        result = snapshot_download(
            repo_id=hf_source_repo,
            allow_patterns=allow_patterns,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=repo_revision,
            **kwargs,
        )

        if (
            not verified_metadata
        ):  # metadata is not up-to-date, update it and check whether the files have been
            # downloaded correctly
            metadata = cls._collect_file_metadata(snapshot_dir, repo_files)

            download_successful = cls._verify_files_from_metadata(
                snapshot_dir, metadata, repo_files=[]
            )
            if not download_successful:
                raise ValueError(
                    "Files have been corrupted during downloading process. "
                    "Please check your internet connection and try again."
                )
            cls._save_file_metadata(snapshot_dir, metadata)

        return result

    @classmethod
    def decompress_to_cache(cls, targz_path: str, cache_dir: str) -> str:
        """
        Decompresses a .tar.gz file to a cache directory.

        Args:
            targz_path (str): Path to the .tar.gz file.
            cache_dir (str): Path to the cache directory.

        Returns:
            cache_dir (str): Path to the cache directory.
        """
        # Check if targz_path exists and is a file
        if not os.path.isfile(targz_path):
            raise ValueError(f"{targz_path} does not exist or is not a file.")

        # Check if targz_path is a .tar.gz file
        if not targz_path.endswith(".tar.gz"):
            raise ValueError(f"{targz_path} is not a .tar.gz file.")

        try:
            # Open the tar.gz file
            with tarfile.open(targz_path, "r:gz") as tar:
                # Extract all files into the cache directory securely
                target_dir = os.path.abspath(cache_dir)

                safe_members = []
                for member in tar.getmembers():
                    if os.path.isabs(member.name) or member.name.startswith("/"):
                        raise tarfile.TarError(
                            f"Attempted path traversal in tar file: {member.name}"
                        )
                    member_path = os.path.abspath(os.path.join(target_dir, member.name))
                    if (
                        not member_path.startswith(target_dir + os.sep)
                        and member_path != target_dir
                    ):
                        raise tarfile.TarError(
                            f"Attempted path traversal in tar file: {member.name}"
                        )
                    # SECURITY: Validate symlink and hardlink targets to prevent
                    # arbitrary file writes outside the extraction directory.
                    if member.issym() or member.islnk():
                        if os.path.isabs(member.linkname):
                            raise tarfile.TarError(
                                f"Attempted absolute path traversal in symlink/hardlink: {member.name} -> {member.linkname}"
                            )
                        link_target_path = os.path.abspath(
                            os.path.join(os.path.dirname(member_path), member.linkname)
                        )
                        if (
                            not link_target_path.startswith(target_dir + os.sep)
                            and link_target_path != target_dir
                        ):
                            raise tarfile.TarError(
                                f"Attempted path traversal in symlink/hardlink: {member.name} -> {member.linkname}"
                            )

                    safe_members.append(member)

                if hasattr(tarfile, "data_filter"):
                    tar.extractall(path=cache_dir, members=safe_members, filter="data")
                else:
                    tar.extractall(path=cache_dir, members=safe_members)
        except tarfile.TarError as e:
            # If decompression fails, remove the partially extracted directory
            shutil.rmtree(cache_dir, ignore_errors=True)
            logger.error(f"Failed to decompress {targz_path}: {e}")
            raise e

        return cache_dir

    @classmethod
    def retrieve_model_gcs(
        cls,
        model_name: str,
        source_url: str,
        cache_dir: str,
        deprecated_tar_struct: bool = False,
        local_files_only: bool = False,
    ) -> Path:
        fast_model_name = f"{'fast-' if deprecated_tar_struct else ''}{model_name.split('/')[-1]}"
        cache_tmp_dir = Path(cache_dir) / "tmp"
        model_tmp_dir = cache_tmp_dir / fast_model_name
        model_dir = Path(cache_dir) / fast_model_name

        # check if the model_dir and the model files are both present for macOS
        if model_dir.exists() and len(list(model_dir.glob("*"))) > 0:
            return model_dir

        if model_tmp_dir.exists():
            shutil.rmtree(model_tmp_dir)

        cache_tmp_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(cache_tmp_dir, 0o700)

        model_tar_gz = Path(cache_dir) / f"{fast_model_name}.tar.gz"

        if model_tar_gz.exists():
            model_tar_gz.unlink()

        if not local_files_only:
            cls.download_file_from_gcs(
                source_url,
                output_path=str(model_tar_gz),
            )

            cls.decompress_to_cache(targz_path=str(model_tar_gz), cache_dir=str(cache_tmp_dir))
            if not model_tmp_dir.exists():
                raise ValueError(f"Could not find {model_tmp_dir} in {cache_tmp_dir}")

            model_tar_gz.unlink()
            # Rename from tmp to final name is atomic
            model_tmp_dir.rename(model_dir)
        else:
            logger.error(
                f"Could not find the model tar.gz file at {model_dir} and local_files_only=True."
            )
            raise ValueError(
                f"Could not find the model tar.gz file at {model_dir} and local_files_only=True."
            )

        return model_dir

    @classmethod
    def _check_hf_cache(
        cls,
        hf_source: str,
        cache_dir: str,
        extra_patterns: list[str],
        model_file: str,
        **kwargs: Any,
    ) -> Path | None:
        try:
            cache_kwargs = deepcopy(kwargs)
            cache_kwargs["local_files_only"] = True
            cached_path = Path(
                cls.download_files_from_huggingface(
                    hf_source,
                    cache_dir=cache_dir,
                    extra_patterns=extra_patterns,
                    **cache_kwargs,
                )
            )
            # Verify the required model file actually exists in the cached snapshot
            if (cached_path / model_file).exists():
                return cached_path
        except Exception:
            logger.debug("Model not found in cache, will attempt download")
        finally:
            enable_progress_bars()
        return None

    @classmethod
    def _download_from_hf(
        cls, hf_source: str, cache_dir: str, extra_patterns: list[str], **kwargs: Any
    ) -> Path | None:
        local_files_only = kwargs.get("local_files_only", False)
        try:
            return Path(
                cls.download_files_from_huggingface(
                    hf_source,
                    cache_dir=cache_dir,
                    extra_patterns=extra_patterns,
                    **kwargs,
                )
            )
        except (OSError, RepositoryNotFoundError, ValueError) as e:
            if not local_files_only:
                logger.error(
                    f"Could not download model from HuggingFace: {e} "
                    "Falling back to other sources."
                )
        finally:
            enable_progress_bars()
        return None

    @classmethod
    def _download_from_gcs(
        cls,
        model_name: str,
        url_source: str | None,
        cache_dir: str,
        deprecated_tar_struct: bool,
        local_files_only: bool,
    ) -> Path | None:
        try:
            return cls.retrieve_model_gcs(
                model_name,
                str(url_source),
                str(cache_dir),
                deprecated_tar_struct=deprecated_tar_struct,
                local_files_only=local_files_only,
            )
        except Exception:
            if not local_files_only:
                logger.error(f"Could not download model from url: {url_source}")
        return None

    @classmethod
    def download_model(cls, model: T, cache_dir: str, retries: int = 3, **kwargs: Any) -> Path:
        """
        Downloads a model from HuggingFace Hub or Google Cloud Storage.

        Args:
            model (T): The model description.
                Example:
                ```
                {
                    "model": "BAAI/bge-base-en-v1.5",
                    "dim": 768,
                    "description": "Base English model, v1.5",
                    "size_in_GB": 0.44,
                    "sources": {
                        "url": "https://storage.googleapis.com/qdrant-qwen3_embed/fast-bge-base-en-v1.5.tar.gz",
                        "hf": "qdrant/bge-base-en-v1.5-onnx-q",
                    }
                }
                ```
            cache_dir (str): The path to the cache directory.
            retries: (int): The number of times to retry (including the first attempt)

        Returns:
            Path: The path to the downloaded model directory.
        """
        local_files_only = kwargs.get("local_files_only", False)
        specific_model_path: str | None = kwargs.pop("specific_model_path", None)
        if specific_model_path:
            return Path(specific_model_path)
        retries = 1 if local_files_only else retries
        hf_source = model.sources.hf
        url_source = model.sources.url

        extra_patterns = [model.model_file]
        extra_patterns.extend(model.additional_files)

        if hf_source:
            cached_path = cls._check_hf_cache(
                hf_source=hf_source,
                cache_dir=cache_dir,
                extra_patterns=extra_patterns,
                model_file=model.model_file,
                **kwargs,
            )
            if cached_path:
                return cached_path

        sleep = 3.0
        while retries > 0:
            retries -= 1

            if hf_source and not local_files_only:
                hf_path = cls._download_from_hf(
                    hf_source=hf_source,
                    cache_dir=cache_dir,
                    extra_patterns=extra_patterns,
                    **kwargs,
                )
                if hf_path:
                    return hf_path

            if url_source or local_files_only:
                gcs_path = cls._download_from_gcs(
                    model_name=model.model,
                    url_source=url_source,
                    cache_dir=cache_dir,
                    deprecated_tar_struct=model.sources.deprecated_tar_struct,
                    local_files_only=local_files_only,
                )
                if gcs_path:
                    return gcs_path

            if local_files_only:
                logger.error("Could not find model in cache_dir")
                break
            else:
                logger.error(
                    f"Could not download model from either source, sleeping for {sleep} seconds, {retries} retries left."
                )
                time.sleep(sleep)
                sleep *= 3

        raise ValueError(f"Could not load model {model.model} from any source.")
