from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray


class Device(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


PathInput: TypeAlias = str | Path

OnnxProvider: TypeAlias = str | tuple[str, dict[Any, Any]]
NumpyArray: TypeAlias = (
    NDArray[np.float64]
    | NDArray[np.float32]
    | NDArray[np.float16]
    | NDArray[np.int8]
    | NDArray[np.int64]
    | NDArray[np.int32]
)


@dataclass
class ExecutionConfig:
    """Configuration for model execution."""

    cache_dir: str | None = None
    threads: int | None = None
    providers: Sequence[OnnxProvider] | None = None
    cuda: bool | Device = Device.AUTO
    device_ids: list[int] | None = None
    lazy_load: bool = False
    device_id: int | None = None
    specific_model_path: str | None = None
    local_files_only: bool = False
