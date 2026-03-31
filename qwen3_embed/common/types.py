from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


PathInput: TypeAlias = str | Path

OnnxProvider: TypeAlias = str | tuple[str, dict[Any, Any]]


@dataclass
class RerankWorkerParams:
    model_name: str
    cache_dir: str
    parallel: int | None = None
    providers: Sequence[OnnxProvider] | None = None
    cuda: bool | Device = Device.AUTO
    device_ids: list[int] | None = None
    local_files_only: bool = False
    specific_model_path: str | None = None
    extra_session_options: dict[str, Any] | None = None


NumpyArray: TypeAlias = (
    NDArray[np.float64]
    | NDArray[np.float32]
    | NDArray[np.float16]
    | NDArray[np.int8]
    | NDArray[np.int64]
    | NDArray[np.int32]
)
