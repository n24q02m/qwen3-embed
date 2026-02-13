from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class Device(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


type PathInput = str | Path

type OnnxProvider = str | tuple[str, dict[Any, Any]]
type NumpyArray = (
    NDArray[np.float64]
    | NDArray[np.float32]
    | NDArray[np.float16]
    | NDArray[np.int8]
    | NDArray[np.int64]
    | NDArray[np.int32]
)
