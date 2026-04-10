from enum import StrEnum
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

NumpyArray: TypeAlias = npt.NDArray[np.float32]
OnnxProvider: TypeAlias = str | tuple[str, dict[str, Any]]
PathInput: TypeAlias = str | Path


class Device(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"
