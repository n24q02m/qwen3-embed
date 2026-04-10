"""Tests for common type definitions."""

import typing
from pathlib import Path

import numpy as np

from qwen3_embed.common.types import Device, NumpyArray, OnnxProvider, PathInput


def test_device_enum() -> None:
    """Test Device enum values."""
    assert Device.CPU == "cpu"
    assert Device.CUDA == "cuda"
    assert Device.AUTO == "auto"

    assert Device.CPU.value == "cpu"
    assert Device.CUDA.value == "cuda"
    assert Device.AUTO.value == "auto"

    assert list(Device) == [Device.CPU, Device.CUDA, Device.AUTO]


def test_path_input_type() -> None:
    """Test PathInput type definition."""
    # We can inspect the args of the union type
    args = typing.get_args(PathInput)
    assert str in args
    assert Path in args
    assert len(args) == 2


def test_onnx_provider_type() -> None:
    """Test OnnxProvider type definition."""
    args = typing.get_args(OnnxProvider)
    assert str in args
    # Be more robust with generic dict matching
    found_tuple = False
    for arg in args:
        if typing.get_origin(arg) is tuple:
            tuple_args = typing.get_args(arg)
            if (
                len(tuple_args) == 2
                and tuple_args[0] is str
                and typing.get_origin(tuple_args[1]) is dict
            ):
                found_tuple = True
                break
    assert found_tuple
    assert len(args) == 2


def test_numpy_array_type() -> None:
    """Test NumpyArray type definition."""
    # NumpyArray is npt.NDArray[np.float32]
    # Under NumPy 2.x, NDArray is an alias for numpy.ndarray[Any, numpy.dtype[ScalarType]]
    origin = typing.get_origin(NumpyArray)
    # npt.NDArray might resolve to np.ndarray or be a special type depending on numpy version
    assert origin is np.ndarray or str(origin).endswith("ndarray")

    args = typing.get_args(NumpyArray)
    # args[0] is Any (shape), args[1] is the dtype
    dtype_arg = args[1]

    # Extract the scalar type from the dtype
    if typing.get_origin(dtype_arg) is np.dtype:
        scalar_type = typing.get_args(dtype_arg)[0]
    else:
        # Fallback for different numpy/typing versions
        scalar_type = dtype_arg

    assert scalar_type is np.float32 or str(scalar_type).strip("'").endswith("float32")
