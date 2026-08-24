import numpy as np
import pytest

from ml3d.vis.visualizer import Model


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
def test_unsigned_integer_attributes_are_converted_to_float32(dtype):
    values = np.array([0, 1, 2], dtype=dtype)

    converted = Model()._convert_to_numpy(values)

    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted, values)
