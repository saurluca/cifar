import pytest
from custom_nn import ReLU


def test_relu_forward():
    relu = ReLU()
    assert relu.forward(1) == 1
    assert relu.forward(0) == 0
    assert relu.forward(-0.2) == 0
