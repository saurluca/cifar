import numpy as np
from custom_nn import ReLU, Sigmoid, Softmax


def test_relu_forward():
    relu = ReLU()
    assert relu.forward(1) == 1
    assert relu.forward(0) == 0
    assert relu.forward(-0.2) == 0


def test_sigmoid_forward():
    sigmoid = Sigmoid()
    # Test sigmoid(0) = 0.5
    assert np.isclose(sigmoid.forward(0), 0.5)
    # Test sigmoid(1) ≈ 0.731
    assert np.isclose(sigmoid.forward(1), 0.7310585786300049)
    # Test sigmoid(-1) ≈ 0.269
    assert np.isclose(sigmoid.forward(-1), 0.2689414213699951)
    # Test sigmoid of large positive number approaches 1
    assert np.isclose(sigmoid.forward(10), 0.9999546021312976)
    # Test sigmoid of large negative number approaches 0
    assert np.isclose(sigmoid.forward(-10), 4.5397868702434395e-05)


def test_softmax_forward():
    softmax = Softmax()
    # Test simple case with 2 values
    x = np.array([1.0, 2.0])
    result = softmax.forward(x)
    expected = np.array([0.26894142, 0.73105858])
    assert np.allclose(result, expected)

    # Test case with all equal values
    x = np.array([1.0, 1.0, 1.0])
    result = softmax.forward(x)
    expected = np.array([0.33333333, 0.33333333, 0.33333333])
    assert np.allclose(result, expected)

    # Test case with negative values
    x = np.array([-1.0, 0.0, 1.0])
    result = softmax.forward(x)
    expected = np.array([0.09003057, 0.24472847, 0.66524096])
    assert np.allclose(result, expected)

    # Test that output sums to 1
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = softmax.forward(x)
    assert np.isclose(np.sum(result), 1.0)
