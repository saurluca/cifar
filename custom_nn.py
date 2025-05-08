import numpy as np


class SigmoidCrossEntropy:
    def __init__(self):
        self.input = None

    def forward(self, input):
        raise NotImplementedError("Not implemented")

    def backward(self, grad):
        raise NotImplementedError("Not implemented")


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, input):
        raise NotImplementedError("Not implemented")

    def backward(self, grad):
        raise NotImplementedError("Not implemented")

    # needed?
    def step(self):
        raise NotImplementedError("Not implemented")


class LinearLayer:
    def __init__(self):
        self.input = None

    def forward(self, input):
        raise NotImplementedError("Not implemented")

    def backward(self, grad):
        raise NotImplementedError("Not implemented")

    # needed?
    def step(self):
        raise NotImplementedError("Not implemented")


class FeedForwardNeuralNetwork:
    def __init__(self):
        self.input = None

    def forward(self, input):
        raise NotImplementedError("Not implemented")

    def backward(self, grad):
        raise NotImplementedError("Not implemented")

    # needed?
    def step(self):
        raise NotImplementedError("Not implemented")


def load_data():
    raise NotImplementedError("Not implemented")


def train():
    raise NotImplementedError("Not implemented")


def evaluate():
    raise NotImplementedError("Not implemented")


def plot_loss():
    raise NotImplementedError("Not implemented")


def plot_accuracy():
    raise NotImplementedError("Not implemented")


def main():
    load_data()


if __name__ == "__main__":
    main()
