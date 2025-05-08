import numpy as np

VERBOSE = True

class SigmoidCrossEntropy:
    def __init__(self):
        self.input = None

    def forward(self, input):
        raise NotImplementedError("Not implemented")

    def backward(self, grad):
        raise NotImplementedError("Not implemented")

    def __call__(self, x):
        return self.forward(x)


class ReLU:
    def __init__(self):
        self.input = None
    
    def forward(self, x):
        return np.maximum(0, x)
        raise NotImplementedError("Not implemented")

    def backward(self, grad):
        raise NotImplementedError("Not implemented")

    # needed?
    def step(self):
        raise NotImplementedError("Not implemented")
    
    def __call__(self, x):
        return self.forward(x)


class LinearLayer:
    def __init__(self, input_d, output_d, weights=None, bias=None, use_bias=True):
        self.input = None
        self.input_d = input_d
        self.output_d = output_d
        if bias:
            assert len(bias) == input_d
            self.bias = bias
        else:
            self.bias = np.zeros(input_d)
        
        if weights:
            print(f"len(weights): {weights}, input_d: {input_d}")
            self.weights = weights
        else:
            # using He initialization of weights
            self.weights = np.random.normal(loc=0, scale=np.sqrt(2.0/input_d), size=(input_d, output_d))


    def forward(self, x):
        if VERBOSE:
            print(f"multiplying {x} with {self.weights}")
        y = np.matmul(x, self.weights)
        if VERBOSE:
            print(f"result { y }")
        # y += self.bias
        if VERBOSE:
            print(f"result with bias { y }")
        return y
        raise NotImplementedError("Not implemented")

    def backward(self, grad):
        raise NotImplementedError("Not implemented")

    # needed?
    def step(self):
        raise NotImplementedError("Not implemented")
    
    # if object of this class is called, invoke forward of input x
    def __call__(self, x):
        return self.forward(x)


class FeedForwardNeuralNetwork:
    def __init__(self, n_layers, model_d, input_d, output_d, weights=None):
        self.n_layers = n_layers
        self.model_d = model_d
        self.input_d = input_d
        self.output_d = output_d
        self.activation_fn = ReLU()

        # assert len(weights) == 2 + n_layers
        # assert len(weights[0]) == input_d
        # for i in range(1, len(weights)-1):
        #     assert len(weights[i]) == model_d
        # assert len(weights[-1]) == model_d

        # model Architecture
        first_layer_weights = weights[0] if weights else None
        self.layer_stack = [
            LinearLayer(input_d, model_d, weights=first_layer_weights),
        ]
        
        for i in range(n_layers):
            layer_weights = weights[i+1] if weights else None
            self.layer_stack.append(
                LinearLayer(model_d, model_d, weights = layer_weights),
            )
            
        final_layer_weights = weights[-1] if weights else None
        self.final_layer = LinearLayer(model_d, output_d, weights=final_layer_weights)        

    def forward(self, x):
        for layer in self.layer_stack:
            x = layer(x)
            x = self.activation_fn(x)
        x = self.final_layer(x)
        return x
        raise NotImplementedError("Not implemented")

    def backward(self, grad):
        raise NotImplementedError("Not implemented")

    # needed?
    def step(self):
        raise NotImplementedError("Not implemented")
    
    def __call__(self, x):
        return self.forward(x)


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
    input_d = 2
    output_d = 2
    model_d = 3
    n_layers = 1
    
    # Create weights with all 1s:
    # First layer: input_d -> model_d (2x3)
    # Hidden layer: model_d -> model_d (3x3) 
    # Final layer: model_d -> output_d (3x2)
    weights = [
        [[1.0, 1.0, 1.0],  # First layer weights (2x3)
         [1.0, 1.0, 1.0]], 
        [[1.0, 1.0, 1.0],  # Hidden layer weights (3x3)
         [1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0]],
        [[1.0, 1.0],       # Final layer weights (3x2)
         [1.0, 1.0],
         [1.0, 1.0]]
    ]
    weights= None
    model = FeedForwardNeuralNetwork(n_layers, model_d, input_d, output_d, weights=weights)
    print(model)
    
    sample_data = np.array([2, 2])
    result = model(sample_data)
    
    # relu = ReLU()
    # result = relu(sample_data)
    
    print("result", result)

if __name__ == "__main__":
    main()
