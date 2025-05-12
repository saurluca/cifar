import numpy as np

VERBOSE = False

class Module:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        raise NotImplementedError("Not implemented")
        
    def backward(self, grad):
        raise NotImplementedError("Not implemented")

    def __call__(self, x):
        return self.forward(x)

         
class Softmax(Module):   
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, x):
        exp_values = np.exp(x)
        return exp_values / np.sum(exp_values)

    def backward(self, grad):
        raise NotImplementedError("Not implemented")


class CrossEntropyLoss(Module):
    def __init__(self):
        self.input = None

    def forward(self, y_pred, y):
        loss = 0 
        
        # TODO make more efficent
        for i in range(len(y_pred)):
            loss += (-1 * y[i] * np.log(y_pred[i]))
        
        return loss
        
    def backward(self, grad):
        raise NotImplementedError("Not implemented")


class Sigmoid(Module):
    def __init__(self):
        self.input = None
        self.output = None
        self.grad = None
    
    def _get_sigmoid_value(self, x):
        return 1 / (1 + np.exp(-x))
        
    def forward(self, x):
        self.input = x
        self.output = self._get_sigmoid_value(x)
        return self.output
    
    def backward(self, grad):
        sigma_input = self._get_sigmoid_value(self.input)
        self.grad =  sigma_input * (1 - sigma_input)
        return grad * self.grad
        

class ReLU(Module):
    def __init__(self):
        self.input = None
        self.output = None
        self.grad = None
    
    def forward(self, x):
        self.input = x
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, grad):
        self.grad = 1 if self.output > 0 else 0
        return grad * self.grad


class LinearLayer(Module):
    def __init__(self, input_d, output_d, weights=None, bias=None, use_bias=True):
        self.input = None
        self.output = None
        self.grad_weights = None
        self.grad_bias = None
        self.input_d = input_d
        self.output_d = output_d

        
        # if bias:
        #     assert len(bias) == input_d
        #     self.bias = bias
        # else:
        #     self.bias = np.zeros(input_d)
        
        if weights:
            print(f"len(weights): {weights}, input_d: {input_d}")
            self.weights = weights
        else:
            # using He initialization of weights
            self.weights = np.random.normal(loc=0, scale=np.sqrt(2.0/input_d), size=(input_d, output_d))

    def forward(self, x):
        self.input = x
        if np.isscalar(x):
           x = np.array([x])
           if VERBOSE:
               print("x is scalar, converting to 1D array")
               
        if VERBOSE:
            print(f"multiplying {x} with {self.weights}")
        y = np.matmul(x, self.weights)
        if VERBOSE:
            print(f"result { y }")
        # y += self.bias
        if VERBOSE:
            print(f"result with bias { y }")
        self.output = y
        return y

    def backward(self, grad):
        self.grad_weights = 0
        
        
        raise NotImplementedError("Not implemented")

    # needed?
    def step(self):
        raise NotImplementedError("Not implemented")
    

class FeedForwardNeuralNetwork(Module):
    def __init__(self, n_layers, model_d, input_d, output_d, weights=None):
        self.n_layers = n_layers
        self.model_d = model_d
        self.input_d = input_d
        self.output_d = output_d
        self.activation_fn = ReLU()
        self.final_activation_fn = Sigmoid()
        self.l_stack = []
        
        self.l_stack.append(LinearLayer(input_d, model_d))
        self.l_stack.append(ReLU())
        self.l_stack.append(LinearLayer(model_d, output_d))
        self.l_stack.append(Sigmoid())  

        # assert len(weights) == 2 + n_layers
        # assert len(weights[0]) == input_d
        # for i in range(1, len(weights)-1):
        #     assert len(weights[i]) == model_d
        # assert len(weights[-1]) == model_d

        # model Architecture
        # first_layer_weights = weights[0] if weights else None
        # self.layer_stack = [
        #     LinearLayer(input_d, model_d, weights=first_layer_weights),
        # ]
        
        # for i in range(n_layers):
        #     layer_weights = weights[i+1] if weights else None
        #     self.layer_stack.append(
        #         LinearLayer(model_d, model_d, weights = layer_weights),
        #     )
            
        # final_layer_weights = weights[-1] if weights else None
        # self.final_layer = LinearLayer(model_d, output_d, weights=final_layer_weights)        

    def forward(self, x):
        # for layer in self.layer_stack:
        #     x = layer(x)
        #     x = self.activation_fn(x)
        # x = self.final_layer(x)
        
        # if self.final_activation_fn:
        #     x = self.final_activation_fn(x)
        for layer in self.l_stack:
            x = layer(x)
        return x

    def backward(self, grad):
        for layer in self.l_stack:
            grad = layer.backward(grad)
            
        raise NotImplementedError("Not implemented")

    # needed?
    def step(self):
        raise NotImplementedError("Not implemented")
    
    
class SochasticGradientDescent(Module):
    def __init__(self, lr, params, objective):
        self.lr = lr
        self.params = params
        self.objective = objective
            
    def step(self):
        raise NotImplementedError("Not implemented")
    
    def zero_grad(self):
        raise NotImplementedError("Not implemented")


def load_data(n_samples=10):
    x = np.random.uniform(-10, 10, 10) 
    # generates label 0 for negative 1 for positve
    y = np.maximum(0, np.sign(x))
    data = list(zip(x,y))
    return data


def train(model, loss_fn, train_set):
    train_loss = 0
    correct_n = 0
    
    for x, y in train_set:
        y_pred = model(x)
        train_loss += loss_fn(y_pred, y)
        
        grad = loss_fn.backward()
        model.backward(grad)
        
        if np.round(y_pred) == y:
            correct_n += 1

        
        
    accuracy = correct_n / len(train_set)
    
    print(f"correct_n: {correct_n}, len train_set: {len(train_set)}")
    print("accuracy", accuracy)
    print("loss", train_loss)
        
    return train_loss, accuracy


def evaluate():
    raise NotImplementedError("Not implemented")


def plot_loss():
    raise NotImplementedError("Not implemented")


def plot_accuracy():
    raise NotImplementedError("Not implemented")


def fake_loss_fn(y_pred, y):
    return 1  


class BCELoss(Module):
    def __init__(self):
        self.input_y_pred = None
        self.input_y = None
    
    def forward(self, y_pred, y):
        self.input_y_pred = y_pred
        self.input_y = y
        return -np.mean(y * np.log(y_pred) + (1-y)*np.log(1-y_pred))
    
    def backward(self):
        return -self.input_y / self.input_y_pred + (1 - self.input_y) / (1 - self.input_y_pred)
    
    def __call__(self, y_pred, y):
        return self.forward(y_pred, y)
    

def main():  
    input_d = 1
    model_d = 2
    n_layers = 0
    output_d = 1
    np.random.seed(42)
    
    # lets build a postive / negative number classifer
    train_set = load_data(10)

    model = FeedForwardNeuralNetwork(n_layers, model_d, input_d, output_d)

    loss_fn = BCELoss()
        
    train_loss, accuracy = train(model, loss_fn, train_set)


if __name__ == "__main__":
    main()
