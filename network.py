import itertools
import collections

class MSNet:

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_step(self, input_samples):
        """
        Compute and return the forward activation of each layer in layers.
        Input:
            input_samples: A matrix of input samples (each row 
                           is an input vector)
            layers: A list of Layers
        Output:
            A list of activations where the activation at each index 
            i+1 corresponds to the activation of layer i in layers. 
            activations[0] contains the input samples.  
        """
        activations = [input_samples]
        X = input_samples
        for layer in self.layers:
            Y = layer.get_output(X)
            activations.append(Y)
            X = activations[-1]
        
        return activations

    def backward_step(self, activations, targets):
        """
        Perform the backpropagation step over all the layers and return the parameter gradients.
        Input:
            activations: A list of forward step activations where the activation at 
                each index i+1 corresponds to the activation of layer i in layers. 
                activations[0] contains the input samples. 
            targets: The output targets of the output layer.
            layers: A list of Layers corresponding that generated the outputs in activations.
        Output:
            A list of parameter gradients where the gradients at each index corresponds to
            the parameters gradients of the layer at the same index in layers. 
        """
        param_grads = collections.deque()
        output_grad = None
        for layer in reversed(self.layers):
            Y = activations.pop()
            if output_grad is None:
                input_grad = layer.get_input_grad(Y, targets)
            else:
                input_grad = layer.get_input_grad(Y, output_grad)
            X = activations[-1]
            grads = layer.get_params_grad(X, output_grad)
            param_grads.appendleft(grads)
            output_grad = input_grad
        
        return list(param_grads)

    def update_params(self, param_grads, learning_rate):
        for layer, layer_backprop_grads in zip(self.layers, param_grads):
            for param, grad in zip(layer.get_params_iter(),
                                   layer_backprop_grads):
                param -= learning_rate * grad