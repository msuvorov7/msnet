import numpy as np
import itertools
import activations as active

class Layer(object):
    
    def get_params_iter(self):
        """Return an iterator over the parameters (if any)"""
        return []
    
    def get_params_grad(self, X, output_grad):
        """Return a list of gradients over the parameters"""
        return []
    
    def get_output(self, X):
        """Perform the forward step linear transformation"""
        pass
    
    def get_input_grad(self, Y, output_grad=None, T=None):
        """Return the gradient at the inputs of this layer"""
        pass

class LinearLayer(Layer):
    
    def __init__(self, n_in, n_out):
        self.W = np.random.randn(n_in, n_out) * 0.1
        self.b = np.zeros(n_out) + 0.1
        
    def get_params_iter(self):
        return itertools.chain(
            np.nditer(self.W, op_flags=['readwrite']),
            np.nditer(self.b, op_flags=['readwrite']))
    
    def get_output(self, X):
        return (X @ self.W) + self.b
    
    def get_params_grad(self, X, output_grad):
        JW = X.T @ output_grad
        Jb = np.sum(output_grad, axis=0)
        return [g for g in itertools.chain(
            np.nditer(JW), np.nditer(Jb))]
    
    def get_input_grad(self, Y, output_grad):
        return output_grad @ self.W.T

class LogisticLayer(Layer):
    
    def get_output(self, X):
        return active.logistic(X)
    
    def get_input_grad(self, Y, ouput_grad):
        return np.multiply(active.logistic_deriv(Y), ouput_grad)
    
    def get_cost(self, Y, T):
        return np.mean((T - Y)**2).sum() / Y.shape[0]

class SoftmaxOutputLayer(Layer):
    
    def get_output(self, X):
        return active.softmax(X)
    
    def get_input_grad(self, Y, T):
        return (Y - T) / Y.shape[0]
    
    def get_cost(self, Y, T):
        return -np.multiply(T, np.log(Y)).sum() / Y.shape[0]

class LinearActivationLayer(Layer):
    
    def get_output(self, X):
        return X
    
    def get_input_grad(self, Y, T):
        return (Y - T) / Y.shape[0]
    
    def get_cost(self, Y, T):
        return np.mean((T - Y)**2).sum() / Y.shape[0]
