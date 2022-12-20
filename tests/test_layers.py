import unittest
import torch
from torch.autograd import gradcheck

from src.layers.linear import LinearFunction
from src.layers.relu import ReLUFunction
from src.layers.sigmoid import SigmoidFunction


class TestLayers(unittest.TestCase):
    def test_linear_autograd(self):
        lin_func = LinearFunction.apply
        x = torch.randn(2, 3, requires_grad=True).double()
        w = torch.randn(5, 3, requires_grad=True).double()
        b = torch.zeros(5, requires_grad=True).double()
        self.assertTrue(gradcheck(lin_func, (x, w, b)))

    def test_relu_autograd(self):
        relu_func = ReLUFunction.apply
        x = torch.randn(2, 3, requires_grad=True).double()
        self.assertTrue(gradcheck(relu_func, x))

    def test_sigmoid_autograd(self):
        sigmoid_func = SigmoidFunction.apply
        x = torch.randn(2, 3, requires_grad=True).double()
        self.assertTrue(gradcheck(sigmoid_func, x))
