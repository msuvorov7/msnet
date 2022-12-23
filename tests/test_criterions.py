import unittest
import torch
from torch.autograd import gradcheck

from src.criterions.neg_log_likelihood_loss import NLLLossFunction


class TestCriterions(unittest.TestCase):

    def test_nllloss_autograd(self):
        neg_log_loss_func = NLLLossFunction.apply
        x = torch.randn(3, 4, requires_grad=True).double()
        y = torch.randint(4, (3,))
        self.assertTrue(gradcheck(neg_log_loss_func, (x, y)))
