from typing import Any

import torch.nn as nn
import torch


class LinearFunction(torch.autograd.Function):
    """
    Функция для реализации прямого и обратного прохода
    линейного (полносвязного) слоя нейронной сети
    """
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Классическая линейная реализация X @ W.T + b
        :param ctx: контекст для сохранения производных при прямом обходе
        :param args: аргументы в порядке [X, W, b]
        :param kwargs:
        :return:
        """
        inputs = args[0]
        weight = args[1]
        bias = args[2]

        ctx.save_for_backward(inputs, weight, bias)

        return inputs @ weight.T + bias

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """
        Обратное распростанение градиента для X @ W.T + b

        dL/dX = dL/dz * dz/dX = dL/dz * W
        dz/dW = dL/dz * dz/dW = dL/dz * X
        dz/db = dL/dz * dz/db = dL/dz * 1

        :param ctx: контекст с сохраненными значениями при прямом проходе
        :param grad_outputs: приходящий градиент
        :return: градиенты по dX, dW, db
        """
        inputs, weight, bias = ctx.saved_tensors

        der_inputs = grad_outputs[0] @ weight
        der_weight = grad_outputs[0].T @ inputs
        der_bias = grad_outputs[0].sum(axis=0)

        return der_inputs, der_weight, der_bias


class Linear(nn.Module):
    """
    Реализация линейного (полносвязного) слоя нейронной сети.
    Для переопределения метода backward требуется создать свою функцию
    и вызывать ее методы.

    Веса имеют нормальное распределение при инициализации * 1e-3.
    Важно инициализировать bias нулевым или очень малым значением.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 1e-3)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.linear = LinearFunction.apply

    def forward(self, x):
        return self.linear(x, self.weight, self.bias)
